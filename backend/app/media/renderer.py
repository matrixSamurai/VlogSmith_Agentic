"""Video assembly: timeline -> normalized segments -> transitions -> final mix.

Every clip (title card, photo, video) is first rendered into a segment with
identical codec parameters, so segments can be joined either by stream copy
(hard cuts / fade-through-black) or by an xfade chain (crossfade, slide...).
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Awaitable, Callable, Literal

from PIL import Image, ImageFilter, ImageOps

from .ffmpeg import run_ffmpeg

log = logging.getLogger(__name__)

XFADE_NAMES = {"crossfade": "fade", "slide": "slideleft", "zoom": "zoomin", "wipe": "wipeleft"}
PRESETS = {"fast": "veryfast", "balanced": "fast", "high": "medium"}
KB_UPSCALE = {"fast": 1.5, "balanced": 2.0, "high": 2.5}


@dataclass
class Clip:
    kind: Literal["card", "image", "video"]
    src: Path
    duration: float
    overlay: Path | None = None
    clip_start: float = 0.0
    source_duration: float | None = None  # for videos
    has_audio: bool = False
    motion: int = 0  # Ken Burns variant
    section_index: int | None = None
    hold: float = 0.0  # extra frozen-frame time appended to a video


@dataclass
class RenderSpec:
    width: int
    height: int
    fps: int
    transition: str
    transition_duration: float
    ken_burns: bool
    quality: str
    clip_audio: str
    clip_audio_volume: float

    @property
    def uses_xfade(self) -> bool:
        return self.transition in XFADE_NAMES

    @property
    def overlap(self) -> float:
        return self.transition_duration if self.uses_xfade else 0.0


@dataclass
class Timeline:
    clips: list[Clip]
    starts: list[float] = field(default_factory=list)  # start time of each clip in the final video
    total: float = 0.0


def layout_timeline(clips: list[Clip], spec: RenderSpec) -> Timeline:
    starts, t = [], 0.0
    for i, clip in enumerate(clips):
        starts.append(t)
        t += clip.duration + clip.hold - (spec.overlap if i < len(clips) - 1 else 0)
    return Timeline(clips=clips, starts=starts, total=round(t, 3))


def _encode_args(spec: RenderSpec) -> list[str]:
    return [
        "-c:v", "libx264", "-preset", PRESETS.get(spec.quality, "fast"), "-crf", "20" if spec.quality != "fast" else "23",
        "-pix_fmt", "yuv420p", "-r", str(spec.fps), "-video_track_timescale", "90000",
        "-c:a", "aac", "-b:a", "160k", "-ar", "48000", "-ac", "2",
    ]


def _compose_still(src: Path, dst: Path, size: tuple[int, int]) -> None:
    """Fit an image onto the frame. Mismatched aspect ratios get a blurred backdrop."""
    W, H = size
    with Image.open(src) as im:
        im = ImageOps.exif_transpose(im).convert("RGB")
        src_ratio, dst_ratio = im.width / im.height, W / H
        if abs(src_ratio - dst_ratio) / dst_ratio < 0.18:
            out = ImageOps.fit(im, size, Image.Resampling.LANCZOS)
        else:
            out = ImageOps.fit(im, size, Image.Resampling.BILINEAR).filter(ImageFilter.GaussianBlur(max(W, H) // 30))
            out = Image.blend(out, Image.new("RGB", size, (0, 0, 0)), 0.35)
            fg = ImageOps.contain(im, size, Image.Resampling.LANCZOS)
            out.paste(fg, ((W - fg.width) // 2, (H - fg.height) // 2))
        out.save(dst, "JPEG", quality=93)


def _zoompan(motion: int, frames: int, w: int, h: int, fps: int, strength: float = 0.12) -> str:
    n = max(frames - 1, 1)
    p = f"(on/{n})"
    ease = f"(0.5-0.5*cos(PI*{p}))"
    variants = [
        (f"1+{strength}*{ease}", "iw/2-(iw/zoom/2)", "ih/2-(ih/zoom/2)"),  # zoom in
        (f"{1 + strength}-{strength}*{ease}", "iw/2-(iw/zoom/2)", "ih/2-(ih/zoom/2)"),  # zoom out
        (f"{1 + strength}", f"(iw-iw/zoom)*{ease}", "ih/2-(ih/zoom/2)"),  # pan right
        (f"{1 + strength}", f"(iw-iw/zoom)*(1-{ease})", "ih/2-(ih/zoom/2)"),  # pan left
        (f"1+{strength}*{ease}", "(iw-iw/zoom)*0.3", "(ih-ih/zoom)*0.3"),  # zoom to upper-left third
    ]
    z, x, y = variants[motion % len(variants)]
    return f"zoompan=z='{z}':x='{x}':y='{y}':d={frames}:s={w}x{h}:fps={fps}"


def _fade_filters(spec: RenderSpec, duration: float) -> tuple[str, str]:
    if spec.transition != "fade_black":
        return "", ""
    d = min(spec.transition_duration / 2 + 0.1, duration / 4)
    out = max(duration - d, 0)
    return (
        f",fade=t=in:st=0:d={d:.2f},fade=t=out:st={out:.2f}:d={d:.2f}",
        f",afade=t=in:st=0:d={d:.2f},afade=t=out:st={out:.2f}:d={d:.2f}",
    )


async def render_segment(clip: Clip, spec: RenderSpec, out: Path, work: Path) -> Path:
    W, H, fps = spec.width, spec.height, spec.fps
    dur = round(clip.duration + clip.hold, 3)
    vfade, afade = _fade_filters(spec, dur)
    args: list[str] = []
    filters: list[str] = []

    if clip.kind in ("image", "card"):
        scale = KB_UPSCALE.get(spec.quality, 2.0) if (spec.ken_burns or clip.kind == "card") else 1.0
        still = work / f"{out.stem}_still.jpg"
        await asyncio.to_thread(_compose_still, clip.src, still, (int(W * scale) // 2 * 2, int(H * scale) // 2 * 2))
        frames = int(round(dur * fps))
        args += ["-i", str(still)]
        if spec.ken_burns or clip.kind == "card":
            strength = 0.05 if clip.kind == "card" else 0.12
            filters.append(f"[0:v]{_zoompan(clip.motion, frames, W, H, fps, strength)},setsar=1,format=yuv420p[base]")
        else:
            filters.append(f"[0:v]loop=loop=-1:size=1:start=0,fps={fps},trim=duration={dur},scale={W}:{H},setsar=1,format=yuv420p[base]")
        audio_label = None
    else:
        src_dur = clip.source_duration or (clip.clip_start + clip.duration)
        take = max(0.5, min(clip.duration, src_dur - clip.clip_start))
        args += ["-ss", f"{clip.clip_start:.3f}", "-t", f"{take:.3f}", "-i", str(clip.src)]
        pad = max(0.0, dur - take)
        filters.append(
            f"[0:v]split=2[bgsrc][fgsrc];"
            f"[bgsrc]scale={W}:{H}:force_original_aspect_ratio=increase,crop={W}:{H},boxblur=20:2,eq=brightness=-0.12[bg];"
            f"[fgsrc]scale={W}:{H}:force_original_aspect_ratio=decrease[fg];"
            f"[bg][fg]overlay=(W-w)/2:(H-h)/2,fps={fps},setsar=1,format=yuv420p,"
            f"tpad=stop_mode=clone:stop_duration={pad + 0.1:.3f},trim=duration={dur}[base]"
        )
        audio_label = "0:a" if (clip.has_audio and spec.clip_audio == "keep") else None

    if clip.overlay:
        idx = len([a for a in args if a == "-i"])
        args += ["-loop", "1", "-t", f"{dur}", "-i", str(clip.overlay)]
        fade_in = 0.35 if clip.kind != "card" else 0
        filters.append(
            f"[{idx}:v]format=rgba,fade=t=in:st={fade_in}:d=0.45:alpha=1,"
            f"fade=t=out:st={max(dur - 0.5, 0):.2f}:d=0.4:alpha=1[ov];[base][ov]overlay=0:0:shortest=1[comp]"
        )
        vlabel = "comp"
    else:
        vlabel = "base"
    filters.append(f"[{vlabel}]null{vfade}[vout]")

    if audio_label:
        filters.append(
            f"[{audio_label}]aresample=48000,aformat=channel_layouts=stereo,volume={spec.clip_audio_volume},"
            f"apad,atrim=duration={dur}{afade}[aout]"
        )
    else:
        idx = len([a for a in args if a == "-i"])
        args += ["-f", "lavfi", "-t", f"{dur}", "-i", "anullsrc=channel_layout=stereo:sample_rate=48000"]
        filters.append(f"[{idx}:a]atrim=duration={dur}[aout]")

    await run_ffmpeg(
        [*args, "-filter_complex", ";".join(filters), "-map", "[vout]", "-map", "[aout]",
         "-t", f"{dur}", *_encode_args(spec), str(out)],
        timeout=900,
    )
    return out


async def join_segments(segments: list[Path], durations: list[float], spec: RenderSpec, out: Path, work: Path) -> Path:
    if len(segments) == 1 or not spec.uses_xfade:
        listing = work / "concat.txt"
        listing.write_text("".join(f"file '{p.resolve()}'\n" for p in segments))
        await run_ffmpeg(["-f", "concat", "-safe", "0", "-i", str(listing), "-c", "copy", str(out)])
        return out

    T = spec.transition_duration
    name = XFADE_NAMES[spec.transition]
    args: list[str] = []
    for seg in segments:
        args += ["-i", str(seg)]
    vf, af = [], []
    prev_v, prev_a = "0:v", "0:a"
    offset = 0.0
    for i in range(1, len(segments)):
        offset += durations[i - 1] - T
        nv, na = f"v{i}", f"a{i}"
        vf.append(f"[{prev_v}][{i}:v]xfade=transition={name}:duration={T}:offset={offset:.3f}[{nv}]")
        af.append(f"[{prev_a}][{i}:a]acrossfade=d={T}:c1=tri:c2=tri[{na}]")
        prev_v, prev_a = nv, na
    await run_ffmpeg(
        [*args, "-filter_complex", ";".join(vf + af), "-map", f"[{prev_v}]", "-map", f"[{prev_a}]",
         *_encode_args(spec), str(out)],
        timeout=1800,
    )
    return out


async def build_voice_track(parts: list[tuple[float, Path]], total: float, out: Path) -> Path | None:
    """Place narration clips at their start times on a single full-length track."""
    if not parts:
        return None
    args, filters = [], []
    for i, (start, path) in enumerate(parts):
        args += ["-i", str(path)]
        ms = int(max(start, 0) * 1000)
        filters.append(f"[{i}:a]aresample=48000,aformat=channel_layouts=stereo,adelay={ms}|{ms}[n{i}]")
    mix = "".join(f"[n{i}]" for i in range(len(parts)))
    filters.append(f"{mix}amix=inputs={len(parts)}:normalize=0:dropout_transition=0,apad,atrim=duration={total:.3f}[vo]")
    await run_ffmpeg([*args, "-filter_complex", ";".join(filters), "-map", "[vo]", "-c:a", "pcm_s16le", str(out)])
    return out


async def final_mix(
    video: Path, total: float, out: Path, music: Path | None, music_volume: float, voice: Path | None
) -> Path:
    args = ["-i", str(video)]
    filters = ["[0:a]aresample=48000[clip]"]
    mix = ["[clip]"]
    voice_idx = music_idx = None
    if voice:
        voice_idx = 1
        args += ["-i", str(voice)]
    if music:
        music_idx = 2 if voice else 1
        args += ["-stream_loop", "-1", "-i", str(music)]
    if voice_idx is not None:
        filters.append(f"[{voice_idx}:a]aresample=48000,aformat=channel_layouts=stereo,volume=1.15,asplit=2[vo][vosc]")
        mix.append("[vo]")
    if music_idx is not None:
        fade_out = max(total - 2.5, 0)
        filters.append(
            f"[{music_idx}:a]aresample=48000,aformat=channel_layouts=stereo,atrim=duration={total:.3f},"
            f"volume={music_volume:.3f},afade=t=in:st=0:d=1.2,afade=t=out:st={fade_out:.2f}:d=2.5[mus]"
        )
        if voice_idx is not None:
            filters.append("[mus][vosc]sidechaincompress=threshold=0.02:ratio=8:attack=30:release=500[musd]")
            mix.append("[musd]")
        else:
            mix.append("[mus]")
    elif voice_idx is not None:
        filters.append("[vosc]anullsink")
    filters.append(
        f"{''.join(mix)}amix=inputs={len(mix)}:normalize=0:dropout_transition=0,"
        f"alimiter=limit=0.95,atrim=duration={total:.3f}[aout]"
    )
    await run_ffmpeg(
        [*args, "-filter_complex", ";".join(filters), "-map", "0:v", "-map", "[aout]",
         "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", "-t", f"{total:.3f}", str(out)],
        timeout=1200,
    )
    return out


ProgressFn = Callable[[int, int], Awaitable[None]]


async def render_timeline(
    timeline: Timeline,
    spec: RenderSpec,
    work: Path,
    out: Path,
    music: Path | None,
    music_volume: float,
    voice_parts: list[tuple[float, Path]],
    concurrency: int = 2,
    on_progress: ProgressFn | None = None,
) -> Path:
    work.mkdir(parents=True, exist_ok=True)
    sem = asyncio.Semaphore(max(1, concurrency))
    done = 0

    async def one(i: int, clip: Clip) -> Path:
        nonlocal done
        async with sem:
            path = await render_segment(clip, spec, work / f"seg_{i:03d}.mp4", work)
        done += 1
        if on_progress:
            await on_progress(done, len(timeline.clips))
        return path

    segments = await asyncio.gather(*(one(i, c) for i, c in enumerate(timeline.clips)))
    durations = [c.duration + c.hold for c in timeline.clips]
    joined = await join_segments(list(segments), durations, spec, work / "joined.mp4", work)
    voice = await build_voice_track(voice_parts, timeline.total, work / "voice.wav")
    return await final_mix(joined, timeline.total, out, music, music_volume, voice)
