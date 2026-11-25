"""Video assembly: timeline -> graded, framed segments with animated overlays ->
per-boundary transitions -> final pass (subtitles, music, narration, SFX).

Every clip is first rendered into a segment with identical codec parameters so
segments can be joined by stream copy (hard cuts / fade-through-black) or by an
xfade chain whose transition can differ at every boundary.
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

LOOKS = {
    "natural": "eq=contrast=1.03:saturation=1.06",
    "warm": "colorbalance=rs=0.05:gs=0.01:bs=-0.06:rm=0.04:bm=-0.04,eq=contrast=1.05:saturation=1.12:gamma=1.02",
    "cinematic": "colorbalance=rs=-0.06:gs=-0.01:bs=0.08:rh=0.07:gh=0.01:bh=-0.07,eq=contrast=1.1:saturation=0.92,"
                 "vignette=angle=PI/4.6",
    "vibrant": "eq=contrast=1.07:saturation=1.38:gamma=1.02,unsharp=5:5:0.45",
    "moody": "colorbalance=bs=0.05:bm=0.03,eq=contrast=1.13:brightness=-0.035:saturation=0.78,vignette=angle=PI/4",
    "bw": "hue=s=0,eq=contrast=1.18:gamma=0.98,vignette=angle=PI/5",
    "none": "",
}


@dataclass
class Clip:
    kind: Literal["card", "image", "video"]
    src: Path
    duration: float
    overlay: Path | None = None  # ffconcat overlay track (animated) or a static PNG
    clip_start: float = 0.0
    source_duration: float | None = None  # for videos
    has_audio: bool = False
    motion: int = 0  # Ken Burns variant
    section_index: int | None = None
    hold: float = 0.0  # extra frozen-frame time appended to a video
    focus: tuple[float, float] = (0.5, 0.5)  # subject position (0-1) for crops and Ken Burns
    graded: bool = True  # apply the colour look (title/end cards on blur are graded too)

    @property
    def length(self) -> float:
        return round(self.duration + self.hold, 3)


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
    look: str = "natural"
    letterbox: bool = False
    framing: str = "smart_crop"  # smart_crop | fit_blur

    @property
    def uses_xfade(self) -> bool:
        return self.transition in XFADE_NAMES or self.transition == "smart"


@dataclass
class Boundary:
    name: str  # xfade transition name
    duration: float


@dataclass
class Timeline:
    clips: list[Clip]
    boundaries: list[Boundary]
    starts: list[float] = field(default_factory=list)
    total: float = 0.0


def plan_boundaries(clips: list[Clip], spec: RenderSpec) -> list[Boundary]:
    """Transition at each cut. 'smart' = quick dissolves inside a section, bigger moves between sections."""
    out: list[Boundary] = []
    if not spec.uses_xfade:
        return [Boundary("cut", 0.0) for _ in clips[1:]]
    big = ["zoomin", "smoothleft", "circleopen", "slideleft", "smoothup", "radial"]
    k = 0
    for a, b in zip(clips, clips[1:]):
        if spec.transition != "smart":
            out.append(Boundary(XFADE_NAMES[spec.transition], spec.transition_duration))
        elif b.kind == "card":  # into the end screen
            out.append(Boundary("fadeblack", 0.7))
        elif a.kind == "card" or a.section_index != b.section_index:
            out.append(Boundary(big[k % len(big)], 0.55))
            k += 1
        else:
            out.append(Boundary("fade", 0.2))
    return out


def layout_timeline(clips: list[Clip], spec: RenderSpec, boundaries: list[Boundary] | None = None) -> Timeline:
    bounds = boundaries if boundaries is not None else plan_boundaries(clips, spec)
    starts, t = [], 0.0
    for i, clip in enumerate(clips):
        starts.append(round(t, 3))
        t += clip.length - (bounds[i].duration if i < len(bounds) else 0)
    return Timeline(clips=clips, boundaries=bounds, starts=starts, total=round(t, 3))


def _encode_args(spec: RenderSpec, audio: bool = True) -> list[str]:
    args = [
        "-c:v", "libx264", "-preset", PRESETS.get(spec.quality, "fast"), "-crf", "19" if spec.quality == "high" else "21",
        "-pix_fmt", "yuv420p", "-r", str(spec.fps), "-video_track_timescale", "90000",
    ]
    if audio:
        args += ["-c:a", "aac", "-b:a", "160k", "-ar", "48000", "-ac", "2"]
    return args


def _compose_still(src: Path, dst: Path, size: tuple[int, int], focus: tuple[float, float], framing: str,
                   blur_bg: bool = False) -> None:
    """Fit an image onto the frame around its subject; optionally a fully blurred backdrop (end cards)."""
    W, H = size
    with Image.open(src) as im:
        im = ImageOps.exif_transpose(im).convert("RGB")
        if blur_bg:
            out = ImageOps.fit(im, size, Image.Resampling.BILINEAR, centering=focus)
            out = out.filter(ImageFilter.GaussianBlur(max(W, H) // 45))
            out = Image.blend(out, Image.new("RGB", size, (6, 6, 12)), 0.5)
        elif framing == "smart_crop" or abs(im.width / im.height - W / H) / (W / H) < 0.18:
            out = ImageOps.fit(im, size, Image.Resampling.LANCZOS, centering=focus)
        else:
            out = ImageOps.fit(im, size, Image.Resampling.BILINEAR, centering=focus).filter(
                ImageFilter.GaussianBlur(max(W, H) // 30))
            out = Image.blend(out, Image.new("RGB", size, (0, 0, 0)), 0.35)
            fg = ImageOps.contain(im, size, Image.Resampling.LANCZOS)
            out.paste(fg, ((W - fg.width) // 2, (H - fg.height) // 2))
        out.save(dst, "JPEG", quality=93)


def _zoompan(motion: int, frames: int, w: int, h: int, fps: int, focus: tuple[float, float], strength: float) -> str:
    n = max(frames - 1, 1)
    ease = f"(0.5-0.5*cos(PI*on/{n}))"
    fx, fy = (min(max(v, 0.0), 1.0) for v in focus)
    to_focus_x = f"(iw-iw/zoom)*{fx:.3f}"
    to_focus_y = f"(ih-ih/zoom)*{fy:.3f}"
    centre_x, centre_y = "(iw-iw/zoom)/2", "(ih-ih/zoom)/2"
    variants = [
        (f"1+{strength}*{ease}", f"{centre_x}+({to_focus_x}-{centre_x})*{ease}", f"{centre_y}+({to_focus_y}-{centre_y})*{ease}"),
        (f"{1 + strength}-{strength}*{ease}", to_focus_x, to_focus_y),  # pull out from the subject
        (f"{1 + strength}", f"(iw-iw/zoom)*{ease}", to_focus_y),  # pan right
        (f"{1 + strength}", f"(iw-iw/zoom)*(1-{ease})", to_focus_y),  # pan left
        (f"{1 + strength}", to_focus_x, f"(ih-ih/zoom)*(1-{ease})"),  # tilt up
    ]
    z, x, y = variants[motion % len(variants)]
    return f"zoompan=z='{z}':x='{x}':y='{y}':d={frames}:s={w}x{h}:fps={fps}"


def _grade(spec: RenderSpec, clip: Clip) -> str:
    look = LOOKS.get(spec.look, "") if clip.graded else ""
    parts = [look] if look else []
    if spec.letterbox and spec.width > spec.height:
        bar = int(spec.height * 0.1)
        parts.append(f"drawbox=x=0:y=0:w=iw:h={bar}:color=black:t=fill,drawbox=x=0:y=ih-{bar}:w=iw:h={bar}:color=black:t=fill")
    return ("," + ",".join(parts)) if parts else ""


def _fade_filters(spec: RenderSpec, duration: float) -> tuple[str, str]:
    if spec.transition != "fade_black":
        return "", ""
    d = min(spec.transition_duration / 2 + 0.1, duration / 4)
    out = max(duration - d, 0)
    return (
        f",fade=t=in:st=0:d={d:.2f},fade=t=out:st={out:.2f}:d={d:.2f}",
        f",afade=t=in:st=0:d={d:.2f},afade=t=out:st={out:.2f}:d={d:.2f}",
    )


def _overlay_input(path: Path, dur: float) -> list[str]:
    if path.suffix == ".ffconcat":
        return ["-f", "concat", "-safe", "0", "-i", str(path)]
    return ["-loop", "1", "-t", f"{dur}", "-i", str(path)]


async def render_segment(clip: Clip, spec: RenderSpec, out: Path, work: Path) -> Path:
    W, H, fps = spec.width, spec.height, spec.fps
    dur = clip.length
    vfade, afade = _fade_filters(spec, dur)
    grade = _grade(spec, clip)
    args: list[str] = []
    filters: list[str] = []
    n_inputs = 0

    if clip.kind in ("image", "card"):
        animate = spec.ken_burns or clip.kind == "card"
        scale = KB_UPSCALE.get(spec.quality, 2.0) if animate else 1.0
        still = work / f"{out.stem}_still.jpg"
        await asyncio.to_thread(_compose_still, clip.src, still, (int(W * scale) // 2 * 2, int(H * scale) // 2 * 2),
                                clip.focus, spec.framing, clip.kind == "card")
        frames = int(round(dur * fps))
        args += ["-i", str(still)]
        n_inputs += 1
        if animate:
            strength = 0.06 if clip.kind == "card" else 0.13
            filters.append(f"[0:v]{_zoompan(clip.motion, frames, W, H, fps, clip.focus, strength)},setsar=1{grade},format=yuv420p[base]")
        else:
            filters.append(f"[0:v]loop=loop=-1:size=1:start=0,fps={fps},trim=duration={dur},scale={W}:{H},setsar=1{grade},format=yuv420p[base]")
        audio_label = None
    else:
        src_dur = clip.source_duration or (clip.clip_start + clip.duration)
        take = max(0.5, min(clip.duration, src_dur - clip.clip_start))
        args += ["-ss", f"{clip.clip_start:.3f}", "-t", f"{take:.3f}", "-i", str(clip.src)]
        n_inputs += 1
        pad = max(0.0, dur - take)
        fx, fy = clip.focus
        tail = f"fps={fps},setsar=1{grade},format=yuv420p,tpad=stop_mode=clone:stop_duration={pad + 0.1:.3f},trim=duration={dur}[base]"
        if spec.framing == "smart_crop":
            filters.append(
                f"[0:v]scale={W}:{H}:force_original_aspect_ratio=increase,"
                f"crop={W}:{H}:x='(iw-ow)*{fx:.3f}':y='(ih-oh)*{fy:.3f}',{tail}")
        else:
            filters.append(
                f"[0:v]split=2[bgsrc][fgsrc];"
                f"[bgsrc]scale={W}:{H}:force_original_aspect_ratio=increase,crop={W}:{H},boxblur=20:2,eq=brightness=-0.12[bg];"
                f"[fgsrc]scale={W}:{H}:force_original_aspect_ratio=decrease[fg];"
                f"[bg][fg]overlay=(W-w)/2:(H-h)/2,{tail}")
        audio_label = "0:a" if (clip.has_audio and spec.clip_audio == "keep") else None

    if clip.overlay:
        args += _overlay_input(clip.overlay, dur)
        filters.append(f"[{n_inputs}:v]fps={fps},format=rgba[ov];[base][ov]overlay=0:0:eof_action=pass:format=auto[comp]")
        n_inputs += 1
        vlabel = "comp"
    else:
        vlabel = "base"
    filters.append(f"[{vlabel}]format=yuv420p{vfade}[vout]")

    if audio_label:
        filters.append(
            f"[{audio_label}]aresample=48000,aformat=channel_layouts=stereo,volume={spec.clip_audio_volume},"
            f"apad,atrim=duration={dur}{afade}[aout]")
    else:
        args += ["-f", "lavfi", "-t", f"{dur}", "-i", "anullsrc=channel_layout=stereo:sample_rate=48000"]
        filters.append(f"[{n_inputs}:a]atrim=duration={dur}[aout]")
        n_inputs += 1

    await run_ffmpeg([*args, "-filter_complex", ";".join(filters), "-map", "[vout]", "-map", "[aout]",
                      "-t", f"{dur}", *_encode_args(spec), str(out)], timeout=900)
    return out


async def join_segments(segments: list[Path], timeline: Timeline, spec: RenderSpec, out: Path, work: Path) -> Path:
    if len(segments) == 1 or not spec.uses_xfade:
        listing = work / "concat.txt"
        listing.write_text("".join(f"file '{p.resolve()}'\n" for p in segments))
        await run_ffmpeg(["-f", "concat", "-safe", "0", "-i", str(listing), "-c", "copy", str(out)])
        return out
    args: list[str] = []
    for seg in segments:
        args += ["-i", str(seg)]
    vf, af = [], []
    prev_v, prev_a = "0:v", "0:a"
    for i in range(1, len(segments)):
        b = timeline.boundaries[i - 1]
        offset = timeline.starts[i]
        nv, na = f"v{i}", f"a{i}"
        vf.append(f"[{prev_v}][{i}:v]xfade=transition={b.name}:duration={b.duration:.3f}:offset={offset:.3f}[{nv}]")
        af.append(f"[{prev_a}][{i}:a]acrossfade=d={b.duration:.3f}:c1=tri:c2=tri[{na}]")
        prev_v, prev_a = nv, na
    await run_ffmpeg([*args, "-filter_complex", ";".join(vf + af), "-map", f"[{prev_v}]", "-map", f"[{prev_a}]",
                      *_encode_args(spec), str(out)], timeout=1800)
    return out


async def build_fx_track(parts: list[tuple[float, Path, float]], total: float, out: Path) -> Path | None:
    """Place audio clips (narration, SFX) at their start times, with gains, on one full-length track."""
    if not parts:
        return None
    args, filters = [], []
    for i, (start, path, gain) in enumerate(parts):
        args += ["-i", str(path)]
        ms = int(max(start, 0) * 1000)
        filters.append(f"[{i}:a]aresample=48000,aformat=channel_layouts=stereo,volume={gain:.3f},adelay={ms}|{ms}[n{i}]")
    mix = "".join(f"[n{i}]" for i in range(len(parts)))
    filters.append(f"{mix}amix=inputs={len(parts)}:normalize=0:dropout_transition=0,apad,atrim=duration={total:.3f}[vo]")
    await run_ffmpeg([*args, "-filter_complex", ";".join(filters), "-map", "[vo]", "-c:a", "pcm_s16le", str(out)])
    return out


async def final_mix(video: Path, total: float, out: Path, music: Path | None, music_volume: float,
                    voice: Path | None, sfx: Path | None, subtitles: Path | None, spec: RenderSpec) -> Path:
    args = ["-i", str(video)]
    filters = ["[0:a]aresample=48000[clip]"]
    mix = ["[clip]"]
    idx = 1
    voice_idx = music_idx = sfx_idx = sub_idx = None
    if voice:
        voice_idx, idx = idx, idx + 1
        args += ["-i", str(voice)]
    if sfx:
        sfx_idx, idx = idx, idx + 1
        args += ["-i", str(sfx)]
    if music:
        music_idx, idx = idx, idx + 1
        args += ["-stream_loop", "-1", "-i", str(music)]
    if subtitles:
        sub_idx, idx = idx, idx + 1
        args += _overlay_input(subtitles, total)
    if voice_idx is not None:
        filters.append(f"[{voice_idx}:a]aresample=48000,aformat=channel_layouts=stereo,volume=1.15,asplit=2[vo][vosc]")
        mix.append("[vo]")
    if sfx_idx is not None:
        filters.append(f"[{sfx_idx}:a]aresample=48000,aformat=channel_layouts=stereo[sfx]")
        mix.append("[sfx]")
    if music_idx is not None:
        fade_out = max(total - 2.5, 0)
        filters.append(
            f"[{music_idx}:a]aresample=48000,aformat=channel_layouts=stereo,atrim=duration={total:.3f},"
            f"volume={music_volume:.3f},afade=t=in:st=0:d=0.8,afade=t=out:st={fade_out:.2f}:d=2.5[mus]")
        if voice_idx is not None:
            filters.append("[mus][vosc]sidechaincompress=threshold=0.02:ratio=8:attack=30:release=500[musd]")
            mix.append("[musd]")
        else:
            mix.append("[mus]")
    elif voice_idx is not None:
        filters.append("[vosc]anullsink")
    filters.append(f"{''.join(mix)}amix=inputs={len(mix)}:normalize=0:dropout_transition=0,"
                   f"alimiter=limit=0.95,atrim=duration={total:.3f}[aout]")
    if sub_idx is not None:
        filters.append(f"[{sub_idx}:v]fps={spec.fps},format=rgba[subs];[0:v][subs]overlay=0:0:eof_action=pass:format=auto,"
                       f"format=yuv420p[vout]")
        video_map, video_codec = "[vout]", _encode_args(spec, audio=False)
    else:
        video_map, video_codec = "0:v", ["-c:v", "copy"]
    await run_ffmpeg([*args, "-filter_complex", ";".join(filters), "-map", video_map, "-map", "[aout]",
                      *video_codec, "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", "-t", f"{total:.3f}",
                      str(out)], timeout=1800)
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
    sfx_parts: list[tuple[float, Path, float]] | None = None,
    subtitles: Path | None = None,
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
    joined = await join_segments(list(segments), timeline, spec, work / "joined.mp4", work)
    voice = await build_fx_track([(s, p, 1.0) for s, p in voice_parts], timeline.total, work / "voice.wav")
    sfx = await build_fx_track(sfx_parts or [], timeline.total, work / "sfx.wav")
    return await final_mix(joined, timeline.total, out, music, music_volume, voice, sfx, subtitles, spec)
