"""Thin async wrappers around the ffmpeg binary."""

from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import dataclass
from pathlib import Path

from ..config import get_settings

log = logging.getLogger(__name__)


class FFmpegError(RuntimeError):
    pass


@dataclass
class ProbeResult:
    duration: float
    width: int | None
    height: int | None
    has_video: bool
    has_audio: bool


_DURATION_RE = re.compile(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)")
_VIDEO_RE = re.compile(r"Stream #\S+.*?: Video: .*?(\d{2,5})x(\d{2,5})")
_ROTATION_RE = re.compile(r"rotation of (-?\d+(?:\.\d+)?) degrees")


def ffmpeg_bin() -> str:
    return get_settings().resolve_ffmpeg()


async def run_ffmpeg(args: list[str], timeout: float = 900) -> str:
    """Run ffmpeg with the given args; returns stderr. Raises FFmpegError on failure."""
    cmd = [ffmpeg_bin(), "-hide_banner", "-nostdin", "-y", *args]
    log.debug("ffmpeg %s", " ".join(args))
    proc = await asyncio.create_subprocess_exec(
        *cmd, stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.PIPE
    )
    try:
        _, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except (asyncio.TimeoutError, asyncio.CancelledError):
        proc.kill()
        await proc.wait()
        raise
    text = stderr.decode(errors="replace")
    if proc.returncode != 0:
        tail = "\n".join(text.strip().splitlines()[-12:])
        raise FFmpegError(f"ffmpeg failed ({proc.returncode}): {tail}")
    return text


async def probe(path: Path) -> ProbeResult:
    """Probe a media file by parsing `ffmpeg -i` output (no ffprobe dependency)."""
    proc = await asyncio.create_subprocess_exec(
        ffmpeg_bin(), "-hide_banner", "-nostdin", "-i", str(path),
        stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.PIPE,
    )
    _, stderr = await proc.communicate()
    text = stderr.decode(errors="replace")
    if "Invalid data found" in text or "No such file" in text:
        raise FFmpegError(f"unreadable media file: {path.name}")
    duration = 0.0
    if m := _DURATION_RE.search(text):
        h, mnt, s = m.groups()
        duration = int(h) * 3600 + int(mnt) * 60 + float(s)
    width = height = None
    if m := _VIDEO_RE.search(text):
        width, height = int(m.group(1)), int(m.group(2))
        if (r := _ROTATION_RE.search(text)) and abs(round(float(r.group(1)))) % 180 == 90:
            width, height = height, width  # ffmpeg auto-rotates on decode
    has_audio = bool(re.search(r"Stream #\S+.*?: Audio:", text))
    return ProbeResult(duration, width, height, width is not None, has_audio)


async def extract_frames(video: Path, out_dir: Path, count: int, duration: float, width: int = 768) -> list[tuple[float, Path]]:
    """Grab `count` evenly spaced JPEG frames; returns (timestamp, path) pairs."""
    out_dir.mkdir(parents=True, exist_ok=True)
    count = max(1, count)
    usable = max(duration - 0.2, 0.1)
    stamps = [round(usable * (i + 0.5) / count, 2) for i in range(count)]
    results: list[tuple[float, Path]] = []
    for i, t in enumerate(stamps):
        out = out_dir / f"{video.stem}_f{i:02d}.jpg"
        try:
            await run_ffmpeg(
                ["-ss", f"{t}", "-i", str(video), "-frames:v", "1", "-vf", f"scale={width}:-2", "-q:v", "4", str(out)],
                timeout=120,
            )
            if out.exists():
                results.append((t, out))
        except FFmpegError as exc:
            log.warning("frame extraction failed at %.2fs: %s", t, exc)
    return results
