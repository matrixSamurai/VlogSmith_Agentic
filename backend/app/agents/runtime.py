"""Per-job runtime objects that are not JSON-serializable (paths, clients, callbacks).

ADK session state must stay serializable, so workflow nodes only keep the
job id in state and look up everything else here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable

from google.adk.models.base_llm import BaseLlm

from ..costs import CostTracker
from ..models import BlogSettings, MediaRef
from ..storage import Storage

ReportFn = Callable[[str, str, float | None, str], Awaitable[None]]


@dataclass
class LocalMedia:
    ref: MediaRef
    path: Path  # normalized local file (JPEG for images)
    width: int | None = None
    height: int | None = None
    duration: float = 0.0
    has_audio: bool = False
    frames: list[tuple[float, Path]] = field(default_factory=list)


@dataclass
class JobRuntime:
    job_id: str
    settings: BlogSettings
    media: list[MediaRef]
    storage: Storage
    work_dir: Path
    report: ReportFn
    models: dict[str, str | BaseLlm]
    local: dict[str, LocalMedia] = field(default_factory=dict)
    music_upload: MediaRef | None = None
    extras: dict[str, Any] = field(default_factory=dict)
    costs: CostTracker = field(default_factory=CostTracker)

    async def info(self, stage: str, message: str, progress: float | None = None) -> None:
        await self.report(stage, message, progress, "info")

    async def warn(self, stage: str, message: str, progress: float | None = None) -> None:
        await self.report(stage, message, progress, "warning")


_RUNTIMES: dict[str, JobRuntime] = {}


def register(rt: JobRuntime) -> None:
    _RUNTIMES[rt.job_id] = rt


def unregister(job_id: str) -> None:
    _RUNTIMES.pop(job_id, None)


def get_runtime(job_id: str) -> JobRuntime:
    return _RUNTIMES[job_id]
