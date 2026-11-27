"""API-facing data models: blog settings, uploads, jobs and events."""

from __future__ import annotations

import time
import uuid
from typing import Literal

from pydantic import BaseModel, Field, field_validator

from .costs import JobCost

MediaKind = Literal["image", "video", "audio"]
JobStatus = Literal["queued", "running", "succeeded", "failed", "cancelled"]

TONES = ["casual", "inspirational", "professional", "humorous", "cinematic", "educational", "romantic", "adventurous"]
MUSIC_MOODS = ["auto", "upbeat", "calm", "epic", "romantic", "adventurous", "dreamy", "corporate", "playful"]
VOICES = ["Kore", "Puck", "Charon", "Aoede", "Fenrir", "Leda", "Orus", "Zephyr"]
TRANSITIONS = ["smart", "crossfade", "fade_black", "slide", "zoom", "wipe", "none"]
LOOKS = ["auto", "natural", "warm", "cinematic", "vibrant", "moody", "bw", "none"]


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:16]}"


def now() -> float:
    return time.time()


class BlogSettings(BaseModel):
    """Everything the user can configure for a single blog run."""

    title: str | None = Field(default=None, max_length=140)
    context: str = Field(min_length=3, max_length=4000, description="What the blog is about and how it should feel.")
    tone: Literal[tuple(TONES)] = "inspirational"  # type: ignore[valid-type]
    language: str = Field(default="English", max_length=40)
    audience: str | None = Field(default=None, max_length=200)

    # Video
    target_duration: int = Field(default=75, ge=20, le=600, description="Approximate vlog length in seconds.")
    aspect_ratio: Literal["16:9", "9:16", "1:1"] = "16:9"
    resolution: Literal["720p", "1080p"] = "1080p"
    fps: Literal[24, 30] = 30
    caption_style: Literal["bold", "minimal", "cinematic", "pop", "none"] = "bold"
    caption_position: Literal["bottom", "center", "top"] = "bottom"
    subtitles: bool = Field(default=True, description="Burn in word-highlighted subtitles (needs voiceover).")
    color_look: Literal[tuple(LOOKS)] = "auto"  # type: ignore[valid-type]
    letterbox: bool = False
    framing: Literal["smart_crop", "fit_blur"] = "smart_crop"
    beat_sync: bool = True
    sfx: bool = True
    channel_name: str | None = Field(default=None, max_length=60)
    transition: Literal[tuple(TRANSITIONS)] = "smart"  # type: ignore[valid-type]
    transition_duration: float = Field(default=0.6, ge=0.2, le=1.5)
    ken_burns: bool = True
    include_title_card: bool = True
    include_outro: bool = True
    outro_text: str | None = Field(default=None, max_length=120)
    max_clip_seconds: float = Field(default=8, ge=2, le=60)
    clip_audio: Literal["keep", "mute"] = "keep"
    clip_audio_volume: float = Field(default=0.6, ge=0, le=1)

    # Audio
    music_source: Literal["generated", "ai", "upload", "none"] = "generated"
    music_mood: Literal[tuple(MUSIC_MOODS)] = "auto"  # type: ignore[valid-type]
    music_volume: float = Field(default=0.22, ge=0, le=1)
    music_upload_id: str | None = None
    voiceover: bool = False
    voice: Literal[tuple(VOICES)] = "Kore"  # type: ignore[valid-type]

    # Article + agent behaviour
    generate_article: bool = True
    article_length: Literal["short", "medium", "long"] = "medium"
    review_rounds: int = Field(default=1, ge=0, le=3)
    quality: Literal["fast", "balanced", "high"] = "balanced"

    @field_validator("title", "outro_text", "audience", "channel_name", mode="before")
    @classmethod
    def _blank_to_none(cls, v):
        if isinstance(v, str) and not v.strip():
            return None
        return v

    @property
    def frame_size(self) -> tuple[int, int]:
        short = 720 if self.resolution == "720p" else 1080
        long = 1280 if self.resolution == "720p" else 1920
        if self.aspect_ratio == "16:9":
            return long, short
        if self.aspect_ratio == "9:16":
            return short, long
        return short, short


class UploadRequest(BaseModel):
    filename: str = Field(min_length=1, max_length=255)
    content_type: str = Field(min_length=3, max_length=120)
    size: int = Field(gt=0)


class UploadRecord(BaseModel):
    id: str
    filename: str
    content_type: str
    kind: MediaKind
    size: int
    storage_key: str
    status: Literal["pending", "complete"] = "pending"
    created_at: float = Field(default_factory=now)


class UploadTicket(BaseModel):
    upload_id: str
    upload_url: str
    method: Literal["PUT"] = "PUT"
    headers: dict[str, str] = {}


class CreateJobRequest(BaseModel):
    upload_ids: list[str] = Field(min_length=1)
    settings: BlogSettings


class MediaRef(BaseModel):
    id: str  # short stable id used inside the pipeline ("m1", "m2"...)
    upload_id: str
    filename: str
    kind: MediaKind
    content_type: str
    storage_key: str


class JobEvent(BaseModel):
    seq: int = 0
    ts: float = Field(default_factory=now)
    stage: str
    level: Literal["info", "warning", "error", "success"] = "info"
    message: str
    progress: float | None = None
    data: dict | None = None  # structured payload, e.g. ADK trace records


class JobOutputs(BaseModel):
    video: str | None = None
    thumbnail: str | None = None
    article_html: str | None = None
    article_md: str | None = None
    package: str | None = None
    youtube: str | None = None


class Job(BaseModel):
    id: str
    status: JobStatus = "queued"
    stage: str = "queued"
    progress: float = 0
    settings: BlogSettings
    media: list[MediaRef]
    outputs: JobOutputs = Field(default_factory=JobOutputs)
    title: str | None = None
    summary: str | None = None
    duration: float | None = None
    error: str | None = None
    cost: JobCost | None = None
    cancel_requested: bool = False
    created_at: float = Field(default_factory=now)
    updated_at: float = Field(default_factory=now)
    heartbeat_at: float = Field(default_factory=now)
    started_at: float | None = None
    finished_at: float | None = None
