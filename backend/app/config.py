"""Application configuration, loaded from environment variables (and `.env`)."""

from __future__ import annotations

import os
import shutil
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

APP_DIR = Path(__file__).resolve().parent
ASSETS_DIR = APP_DIR / "assets"
FONTS_DIR = ASSETS_DIR / "fonts"


class Settings(BaseSettings):
    # Later files win: the project-root .env (shared with docker compose) overrides backend/.env.
    model_config = SettingsConfigDict(
        env_file=(APP_DIR.parent / ".env", APP_DIR.parent.parent / ".env"), env_file_encoding="utf-8", extra="ignore"
    )

    app_name: str = "Vlogsmith"
    environment: Literal["local", "cloud"] = "local"

    # --- Storage / persistence -------------------------------------------------
    storage_backend: Literal["local", "gcs"] = "local"
    local_data_dir: Path = Path("./data")
    gcs_bucket: str | None = None
    jobstore_backend: Literal["sqlite", "firestore"] = "sqlite"
    firestore_database: str = "(default)"
    work_dir: Path = Path(os.environ.get("TMPDIR", "/tmp")) / "vlogsmith-work"

    # --- Models ----------------------------------------------------------------
    # Auth follows the google-genai conventions: either GOOGLE_API_KEY, or
    # GOOGLE_GENAI_USE_VERTEXAI=TRUE + GOOGLE_CLOUD_PROJECT + GOOGLE_CLOUD_LOCATION.
    model_fast: str = "gemini-4-neon"
    model_smart: str = "gemini-4-neon"
    model_tts: str = "gemini-2.5-flash-preview-tts"
    lyria_model: str = "lyria-002"
    google_cloud_project: str | None = None
    google_cloud_location: str = "us-central1"
    google_genai_use_vertexai: bool = False
    google_api_key: str | None = None

    # --- Limits ----------------------------------------------------------------
    max_concurrent_jobs: int = 2
    # Size of the compute instance, used for the per-job compute cost (0 = local run, not billed).
    compute_vcpu: float = 0
    compute_memory_gib: float = 0
    max_files_per_job: int = 40
    max_image_mb: int = 25
    max_video_mb: int = 1024
    max_audio_mb: int = 50
    render_concurrency: int = Field(default_factory=lambda: max(1, (os.cpu_count() or 2) // 2))
    llm_concurrency: int = 6
    job_heartbeat_seconds: int = 15
    job_stale_seconds: int = 120

    # --- Web -------------------------------------------------------------------
    cors_origins: list[str] = ["http://localhost:5173"]
    frontend_dist: Path = APP_DIR.parent.parent / "frontend" / "dist"
    ffmpeg_bin: str | None = None

    @property
    def vertex_enabled(self) -> bool:
        return self.google_genai_use_vertexai and bool(self.google_cloud_project)

    @property
    def llm_configured(self) -> bool:
        return self.vertex_enabled or bool(self.google_api_key or os.environ.get("GEMINI_API_KEY"))

    def resolve_ffmpeg(self) -> str:
        if self.ffmpeg_bin:
            return self.ffmpeg_bin
        found = shutil.which("ffmpeg")
        if found:
            return found
        import imageio_ffmpeg  # bundled static binary, used for local development

        return imageio_ffmpeg.get_ffmpeg_exe()


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    # google-genai / ADK read these from the process environment.
    if settings.google_genai_use_vertexai:
        os.environ.setdefault("GOOGLE_GENAI_USE_VERTEXAI", "TRUE")
        if settings.google_cloud_project:
            os.environ.setdefault("GOOGLE_CLOUD_PROJECT", settings.google_cloud_project)
        os.environ.setdefault("GOOGLE_CLOUD_LOCATION", settings.google_cloud_location)
    elif settings.google_api_key:
        os.environ.setdefault("GOOGLE_API_KEY", settings.google_api_key)
    return settings
