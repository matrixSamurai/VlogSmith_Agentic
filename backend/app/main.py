"""FastAPI application: REST API, SSE progress stream and the built web UI."""

from __future__ import annotations

import asyncio
import json
import logging
import re
from contextlib import asynccontextmanager
from pathlib import Path, PurePosixPath

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles

from .config import get_settings
from .jobstore import build_jobstore
from .models import (
    MUSIC_MOODS, TONES, TRANSITIONS, VOICES, BlogSettings, CreateJobRequest, Job, UploadRecord, UploadRequest,
    UploadTicket, new_id,
)
from .storage import build_storage
from .worker import JobManager, media_ref_from_upload

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("vlogsmith")

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp"}
VIDEO_EXT = {".mp4", ".mov", ".m4v", ".webm", ".mkv", ".avi"}
AUDIO_EXT = {".mp3", ".wav", ".m4a", ".aac", ".ogg", ".flac"}
OUTPUT_FILE_RE = re.compile(r"^(vlog\.mp4|thumbnail\.jpg|article\.(html|md)|blog_package\.zip|media/m\d+(_poster)?\.(jpg|mp4))$")
CONTENT_TYPES = {".mp4": "video/mp4", ".jpg": "image/jpeg", ".html": "text/html; charset=utf-8",
                 ".md": "text/markdown; charset=utf-8", ".zip": "application/zip"}


def classify(filename: str, content_type: str) -> str | None:
    ext = PurePosixPath(filename.lower()).suffix
    if ext in IMAGE_EXT and content_type.startswith("image/"):
        return "image"
    if ext in VIDEO_EXT and content_type.startswith("video/"):
        return "video"
    if ext in AUDIO_EXT and content_type.startswith("audio/"):
        return "audio"
    return None


def safe_name(filename: str) -> str:
    name = PurePosixPath(filename).name
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("._") or "file"
    return stem[-120:]


def create_app(manager: JobManager | None = None) -> FastAPI:
    settings = get_settings()
    store = manager.store if manager else build_jobstore(settings)
    storage = manager.storage if manager else build_storage(settings)
    jobs = manager or JobManager(settings, store, storage)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        await jobs.recover()
        if not settings.llm_configured:
            log.warning("No Gemini credentials found: set GOOGLE_API_KEY or GOOGLE_GENAI_USE_VERTEXAI=TRUE")
        yield
        await jobs.shutdown()

    app = FastAPI(title=f"{settings.app_name} API", version="1.0.0", lifespan=lifespan)
    app.state.jobs = jobs
    app.add_middleware(CORSMiddleware, allow_origins=settings.cors_origins, allow_methods=["*"], allow_headers=["*"])

    limits = {"image": settings.max_image_mb, "video": settings.max_video_mb, "audio": settings.max_audio_mb}

    # ------------------------------------------------------------------ meta
    @app.get("/api/health")
    async def health():
        return {"ok": True}

    @app.get("/api/config")
    async def config():
        return {
            "app_name": settings.app_name,
            "llm_configured": settings.llm_configured,
            "vertex_enabled": settings.vertex_enabled,
            "defaults": json.loads(BlogSettings(context="...").model_dump_json()),
            "options": {"tones": TONES, "music_moods": MUSIC_MOODS, "voices": VOICES, "transitions": TRANSITIONS},
            "limits": {"max_files": settings.max_files_per_job, "max_mb": limits},
            "accept": {"image": sorted(IMAGE_EXT), "video": sorted(VIDEO_EXT), "audio": sorted(AUDIO_EXT)},
        }

    # ------------------------------------------------------------------ uploads
    @app.post("/api/uploads", response_model=UploadTicket)
    async def create_upload(req: UploadRequest):
        kind = classify(req.filename, req.content_type)
        if not kind:
            raise HTTPException(415, f"Unsupported file type: {req.filename}")
        if req.size > limits[kind] * 1024 * 1024:
            raise HTTPException(413, f"{req.filename} is larger than {limits[kind]} MB")
        upload_id = new_id("up")
        key = f"uploads/{upload_id}/{safe_name(req.filename)}"
        rec = UploadRecord(id=upload_id, filename=PurePosixPath(req.filename).name[:255], content_type=req.content_type,
                           kind=kind, size=req.size, storage_key=key)
        await store.put_upload(rec)
        url, headers = storage.create_upload_url(key, req.content_type, upload_id)
        return UploadTicket(upload_id=upload_id, upload_url=url, headers=headers)

    @app.put("/api/uploads/{upload_id}/content")
    async def put_upload_content(upload_id: str, request: Request):
        rec = await store.get_upload(upload_id)
        if not rec or rec.status == "complete":
            raise HTTPException(404, "Unknown or already completed upload")
        if settings.storage_backend != "local":
            raise HTTPException(400, "Upload directly to the signed URL")
        try:
            written = await storage.write_stream(rec.storage_key, request.stream(), limits[rec.kind] * 1024 * 1024)
        except ValueError as exc:
            raise HTTPException(413, str(exc)) from exc
        rec.size, rec.status = written, "complete"
        await store.put_upload(rec)
        return {"ok": True, "size": written}

    @app.post("/api/uploads/{upload_id}/complete")
    async def complete_upload(upload_id: str):
        rec = await store.get_upload(upload_id)
        if not rec:
            raise HTTPException(404, "Unknown upload")
        size = await storage.size(rec.storage_key)
        if size is None:
            raise HTTPException(409, "File has not been uploaded yet")
        if size > limits[rec.kind] * 1024 * 1024:
            raise HTTPException(413, "File exceeds the size limit")
        rec.size, rec.status = size, "complete"
        await store.put_upload(rec)
        return {"ok": True, "kind": rec.kind, "size": size}

    # ------------------------------------------------------------------ jobs
    @app.post("/api/jobs", response_model=Job, status_code=201)
    async def create_job(req: CreateJobRequest):
        if not settings.llm_configured and manager is None:
            raise HTTPException(503, "The server has no Gemini credentials configured.")
        ids = list(dict.fromkeys(req.upload_ids + ([req.settings.music_upload_id] if req.settings.music_upload_id else [])))
        if len(ids) > settings.max_files_per_job + 1:
            raise HTTPException(400, f"At most {settings.max_files_per_job} files per blog")
        uploads = []
        for uid in ids:
            rec = await store.get_upload(uid)
            if not rec or rec.status != "complete":
                raise HTTPException(400, f"Upload {uid} is missing or incomplete")
            uploads.append(rec)
        visual = [u for u in uploads if u.kind in ("image", "video")]
        if not visual:
            raise HTTPException(400, "Add at least one photo or video")
        music = [u for u in uploads if u.kind == "audio"][:1]
        if req.settings.music_source == "upload" and not music:
            raise HTTPException(400, "Choose a music file or another music source")
        media = [media_ref_from_upload(i, u) for i, u in enumerate(visual)]
        media += [media_ref_from_upload(len(media), u) for u in music]
        job = Job(id=new_id("job"), settings=req.settings, media=media, title=req.settings.title)
        await store.create_job(job)
        jobs.submit(job)
        return job

    @app.get("/api/jobs", response_model=list[Job])
    async def list_jobs(limit: int = 50):
        return await store.list_jobs(min(max(limit, 1), 200))

    async def _job(job_id: str) -> Job:
        job = await store.get_job(job_id)
        if not job:
            raise HTTPException(404, "Job not found")
        return job

    @app.get("/api/jobs/{job_id}", response_model=Job)
    async def get_job(job_id: str):
        return await _job(job_id)

    @app.post("/api/jobs/{job_id}/cancel")
    async def cancel_job(job_id: str):
        job = await _job(job_id)
        if job.status in ("queued", "running"):
            await jobs.cancel(job_id)
        return {"ok": True}

    @app.post("/api/jobs/{job_id}/retry", response_model=Job, status_code=201)
    async def retry_job(job_id: str):
        old = await _job(job_id)
        if old.status in ("queued", "running"):
            raise HTTPException(409, "Job is still running")
        job = Job(id=new_id("job"), settings=old.settings, media=old.media, title=old.settings.title)
        await store.create_job(job)
        jobs.submit(job)
        return job

    @app.delete("/api/jobs/{job_id}", status_code=204)
    async def delete_job(job_id: str):
        job = await _job(job_id)
        if job.status in ("queued", "running"):
            raise HTTPException(409, "Cancel the job before deleting it")
        await store.delete_job(job_id)
        return Response(status_code=204)

    @app.get("/api/jobs/{job_id}/events")
    async def job_events(job_id: str, request: Request):
        await _job(job_id)
        try:
            last = int(request.headers.get("last-event-id") or request.query_params.get("after") or 0)
        except ValueError:
            last = 0

        async def stream():
            nonlocal last
            idle = 0
            while True:
                if await request.is_disconnected():
                    return
                events = await store.list_events(job_id, last)
                for ev in events:
                    last = ev.seq
                    yield f"id: {ev.seq}\nevent: progress\ndata: {ev.model_dump_json()}\n\n"
                job = await store.get_job(job_id)
                if job is None:
                    return
                if job.status in ("succeeded", "failed", "cancelled") and not events:
                    yield f"event: job\ndata: {job.model_dump_json()}\n\n"
                    return
                if events:
                    yield f"event: job\ndata: {job.model_dump_json()}\n\n"
                    idle = 0
                else:
                    idle += 1
                    if idle % 15 == 0:
                        yield ": keep-alive\n\n"
                await asyncio.sleep(1.0)

        return StreamingResponse(stream(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    @app.get("/api/jobs/{job_id}/files/{name:path}")
    async def job_file(job_id: str, name: str, download: bool = False):
        if not OUTPUT_FILE_RE.match(name):
            raise HTTPException(404, "Not found")
        job = await _job(job_id)
        if job.status != "succeeded":
            raise HTTPException(404, "Outputs are not ready")
        key = f"jobs/{job_id}/outputs/{name}"
        ext = PurePosixPath(name).suffix
        filename = f"{_slug(job.title or 'vlog')}{ext}" if download else None
        if ext == ".html":  # always proxy so relative media links keep resolving through this API
            data = await storage.read_bytes(key)
            if data is None:
                raise HTTPException(404, "Not found")
            return Response(data, media_type=CONTENT_TYPES[ext], headers={
                "Content-Security-Policy": "default-src 'self'; img-src 'self' https://storage.googleapis.com; "
                                           "media-src 'self' https://storage.googleapis.com; style-src 'unsafe-inline'; "
                                           "script-src 'none'",
                **({"Content-Disposition": f'attachment; filename="{filename}"'} if filename else {})})
        if path := storage.local_path(key):
            return FileResponse(path, media_type=CONTENT_TYPES.get(ext), filename=filename)
        url = storage.download_url(key, filename)
        if not url:
            raise HTTPException(404, "Not found")
        return RedirectResponse(url, status_code=307)

    # ------------------------------------------------------------------ web UI
    dist = Path(settings.frontend_dist)
    if dist.exists():
        app.mount("/assets", StaticFiles(directory=dist / "assets"), name="assets")

        @app.get("/{full_path:path}", include_in_schema=False)
        async def spa(full_path: str):
            if full_path.startswith("api/"):
                return JSONResponse({"detail": "Not found"}, status_code=404)
            candidate = (dist / full_path).resolve()
            if full_path and candidate.is_file() and dist.resolve() in candidate.parents:
                return FileResponse(candidate)
            return FileResponse(dist / "index.html")

    return app


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:60] or "vlog"
# Run with: uvicorn app.main:create_app --factory
