"""Runs blog pipelines in the background and records progress events."""

from __future__ import annotations

import asyncio
import logging
import shutil
import traceback
from pathlib import Path

from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

from .agents import runtime
from .agents.pipeline import build_workflow
from .agents.runtime import JobRuntime
from .agents.trace import TracePlugin, summarize
from .config import Settings
from .costs import format_usd
from .jobstore import JobStore
from .models import Job, JobEvent, JobOutputs, MediaRef, now
from .storage import Storage

log = logging.getLogger(__name__)

APP_NAME = "vlogsmith"


class JobCancelled(Exception):
    pass


class JobManager:
    def __init__(self, settings: Settings, store: JobStore, storage: Storage, model_overrides: dict | None = None):
        self.settings = settings
        self.store = store
        self.storage = storage
        self._sem = asyncio.Semaphore(settings.max_concurrent_jobs)
        self._tasks: dict[str, asyncio.Task] = {}
        self._models = model_overrides or {"fast": settings.model_fast, "smart": settings.model_smart}

    # ------------------------------------------------------------- lifecycle
    async def recover(self) -> None:
        """Fail jobs whose owning instance stopped heartbeating (e.g. instance restart)."""
        for job in await self.store.list_active_jobs():
            if job.id in self._tasks:
                continue
            if now() - job.heartbeat_at > self.settings.job_stale_seconds:
                await self.store.update_job(job.id, status="failed", stage="failed", finished_at=now(),
                                            error="The server restarted while this blog was being created. Please retry.")
                await self.store.add_event(job.id, JobEvent(stage="failed", level="error",
                                                             message="Interrupted by a server restart"))

    async def shutdown(self) -> None:
        for task in list(self._tasks.values()):
            task.cancel()
        await asyncio.gather(*self._tasks.values(), return_exceptions=True)

    def submit(self, job: Job) -> None:
        task = asyncio.create_task(self._run(job.id), name=f"job-{job.id}")
        self._tasks[job.id] = task
        task.add_done_callback(lambda _: self._tasks.pop(job.id, None))

    def is_local(self, job_id: str) -> bool:
        return job_id in self._tasks

    async def cancel(self, job_id: str) -> None:
        await self.store.update_job(job_id, cancel_requested=True)
        if task := self._tasks.get(job_id):
            task.cancel()

    # ------------------------------------------------------------- execution
    async def _report(self, job_id: str, stage: str, message: str, progress: float | None, level: str) -> None:
        await self.store.add_event(job_id, JobEvent(stage=stage, message=message, progress=progress, level=level))
        fields = {"stage": stage, "heartbeat_at": now()}
        if progress is not None:
            fields["progress"] = round(min(progress, 0.99), 3)
        await self.store.update_job(job_id, **fields)

    async def _heartbeat(self, job_id: str, abort: asyncio.Event) -> None:
        while True:
            await asyncio.sleep(self.settings.job_heartbeat_seconds)
            await self.store.update_job(job_id, heartbeat_at=now())
            job = await self.store.get_job(job_id)
            if job and job.cancel_requested:
                abort.set()
                if task := self._tasks.get(job_id):
                    task.cancel()
                return

    async def _run(self, job_id: str) -> None:
        job = await self.store.get_job(job_id)
        if not job:
            return
        await self.store.add_event(job_id, JobEvent(stage="queued", message="Waiting for a free worker"))
        async with self._sem:
            await self._execute(job)

    async def _execute(self, job: Job) -> None:
        job_id = job.id
        work = Path(self.settings.work_dir) / job_id
        work.mkdir(parents=True, exist_ok=True)
        await self.store.update_job(job_id, status="running", stage="ingest", started_at=now(), heartbeat_at=now())

        media = [m for m in job.media if m.kind != "audio"]
        music = next((m for m in job.media if m.kind == "audio"), None)

        async def report(stage, message, progress, level):
            await self._report(job_id, stage, message, progress, level)

        rt = JobRuntime(job_id=job_id, settings=job.settings, media=media, storage=self.storage, work_dir=work,
                        report=report, models=self._models, music_upload=music)
        runtime.register(rt)
        abort = asyncio.Event()
        hb = asyncio.create_task(self._heartbeat(job_id, abort))
        started = now()
        try:
            result = await self._run_workflow(rt, abort)
            await self.store.add_event(job_id, JobEvent(stage="done", level="success", progress=1.0,
                                                        message="Your blog is ready"))
            # Cost is recorded before the terminal status so progress streams always deliver it as the last event.
            await self._record_cost(job_id, rt, started, final=True)
            await self.store.update_job(
                job_id, status="succeeded", stage="done", progress=1.0, finished_at=now(),
                outputs=JobOutputs.model_validate(result["outputs"]), title=result.get("title"),
                summary=result.get("summary"), duration=result.get("duration"))
        except asyncio.CancelledError:
            await self.store.add_event(job_id, JobEvent(stage="cancelled", level="warning", message="Cancelled"))
            await self._record_cost(job_id, rt, started, final=False)
            await self.store.update_job(job_id, status="cancelled", stage="cancelled", finished_at=now())
        except Exception as exc:  # noqa: BLE001 - surface any failure to the user
            log.error("job %s failed: %s\n%s", job_id, exc, traceback.format_exc())
            message = _friendly_error(exc)
            await self.store.add_event(job_id, JobEvent(stage="failed", level="error", message=message))
            await self._record_cost(job_id, rt, started, final=False)
            await self.store.update_job(job_id, status="failed", stage="failed", finished_at=now(), error=message)
        finally:
            hb.cancel()
            runtime.unregister(job_id)
            shutil.rmtree(work, ignore_errors=True)

    async def _record_cost(self, job_id: str, rt: JobRuntime, started: float, final: bool) -> None:
        """Store the job's cost and announce it as the last activity event (also for failed/cancelled runs)."""
        try:
            cost = rt.costs.summary(now() - started, self.settings.compute_vcpu, self.settings.compute_memory_gib)
            await self.store.update_job(job_id, cost=cost)
            parts = [f"AI {format_usd(cost.ai_usd)}"]
            if cost.compute_usd:
                parts.append(f"compute {format_usd(cost.compute_usd)}")
            tokens = sum(l.input_tokens + l.output_tokens for l in cost.lines)
            label = "Generation cost" if final else "Spent before stopping"
            msg = f"{label}: {format_usd(cost.total_usd)} ({' · '.join(parts)}; {tokens:,} tokens in {sum(l.calls for l in cost.lines)} model calls)"
            if cost.unpriced_models:
                msg += f" — no price set for {', '.join(cost.unpriced_models)}"
            await self.store.add_event(job_id, JobEvent(stage="cost", level="success" if final else "info", message=msg))
        except Exception as exc:  # cost reporting must never fail a job
            log.warning("cost summary failed for %s: %s", job_id, exc)

    async def _run_workflow(self, rt: JobRuntime, abort: asyncio.Event) -> dict:
        workflow = build_workflow(rt)
        sessions = InMemorySessionService()
        async def emit(record: dict) -> None:
            await self.store.add_event(rt.job_id, JobEvent(stage="trace", message=summarize(record), data=record))

        tracer = TracePlugin(workflow, emit, rt.extras.get("node_agents", {}))
        runner = Runner(app_name=APP_NAME, node=workflow, session_service=sessions, plugins=[tracer])
        session = await sessions.create_session(app_name=APP_NAME, user_id="creator", session_id=rt.job_id)
        message = types.Content(role="user", parts=[types.Part(text=rt.settings.context)])
        async for event in runner.run_async(user_id="creator", session_id=session.id, new_message=message,
                                            abort_signal=abort):
            if event.error_message:
                raise RuntimeError(event.error_message)
        if abort.is_set():
            raise asyncio.CancelledError()
        final = await sessions.get_session(app_name=APP_NAME, user_id="creator", session_id=session.id)
        state = final.state if final else {}
        if "outputs" not in state:
            raise RuntimeError("The workflow finished without producing outputs.")
        plan = state.get("plan", {})
        return {"outputs": state["outputs"], "title": plan.get("title"), "summary": plan.get("subtitle"),
                "duration": state.get("duration")}


def _friendly_error(exc: Exception) -> str:
    text = str(exc)
    low = text.lower()
    if "api key" in low or "permission" in low or "unauthenticated" in low or "403" in low:
        return "The AI model rejected the request (check GOOGLE_API_KEY / Vertex AI permissions)."
    if "prepayment" in low or "billing" in low or "402" in low:
        return "The Gemini API project is out of credits. Add billing or credits at https://ai.studio/projects and retry."
    if "quota" in low or "429" in low or "resource_exhausted" in low:
        return "The AI model quota was exceeded. Please try again in a minute."
    if "ffmpeg" in low:
        return f"Video rendering failed: {text[-300:]}"
    return text[:500] or exc.__class__.__name__


def media_ref_from_upload(index: int, upload) -> MediaRef:
    return MediaRef(id=f"m{index + 1}", upload_id=upload.id, filename=upload.filename, kind=upload.kind,
                    content_type=upload.content_type, storage_key=upload.storage_key)
