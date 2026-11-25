"""Persistence for uploads, jobs and job progress events.

SQLite for local development, Firestore for Cloud Run (so any instance can
serve progress for a job running on another instance).
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
import threading
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from .config import Settings
from .models import Job, JobEvent, UploadRecord, now


class JobStore(ABC):
    @abstractmethod
    async def put_upload(self, rec: UploadRecord) -> None: ...

    @abstractmethod
    async def get_upload(self, upload_id: str) -> UploadRecord | None: ...

    @abstractmethod
    async def create_job(self, job: Job) -> None: ...

    @abstractmethod
    async def get_job(self, job_id: str) -> Job | None: ...

    @abstractmethod
    async def update_job(self, job_id: str, **fields: Any) -> None: ...

    @abstractmethod
    async def list_jobs(self, limit: int = 50) -> list[Job]: ...

    @abstractmethod
    async def delete_job(self, job_id: str) -> None: ...

    @abstractmethod
    async def add_event(self, job_id: str, event: JobEvent) -> JobEvent: ...

    @abstractmethod
    async def list_events(self, job_id: str, after_seq: int = 0) -> list[JobEvent]: ...

    @abstractmethod
    async def list_active_jobs(self) -> list[Job]: ...


def _dump(model) -> dict:
    return model.model_dump(mode="json")


class SQLiteJobStore(JobStore):
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(path), check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._lock = threading.Lock()
        with self._lock:
            self._conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS uploads (id TEXT PRIMARY KEY, data TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS jobs (id TEXT PRIMARY KEY, created_at REAL, status TEXT, data TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS events (job_id TEXT, seq INTEGER, data TEXT NOT NULL, PRIMARY KEY (job_id, seq));
                CREATE INDEX IF NOT EXISTS jobs_created ON jobs(created_at DESC);
                """
            )
            self._conn.commit()

    async def _run(self, fn, *args):
        def locked():
            with self._lock:
                return fn(*args)

        return await asyncio.to_thread(locked)

    async def put_upload(self, rec):
        def op():
            self._conn.execute("INSERT OR REPLACE INTO uploads VALUES (?, ?)", (rec.id, json.dumps(_dump(rec))))
            self._conn.commit()

        await self._run(op)

    async def get_upload(self, upload_id):
        row = await self._run(lambda: self._conn.execute("SELECT data FROM uploads WHERE id=?", (upload_id,)).fetchone())
        return UploadRecord.model_validate_json(row[0]) if row else None

    async def create_job(self, job):
        def op():
            self._conn.execute(
                "INSERT INTO jobs VALUES (?, ?, ?, ?)", (job.id, job.created_at, job.status, json.dumps(_dump(job)))
            )
            self._conn.commit()

        await self._run(op)

    async def get_job(self, job_id):
        row = await self._run(lambda: self._conn.execute("SELECT data FROM jobs WHERE id=?", (job_id,)).fetchone())
        return Job.model_validate_json(row[0]) if row else None

    async def update_job(self, job_id, **fields):
        def op():
            row = self._conn.execute("SELECT data FROM jobs WHERE id=?", (job_id,)).fetchone()
            if not row:
                return
            data = json.loads(row[0])
            for k, v in fields.items():
                data[k] = v.model_dump(mode="json") if hasattr(v, "model_dump") else v
            data["updated_at"] = now()
            job = Job.model_validate(data)
            self._conn.execute(
                "UPDATE jobs SET status=?, data=? WHERE id=?", (job.status, json.dumps(_dump(job)), job_id)
            )
            self._conn.commit()

        await self._run(op)

    async def list_jobs(self, limit=50):
        rows = await self._run(
            lambda: self._conn.execute("SELECT data FROM jobs ORDER BY created_at DESC LIMIT ?", (limit,)).fetchall()
        )
        return [Job.model_validate_json(r[0]) for r in rows]

    async def delete_job(self, job_id):
        def op():
            self._conn.execute("DELETE FROM jobs WHERE id=?", (job_id,))
            self._conn.execute("DELETE FROM events WHERE job_id=?", (job_id,))
            self._conn.commit()

        await self._run(op)

    async def add_event(self, job_id, event):
        def op():
            row = self._conn.execute("SELECT COALESCE(MAX(seq), 0) FROM events WHERE job_id=?", (job_id,)).fetchone()
            event.seq = row[0] + 1
            self._conn.execute("INSERT INTO events VALUES (?, ?, ?)", (job_id, event.seq, json.dumps(_dump(event))))
            self._conn.commit()
            return event

        return await self._run(op)

    async def list_events(self, job_id, after_seq=0):
        rows = await self._run(
            lambda: self._conn.execute(
                "SELECT data FROM events WHERE job_id=? AND seq>? ORDER BY seq", (job_id, after_seq)
            ).fetchall()
        )
        return [JobEvent.model_validate_json(r[0]) for r in rows]

    async def list_active_jobs(self):
        rows = await self._run(
            lambda: self._conn.execute("SELECT data FROM jobs WHERE status IN ('queued','running')").fetchall()
        )
        return [Job.model_validate_json(r[0]) for r in rows]


class FirestoreJobStore(JobStore):
    def __init__(self, project: str | None, database: str):
        from google.cloud import firestore

        self._fs = firestore
        self._db = firestore.AsyncClient(project=project, database=database)
        self._uploads = self._db.collection("uploads")
        self._jobs = self._db.collection("jobs")

    async def put_upload(self, rec):
        await self._uploads.document(rec.id).set(_dump(rec))

    async def get_upload(self, upload_id):
        snap = await self._uploads.document(upload_id).get()
        return UploadRecord.model_validate(snap.to_dict()) if snap.exists else None

    async def create_job(self, job):
        await self._jobs.document(job.id).set({**_dump(job), "event_seq": 0})

    async def get_job(self, job_id):
        snap = await self._jobs.document(job_id).get()
        return Job.model_validate(snap.to_dict()) if snap.exists else None

    async def update_job(self, job_id, **fields):
        data = {k: (v.model_dump(mode="json") if hasattr(v, "model_dump") else v) for k, v in fields.items()}
        data["updated_at"] = now()
        await self._jobs.document(job_id).update(data)

    async def list_jobs(self, limit=50):
        q = self._jobs.order_by("created_at", direction=self._fs.Query.DESCENDING).limit(limit)
        return [Job.model_validate(s.to_dict()) async for s in q.stream()]

    async def delete_job(self, job_id):
        ref = self._jobs.document(job_id)
        async for ev in ref.collection("events").stream():
            await ev.reference.delete()
        await ref.delete()

    async def add_event(self, job_id, event):
        ref = self._jobs.document(job_id)
        transaction = self._db.transaction()

        @self._fs.async_transactional
        async def txn(tx):
            snap = await ref.get(transaction=tx)
            seq = int((snap.to_dict() or {}).get("event_seq", 0)) + 1
            event.seq = seq
            tx.update(ref, {"event_seq": seq})
            tx.set(ref.collection("events").document(f"{seq:06d}"), _dump(event))

        await txn(transaction)
        return event

    async def list_events(self, job_id, after_seq=0):
        q = self._jobs.document(job_id).collection("events").where(
            filter=self._fs.FieldFilter("seq", ">", after_seq)
        ).order_by("seq")
        return [JobEvent.model_validate(s.to_dict()) async for s in q.stream()]

    async def list_active_jobs(self):
        q = self._jobs.where(filter=self._fs.FieldFilter("status", "in", ["queued", "running"]))
        return [Job.model_validate(s.to_dict()) async for s in q.stream()]


def build_jobstore(settings: Settings) -> JobStore:
    if settings.jobstore_backend == "firestore":
        return FirestoreJobStore(settings.google_cloud_project, settings.firestore_database)
    return SQLiteJobStore(settings.local_data_dir / "vlogsmith.db")
