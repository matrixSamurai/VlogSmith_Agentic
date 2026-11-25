"""Blob storage abstraction: local filesystem for development, GCS for Cloud Run.

Uploads always go through a two-step "ticket" flow so large videos never pass
through the API server on Cloud Run (which caps request bodies at 32 MB):

1. ``create_upload_url`` returns a URL the browser PUTs the file to.
   Local: an API endpoint. GCS: a V4 signed URL straight to the bucket.
2. The job worker later downloads inputs with ``download_to``.
"""

from __future__ import annotations

import asyncio
import datetime as dt
import shutil
from abc import ABC, abstractmethod
from pathlib import Path
from typing import AsyncIterator

from .config import Settings


class Storage(ABC):
    @abstractmethod
    def create_upload_url(self, key: str, content_type: str, upload_id: str) -> tuple[str, dict[str, str]]: ...

    @abstractmethod
    async def size(self, key: str) -> int | None: ...

    @abstractmethod
    async def download_to(self, key: str, dest: Path) -> Path: ...

    @abstractmethod
    async def upload_file(self, src: Path, key: str, content_type: str) -> None: ...

    @abstractmethod
    def download_url(self, key: str, filename: str | None = None) -> str | None:
        """A direct URL for clients, or None if the API must stream the file."""

    def local_path(self, key: str) -> Path | None:
        return None

    @abstractmethod
    async def read_bytes(self, key: str) -> bytes | None: ...

    async def write_stream(self, key: str, chunks: AsyncIterator[bytes], max_bytes: int) -> int:
        raise NotImplementedError


class LocalStorage(Storage):
    def __init__(self, root: Path):
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        path = (self.root / key).resolve()
        if self.root not in path.parents:
            raise ValueError("invalid storage key")
        return path

    def create_upload_url(self, key, content_type, upload_id):
        return f"/api/uploads/{upload_id}/content", {"Content-Type": content_type}

    async def size(self, key):
        p = self._path(key)
        return p.stat().st_size if p.exists() else None

    async def download_to(self, key, dest):
        dest.parent.mkdir(parents=True, exist_ok=True)
        await asyncio.to_thread(shutil.copyfile, self._path(key), dest)
        return dest

    async def upload_file(self, src, key, content_type):
        p = self._path(key)
        p.parent.mkdir(parents=True, exist_ok=True)
        await asyncio.to_thread(shutil.copyfile, src, p)

    def download_url(self, key, filename=None):
        return None

    def local_path(self, key):
        p = self._path(key)
        return p if p.exists() else None

    async def read_bytes(self, key):
        p = self._path(key)
        return await asyncio.to_thread(p.read_bytes) if p.exists() else None

    async def write_stream(self, key, chunks, max_bytes):
        p = self._path(key)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(p.suffix + ".part")
        written = 0
        try:
            with tmp.open("wb") as fh:
                async for chunk in chunks:
                    written += len(chunk)
                    if written > max_bytes:
                        raise ValueError("file exceeds the size limit")
                    fh.write(chunk)
            tmp.replace(p)
        finally:
            tmp.unlink(missing_ok=True)
        return written


class GCSStorage(Storage):
    def __init__(self, bucket: str):
        from google.cloud import storage as gcs

        self._client = gcs.Client()
        self._bucket = self._client.bucket(bucket)
        self._signing_creds = None

    def _sign_kwargs(self) -> dict:
        """On Cloud Run the runtime credentials have no private key, so sign via IAM."""
        import google.auth
        from google.auth.transport import requests as ga_requests
        from google.oauth2 import service_account

        if self._signing_creds is None:
            self._signing_creds, _ = google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
        creds = self._signing_creds
        if isinstance(creds, service_account.Credentials):
            return {"credentials": creds}  # key file: signs locally
        if not creds.valid:
            creds.refresh(ga_requests.Request())
        return {"service_account_email": creds.service_account_email, "access_token": creds.token}

    def create_upload_url(self, key, content_type, upload_id):
        blob = self._bucket.blob(key)
        url = blob.generate_signed_url(
            version="v4",
            expiration=dt.timedelta(hours=1),
            method="PUT",
            content_type=content_type,
            **self._sign_kwargs(),
        )
        return url, {"Content-Type": content_type}

    async def size(self, key):
        blob = await asyncio.to_thread(self._bucket.get_blob, key)
        return blob.size if blob else None

    async def download_to(self, key, dest):
        dest.parent.mkdir(parents=True, exist_ok=True)
        await asyncio.to_thread(self._bucket.blob(key).download_to_filename, str(dest))
        return dest

    async def upload_file(self, src, key, content_type):
        blob = self._bucket.blob(key)
        await asyncio.to_thread(blob.upload_from_filename, str(src), content_type=content_type)

    async def read_bytes(self, key):
        blob = await asyncio.to_thread(self._bucket.get_blob, key)
        return await asyncio.to_thread(blob.download_as_bytes) if blob else None

    def download_url(self, key, filename=None):
        disposition = f'attachment; filename="{filename}"' if filename else None
        return self._bucket.blob(key).generate_signed_url(
            version="v4",
            expiration=dt.timedelta(hours=6),
            method="GET",
            response_disposition=disposition,
            **self._sign_kwargs(),
        )


def build_storage(settings: Settings) -> Storage:
    if settings.storage_backend == "gcs":
        if not settings.gcs_bucket:
            raise RuntimeError("GCS_BUCKET must be set when STORAGE_BACKEND=gcs")
        return GCSStorage(settings.gcs_bucket)
    return LocalStorage(settings.local_data_dir / "blobs")
