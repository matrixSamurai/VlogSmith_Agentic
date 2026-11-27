"""Full pipeline through the HTTP API with a fake model and real ffmpeg rendering."""

from __future__ import annotations

import asyncio
import io
import json
import wave
import zipfile
from pathlib import Path

import httpx
import numpy as np
import pytest
from PIL import Image, ImageDraw

from app.config import Settings
from app.jobstore import SQLiteJobStore
from app.main import create_app
from app.media import audio
from app.media.ffmpeg import probe, run_ffmpeg
from app.storage import LocalStorage
from app.worker import JobManager

from .fake_llm import FakeGemini


def _jpeg(color, size) -> bytes:
    im = Image.new("RGB", size, color)
    d = ImageDraw.Draw(im)
    for x in range(0, size[0], 60):
        d.line((x, 0, x, size[1]), fill=(255, 255, 255), width=4)
    buf = io.BytesIO()
    im.save(buf, "JPEG")
    return buf.getvalue()


async def _video(path: Path) -> bytes:
    await run_ffmpeg(["-f", "lavfi", "-i", "testsrc2=size=640x360:rate=30", "-f", "lavfi", "-i",
                      "sine=frequency=330:sample_rate=44100", "-t", "6", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                      "-c:a", "aac", "-shortest", str(path)])
    return path.read_bytes()


@pytest.fixture
def env(tmp_path, monkeypatch):
    settings = Settings(local_data_dir=tmp_path / "data", work_dir=tmp_path / "work", frontend_dist=tmp_path / "nofront",
                        render_concurrency=3, job_heartbeat_seconds=1)
    monkeypatch.setattr("app.main.get_settings", lambda: settings)
    monkeypatch.setattr("app.agents.pipeline.get_settings", lambda: settings)

    async def fake_tts(text, voice, out, model, style_hint="", on_usage=None):
        if on_usage:
            from google.genai import types as gt
            on_usage(model, gt.GenerateContentResponseUsageMetadata(prompt_token_count=50, candidates_token_count=1000))
        sr = 24000
        t = np.arange(int(sr * 2.5)) / sr
        pcm = (np.sin(2 * np.pi * 200 * t) * 8000).astype("<i2")
        out.parent.mkdir(parents=True, exist_ok=True)
        with wave.open(str(out), "wb") as wf:
            wf.setnchannels(1), wf.setsampwidth(2), wf.setframerate(sr), wf.writeframes(pcm.tobytes())
        return out

    monkeypatch.setattr(audio, "synthesize_speech", fake_tts)
    monkeypatch.setenv("PRICING_JSON", '{"fake-gemini": {"input": 3.0, "output": 15.0, "cached": 0.15}}')
    store = SQLiteJobStore(settings.local_data_dir / "t.db")
    storage = LocalStorage(settings.local_data_dir / "blobs")
    fake = FakeGemini(calls={})
    manager = JobManager(settings, store, storage, model_overrides={"fast": fake, "smart": fake})
    app = create_app(manager)
    return app, fake, tmp_path


async def _upload(client, name, ctype, data) -> str:
    r = await client.post("/api/uploads", json={"filename": name, "content_type": ctype, "size": len(data)})
    assert r.status_code == 200, r.text
    ticket = r.json()
    r = await client.put(ticket["upload_url"], content=data, headers=ticket["headers"])
    assert r.status_code == 200, r.text
    r = await client.post(f"/api/uploads/{ticket['upload_id']}/complete")
    assert r.status_code == 200, r.text
    return ticket["upload_id"]


async def _wait(client, job_id, timeout=240):
    for _ in range(timeout * 2):
        job = (await client.get(f"/api/jobs/{job_id}")).json()
        if job["status"] in ("succeeded", "failed", "cancelled"):
            return job
        await asyncio.sleep(0.5)
    raise AssertionError("job timed out")


@pytest.mark.asyncio
@pytest.mark.parametrize("transition,voiceover,aspect", [("crossfade", True, "16:9"), ("fade_black", False, "9:16")])
async def test_full_pipeline(env, transition, voiceover, aspect):
    app, fake, tmp = env
    transport = httpx.ASGITransport(app=app)
    async with app.router.lifespan_context(app), httpx.AsyncClient(transport=transport, base_url="http://t") as client:
        ids = [
            await _upload(client, "lake.jpg", "image/jpeg", _jpeg((30, 110, 190), (1600, 1067))),
            await _upload(client, "portrait.jpg", "image/jpeg", _jpeg((190, 80, 50), (900, 1350))),
            await _upload(client, "train.mp4", "video/mp4", await _video(tmp / "train.mp4")),
            await _upload(client, "broken.jpg", "image/jpeg", b"definitely not a jpeg"),
        ]
        settings = {"context": "A weekend trip to the Swiss Alps", "target_duration": 25, "resolution": "720p",
                    "aspect_ratio": aspect, "transition": transition, "voiceover": voiceover, "quality": "fast",
                    "review_rounds": 2, "music_source": "generated"}
        r = await client.post("/api/jobs", json={"upload_ids": ids, "settings": settings})
        assert r.status_code == 201, r.text
        job = await _wait(client, r.json()["id"])
        events = (await client.get(f"/api/jobs/{job['id']}/events")).text

        assert job["status"] == "succeeded", (job, events)
        assert job["title"] == "Chasing Light in the Alps"
        assert "Skipping broken.jpg" in events
        assert "revising 1 section" in events and "Draft approved" in events
        assert fake.calls["MediaInsight"] == 3  # broken file skipped, others analysed in parallel
        assert fake.calls["SectionScript"] == 3  # 2 sections + 1 revision
        assert fake.calls["ScriptReview"] == 2

        video = await client.get(f"/api/jobs/{job['id']}/files/vlog.mp4")
        assert video.status_code == 200
        out = tmp / f"out_{transition}.mp4"
        out.write_bytes(video.content)
        info = await probe(out)
        assert info.has_audio and info.has_video
        assert (info.width, info.height) == ((1280, 720) if aspect == "16:9" else (720, 1280))
        assert abs(info.duration - job["duration"]) < 0.5

        page = (await client.get(f"/api/jobs/{job['id']}/files/article.html")).text
        assert "Chasing Light in the Alps" in page and "<script>" not in page
        assert "<strong>light</strong>" in page and 'src="media/m3.mp4"' in page
        pkg = await client.get(f"/api/jobs/{job['id']}/files/blog_package.zip")
        names = zipfile.ZipFile(io.BytesIO(pkg.content)).namelist()
        assert {"vlog.mp4", "article.html", "article.md", "media/m1.jpg", "media/m3.mp4"} <= set(names)

        yt = (await client.get(f"/api/jobs/{job['id']}/files/youtube.txt")).text
        assert "Chasing Light in the Alps" in yt and "0:00 Intro" in yt and "Come ride the rails." in yt
        assert "Motion graphics:" in events
        traces = [json.loads(l[6:])["data"] for l in events.splitlines()
                  if l.startswith("data:") and '"stage":"trace"' in l]
        graph = next(t for t in traces if t["type"] == "graph")
        assert {"review_script", "production_join", "analyze_media"} <= {n["id"] for n in graph["nodes"]}
        handoffs = [t for t in traces if t["type"] == "handoff"]
        assert any(h["from"] == "review_script" and h["route"] == "revise" and h["to"] == ["rewrite_sections"] for h in handoffs)
        assert any(h["from"] == "review_script" and h["route"] == "approved" and len(h["to"]) == 3 for h in handoffs)
        assert any(h["from"] == "plan_story" and h["shape"].startswith("list[2]") for h in handoffs)
        llm = [t for t in traces if t["type"] == "llm_start"]
        assert len(llm) == sum(fake.calls.values()) and any(t["schema"] == "SectionScript" for t in llm)
        Path(tmp / f"trace_{transition}.txt").write_text("\n".join(json.dumps(t)[:300] for t in traces))
        # Cost: every fake call = 1000 in + (200 out + 300 thinking) -> $0.003 + $0.0075 = $0.0105
        cost = job["cost"]
        fake_lines = [l for l in cost["lines"] if l["model"] == "fake-gemini"]
        calls = sum(l["calls"] for l in fake_lines)
        assert calls == sum(fake.calls.values())
        assert abs(sum(l["cost_usd"] for l in fake_lines) - calls * 0.0105) < 1e-6
        assert abs(cost["ai_usd"] - sum(l["cost_usd"] for l in cost["lines"])) < 1e-6
        if voiceover:
            tts = next(l for l in cost["lines"] if l["label"] == "Voiceover (TTS)")
            assert tts["calls"] == 2 and abs(tts["cost_usd"] - 2 * (50 * 0.5 + 1000 * 10) / 1e6) < 1e-9
        progress = [l for l in events.splitlines() if l.startswith("data:") and '"seq"' in l]
        assert "Generation cost: $" in progress[-1]  # announced as the very last activity event
        if voiceover:
            assert "subtitle words" in events
        assert (await client.get(f"/api/jobs/{job['id']}/files/../../t.db")).status_code == 404


@pytest.mark.asyncio
async def test_upload_validation(env):
    app, _, _ = env
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t") as client:
        r = await client.post("/api/uploads", json={"filename": "x.exe", "content_type": "application/x", "size": 10})
        assert r.status_code == 415
        r = await client.post("/api/uploads", json={"filename": "x.jpg", "content_type": "image/jpeg", "size": 10**10})
        assert r.status_code == 413
        r = await client.post("/api/jobs", json={"upload_ids": ["up_missing"], "settings": {"context": "hello"}})
        assert r.status_code == 400
        cfg = (await client.get("/api/config")).json()
        assert "crossfade" in cfg["options"]["transitions"]
