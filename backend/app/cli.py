"""Create a vlog from the terminal with one command.

    python -m app.cli create trip/*.jpg trip/*.mp4 \
        --context "A weekend in the Swiss Alps, dreamy and warm" --aspect 9:16 --voiceover

Talks to a running Vlogsmith server (local or Cloud Run). For a private Cloud Run
service pass --token "$(gcloud auth print-identity-token)".
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import sys
import time
from pathlib import Path

import httpx

from .models import BlogSettings


def _client(server: str, token: str | None) -> httpx.Client:
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    return httpx.Client(base_url=server.rstrip("/"), headers=headers, timeout=httpx.Timeout(60, read=600))


def _check(r: httpx.Response) -> dict:
    if r.status_code >= 400:
        try:
            detail = r.json().get("detail")
        except ValueError:
            detail = r.text
        sys.exit(f"error: {r.status_code} {detail}")
    return r.json() if r.content else {}


def _upload(client: httpx.Client, path: Path) -> str:
    ctype = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    ticket = _check(client.post("/api/uploads", json={"filename": path.name, "content_type": ctype,
                                                      "size": path.stat().st_size}))
    url = ticket["upload_url"]
    with path.open("rb") as fh:
        if url.startswith("/"):
            r = client.put(url, content=fh, headers=ticket["headers"])
        else:  # signed GCS URL: must not carry our bearer token
            r = httpx.put(url, content=fh, headers=ticket["headers"], timeout=None)
    if r.status_code >= 300:
        sys.exit(f"error: upload of {path.name} failed ({r.status_code})")
    _check(client.post(f"/api/uploads/{ticket['upload_id']}/complete"))
    return ticket["upload_id"]


def create(args: argparse.Namespace) -> None:
    files = [Path(f) for f in args.files]
    missing = [str(f) for f in files if not f.is_file()]
    if missing:
        sys.exit(f"error: not found: {', '.join(missing)}")
    overrides = {
        "context": args.context, "title": args.title, "tone": args.tone, "language": args.language,
        "aspect_ratio": args.aspect, "target_duration": args.duration, "resolution": args.resolution,
        "voiceover": args.voiceover, "music_source": "upload" if args.music else args.music_source,
        "transition": args.transition, "caption_style": args.captions, "quality": args.quality,
    }
    if args.settings:
        overrides = {**json.loads(Path(args.settings).read_text()), **{k: v for k, v in overrides.items() if v is not None}}
    settings = BlogSettings.model_validate({k: v for k, v in overrides.items() if v is not None})

    with _client(args.server, args.token) as client:
        ids = []
        for i, f in enumerate(files, 1):
            print(f"[upload {i}/{len(files)}] {f.name}")
            ids.append(_upload(client, f))
        if args.music:
            print(f"[upload] music {Path(args.music).name}")
            settings.music_upload_id = _upload(client, Path(args.music))
        job = _check(client.post("/api/jobs", json={"upload_ids": ids, "settings": settings.model_dump()}))
        job_id = job["id"]
        print(f"[job] {job_id} started — {args.server.rstrip('/')}/jobs/{job_id}")

        seen = 0
        while True:
            with client.stream("GET", f"/api/jobs/{job_id}/events", params={"after": seen}) as stream:
                event = None
                for line in stream.iter_lines():
                    if line.startswith("event: "):
                        event = line[7:]
                    elif line.startswith("data: ") and event == "progress":
                        ev = json.loads(line[6:])
                        seen = ev["seq"]
                        if ev["stage"] == "trace":
                            if args.trace and ev.get("data", {}).get("type") in ("graph", "handoff", "llm_start", "route"):
                                print(f"       ⇢ ADK  {ev['message']}")
                            continue
                        pct = f"{ev['progress'] * 100:3.0f}%" if ev.get("progress") is not None else "    "
                        print(f"  {pct} [{ev['stage']}] {ev['message']}")
            job = _check(client.get(f"/api/jobs/{job_id}"))
            if job["status"] in ("succeeded", "failed", "cancelled"):
                break
            time.sleep(1)

        if job["status"] != "succeeded":
            sys.exit(f"job {job['status']}: {job.get('error') or ''}")
        out = Path(args.out) / job_id
        out.mkdir(parents=True, exist_ok=True)
        for name in filter(None, [job["outputs"].get("video"), job["outputs"].get("package")]):
            dest = out / name
            with client.stream("GET", f"/api/jobs/{job_id}/files/{name}", follow_redirects=True) as r:
                r.raise_for_status()
                with dest.open("wb") as fh:
                    for chunk in r.iter_bytes():
                        fh.write(chunk)
            print(f"[saved] {dest}")
        print(f"Done: “{job.get('title')}” ({job.get('duration') or 0:.0f}s)")
        _print_cost(job.get("cost"))


def _print_cost(cost: dict | None) -> None:
    from .costs import format_usd

    if not cost:
        return
    print("\nGeneration cost")
    for line in cost["lines"]:
        tokens = f"{line['input_tokens']:,} in / {line['output_tokens']:,} out"
        print(f"  {line['label']:<26} {line['model']:<30} {line['calls']:>3} calls  {tokens:<26} {format_usd(line['cost_usd']):>9}")
    if cost.get("compute_usd"):
        print(f"  {'Cloud Run compute':<26} {'':<30} {cost['compute_seconds']:>5.0f}s {'':<29} {format_usd(cost['compute_usd']):>9}")
    print(f"  {'TOTAL':<26} {'':<69} {format_usd(cost['total_usd']):>9}")
    for note in cost.get("notes", []):
        print(f"  * {note}")


def main() -> None:
    sys.stdout.reconfigure(line_buffering=True)  # live progress even when piped
    p = argparse.ArgumentParser(prog="vlogsmith", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("create", help="upload media and create a vlog")
    c.add_argument("files", nargs="+", help="photos and videos, in the order you'd like them considered")
    c.add_argument("--context", required=True, help="what the blog is about and how it should feel")
    c.add_argument("--title")
    c.add_argument("--tone")
    c.add_argument("--language")
    c.add_argument("--aspect", choices=["16:9", "9:16", "1:1"])
    c.add_argument("--duration", type=int, help="target length in seconds")
    c.add_argument("--resolution", choices=["720p", "1080p"])
    c.add_argument("--transition")
    c.add_argument("--captions", choices=["bold", "minimal", "cinematic", "none"])
    c.add_argument("--quality", choices=["fast", "balanced", "high"])
    c.add_argument("--voiceover", action="store_true", default=None)
    c.add_argument("--music", help="path to your own music track")
    c.add_argument("--music-source", choices=["generated", "ai", "none"])
    c.add_argument("--settings", help="JSON file with any BlogSettings fields")
    c.add_argument("--server", default="http://localhost:8080")
    c.add_argument("--token", help="bearer token for a private Cloud Run service")
    c.add_argument("--out", default="./vlogs")
    c.add_argument("--trace", action="store_true", help="print ADK hand-offs, routes and model calls live")
    args = p.parse_args()
    if args.cmd == "create":
        create(args)


if __name__ == "__main__":
    main()
