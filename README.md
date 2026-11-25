# Vlogsmith: AI video blog studio (Google ADK)

Upload photos and video clips, describe the story, and press **Create**. A multi-agent workflow built on
Google's Agent Development Kit (ADK 2.x `Workflow` graph API) and Gemini produces:

- **`vlog.mp4`**: a YouTube-style edit. It opens cold on your best shot with an animated title, then moves through
  chapters with animated section titles, 📍 location tags, lower thirds, creator-style captions with a highlighted
  keyword, and sticker callouts. It adds Ken Burns motion aimed at each shot's subject, a colour grade, cuts on the
  music beat, bigger transitions and whooshes between chapters, an optional AI voiceover with word-by-word subtitles,
  and an end screen with an animated subscribe button.
- **`thumbnail.jpg`**: a 1280×720 YouTube thumbnail with huge outlined text placed away from the subject.
- **`youtube.txt`**: title, description, chapter timestamps and tags, ready to paste.
- **`article.html` / `article.md`**: a written blog post with your photos and short web clips embedded between sections.
- **`blog_package.zip`**: everything above in one download.

### Motion graphics

The overlay engine is in `backend/app/media/overlays.py`. Each graphic is a layer that draws itself at time *t*
with easing (pop, slide, wipe, word-by-word). A shot's layers are rendered into PNG keyframes, one per visual change,
and FFmpeg overlays them through the concat demuxer, so animations are frame-accurate and cheap. There are four
themes: **Creator** (bold, outlined, yellow keyword), **Pop** (condensed, colourful stickers), **Clean** (minimal,
keyword box) and **Cinematic** (serif, italic gold accent, optional letterbox). Subtitle word timings are estimated
from the TTS audio's speech segments (`media/timing.py`). Text in Devanagari and other non-Latin scripts falls back
to Noto fonts.

## The agent workflow

```
START → ingest → analyze_media ×N (parallel) → plan_story → write_sections ×M (parallel) → review_script
                                                                   ↑                           │
                          merge_revisions ← rewrite_sections ×K ←──┴──────────── "revise" ─────┤
                                                                                 "approved" ───┘
        → ( narrate ∥ compose_music ∥ write_article ) → production_join → render_video → publish
```

| Node | Kind | What it does |
|---|---|---|
| `ingest` | function | Downloads inputs, fixes EXIF rotation, probes videos, samples keyframes. Unreadable files are skipped with a warning. |
| `analyze_media` | parallel worker, Gemini vision `LlmAgent` | One agent per photo or video (videos are sent as timestamped keyframes). Returns a description, mood, quality score, caption idea and the best moment in the clip. |
| `plan_story` | `LlmAgent` (Pro in balanced/high quality) | Title, narrative arc, sections, media order, music mood. The output is repaired so every media item is used exactly once. |
| `write_sections` | parallel worker `LlmAgent` | Writes each section: per-shot captions and timings, voiceover script, article prose, alt text. |
| `review_script` | `LlmAgent` + routing | Editor-in-chief critique. Sends flagged sections back to `rewrite_sections` (in parallel) until approved or `review_rounds` is used up. |
| `narrate` / `compose_music` / `write_article` | fan-out + `JoinNode` | Gemini TTS voiceover, the soundtrack (built-in composer, Lyria on Vertex AI, or your uploaded track), and the article intro, conclusion and SEO metadata. |
| `render_video` | ffmpeg | Each shot is rendered to an identical-codec segment, then joined with xfade or concat, then mixed with the music (ducked under the voiceover) and loudness-limited. |
| `publish` | function | Builds the article HTML and Markdown, the zip package, and uploads all outputs. |

All structured outputs are enforced with Pydantic `output_schema` (`backend/app/agents/schemas.py`).
Prompts are in `backend/app/agents/prompts.py`.

## Settings users can configure

Tone, language, audience, title, target length (20–600 s), aspect ratio (16:9 / 9:16 / 1:1), 720p/1080p, 24/30 fps,
motion-graphics theme and caption position, colour grade, smart crop vs fit-with-blur, beat-synced cuts, letterbox,
animated subtitles, transition SFX, channel name for the end screen, transition type and duration, Ken Burns motion, title and outro cards, outro text, max clip
length, keep or mute clip audio, music source, mood and volume, voiceover and voice, article on/off and length,
editor review rounds, and quality (fast/balanced/high, which picks Flash or Pro and the x264 preset). The UI also has
one-click presets (YouTube, Reel/Short, Cinematic, Square).

## Run locally

Requirements: Python 3.12, Node ≥ 20.19, and a Gemini API key (https://aistudio.google.com/apikey) or Vertex AI.
ffmpeg is optional locally because a bundled static build (`imageio-ffmpeg`) is used if none is on PATH.

```bash
cp .env.example .env                   # set GOOGLE_API_KEY=... (project root)
cd frontend && npm ci && npm run build && cd ..
cd backend && uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python -e ".[dev]"
.venv/bin/uvicorn app.main:create_app --factory --port 8080
# open http://localhost:8080
```

For frontend development, run `npm run dev` in `frontend/`. It proxies `/api` to port 8080.

With Docker: `cp .env.example .env && docker compose up --build`.

### One-command CLI

```bash
cd backend
.venv/bin/python -m app.cli create ~/trip/*.jpg ~/trip/*.mp4 \
  --context "3 days in the Swiss Alps by train, dreamy and a bit funny" --aspect 9:16 --duration 45 --voiceover
# private Cloud Run: add --server https://... --token "$(gcloud auth print-identity-token)"
```

## Deploy to Cloud Run

```bash
PROJECT_ID=my-project REGION=us-central1 ./deploy/cloudrun.sh          # private service
PROJECT_ID=my-project PUBLIC=true ./deploy/cloudrun.sh                 # public URL
```

The script is idempotent. It enables the APIs, creates a runtime service account (Vertex AI user, Firestore user,
bucket object admin, and token creator on itself for keyless URL signing), a private bucket with upload CORS and
lifecycle rules, and a Firestore database. It then deploys with 4 vCPU / 8 GiB, always-on CPU and a 60-minute request timeout.

How it works in the cloud:

- **Uploads** go from the browser directly to GCS via V4 signed URLs, which avoids Cloud Run's 32 MB request limit.
- **Jobs and progress events** are stored in Firestore, so any instance can stream progress (SSE) or accept a cancel.
- **Models** run on Vertex AI through the service account, with no API keys in the service.
- **Downloads** are served as short-lived signed GCS URLs. The article HTML is proxied with a strict CSP.

## Tests

```bash
cd backend && .venv/bin/python -m pytest -q
```

The end-to-end test runs the real ADK workflow through the HTTP API with a deterministic fake Gemini
(`tests/fake_llm.py`) and real ffmpeg rendering. It covers parallel analysis, a corrupt upload, a review → revise → approve
loop, voiceover placement, crossfade and fade-through-black, 16:9 and 9:16, article escaping, the zip package and
path-traversal protection.

## Known limits and next steps

- **Authentication:** there are no user accounts. Keep the service private (IAM/IAP) or add an auth layer before opening it publicly.
- **Job durability:** jobs run inside the Cloud Run instance that accepted them. If that instance stops, the job is
  marked failed after 2 minutes without a heartbeat and the user can press Retry. For higher durability, move execution to
  Cloud Run Jobs or Cloud Tasks; the worker is already isolated in `app/worker.py`.
- **Retention:** uploads are deleted after 7 days and outputs after 90 days (`deploy/lifecycle.json`). Retrying a job
  whose uploads have expired fails at ingest.
- **Model names** are configurable via env (`MODEL_FAST`, `MODEL_SMART`, `MODEL_TTS`, `LYRIA_MODEL`). TTS and Lyria model IDs
  differ between AI Studio and Vertex AI and change over time, so check them for your project.
- **HEIC photos** (iPhone default) are not supported yet. Export them as JPEG.
