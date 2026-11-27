"""The vlog creation workflow, built on the ADK 2.x graph `Workflow` API.

Graph:

    START
      -> ingest                      (download, normalize, probe, sample video keyframes)
      -> analyze_media   [parallel]  (one Gemini vision agent per photo / video)
      -> plan_story                  (showrunner agent: title, arc, sections, music)
      -> write_sections  [parallel]  (one writer agent per section: captions, narration, article)
      -> review_script               (editor agent; routes "revise" or "approved")
           revise  -> rewrite_sections [parallel] -> merge_revisions -> review_script   (loop)
           approved -> (narrate | compose_music | write_article)   [fan-out]
      -> production_join             (JoinNode waits for all three)
      -> render_video                (ffmpeg: cards, Ken Burns, captions, clips, transitions, mix)
      -> publish                     (article HTML/MD with embedded media, package zip, uploads)
"""

from __future__ import annotations

import asyncio
import html
import json
import logging
import re
import math
import shutil
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from google.adk.agents import Context, LlmAgent
from google.adk.workflow import START, JoinNode, Workflow, node
from google.genai import types

from ..config import get_settings
from ..media import audio, graphics, timing
from ..media import overlays as O
from ..media.ffmpeg import FFmpegError, extract_frames, probe, run_ffmpeg
from ..media.renderer import Clip, RenderSpec, layout_timeline, plan_boundaries, render_timeline
from ..models import JobOutputs
from . import prompts
from .runtime import JobRuntime, LocalMedia
from .schemas import LOOKS, OVERLAYS, ArticleExtras, MediaInsight, ScriptReview, SectionScript, StoryPlan

log = logging.getLogger(__name__)

TITLE_SECONDS = 3.2
OUTRO_SECONDS = 3.0


# ----------------------------------------------------------------- helpers
def _gen_config(temperature: float) -> types.GenerateContentConfig:
    return types.GenerateContentConfig(
        temperature=temperature,
        http_options=types.HttpOptions(retry_options=types.HttpRetryOptions(attempts=4, initial_delay=2.0)),
    )


def _brief(rt: JobRuntime) -> str:
    s = rt.settings
    lines = [
        f"Creator brief: {s.context}",
        f"Working title: {s.title}" if s.title else "",
        f"Tone: {s.tone}. Language: {s.language}.",
        f"Audience: {s.audience}" if s.audience else "",
        f"Target video length: about {s.target_duration} seconds, aspect ratio {s.aspect_ratio}.",
        f"Article length: {s.article_length}. Voiceover: {'yes' if s.voiceover else 'no'}.",
        f"Max seconds per video clip: {s.max_clip_seconds}.",
    ]
    return "\n".join(line for line in lines if line)


def _user(text: str, extra: list[types.Part] | None = None) -> types.Content:
    return types.Content(role="user", parts=[types.Part(text=text), *(extra or [])])


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def _media_summary(rt: JobRuntime, insights: dict[str, dict], ids: list[str] | None = None) -> str:
    rows = []
    for mid in ids or list(insights):
        lm = rt.local.get(mid)
        ins = insights.get(mid)
        if not lm or not ins:
            continue
        extra = (
            f"video {lm.duration:.1f}s, highlight {ins['highlight_start']:.1f}-{ins['highlight_end']:.1f}s"
            if lm.ref.kind == "video"
            else "photo"
        )
        rows.append(
            f"- id={mid} ({extra}; quality {ins['quality']}/10; mood {ins['mood']}): {ins['description']} "
            f"Setting: {ins['setting']}." + (f" Location: {ins['location']}." if ins.get("location") else "")
            + f" Subjects: {', '.join(ins['subjects'])}. Caption idea: {ins['caption_idea']}"
        )
    return "\n".join(rows)


def _md_inline(text: str) -> str:
    """Escape, then apply a tiny safe subset of markdown (bold, italic, links)."""
    t = html.escape(text)
    t = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", t)
    t = re.sub(r"(?<!\*)\*(?!\s)(.+?)(?<!\s)\*(?!\*)", r"<em>\1</em>", t)
    t = re.sub(r"\[([^\]]+)\]\((https?://[^)\s]+)\)", r'<a href="\2" rel="nofollow noopener">\1</a>', t)
    return t


def _md_blocks(text: str) -> str:
    out = []
    for block in re.split(r"\n\s*\n", text.strip()):
        lines = [l.strip() for l in block.strip().splitlines() if l.strip()]
        if not lines:
            continue
        if all(re.match(r"^[-*] ", l) for l in lines):
            out.append("<ul>" + "".join(f"<li>{_md_inline(l[2:])}</li>" for l in lines) + "</ul>")
        elif lines[0].startswith(">"):
            out.append(f"<blockquote>{_md_inline(' '.join(l.lstrip('> ') for l in lines))}</blockquote>")
        else:
            out.append(f"<p>{_md_inline(' '.join(lines))}</p>")
    return "\n".join(out)


# ------------------------------------------------------------ workflow
def build_workflow(rt: JobRuntime) -> Workflow:
    cfg = get_settings()
    s = rt.settings
    fast, smart = rt.models["fast"], rt.models["smart"]
    planner_model = fast if s.quality == "fast" else smart
    writer_model = smart if s.quality == "high" else fast

    analyst = LlmAgent(
        name="media_analyst", model=fast, instruction=prompts.ANALYST, output_schema=MediaInsight,
        include_contents="none", generate_content_config=_gen_config(0.4),
    )
    planner = LlmAgent(
        name="story_planner", model=planner_model, instruction=prompts.PLANNER, output_schema=StoryPlan,
        include_contents="none", generate_content_config=_gen_config(0.8),
    )
    writer = LlmAgent(
        name="section_writer", model=writer_model, instruction=prompts.WRITER, output_schema=SectionScript,
        include_contents="none", generate_content_config=_gen_config(0.9),
    )
    reviewer = LlmAgent(
        name="editor_in_chief", model=planner_model, instruction=prompts.REVIEWER, output_schema=ScriptReview,
        include_contents="none", generate_content_config=_gen_config(0.2),
    )
    article_editor = LlmAgent(
        name="article_editor", model=fast, instruction=prompts.ARTICLE, output_schema=ArticleExtras,
        include_contents="none", generate_content_config=_gen_config(0.7),
    )
    for agent in (analyst, planner, writer, reviewer, article_editor):
        agent.after_model_callback = _usage_tracker(rt, agent)

    def card(agent: LlmAgent) -> dict:
        model = agent.model if isinstance(agent.model, str) else getattr(agent.model, "model", "?")
        return {"agent": agent.name, "model": model, "schema": agent.output_schema.__name__ if agent.output_schema else None}

    # Which ADK LlmAgents each graph node runs (via ctx.run_node), shown in the live trace.
    rt.extras["node_agents"] = {
        "analyze_media": [card(analyst)], "plan_story": [card(planner)], "write_sections": [card(writer)],
        "rewrite_sections": [card(writer)], "review_script": [card(reviewer)], "write_article": [card(article_editor)],
        "narrate": [{"agent": "gemini_tts", "model": cfg.model_tts, "schema": "audio"}] if s.voiceover else [],
    }
    work = rt.work_dir

    # ---------------------------------------------------------------- ingest
    async def ingest(ctx: Context, node_input: Any):
        await rt.info("ingest", f"Preparing {len(rt.media)} media files", 0.02)
        sem = asyncio.Semaphore(4)

        async def prep(ref):
            async with sem:
                raw = await rt.storage.download_to(ref.storage_key, work / "raw" / f"{ref.id}_{Path(ref.filename).name}")
                if ref.kind == "image":
                    dst = work / "media" / f"{ref.id}.jpg"
                    w, h = await asyncio.to_thread(graphics.normalize_image, raw, dst)
                    return LocalMedia(ref=ref, path=dst, width=w, height=h)
                info = await probe(raw)
                if not info.has_video or info.duration <= 0.3:
                    raise FFmpegError(f"{ref.filename} has no usable video stream")
                n = int(_clamp(info.duration / 4, 3, 8))
                frames = await extract_frames(raw, work / "frames", n, info.duration)
                return LocalMedia(ref=ref, path=raw, width=info.width, height=info.height,
                                  duration=info.duration, has_audio=info.has_audio, frames=frames)

        results = await asyncio.gather(*(prep(r) for r in rt.media), return_exceptions=True)
        items = []
        for ref, res in zip(rt.media, results):
            if isinstance(res, Exception):
                await rt.warn("ingest", f"Skipping {ref.filename}: {res}")
                continue
            rt.local[ref.id] = res
            items.append({"media_id": ref.id})
        if not items:
            raise RuntimeError("None of the uploaded files could be read.")
        if s.music_source == "upload" and rt.music_upload:
            rt.extras["music_upload_path"] = await rt.storage.download_to(
                rt.music_upload.storage_key, work / "raw" / f"music_{Path(rt.music_upload.filename).name}"
            )
        photos = sum(1 for m in rt.local.values() if m.ref.kind == "image")
        videos = len(items) - photos
        await rt.info("ingest", f"Ready: {photos} photo{'s' * (photos != 1)}, {videos} video{'s' * (videos != 1)}", 0.08)
        return items

    # ---------------------------------------------------------------- analyze
    @node(name="analyze_media", parallel_worker=True, max_parallel_workers=cfg.llm_concurrency, rerun_on_resume=True)
    async def analyze_media(ctx: Context, node_input: dict):
        mid = node_input["media_id"]
        lm = rt.local[mid]
        parts: list[types.Part] = []
        if lm.ref.kind == "image":
            preview = work / "preview" / f"{mid}.jpg"
            preview.parent.mkdir(parents=True, exist_ok=True)
            await asyncio.to_thread(graphics.thumbnail, lm.path, preview, 1280)
            parts.append(types.Part.from_bytes(data=preview.read_bytes(), mime_type="image/jpeg"))
            desc = f"Media id {mid}: a PHOTO ({lm.width}x{lm.height}), file name '{lm.ref.filename}'."
        else:
            for t, frame in lm.frames:
                parts.append(types.Part(text=f"Frame at t={t:.1f}s:"))
                parts.append(types.Part.from_bytes(data=frame.read_bytes(), mime_type="image/jpeg"))
            desc = (f"Media id {mid}: a VIDEO of {lm.duration:.1f}s ({'with' if lm.has_audio else 'without'} sound), "
                    f"file name '{lm.ref.filename}'. {len(lm.frames)} keyframes follow.")
        try:
            insight = await ctx.run_node(analyst, node_input=_user(f"{_brief(rt)}\n\n{desc}", parts))
        except Exception as exc:  # one bad item must not sink the job
            log.warning("analysis failed for %s: %s", mid, exc)
            await rt.warn("analyze", f"Could not analyse {lm.ref.filename}; using defaults")
            insight = MediaInsight(
                description=lm.ref.filename, subjects=[], setting="", location="", mood="neutral", quality=5,
                caption_idea="", focus_x=0.5, focus_y=0.45, highlight_start=0,
                highlight_end=min(lm.duration, s.max_clip_seconds), tags=[],
            ).model_dump()
        insight["focus_x"] = _clamp(float(insight.get("focus_x", 0.5)), 0.0, 1.0)
        insight["focus_y"] = _clamp(float(insight.get("focus_y", 0.45)), 0.0, 1.0)
        done = rt.extras["analyzed"] = rt.extras.get("analyzed", 0) + 1
        await rt.info("analyze", f"Analysed {lm.ref.filename} ({done}/{len(rt.local)})", 0.1 + 0.15 * done / len(rt.local))
        return {"media_id": mid, **insight}

    # ---------------------------------------------------------------- plan
    @node(name="plan_story", rerun_on_resume=True)
    async def plan_story(ctx: Context, node_input: list[dict]):
        insights = {item["media_id"]: {k: v for k, v in item.items() if k != "media_id"} for item in node_input}
        ctx.state["insights"] = insights
        await rt.info("plan", "Story planner is designing the narrative arc", 0.27)
        prompt = (
            f"{_brief(rt)}\n\nMedia library ({len(insights)} items):\n{_media_summary(rt, insights)}\n\n"
            "Design the story plan."
        )
        plan = StoryPlan.model_validate(await ctx.run_node(planner, node_input=_user(prompt)))
        plan = _repair_plan(plan, insights)
        if s.title:
            plan.title = s.title
        ctx.state["plan"] = plan.model_dump()
        ctx.state["review_round"] = 0
        await rt.report("plan", f"Plan ready: “{plan.title}” with {len(plan.sections)} sections", 0.32, "success")
        return [{"index": i, "feedback": ""} for i in range(len(plan.sections))]

    # ---------------------------------------------------------------- write
    async def _write(ctx: Context, task: dict) -> dict:
        plan = StoryPlan.model_validate(ctx.state["plan"])
        insights = ctx.state["insights"]
        i = task["index"]
        sec = plan.sections[i]
        outline = "\n".join(f"{n + 1}. {p.heading} — {p.purpose}" for n, p in enumerate(plan.sections))
        prompt = (
            f"{_brief(rt)}\n\nTitle: {plan.title}\nHook: {plan.hook}\nStory outline:\n{outline}\n\n"
            f"Write section {i + 1} of {len(plan.sections)}: '{sec.heading}' (kicker '{sec.kicker}').\n"
            f"Purpose: {sec.purpose}\nTarget screen time: {sec.seconds:.0f}s\n"
            f"Media in order:\n{_media_summary(rt, insights, sec.media_ids)}\n"
        )
        if task.get("feedback"):
            prompt += f"\nEditor feedback you must address:\n{task['feedback']}\n"
            if prev := task.get("previous"):
                prompt += f"\nYour previous draft:\n{json.dumps(prev, ensure_ascii=False)}\n"
        script = SectionScript.model_validate(await ctx.run_node(writer, node_input=_user(prompt)))
        script = _repair_script(script, sec.media_ids, insights, rt)
        script.heading = script.heading or sec.heading
        return {"index": i, **script.model_dump()}

    @node(name="write_sections", parallel_worker=True, max_parallel_workers=cfg.llm_concurrency, rerun_on_resume=True)
    async def write_sections(ctx: Context, node_input: dict):
        out = await _write(ctx, node_input)
        n = len(ctx.state["plan"]["sections"])
        done = rt.extras["written"] = rt.extras.get("written", 0) + 1
        await rt.info("write", f"Wrote section “{out['heading']}” ({done}/{n})", 0.32 + 0.13 * done / n)
        return out

    @node(name="rewrite_sections", parallel_worker=True, max_parallel_workers=cfg.llm_concurrency, rerun_on_resume=True)
    async def rewrite_sections(ctx: Context, node_input: dict):
        out = await _write(ctx, node_input)
        await rt.info("review", f"Revised section “{out['heading']}”")
        return out

    def merge_revisions(ctx: Context, node_input: list[dict]):
        sections = list(ctx.state["sections"])
        for rev in node_input:
            sections[rev["index"]] = rev
        ctx.state["sections"] = sections
        return {"revised": [r["index"] for r in node_input]}

    # ---------------------------------------------------------------- review loop
    @node(name="review_script", rerun_on_resume=True)
    async def review_script(ctx: Context, node_input: Any):
        if isinstance(node_input, list):  # first pass: output of write_sections
            ctx.state["sections"] = sorted(node_input, key=lambda x: x["index"])
        rnd = ctx.state.get("review_round", 0)
        if rnd >= s.review_rounds:
            if rnd:
                await rt.report("review", f"Draft finalised after {rnd} review round(s)", 0.55, "success")
            ctx.route = "approved"
            return {"approved": True}
        ctx.state["review_round"] = rnd + 1
        await rt.info("review", f"Editor-in-chief reviewing draft (round {rnd + 1}/{s.review_rounds})", 0.47)
        plan = ctx.state["plan"]
        draft = {
            "title": plan["title"], "hook": plan["hook"],
            "sections": [
                {"index": x["index"], "heading": x["heading"], "narration": x["narration"],
                 "captions": [sh["caption"] for sh in x["shots"]],
                 "seconds": round(sum(sh["seconds"] for sh in x["shots"]), 1), "article_body": x["article_body"]}
                for x in ctx.state["sections"]
            ],
        }
        prompt = (f"{_brief(rt)}\n\nMedia notes:\n{_media_summary(rt, ctx.state['insights'])}\n\n"
                  f"Draft:\n{json.dumps(draft, ensure_ascii=False, indent=1)}")
        try:
            review = ScriptReview.model_validate(await ctx.run_node(reviewer, node_input=_user(prompt)))
        except Exception as exc:
            log.warning("review failed: %s", exc)
            await rt.warn("review", "Review step failed; continuing with the current draft")
            ctx.route = "approved"
            return {"approved": True}
        history = list(ctx.state.get("reviews", []))
        history.append(review.model_dump())
        ctx.state["reviews"] = history
        n = len(ctx.state["sections"])
        issues = [x for x in review.issues if 0 <= x.section_index < n]
        if review.approved or not issues:
            await rt.report("review", f"Draft approved (score {review.score}/10)", 0.55, "success")
            ctx.route = "approved"
            return {"approved": True}
        await rt.info("review", f"Score {review.score}/10 — revising {len({x.section_index for x in issues})} section(s): {review.summary}")
        feedback: dict[int, list[str]] = {}
        for iss in issues:
            feedback.setdefault(iss.section_index, []).append(f"- {iss.problem} → {iss.fix}")
        ctx.route = "revise"
        return [
            {"index": i, "feedback": "\n".join(fb), "previous": ctx.state["sections"][i]}
            for i, fb in sorted(feedback.items())
        ]

    # ---------------------------------------------------------------- production (parallel)
    @node(name="narrate", rerun_on_resume=True)
    async def narrate(ctx: Context, node_input: Any):
        if not s.voiceover:
            return {"voice": []}
        sections = ctx.state["sections"]
        await rt.info("production", f"Recording voiceover with voice {s.voice}", 0.57)
        style = f"Read this as a {s.tone} vlog narrator, natural pacing, in {s.language}:"
        sem = asyncio.Semaphore(4)

        async def one(sec):
            if not sec["narration"].strip():
                return None
            async with sem:
                p = await audio.synthesize_speech(sec["narration"], s.voice, work / "voice" / f"s{sec['index']}.wav",
                                                  cfg.model_tts, style,
                                                  on_usage=lambda m, u: rt.costs.record("tts", m, u))
            return str(p)

        results = await asyncio.gather(*(one(x) for x in sections), return_exceptions=True)
        paths: list[str | None] = []
        for r in results:
            if isinstance(r, Exception):
                log.warning("TTS failed: %s", r)
                paths.append(None)
            else:
                paths.append(r)
        if not any(paths):
            await rt.warn("production", "Voiceover generation failed; continuing without narration")
        rt.extras["voice"] = paths
        return {"voice": [p is not None for p in paths]}

    @node(name="compose_music", rerun_on_resume=True)
    async def compose_music(ctx: Context, node_input: Any):
        plan = ctx.state["plan"]
        mood = s.music_mood if s.music_mood != "auto" else plan.get("music_mood", "calm")
        mood = mood if mood in audio.MOODS else "calm"
        length = _clamp(s.target_duration * 1.6 + 20, 40, 4200)
        track: Path | None = None
        source = s.music_source
        if source == "upload":
            track = rt.extras.get("music_upload_path")
            if not track:
                await rt.warn("production", "No music file uploaded; generating a soundtrack instead")
                source = "generated"
        if source == "ai":
            if cfg.vertex_enabled:
                await rt.info("production", f"Composing {mood} music with Lyria", 0.58)
                try:
                    track = await audio.generate_lyria(
                        f"{plan.get('music_prompt', '')}. Mood: {mood}. Instrumental background music for a vlog.",
                        work / "music_ai.wav", cfg.google_cloud_project, cfg.google_cloud_location, cfg.lyria_model)
                except Exception as exc:
                    log.warning("Lyria failed: %s", exc)
                    await rt.warn("production", "AI music generation failed; using the built-in composer")
            else:
                await rt.warn("production", "AI music needs Vertex AI; using the built-in composer")
            if track is None:
                source = "generated"
        if source == "generated":
            await rt.info("production", f"Composing the soundtrack (mood: {mood})", 0.58)
            track = await asyncio.to_thread(audio.generate_soundtrack, mood, length, work / "music.wav",
                                            abs(hash(rt.job_id)) % 10_000)
            rt.extras["bpm"] = audio.mood_bpm(mood)  # known tempo -> cuts can land on the beat
        rt.extras["music"] = track
        return {"music": source if track else "none", "mood": mood}

    @node(name="write_article", rerun_on_resume=True)
    async def write_article(ctx: Context, node_input: Any):
        what = "article intro, conclusion, SEO and YouTube description" if s.generate_article else "YouTube description"
        await rt.info("production", f"Channel editor is writing the {what}", 0.59)
        plan = ctx.state["plan"]
        body = "\n\n".join(f"## {x['heading']}\n{x['article_body']}" for x in ctx.state["sections"])
        try:
            extras = await ctx.run_node(article_editor, node_input=_user(
                f"{_brief(rt)}\n\nTitle: {plan['title']}\nSubtitle: {plan['subtitle']}\nHook: {plan['hook']}\n\n{body}"))
            ctx.state["article"] = ArticleExtras.model_validate(extras).model_dump()
        except Exception as exc:
            log.warning("article editor failed: %s", exc)
            ctx.state["article"] = ArticleExtras(
                intro=plan["hook"], conclusion="", seo_title=plan["title"], seo_description=plan["subtitle"][:155],
                youtube_description=f"{plan['hook']}\n\n{plan['subtitle']}", tags=[]).model_dump()
        return {"article": True}

    production_join = JoinNode(name="production_join")

    # ---------------------------------------------------------------- render
    @node(name="render_video", rerun_on_resume=True)
    async def render_video(ctx: Context, node_input: Any):
        plan = StoryPlan.model_validate(ctx.state["plan"])
        sections = ctx.state["sections"]
        insights = ctx.state["insights"]
        W, H = s.frame_size
        look = s.color_look if s.color_look != "auto" else (plan.look if plan.look in LOOKS else "natural")
        spec = RenderSpec(W, H, s.fps, s.transition, s.transition_duration, s.ken_burns, s.quality,
                          s.clip_audio, s.clip_audio_volume, look=look, letterbox=s.letterbox, framing=s.framing)
        edit = await asyncio.to_thread(_build_edit, rt, plan, sections, spec, insights)
        timeline = edit.timeline
        await rt.info("render", f"Motion graphics: {edit.graphics} animated overlays"
                      + (f", {edit.subtitle_words} subtitle words" if edit.subtitle_words else "")
                      + f" · look: {look}" + (f" · cuts on the beat ({rt.extras['bpm']:.0f} BPM)" if edit.beat_synced else ""),
                      0.61)
        await rt.info("render", f"Rendering {len(timeline.clips)} shots ({timeline.total:.0f}s at {W}x{H})", 0.62)

        async def progress(done: int, total: int):
            await rt.info("render", f"Rendered shot {done}/{total}", 0.62 + 0.25 * done / total)

        out = work / "out" / "vlog.mp4"
        out.parent.mkdir(parents=True, exist_ok=True)
        await render_timeline(timeline, spec, work / "render", out, rt.extras.get("music"), s.music_volume,
                              edit.voice_parts, concurrency=cfg.render_concurrency, on_progress=progress,
                              sfx_parts=edit.sfx_parts, subtitles=edit.subtitles)
        thumb = work / "out" / "thumbnail.jpg"
        await asyncio.to_thread(graphics.youtube_thumbnail, thumb, edit.hero_still, plan.thumbnail_text or plan.title,
                                plan.title_emphasis, s.caption_style, edit.hero_focus)
        rt.extras["video"], rt.extras["thumbnail"] = out, thumb
        ctx.state["duration"] = timeline.total
        ctx.state["chapters"] = edit.chapters
        await rt.report("render", f"Video rendered ({timeline.total:.0f}s)", 0.9, "success")
        return {"video": str(out), "duration": timeline.total}

    # ---------------------------------------------------------------- publish
    @node(name="publish", rerun_on_resume=True)
    async def publish(ctx: Context, node_input: Any):
        await rt.info("publish", "Packaging the blog post and uploading assets", 0.92)
        plan = StoryPlan.model_validate(ctx.state["plan"])
        sections = ctx.state["sections"]
        out_dir = work / "out"
        media_dir = out_dir / "media"
        media_dir.mkdir(parents=True, exist_ok=True)
        outputs = JobOutputs(video="vlog.mp4", thumbnail="thumbnail.jpg", youtube="youtube.txt")
        meta = ArticleExtras.model_validate(ctx.state["article"])
        (out_dir / "youtube.txt").write_text(_youtube_text(plan, meta, ctx.state.get("chapters", [])), encoding="utf-8")
        files: list[tuple[Path, str, str]] = [
            (rt.extras["video"], "vlog.mp4", "video/mp4"), (rt.extras["thumbnail"], "thumbnail.jpg", "image/jpeg"),
            (out_dir / "youtube.txt", "youtube.txt", "text/plain; charset=utf-8")]

        if s.generate_article:
            article = ArticleExtras.model_validate(ctx.state["article"])
            md, page = await _build_article(rt, plan, sections, article, media_dir)
            (out_dir / "article.md").write_text(md, encoding="utf-8")
            (out_dir / "article.html").write_text(page, encoding="utf-8")
            outputs.article_md, outputs.article_html = "article.md", "article.html"
            files += [(out_dir / "article.md", "article.md", "text/markdown; charset=utf-8"),
                      (out_dir / "article.html", "article.html", "text/html; charset=utf-8")]
            for f in sorted(media_dir.iterdir()):
                files.append((f, f"media/{f.name}", "video/mp4" if f.suffix == ".mp4" else "image/jpeg"))

        package = out_dir / "blog_package.zip"

        def make_zip():
            with zipfile.ZipFile(package, "w", zipfile.ZIP_STORED) as zf:
                for path, name, _ in files:
                    zf.write(path, name)

        await asyncio.to_thread(make_zip)
        outputs.package = "blog_package.zip"
        files.append((package, "blog_package.zip", "application/zip"))
        for path, name, ctype in files:
            await rt.storage.upload_file(path, f"jobs/{rt.job_id}/outputs/{name}", ctype)
        ctx.state["outputs"] = outputs.model_dump()
        return {"outputs": outputs.model_dump(), "title": plan.title, "summary": plan.subtitle,
                "duration": ctx.state.get("duration")}

    return Workflow(
        name="vlog_pipeline",
        edges=[
            (START, ingest, analyze_media, plan_story, write_sections, review_script),
            (review_script, {"revise": rewrite_sections, "approved": (narrate, compose_music, write_article)}),
            (rewrite_sections, merge_revisions, review_script),
            ((narrate, compose_music, write_article), production_join, render_video, publish),
        ],
    )


def _usage_tracker(rt: JobRuntime, agent: LlmAgent):
    """after_model_callback that adds each Gemini call's token usage to the job's cost tracker."""
    model_name = agent.model if isinstance(agent.model, str) else getattr(agent.model, "model", "unknown")

    def track(callback_context, llm_response):
        if not getattr(llm_response, "partial", False):
            rt.costs.record(agent.name, model_name, getattr(llm_response, "usage_metadata", None))
        return None

    return track


# ------------------------------------------------------------ repairs (LLM output hardening)
def _repair_plan(plan: StoryPlan, insights: dict[str, dict]) -> StoryPlan:
    """Guarantee every media item appears exactly once and no section is empty."""
    valid = list(insights)
    seen: set[str] = set()
    for sec in plan.sections:
        sec.media_ids = [m for m in sec.media_ids if m in insights and not (m in seen or seen.add(m))]
    plan.sections = [sec for sec in plan.sections if sec.media_ids]
    missing = [m for m in valid if m not in seen]
    if not plan.sections:
        from .schemas import PlannedSection

        plan.sections = [PlannedSection(heading=plan.title or "The story", heading_emphasis="", kicker="", purpose="",
                                       media_ids=[], seconds=30)]
    for m in missing:  # append leftovers to the smallest section
        min(plan.sections, key=lambda x: len(x.media_ids)).media_ids.append(m)
    return plan


def _repair_script(script: SectionScript, media_ids: list[str], insights: dict, rt: JobRuntime) -> SectionScript:
    from .schemas import Shot

    by_id = {sh.media_id: sh for sh in script.shots if sh.media_id in media_ids}
    shots, alts = [], []
    for n, mid in enumerate(media_ids):
        ins = insights.get(mid, {})
        sh = by_id.get(mid) or Shot(media_id=mid, overlay="caption" if n else "none", caption=ins.get("caption_idea", ""),
                                    emphasis="", subline="", seconds=4, clip_start=ins.get("highlight_start", 0))
        sh.overlay = sh.overlay.strip().lower().replace("-", "_").replace(" ", "_")
        if sh.overlay not in OVERLAYS or (sh.overlay != "none" and not sh.caption.strip()):
            sh.overlay = "none" if not sh.caption.strip() else "caption"
        shots.append(sh)
        alts.append(script.image_alt[n] if n < len(script.image_alt) else ins.get("description", ""))
    script.shots, script.image_alt = shots, alts
    return script


# ------------------------------------------------------------ the edit (timeline + motion graphics)
@dataclass
class Edit:
    timeline: Any
    voice_parts: list[tuple[float, Path]]
    sfx_parts: list[tuple[float, Path, float]]
    subtitles: Path | None
    chapters: list[tuple[float, str]]
    hero_still: Path | None
    hero_focus: tuple[float, float]
    graphics: int
    subtitle_words: int
    beat_synced: bool


def _build_edit(rt: JobRuntime, plan: StoryPlan, sections: list[dict], spec: RenderSpec, insights: dict) -> Edit:
    """Decide every shot, its length, graphics, transitions, SFX and subtitles. Runs in a worker thread."""
    s = rt.settings
    W, H = spec.width, spec.height
    size = (W, H)
    gfx = rt.work_dir / "gfx"
    gfx.mkdir(parents=True, exist_ok=True)
    texts = s.caption_style != "none"
    theme = O.get_theme(s.caption_style)
    voice = rt.extras.get("voice", [])
    use_subs = texts and s.subtitles and s.voiceover and any(voice)

    def focus_of(mid: str) -> tuple[float, float]:
        ins = insights.get(mid, {})
        return float(ins.get("focus_x", 0.5)), float(ins.get("focus_y", 0.45))

    # Hero = the strongest shot; it opens the video (cold open) and fills the thumbnail.
    ranked = sorted(rt.local.values(), key=lambda m: (insights.get(m.ref.id, {}).get("quality", 0), m.ref.kind == "image"),
                    reverse=True)
    hero = ranked[0] if ranked else None
    hero_still: Path | None = None
    if hero is not None:
        if hero.ref.kind == "image":
            hero_still = hero.path
        elif hero.frames:
            hero_still = min(hero.frames, key=lambda f: abs(f[0] - insights.get(hero.ref.id, {}).get("highlight_start", 0)))[1]
    hero_focus = focus_of(hero.ref.id) if hero else (0.5, 0.45)

    clips: list[Clip] = []
    layer_fns: list[Any] = []  # per clip: fn(duration) -> list[Layer]
    section_first: dict[int, int] = {}

    if s.include_title_card and hero is not None:
        dur = TITLE_SECONDS + 0.4
        if hero.ref.kind == "image":
            clip = Clip("image", hero.path, dur, motion=0, focus=hero_focus)
        else:
            start = float(insights.get(hero.ref.id, {}).get("highlight_start", 0))
            start = _clamp(start, 0, max(0, hero.duration - 2))
            clip = Clip("video", hero.path, min(dur, max(hero.duration - start, 1.5)), clip_start=start,
                        source_duration=hero.duration, has_audio=hero.has_audio, focus=hero_focus)
        clips.append(clip)
        layer_fns.append(lambda d: [O.HookTitle(0.05, d - 0.05, theme, size, title=plan.title, subtitle=plan.subtitle,
                                                emphasis=plan.title_emphasis)] if texts else [])

    raw_total = sum(sh["seconds"] for sec in sections for sh in sec["shots"])
    budget = max(10.0, s.target_duration - (clips[0].duration if clips else 0) - (OUTRO_SECONDS if s.include_outro else 0))
    factor = _clamp(budget / raw_total, 0.6, 1.8) if raw_total > 0 else 1.0
    min_shot = 1.8
    motion = 1
    for si, sec in enumerate(sections):
        psec = plan.sections[si] if si < len(plan.sections) else None
        first = True
        section_clips: list[tuple[Clip, Any]] = []
        for sh in sec["shots"]:
            lm = rt.local.get(sh["media_id"])
            if not lm:
                continue
            focus = focus_of(lm.ref.id)
            if lm.ref.kind == "image":
                clip = Clip("image", lm.path, _clamp(sh["seconds"] * factor, 2.4, 7.0), motion=motion, section_index=si,
                            focus=focus)
                motion += 1
            else:
                start = _clamp(float(sh.get("clip_start") or 0), 0, max(0, lm.duration - min_shot))
                avail = lm.duration - start
                dur = _clamp(sh["seconds"] * factor, min(min_shot, avail), min(s.max_clip_seconds, avail))
                clip = Clip("video", lm.path, max(dur, 0.8), clip_start=start, source_duration=lm.duration,
                            has_audio=lm.has_audio, section_index=si, focus=focus)
            section_clips.append((clip, _shot_layers(sh, first, psec, sec, theme, size, texts, use_subs, s)))
            first = False
        if not section_clips:
            continue
        # Make the section long enough for its narration.
        vpath = voice[si] if si < len(voice) else None
        if vpath:
            # narration starts 0.35 s in; leave room for the outgoing transition (up to 0.7 s) + a breath
            needed = audio.wav_duration(Path(vpath)) + 0.35 + 0.7 + 0.4
            have = sum(c.length for c, _ in section_clips) - 0.2 * (len(section_clips) - 1)
            if needed > have:
                extra = needed - have
                images = [c for c, _ in section_clips if c.kind == "image"]
                if images:
                    for c in images:
                        c.duration += extra / len(images)
                else:
                    section_clips[-1][0].hold = extra
        section_first[si] = len(clips)
        for c, fn in section_clips:
            clips.append(c)
            layer_fns.append(fn)

    if s.include_outro:
        outro_text = s.outro_text or plan.outro_text or "Thanks for watching"
        sub = f"Subscribe to {s.channel_name}" if s.channel_name else "Like · Comment · Subscribe"
        clips.append(Clip("card", hero_still or (clips[0].src if clips else None), OUTRO_SECONDS + 0.6, motion=0,
                          focus=hero_focus))
        layer_fns.append(lambda d: [O.EndScreen(0.0, d, theme if texts else O.get_theme("bold"), size, text=outro_text,
                                                sub=sub)])
    if clips and clips[-1].kind == "card" and clips[-1].src is None:
        clips.pop()
        layer_fns.pop()

    boundaries = plan_boundaries(clips, spec)
    bpm = rt.extras.get("bpm")
    beat_synced = bool(s.beat_sync and bpm)
    if beat_synced:
        beat = 60.0 / float(bpm)
        for i, clip in enumerate(clips[:-1]):
            b = boundaries[i].duration
            eff = clip.length - b
            beats = max(2, math.ceil(eff / beat - 0.3))
            target = beats * beat + b - clip.hold
            if clip.kind == "video" and clip.source_duration:
                avail = clip.source_duration - clip.clip_start
                if target > avail:
                    target = max(beat, math.floor((avail - b) / beat) * beat) + b
                    clip.hold = 0.0
            clip.duration = round(max(target, b + 0.5), 3)
    timeline = layout_timeline(clips, spec, boundaries)

    # Render every shot's animated graphics track.
    n_layers = 0
    for i, (clip, fn) in enumerate(zip(clips, layer_fns)):
        layers = fn(clip.length) if fn else []
        n_layers += len(layers)
        if layers:
            clip.overlay = O.render_track(layers, clip.length, size, spec.fps, gfx / f"shot{i:03d}", f"s{i:03d}")

    voice_parts: list[tuple[float, Path]] = []
    all_words: list[tuple[str, float, float]] = []
    for si, path in enumerate(voice):
        if path and si in section_first:
            t0 = timeline.starts[section_first[si]] + 0.35
            voice_parts.append((t0, Path(path)))
            if use_subs:
                narration = sections[si]["narration"]
                all_words += [(w, a + t0, b + t0) for w, a, b in timing.word_timings(narration, Path(path))]
    subtitles = None
    # Subtitles never run over the end screen (it has its own text and the subscribe button).
    subs_end = timeline.starts[-1] if clips and clips[-1].kind == "card" else timeline.total
    all_words = [(w, a, min(b, subs_end - 0.05)) for w, a, b in all_words if a < subs_end - 0.15]
    if all_words:
        subtitles = O.render_track([O.Subtitles(0, subs_end, theme, size, words=all_words)], timeline.total, size,
                                   spec.fps, gfx / "subs", "subs")

    sfx_parts: list[tuple[float, Path, float]] = []
    if s.sfx and spec.uses_xfade:
        whoosh = audio.generate_whoosh(gfx / "whoosh.wav")
        for i, b in enumerate(boundaries):
            if b.duration >= 0.5:  # section-level transitions only
                at = timeline.starts[i + 1] + b.duration / 2 - audio.WHOOSH_PEAK
                sfx_parts.append((max(0.0, at), whoosh, 0.32))

    chapters: list[tuple[float, str]] = [(0.0, "Intro")] if clips and clips[0].section_index is None else []
    for si, idx in sorted(section_first.items()):
        chapters.append((timeline.starts[idx], sections[si]["heading"]))
    if chapters:
        chapters[0] = (0.0, chapters[0][1])
    return Edit(timeline, voice_parts, sfx_parts, subtitles, chapters, hero_still, hero_focus, n_layers, len(all_words),
                beat_synced)


def _shot_layers(sh: dict, first: bool, psec, sec: dict, theme, size, texts: bool, subs_on: bool, s):
    """Graphics for one shot, created once its final duration is known."""
    if not texts:
        return None
    kind, caption, sub = sh.get("overlay", "none"), (sh.get("caption") or "").strip(), (sh.get("subline") or "").strip()
    emphasis = (sh.get("emphasis") or "").strip()
    heading = sec.get("heading") or (psec.heading if psec else "")
    kicker = psec.kicker if psec else ""
    heading_emph = psec.heading_emphasis if psec else ""

    def make(d: float):
        layers = []
        if d < 1.4:
            return layers
        if first:
            layers.append(O.ChapterTitle(0.12, min(d - 0.15, 2.9), theme, size, kicker=kicker, title=heading,
                                         emphasis=heading_emph))
            if kind == "location" and caption:
                layers.append(O.LocationTag(0.5, d - 0.2, theme, size, place=caption, region=sub))
            return layers
        if kind == "caption" and caption:
            pos = "top" if subs_on else s.caption_position
            layers.append(O.Caption(0.25, d - 0.15, theme, size, text=caption, emphasis=emphasis, position=pos))
        elif kind == "lower_third" and caption:
            layers.append(O.LowerThird(0.3, d - 0.2, theme, size, title=caption, sub=sub))
        elif kind == "location" and caption:
            layers.append(O.LocationTag(0.25, d - 0.2, theme, size, place=caption, region=sub))
        elif kind == "callout" and caption:
            layers.append(O.Callout(0.35, d - 0.25, theme, size, text=caption))
        return layers

    return make


def _fmt_ts(t: float) -> str:
    t = int(t)
    return f"{t // 3600}:{t % 3600 // 60:02d}:{t % 60:02d}" if t >= 3600 else f"{t // 60}:{t % 60:02d}"


def _youtube_text(plan: StoryPlan, meta: ArticleExtras, chapters: list) -> str:
    lines = [f"TITLE\n{plan.title}", "", "DESCRIPTION", meta.youtube_description.strip(), ""]
    if chapters:
        lines.append("Chapters")
        lines += [f"{_fmt_ts(t)} {name}" for t, name in chapters]
        lines.append("")
    if meta.tags:
        lines += ["TAGS", ", ".join(meta.tags), "",
                  " ".join("#" + re.sub(r"[^\w]", "", t) for t in meta.tags[:3] if re.sub(r"[^\w]", "", t))]
    return "\n".join(lines).strip() + "\n"


# ------------------------------------------------------------ article
async def _build_article(rt: JobRuntime, plan: StoryPlan, sections: list[dict], art: ArticleExtras, media_dir: Path):
    """Markdown + standalone HTML article with photos and short web clips between sections."""

    async def export(mid: str, shot: dict) -> tuple[str, str]:
        lm = rt.local[mid]
        if lm.ref.kind == "image":
            dst = media_dir / f"{mid}.jpg"
            await asyncio.to_thread(graphics.thumbnail, lm.path, dst, 1600)
            return "image", dst.name
        dst = media_dir / f"{mid}.mp4"
        poster = media_dir / f"{mid}_poster.jpg"
        start = float(shot.get("clip_start") or 0)
        length = min(rt.settings.max_clip_seconds, max(lm.duration - start, 1))
        await run_ffmpeg(["-ss", f"{start:.2f}", "-t", f"{length:.2f}", "-i", str(lm.path),
                          "-vf", "scale='min(1280,iw)':-2,fps=30", "-c:v", "libx264", "-preset", "veryfast", "-crf", "24",
                          "-pix_fmt", "yuv420p", "-an", "-movflags", "+faststart", str(dst)])
        await run_ffmpeg(["-ss", f"{min(0.5, length / 2):.2f}", "-i", str(dst), "-frames:v", "1", "-q:v", "4", str(poster)])
        return "video", dst.name

    md = [f"# {plan.title}", f"*{plan.subtitle}*", "", "[▶ Watch the video](vlog.mp4)", "", art.intro, ""]
    body = [
        f"<h1>{html.escape(plan.title)}</h1>",
        f'<p class="subtitle">{html.escape(plan.subtitle)}</p>',
        '<video class="hero" src="vlog.mp4" poster="thumbnail.jpg" controls playsinline preload="metadata"></video>',
        _md_blocks(art.intro),
    ]
    for sec in sections:
        md += [f"## {sec['heading']}", "", sec["article_body"], ""]
        body += [f"<h2>{html.escape(sec['heading'])}</h2>", _md_blocks(sec["article_body"])]
        figures = []
        for n, sh in enumerate(sec["shots"]):
            if sh["media_id"] not in rt.local:
                continue
            kind, name = await export(sh["media_id"], sh)
            alt = sec["image_alt"][n] if n < len(sec["image_alt"]) else ""
            cap = sh.get("caption") or ""
            if kind == "image":
                md.append(f"![{alt}](media/{name})" + (f"\n*{cap}*" if cap else ""))
                figures.append(f'<figure><img src="media/{name}" alt="{html.escape(alt)}" loading="lazy">'
                               + (f"<figcaption>{html.escape(cap)}</figcaption>" if cap else "") + "</figure>")
            else:
                poster = name.replace(".mp4", "_poster.jpg")
                md.append(f'<video src="media/{name}" poster="media/{poster}" controls muted playsinline></video>')
                figures.append(f'<figure><video src="media/{name}" poster="media/{poster}" controls muted loop playsinline '
                               f'preload="none"></video>' + (f"<figcaption>{html.escape(cap)}</figcaption>" if cap else "") + "</figure>")
            md.append("")
        body.append(f'<div class="gallery g{min(len(figures), 3)}">' + "".join(figures) + "</div>")
    md += ["## Wrapping up", "", art.conclusion, "", " ".join(f"#{t.replace(' ', '')}" for t in art.tags)]
    body += ["<h2>Wrapping up</h2>", _md_blocks(art.conclusion),
             '<p class="tags">' + " ".join(f"<span>#{html.escape(t.replace(' ', ''))}</span>" for t in art.tags) + "</p>"]
    page = ARTICLE_TEMPLATE.format(
        lang=html.escape(rt.settings.language[:2].lower()), title=html.escape(art.seo_title or plan.title),
        description=html.escape(art.seo_description), body="\n".join(body))
    return "\n".join(md), page


ARTICLE_TEMPLATE = """<!doctype html>
<html lang="{lang}"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{title}</title><meta name="description" content="{description}">
<meta property="og:title" content="{title}"><meta property="og:description" content="{description}">
<meta property="og:image" content="thumbnail.jpg"><meta property="og:type" content="article">
<style>
:root{{--ink:#16161d;--muted:#5d5d6e;--accent:#ff5a3c}}
*{{box-sizing:border-box}}body{{margin:0;font:18px/1.75 Georgia,'Iowan Old Style',serif;color:var(--ink);background:#fbfaf8}}
main{{max-width:760px;margin:0 auto;padding:56px 22px 96px}}
h1,h2{{font-family:-apple-system,'Segoe UI',Inter,Helvetica,Arial,sans-serif;line-height:1.2;letter-spacing:-.02em}}
h1{{font-size:clamp(34px,6vw,52px);margin:0 0 8px}}h2{{font-size:28px;margin:56px 0 12px}}
.subtitle{{color:var(--muted);font-size:20px;margin:0 0 28px;font-style:italic}}
video,img{{width:100%;border-radius:14px;display:block;background:#000}}
.hero{{margin:0 0 36px;box-shadow:0 18px 50px rgba(0,0,0,.18)}}
figure{{margin:0}}figcaption{{font:14px/1.4 -apple-system,'Segoe UI',sans-serif;color:var(--muted);margin-top:8px}}
.gallery{{display:grid;gap:14px;margin:26px 0}}.gallery.g2{{grid-template-columns:1fr 1fr}}
.gallery.g3{{grid-template-columns:repeat(3,1fr)}}.gallery.g3 img,.gallery.g3 video,.gallery.g2 img,.gallery.g2 video{{aspect-ratio:4/5;object-fit:cover}}
@media(max-width:620px){{.gallery.g2,.gallery.g3{{grid-template-columns:1fr}}}}
blockquote{{border-left:4px solid var(--accent);margin:24px 0;padding:4px 18px;color:var(--muted)}}
.tags span{{display:inline-block;margin:4px 6px 0 0;padding:2px 10px;border-radius:99px;background:#efece6;font:14px sans-serif}}
a{{color:var(--accent)}}
</style></head><body><main>
{body}
</main></body></html>
"""
