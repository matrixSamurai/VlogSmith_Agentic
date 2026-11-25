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
import shutil
import zipfile
from pathlib import Path
from typing import Any

from google.adk.agents import Context, LlmAgent
from google.adk.workflow import START, JoinNode, Workflow, node
from google.genai import types

from ..config import get_settings
from ..media import audio, graphics
from ..media.ffmpeg import FFmpegError, extract_frames, probe, run_ffmpeg
from ..media.renderer import Clip, RenderSpec, layout_timeline, render_timeline
from ..models import JobOutputs
from . import prompts
from .runtime import JobRuntime, LocalMedia
from .schemas import ArticleExtras, MediaInsight, ScriptReview, SectionScript, StoryPlan

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
            f"Setting: {ins['setting']}. Subjects: {', '.join(ins['subjects'])}. Caption idea: {ins['caption_idea']}"
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
                description=lm.ref.filename, subjects=[], setting="", mood="neutral", quality=5,
                caption_idea="", highlight_start=0, highlight_end=min(lm.duration, s.max_clip_seconds), tags=[],
            ).model_dump()
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
                                                  cfg.model_tts, style)
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
        length = _clamp(s.target_duration * 1.6 + 20, 40, 900)
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
        rt.extras["music"] = track
        return {"music": source if track else "none", "mood": mood}

    @node(name="write_article", rerun_on_resume=True)
    async def write_article(ctx: Context, node_input: Any):
        if not s.generate_article:
            return {"article": False}
        await rt.info("production", "Article editor is writing the intro, conclusion and SEO metadata", 0.59)
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
                tags=[]).model_dump()
        return {"article": True}

    production_join = JoinNode(name="production_join")

    # ---------------------------------------------------------------- render
    @node(name="render_video", rerun_on_resume=True)
    async def render_video(ctx: Context, node_input: Any):
        plan = StoryPlan.model_validate(ctx.state["plan"])
        sections = ctx.state["sections"]
        W, H = s.frame_size
        spec = RenderSpec(W, H, s.fps, s.transition, s.transition_duration, s.ken_burns, s.quality,
                          s.clip_audio, s.clip_audio_volume)
        clips, section_first, cover = _build_clips(rt, plan, sections, spec, ctx.state["insights"])
        timeline = layout_timeline(clips, spec)
        voice_parts = []
        for i, path in enumerate(rt.extras.get("voice", [])):
            if path and i in section_first:
                voice_parts.append((timeline.starts[section_first[i]] + 0.35, Path(path)))
        await rt.info("render", f"Rendering {len(clips)} shots ({timeline.total:.0f}s at {W}x{H})", 0.62)

        async def progress(done: int, total: int):
            await rt.info("render", f"Rendered shot {done}/{total}", 0.62 + 0.25 * done / total)

        out = work / "out" / "vlog.mp4"
        out.parent.mkdir(parents=True, exist_ok=True)
        await render_timeline(timeline, spec, work / "render", out, rt.extras.get("music"), s.music_volume,
                              voice_parts, concurrency=cfg.render_concurrency, on_progress=progress)
        thumb = work / "out" / "thumbnail.jpg"
        await asyncio.to_thread(graphics.title_card, thumb, (1280, int(1280 * H / W)) if W >= H else (720, int(720 * H / W)),
                                plan.title, plan.subtitle, cover, s.caption_style)
        rt.extras["video"], rt.extras["thumbnail"], rt.extras["cover"] = out, thumb, cover
        ctx.state["duration"] = timeline.total
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
        outputs = JobOutputs(video="vlog.mp4", thumbnail="thumbnail.jpg")
        files: list[tuple[Path, str, str]] = [
            (rt.extras["video"], "vlog.mp4", "video/mp4"), (rt.extras["thumbnail"], "thumbnail.jpg", "image/jpeg")]

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

        plan.sections = [PlannedSection(heading=plan.title or "The story", kicker="", purpose="", media_ids=[], seconds=30)]
    for m in missing:  # append leftovers to the smallest section
        min(plan.sections, key=lambda x: len(x.media_ids)).media_ids.append(m)
    return plan


def _repair_script(script: SectionScript, media_ids: list[str], insights: dict, rt: JobRuntime) -> SectionScript:
    from .schemas import Shot

    by_id = {sh.media_id: sh for sh in script.shots if sh.media_id in media_ids}
    shots, alts = [], []
    for n, mid in enumerate(media_ids):
        ins = insights.get(mid, {})
        sh = by_id.get(mid) or Shot(media_id=mid, caption=ins.get("caption_idea", ""), seconds=4,
                                    clip_start=ins.get("highlight_start", 0))
        shots.append(sh)
        alts.append(script.image_alt[n] if n < len(script.image_alt) else ins.get("description", ""))
    script.shots, script.image_alt = shots, alts
    return script


# ------------------------------------------------------------ timeline construction
def _build_clips(rt: JobRuntime, plan: StoryPlan, sections: list[dict], spec: RenderSpec, insights: dict):
    s = rt.settings
    W, H = spec.width, spec.height
    gfx = rt.work_dir / "gfx"
    min_shot = max(1.6, 2 * spec.overlap + 0.4)
    clips: list[Clip] = []
    section_first: dict[int, int] = {}

    photos = [m for m in rt.local.values() if m.ref.kind == "image"]
    cover_media = max(photos, key=lambda m: insights.get(m.ref.id, {}).get("quality", 0), default=None)
    cover = cover_media.path if cover_media else None
    if cover is None:
        vids = [m for m in rt.local.values() if m.frames]
        cover = vids[0].frames[len(vids[0].frames) // 2][1] if vids else None

    if s.include_title_card:
        card = graphics.title_card(gfx / "title.jpg", (W, H), plan.title, plan.subtitle, cover, s.caption_style)
        clips.append(Clip("card", card, TITLE_SECONDS, motion=0))

    # Scale shot lengths towards the target duration.
    raw_total = sum(sh["seconds"] for sec in sections for sh in sec["shots"])
    budget = max(10.0, s.target_duration - (TITLE_SECONDS if s.include_title_card else 0) - (OUTRO_SECONDS if s.include_outro else 0))
    factor = _clamp(budget / raw_total, 0.6, 1.8) if raw_total > 0 else 1.0

    voice = rt.extras.get("voice", [])
    motion = 0
    for si, sec in enumerate(sections):
        section_clips: list[Clip] = []
        for n, sh in enumerate(sec["shots"]):
            lm = rt.local.get(sh["media_id"])
            if not lm:
                continue
            caption = sh["caption"].strip() if sh.get("caption") else ""
            kicker = (plan.sections[si].kicker if si < len(plan.sections) else None) if n == 0 else None
            overlay = graphics.caption_overlay(gfx / f"ov_{si}_{n}.png", (W, H), caption or None, s.caption_style,
                                               s.caption_position, kicker=kicker or None)
            if lm.ref.kind == "image":
                section_clips.append(Clip("image", lm.path, _clamp(sh["seconds"] * factor, max(min_shot, 2.2), 8.0),
                                          overlay, motion=motion, section_index=si))
                motion += 1
            else:
                start = _clamp(float(sh.get("clip_start") or 0), 0, max(0, lm.duration - min_shot))
                avail = lm.duration - start
                dur = _clamp(sh["seconds"] * factor, min(min_shot, avail), min(s.max_clip_seconds, avail))
                section_clips.append(Clip("video", lm.path, max(dur, 0.8), overlay, clip_start=start,
                                          source_duration=lm.duration, has_audio=lm.has_audio, section_index=si))
        if not section_clips:
            continue
        # Make sure the section is long enough for its narration.
        vpath = voice[si] if si < len(voice) else None
        if vpath:
            needed = audio.wav_duration(Path(vpath)) + 0.9
            have = sum(c.duration for c in section_clips) - spec.overlap * (len(section_clips) - 1)
            if needed > have:
                extra = needed - have
                images = [c for c in section_clips if c.kind == "image"]
                if images:
                    for c in images:
                        c.duration += extra / len(images)
                else:
                    section_clips[-1].hold = extra
        section_first[si] = len(clips)
        clips.extend(section_clips)

    if s.include_outro:
        outro = graphics.title_card(gfx / "outro.jpg", (W, H), s.outro_text or plan.outro_text or "Thanks for watching",
                                    plan.title, cover, s.caption_style)
        clips.append(Clip("card", outro, OUTRO_SECONDS, motion=1))
    return clips, section_first, cover


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
