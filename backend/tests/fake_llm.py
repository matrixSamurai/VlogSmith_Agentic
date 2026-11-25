"""Deterministic stand-in for Gemini so the full workflow can run in tests/CI."""

from __future__ import annotations

import json
import re
from typing import AsyncGenerator

from google.adk.models.base_llm import BaseLlm
from google.adk.models.llm_response import LlmResponse
from google.genai import types


class FakeGemini(BaseLlm):
    model: str = "fake-gemini"
    calls: dict = {}
    reject_first_review: bool = True

    def _text(self, req) -> str:
        return "\n".join(p.text for c in req.contents for p in (c.parts or []) if p.text)

    def _count(self, name: str) -> int:
        self.calls[name] = self.calls.get(name, 0) + 1
        return self.calls[name]

    async def generate_content_async(self, llm_request, stream: bool = False) -> AsyncGenerator[LlmResponse, None]:
        schema = getattr(llm_request.config, "response_schema", None)
        name = getattr(schema, "__name__", "")
        text = self._text(llm_request)
        images = sum(1 for c in llm_request.contents for p in (c.parts or []) if p.inline_data)
        self._count(name)
        if name == "MediaInsight":
            mid = re.search(r"Media id (m\d+)", text).group(1)
            is_video = "a VIDEO" in text
            out = dict(description=f"Scenic shot {mid} with {images} frame(s).", subjects=["lake", "mountains"],
                       setting="alpine lake", mood="serene", quality=7 if mid == "m1" else 6,
                       caption_idea=f"Moment {mid}", highlight_start=1.0 if is_video else 0,
                       highlight_end=4.0 if is_video else 0, tags=["travel", "alps"])
        elif name == "StoryPlan":
            ids = list(dict.fromkeys(re.findall(r"id=(m\d+)", text)))
            half = max(1, len(ids) // 2)
            groups = [ids[:half], ids[half:]] if len(ids) > 1 else [ids]
            out = dict(title="Chasing Light in the Alps", subtitle="Three days of lakes and trains",
                       hook="We came for the views and stayed for the fondue.", music_mood="adventurous",
                       music_prompt="warm acoustic adventure", outro_text="See you on the next trail",
                       sections=[dict(heading=f"Part {i + 1}", kicker=f"Day {i + 1}", purpose="story beat",
                                      media_ids=g, seconds=8) for i, g in enumerate(groups) if g])
        elif name == "SectionScript":
            m = re.search(r"Media in order:\n(.*?)(?:\n\n|\nEditor|$)", text, re.S)
            ids = re.findall(r"id=(m\d+)", m.group(1) if m else text)
            revised = "Editor feedback" in text
            out = dict(heading="Revised heading" if revised else "Golden hour",
                       narration="The air was crisp and the lake perfectly still.",
                       shots=[dict(media_id=i, caption=f"Caption for {i}", seconds=3, clip_start=1.0) for i in ids],
                       article_body="We woke before dawn.\n\nThe **light** was *unreal*.", image_alt=[f"alt {i}" for i in ids])
        elif name == "ScriptReview":
            n = self.calls[name]
            if self.reject_first_review and n == 1:
                out = dict(approved=False, score=6, summary="Opening needs punch",
                           issues=[dict(section_index=0, problem="Weak hook", fix="Lead with the sunrise")])
            else:
                out = dict(approved=True, score=9, summary="Great", issues=[])
        elif name == "ArticleExtras":
            out = dict(intro="An intro <script>alert(1)</script> paragraph.", conclusion="Go see it yourself.",
                       seo_title="Alps vlog", seo_description="A trip to the Alps.", tags=["alps", "travel"])
        else:
            out = {}
        yield LlmResponse(content=types.Content(role="model", parts=[types.Part(text=json.dumps(out))]))
