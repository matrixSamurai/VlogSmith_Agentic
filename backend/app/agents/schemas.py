"""Structured outputs the LLM agents must produce (enforced via output_schema)."""

from __future__ import annotations

from pydantic import BaseModel, Field


class MediaInsight(BaseModel):
    description: str = Field(description="Two vivid sentences describing what is shown.")
    subjects: list[str] = Field(description="Main subjects, places or objects.")
    setting: str = Field(description="Where / when this appears to be (e.g. 'mountain lake at sunrise').")
    mood: str = Field(description="One or two words for the emotional tone.")
    quality: int = Field(description="Visual quality and storytelling value, 1 (unusable) to 10 (hero shot).")
    caption_idea: str = Field(description="A punchy on-screen caption idea, at most 8 words.")
    highlight_start: float = Field(description="Video only: best moment start in seconds. 0 for images.")
    highlight_end: float = Field(description="Video only: best moment end in seconds. 0 for images.")
    tags: list[str] = Field(description="3-6 short topical tags.")


class PlannedSection(BaseModel):
    heading: str = Field(description="Section heading for the article, at most 8 words.")
    kicker: str = Field(description="1-3 word label shown on screen when the section starts (e.g. 'Day One').")
    purpose: str = Field(description="What this section contributes to the story.")
    media_ids: list[str] = Field(description="Ordered media ids shown in this section.")
    seconds: float = Field(description="Approximate screen time for this section in seconds.")


class StoryPlan(BaseModel):
    title: str = Field(description="Catchy blog / video title, at most 70 characters.")
    subtitle: str = Field(description="One-line subtitle shown on the title card.")
    hook: str = Field(description="One or two sentences that open the story.")
    music_mood: str = Field(description="One of: upbeat, calm, epic, romantic, adventurous, dreamy, corporate, playful.")
    music_prompt: str = Field(description="A short instrumental music description for an AI music generator.")
    outro_text: str = Field(description="Closing on-screen line, at most 10 words.")
    sections: list[PlannedSection]


class Shot(BaseModel):
    media_id: str
    caption: str = Field(description="On-screen text for this shot, at most 9 words. Empty string for none.")
    seconds: float = Field(description="Screen time for this shot in seconds.")
    clip_start: float = Field(description="Video only: start offset in seconds into the source clip. 0 for images.")


class SectionScript(BaseModel):
    heading: str
    narration: str = Field(description="Voiceover script for this section. Spoken, warm, no stage directions.")
    shots: list[Shot]
    article_body: str = Field(description="Markdown paragraphs for the written blog section. No headings, no images.")
    image_alt: list[str] = Field(description="Alt text for each shot, same order as shots.")


class ReviewIssue(BaseModel):
    section_index: int = Field(description="0-based index of the section with the problem.")
    problem: str
    fix: str = Field(description="Concrete instruction for the writer.")


class ScriptReview(BaseModel):
    approved: bool = Field(description="True if the script is ready to publish as is.")
    score: int = Field(description="Overall quality 1-10.")
    summary: str
    issues: list[ReviewIssue]


class ArticleExtras(BaseModel):
    intro: str = Field(description="Markdown introduction, 1-2 paragraphs.")
    conclusion: str = Field(description="Markdown closing paragraph with a gentle call to action.")
    seo_title: str
    seo_description: str = Field(description="Meta description, at most 155 characters.")
    tags: list[str]
