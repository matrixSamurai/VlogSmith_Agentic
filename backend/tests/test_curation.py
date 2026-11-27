"""Unit tests for photo curation in the story plan."""

from pathlib import Path
from types import SimpleNamespace

from app.agents.pipeline import _photo_budget, _repair_plan
from app.agents.schemas import PlannedSection, StoryPlan
from app.models import BlogSettings


def _rt(kinds: dict[str, str]):
    local = {mid: SimpleNamespace(ref=SimpleNamespace(kind=k)) for mid, k in kinds.items()}
    return SimpleNamespace(local=local)


def _plan(media_per_section):
    return StoryPlan(
        title="T", title_emphasis="", subtitle="s", hook="h", thumbnail_text="t", look="natural",
        music_mood="calm", music_prompt="p", outro_text="o",
        sections=[PlannedSection(heading=f"S{i}", heading_emphasis="", kicker="", purpose="", media_ids=ids, seconds=10)
                  for i, ids in enumerate(media_per_section)],
    )


def test_budget_from_length_and_cap():
    assert _photo_budget(BlogSettings(context="abc", target_duration=120), 50) == 30  # ~1 per 4s
    assert _photo_budget(BlogSettings(context="abc", target_duration=120), 10) == 10  # never exceeds uploaded
    assert _photo_budget(BlogSettings(context="abc", target_duration=600, max_photos=12), 50) == 12  # explicit cap
    assert _photo_budget(BlogSettings(context="abc", target_duration=20), 50) == 6   # floor of 6


def test_curate_keeps_planner_selection_and_all_videos():
    insights = {f"m{i}": {"quality": i} for i in range(1, 9)}  # m1..m8
    kinds = {f"m{i}": "image" for i in range(1, 8)} | {"m8": "video"}
    # Planner used only m5, m6, m7 and dropped the rest (including the video m8 by mistake).
    plan = _repair_plan(_plan([["m5", "m6", "m7"]]), insights, _rt(kinds), curate=True, max_photos=5)
    used = {m for sec in plan.sections for m in sec.media_ids}
    assert "m8" in used, "every video must be kept"
    assert used == {"m5", "m6", "m7", "m8"}, "skipped photos stay skipped"


def test_curate_enforces_photo_budget_dropping_weakest():
    insights = {f"m{i}": {"quality": i} for i in range(1, 9)}  # quality == index
    kinds = {f"m{i}": "image" for i in range(1, 9)}
    plan = _repair_plan(_plan([[f"m{i}" for i in range(1, 9)]]), insights, _rt(kinds), curate=True, max_photos=3)
    used = {m for sec in plan.sections for m in sec.media_ids}
    assert used == {"m6", "m7", "m8"}, "keeps the 3 highest-quality photos"


def test_no_curate_keeps_everything():
    insights = {f"m{i}": {"quality": 5} for i in range(1, 6)}
    kinds = {f"m{i}": "image" for i in range(1, 6)}
    plan = _repair_plan(_plan([["m1"]]), insights, _rt(kinds), curate=False)
    used = {m for sec in plan.sections for m in sec.media_ids}
    assert used == set(insights), "legacy mode uses every file"
