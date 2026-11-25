"""Per-job generation cost: token usage reported by Gemini x published prices, plus Cloud Run compute.

Prices are USD per 1M tokens, Gemini Developer API paid tier (Standard), from
https://ai.google.dev/gemini-api/docs/pricing. Override or extend them without a code change with
PRICING_JSON='{"model-name": {"input": 1.0, "output": 2.0, "cached": 0.1}}'.
Thinking tokens are billed as output tokens. Models without a known price are reported with their token
counts and flagged as unpriced rather than guessed.
"""

from __future__ import annotations

import json
import os
import threading
from dataclasses import dataclass, field

from pydantic import BaseModel

DEFAULT_PRICES: dict[str, dict[str, float]] = {
    "gemini-4-neon": {"input": 3.00, "output": 15.00, "cached": 0.15},
    "gemini-2.5-flash-preview-tts": {"input": 0.50, "output": 10.00, "cached": 0.0},
}

# Cloud Run instance-based billing (CPU always allocated), us-central1, before the monthly free tier.
CLOUD_RUN_VCPU_SECOND = 0.000018
CLOUD_RUN_GIB_SECOND = 0.000002


def prices() -> dict[str, dict[str, float]]:
    table = {k: dict(v) for k, v in DEFAULT_PRICES.items()}
    raw = os.environ.get("PRICING_JSON")
    if raw:
        try:
            for model, p in json.loads(raw).items():
                table[model] = {"input": float(p.get("input", 0)), "output": float(p.get("output", 0)),
                                "cached": float(p.get("cached", 0))}
        except (ValueError, AttributeError, TypeError):
            pass
    return table


def _price_for(model: str) -> dict[str, float] | None:
    table = prices()
    name = model.split("/")[-1]
    return table.get(name)


class CostLine(BaseModel):
    label: str  # e.g. "Media analysis", "Voiceover"
    model: str
    calls: int = 0
    input_tokens: int = 0
    cached_tokens: int = 0
    output_tokens: int = 0  # includes thinking tokens
    thinking_tokens: int = 0
    cost_usd: float | None = 0.0  # None = model has no known price


class JobCost(BaseModel):
    total_usd: float = 0.0
    ai_usd: float = 0.0
    compute_usd: float = 0.0
    compute_seconds: float = 0.0
    lines: list[CostLine] = []
    unpriced_models: list[str] = []
    currency: str = "USD"
    notes: list[str] = []


AGENT_LABELS = {
    "media_analyst": "Media analysis (vision)",
    "story_planner": "Story planning",
    "section_writer": "Section writing",
    "editor_in_chief": "Editor review",
    "article_editor": "Article & YouTube copy",
    "tts": "Voiceover (TTS)",
}


@dataclass
class CostTracker:
    _lines: dict[tuple[str, str], CostLine] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def record(self, label_key: str, model: str, usage) -> None:
        """`usage` is a google.genai GenerateContentResponseUsageMetadata (or None)."""
        if usage is None:
            return
        prompt = int(getattr(usage, "prompt_token_count", 0) or 0)
        cached = int(getattr(usage, "cached_content_token_count", 0) or 0)
        out = int(getattr(usage, "candidates_token_count", 0) or 0)
        thoughts = int(getattr(usage, "thoughts_token_count", 0) or 0)
        tool = int(getattr(usage, "tool_use_prompt_token_count", 0) or 0)
        label = AGENT_LABELS.get(label_key, label_key)
        model = model.split("/")[-1]
        with self._lock:
            line = self._lines.setdefault((label, model), CostLine(label=label, model=model))
            line.calls += 1
            line.input_tokens += prompt + tool
            line.cached_tokens += cached
            line.output_tokens += out + thoughts
            line.thinking_tokens += thoughts

    def summary(self, compute_seconds: float = 0.0, vcpu: float = 0.0, memory_gib: float = 0.0) -> JobCost:
        lines, unpriced, ai = [], set(), 0.0
        with self._lock:
            items = [line.model_copy() for line in self._lines.values()]
        for line in items:
            p = _price_for(line.model)
            if p is None:
                line.cost_usd = None
                unpriced.add(line.model)
            else:
                billable_in = max(line.input_tokens - line.cached_tokens, 0)
                line.cost_usd = round((billable_in * p["input"] + line.cached_tokens * p["cached"]
                                       + line.output_tokens * p["output"]) / 1_000_000, 6)
                ai += line.cost_usd
            lines.append(line)
        compute = round(compute_seconds * (vcpu * CLOUD_RUN_VCPU_SECOND + memory_gib * CLOUD_RUN_GIB_SECOND), 6)
        notes = []
        if compute_seconds and vcpu:
            notes.append(f"Compute: {compute_seconds:.0f}s of a {vcpu:g} vCPU / {memory_gib:g} GiB Cloud Run instance "
                         "(upper bound: the instance may be shared by concurrent jobs; the monthly free tier is not applied).")
        if unpriced:
            notes.append("No published price configured for: " + ", ".join(sorted(unpriced))
                         + ". Set PRICING_JSON to include them.")
        order = list(AGENT_LABELS.values())
        lines.sort(key=lambda x: order.index(x.label) if x.label in order else 99)
        return JobCost(total_usd=round(ai + compute, 6), ai_usd=round(ai, 6), compute_usd=compute,
                       compute_seconds=round(compute_seconds, 1), lines=lines, unpriced_models=sorted(unpriced),
                       notes=notes)


def format_usd(v: float | None) -> str:
    if v is None:
        return "n/a"
    if v == 0:
        return "$0.00"
    return f"${v:.4f}" if v < 0.1 else f"${v:.2f}"
