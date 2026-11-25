"""Word-level timing for narration subtitles, estimated from the TTS audio itself.

Gemini TTS returns audio without timestamps. We find the speech segments
(energy above a noise floor, split at pauses), map text phrases onto them, and
distribute each phrase's words by their length. Accurate to ~100 ms, which is
what burned-in vlog subtitles need.
"""

from __future__ import annotations

import re
import wave
from pathlib import Path

import numpy as np


def _speech_segments(path: Path, min_pause: float = 0.18) -> tuple[list[tuple[float, float]], float]:
    with wave.open(str(path), "rb") as wf:
        sr, n, ch, sw = wf.getframerate(), wf.getnframes(), wf.getnchannels(), wf.getsampwidth()
        raw = wf.readframes(n)
    if sw != 2 or n == 0:
        return [], n / float(sr or 1)
    x = np.frombuffer(raw, dtype="<i2").astype(np.float32)
    if ch > 1:
        x = x.reshape(-1, ch).mean(axis=1)
    hop = int(sr * 0.01)
    frames = len(x) // hop
    if frames == 0:
        return [], len(x) / sr
    rms = np.sqrt(np.mean(x[: frames * hop].reshape(frames, hop) ** 2, axis=1) + 1e-9)
    thresh = max(np.percentile(rms, 20) * 2.5, rms.max() * 0.04)
    voiced = rms > thresh
    segs, start, silence = [], None, 0
    gap = int(min_pause / 0.01)
    for i, v in enumerate(voiced):
        if v:
            if start is None:
                start = i
            silence = 0
        elif start is not None:
            silence += 1
            if silence >= gap:
                segs.append((start * 0.01, (i - silence + 1) * 0.01))
                start, silence = None, 0
    if start is not None:
        segs.append((start * 0.01, (frames - silence) * 0.01))
    return [(a, b) for a, b in segs if b - a > 0.05], len(x) / sr


def _weight(word: str) -> float:
    core = re.sub(r"[^\w]", "", word)
    return max(1.0, len(core)) + (2.5 if re.search(r"[.!?]$", word) else 1.2 if re.search(r"[,;:]$", word) else 0)


def _spread(words: list[str], a: float, b: float) -> list[tuple[str, float, float]]:
    weights = [_weight(w) for w in words]
    total = sum(weights) or 1
    out, t = [], a
    for w, wt in zip(words, weights):
        d = (b - a) * wt / total
        out.append((w, round(t, 3), round(t + d, 3)))
        t += d
    return out


def word_timings(text: str, audio: Path) -> list[tuple[str, float, float]]:
    """Returns [(word, start, end)] relative to the start of `audio`."""
    words = text.split()
    if not words:
        return []
    segs, duration = _speech_segments(audio)
    if not segs:
        return _spread(words, 0.05, max(duration - 0.05, 0.5))
    # Phrases split at punctuation; greedily merge segments/phrases until counts match.
    phrases: list[list[str]] = [[]]
    for w in words:
        phrases[-1].append(w)
        if re.search(r"[.!?,;:]$", w):
            phrases.append([])
    phrases = [p for p in phrases if p]
    while len(segs) > len(phrases):  # merge the closest pair of segments
        gaps = [segs[i + 1][0] - segs[i][1] for i in range(len(segs) - 1)]
        i = int(np.argmin(gaps))
        segs[i:i + 2] = [(segs[i][0], segs[i + 1][1])]
    while len(phrases) > len(segs):  # merge the shortest adjacent phrases
        lens = [len(phrases[i]) + len(phrases[i + 1]) for i in range(len(phrases) - 1)]
        i = int(np.argmin(lens))
        phrases[i:i + 2] = [phrases[i] + phrases[i + 1]]
    out: list[tuple[str, float, float]] = []
    for phrase, (a, b) in zip(phrases, segs):
        out += _spread(phrase, a, b)
    return out
