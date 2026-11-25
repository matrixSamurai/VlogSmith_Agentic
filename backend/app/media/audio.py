"""Audio generation: narration (Gemini TTS), procedural soundtrack, Lyria music."""

from __future__ import annotations

import asyncio
import base64
import logging
import wave
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)
SR = 48000


# --------------------------------------------------------------------------- TTS
async def synthesize_speech(text: str, voice: str, out: Path, model: str, style_hint: str = "",
                           on_usage=None) -> Path:
    """Gemini TTS returns 24 kHz mono 16-bit PCM; wrap it into a WAV file."""
    from google import genai
    from google.genai import types

    client = genai.Client()
    prompt = f"{style_hint.strip()}\n{text}" if style_hint else text
    resp = await client.aio.models.generate_content(
        model=model,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_modalities=["AUDIO"],
            speech_config=types.SpeechConfig(
                voice_config=types.VoiceConfig(prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=voice))
            ),
        ),
    )
    if on_usage is not None:
        on_usage(model, resp.usage_metadata)
    part = resp.candidates[0].content.parts[0]
    pcm = part.inline_data.data
    if isinstance(pcm, str):
        pcm = base64.b64decode(pcm)
    rate = 24000
    mime = part.inline_data.mime_type or ""
    if "rate=" in mime:
        try:
            rate = int(mime.split("rate=")[1].split(";")[0])
        except ValueError:
            pass
    out.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(out), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        wf.writeframes(pcm)
    return out


def wav_duration(path: Path) -> float:
    with wave.open(str(path), "rb") as wf:
        return wf.getnframes() / float(wf.getframerate())


# ------------------------------------------------------------------- SFX
WHOOSH_PEAK = 0.42  # seconds into the whoosh file where it is loudest


def generate_whoosh(out: Path, seed: int = 3) -> Path:
    """Airy transition whoosh: band-passed noise with a rising-then-falling sweep."""
    rng = np.random.default_rng(seed)
    n = int(0.8 * SR)
    t = np.arange(n) / SR
    noise = rng.standard_normal(n)
    env = np.exp(-((t - WHOOSH_PEAK) ** 2) / (2 * 0.11 ** 2))
    # Time-varying one-pole low-pass: cutoff sweeps up to the peak and back down.
    cutoff = 400 + 5200 * env
    alpha = 1 - np.exp(-2 * np.pi * cutoff / SR)
    y = np.zeros(n)
    acc = 0.0
    for i in range(n):
        acc += alpha[i] * (noise[i] - acc)
        y[i] = acc
    y = (y - np.convolve(y, np.ones(24) / 24, mode="same")) * env  # remove rumble
    y /= max(1e-6, np.abs(y).max())
    pan = 0.5 + 0.45 * np.tanh((t - WHOOSH_PEAK) * 8)  # sweeps left -> right
    stereo = np.stack([y * (1 - pan), y * pan], axis=1) * 0.9
    out.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(out), "wb") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(SR)
        wf.writeframes((stereo * 32767).astype("<i2").tobytes())
    return out


def mood_bpm(mood: str) -> float:
    return float(MOODS.get(mood, MOODS["calm"])[0])


# ------------------------------------------------------------- Lyria (Vertex AI)
async def generate_lyria(prompt: str, out: Path, project: str, location: str, model: str) -> Path:
    """Vertex AI Lyria returns a ~30 s 48 kHz WAV clip; the mixer loops it."""
    import google.auth
    import httpx
    from google.auth.transport.requests import Request

    creds, _ = google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
    await asyncio.to_thread(creds.refresh, Request())
    url = (
        f"https://{location}-aiplatform.googleapis.com/v1/projects/{project}/locations/{location}"
        f"/publishers/google/models/{model}:predict"
    )
    body = {"instances": [{"prompt": prompt, "negative_prompt": "vocals, singing, lyrics, harsh noise"}], "parameters": {}}
    async with httpx.AsyncClient(timeout=180) as client:
        r = await client.post(url, json=body, headers={"Authorization": f"Bearer {creds.token}"})
        r.raise_for_status()
        pred = r.json()["predictions"][0]
    data = pred.get("bytesBase64Encoded") or pred.get("audioContent")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(base64.b64decode(data))
    return out


# ------------------------------------------------------- Procedural soundtrack
MOODS = {
    #           bpm  root   mode      progression (scale degrees)   drums  brightness
    "upbeat": (118, 60, "major", [0, 4, 5, 3], True, 0.9),
    "calm": (72, 57, "major", [0, 5, 3, 4], False, 0.4),
    "epic": (90, 50, "minor", [0, 5, 2, 6], True, 0.7),
    "romantic": (68, 55, "major", [0, 2, 3, 4], False, 0.5),
    "adventurous": (108, 52, "minor", [0, 6, 5, 6], True, 0.8),
    "dreamy": (64, 58, "major", [0, 3, 5, 4], False, 0.3),
    "corporate": (104, 60, "major", [0, 5, 3, 4], True, 0.75),
    "playful": (124, 62, "major", [0, 3, 4, 0], True, 1.0),
}
SCALES = {"major": [0, 2, 4, 5, 7, 9, 11], "minor": [0, 2, 3, 5, 7, 8, 10]}


def _midi_hz(n: float) -> float:
    return 440.0 * 2 ** ((n - 69) / 12)


def _chord(root: int, mode: str, degree: int) -> list[int]:
    sc = SCALES[mode]
    notes = []
    for step in (0, 2, 4, 6 if degree % 3 == 0 else 7):
        idx = degree + step
        notes.append(root + sc[idx % 7] + 12 * (idx // 7))
    return notes


def _tone(freq: float, n: int, bright: float, detune: float = 0.0) -> np.ndarray:
    t = np.arange(n) / SR
    out = np.zeros(n)
    for h, amp in ((1, 1.0), (2, 0.35 * bright), (3, 0.18 * bright), (4, 0.08 * bright)):
        out += amp * np.sin(2 * np.pi * freq * h * (1 + detune) * t)
    return out


def _envelope(n: int, attack: float, release: float) -> np.ndarray:
    a, r = int(attack * SR), int(release * SR)
    env = np.ones(n)
    if a:
        env[: min(a, n)] = np.linspace(0, 1, min(a, n))
    if r:
        env[max(0, n - r):] *= np.linspace(1, 0, n - max(0, n - r))
    return env


def generate_soundtrack(mood: str, duration: float, out: Path, seed: int = 7) -> Path:
    """A gentle, royalty-free bed: pads + bass + arpeggio (+ light drums)."""
    bpm, root, mode, prog, drums, bright = MOODS.get(mood, MOODS["calm"])
    rng = np.random.default_rng(seed)
    beat = 60 / bpm
    bar = beat * 4
    total = int((duration + 1) * SR)
    L = np.zeros(total)
    R = np.zeros(total)
    bar_n = int(bar * SR)
    n_bars = int(np.ceil(total / bar_n))
    for b in range(n_bars):
        start = b * bar_n
        n = min(bar_n * 2, total - start)  # pads ring into the next bar
        if n <= 0:
            break
        chord = _chord(root, mode, prog[b % len(prog)])
        pad = sum(_tone(_midi_hz(note), n, bright * 0.5, d) for note, d in zip(chord, (0, 0.002, -0.002, 0.001)))
        pad *= _envelope(n, bar * 0.35, bar * 0.9) * 0.07
        L[start:start + n] += pad
        R[start:start + n] += np.roll(pad, int(0.012 * SR))
        bass_n = min(bar_n, total - start)
        bass = np.sin(2 * np.pi * _midi_hz(chord[0] - 12) * np.arange(bass_n) / SR)
        bass *= _envelope(bass_n, 0.02, bar * 0.4) * 0.12
        L[start:start + bass_n] += bass
        R[start:start + bass_n] += bass
        # Arpeggio in eighth notes.
        step_n = int(beat / 2 * SR)
        pattern = [0, 1, 2, 3, 2, 1, 2, 3] if bpm > 90 else [0, 2, 1, 3]
        step_len = step_n if bpm > 90 else step_n * 2
        for i, idx in enumerate(pattern):
            s = start + i * step_len
            m = min(int(step_len * 1.6), total - s)
            if m <= 0:
                break
            note = chord[idx % len(chord)] + 12
            pluck = _tone(_midi_hz(note), m, bright) * np.exp(-np.arange(m) / (SR * 0.18)) * 0.045
            pan = 0.5 + 0.35 * np.sin(i)
            L[s:s + m] += pluck * (1 - pan)
            R[s:s + m] += pluck * pan
        if drums:
            for i in range(4):
                s = start + int(i * beat * SR)
                kn = min(int(0.25 * SR), total - s)
                if kn <= 0:
                    break
                tk = np.arange(kn) / SR
                kick = np.sin(2 * np.pi * (50 + 90 * np.exp(-tk * 30)) * tk) * np.exp(-tk * 12) * 0.22
                L[s:s + kn] += kick
                R[s:s + kn] += kick
                hs = s + int(beat / 2 * SR)
                hn = min(int(0.05 * SR), total - hs)
                if hn > 0:
                    hat = rng.standard_normal(hn) * np.exp(-np.arange(hn) / (SR * 0.01)) * 0.025
                    hat = np.diff(hat, prepend=0)
                    L[hs:hs + hn] += hat
                    R[hs:hs + hn] += hat * 0.8
    # Cheap stereo "room": two feedback delays.
    for d, g in ((int(0.083 * SR), 0.25), (int(0.127 * SR), 0.2)):
        L[d:] += g * R[:-d]
        R[d:] += g * L[:-d]
    stereo = np.stack([L, R], axis=1)
    stereo /= max(1e-6, np.abs(stereo).max()) / 0.85
    fade = int(2 * SR)
    stereo[:fade] *= np.linspace(0, 1, fade)[:, None]
    stereo[-fade:] *= np.linspace(1, 0, fade)[:, None]
    pcm = (stereo * 32767).astype("<i2")
    out.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(out), "wb") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(SR)
        wf.writeframes(pcm.tobytes())
    return out
