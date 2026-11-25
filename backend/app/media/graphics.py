"""Caption overlays and title cards rendered with Pillow.

Rendering text to transparent PNGs (instead of ffmpeg drawtext) gives us
proper fonts, word wrapping, emoji-safe escaping and styled backgrounds.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont, ImageOps

from ..config import FONTS_DIR

_FONT_FILES = {
    "bold": "Poppins-Bold.ttf",
    "semibold": "Poppins-SemiBold.ttf",
    "regular": "Poppins-Regular.ttf",
    "serif": "PlayfairDisplay.ttf",
}


@lru_cache(maxsize=64)
def font(kind: str, size: int) -> ImageFont.FreeTypeFont:
    path = FONTS_DIR / _FONT_FILES[kind]
    try:
        f = ImageFont.truetype(str(path), size)
        if kind == "serif":
            try:
                f.set_variation_by_name("Bold")
            except Exception:
                pass
        return f
    except OSError:
        return ImageFont.load_default(size)


def _wrap(draw: ImageDraw.ImageDraw, text: str, fnt, max_width: int, max_lines: int = 3) -> list[str]:
    words = text.split()
    lines: list[str] = []
    current = ""
    for word in words:
        trial = f"{current} {word}".strip()
        if draw.textlength(trial, font=fnt) <= max_width or not current:
            current = trial
        else:
            lines.append(current)
            current = word
    if current:
        lines.append(current)
    if len(lines) > max_lines:
        lines = lines[:max_lines]
        lines[-1] = lines[-1].rstrip(".,;:") + "…"
    return lines


def _text_block_size(draw, lines, fnt, spacing) -> tuple[int, int, int]:
    widths, heights = [], []
    for line in lines:
        l, t, r, b = draw.textbbox((0, 0), line, font=fnt)
        widths.append(r - l)
        heights.append(b - t)
    line_h = int(fnt.size * 1.18)
    return (max(widths) if widths else 0), line_h * len(lines) + spacing * (len(lines) - 1), line_h


def caption_overlay(
    out: Path,
    size: tuple[int, int],
    caption: str | None,
    style: str = "bold",
    position: str = "bottom",
    kicker: str | None = None,
) -> Path | None:
    """Full-frame transparent PNG with a caption (and optional section kicker)."""
    if style == "none" or not (caption or kicker):
        return None
    W, H = size
    base = min(W, H)
    img = Image.new("RGBA", size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    margin = int(base * 0.06)

    if style == "cinematic":
        # Soft gradient for legibility.
        grad = Image.new("L", (1, H))
        for y in range(H):
            rel = (y / H - 0.55) / 0.45 if position == "bottom" else (0.45 - y / H) / 0.45 if position == "top" else 0
            grad.putpixel((0, y), int(max(0, min(1, rel)) * 170))
        shade = Image.new("RGBA", size, (0, 0, 0, 255))
        shade.putalpha(grad.resize(size))
        img = Image.alpha_composite(img, shade)
        draw = ImageDraw.Draw(img)

    fnt_kind, scale, color = {
        "bold": ("bold", 0.062, (255, 255, 255, 255)),
        "minimal": ("semibold", 0.045, (255, 255, 255, 255)),
        "cinematic": ("serif", 0.06, (250, 246, 238, 255)),
    }.get(style, ("bold", 0.06, (255, 255, 255, 255)))
    fnt = font(fnt_kind, max(18, int(base * scale)))
    spacing = int(fnt.size * 0.15)
    max_w = int(W * (0.84 if W < H else 0.72))
    lines = _wrap(draw, caption or "", fnt, max_w) if caption else []
    block_w, block_h, line_h = _text_block_size(draw, lines, fnt, spacing) if lines else (0, 0, 0)

    kicker_fnt = font("semibold", max(14, int(base * 0.028)))
    kicker_h = int(kicker_fnt.size * 1.9) if kicker else 0
    gap = int(base * 0.018) if kicker and lines else 0
    total_h = kicker_h + gap + block_h

    if position == "top":
        y0 = margin + (int(H * 0.04) if W < H else 0)
    elif position == "center":
        y0 = (H - total_h) // 2
    else:
        y0 = H - margin - total_h - (int(H * 0.08) if W < H else 0)  # keep clear of mobile UI

    if kicker:
        label = kicker.upper()
        tw = draw.textlength(label, font=kicker_fnt)
        pad_x = int(kicker_fnt.size * 0.9)
        kx = (W - int(tw) - 2 * pad_x) // 2
        draw.rounded_rectangle(
            (kx, y0, kx + int(tw) + 2 * pad_x, y0 + kicker_h), radius=kicker_h // 2, fill=(255, 90, 60, 235)
        )
        draw.text((kx + pad_x, y0 + kicker_h / 2), label, font=kicker_fnt, fill=(255, 255, 255, 255), anchor="lm")

    if lines:
        ty = y0 + kicker_h + gap
        if style == "bold":
            pad = int(fnt.size * 0.45)
            box = ((W - block_w) // 2 - pad, ty - pad // 2, (W + block_w) // 2 + pad, ty + block_h + pad // 2)
            draw.rounded_rectangle(box, radius=int(fnt.size * 0.35), fill=(10, 10, 14, 165))
        if style in ("minimal", "cinematic"):
            shadow = Image.new("RGBA", size, (0, 0, 0, 0))
            sd = ImageDraw.Draw(shadow)
            for i, line in enumerate(lines):
                sd.text((W / 2 + 2, ty + i * (line_h + spacing) + 3), line, font=fnt, fill=(0, 0, 0, 200), anchor="ma")
            img = Image.alpha_composite(img, shadow.filter(ImageFilter.GaussianBlur(max(2, fnt.size // 10))))
            draw = ImageDraw.Draw(img)
        for i, line in enumerate(lines):
            draw.text((W / 2, ty + i * (line_h + spacing)), line, font=fnt, fill=color, anchor="ma")

    out.parent.mkdir(parents=True, exist_ok=True)
    img.save(out)
    return out


def _cover_background(size: tuple[int, int], cover: Path | None, darken: float = 0.55) -> Image.Image:
    W, H = size
    if cover and cover.exists():
        with Image.open(cover) as src:
            bg = ImageOps.fit(ImageOps.exif_transpose(src).convert("RGB"), size, Image.Resampling.LANCZOS)
        bg = bg.filter(ImageFilter.GaussianBlur(max(8, min(W, H) // 40)))
        overlay = Image.new("RGB", size, (8, 8, 16))
        return Image.blend(bg, overlay, darken)
    # Fallback gradient.
    import numpy as np

    t = (np.linspace(0, 1, W)[None, :] + np.linspace(0, 1, H)[:, None]) / 2
    rgb = np.stack([30 + 60 * t, 20 + 30 * t, 60 + 90 * (1 - t)], axis=-1).astype("uint8")
    return Image.fromarray(rgb, "RGB")


def title_card(
    out: Path,
    size: tuple[int, int],
    title: str,
    subtitle: str | None = None,
    cover: Path | None = None,
    style: str = "bold",
) -> Path:
    W, H = size
    base = min(W, H)
    img = _cover_background(size, cover).convert("RGBA")
    draw = ImageDraw.Draw(img)
    title_font = font("serif" if style == "cinematic" else "bold", int(base * (0.095 if W >= H else 0.085)))
    sub_font = font("regular", int(base * 0.036))
    max_w = int(W * 0.82)
    t_lines = _wrap(draw, title, title_font, max_w, max_lines=3)
    s_lines = _wrap(draw, subtitle, sub_font, max_w, max_lines=2) if subtitle else []
    _, th, tlh = _text_block_size(draw, t_lines, title_font, int(title_font.size * 0.1))
    _, sh, slh = _text_block_size(draw, s_lines, sub_font, 6) if s_lines else (0, 0, 0)
    accent_h = int(base * 0.008)
    gap = int(base * 0.035)
    total = th + (gap + accent_h + gap + sh if s_lines else 0)
    y = (H - total) // 2
    for i, line in enumerate(t_lines):
        draw.text((W / 2, y + i * (tlh + int(title_font.size * 0.1))), line, font=title_font, fill="white", anchor="ma")
    if s_lines:
        y += th + gap
        aw = int(base * 0.12)
        draw.rounded_rectangle(((W - aw) // 2, y, (W + aw) // 2, y + accent_h), radius=accent_h, fill=(255, 90, 60))
        y += accent_h + gap
        for i, line in enumerate(s_lines):
            draw.text((W / 2, y + i * (slh + 6)), line, font=sub_font, fill=(235, 235, 240), anchor="ma")
    out.parent.mkdir(parents=True, exist_ok=True)
    img.convert("RGB").save(out, quality=92)
    return out


def normalize_image(src: Path, dst: Path, max_side: int = 2560) -> tuple[int, int]:
    """EXIF-rotate, convert to RGB JPEG and cap the resolution."""
    with Image.open(src) as im:
        im = ImageOps.exif_transpose(im)
        if im.mode in ("RGBA", "LA", "P"):
            im = im.convert("RGBA")
            bg = Image.new("RGB", im.size, (0, 0, 0))
            bg.paste(im, mask=im.split()[-1])
            im = bg
        else:
            im = im.convert("RGB")
        im.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
        dst.parent.mkdir(parents=True, exist_ok=True)
        im.save(dst, "JPEG", quality=92)
        return im.size


def thumbnail(src: Path, dst: Path, width: int = 640) -> Path:
    with Image.open(src) as im:
        im = im.convert("RGB")
        im.thumbnail((width, width), Image.Resampling.LANCZOS)
        im.save(dst, "JPEG", quality=85)
    return dst
