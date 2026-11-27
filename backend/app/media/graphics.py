"""Still-image graphics: input normalization, previews and the YouTube-style thumbnail."""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageOps

from . import overlays as O


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


def youtube_thumbnail(out: Path, hero: Path | None, text: str, emphasis: str = "", theme: str = "bold",
                      focus: tuple[float, float] = (0.5, 0.5), size: tuple[int, int] = (1280, 720)) -> Path:
    """1280x720 click-worthy thumbnail: punchy graded hero, vignette, huge outlined text."""
    W, H = size
    if hero and hero.exists():
        with Image.open(hero) as src:
            img = ImageOps.fit(ImageOps.exif_transpose(src).convert("RGB"), size, Image.Resampling.LANCZOS,
                               centering=focus)
        img = ImageEnhance.Color(img).enhance(1.35)
        img = ImageEnhance.Contrast(img).enhance(1.12)
        img = img.filter(ImageFilter.UnsharpMask(radius=2, percent=80, threshold=2))
    else:
        img = Image.new("RGB", size, (30, 30, 50))
    canvas = img.convert("RGBA")
    canvas.alpha_composite(O.scrim(size, "radial", 90))
    th = O.get_theme(theme if theme in O.THEMES else "bold")
    # Text sits on the side opposite the subject so faces / landmarks stay visible.
    text_left = focus[0] >= 0.5
    words = O._tokens(text, emphasis, False)
    if words and not any(e for _, e in words):  # always pop one word: the last one
        words[-1] = (words[-1][0], True)
    kind = th.display  # same bold script as the video titles
    size_px = int(H * 0.2)
    stroke = lambda px: int(px * 0.06)  # noqa: E731
    st = O.WordStyle(kind, size_px, (255, 255, 255, 255), (0, 0, 0, 255), stroke(size_px), int(size_px * 0.08))
    em = O.WordStyle(kind, size_px, th.emph if th.emph_mode == "color" else (255, 208, 64, 255), (0, 0, 0, 255),
                     stroke(size_px), int(size_px * 0.08))
    placed, bw, bh = O.layout(words, st, em, W * 0.56, align="left", line_gap=-0.12, max_lines=3)
    while bh > H * 0.8 and size_px > 40:  # shrink until it fits
        size_px = int(size_px * 0.88)
        st = O.WordStyle(kind, size_px, st.fill, st.stroke, stroke(size_px), int(size_px * 0.08))
        em = O.WordStyle(kind, size_px, em.fill, em.stroke, stroke(size_px), int(size_px * 0.08))
        placed, bw, bh = O.layout(words, st, em, W * 0.56, align="left", line_gap=-0.12, max_lines=3)
    side = O.scrim(size, "left", 150)
    canvas.alpha_composite(side if text_left else side.transpose(Image.Transpose.FLIP_LEFT_RIGHT))
    ox = W * 0.05 if text_left else W * 0.95 - bw
    oy = (H - bh) / 2
    for pw in placed:
        O.draw_word(canvas, pw, ox, oy)
    # Accent frame bar, like many creator thumbnails.
    ImageDraw.Draw(canvas).rectangle((0, H - 14, W, H), fill=th.accent)
    out.parent.mkdir(parents=True, exist_ok=True)
    canvas.convert("RGB").save(out, "JPEG", quality=92)
    return out
