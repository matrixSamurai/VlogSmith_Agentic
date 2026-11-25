"""Animated motion-graphics overlays (YouTube-style titles, lower thirds, captions...).

Each overlay is a `Layer` that can draw itself at any time `t`. A shot's layers are
rendered into an "overlay track": RGBA PNG keyframes, one per visual change
(every frame while something animates, a single long-held frame while static).
The track is fed to ffmpeg through the concat demuxer, so a 6-second shot with a
0.8-second entrance animation costs ~30 PNGs instead of 180.
"""

from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont

from ..config import FONTS_DIR

RGBA = tuple[int, int, int, int]

# ----------------------------------------------------------------------------- fonts
_FILES = {
    "black": "Poppins-Black.ttf", "extrabold": "Poppins-ExtraBold.ttf", "bold": "Poppins-Bold.ttf",
    "semibold": "Poppins-SemiBold.ttf", "medium": "Poppins-Medium.ttf", "regular": "Poppins-Regular.ttf",
    "anton": "Anton-Regular.ttf", "serif": "PlayfairDisplay.ttf", "serif_italic": "PlayfairDisplay-Italic.ttf",
}
_WEIGHT = {"black": "Black", "extrabold": "ExtraBold", "bold": "Bold", "semibold": "SemiBold", "medium": "Medium",
           "regular": "Regular", "anton": "Black", "serif": "Bold", "serif_italic": "Bold Italic"}


def _script(text: str) -> str:
    if any("\u0900" <= ch <= "\u097f" for ch in text):
        return "deva"
    if any(ord(ch) > 0x24F and not (0x2000 <= ord(ch) <= 0x206F) for ch in text):
        return "noto"
    return "latin"


@lru_cache(maxsize=256)
def _font(kind: str, size: int, script: str = "latin") -> ImageFont.FreeTypeFont:
    if script == "deva":
        path, variation = FONTS_DIR / "NotoSansDevanagari.ttf", _WEIGHT.get(kind, "Bold").split()[0]
    elif script == "noto":
        path, variation = FONTS_DIR / "NotoSans.ttf", _WEIGHT.get(kind, "Bold").split()[0]
    else:
        path = FONTS_DIR / _FILES[kind]
        variation = _WEIGHT[kind] if kind.startswith("serif") else None
    try:
        f = ImageFont.truetype(str(path), max(8, size))
    except OSError:
        return ImageFont.load_default(size)
    if variation:
        for name in (variation, variation.replace(" ", ""), "Bold"):
            try:
                f.set_variation_by_name(name)
                break
            except Exception:  # noqa: BLE001 - not a variable font or unknown instance
                continue
    return f


def font(kind: str, size: int, text: str = "") -> ImageFont.FreeTypeFont:
    return _font(kind, int(size), _script(text))


# ----------------------------------------------------------------------------- themes
@dataclass(frozen=True)
class Theme:
    name: str
    display: str  # font kind for big titles
    caption: str  # font kind for captions / subtitles
    body: str
    upper: bool  # uppercase display text
    text: RGBA
    stroke: RGBA | None
    stroke_ratio: float
    emph_mode: str  # color | box | italic
    emph: RGBA
    emph_text: RGBA
    accent: RGBA
    panel: RGBA
    stickers: tuple[RGBA, ...]
    sub_mode: str  # color | box
    sub_emph: RGBA
    tracking: float = 0.0  # extra letter spacing for kickers


THEMES: dict[str, Theme] = {
    "bold": Theme("bold", "black", "extrabold", "semibold", True, (255, 255, 255, 255), (0, 0, 0, 255), 0.085,
                  "color", (255, 214, 0, 255), (0, 0, 0, 255), (255, 59, 48, 255), (12, 12, 16, 215),
                  ((255, 214, 0, 255),), "color", (255, 214, 0, 255)),
    "minimal": Theme("minimal", "bold", "semibold", "medium", False, (255, 255, 255, 255), None, 0.0,
                     "box", (255, 90, 60, 255), (255, 255, 255, 255), (255, 90, 60, 255), (14, 14, 18, 190),
                     ((255, 255, 255, 255),), "box", (255, 255, 255, 235)),
    "cinematic": Theme("cinematic", "serif", "serif", "medium", False, (250, 244, 232, 255), None, 0.0,
                       "italic", (232, 194, 122, 255), (20, 16, 10, 255), (232, 194, 122, 255), (10, 9, 8, 170),
                       ((232, 194, 122, 255),), "color", (232, 194, 122, 255), 0.28),
    "pop": Theme("pop", "anton", "anton", "bold", True, (255, 255, 255, 255), (0, 0, 0, 255), 0.06,
                 "box", (255, 61, 127, 255), (255, 255, 255, 255), (60, 240, 110, 255), (20, 20, 28, 220),
                 ((255, 61, 127, 255), (255, 214, 0, 255), (0, 200, 255, 255), (130, 90, 255, 255)),
                 "box", (60, 240, 110, 255)),
}


def get_theme(name: str) -> Theme:
    return THEMES.get(name, THEMES["bold"])


# ----------------------------------------------------------------------------- easing
def clamp01(x: float) -> float:
    return 0.0 if x < 0 else 1.0 if x > 1 else x


def ease_out_cubic(p: float) -> float:
    p = clamp01(p)
    return 1 - (1 - p) ** 3


def ease_in_out(p: float) -> float:
    p = clamp01(p)
    return 0.5 - 0.5 * math.cos(math.pi * p)


def ease_out_back(p: float, s: float = 1.9) -> float:
    p = clamp01(p) - 1
    return 1 + (s + 1) * p ** 3 + s * p ** 2


# ----------------------------------------------------------------------------- drawing primitives
def _with_alpha(img: Image.Image, alpha: float) -> Image.Image:
    if alpha >= 0.999:
        return img
    out = img.copy()
    out.putalpha(img.getchannel("A").point(lambda a: int(a * max(0.0, alpha))))
    return out


def comp(canvas: Image.Image, im: Image.Image, x: float, y: float, alpha: float = 1.0, scale: float = 1.0,
         angle: float = 0.0, center: bool = False) -> None:
    """Alpha-composite `im` onto `canvas` with transform; (x, y) is top-left, or the centre if center=True."""
    if alpha <= 0.003 or scale <= 0.01:
        return
    if abs(scale - 1) > 0.003:
        im = im.resize((max(1, int(im.width * scale)), max(1, int(im.height * scale))), Image.Resampling.BILINEAR)
    if abs(angle) > 0.05:
        im = im.rotate(angle, resample=Image.Resampling.BICUBIC, expand=True)
    im = _with_alpha(im, alpha)
    if center:
        x, y = x - im.width / 2, y - im.height / 2
    x, y = int(round(x)), int(round(y))
    sx, sy = max(0, -x), max(0, -y)
    dx, dy = max(0, x), max(0, y)
    w, h = min(im.width - sx, canvas.width - dx), min(im.height - sy, canvas.height - dy)
    if w <= 0 or h <= 0:
        return
    if sx or sy or w != im.width or h != im.height:
        im = im.crop((sx, sy, sx + w, sy + h))
    canvas.alpha_composite(im, (dx, dy))


@dataclass(frozen=True)
class WordStyle:
    kind: str
    size: int
    fill: RGBA
    stroke: RGBA | None = None
    stroke_w: int = 0
    shadow: int = 0  # blur radius, 0 = none
    box: RGBA | None = None
    tracking: float = 0.0


@lru_cache(maxsize=2048)
def word_image(text: str, st: WordStyle) -> tuple[Image.Image, int, int, int]:
    """Render one word; returns (image, pad, advance_width, line_height)."""
    f = font(st.kind, st.size, text)
    asc, desc = f.getmetrics()
    if st.tracking:
        adv = int(sum(f.getlength(c) for c in text) + st.tracking * st.size * max(len(text) - 1, 0))
    else:
        adv = int(math.ceil(f.getlength(text)))
    sw = st.stroke_w
    box_pad_x = int(st.size * 0.22) if st.box else 0
    box_pad_y = int(st.size * 0.06) if st.box else 0
    content_w = adv + 2 * sw + 2 * box_pad_x
    line_h = asc + desc + 2 * sw + 2 * box_pad_y
    pad = max(st.shadow * 3, 2)
    img = Image.new("RGBA", (content_w + 2 * pad, line_h + 2 * pad), (0, 0, 0, 0))

    def draw_text(d: ImageDraw.ImageDraw, ox: int, oy: int, fill, stroke=None, swidth=0):
        if st.tracking:
            x = ox
            for c in text:
                d.text((x, oy), c, font=f, fill=fill, anchor="la", stroke_width=swidth, stroke_fill=stroke)
                x += f.getlength(c) + st.tracking * st.size
        else:
            d.text((ox, oy), text, font=f, fill=fill, anchor="la", stroke_width=swidth, stroke_fill=stroke)

    tx, ty = pad + sw + box_pad_x, pad + sw + box_pad_y
    if st.shadow:
        sh = Image.new("RGBA", img.size, (0, 0, 0, 0))
        sd = ImageDraw.Draw(sh)
        if st.box:
            sd.rounded_rectangle((pad, pad + st.shadow // 2, pad + content_w, pad + line_h + st.shadow // 2),
                                 radius=int(st.size * 0.18), fill=(0, 0, 0, 120))
        else:
            draw_text(sd, tx, ty + max(2, st.size // 28), (0, 0, 0, 170), (0, 0, 0, 170), sw)
        img = Image.alpha_composite(img, sh.filter(ImageFilter.GaussianBlur(st.shadow)))
    d = ImageDraw.Draw(img)
    if st.box:
        d.rounded_rectangle((pad, pad, pad + content_w, pad + line_h), radius=int(st.size * 0.18), fill=st.box)
    draw_text(d, tx, ty, st.fill, st.stroke, sw)
    return img, pad, content_w, line_h


@dataclass
class PlacedWord:
    text: str
    x: float  # content box top-left, relative to block origin
    y: float
    w: int
    h: int
    line: int
    emph: bool
    style: WordStyle


def layout(words: list[tuple[str, bool]], style: WordStyle, emph_style: WordStyle, max_w: float,
           align: str = "center", line_gap: float = 0.0, max_lines: int = 4) -> tuple[list[PlacedWord], float, float]:
    """Wrap words into lines; returns placed words, block width and block height."""
    space = font(style.kind, style.size).getlength(" ") * 0.9
    lines: list[list[tuple[str, bool, int, int, WordStyle]]] = [[]]
    width = 0.0
    for text, emph in words:
        st = emph_style if emph else style
        _, _, w, h = word_image(text, st)
        if lines[-1] and width + space + w > max_w:
            if len(lines) >= max_lines:
                break
            lines.append([])
            width = 0.0
        lines[-1].append((text, emph, w, h, st))
        width += (space if width else 0) + w
    line_h = max((h for line in lines for *_, h, _ in line), default=style.size)
    step = line_h * (0.92 + line_gap)
    placed: list[PlacedWord] = []
    widths = [sum(w for _, _, w, _, _ in line) + space * max(len(line) - 1, 0) for line in lines]
    block_w = max(widths, default=0)
    for li, line in enumerate(lines):
        x = (block_w - widths[li]) / 2 if align == "center" else 0
        for text, emph, w, h, st in line:
            placed.append(PlacedWord(text, x, li * step + (line_h - h) / 2, w, h, li, emph, st))
            x += w + space
    return placed, block_w, step * (len(lines) - 1) + line_h


def draw_word(canvas: Image.Image, pw: PlacedWord, ox: float, oy: float, alpha=1.0, scale=1.0, dx=0.0, dy=0.0,
              style: WordStyle | None = None) -> None:
    img, pad, _, _ = word_image(pw.text, style or pw.style)
    cx = ox + pw.x + pw.w / 2 + dx
    cy = oy + pw.y + pw.h / 2 + dy
    comp(canvas, img, cx, cy, alpha=alpha, scale=scale, center=True)


def _tokens(text: str, emphasis: str, upper: bool) -> list[tuple[str, bool]]:
    emph = {w.strip(".,!?:;\"'").lower() for w in (emphasis or "").split() if w.strip()}
    out = []
    for w in text.split():
        key = w.strip(".,!?:;\"'").lower()
        out.append((w.upper() if upper else w, bool(key) and key in emph))
    return out


# ----------------------------------------------------------------------------- scrims & icons
@lru_cache(maxsize=32)
def scrim(size: tuple[int, int], kind: str, strength: int = 150) -> Image.Image:
    """Soft darkening behind text so it reads on any footage."""
    import numpy as np

    W, H = size
    ys = np.linspace(0, 1, H)[:, None]
    xs = np.linspace(0, 1, W)[None, :]
    if kind == "bottom":
        a = np.clip((ys - 0.45) / 0.55, 0, 1) ** 1.4 * np.ones_like(xs)
    elif kind == "top":
        a = np.clip((0.4 - ys) / 0.4, 0, 1) ** 1.4 * np.ones_like(xs)
    elif kind == "left":
        a = np.clip((0.75 - xs) / 0.75, 0, 1) ** 1.2 * np.ones_like(ys)
    else:  # radial centre
        a = np.clip(1.15 - np.sqrt(((xs - 0.5) / 0.75) ** 2 + ((ys - 0.48) / 0.6) ** 2), 0, 1) ** 0.8
    alpha = (a * strength).astype("uint8")
    img = Image.new("RGBA", size, (0, 0, 0, 0))
    img.putalpha(Image.fromarray(alpha, "L"))
    return img


@lru_cache(maxsize=16)
def pin_icon(size: int, color: RGBA) -> Image.Image:
    s = size * 4
    img = Image.new("RGBA", (s, int(s * 1.3)), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    r = s / 2
    d.ellipse((0, 0, s, s), fill=color)
    d.polygon([(s * 0.14, r * 1.35), (s * 0.86, r * 1.35), (r, s * 1.3)], fill=color)
    d.ellipse((r - s * 0.17, r - s * 0.17, r + s * 0.17, r + s * 0.17), fill=(0, 0, 0, 0))
    return img.resize((size, int(size * 1.3)), Image.Resampling.LANCZOS)


@lru_cache(maxsize=8)
def cursor_icon(size: int) -> Image.Image:
    s = size * 4
    img = Image.new("RGBA", (s, int(s * 1.45)), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    pts = [(0, 0), (0, s * 1.15), (s * 0.3, s * 0.88), (s * 0.52, s * 1.38), (s * 0.7, s * 1.3), (s * 0.48, s * 0.82),
           (s * 0.88, s * 0.82)]
    d.polygon(pts, fill=(255, 255, 255, 255), outline=(0, 0, 0, 255), width=int(s * 0.06))
    return img.resize((size, int(size * 1.45)), Image.Resampling.LANCZOS)


@lru_cache(maxsize=64)
def rounded(size: tuple[int, int], radius: int, color: RGBA) -> Image.Image:
    img = Image.new("RGBA", (max(1, size[0]), max(1, size[1])), (0, 0, 0, 0))
    ImageDraw.Draw(img).rounded_rectangle((0, 0, size[0] - 1, size[1] - 1), radius=radius, fill=color)
    return img


# ----------------------------------------------------------------------------- layers
@dataclass
class Layer:
    """Base overlay layer living in [start, end] of a shot (seconds)."""

    start: float
    end: float
    theme: Theme
    size: tuple[int, int]

    intro: float = 0.6
    outro: float = 0.3

    @property
    def W(self) -> int:
        return self.size[0]

    @property
    def H(self) -> int:
        return self.size[1]

    @property
    def base(self) -> int:
        return min(self.size)

    @property
    def vertical(self) -> bool:
        return self.H > self.W

    def windows(self) -> list[tuple[float, float]]:
        """Time ranges where the layer is animating (sampled every frame)."""
        return [(self.start, self.start + self.intro), (self.end - self.outro, self.end)]

    def out_alpha(self, t: float) -> float:
        return clamp01((self.end - t) / self.outro) if self.outro > 0 else 1.0

    def draw(self, canvas: Image.Image, t: float) -> None:  # pragma: no cover - abstract
        raise NotImplementedError


def _display_style(th: Theme, size: int, emph: bool = False, kind: str | None = None) -> WordStyle:
    k = kind or th.display
    stroke_w = int(size * th.stroke_ratio) if th.stroke else 0
    if emph:
        if th.emph_mode == "box":
            return WordStyle(k, size, th.emph_text, None, 0, max(3, size // 14), th.emph)
        if th.emph_mode == "italic":
            return WordStyle("serif_italic" if k.startswith("serif") else k, size, th.emph, th.stroke, stroke_w,
                             max(3, size // 12))
        return WordStyle(k, size, th.emph, th.stroke, stroke_w, max(3, size // 12))
    return WordStyle(k, size, th.text, th.stroke, stroke_w, max(3, size // 12))


@dataclass
class HookTitle(Layer):
    """Cold-open title over the hero shot: words pop in one by one, subtitle + accent bar follow."""

    title: str = ""
    subtitle: str = ""
    emphasis: str = ""

    def __post_init__(self):
        th = self.theme
        size = int(self.base * (0.125 if self.vertical else 0.11))
        words = _tokens(self.title, self.emphasis, th.upper)
        self.placed, self.bw, self.bh = layout(words, _display_style(th, size), _display_style(th, size, True),
                                               self.W * (0.86 if self.vertical else 0.8), max_lines=4)
        self.sub_style = WordStyle(th.body, int(self.base * 0.04), (235, 235, 240, 255), None, 0, 4)
        self.stagger, self.word_in = 0.1, 0.34
        self.sub_at = 0.2 + len(self.placed) * self.stagger + 0.1
        self.intro = self.sub_at + 0.5

    def draw(self, canvas, t):
        lt = t - self.start
        if lt < 0 or t > self.end:
            return
        out = self.out_alpha(t)
        comp(canvas, scrim(self.size, "radial", 165), 0, 0, alpha=ease_out_cubic(lt / 0.45) * out)
        ox = (self.W - self.bw) / 2
        oy = self.H * (0.4 if self.vertical else 0.43) - self.bh / 2
        lift = (1 - out) * self.base * 0.03
        for i, pw in enumerate(self.placed):
            p = (lt - 0.2 - i * self.stagger) / self.word_in
            if p <= 0:
                continue
            draw_word(canvas, pw, ox, oy - lift, alpha=clamp01(p * 2.5) * out, scale=0.5 + 0.5 * ease_out_back(p))
        if self.subtitle:
            p = (lt - self.sub_at) / 0.4
            if p > 0:
                bar_w = int(self.base * 0.14 * ease_out_cubic(p))
                bar_h = max(4, int(self.base * 0.008))
                by = oy + self.bh + self.base * 0.035
                if bar_w > 2:
                    comp(canvas, rounded((bar_w, bar_h), bar_h // 2, self.theme.accent), (self.W - bar_w) / 2, by - lift,
                         alpha=out)
                sub_words = [(w, False) for w in self.subtitle.split()]
                placed, sw, _ = layout(sub_words, self.sub_style, self.sub_style, self.W * 0.8, max_lines=2)
                sy = by + bar_h + self.base * 0.03 + (1 - ease_out_cubic(p)) * self.base * 0.02
                for pw in placed:
                    draw_word(canvas, pw, (self.W - sw) / 2, sy - lift, alpha=clamp01(p) * out)


@dataclass
class ChapterTitle(Layer):
    """Section opener: accent kicker pill slides in, the title rises line by line."""

    kicker: str = ""
    title: str = ""
    emphasis: str = ""
    intro: float = 1.0

    def __post_init__(self):
        th = self.theme
        size = int(self.base * (0.09 if self.vertical else 0.085))
        self.align = "center" if self.vertical else "left"
        words = _tokens(self.title, self.emphasis, th.upper)
        self.placed, self.bw, self.bh = layout(words, _display_style(th, size), _display_style(th, size, True),
                                               self.W * (0.84 if self.vertical else 0.6), align=self.align, max_lines=3)
        ks = int(self.base * (0.038 if self.vertical else 0.03))
        self.kicker_style = WordStyle(th.body if th.name != "pop" else "anton", ks, (255, 255, 255, 255),
                                      tracking=0.12 if th.name != "pop" else 0.04)
        self.lines = max((pw.line for pw in self.placed), default=0) + 1

    def draw(self, canvas, t):
        lt = t - self.start
        if lt < 0 or t > self.end:
            return
        out = self.out_alpha(t)
        comp(canvas, scrim(self.size, "radial" if self.vertical else "left", 175), 0, 0,
             alpha=ease_out_cubic(lt / 0.4) * out)
        x0 = (self.W - self.bw) / 2 if self.vertical else self.W * 0.07
        y0 = self.H * (0.42 if self.vertical else 0.5) - self.bh / 2
        if self.kicker:
            p = ease_out_cubic((lt - 0.05) / 0.35)
            img, pad, kw, kh = word_image(self.kicker.upper(), self.kicker_style)
            pill = rounded((kw + int(kh * 0.9), int(kh * 1.25)), int(kh * 0.62), self.theme.accent)
            px = (self.W - pill.width) / 2 if self.vertical else x0
            py = y0 - pill.height - self.base * 0.03
            dx = (1 - p) * -self.base * 0.08
            comp(canvas, pill, px + dx, py, alpha=p * out)
            comp(canvas, img, px + dx + pill.width / 2, py + pill.height / 2, alpha=p * out, center=True)
        for pw in self.placed:
            p = (lt - 0.22 - pw.line * 0.09) / 0.5
            if p <= 0:
                continue
            e = ease_out_cubic(p)
            draw_word(canvas, pw, x0, y0, alpha=clamp01(p * 1.6) * out, dy=(1 - e) * pw.h * 0.6)
        p = (lt - 0.45 - self.lines * 0.09) / 0.45
        if p > 0:
            bar_h = max(4, int(self.base * 0.007))
            bar_w = int(self.base * 0.12 * ease_out_cubic(p))
            bx = (self.W - bar_w) / 2 if self.vertical else x0
            if bar_w > 2:
                comp(canvas, rounded((bar_w, bar_h), bar_h // 2, self.theme.accent), bx, y0 + self.bh + self.base * 0.03,
                     alpha=out)


@dataclass
class LowerThird(Layer):
    """Name / place strip in the lower-left corner with an accent edge."""

    title: str = ""
    sub: str = ""
    intro: float = 0.85

    def __post_init__(self):
        th = self.theme
        self.t_style = WordStyle(th.caption if th.name != "cinematic" else "serif", int(self.base * 0.046),
                                 (255, 255, 255, 255))
        self.s_style = WordStyle(th.body, int(self.base * 0.026), (210, 210, 220, 255), tracking=0.08)
        self.tw = word_image(self.title, self.t_style)[2]
        self.sw = word_image(self.sub.upper(), self.s_style)[2] if self.sub else 0
        self.th_ = word_image(self.title, self.t_style)[3]
        self.sh_ = word_image(self.sub.upper(), self.s_style)[3] if self.sub else 0

    def draw(self, canvas, t):
        lt = t - self.start
        if lt < 0 or t > self.end:
            return
        out = self.out_alpha(t)
        pad = int(self.base * 0.028)
        edge = max(5, int(self.base * 0.009))
        pw_full = max(self.tw, self.sw) + pad * 2 + edge
        ph = self.th_ + (self.sh_ + pad * 0.2 if self.sub else 0) + pad * 1.2
        x = self.W * (0.07 if self.vertical else 0.055)
        y = self.H * (0.7 if self.vertical else 0.8) - ph / 2
        p = ease_out_cubic(lt / 0.35)
        w = int(pw_full * p)
        if w > edge:
            comp(canvas, rounded((w, int(ph)), int(self.base * 0.012), self.theme.panel), x, y, alpha=out)
            comp(canvas, rounded((edge, int(ph)), edge // 2, self.theme.accent), x, y, alpha=out)
        pt = ease_out_cubic((lt - 0.2) / 0.35)
        if pt > 0:
            img = word_image(self.title, self.t_style)[0]
            comp(canvas, img, x + edge + pad + self.tw / 2 - (1 - pt) * self.base * 0.04, y + pad * 0.6 + self.th_ / 2,
                 alpha=pt * out, center=True)
        if self.sub:
            ps = ease_out_cubic((lt - 0.38) / 0.35)
            if ps > 0:
                img = word_image(self.sub.upper(), self.s_style)[0]
                comp(canvas, img, x + edge + pad + self.sw / 2, y + pad * 0.6 + self.th_ + self.sh_ / 2,
                     alpha=ps * out, center=True)


@dataclass
class LocationTag(Layer):
    """📍 Place · Region pill in the top corner."""

    place: str = ""
    region: str = ""
    intro: float = 0.75

    def __post_init__(self):
        th = self.theme
        self.p_style = WordStyle(th.caption if th.name != "cinematic" else "serif", int(self.base * 0.036),
                                 (255, 255, 255, 255))
        self.r_style = WordStyle(th.body, int(self.base * 0.022), (215, 215, 225, 255), tracking=0.1)

    def draw(self, canvas, t):
        lt = t - self.start
        if lt < 0 or t > self.end:
            return
        out = self.out_alpha(t)
        pimg, _, pw, ph = word_image(self.place, self.p_style)
        rimg, _, rw, rh = word_image(self.region.upper(), self.r_style) if self.region else (None, 0, 0, 0)
        icon_h = int(ph * 0.95)
        icon = pin_icon(int(icon_h / 1.3), self.theme.accent)
        pad = int(self.base * 0.02)
        text_w = max(pw, rw)
        text_h = ph + (rh * 0.9 if self.region else 0)
        box_w = int(pad * 2.2 + icon.width + pad * 0.7 + text_w)
        box_h = int(text_h + pad * 1.2)
        x = (self.W - box_w) / 2 if self.vertical else self.W * 0.05
        y = self.H * (0.1 if self.vertical else 0.075)
        p = ease_out_cubic(lt / 0.35)
        comp(canvas, rounded((box_w, box_h), box_h // 2, (0, 0, 0, 150)), x, y - (1 - p) * self.base * 0.03,
             alpha=p * out)
        pi = (lt - 0.1) / 0.4
        if pi > 0:
            comp(canvas, icon, x + pad * 1.1 + icon.width / 2, y + box_h / 2, alpha=clamp01(pi * 2) * out,
                 scale=ease_out_back(pi), center=True)
        pt = ease_out_cubic((lt - 0.22) / 0.4)
        tx = x + pad * 1.1 + icon.width + pad * 0.7
        if pt > 0:
            ty = y + (box_h - text_h) / 2
            comp(canvas, pimg, tx + pw / 2 - (1 - pt) * self.base * 0.03, ty + ph / 2, alpha=pt * out, center=True)
            if rimg is not None:
                comp(canvas, rimg, tx + rw / 2, ty + ph * 0.92 + rh / 2, alpha=pt * out, center=True)


@dataclass
class Caption(Layer):
    """Creator-style caption: quick word pops, one keyword highlighted."""

    text: str = ""
    emphasis: str = ""
    position: str = "bottom"
    outro: float = 0.25

    def __post_init__(self):
        th = self.theme
        size = int(self.base * (0.074 if self.vertical else 0.066))
        kind = th.caption
        words = _tokens(self.text, self.emphasis, th.upper)
        self.placed, self.bw, self.bh = layout(words, _display_style(th, size, kind=kind),
                                               _display_style(th, size, True, kind=kind),
                                               self.W * (0.84 if self.vertical else 0.72), max_lines=3)
        self.stagger, self.word_in = 0.055, 0.26
        self.intro = 0.1 + len(self.placed) * self.stagger + self.word_in
        self.lines = max((pw.line for pw in self.placed), default=0) + 1

    def origin(self) -> tuple[float, float]:
        if self.position == "top":
            y = self.H * (0.15 if self.vertical else 0.1)
        elif self.position == "center":
            y = (self.H - self.bh) / 2
        else:
            y = self.H * (0.74 if self.vertical else 0.87) - self.bh
        return (self.W - self.bw) / 2, y

    def draw(self, canvas, t):
        lt = t - self.start
        if lt < 0 or t > self.end:
            return
        out = self.out_alpha(t)
        ox, oy = self.origin()
        kind = "top" if self.position == "top" else "bottom" if self.position == "bottom" else "radial"
        comp(canvas, scrim(self.size, kind, 120), 0, 0, alpha=ease_out_cubic(lt / 0.3) * out)
        th = self.theme
        if th.name == "pop":  # sticker strip behind every line
            for li in range(self.lines):
                ws = [pw for pw in self.placed if pw.line == li]
                if not ws:
                    continue
                p = ease_out_back((lt - 0.02 - li * 0.06) / 0.3)
                if p <= 0:
                    continue
                pad = ws[0].h * 0.18
                x0, x1 = ws[0].x - pad, ws[-1].x + ws[-1].w + pad
                h = max(w.h for w in ws) + pad * 0.6
                box = rounded((int(x1 - x0), int(h)), int(h * 0.18), th.stickers[li % len(th.stickers)])
                comp(canvas, box, ox + (x0 + x1) / 2, oy + ws[0].y + ws[0].h / 2, alpha=out, scale=p,
                     angle=(-2.0 if li % 2 == 0 else 1.5), center=True)
        for i, pw in enumerate(self.placed):
            p = (lt - 0.1 - i * self.stagger) / self.word_in
            if p <= 0:
                continue
            draw_word(canvas, pw, ox, oy, alpha=clamp01(p * 3) * out, scale=0.78 + 0.22 * ease_out_back(p))


@dataclass
class Callout(Layer):
    """Tilted sticker ("MUST TRY!") that pops in with overshoot."""

    text: str = ""
    intro: float = 0.5

    def __post_init__(self):
        th = self.theme
        self.style = WordStyle("anton" if th.name in ("pop", "bold") else th.caption, int(self.base * 0.05),
                               (255, 255, 255, 255) if th.name != "bold" else (0, 0, 0, 255))
        self.color = th.stickers[0] if th.name in ("pop", "bold") else th.accent

    def draw(self, canvas, t):
        lt = t - self.start
        if lt < 0 or t > self.end:
            return
        out = self.out_alpha(t)
        img, _, w, h = word_image(self.text.upper(), self.style)
        pad = int(h * 0.35)
        sticker = rounded((w + pad * 2, h + pad), int(h * 0.25), self.color)
        sticker = sticker.copy()
        sticker.alpha_composite(img, ((sticker.width - img.width) // 2, (sticker.height - img.height) // 2))
        x = self.W * (0.72 if not self.vertical else 0.68)
        y = self.H * (0.2 if not self.vertical else 0.27)
        p = lt / 0.45
        comp(canvas, sticker, x, y, alpha=clamp01(p * 3) * out, scale=0.2 + 0.8 * ease_out_back(p, 2.6), angle=6,
             center=True)


@dataclass
class EndScreen(Layer):
    """Outro line + animated SUBSCRIBE button with a cursor click."""

    text: str = ""
    sub: str = ""
    intro: float = 2.3
    outro: float = 0.0

    def __post_init__(self):
        th = self.theme
        size = int(self.base * (0.09 if self.vertical else 0.085))
        words = _tokens(self.text, "", th.upper)
        self.placed, self.bw, self.bh = layout(words, _display_style(th, size), _display_style(th, size, True),
                                               self.W * 0.82, max_lines=3)
        self.btn_style = WordStyle("bold", int(self.base * 0.04), (255, 255, 255, 255), tracking=0.04)
        self.sub_style = WordStyle(th.body, int(self.base * (0.04 if self.vertical else 0.034)), (225, 225, 232, 255))

    def draw(self, canvas, t):
        lt = t - self.start
        if lt < 0 or t > self.end:
            return
        ox = (self.W - self.bw) / 2
        oy = self.H * 0.38 - self.bh / 2
        for i, pw in enumerate(self.placed):
            p = (lt - 0.1 - i * 0.08) / 0.32
            if p > 0:
                draw_word(canvas, pw, ox, oy, alpha=clamp01(p * 2.5), scale=0.55 + 0.45 * ease_out_back(p))
        if self.sub:
            p = ease_out_cubic((lt - 0.45) / 0.4)
            if p > 0:
                img, _, w, h = word_image(self.sub, self.sub_style)
                comp(canvas, img, self.W / 2, oy + self.bh + self.base * 0.05 + h / 2, alpha=p, center=True)
        # Subscribe button.
        clicked = lt >= 1.75
        label = "SUBSCRIBED ✓" if clicked else "SUBSCRIBE"
        img, _, w, h = word_image(label.replace(" ✓", ""), self.btn_style)
        bw, bh = w + int(h * 1.6), int(h * 1.55)
        color = (64, 64, 70, 255) if clicked else (255, 0, 0, 255)
        btn = rounded((bw, bh), bh // 2, color).copy()
        btn.alpha_composite(img, ((bw - img.width) // 2, (bh - img.height) // 2))
        bx, by = self.W / 2, self.H * (0.66 if self.vertical else 0.7)
        p = (lt - 0.75) / 0.4
        if p > 0:
            press = 1 - 0.08 * math.sin(math.pi * clamp01((lt - 1.68) / 0.18)) if 1.68 <= lt <= 1.86 else 1
            comp(canvas, btn, bx, by + (1 - ease_out_cubic(p)) * self.base * 0.05, alpha=clamp01(p * 2),
                 scale=ease_out_back(p) * press, center=True)
        # Cursor flies in and clicks.
        pc = (lt - 1.05) / 0.6
        if pc > 0 and lt < 2.6:
            e = ease_in_out(pc)
            cur = cursor_icon(int(self.base * 0.045))
            sx, sy = self.W * 0.8, self.H * 0.95
            tx, ty = bx + bw * 0.18, by + bh * 0.1
            fade = clamp01((2.6 - lt) / 0.3)
            comp(canvas, cur, sx + (tx - sx) * e, sy + (ty - sy) * e, alpha=fade,
                 scale=0.9 if 1.68 <= lt <= 1.86 else 1.0)


@dataclass
class Subtitles(Layer):
    """Burned-in subtitles synced to narration, current word highlighted (karaoke)."""

    words: list[tuple[str, float, float]] = field(default_factory=list)  # (word, t0, t1) on this track's clock
    intro: float = 0.0
    outro: float = 0.0

    def __post_init__(self):
        th = self.theme
        size = int(self.base * (0.07 if self.vertical else 0.05))
        kind = th.caption
        sw = int(size * max(th.stroke_ratio, 0.07)) if th.name != "cinematic" else 0
        stroke = (0, 0, 0, 255) if sw else None
        self.style = WordStyle(kind, size, th.text, stroke, sw, max(3, size // 10))
        if th.sub_mode == "box":
            self.hi = WordStyle(kind, size, (10, 10, 14, 255), None, 0, max(3, size // 12), th.sub_emph)
        else:
            self.hi = WordStyle(kind, size, th.sub_emph, stroke, sw, max(3, size // 10))
        self.upper = th.upper
        self.groups = self._group(3 if self.vertical else 6, 16 if self.vertical else 34)

    def _group(self, max_words: int, max_chars: int) -> list[list[tuple[str, float, float]]]:
        groups, cur = [], []
        for w in self.words:
            text = w[0]
            if cur and (len(cur) >= max_words or sum(len(x[0]) + 1 for x in cur) + len(text) > max_chars
                        or (w[1] - cur[-1][2]) > 0.6):
                groups.append(cur)
                cur = []
            cur.append(w)
            if re.search(r"[.!?;:,]$", text) and len(cur) >= 2:
                groups.append(cur)
                cur = []
        if cur:
            groups.append(cur)
        return groups

    def windows(self):
        wins = []
        for g in self.groups:
            wins.append((g[0][1], g[0][1] + 0.12))
        return wins

    def keytimes(self) -> list[float]:
        times = []
        for gi, g in enumerate(self.groups):
            times += [w[1] for w in g]
            end = self._group_end(gi)
            times.append(end)
        return times

    def _group_end(self, gi: int) -> float:
        g = self.groups[gi]
        nxt = self.groups[gi + 1][0][1] if gi + 1 < len(self.groups) else g[-1][2] + 0.5
        return min(nxt, g[-1][2] + 0.6, self.end)

    def draw(self, canvas, t):
        for gi, g in enumerate(self.groups):
            if not (g[0][1] <= t < self._group_end(gi)):
                continue
            tokens = [(w[0].upper() if self.upper else w[0], False) for w in g]
            placed, bw, bh = layout(tokens, self.style, self.style, self.W * (0.86 if self.vertical else 0.8),
                                    max_lines=2)
            ox = (self.W - bw) / 2
            oy = self.H * (0.66 if self.vertical else 0.9) - bh / 2 - (0 if self.vertical else bh / 2)
            pop = 0.9 + 0.1 * ease_out_back((t - g[0][1]) / 0.12)
            active = max((i for i, w in enumerate(g) if w[1] <= t), default=-1)
            for i, pw in enumerate(placed):
                st = self.hi if i == active else None
                draw_word(canvas, pw, ox, oy, scale=pop * (1.06 if i == active else 1.0), style=st)
            return


# ----------------------------------------------------------------------------- track rendering
def render_track(layers: list[Layer], duration: float, size: tuple[int, int], fps: int, out_dir: Path,
                 prefix: str) -> Path | None:
    """Render layers into PNG keyframes + an ffconcat list. Returns the list path (None if empty)."""
    layers = [layer for layer in layers if layer.end > layer.start]
    if not layers:
        return None
    out_dir.mkdir(parents=True, exist_ok=True)
    total = max(1, int(round(duration * fps)))
    marks: set[int] = {0}
    for layer in layers:
        marks.add(int(round(layer.start * fps)))
        marks.add(int(round(layer.end * fps)))
        for a, b in layer.windows():
            for k in range(int(math.floor(a * fps)), int(math.ceil(b * fps)) + 1):
                marks.add(k)
        if isinstance(layer, Subtitles):
            marks.update(int(round(x * fps)) for x in layer.keytimes())
    frames = sorted(k for k in marks if 0 <= k < total)
    entries: list[tuple[Path, int]] = []  # (file, frame count)
    last_hash = None
    for idx, k in enumerate(frames):
        nxt = frames[idx + 1] if idx + 1 < len(frames) else total
        t = k / fps
        canvas = Image.new("RGBA", size, (0, 0, 0, 0))
        for layer in layers:
            layer.draw(canvas, t)
        digest = hashlib.md5(canvas.tobytes()).digest()
        if digest == last_hash and entries:
            entries[-1] = (entries[-1][0], entries[-1][1] + (nxt - k))
            continue
        path = out_dir / f"{prefix}_{idx:04d}.png"
        canvas.save(path, compress_level=1)
        entries.append((path, nxt - k))
        last_hash = digest
    listing = out_dir / f"{prefix}.ffconcat"
    lines = ["ffconcat version 1.0"]
    for path, count in entries:
        lines += [f"file '{path.resolve()}'", f"duration {count / fps:.6f}"]
    lines.append(f"file '{entries[-1][0].resolve()}'")  # concat demuxer quirk: repeat the last file
    listing.write_text("\n".join(lines) + "\n")
    return listing
