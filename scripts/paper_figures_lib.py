"""SVG helpers for the research-paper figures.

Text is measured with the same Nimbus Sans font files the PDF uses, so
every box is sized from its content and no label can overflow its frame.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from xml.sax.saxutils import escape

from PIL import ImageFont

FONT_FAMILY = "'Nimbus Sans', Helvetica, Arial, sans-serif"
_FONT_FILES = {
    False: "/usr/share/fonts/OTF/NimbusSans-Regular.otf",
    True: "/usr/share/fonts/OTF/NimbusSans-Bold.otf",
}
INK = "#0f172a"
MUTED = "#475569"
EDGE = "#64748b"
GRID = "#e2e8f0"

# name -> (fill, stroke, title colour)
TONES = {
    "slate": ("#f8fafc", "#cbd5e1", "#0f172a"),
    "blue": ("#eff6ff", "#93c5fd", "#1e3a8a"),
    "green": ("#f0fdf4", "#86efac", "#14532d"),
    "amber": ("#fffbeb", "#fcd34d", "#78350f"),
    "red": ("#fef2f2", "#fca5a5", "#7f1d1d"),
    "violet": ("#f5f3ff", "#c4b5fd", "#4c1d95"),
    "white": ("#ffffff", "#cbd5e1", "#0f172a"),
}
SERIES = ("#94a3b8", "#60a5fa", "#1d4ed8", "#15803d")

TITLE_SIZE = 12.5
BODY_SIZE = 10.5
LABEL_SIZE = 9.5
LINE_GAP = 1.35
PAD_X = 12
PAD_Y = 9


@lru_cache(maxsize=None)
def _font(size_tenths: int, bold: bool) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(_FONT_FILES[bold], size_tenths)


def text_width(text: str, size: float, bold: bool = False) -> float:
    """Advance width in SVG px of `text` at `size` px."""
    return _font(round(size * 10), bold).getlength(text) / 10.0


def box_size(title: str, lines: tuple[str, ...] = ()) -> tuple[float, float]:
    widths = [text_width(title, TITLE_SIZE, True)] + [text_width(s, BODY_SIZE) for s in lines]
    w = max(widths) + 2 * PAD_X
    h = 2 * PAD_Y + TITLE_SIZE * LINE_GAP + len(lines) * BODY_SIZE * LINE_GAP
    return w, h


def label_gap(*labels: str, floor: float = 30) -> float:
    """Column gap wide enough for any edge label placed inside it."""
    return max([floor] + [text_width(s, LABEL_SIZE) + 30 for s in labels])


@dataclass
class Box:
    x: float
    y: float
    w: float
    h: float

    @property
    def cx(self) -> float:
        return self.x + self.w / 2

    @property
    def cy(self) -> float:
        return self.y + self.h / 2

    @property
    def right(self) -> float:
        return self.x + self.w

    @property
    def bottom(self) -> float:
        return self.y + self.h

    def left_mid(self) -> tuple[float, float]:
        return (self.x, self.cy)

    def right_mid(self) -> tuple[float, float]:
        return (self.right, self.cy)

    def top_mid(self) -> tuple[float, float]:
        return (self.cx, self.y)

    def bottom_mid(self) -> tuple[float, float]:
        return (self.cx, self.bottom)


class Canvas:
    def __init__(self, width: float, height: float, label: str):
        self.width = width
        self.height = height
        self.label = label
        self.parts: list[str] = []

    # -- primitives ------------------------------------------------------
    def text(self, x, y, s, size=BODY_SIZE, bold=False, color=MUTED, anchor="start",
             italic=False):
        weight = ' font-weight="700"' if bold else ""
        style = ' font-style="italic"' if italic else ""
        self.parts.append(
            f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}"{weight}{style} '
            f'fill="{color}" text-anchor="{anchor}">{escape(s)}</text>'
        )

    def rect(self, x, y, w, h, fill, stroke="none", rx=6, width=1.0, dash=None):
        dashed = f' stroke-dasharray="{dash}"' if dash else ""
        self.parts.append(
            f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="{rx}" '
            f'fill="{fill}" stroke="{stroke}" stroke-width="{width}"{dashed}/>'
        )

    def line(self, x1, y1, x2, y2, color=GRID, width=1.0, dash=None):
        dashed = f' stroke-dasharray="{dash}"' if dash else ""
        self.parts.append(
            f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
            f'stroke="{color}" stroke-width="{width}"{dashed}/>'
        )

    def pill(self, x, y, s, size=LABEL_SIZE, color=MUTED, fill="#ffffff"):
        """Centred label on a white backing so it stays legible over edges."""
        w = text_width(s, size) + 8
        self.rect(x - w / 2, y - size * 0.9, w, size * 1.35, fill, rx=3)
        self.text(x, y + size * 0.12, s, size=size, color=color, anchor="middle")

    # -- composites ------------------------------------------------------
    def box(self, x, y, title, lines=(), tone="slate", min_w=0, min_h=0, dash=None) -> Box:
        fill, stroke, title_color = TONES[tone]
        w, h = box_size(title, tuple(lines))
        w, h = max(w, min_w), max(h, min_h)
        self.rect(x, y, w, h, fill, stroke, rx=7, dash=dash)
        cx = x + w / 2
        block = TITLE_SIZE * LINE_GAP + len(lines) * BODY_SIZE * LINE_GAP
        ty = y + (h - block) / 2 + TITLE_SIZE * 0.95
        self.text(cx, ty, title, TITLE_SIZE, True, title_color, "middle")
        for i, s in enumerate(lines):
            self.text(cx, ty + (i + 1) * BODY_SIZE * LINE_GAP + 1, s, BODY_SIZE, False,
                      MUTED, "middle")
        return Box(x, y, w, h)

    def grid(self, spec, place, tones, x0, y0, col_gaps, row_gaps, dashed=()):
        """Draw boxes on a grid where each column shares a width and each row a
        height, so edges between neighbours stay straight.

        spec: key -> (title, lines); place: key -> (col, row).
        col_gaps / row_gaps: gap after each column / row (length n - 1).
        Returns (boxes, right edge, bottom edge)."""
        n_col = 1 + max(c for c, _ in place.values())
        n_row = 1 + max(r for _, r in place.values())
        col_w = [0.0] * n_col
        row_h = [0.0] * n_row
        for key, (col, row) in place.items():
            w, h = box_size(*spec[key])
            col_w[col] = max(col_w[col], w)
            row_h[row] = max(row_h[row], h)
        xs = [x0]
        for i in range(1, n_col):
            xs.append(xs[-1] + col_w[i - 1] + col_gaps[i - 1])
        ys = [y0]
        for i in range(1, n_row):
            ys.append(ys[-1] + row_h[i - 1] + row_gaps[i - 1])
        boxes = {}
        for key, (col, row) in place.items():
            title, lines = spec[key]
            boxes[key] = self.box(xs[col], ys[row], title, lines, tones[key], min_w=col_w[col],
                                  min_h=row_h[row], dash="4 3" if key in dashed else None)
        return boxes, xs[-1] + col_w[-1], ys[-1] + row_h[-1]

    def arrow(self, points, color=EDGE, dash=None, label=None, label_at=None):
        path = " L ".join(f"{px:.1f} {py:.1f}" for px, py in points)
        dashed = f' stroke-dasharray="{dash}"' if dash else ""
        self.parts.append(
            f'<path d="M {path}" fill="none" stroke="{color}" stroke-width="1.3"'
            f'{dashed} marker-end="url(#{self.label}-arrow)"/>'
        )
        if label:
            if label_at is None:
                (x1, y1), (x2, y2) = points[0], points[1]
                label_at = ((x1 + x2) / 2, (y1 + y2) / 2)
            self.pill(label_at[0], label_at[1], label)

    def heading(self, x, y, s, size=11, color=MUTED):
        self.text(x, y, s.upper(), size, True, color)

    # -- output ----------------------------------------------------------
    def svg(self) -> str:
        head = (
            f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {self.width:.0f} '
            f'{self.height:.0f}" role="img" aria-label="{escape(self.label)}" '
            f'font-family="{FONT_FAMILY}">\n'
            f'<defs><marker id="{self.label}-arrow" viewBox="0 0 10 10" refX="9" refY="5" '
            'markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
            f'<path d="M 0 1.5 L 9 5 L 0 8.5 z" fill="{EDGE}"/></marker></defs>\n'
            f'<rect width="{self.width:.0f}" height="{self.height:.0f}" fill="#ffffff"/>\n'
        )
        return head + "\n".join(self.parts) + "\n</svg>\n"


def hbar_panel(c: Canvas, x, y, w, title, rows, scale_max, unit_fmt, label_rows=None):
    """Horizontal bar panel. rows: (label, value, value_text, sub_text, colour).
    label_rows sizes the label column, so stacked panels can share bar axes.
    Returns the y coordinate below the axis labels."""
    c.heading(x, y, title)
    sizing = label_rows or rows
    label_w = max(max(text_width(r[0], BODY_SIZE, True), text_width(r[3], LABEL_SIZE))
                  for r in sizing) + 16
    value_w = max(text_width(r[2], BODY_SIZE, True) for r in sizing) + 12
    bar_x = x + label_w
    bar_w = w - label_w - value_w
    row_h = 42
    top = y + 18
    for frac in (0, 0.25, 0.5, 0.75, 1.0):
        gx = bar_x + bar_w * frac
        c.line(gx, top, gx, top + row_h * len(rows), GRID)
        c.text(gx, top + row_h * len(rows) + 13, unit_fmt(scale_max * frac), 9, color=MUTED,
               anchor="middle")
    for i, (label, value, value_text, sub, color) in enumerate(rows):
        ry = top + i * row_h + 8
        c.text(x, ry + 11, label, BODY_SIZE, True, INK)
        if sub:
            c.text(x, ry + 25, sub, LABEL_SIZE, False, MUTED)
        bw = max(bar_w * value / scale_max, 2)
        c.rect(bar_x, ry, bw, 16, color, rx=3)
        c.text(bar_x + bw + 6, ry + 12, value_text, BODY_SIZE, True, INK)
    return top + row_h * len(rows) + 18
