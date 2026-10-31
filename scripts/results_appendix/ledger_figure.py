"""Figure A1: one stacked bar per result group, coloured by outcome."""
from __future__ import annotations

from pathlib import Path

from paper_figures_lib import BODY_SIZE, INK, LABEL_SIZE, MUTED, Canvas, text_width

from .render import LABELS, OUTCOMES, tally

COLORS = {"positive": "#15803d", "mixed": "#b45309", "null": "#64748b", "negative": "#b91c1c"}
UNIT = 26.0
BAR_H = 22.0
ROW_H = 40.0


def write_ledger(groups, path: Path) -> None:
    counts = tally(groups)
    total = {o: sum(c[o] for c in counts.values()) for o in OUTCOMES}
    rows = list(counts.items()) + [("All groups", total)]
    label_w = max(text_width(name, BODY_SIZE, True) for name, _ in rows) + 18
    bar_max = max(sum(c.values()) for _, c in rows[:-1])
    right = max(text_width(f"{sum(c.values())} tables", LABEL_SIZE) for _, c in rows) + 12
    width = 20 + label_w + max(bar_max, sum(total.values()) / 3) * UNIT + right + 20
    # the total row is drawn at one third scale so it does not dwarf the groups
    top = 54.0
    height = top + ROW_H * len(rows) + 20
    c = Canvas(width, height, "results-ledger")
    x = 20.0
    for outcome in OUTCOMES:
        c.rect(x, 16, 12, 12, COLORS[outcome], rx=2)
        c.text(x + 18, 26.5, LABELS[outcome], BODY_SIZE, True, INK)
        x += 18 + text_width(LABELS[outcome], BODY_SIZE, True) + 22
    for i, (name, count) in enumerate(rows):
        y = top + i * ROW_H
        last = i == len(rows) - 1
        if last:
            c.line(20, y - 8, width - 20, y - 8)
        c.text(20, y + 15, name, BODY_SIZE, True, INK)
        unit = UNIT / 3 if last else UNIT
        x = 20 + label_w
        for outcome in OUTCOMES:
            n = count[outcome]
            if not n:
                continue
            w = n * unit
            c.rect(x, y, w, BAR_H, COLORS[outcome], rx=3)
            text = str(n)
            if text_width(text, BODY_SIZE, True) + 6 <= w:
                c.text(x + w / 2, y + 15, text, BODY_SIZE, True, "#ffffff", "middle")
            x += w
        c.text(x + 8, y + 15, f"{sum(count.values())} tables", LABEL_SIZE, False, MUTED)
    path.write_text(c.svg(), encoding="utf-8")
