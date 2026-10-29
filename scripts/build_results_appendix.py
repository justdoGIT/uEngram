#!/usr/bin/env python3
"""Regenerate the standalone results ledger from files under results/.

Usage: .venv/bin/python scripts/build_results_appendix.py [--print]
Needs the "docs" extra (markdown, pygments, pillow): `uv sync --extra docs`.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))

from results_appendix.render import collect, render  # noqa: E402

LEDGER = REPO / "docs/results-ledger.md"
FIGURE = ("![Figure A1. Outcome of every result table, by group. Green: positive. Amber: mixed. "
          "Grey: null. Red: negative.](research-paper/figures/results-ledger.svg)")


def main() -> None:
    groups = collect(REPO)
    if "--print" in sys.argv:
        print(render(groups, FIGURE))
        return
    from results_appendix.ledger_figure import write_ledger
    write_ledger(groups, REPO / "docs/research-paper/figures/results-ledger.svg")
    LEDGER.write_text("# Supplementary Results Ledger\n\n" + render(groups, FIGURE), encoding="utf-8")
    print(f"ledger written: {sum(len(t) for _, _, t in groups)} tables")

if __name__ == "__main__":
    main()
