#!/usr/bin/env python3
"""Regenerate Appendix A of the research paper from the files under results/.

Usage: python3 scripts/build_results_appendix.py [--print]
Replaces the text between the RESULTS-APPENDIX markers in the paper Markdown.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))

from results_appendix.render import collect, render  # noqa: E402

PAPER = REPO / "docs/research-paper/From_Sensor_Streams_to_Bounded_Machine_Memory.md"
BEGIN, END = "<!-- RESULTS-APPENDIX:BEGIN -->", "<!-- RESULTS-APPENDIX:END -->"
FIGURE = ("![Figure A1. Outcome of every result table, by group. Green: positive. Amber: mixed. "
          "Grey: null. Red: negative.](figures/results-ledger.svg)")


def main() -> None:
    groups = collect(REPO)
    if "--print" in sys.argv:
        print(render(groups, FIGURE))
        return
    from results_appendix.ledger_figure import write_ledger
    write_ledger(groups, REPO / "docs/research-paper/figures/results-ledger.svg")
    text = PAPER.read_text(encoding="utf-8")
    start, stop = text.find(BEGIN), text.find(END)
    if start < 0 or stop < start:
        raise SystemExit(f"paper is missing the {BEGIN} / {END} markers")
    body = render(groups, FIGURE)
    PAPER.write_text(text[: start + len(BEGIN)] + "\n\n" + body + "\n" + text[stop:], encoding="utf-8")
    print(f"appendix written: {sum(len(t) for _, _, t in groups)} tables")


if __name__ == "__main__":
    main()
