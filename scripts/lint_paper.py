#!/usr/bin/env python3
"""Lint standalone-paper content for repository leakage and structural consistency."""
from __future__ import annotations

import argparse
import re
from pathlib import Path

# The named rule list is intentionally easy to tune as manuscript policy evolves.
REPOSITORY_LEAKAGE_RULES = [
    ("repository paths", re.compile(r"(?<![\w.-])(?:src|bench|scripts|results|tests|docs)/", re.I)),
    ("repository file extensions", re.compile(r"\.(?:py|json|jsonl|csv|cff|toml)\b", re.I)),
    ("line-range citations", re.compile(r"(?<!\d):\d+(?:-\d+)?\b")),
    ("commit hashes", re.compile(r"\b(?:commit(?:ted)?\s+)?[0-9a-f]{7,40}\b", re.I)),
    ("internal labels", re.compile(r"\[INFERENCE\]|\bTODO\b|v2_|\bv8\b|\bTable A\d+\b", re.I)),
    ("repository terminology", re.compile(r"\b(?:repository|untracked)\b", re.I)),
    ("backticked Python identifiers", re.compile(r"`[A-Za-z][A-Za-z0-9]*_[A-Za-z0-9_]*`")),
]

CITATION = re.compile(r"\[(\d+(?:\s*(?:,|\u2013|-)\s*\d+)*)\]")
REFERENCE_ITEM = re.compile(r"^\s*(\d+)[.)]\s+", re.M)
SECTION = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
FIGURE = re.compile(r"!\[Figure\s+(\d+)\.", re.I)
TABLE = re.compile(r"^\s*\*\*Table\s+(\d+)\.\*\*", re.I | re.M)
ALGORITHM = re.compile(r"^```algorithm\s*\|\s*Algorithm\s+(\d+)\.", re.I | re.M)


def _sections(lines: list[str]) -> tuple[list[tuple[int, str]], list[tuple[int, str]]]:
    excluded: set[int] = set()
    references: list[tuple[int, str]] = []
    for index, line in enumerate(lines):
        heading = SECTION.match(line)
        if not heading or len(heading.group(1)) != 2:
            continue
        title = heading.group(2).strip().casefold()
        if title not in {"references", "data and code availability"}:
            continue
        end = len(lines)
        for later in range(index + 1, len(lines)):
            next_heading = SECTION.match(lines[later])
            if next_heading and len(next_heading.group(1)) <= 2:
                end = later
                break
        excluded.update(range(index, end))
        if title == "references":
            references = [(index + 1, "\n".join(lines[index + 1:end]))]
    active = [(i + 1, line) for i, line in enumerate(lines) if i not in excluded]
    return active, references


def _citation_numbers(text: str) -> set[int]:
    found: set[int] = set()
    for match in CITATION.finditer(text):
        chunks = re.split(r"\s*(,|\u2013|-)\s*", match.group(1))
        values = [int(value) for value in chunks[::2]]
        found.update(values)
        for index, separator in enumerate(chunks[1::2]):
            if separator in {"-", "\u2013"}:
                first, last = values[index], values[index + 1]
                found.update(range(min(first, last), max(first, last) + 1))
    return found


def lint_markdown(text: str) -> list[tuple[int, str]]:
    lines = text.splitlines()
    active, reference_sections = _sections(lines)
    findings: list[tuple[int, str]] = []
    for line_number, line in active:
        for name, pattern in REPOSITORY_LEAKAGE_RULES:
            if pattern.search(line):
                findings.append((line_number, f"repository-specific leakage: {name}"))

    refs_text = "\n".join(section for _, section in reference_sections)
    reference_items = [(reference_sections[0][0] + offset + 1, int(match.group(1)))
                       for offset, line in enumerate(refs_text.splitlines())
                       if (match := REFERENCE_ITEM.match(line))] if reference_sections else []
    reference_numbers = {number for _, number in reference_items}
    if reference_sections:
        ref_line = reference_sections[0][0]
        if len(reference_items) != len(reference_numbers):
            findings.append((ref_line, "reference numbers must be unique"))
        for expected, (line_number, actual) in enumerate(reference_items, 1):
            if actual != expected:
                findings.append((line_number, f"reference numbering must be consecutive (expected {expected}, found {actual})"))

    cited_numbers: set[int] = set()
    for line_number, line in active:
        line_citations = _citation_numbers(line)
        cited_numbers.update(line_citations)
        for number in sorted(line_citations - reference_numbers):
            findings.append((line_number, f"citation [{number}] has no matching reference"))
    for line_number, number in reference_items:
        if number not in cited_numbers:
            findings.append((line_number, f"reference {number} is never cited"))

    for label, pattern in (("Figure", FIGURE), ("Table", TABLE), ("Algorithm", ALGORITHM)):
        items = [(number, int(match.group(1)))
                 for number, line in active for match in pattern.finditer(line)]
        declaration_lines = {number for number, _ in items}
        for index, (line_number, actual) in enumerate(items, 1):
            if actual != index:
                findings.append((line_number, f"{label} numbering must be consecutive (expected {index}, found {actual})"))
            mention = re.compile(rf"\b{label}\s+{actual}\b", re.I)
            if not any(number not in declaration_lines and mention.search(line)
                       for number, line in active):
                findings.append((line_number, f"{label} {actual} is not referenced in text"))
    return sorted(findings)


DEFAULT_PAPER = (Path(__file__).resolve().parents[1] / "docs" / "research-paper" /
                 "From_Sensor_Streams_to_Bounded_Machine_Memory.md")
def main() -> int:
    parser = argparse.ArgumentParser(description="Check a research-paper Markdown file.")
    parser.add_argument("markdown", type=Path, nargs="?", default=DEFAULT_PAPER,
                        help="Markdown file to lint (defaults to the standalone paper)")
    args = parser.parse_args()
    try:
        text = args.markdown.read_text(encoding="utf-8")
    except OSError as error:
        print(f"1: cannot read Markdown file: {error}")
        return 1
    findings = lint_markdown(text)
    for line_number, message in findings:
        print(f"{line_number}: {message}")
    return 1 if findings else 0


if __name__ == "__main__":
    raise SystemExit(main())
