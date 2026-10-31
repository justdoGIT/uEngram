import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from pypdf import PdfReader

ROOT = Path(__file__).resolve().parents[1]
BUILD_SCRIPT = ROOT / "scripts" / "build_paper_pdf.py"


@pytest.mark.skipif(shutil.which("chromium") is None, reason="Chromium is unavailable")
def test_pdf_contains_markdown_metadata_and_internal_citation_link(tmp_path):
    paper = tmp_path / "metadata_fixture.md"
    paper.write_text(
        """# Metadata Test: Short Title

**Example Author** · author@example.org

Example Institute

**Keywords:** sensor memory · embedded systems

Preprint · 2026

## Abstract

A carefully chosen abstract subject.

---

## 1. Introduction

Prior work [1] motivates this paper.

## References

1. Example reference.
""",
        encoding="utf-8",
    )
    output_dir = tmp_path / "output"
    build = subprocess.run(
        [sys.executable, str(BUILD_SCRIPT), "--paper", str(paper), "--out-dir", str(output_dir)],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    link_counts = re.search(
        r"PDF link annotations: (\d+) before metadata update, (\d+) after\.",
        build.stdout,
    )
    assert link_counts is not None
    assert link_counts.group(1) == link_counts.group(2)

    pdf_path = output_dir / "metadata_fixture.pdf"
    metadata = PdfReader(pdf_path).metadata
    assert metadata.title == "Metadata Test: Short Title"
    assert metadata.author == "Example Author"
    assert metadata.subject == "A carefully chosen abstract subject."
    assert metadata.get("/Keywords") == "sensor memory · embedded systems"
    assert metadata.creator == "paper build pipeline"

    reader = PdfReader(pdf_path)
    links = [
        annotation.get_object()
        for page in reader.pages
        for annotation in page.get("/Annots", [])
        if annotation.get_object().get("/Subtype") == "/Link"
    ]
    assert any(link.get("/Dest") == "/ref-1" for link in links)
