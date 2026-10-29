#!/usr/bin/env python3
"""Build a research-paper PDF from its Markdown source.

Needs the docs extra: `uv sync --extra docs`.
Usage: `.venv/bin/python scripts/build_paper_pdf.py [--paper PATH] [--out-dir DIR]`.
"""
from __future__ import annotations

import subprocess
from html import escape
from pathlib import Path

import markdown
from pygments.formatters import HtmlFormatter

from paper_render_lib import (
    MD_EXTENSIONS,
    Stash,
    caption_tables,
    link_references,
    polish_text,
    protect_math,
    reference_count_from_markdown,
    render_front,
    set_pdf_metadata,
    stash_code_blocks,
    stash_figures,
)

ROOT = Path(__file__).resolve().parents[1]
PAPER_DIR = ROOT / "docs" / "research-paper"
PAPER_MD = PAPER_DIR / "From_Sensor_Streams_to_Bounded_Machine_Memory.md"
STYLE = PAPER_DIR / "paper.css"
KATEX = "vendor/katex"


def build_html(paper_md: Path = PAPER_MD, html_out: Path | None = None,
               metadata_out: dict[str, str] | None = None) -> Path:
    blocks, maths = Stash("BLOCK"), Stash("MATH")
    content = paper_md.read_text(encoding="utf-8")
    reference_count = reference_count_from_markdown(content)
    content = stash_code_blocks(stash_figures(content, blocks, paper_md.parent / "figures"), blocks)
    content = protect_math(content, maths)
    head, separator, body = content.partition("\n---\n")
    if not separator:
        raise ValueError("paper needs a '---' line after the front matter")
    front, short_title, metadata = render_front(head, reference_count)
    if metadata_out is not None:
        metadata_out.update(metadata)
    body_html = polish_text(link_references(caption_tables(
        markdown.markdown(body, extensions=MD_EXTENSIONS)), reference_count))
    page = blocks.restore(maths.restore(front + body_html, escape_items=True))
    katex_uri = (PAPER_DIR / KATEX).resolve().as_uri()
    css_short_title = short_title.replace("\\", "\\\\").replace('"', '\\"').replace("\n", " ")
    paper_css = STYLE.read_text(encoding="utf-8").replace("__SHORT_TITLE__", css_short_title)
    title = metadata["title"]
    pygments_css = HtmlFormatter(style="friendly").get_style_defs("figure.listing pre")
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{escape(title)}</title>
<meta name="author" content="{escape(metadata['author'], quote=True)}">
<meta name="subject" content="{escape(metadata['subject'], quote=True)}">
<meta name="keywords" content="{escape(metadata['keywords'], quote=True)}">
<meta name="description" content="{escape(metadata['subject'], quote=True)}">
<link rel="stylesheet" href="{katex_uri}/katex.min.css">
<script src="{katex_uri}/katex.min.js"></script>
<script src="{katex_uri}/contrib/auto-render.min.js"></script>
<script>
document.addEventListener("DOMContentLoaded", () => renderMathInElement(document.body, {{
    delimiters: [{{left: "$$", right: "$$", display: true}},
                 {{left: "$", right: "$", display: false}}],
    throwOnError: false
}}));
</script>
<style>
{paper_css}
{pygments_css}
figure.listing pre {{ background: none; }}
</style>
</head>
<body>
{page}
</body>
</html>
"""
    output = html_out or paper_md.with_suffix(".html")
    output.write_text(html, encoding="utf-8")
    return output


def compile_pdf(html_path: Path, pdf_out: Path, metadata: dict[str, str]) -> Path:
    subprocess.run([
        "chromium", "--headless=new", "--no-sandbox", "--disable-gpu",
        "--no-pdf-header-footer", "--run-all-compositor-stages-before-draw",
        "--virtual-time-budget=15000", f"--print-to-pdf={pdf_out}",
        f"file://{html_path.resolve()}",
    ], check=True, stderr=subprocess.DEVNULL)
    set_pdf_metadata(pdf_out, metadata)
    return pdf_out


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Build the research paper PDF.")
    parser.add_argument("--paper", type=Path, default=PAPER_MD)
    parser.add_argument("--out-dir", type=Path)
    args = parser.parse_args()
    output_dir = args.out_dir or args.paper.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata: dict[str, str] = {}
    html_path = build_html(args.paper, output_dir / args.paper.with_suffix(".html").name,
                           metadata_out=metadata)
    print(f"HTML generated at: {html_path}")
    subprocess.run([".venv/bin/python", str(ROOT / "scripts" / "lint_paper.py"),
                    str(args.paper)], cwd=ROOT, check=True)
    pdf_path = output_dir / args.paper.with_suffix(".pdf").name
    print(f"PDF successfully compiled at: {compile_pdf(html_path, pdf_path, metadata)}")


if __name__ == "__main__":
    main()
