#!/usr/bin/env python3
"""Convert markdown research paper to publication-quality PDF via Chromium headless."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import markdown


def convert_markdown_to_pdf(
    md_path: Path,
    pdf_path: Path | None = None,
    html_path: Path | None = None,
) -> Path:
    if not md_path.exists():
        raise FileNotFoundError(f"Markdown file not found: {md_path}")

    if pdf_path is None:
        pdf_path = md_path.with_suffix(".pdf")
    if html_path is None:
        html_path = md_path.with_suffix(".html")

    md_content = md_path.read_text(encoding="utf-8")

    # Render markdown to HTML
    body_html = markdown.markdown(
        md_content,
        extensions=["tables", "fenced_code", "toc", "def_list"],
    )

    title = md_path.stem.replace("-", " ").title()
    # Extract title from first h1 if available
    lines = md_content.splitlines()
    for line in lines:
        if line.startswith("# "):
            title = line[2:].strip()
            break

    full_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{title}</title>
<!-- KaTeX for crisp mathematical formulas -->
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.8/dist/katex.min.css">
<script src="https://cdn.jsdelivr.net/npm/katex@0.16.8/dist/katex.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/katex@0.16.8/dist/contrib/auto-render.min.js"></script>
<script>
document.addEventListener("DOMContentLoaded", function() {{
    renderMathInElement(document.body, {{
        delimiters: [
            {{left: "$$", right: "$$", display: true}},
            {{left: "$", right: "$", display: false}}
        ],
        throwOnError: false
    }});
}});
</script>
<style>
@page {{
    size: A4;
    margin: 20mm 16mm 20mm 16mm;
    @top-right {{
        content: "uEngram Research Report";
        font-family: Inter, system-ui, -apple-system, sans-serif;
        font-size: 8pt;
        color: #94a3b8;
    }}
    @bottom-right {{
        content: "Page " counter(page);
        font-family: Inter, system-ui, -apple-system, sans-serif;
        font-size: 8.5pt;
        color: #64748b;
    }}
}}
*, *:before, *:after {{
    box-sizing: border-box;
}}
body {{
    font-family: 'Liberation Serif', Georgia, 'Times New Roman', Times, serif;
    font-size: 10pt;
    line-height: 1.55;
    color: #1e293b;
    margin: 0;
    padding: 0;
}}
h1, h2, h3, h4, h5, h6 {{
    font-family: Inter, system-ui, -apple-system, 'Segoe UI', Roboto, sans-serif;
    color: #0f172a;
    font-weight: 700;
    page-break-after: avoid;
    break-after: avoid;
}}
h1 {{
    font-size: 19pt;
    line-height: 1.25;
    margin-top: 0;
    margin-bottom: 12pt;
    border-bottom: 2px solid #0f172a;
    padding-bottom: 6pt;
}}
h2 {{
    font-size: 13pt;
    line-height: 1.3;
    margin-top: 18pt;
    margin-bottom: 6pt;
    border-bottom: 1px solid #cbd5e1;
    padding-bottom: 3pt;
}}
h3 {{
    font-size: 11pt;
    line-height: 1.35;
    margin-top: 13pt;
    margin-bottom: 4pt;
}}
h4 {{
    font-size: 10pt;
    margin-top: 9pt;
    margin-bottom: 3pt;
}}
p {{
    margin-top: 0;
    margin-bottom: 7pt;
    text-align: justify;
    text-justify: inter-word;
}}
blockquote {{
    border-left: 3px solid #2563eb;
    margin: 10pt 0;
    padding: 6pt 12pt;
    background-color: #f8fafc;
    color: #334155;
    font-style: italic;
    page-break-inside: avoid;
    break-inside: avoid;
}}
table {{
    width: 100%;
    border-collapse: collapse;
    margin: 12pt 0;
    font-size: 8.5pt;
    line-height: 1.35;
    page-break-inside: avoid;
    break-inside: avoid;
    font-family: Inter, system-ui, -apple-system, sans-serif;
}}
th, td {{
    border: 1px solid #cbd5e1;
    padding: 5pt 6pt;
    text-align: left;
    vertical-align: top;
}}
th {{
    background-color: #f1f5f9;
    font-weight: 600;
    color: #0f172a;
}}
tr:nth-child(even) {{
    background-color: #f8fafc;
}}
pre, code {{
    font-family: 'Liberation Mono', 'Fira Code', 'DejaVu Sans Mono', Consolas, monospace;
    font-size: 8pt;
}}
pre {{
    background-color: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 4px;
    padding: 8pt 10pt;
    overflow-x: auto;
    page-break-inside: avoid;
    break-inside: avoid;
    line-height: 1.35;
}}
code {{
    background-color: #f1f5f9;
    padding: 1pt 3pt;
    border-radius: 3px;
}}
pre code {{
    background: none;
    padding: 0;
}}
ul, ol {{
    margin-top: 0;
    margin-bottom: 7pt;
    padding-left: 18pt;
}}
li {{
    margin-bottom: 3pt;
}}
hr {{
    border: none;
    border-top: 1px solid #e2e8f0;
    margin: 16pt 0;
}}
.katex-display {{
    margin: 8pt 0;
    overflow-x: auto;
    overflow-y: hidden;
}}
</style>
</head>
<body>
{body_html}
</body>
</html>
"""
    html_path.write_text(full_html, encoding="utf-8")

    cmd = [
        "chromium",
        "--headless=new",
        "--no-sandbox",
        "--disable-gpu",
        "--run-all-compositor-stages-before-draw",
        "--virtual-time-budget=10000",
        f"--print-to-pdf={pdf_path}",
        f"file://{html_path.resolve()}",
    ]
    subprocess.run(cmd, check=True)
    return pdf_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, nargs="?", default=Path("docs/research-paper/sensor-state-memory-and-anamoly-detection.md"))
    parser.add_argument("-o", "--output", type=Path)
    args = parser.parse_args()

    input_path = args.input.resolve()
    output_pdf = args.output.resolve() if args.output else input_path.with_suffix(".pdf")

    print(f"Converting: {input_path}")
    print(f"Target PDF: {output_pdf}")
    pdf_out = convert_markdown_to_pdf(input_path, output_pdf)
    print(f"Successfully generated PDF ({pdf_out.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
