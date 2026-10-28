#!/usr/bin/env python3
"""Build publication-quality PDF of the uEngram sensor-state-memory research paper."""
from __future__ import annotations

import re
import subprocess
from pathlib import Path

import markdown

ROOT = Path(__file__).resolve().parents[1]
PAPER_MD = ROOT / "docs" / "research-paper" / "sensor-state-memory.md"
FIGURES_DIR = ROOT / "docs" / "research-paper" / "figures"
HTML_OUT = ROOT / "docs" / "research-paper" / "sensor-state-memory.html"
PDF_OUT = ROOT / "docs" / "research-paper" / "sensor-state-memory.pdf"


def inline_svgs(content: str) -> str:
    """Replace markdown SVG image links with inline SVG markup."""
    pattern = re.compile(r"!\[(.*?)\]\(figures/([a-zA-Z0-9_-]+\.svg)\)")

    def replacer(match):
        caption = match.group(1)
        filename = match.group(2)
        svg_path = FIGURES_DIR / filename
        if svg_path.exists():
            svg_text = svg_path.read_text(encoding="utf-8")
            # Strip XML declaration if present
            svg_text = re.sub(r"<\?xml.*?\?>", "", svg_text).strip()
            return f'<figure class="paper-figure"><div class="svg-container">{svg_text}</div><figcaption>{caption}</figcaption></figure>'
        return match.group(0)

    return pattern.sub(replacer, content)


def prepare_mermaid_blocks(content: str) -> str:
    """Convert mermaid fenced code blocks into mermaid div tags."""
    pattern = re.compile(r"```mermaid\n(.*?)```", re.DOTALL)
    return pattern.sub(r'<div class="mermaid">\n\1\n</div>', content)


def build_html() -> Path:
    md_text = PAPER_MD.read_text(encoding="utf-8")

    # 1. Inline SVGs
    md_text = inline_svgs(md_text)

    # 2. Extract and preserve mermaid blocks before markdown parsing
    md_text = prepare_mermaid_blocks(md_text)

    # 3. Convert markdown to HTML
    body_html = markdown.markdown(
        md_text,
        extensions=["tables", "fenced_code", "toc"],
    )

    full_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>From Sensor Streams to Machine Memory: What uEngram Shows—and What It Does Not</title>
<!-- KaTeX for math rendering -->
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.8/dist/katex.min.css">
<script src="https://cdn.jsdelivr.net/npm/katex@0.16.8/dist/katex.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/katex@0.16.8/dist/contrib/auto-render.min.js"></script>
<!-- Mermaid for diagrams -->
<script src="https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.js"></script>
<script>
document.addEventListener("DOMContentLoaded", function() {{
    renderMathInElement(document.body, {{
        delimiters: [
            {{left: "$$", right: "$$", display: true}},
            {{left: "$", right: "$", display: false}}
        ],
        throwOnError: false
    }});
    mermaid.initialize({{
        startOnLoad: true,
        theme: 'neutral',
        fontFamily: 'Inter, system-ui, -apple-system, sans-serif'
    }});
}});
</script>
<style>
@page {{
    size: A4;
    margin: 20mm 15mm 20mm 15mm;
    @bottom-right {{
        content: counter(page);
        font-family: Inter, system-ui, sans-serif;
        font-size: 9pt;
        color: #64748b;
    }}
}}
body {{
    font-family: 'Liberation Serif', Georgia, Cambria, 'Times New Roman', Times, serif;
    font-size: 10.5pt;
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
}}
h1 {{
    font-size: 21pt;
    line-height: 1.25;
    margin-bottom: 8pt;
    border-bottom: 2px solid #0f172a;
    padding-bottom: 6pt;
}}
h2 {{
    font-size: 14pt;
    line-height: 1.3;
    margin-top: 18pt;
    margin-bottom: 6pt;
    border-bottom: 1px solid #cbd5e1;
    padding-bottom: 3pt;
}}
h3 {{
    font-size: 12pt;
    line-height: 1.35;
    margin-top: 14pt;
    margin-bottom: 4pt;
}}
h4 {{
    font-size: 11pt;
    margin-top: 10pt;
    margin-bottom: 3pt;
}}
p {{
    margin-top: 0;
    margin-bottom: 7pt;
    text-align: justify;
    text-justify: inter-word;
}}
table {{
    width: 100%;
    border-collapse: collapse;
    margin: 12pt 0;
    font-size: 9pt;
    line-height: 1.35;
    page-break-inside: avoid;
    font-family: Inter, system-ui, sans-serif;
}}
th, td {{
    border: 1px solid #cbd5e1;
    padding: 4.5pt 6pt;
    text-align: left;
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
    font-size: 8.5pt;
}}
pre {{
    background-color: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 4px;
    padding: 8pt 10pt;
    overflow-x: auto;
    page-break-inside: avoid;
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
blockquote {{
    border-left: 3px solid #3b82f6;
    margin: 10pt 0;
    padding: 4pt 12pt;
    background-color: #f0f7ff;
    color: #1e3a8a;
    font-style: italic;
    page-break-inside: avoid;
}}
figure.paper-figure {{
    margin: 16pt 0;
    text-align: center;
    page-break-inside: avoid;
}}
figure.paper-figure .svg-container {{
    display: inline-block;
    max-width: 95%;
    background-color: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 6px;
    padding: 8pt;
}}
figure.paper-figure .svg-container svg {{
    max-width: 100%;
    height: auto;
}}
figcaption {{
    font-size: 9pt;
    color: #475569;
    margin-top: 6pt;
    font-family: Inter, system-ui, sans-serif;
}}
.mermaid {{
    margin: 14pt 0;
    text-align: center;
    page-break-inside: avoid;
    background-color: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 6px;
    padding: 10pt;
}}
hr {{
    border: none;
    border-top: 1px solid #e2e8f0;
    margin: 16pt 0;
}}
ul, ol {{
    margin-top: 0;
    margin-bottom: 7pt;
    padding-left: 20pt;
}}
li {{
    margin-bottom: 3pt;
}}
</style>
</head>
<body>
{body_html}
</body>
</html>
"""
    HTML_OUT.write_text(full_html, encoding="utf-8")
    return HTML_OUT


def compile_pdf(html_path: Path) -> Path:
    cmd = [
        "chromium",
        "--headless=new",
        "--no-sandbox",
        "--disable-gpu",
        "--run-all-compositor-stages-before-draw",
        "--virtual-time-budget=12000",
        f"--print-to-pdf={PDF_OUT}",
        f"file://{html_path.resolve()}",
    ]
    subprocess.run(cmd, check=True)
    return PDF_OUT


def main():
    print("Building HTML with KaTeX, Mermaid, and inlined figures...")
    html_path = build_html()
    print(f"HTML generated at: {html_path}")
    print("Compiling PDF with Chromium headless...")
    pdf_path = compile_pdf(html_path)
    print(f"PDF successfully compiled at: {pdf_path}")


if __name__ == "__main__":
    main()
