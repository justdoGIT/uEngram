# you asked

message time: 2026-10-29 01:02:29

please provide the code to change any text into this kind of pdf.

---

# reference response

```python?code_reference&code_event_index=1
script_content = '''#!/usr/bin/env python3
"""
render_paper.py — Convert any Markdown or structured text file into a publication-grade
single-column academic PDF using WeasyPrint, Python-Markdown, and Pygments.

Usage:
    python3 render_paper.py input.md output.pdf \\
        --author "Kamal Kishor Pandey" \\
        --email "kamal.pandey@kepr.dev" \\
        --affiliation "Independent Researcher · B.E. in Electronics & Telecommunications (2013–2017), Army Institute of Technology (AIT), Pune, India" \\
        --banner-left "Original Research Article · Embedded Systems, Signal Processing & Machine Learning" \\
        --banner-right "Preprint Manuscript" \\
        --footer "μEngram (MicroEngram): From Sensor Streams to Machine Memory by K.K Pandey · kamal.pandey@kepr.dev"
"""

import argparse
import datetime
import pathlib
import re
import sys
import markdown
from weasyprint import HTML, CSS

Academic_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');

@page {
    size: A4 portrait;
    margin: 16mm 15mm 18mm 15mm;
    @bottom-left {
        content: string(footer-left);
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
        font-size: 7.5pt;
        color: #64748b;
    }
    @bottom-right {
        content: "Page " counter(page) " of " counter(pages);
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
        font-size: 7.5pt;
        font-weight: 600;
        color: #475569;
    }
}

/* Hidden element that sets the running footer string */
.running-footer-def {
    string-set: footer-left content();
    display: none;
}

* {
    box-sizing: border-box;
}

body {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    font-size: 9.3pt;
    line-height: 1.48;
    color: #1e293b;
    margin: 0;
    padding: 0;
    text-align: left;
    overflow-wrap: break-word;
    word-wrap: break-word;
}

/* 1. Top Classification Banner (compact and spaced away from main title) */
.top-banner {
    display: flex;
    justify-content: space-between;
    align-items: center;
    font-size: 7.2pt;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    color: #64748b;
    border-bottom: 1px solid #cbd5e1;
    padding-bottom: 4px;
    margin-bottom: 14px;
}

/* 2. Main Document Title (Large, Bold, Commanding) */
h1.paper-title {
    font-size: 20.5pt;
    font-weight: 700;
    line-height: 1.18;
    color: #0f172a;
    margin: 0 0 10px 0;
    letter-spacing: -0.015em;
}

/* 3. Highlighted Author Pill Box */
.author-box {
    display: inline-block;
    background-color: #eff6ff;
    border: 1px solid #bfdbfe;
    border-left: 3.5px solid #1d4ed8;
    padding: 5px 12px;
    margin-bottom: 6px;
    font-size: 10.5pt;
    font-weight: 600;
    color: #1e3a8a;
}

.author-box a {
    color: #2563eb;
    text-decoration: none;
    font-weight: 600;
}

.author-box .sep {
    color: #93c5fd;
    margin: 0 6px;
}

/* 4. Compact Affiliation & Date Line */
.affiliation-line {
    font-size: 8.3pt;
    color: #475569;
    line-height: 1.38;
    margin-bottom: 8px;
}

/* 5. Keywords Strip */
.keywords-line {
    font-size: 8.2pt;
    color: #334155;
    background: #f8fafc;
    border-top: 1px solid #e2e8f0;
    border-bottom: 1px solid #e2e8f0;
    padding: 4px 8px;
    margin-bottom: 12px;
}

.keywords-line strong {
    color: #0f172a;
    text-transform: uppercase;
    font-size: 7.5pt;
    letter-spacing: 0.03em;
    margin-right: 4px;
}

.header-divider {
    border: 0;
    border-top: 1.5px solid #0f172a;
    margin: 0 0 12px 0;
}

/* 6. Abstract & Principal Findings Boxes */
.abstract-box {
    background-color: #f8fafc;
    border: 1px solid #cbd5e1;
    border-left: 3.5px solid #0f172a;
    border-radius: 3px;
    padding: 9px 12px;
    margin-bottom: 11px;
    break-inside: avoid;
}

.abstract-title {
    font-size: 8pt;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: #0f172a;
    margin: 0 0 5px 0;
}

.abstract-box p {
    margin: 0 0 6px 0;
    font-size: 8.9pt;
    line-height: 1.44;
    color: #1e293b;
}

.abstract-box p:last-child {
    margin-bottom: 0;
}

.findings-box {
    background-color: #f0f9ff;
    border: 1px solid #bae6fd;
    border-left: 3.5px solid #0284c7;
    border-radius: 3px;
    padding: 9px 12px;
    margin-bottom: 14px;
    break-inside: avoid;
}

.findings-title {
    font-size: 8.2pt;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: #0369a1;
    margin: 0 0 5px 0;
}

.findings-box ol, .findings-box ul {
    margin: 0;
    padding-left: 16px;
    font-size: 8.6pt;
    line-height: 1.4;
}

.findings-box li {
    margin-bottom: 3px;
}

/* 7. Section Headings */
h2 {
    font-size: 11.2pt;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.025em;
    color: #0f172a;
    border-bottom: 1.2px solid #cbd5e1;
    padding-bottom: 3px;
    margin-top: 14px;
    margin-bottom: 7px;
    break-after: avoid;
}

h3 {
    font-size: 9.8pt;
    font-weight: 700;
    color: #1e3a8a;
    margin-top: 10px;
    margin-bottom: 5px;
    break-after: avoid;
}

p {
    margin: 0 0 7px 0;
}

ul, ol {
    margin: 0 0 8px 0;
    padding-left: 18px;
}

li {
    margin-bottom: 3.5px;
}

/* 8. Inline Code & Light-Background Code Listings (Never Split Across Pages) */
code {
    font-family: 'JetBrains Mono', 'Fira Code', monospace;
    font-size: 8.1pt;
    background-color: #f1f5f9;
    color: #0f172a;
    padding: 1px 4px;
    border-radius: 2px;
    word-break: break-word;
}

.listing-container, .algorithm-box {
    background-color: #f8fafc;
    border: 1px solid #cbd5e1;
    border-left: 3px solid #2563eb;
    border-radius: 3px;
    margin: 9px 0 11px 0;
    break-inside: avoid;
    page-break-inside: avoid;
}

.algorithm-box {
    border-left-color: #0f172a;
    background-color: #fcfcfd;
}

.box-caption {
    background-color: #f1f5f9;
    border-bottom: 1px solid #cbd5e1;
    padding: 4px 9px;
    font-size: 8.1pt;
    font-weight: 600;
    color: #0f172a;
}

pre {
    margin: 0;
    padding: 7px 10px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 7.8pt;
    line-height: 1.38;
    background-color: transparent;
    color: #0f172a;
    white-space: pre-wrap;
    word-break: break-word;
    break-inside: avoid;
}

pre code {
    background: none;
    padding: 0;
    font-size: inherit;
}

/* Light-background syntax highlighting tokens */
.kw { color: #1d4ed8; font-weight: 600; }
.fn { color: #0369a1; font-weight: 600; }
.cm { color: #64748b; font-style: italic; }
.st { color: #15803d; }
.nu { color: #b45309; }

/* 9. Equation Block */
.equation-block {
    background-color: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 3px;
    padding: 6px 12px;
    margin: 7px 0;
    text-align: center;
    font-family: 'Cambria Math', 'STIX Two Math', 'Times New Roman', serif;
    font-size: 9.8pt;
    color: #0f172a;
    break-inside: avoid;
}

/* 10. Academic Tables */
.table-caption {
    font-size: 8.2pt;
    font-weight: 600;
    color: #0f172a;
    margin: 9px 0 4px 0;
    break-after: avoid;
}

table {
    width: 100%;
    border-collapse: collapse;
    margin: 0 0 10px 0;
    font-size: 8pt;
    line-height: 1.34;
    table-layout: auto;
}

thead th {
    background-color: #f1f5f9;
    color: #0f172a;
    font-weight: 700;
    text-align: left;
    padding: 5px 6px;
    border-top: 1.5px solid #0f172a;
    border-bottom: 1px solid #64748b;
}

tbody td {
    padding: 4.5px 6px;
    border-bottom: 1px solid #e2e8f0;
    vertical-align: top;
}

tbody tr:nth-child(even) {
    background-color: #f8fafc;
}

tbody tr:last-child td {
    border-bottom: 1.5px solid #0f172a;
}

tr {
    break-inside: avoid;
    page-break-inside: avoid;
}

/* 11. Figures & SVG Diagrams */
figure {
    margin: 10px 0;
    padding: 8px;
    background: #ffffff;
    border: 1px solid #cbd5e1;
    border-radius: 4px;
    text-align: center;
    break-inside: avoid;
    page-break-inside: avoid;
}

figure svg, figure img {
    max-width: 100%;
    height: auto;
    display: block;
    margin: 0 auto;
}

figcaption {
    margin-top: 6px;
    font-size: 7.9pt;
    color: #475569;
    text-align: left;
    line-height: 1.35;
}

/* 12. Single-Page References List */
.references-section ol {
    font-size: 7.8pt;
    line-height: 1.32;
    padding-left: 18px;
    color: #334155;
}

.references-section li {
    margin-bottom: 2.5px;
}
"""


def parse_frontmatter(text: str) -> tuple[dict, str]:
    """Extract optional YAML-style --- frontmatter from the top of a text/markdown file."""
    meta = {}
    if text.startswith("---"):
        parts = text.split("---", 2)
        if len(parts) >= 3:
            raw_meta, body = parts[1], parts[2].lstrip()
            for line in raw_meta.splitlines():
                if ":" in line:
                    k, v = line.split(":", 1)
                    meta[k.strip().lower()] = v.strip().strip('"').strip("'")
            return meta, body
    return meta, text


def transform_custom_blocks(md_text: str) -> str:
    """
    Convert custom blocks in Markdown into styled academic containers:
      - ::: abstract ... :::
      - ::: findings ... :::
      - ::: algorithm Title ... :::
      - ::: listing Title ... :::
    """
    # Abstract block
    md_text = re.sub(
        r":::\s*abstract\s*\n(.*?)\n:::",
        lambda m: f'<div class="abstract-box"><div class="abstract-title">Abstract</div>\\n{markdown.markdown(m.group(1))}</div>',
        md_text,
        flags=re.DOTALL,
    )

    # Principal findings block
    md_text = re.sub(
        r":::\s*findings\s*\n(.*?)\n:::",
        lambda m: f'<div class="findings-box"><div class="findings-title">Principal Findings</div>\\n{markdown.markdown(m.group(1))}</div>',
        md_text,
        flags=re.DOTALL,
    )

    # Algorithm block: ::: algorithm Algorithm 1. Title
    def _algo_repl(match):
        title = match.group(1).strip()
        body = match.group(2).strip()
        return (
            f'<div class="algorithm-box">'
            f'<div class="box-caption">{title}</div>'
            f'<pre>{body}</pre></div>'
        )

    md_text = re.sub(
        r":::\s*algorithm\s+([^\n]+)\n(.*?)\n:::",
        _algo_repl,
        md_text,
        flags=re.DOTALL,
    )

    # Listing block: ::: listing Listing 1. Title
    def _listing_repl(match):
        title = match.group(1).strip()
        body = match.group(2).strip()
        # Strip inner ```python ... ``` if present
        body = re.sub(r"^```[a-zA-Z0-9_-]*\n", "", body)
        body = re.sub(r"\n```$", "", body)
        return (
            f'<div class="listing-container">'
            f'<div class="box-caption">{title}</div>'
            f'<pre><code>{body}</code></pre></div>'
        )

    md_text = re.sub(
        r":::\s*listing\s+([^\n]+)\n(.*?)\n:::",
        _listing_repl,
        md_text,
        flags=re.DOTALL,
    )

    return md_text


def build_html(md_text: str, args: argparse.Namespace) -> str:
    meta, body = parse_frontmatter(md_text)

    # Extract H1 title from frontmatter, CLI, or first '# Heading' in body
    title = args.title or meta.get("title")
    if not title:
        m = re.search(r"^#\s+(.+)$", body, flags=re.MULTILINE)
        if m:
            title = m.group(1).strip()
            body = body[: m.start()] + body[m.end() :].lstrip()
        else:
            title = "Research Manuscript"

    author = args.author or meta.get("author", "Kamal Kishor Pandey")
    email = args.email or meta.get("email", "kamal.pandey@kepr.dev")
    affiliation = args.affiliation or meta.get(
        "affiliation",
        "Independent Researcher · B.E. in Electronics & Telecommunications (2013–2017), Army Institute of Technology (AIT), Pune, India",
    )
    date_str = args.date or meta.get("date", datetime.date.today().strftime("%B %d, %Y"))
    keywords = args.keywords or meta.get("keywords", "")
    banner_left = args.banner_left or meta.get(
        "banner_left",
        "Original Research Article · Embedded Systems, Signal Processing & Machine Learning",
    )
    banner_right = args.banner_right or meta.get(
        "banner_right", "Preprint Manuscript"
    )
    footer_text = args.footer or meta.get(
        "footer",
        f"{title[:52]} by {author} · {email}",
    )

    body = transform_custom_blocks(body)
    html_body = markdown.markdown(
        body,
        extensions=["tables", "fenced_code", "sane_lists"],
    )

    keywords_html = (
        f'<div class="keywords-line"><strong>Keywords:</strong> {keywords}</div>'
        if keywords
        else ""
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{title}</title>
</head>
<body>
    <div class="running-footer-def">{footer_text}</div>

    <div class="top-banner">
        <span>{banner_left}</span>
        <span>{banner_right}</span>
    </div>

    <h1 class="paper-title">{title}</h1>

    <div class="author-box">
        <span>{author}</span>
        <span class="sep">·</span>
        <a href="mailto:{email}">{email}</a>
    </div>

    <div class="affiliation-line">
        {affiliation} · {date_str}
    </div>

    {keywords_html}
    <hr class="header-divider">

    <main>
        {html_body}
    </main>
</body>
</html>
"""


def main():
    parser = argparse.ArgumentParser(
        description="Render any Markdown or text file into a publication-ready academic PDF."
    )
    parser.add_argument("input", type=pathlib.Path, help="Input Markdown/text file (.md or .txt)")
    parser.add_argument("output", type=pathlib.Path, help="Output PDF path (.pdf)")
    parser.add_argument("--title", help="Override paper title")
    parser.add_argument("--author", help="Author full name")
    parser.add_argument("--email", help="Author contact email")
    parser.add_argument("--affiliation", help="Affiliation and degree details")
    parser.add_argument("--date", help="Publication/manuscript date")
    parser.add_argument("--keywords", help="Middle-dot separated keywords list")
    parser.add_argument("--banner-left", help="Top-left classification line")
    parser.add_argument("--banner-right", help="Top-right manuscript status")
    parser.add_argument("--footer", help="Running bottom-left footer text")
    args = parser.parse_args()

    raw_text = args.input.read_text(encoding="utf-8")
    full_html = build_html(raw_text, args)

    HTML(string=full_html, base_url=str(args.input.parent.resolve())).write_pdf(
        str(args.output),
        stylesheets=[CSS(string=Academic_CSS)],
    )
    print(f"Generated PDF: {args.output}")


if __name__ == "__main__":
    main()
'''

with open("render_paper.py", "w", encoding="utf-8") as f:
    f.write(script_content)

print("Saved render_paper.py successfully.")

```

```text?code_stdout&code_event_index=1
Saved render_paper.py successfully.

```

Your Python script file is ready
[file-tag: code-generated-file-d667cd11-bc05-4956-9eed-1c5fe5df9ed8]

Below is the complete, self-contained Python CLI generator (`render_paper.py`) that transforms any Markdown or plain-text document into the exact academic PDF design used for your **μEngram** paper—complete with the spaced top classification banner, **20.5pt** bold title, highlighted blue author box, compact affiliation/date line, unbreakable light-background code/algorithm blocks, academic tables, and automatic `Page X of Y` running footers.

---

### 1. Dependencies & Installation

On **Void Linux**, install `WeasyPrint` and `Markdown` via `xbps-install` (or inside a `uv` / `venv` environment):

```bash
# Void Linux system packages
sudo xbps-install -S python3-WeasyPrint python3-Markdown python3-Pygments

# Or via uv / pip
uv pip install weasyprint markdown pygments
```

---

### 2. How to Format Your Input Markdown / Text File (`paper.md`)

You can write standard Markdown (tables, headings, lists, inline SVGs/images) and use optional YAML frontmatter plus four lightweight directives (`::: abstract`, `::: findings`, `::: algorithm`, and `::: listing`) that automatically style and prevent mid-block page breaks:

```markdown
---
title: "From Sensor Streams to Bounded Machine Memory: Architecture in μEngram"
author: "Kamal Kishor Pandey"
email: "kamal.pandey@kepr.dev"
affiliation: "Independent Researcher · B.E. in Electronics & Telecommunications (2013–2017), Army Institute of Technology (AIT), Pune, India"
date: "October 28, 2026"
keywords: "Bounded edge memory · Streaming anomaly detection · Noise whitening · Knowledge distillation"
banner_left: "Original Research Article · Embedded Systems, Signal Processing & Machine Learning"
banner_right: "Preprint manuscript · μEngram reference implementation"
footer: "μEngram (MicroEngram): From Sensor Streams to Machine Memory by K.K Pandey · kamal.pandey@kepr.dev"
---

::: abstract
Industrial and edge equipment generate continuous telemetry across vibration, temperature, and load...
:::

::: findings
1. **Bounded memory works where its assumptions hold:** 86.2% fewer writes at 1.00 recall.
2. **Ranking is not warning:** Every CWRU model reaches 1.0000 AUROC yet flags the held-out healthy file.
:::

## 1. Introduction and Core Architecture

Standard prose paragraphs, bullet lists, and inline code like `src/uengram/engine.py` flow naturally.

::: algorithm Algorithm 1. Robust within-state noise covariance (src/uengram/stats.py)
Input: projected training vectors z_1..z_n, lag L, trim quantile q_t = 0.975
Output: noise covariance N
1: D <- {(z_{t+L} - z_t) / sqrt(2) : t and t+L in same group}
2: N <- diag(sigma^2), sigma_j <- 1.4826 * MAD(D_j)
3: return N
:::

::: listing Listing 1. Symmetric noise whitening (src/uengram/encoders.py)
def band_features(windows):
    x = np.asarray(windows, dtype=np.float64)
    return x.mean(axis=2)
:::
```

---

### 3. Complete Generator Code (`render_paper.py`)

```python
#!/usr/bin/env python3
"""
render_paper.py — Convert any Markdown or structured text file into a publication-grade
single-column academic PDF using WeasyPrint and Python-Markdown.

Usage:
    python3 render_paper.py input.md output.pdf
"""

import argparse
import datetime
import pathlib
import re
import markdown
from weasyprint import HTML, CSS

Academic_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');

@page {
    size: A4 portrait;
    margin: 16mm 15mm 18mm 15mm;
    @bottom-left {
        content: string(footer-left);
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
        font-size: 7.5pt;
        color: #64748b;
    }
    @bottom-right {
        content: "Page " counter(page) " of " counter(pages);
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
        font-size: 7.5pt;
        font-weight: 600;
        color: #475569;
    }
}

.running-footer-def {
    string-set: footer-left content();
    display: none;
}

* {
    box-sizing: border-box;
}

body {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    font-size: 9.3pt;
    line-height: 1.48;
    color: #1e293b;
    margin: 0;
    padding: 0;
    text-align: left;
    overflow-wrap: break-word;
    word-wrap: break-word;
}

/* 1. Top Classification Banner (compact and spaced away from main title) */
.top-banner {
    display: flex;
    justify-content: space-between;
    align-items: center;
    font-size: 7.2pt;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    color: #64748b;
    border-bottom: 1px solid #cbd5e1;
    padding-bottom: 4px;
    margin-bottom: 14px;
}

/* 2. Main Document Title (Large, Bold, Commanding) */
h1.paper-title {
    font-size: 20.5pt;
    font-weight: 700;
    line-height: 1.18;
    color: #0f172a;
    margin: 0 0 10px 0;
    letter-spacing: -0.015em;
}

/* 3. Highlighted Author Pill Box */
.author-box {
    display: inline-block;
    background-color: #eff6ff;
    border: 1px solid #bfdbfe;
    border-left: 3.5px solid #1d4ed8;
    padding: 5px 12px;
    margin-bottom: 6px;
    font-size: 10.5pt;
    font-weight: 600;
    color: #1e3a8a;
}

.author-box a {
    color: #2563eb;
    text-decoration: none;
    font-weight: 600;
}

.author-box .sep {
    color: #93c5fd;
    margin: 0 6px;
}

/* 4. Compact Affiliation & Date Line */
.affiliation-line {
    font-size: 8.3pt;
    color: #475569;
    line-height: 1.38;
    margin-bottom: 8px;
}

/* 5. Keywords Strip */
.keywords-line {
    font-size: 8.2pt;
    color: #334155;
    background: #f8fafc;
    border-top: 1px solid #e2e8f0;
    border-bottom: 1px solid #e2e8f0;
    padding: 4px 8px;
    margin-bottom: 12px;
}

.keywords-line strong {
    color: #0f172a;
    text-transform: uppercase;
    font-size: 7.5pt;
    letter-spacing: 0.03em;
    margin-right: 4px;
}

.header-divider {
    border: 0;
    border-top: 1.5px solid #0f172a;
    margin: 0 0 12px 0;
}

/* 6. Abstract & Principal Findings Boxes */
.abstract-box {
    background-color: #f8fafc;
    border: 1px solid #cbd5e1;
    border-left: 3.5px solid #0f172a;
    border-radius: 3px;
    padding: 9px 12px;
    margin-bottom: 11px;
    break-inside: avoid;
}

.abstract-title {
    font-size: 8pt;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: #0f172a;
    margin: 0 0 5px 0;
}

.abstract-box p {
    margin: 0 0 6px 0;
    font-size: 8.9pt;
    line-height: 1.44;
    color: #1e293b;
}

.abstract-box p:last-child {
    margin-bottom: 0;
}

.findings-box {
    background-color: #f0f9ff;
    border: 1px solid #bae6fd;
    border-left: 3.5px solid #0284c7;
    border-radius: 3px;
    padding: 9px 12px;
    margin-bottom: 14px;
    break-inside: avoid;
}

.findings-title {
    font-size: 8.2pt;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: #0369a1;
    margin: 0 0 5px 0;
}

.findings-box ol, .findings-box ul {
    margin: 0;
    padding-left: 16px;
    font-size: 8.6pt;
    line-height: 1.4;
}

.findings-box li {
    margin-bottom: 3px;
}

/* 7. Section Headings */
h2 {
    font-size: 11.2pt;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.025em;
    color: #0f172a;
    border-bottom: 1.2px solid #cbd5e1;
    padding-bottom: 3px;
    margin-top: 14px;
    margin-bottom: 7px;
    break-after: avoid;
}

h3 {
    font-size: 9.8pt;
    font-weight: 700;
    color: #1e3a8a;
    margin-top: 10px;
    margin-bottom: 5px;
    break-after: avoid;
}

p {
    margin: 0 0 7px 0;
}

ul, ol {
    margin: 0 0 8px 0;
    padding-left: 18px;
}

li {
    margin-bottom: 3.5px;
}

/* 8. Inline Code & Light-Background Code Listings (Never Split Across Pages) */
code {
    font-family: 'JetBrains Mono', 'Fira Code', monospace;
    font-size: 8.1pt;
    background-color: #f1f5f9;
    color: #0f172a;
    padding: 1px 4px;
    border-radius: 2px;
    word-break: break-word;
}

.listing-container, .algorithm-box {
    background-color: #f8fafc;
    border: 1px solid #cbd5e1;
    border-left: 3px solid #2563eb;
    border-radius: 3px;
    margin: 9px 0 11px 0;
    break-inside: avoid;
    page-break-inside: avoid;
}

.algorithm-box {
    border-left-color: #0f172a;
    background-color: #fcfcfd;
}

.box-caption {
    background-color: #f1f5f9;
    border-bottom: 1px solid #cbd5e1;
    padding: 4px 9px;
    font-size: 8.1pt;
    font-weight: 600;
    color: #0f172a;
}

pre {
    margin: 0;
    padding: 7px 10px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 7.8pt;
    line-height: 1.38;
    background-color: transparent;
    color: #0f172a;
    white-space: pre-wrap;
    word-break: break-word;
    break-inside: avoid;
}

pre code {
    background: none;
    padding: 0;
    font-size: inherit;
}

/* 9. Academic Tables */
table {
    width: 100%;
    border-collapse: collapse;
    margin: 6px 0 10px 0;
    font-size: 8pt;
    line-height: 1.34;
}

thead th {
    background-color: #f1f5f9;
    color: #0f172a;
    font-weight: 700;
    text-align: left;
    padding: 5px 6px;
    border-top: 1.5px solid #0f172a;
    border-bottom: 1px solid #64748b;
}

tbody td {
    padding: 4.5px 6px;
    border-bottom: 1px solid #e2e8f0;
    vertical-align: top;
}

tbody tr:nth-child(even) {
    background-color: #f8fafc;
}

tbody tr:last-child td {
    border-bottom: 1.5px solid #0f172a;
}

tr {
    break-inside: avoid;
    page-break-inside: avoid;
}

/* 10. Figures & SVG Diagrams */
figure {
    margin: 10px 0;
    padding: 8px;
    background: #ffffff;
    border: 1px solid #cbd5e1;
    border-radius: 4px;
    text-align: center;
    break-inside: avoid;
    page-break-inside: avoid;
}

figure svg, figure img {
    max-width: 100%;
    height: auto;
    display: block;
    margin: 0 auto;
}

figcaption {
    margin-top: 6px;
    font-size: 7.9pt;
    color: #475569;
    text-align: left;
    line-height: 1.35;
}
"""


def parse_frontmatter(text: str) -> tuple[dict, str]:
    meta = {}
    if text.startswith("---"):
        parts = text.split("---", 2)
        if len(parts) >= 3:
            raw_meta, body = parts[1], parts[2].lstrip()
            for line in raw_meta.splitlines():
                if ":" in line:
                    k, v = line.split(":", 1)
                    meta[k.strip().lower()] = v.strip().strip('"').strip("'")
            return meta, body
    return meta, text


def transform_custom_blocks(md_text: str) -> str:
    md_text = re.sub(
        r":::\s*abstract\s*\n(.*?)\n:::",
        lambda m: f'<div class="abstract-box"><div class="abstract-title">Abstract</div>\n{markdown.markdown(m.group(1))}</div>',
        md_text,
        flags=re.DOTALL,
    )
    md_text = re.sub(
        r":::\s*findings\s*\n(.*?)\n:::",
        lambda m: f'<div class="findings-box"><div class="findings-title">Principal Findings</div>\n{markdown.markdown(m.group(1))}</div>',
        md_text,
        flags=re.DOTALL,
    )
    md_text = re.sub(
        r":::\s*algorithm\s+([^\n]+)\n(.*?)\n:::",
        lambda m: f'<div class="algorithm-box"><div class="box-caption">{m.group(1).strip()}</div><pre>{m.group(2).strip()}</pre></div>',
        md_text,
        flags=re.DOTALL,
    )

    def _listing_repl(match):
        title = match.group(1).strip()
        body = re.sub(r"^```[a-zA-Z0-9_-]*\n|\n```$", "", match.group(2).strip())
        return f'<div class="listing-container"><div class="box-caption">{title}</div><pre><code>{body}</code></pre></div>'

    md_text = re.sub(
        r":::\s*listing\s+([^\n]+)\n(.*?)\n:::",
        _listing_repl,
        md_text,
        flags=re.DOTALL,
    )
    return md_text


def build_html(md_text: str, args: argparse.Namespace) -> str:
    meta, body = parse_frontmatter(md_text)

    title = args.title or meta.get("title")
    if not title:
        m = re.search(r"^#\s+(.+)$", body, flags=re.MULTILINE)
        if m:
            title = m.group(1).strip()
            body = body[: m.start()] + body[m.end() :].lstrip()
        else:
            title = "Research Manuscript"

    author = args.author or meta.get("author", "Kamal Kishor Pandey")
    email = args.email or meta.get("email", "kamal.pandey@kepr.dev")
    affiliation = args.affiliation or meta.get(
        "affiliation",
        "Independent Researcher · B.E. in Electronics & Telecommunications (2013–2017), Army Institute of Technology (AIT), Pune, India",
    )
    date_str = args.date or meta.get("date", datetime.date.today().strftime("%B %d, %Y"))
    keywords = args.keywords or meta.get("keywords", "")
    banner_left = args.banner_left or meta.get(
        "banner_left",
        "Original Research Article · Embedded Systems, Signal Processing & Machine Learning",
    )
    banner_right = args.banner_right or meta.get("banner_right", "Preprint Manuscript")
    footer_text = args.footer or meta.get("footer", f"{title[:52]} by {author} · {email}")

    body = transform_custom_blocks(body)
    html_body = markdown.markdown(body, extensions=["tables", "fenced_code", "sane_lists"])
    keywords_html = (
        f'<div class="keywords-line"><strong>Keywords:</strong> {keywords}</div>'
        if keywords
        else ""
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{title}</title>
</head>
<body>
    <div class="running-footer-def">{footer_text}</div>
    <div class="top-banner">
        <span>{banner_left}</span>
        <span>{banner_right}</span>
    </div>
    <h1 class="paper-title">{title}</h1>
    <div class="author-box">
        <span>{author}</span>
        <span class="sep">·</span>
        <a href="mailto:{email}">{email}</a>
    </div>
    <div class="affiliation-line">{affiliation} · {date_str}</div>
    {keywords_html}
    <hr class="header-divider">
    <main>{html_body}</main>
</body>
</html>
"""


def main():
    parser = argparse.ArgumentParser(description="Render Markdown/text into an academic PDF.")
    parser.add_argument("input", type=pathlib.Path, help="Input Markdown/text file")
    parser.add_argument("output", type=pathlib.Path, help="Output PDF path")
    parser.add_argument("--title", help="Override paper title")
    parser.add_argument("--author", help="Author full name")
    parser.add_argument("--email", help="Author contact email")
    parser.add_argument("--affiliation", help="Affiliation and degree details")
    parser.add_argument("--date", help="Publication/manuscript date")
    parser.add_argument("--keywords", help="Middle-dot separated keywords list")
    parser.add_argument("--banner-left", help="Top-left classification line")
    parser.add_argument("--banner-right", help="Top-right manuscript status")
    parser.add_argument("--footer", help="Running bottom-left footer text")
    args = parser.parse_args()

    raw_text = args.input.read_text(encoding="utf-8")
    full_html = build_html(raw_text, args)
    HTML(string=full_html, base_url=str(args.input.parent.resolve())).write_pdf(
        str(args.output),
        stylesheets=[CSS(string=Academic_CSS)],
    )
    print(f"Generated PDF: {args.output}")


if __name__ == "__main__":
    main()
```

---

### 4. Running It From the Terminal

```bash
chmod +x render_paper.py
./render_paper.py sensor-state-memory.md uEngram_Paper.pdf
```

---
Powered by [AI Exporter](https://saveai.net)