#!/usr/bin/env python3
"""Build the uEngram research paper PDF from its Markdown source.

Markdown conventions this builder adds:
  * The text before the first "---" is the front matter:
        # Title
        **Author** · affiliation · email
        **Keywords:** a · b · c
        ## Abstract
        paragraphs...
        > **Principal findings**   (blockquote box, optional)
  * ![Figure N. caption](figures/name.svg) inlines the SVG as a numbered figure.
  * ```python | Listing N. caption``` renders a highlighted source listing.
  * ```algorithm | Algorithm N. caption``` renders numbered pseudocode; lines
    starting "Input:" / "Output:" are not numbered, text after "▷" is a comment.
  * A paragraph "**Table N.** caption" directly before a table becomes its caption.
"""
from __future__ import annotations

import re
import subprocess
from html import escape
from pathlib import Path

import markdown
from pygments import highlight
from pygments.formatters import HtmlFormatter
from pygments.lexers import get_lexer_by_name

ROOT = Path(__file__).resolve().parents[1]
PAPER_DIR = ROOT / "docs" / "research-paper"
PAPER_MD = PAPER_DIR / "From_Sensor_Streams_to_Bounded_Machine_Memory.md"
STYLE = PAPER_DIR / "paper.css"
FIGURES_DIR = PAPER_DIR / "figures"
HTML_OUT = PAPER_MD.with_suffix(".html")
PDF_OUT = PAPER_MD.with_suffix(".pdf")
KICKER = ("Original Research Article · Embedded Systems, Signal Processing &amp; Machine Learning",
          "Preprint manuscript · uEngram reference implementation")
MD_EXTENSIONS = ["tables", "fenced_code"]
ALGO_KEYWORDS = ("for each", "for", "if", "else if", "else", "then", "do", "return", "repeat",
                 "raise")
KATEX = "https://cdn.jsdelivr.net/npm/katex@0.16.8/dist"


class Stash:
    """Hold rendered HTML out of the Markdown pass behind unique tokens."""

    def __init__(self, tag: str):
        self.tag = tag
        self.items: list[str] = []

    def put(self, html: str) -> str:
        self.items.append(html)
        return f"{self.tag}{len(self.items) - 1}END"

    def restore(self, html: str, escape_items: bool = False) -> str:
        for index, item in enumerate(self.items):
            value = escape(item, quote=False) if escape_items else item
            token = f"{self.tag}{index}END"
            html = html.replace(f"<p>{token}</p>", value).replace(token, value)
        return html


def inline_markdown(text: str) -> str:
    html = markdown.markdown(text, extensions=MD_EXTENSIONS).strip()
    return re.sub(r"^<p>(.*)</p>$", r"\1", html, flags=re.S)


def caption_html(caption: str) -> str:
    match = re.match(r"((?:Figure|Listing|Algorithm|Table) [A-Z]?\d+\.)\s*(.*)", caption, re.S)
    if not match:
        raise ValueError(f"caption must start with a numbered label: {caption!r}")
    return f"<b>{match.group(1)}</b> {inline_markdown(match.group(2))}"


def stash_figures(content: str, blocks: Stash) -> str:
    def replace(match):
        svg = (FIGURES_DIR / match.group(2)).read_text(encoding="utf-8")
        svg = re.sub(r"<\?xml.*?\?>", "", svg).strip()
        return blocks.put(
            f'<figure class="paper-figure"><div class="frame">{svg}</div>'
            f"<figcaption>{caption_html(match.group(1))}</figcaption></figure>"
        )

    return re.sub(r"!\[(.*?)\]\(figures/([\w-]+\.svg)\)", replace, content)


def render_algorithm(code: str) -> str:
    keyword = re.compile(r"\b(" + "|".join(k.replace(" ", r"\s") for k in ALGO_KEYWORDS) + r")\b")
    lines, number = [], 0
    for raw in code.rstrip("\n").split("\n"):
        body, _, comment = raw.partition("▷")
        io = re.match(r"(\s*)(Input|Output|Procedure)(:?)(.*)", body)
        if io:
            text = (f'<span class="ln"></span>{escape(io.group(1))}<span class="io">'
                    f"{io.group(2)}{io.group(3)}</span>{escape(io.group(4))}")
        else:
            number += 1
            text = f'<span class="ln">{number}</span>' + keyword.sub(
                r'<span class="kw">\1</span>', escape(body))
        if comment:
            text += f'<span class="cm">▷{escape(comment)}</span>'
        lines.append(text)
    return "\n".join(lines)


def stash_code_blocks(content: str, blocks: Stash) -> str:
    formatter = HtmlFormatter(nowrap=True)

    def replace(match):
        language, caption, code = match.group(1), match.group(2).strip(), match.group(3)
        if language == "algorithm":
            body, kind = render_algorithm(code), "code algorithm"
        else:
            body = highlight(code, get_lexer_by_name(language), formatter).rstrip("\n")
            kind = "code listing"
        if code.count("\n") > 18:
            kind += " breakable"
        return blocks.put(
            f'<figure class="{kind}"><figcaption>{caption_html(caption)}</figcaption>'
            f'<pre class="highlight">{body}</pre></figure>'
        )

    return re.sub(r"^```(\w+) \| (.+?)\n(.*?)^```[ \t]*$", replace, content, flags=re.M | re.S)


def protect_math(content: str, maths: Stash) -> str:
    """Swap TeX spans for tokens so Markdown cannot rewrite _ and *."""
    content = re.sub(r"\$\$.*?\$\$", lambda m: maths.put(m.group(0)), content, flags=re.S)
    return re.sub(r"(?<![\\$])\$(?!\s)[^$\n]+?(?<!\s)\$", lambda m: maths.put(m.group(0)),
                  content)


def caption_tables(html: str) -> str:
    html = re.sub(r'<p><strong>(Table [A-Z]?\d+\.)</strong>\s*(<span class="outcome (\w+)">.*?)</p>\s*<table>',
                  r'<table class="tone-\3"><caption><b>\1</b> \2</caption>', html, flags=re.S)
    return re.sub(r"<p><strong>(Table [A-Z]?\d+\.)</strong>\s*(.*?)</p>\s*<table>",
                  r"<table><caption><b>\1</b> \2</caption>", html, flags=re.S)


def polish_text(html: str) -> str:
    """Let long inline code paths wrap after '/' and '_', and keep names such as
    MetroPT-3 or Chronos-2 from splitting at their hyphen."""
    out, in_code = [], False
    for piece in re.split(r"(<[^>]+>)", html):
        if piece.startswith("<"):
            in_code = piece.startswith("<code") or (in_code and piece != "</code>")
            out.append(piece)
        elif in_code:
            out.append(re.sub(r"([/_])(?=\w)", r"\1<wbr>", piece))
        else:
            out.append(re.sub(r"\b([A-Za-z]+-\d+)\b", r'<span class="nb">\1</span>', piece))
    return "".join(out)


def render_front(head: str) -> tuple[str, str]:
    match = re.fullmatch(
        r"# (?P<title>.+?)\n\n\*\*(?P<name>.+?)\*\*(?P<meta>.*?)\n\n"
        r"\*\*Keywords:\*\* (?P<keywords>.+?)\n\n## Abstract\n\n(?P<abstract>.+?)"
        r"(?:\n\n(?P<findings>>.+))?",
        head.strip(), flags=re.S)
    if not match:
        raise ValueError("front matter does not follow the documented layout")
    findings = match.group("findings")
    html = (
        f'<div class="kicker">{KICKER[0]}<br>{KICKER[1]}</div>'
        f'<h1 class="title">{match.group("title")}</h1>'
        f'<p class="byline"><span class="name">{match.group("name")}</span>'
        f'<span class="meta">{escape(match.group("meta"))}</span></p>'
        f'<p class="keywords"><b>Keywords:</b> {match.group("keywords")}</p>'
        f'<div class="abstract"><div class="label">ABSTRACT</div>'
        f'{markdown.markdown(match.group("abstract"), extensions=MD_EXTENSIONS)}</div>'
        + (markdown.markdown(findings, extensions=MD_EXTENSIONS) if findings else "")
    )
    return polish_text(html), re.sub(r"\s+", " ",
                                     re.sub(r"\$[^$]*\$|MATH\d+END", "", match.group("title")))


def build_html() -> Path:
    blocks, maths = Stash("BLOCK"), Stash("MATH")
    content = PAPER_MD.read_text(encoding="utf-8")
    content = stash_code_blocks(stash_figures(content, blocks), blocks)
    content = protect_math(content, maths)
    head, separator, body = content.partition("\n---\n")
    if not separator:
        raise ValueError("paper needs a '---' line after the front matter")
    front, title = render_front(head)
    body_html = polish_text(caption_tables(markdown.markdown(body, extensions=MD_EXTENSIONS)))
    page = blocks.restore(maths.restore(front + body_html, escape_items=True))
    pygments_css = HtmlFormatter(style="friendly").get_style_defs("figure.listing pre")
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{title}</title>
<link rel="stylesheet" href="{KATEX}/katex.min.css">
<script src="{KATEX}/katex.min.js"></script>
<script src="{KATEX}/contrib/auto-render.min.js"></script>
<script>
document.addEventListener("DOMContentLoaded", () => renderMathInElement(document.body, {{
    delimiters: [{{left: "$$", right: "$$", display: true}},
                 {{left: "$", right: "$", display: false}}],
    throwOnError: false
}}));
</script>
<style>
{STYLE.read_text(encoding="utf-8")}
{pygments_css}
figure.listing pre {{ background: none; }}
</style>
</head>
<body>
{page}
</body>
</html>
"""
    HTML_OUT.write_text(html, encoding="utf-8")
    return HTML_OUT


def compile_pdf(html_path: Path) -> Path:
    subprocess.run([
        "chromium", "--headless=new", "--no-sandbox", "--disable-gpu",
        "--no-pdf-header-footer", "--run-all-compositor-stages-before-draw",
        "--virtual-time-budget=15000", f"--print-to-pdf={PDF_OUT}",
        f"file://{html_path.resolve()}",
    ], check=True, stderr=subprocess.DEVNULL)
    return PDF_OUT


def main():
    print("Building HTML with KaTeX, listings, and inlined figures...")
    html_path = build_html()
    print(f"HTML generated at: {html_path}")
    print("Compiling PDF with Chromium headless...")
    print(f"PDF successfully compiled at: {compile_pdf(html_path)}")


if __name__ == "__main__":
    main()
