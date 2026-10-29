"""Markdown rendering and PDF metadata helpers for the paper builder."""
from __future__ import annotations

import re
import tempfile
from html import escape
from pathlib import Path

import markdown
from pygments import highlight
from pygments.formatters import HtmlFormatter
from pygments.lexers import get_lexer_by_name

KICKER = ("Preprint", "Machine Learning for Embedded Sensor Systems")
MD_EXTENSIONS = ["tables", "fenced_code"]
ALGO_KEYWORDS = ("for each", "for", "if", "else if", "else", "then", "do", "return", "repeat",
                 "raise")


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


def stash_figures(content: str, blocks: Stash, figure_dir: Path) -> str:
    def replace(match):
        svg = (figure_dir / match.group(2)).read_text(encoding="utf-8")
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
    return re.sub(r"<p><strong>(Table [A-Z]?\d+\.)</strong>\s*(.*?)</p>\s*<table>",
                  r"<table><caption><b>\1</b> \2</caption>", html, flags=re.S)


def polish_text(html: str) -> str:
    """Wrap long inline paths and keep product names such as MetroPT-3 together."""
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


def render_front(head: str, reference_count: int = 0) -> tuple[str, str, dict[str, str]]:
    match = re.fullmatch(
        r"# (?P<title>.+?)\n\n\*\*(?P<name>.+?)\*\* · (?P<email>\S+@\S+)\n\n"
        r"(?P<affiliation>.+?)\n\n\*\*Keywords:\*\* (?P<keywords>.+?)\n\n(?P<date>.+?)\n\n"
        r"## Abstract\n\n(?P<abstract>.+?)(?:\n\n(?P<findings>>.+))?",
        head.strip(), flags=re.S)
    if not match:
        raise ValueError("front matter does not follow the documented layout")
    findings, email = match.group("findings"), match.group("email")
    title = match.group("title")
    keywords = re.sub(r"<[^>]+>", "", markdown.markdown(
        match.group("keywords"), extensions=MD_EXTENSIONS))
    abstract_text = re.sub(r"<[^>]+>", " ", markdown.markdown(match.group("abstract")))
    metadata = {
        "title": title,
        "author": match.group("name"),
        "subject": re.sub(r"\s+", " ", abstract_text).strip(),
        "keywords": keywords,
    }
    abstract_words = re.findall(r"\b[\w’'-]+\b", abstract_text)
    if len(abstract_words) > 250:
        print(f"Warning: abstract contains {len(abstract_words)} words (maximum recommended: 250)")
    short_title = re.sub(r"\s+", " ", title.split(":", 1)[0]).strip()
    html = (
        f'<div class="kicker"><span>{KICKER[0]}</span><span>{KICKER[1]}</span></div>'
        f'<h1 class="title">{title}</h1>'
        f'<div class="author-box"><span class="name">{escape(match.group("name"))}</span>'
        f'<span class="sep">·</span><a href="mailto:{email}">{email}</a></div>'
        f'<p class="affiliation">{escape(match.group("affiliation"))}</p>'
        f'<p class="keywords"><b>Keywords:</b> {match.group("keywords")}</p>'
        f'<div class="date">{markdown.markdown(match.group("date"), extensions=MD_EXTENSIONS)}</div>'
        '<div class="rule"></div>'
        f'<div class="abstract"><div class="label">ABSTRACT</div>'
        f'{markdown.markdown(match.group("abstract"), extensions=MD_EXTENSIONS)}</div>'
        + (markdown.markdown(findings, extensions=MD_EXTENSIONS) if findings else "")
    )
    return polish_text(link_references(html, reference_count)), short_title, metadata


CITATION = re.compile(r"\[(\d+(?:\s*[,\u2013-]\s*\d+)*)\]")


def cite_links(match: re.Match, reference_count: int) -> str:
    parts = re.split(r"(\s*[,\u2013-]\s*)", match.group(1))
    entries = parts[::2]
    numbers = [int(n) for n in entries]
    for index, separator in enumerate(parts[1::2]):
        if "-" in separator or "\u2013" in separator:
            numbers.extend(range(min(int(entries[index]), int(entries[index + 1])),
                                 max(int(entries[index]), int(entries[index + 1])) + 1))
    if not all(1 <= n <= reference_count for n in numbers):
        return match.group(0)
    out = []
    for n, sep in zip(entries, parts[1::2] + [""]):
        out.append(f'<a class="cite" href="#ref-{n}">[{n}]</a>')
        out.append("\u2013" if "\u2013" in sep or "-" in sep else ", " if sep else "")
    return "".join(out)


def reference_count_from_markdown(content: str) -> int:
    match = re.search(r"(?m)^## References\s*$([\s\S]*?)(?=^## |\Z)", content)
    if not match:
        return 0
    return len(re.findall(r"(?m)^\s*\d+[.)]\s+", match.group(1)))


def link_references(html: str, reference_count: int) -> str:
    head, marker, refs = html.partition("<h2>References</h2>")
    pieces, in_code = [], False
    for piece in re.split(r"(<[^>]+>)", head):
        if piece.startswith("<"):
            in_code = piece.startswith(("<code", "<pre", "<a ")) or (
                in_code and piece not in ("</code>", "</pre>", "</a>"))
            pieces.append(piece)
        elif in_code:
            pieces.append(piece)
        else:
            pieces.append(CITATION.sub(lambda m: cite_links(m, reference_count), piece))
    if marker:
        count = iter(range(1, reference_count + 1))
        ref_list, closer, appendix = refs.partition("</ol>")
        ref_list = re.sub(r"<li>", lambda _: f'<li id="ref-{next(count)}">', ref_list)
        refs = ref_list + closer + appendix
    return "".join(pieces) + marker + refs


def count_link_annotations(reader) -> int:
    return sum(
        annotation.get_object().get("/Subtype") == "/Link"
        for page in reader.pages
        for annotation in page.get("/Annots", [])
    )


def set_pdf_metadata(pdf_path: Path, metadata: dict[str, str]) -> None:
    try:
        from pypdf import PdfReader, PdfWriter
    except ImportError:
        print("Warning: pypdf is unavailable; leaving Chromium PDF metadata unchanged.")
        return

    reader = PdfReader(str(pdf_path))
    links_before = count_link_annotations(reader)
    writer = PdfWriter(clone_from=reader)
    writer.add_metadata({
        "/Title": metadata["title"],
        "/Author": metadata["author"],
        "/Subject": metadata["subject"],
        "/Keywords": metadata["keywords"],
        "/Creator": "paper build pipeline",
    })
    with tempfile.NamedTemporaryFile(dir=pdf_path.parent, suffix=".pdf", delete=False) as temp:
        temp_path = Path(temp.name)
    try:
        writer.write(temp_path)
        updated = PdfReader(str(temp_path))
        links_after = count_link_annotations(updated)
        temp_path.replace(pdf_path)
    except Exception:
        temp_path.unlink(missing_ok=True)
        raise
    print(f"PDF link annotations: {links_before} before metadata update, {links_after} after.")
