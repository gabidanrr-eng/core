"""HTML → readable text using only the standard-library parser.

Keeps headings (as ``#`` markers), paragraphs, lists, tables (``a | b`` rows), blockquotes, inline
code and ``pre`` blocks (fenced, with the language from ``language-*`` classes). Drops scripts,
styles, navigation, footers, sidebars, hidden elements and common chrome (cookie banners, menus,
breadcrumbs, heading permalinks). When the page marks its content (``<article>``, then ``<main>`` /
``role=main``) and that region holds real text, only the most specific such region is returned.
"""

from __future__ import annotations

import html
import re
from dataclasses import dataclass, field
from html.parser import HTMLParser
from urllib.parse import urljoin, urlsplit

_SKIP_TAGS = frozenset(
    {
        "script",
        "style",
        "noscript",
        "template",
        "head",
        "svg",
        "math",
        "canvas",
        "iframe",
        "object",
        "embed",
        "audio",
        "video",
        "picture",
        "map",
        "nav",
        "footer",
        "aside",
        "button",
        "select",
        "textarea",
        "dialog",
    }
)
_NEVER_SKIP = frozenset({"html", "body", "main", "article"})
_NO_CLASS_FILTER = frozenset(
    {"pre", "code", "p", "h1", "h2", "h3", "h4", "h5", "h6", "table", "thead", "tbody", "tr", "td", "th"}
)
_NOISE_ROLES = frozenset(
    {"navigation", "banner", "contentinfo", "search", "dialog", "menu", "menubar", "toolbar"}
)
_NOISE_TOKENS = frozenset(
    {
        "nav",
        "navbar",
        "navigation",
        "sidebar",
        "sidenav",
        "menu",
        "breadcrumb",
        "breadcrumbs",
        "cookie",
        "cookies",
        "consent",
        "toc",
        "skip",
        "advert",
        "advertisement",
        "ads",
        "share",
        "social",
        "newsletter",
        "popup",
        "modal",
        "footer",
    }
)
_VOID = frozenset(
    {
        "area",
        "base",
        "br",
        "col",
        "embed",
        "hr",
        "img",
        "input",
        "link",
        "meta",
        "param",
        "source",
        "track",
        "wbr",
    }
)
_HEADINGS = frozenset({"h1", "h2", "h3", "h4", "h5", "h6"})
_BLOCKS = frozenset(
    {
        "p",
        "div",
        "section",
        "article",
        "main",
        "header",
        "figure",
        "figcaption",
        "details",
        "summary",
        "dl",
        "dt",
        "dd",
        "table",
        "caption",
        "address",
        "form",
        "fieldset",
        "legend",
        "center",
    }
)
_TOKEN_SPLIT = re.compile(r"[^a-z0-9]+")
_LANG = re.compile(r"(?:^|\s)(?:(?:language|lang|highlight-source|highlight)-|brush:\s*)([A-Za-z0-9_+#.\-]+)")
_PERMALINK_CLASSES = ("headerlink", "hash-link", "header-anchor", "anchor-link", "anchorjs-link", "permalink")
_WS = re.compile(r"\s+")
MIN_MAIN_CHARS = 200
MAX_LINKS = 50


@dataclass
class ExtractedPage:
    title: str | None
    text: str
    links: list[dict[str, str]] = field(default_factory=list)
    used_main: bool = False


@dataclass
class _Open:
    tag: str
    skip: bool
    root: int  # 2 = <article>, 1 = <main>/role=main, 0 = neither


class _Extractor(HTMLParser):
    def __init__(self, base_url: str | None):
        super().__init__(convert_charrefs=True)
        self.base_url = base_url or ""
        self.stack: list[_Open] = []
        self.skip_depth = 0
        self.root_depth = [0, 0, 0]  # open elements per content-root level
        self.blocks: list[tuple[str, str, int]] = []  # (kind, text, innermost content-root level)
        self.buf: list[str] = []
        self.kind = "p"
        self.prefix = ""
        self.lists: list[list[str | int]] = []
        self.quote_depth = 0
        self.pre_depth = 0
        self.pre_buf: list[str] = []
        self.pre_lang = ""
        self.title_parts: list[str] | None = None
        self.title: str | None = None
        self.og_title: str | None = None
        self.first_h1: str | None = None
        self.link_href: str | None = None
        self.link_text: list[str] = []
        self.links: list[tuple[str, str, int]] = []
        self._seen_links: set[str] = set()

    @property
    def level(self) -> int:
        return 2 if self.root_depth[2] else 1 if self.root_depth[1] else 0

    # ----------------------------------------------------------------- parser hooks
    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        a = {k.lower(): (v or "") for k, v in attrs}
        if tag == "title" and self.title is None and not any(e.tag in {"svg", "math"} for e in self.stack):
            self.title_parts = []
        if tag == "meta":
            if a.get("property", "").lower() == "og:title" and a.get("content") and self.og_title is None:
                self.og_title = " ".join(a["content"].split())
            return
        if tag in _VOID:
            if not self.skip_depth and not self.pre_depth:
                if tag == "br":
                    self.buf.append("\n")
                elif tag == "hr":
                    self._flush()
            return
        skip = self._is_noise(tag, a)
        root = 2 if tag == "article" else 1 if tag == "main" or a.get("role", "").lower() == "main" else 0
        if not self.skip_depth and not self.pre_depth and (skip or root or tag in _BLOCKS):
            # A nested block (e.g. <li><p>) must not discard the pending list/heading prefix.
            self._flush(reset=False)
        self.stack.append(_Open(tag, skip, root))
        if skip:
            self.skip_depth += 1
        if root:
            self.root_depth[root] += 1
        if self.skip_depth:
            return
        self._start(tag, a)

    def handle_endtag(self, tag: str) -> None:
        if tag == "title" and self.title_parts is not None:
            self.title = " ".join("".join(self.title_parts).split()) or None
            self.title_parts = None
        if tag in _VOID:
            return
        idx = next((i for i in range(len(self.stack) - 1, -1, -1) if self.stack[i].tag == tag), None)
        if idx is None:
            return
        # Elements left open inside the closed one (e.g. unclosed <li>) are closed implicitly.
        while len(self.stack) > idx:
            el = self.stack.pop()
            suppressed = self.skip_depth > 0
            if el.skip:
                self.skip_depth -= 1
            if not suppressed:
                self._end(el.tag)
            if el.root:
                self.root_depth[el.root] -= 1

    def handle_data(self, data: str) -> None:
        if self.title_parts is not None:
            self.title_parts.append(data)
            return
        if self.skip_depth:
            return
        if self.pre_depth:
            self.pre_buf.append(data)
            return
        text = _WS.sub(" ", data)
        if text == " ":
            if self.buf and not self.buf[-1].endswith((" ", "\n")):
                self.buf.append(" ")
            return
        self.buf.append(text)
        if self.link_href is not None:
            self.link_text.append(text)

    # ------------------------------------------------------------------ semantics
    def _is_noise(self, tag: str, a: dict[str, str]) -> bool:
        if tag in _NEVER_SKIP:
            return False
        if tag in _SKIP_TAGS:
            # An <aside> inside the main content is usually a callout/admonition, not chrome.
            return not (tag == "aside" and self.level > 0)
        if "hidden" in a or a.get("aria-hidden", "").lower() == "true":
            return True
        role = a.get("role", "").lower()
        if role in _NOISE_ROLES or (role == "complementary" and not self.level):
            return True
        style = a.get("style", "").replace(" ", "").lower()
        if "display:none" in style or "visibility:hidden" in style:
            return True
        if tag == "a" and any(c in a.get("class", "").lower() for c in _PERMALINK_CLASSES):
            return True
        if tag in _NO_CLASS_FILTER:
            return False
        tokens = set(_TOKEN_SPLIT.split(f"{a.get('class', '')} {a.get('id', '')}".lower()))
        return bool(tokens & _NOISE_TOKENS)

    def _start(self, tag: str, a: dict[str, str]) -> None:
        if self.pre_depth:
            if tag == "pre":
                self.pre_depth += 1
            elif tag == "code" and not self.pre_lang:
                self.pre_lang = _language(a)
            return
        if tag == "pre":
            self._flush()
            self.pre_depth = 1
            self.pre_buf = []
            self.pre_lang = _language(a)
        elif tag in _HEADINGS:
            self._flush()
            self.kind = "h"
            self.prefix = "#" * int(tag[1]) + " "
        elif tag in {"ul", "ol"}:
            self._flush()
            start = a.get("start", "")
            self.lists.append([tag, int(start) if tag == "ol" and start.isdigit() else 1])
        elif tag == "li":
            self._flush()
            indent = "  " * max(0, len(self.lists) - 1)
            marker = "- "
            if self.lists and self.lists[-1][0] == "ol":
                number = int(self.lists[-1][1])
                self.lists[-1][1] = number + 1
                marker = f"{number}. "
            self.kind = "li"
            self.prefix = indent + marker
        elif tag == "tr":
            self._flush()
            self.kind = "tr"
        elif tag in {"td", "th"}:
            if "".join(self.buf).strip():
                self.buf.append(" | ")
        elif tag == "blockquote":
            self._flush()
            self.quote_depth += 1
        elif tag == "code":
            self.buf.append("`")
        elif tag == "a":
            self.link_href = a.get("href") or None
            self.link_text = []

    def _end(self, tag: str) -> None:
        if self.pre_depth:
            if tag == "pre":
                self.pre_depth -= 1
                if self.pre_depth == 0:
                    self._emit_pre()
            return
        if tag in _HEADINGS or tag in {"li", "tr", "dt", "dd"} or tag in _BLOCKS:
            self._flush()
        elif tag in {"ul", "ol"}:
            self._flush()
            if self.lists:
                self.lists.pop()
        elif tag == "blockquote":
            self._flush()
            self.quote_depth = max(0, self.quote_depth - 1)
        elif tag == "code":
            self.buf.append("`")
        elif tag == "a":
            self._record_link()

    def _flush(self, *, reset: bool = True) -> None:
        raw = "".join(self.buf)
        self.buf = []
        lines = (" ".join(line.split()) for line in raw.split("\n"))
        text = "\n".join(line for line in lines if line).strip()
        if not text or text == "``":
            if reset:
                self.kind, self.prefix = "p", ""
            return
        kind, prefix = self.kind, self.prefix
        self.kind, self.prefix = "p", ""
        if kind == "h" and prefix == "# " and self.first_h1 is None:
            self.first_h1 = text
        text = prefix + text
        if self.quote_depth:
            text = "\n".join("> " + line for line in text.split("\n"))
        self.blocks.append((kind, text, self.level))

    def _emit_pre(self) -> None:
        code = "".join(self.pre_buf).strip("\n").rstrip()
        self.pre_buf = []
        lang, self.pre_lang = self.pre_lang, ""
        if not code.strip():
            return
        fence = "````" if "```" in code else "```"
        self.blocks.append(("pre", f"{fence}{lang}\n{code}\n{fence}", self.level))

    def _record_link(self) -> None:
        href, self.link_href = self.link_href, None
        text = " ".join("".join(self.link_text).split())
        self.link_text = []
        if not href or not text or len(self.links) >= MAX_LINKS:
            return
        absolute = urljoin(self.base_url, href.strip()).split("#", 1)[0]
        try:
            scheme = urlsplit(absolute).scheme
        except ValueError:
            return
        if scheme in {"http", "https"} and absolute not in self._seen_links:
            self._seen_links.add(absolute)
            self.links.append((text[:120], absolute, self.level))

    def result(self) -> ExtractedPage:
        self._flush()
        if self.pre_depth:
            self.pre_depth = 0
            self._emit_pre()
        floor = 0
        for level in (2, 1):
            if sum(len(b[1]) for b in self.blocks if b[2] >= level) >= MIN_MAIN_CHARS:
                floor = level
                break
        chosen = [b for b in self.blocks if b[2] >= floor]
        parts: list[str] = []
        prev = ""
        for kind, text, _ in chosen:
            if parts:
                parts.append("\n" if kind == prev and kind in {"li", "tr"} else "\n\n")
            parts.append(text)
            prev = kind
        links = [{"text": text, "url": url} for text, url, level in self.links if level >= floor]
        return ExtractedPage(
            title=self.title or self.og_title or self.first_h1,
            text=re.sub(r"\n{3,}", "\n\n", "".join(parts)).strip(),
            links=links,
            used_main=floor > 0,
        )


def _language(attrs: dict[str, str]) -> str:
    match = _LANG.search(attrs.get("class", ""))
    return match.group(1).lower() if match else ""


def _strip_tags(markup: str) -> str:
    body = re.sub(r"(?is)<(script|style|noscript|template)\b.*?</\1\s*>", " ", markup)
    body = re.sub(r"(?s)<[^>]*>", " ", body)
    return _WS.sub(" ", html.unescape(body)).strip()


def html_to_text(markup: str, base_url: str | None = None) -> ExtractedPage:
    parser = _Extractor(base_url)
    try:
        parser.feed(markup)
        parser.close()
    except (AssertionError, ValueError):
        # The stdlib parser can still trip on hostile markup; plain tag stripping is the fallback.
        return ExtractedPage(title=None, text=_strip_tags(markup))
    return parser.result()
