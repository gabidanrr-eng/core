"""Text helpers: truncation that preserves head and tail, token estimates, identifiers."""

from __future__ import annotations

import re

_IDENT_SPLIT = re.compile(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+|\d+")
_WORD = re.compile(r"[A-Za-z_][A-Za-z0-9_]{1,}")


def estimate_tokens(text: str) -> int:
    """Conservative provider-independent token estimate (≈3.6 chars/token for code-heavy text)."""
    if not text:
        return 0
    return max(1, int(len(text) / 3.6) + 1)


def truncate_middle(
    text: str, max_chars: int, *, marker: str = "\n… [{omitted} characters omitted] …\n"
) -> str:
    if len(text) <= max_chars:
        return text
    keep = max(0, max_chars - 64)
    head = keep * 2 // 3
    tail = keep - head
    omitted = len(text) - head - tail
    return text[:head] + marker.format(omitted=omitted) + (text[-tail:] if tail else "")


def split_identifier(word: str) -> list[str]:
    parts: list[str] = []
    for chunk in re.split(r"[_\-.$]+", word):
        parts.extend(p.lower() for p in _IDENT_SPLIT.findall(chunk))
    return [p for p in parts if p]


def identifier_terms(text: str, *, limit: int = 20000) -> str:
    """Expand identifiers (camelCase/snake_case) into searchable words for lexical retrieval."""
    seen: set[str] = set()
    out: list[str] = []
    for match in _WORD.finditer(text[: limit * 8]):
        for part in split_identifier(match.group(0)):
            if len(part) > 1 and part not in seen:
                seen.add(part)
                out.append(part)
                if len(out) >= limit:
                    return " ".join(out)
    return " ".join(out)


STOPWORDS = frozenset(
    [
        "a",
        "an",
        "the",
        "and",
        "or",
        "of",
        "to",
        "in",
        "on",
        "for",
        "with",
        "without",
        "by",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "it",
        "this",
        "that",
        "these",
        "those",
        "how",
        "what",
        "why",
        "where",
        "when",
        "which",
        "who",
        "can",
        "could",
        "should",
        "would",
        "will",
        "please",
        "make",
        "sure",
        "do",
        "does",
        "did",
        "i",
        "we",
        "you",
        "me",
        "my",
        "our",
        "your",
        "us",
        "from",
        "into",
        "at",
        "as",
        "about",
        "all",
        "any",
        "some",
        "there",
        "here",
        "then",
        "than",
        "so",
        "if",
        "not",
        "no",
        "yes",
        "just",
        "use",
        "using",
        "get",
        "set",
        "have",
        "has",
        "had",
        "let",
        "lets",
        "also",
        "more",
        "most",
        "very",
    ]
)


def query_terms(text: str, *, max_terms: int = 24) -> list[str]:
    terms: list[str] = []
    seen: set[str] = set()
    for match in _WORD.finditer(text):
        word = match.group(0)
        candidates = [word.lower(), *split_identifier(word)]
        for cand in candidates:
            if len(cand) < 3 or cand in STOPWORDS or cand in seen:
                continue
            seen.add(cand)
            terms.append(cand)
            if len(terms) >= max_terms:
                return terms
    return terms


def fts_query(terms: list[str]) -> str:
    """Build a safe FTS5 MATCH expression (quoted prefix terms OR'ed together)."""
    safe = [t.replace('"', "") for t in terms if t.strip()]
    return " OR ".join(f'"{t}"*' for t in safe if t)


def one_line(text: str, limit: int = 160) -> str:
    flat = " ".join(text.split())
    return flat if len(flat) <= limit else flat[: limit - 1] + "…"
