"""Patch parsing and application for model-produced edits.

Accepts unified diffs (``--- a/x`` / ``+++ b/x`` / ``@@ -l,s +l,s @@``) and the
``*** Begin Patch`` envelope format (``*** Add File:`` / ``*** Update File:`` /
``*** Delete File:``) that some model families are trained on. Each hunk is applied by
locating its exact old block (context + removed lines); line numbers are only hints used to
disambiguate. A second pass tolerates trailing-whitespace differences. Ambiguous or missing
blocks raise ``PatchError`` with a precise explanation instead of guessing.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


class PatchError(Exception):
    pass


@dataclass
class Hunk:
    old: list[str]
    new: list[str]
    hint_line: int | None = None
    anchor: str | None = None


@dataclass
class FilePatch:
    path: str
    op: str  # add | update | delete
    hunks: list[Hunk] = field(default_factory=list)
    content: list[str] = field(default_factory=list)
    no_newline_at_end: bool = False


_HUNK = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")


def _strip_prefix(path: str) -> str:
    path = path.strip().split("\t")[0]
    if path.startswith(("a/", "b/")):
        return path[2:]
    return path


def parse_patch(text: str) -> list[FilePatch]:
    lines = text.replace("\r\n", "\n").split("\n")
    if any(line.startswith("*** Begin Patch") for line in lines[:5]) or any(
        line.startswith(("*** Update File:", "*** Add File:", "*** Delete File:")) for line in lines
    ):
        return _parse_envelope(lines)
    return _parse_unified(lines)


def _parse_unified(lines: list[str]) -> list[FilePatch]:
    patches: list[FilePatch] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith("--- ") and i + 1 < len(lines) and lines[i + 1].startswith("+++ "):
            old_path = line[4:].strip()
            new_path = lines[i + 1][4:].strip()
            i += 2
            if new_path == "/dev/null":
                fp = FilePatch(_strip_prefix(old_path), "delete")
            elif old_path == "/dev/null":
                fp = FilePatch(_strip_prefix(new_path), "add")
            else:
                fp = FilePatch(_strip_prefix(new_path), "update")
            while i < len(lines) and not (
                lines[i].startswith("--- ") and i + 1 < len(lines) and lines[i + 1].startswith("+++ ")
            ):
                m = _HUNK.match(lines[i])
                if not m:
                    i += 1
                    continue
                hint = int(m.group(1))
                i += 1
                old: list[str] = []
                new: list[str] = []
                while (
                    i < len(lines)
                    and not lines[i].startswith("@@")
                    and not (
                        lines[i].startswith("--- ") and i + 1 < len(lines) and lines[i + 1].startswith("+++ ")
                    )
                ):
                    body = lines[i]
                    if body.startswith("\\"):
                        fp.no_newline_at_end = True
                    elif body.startswith("+"):
                        new.append(body[1:])
                    elif body.startswith("-"):
                        old.append(body[1:])
                    elif body.startswith(" "):
                        old.append(body[1:])
                        new.append(body[1:])
                    elif body == "":
                        # A bare empty line is a blank context line only if more hunk content follows.
                        rest = lines[i + 1 : i + 2]
                        if rest and rest[0][:1] in {" ", "+", "-"}:
                            old.append("")
                            new.append("")
                    i += 1
                if fp.op == "add":
                    fp.content.extend(new)
                else:
                    fp.hunks.append(Hunk(old, new, hint))
            patches.append(fp)
            continue
        i += 1
    if not patches:
        raise PatchError(
            "no file sections found; expected a unified diff (---/+++/@@) or a *** Begin Patch envelope"
        )
    return patches


def _parse_envelope(lines: list[str]) -> list[FilePatch]:
    patches: list[FilePatch] = []
    current: FilePatch | None = None
    hunk: Hunk | None = None
    for line in lines:
        if line.startswith(("*** Begin Patch", "*** End Patch", "*** End of File")):
            continue
        header = re.match(r"^\*\*\* (Add|Update|Delete) File: (.+)$", line)
        if header:
            op = header.group(1).lower()
            current = FilePatch(header.group(2).strip(), op)
            patches.append(current)
            hunk = None
            continue
        if line.startswith("*** Move to:"):
            raise PatchError("file moves are not supported in patches; use run_command with mv")
        if current is None:
            continue
        if current.op == "add":
            if line.startswith("+"):
                current.content.append(line[1:])
            continue
        if current.op == "delete":
            continue
        if line.startswith("@@"):
            hunk = Hunk([], [], None, line[2:].strip() or None)
            current.hunks.append(hunk)
            continue
        if hunk is None:
            hunk = Hunk([], [])
            current.hunks.append(hunk)
        if line.startswith("+"):
            hunk.new.append(line[1:])
        elif line.startswith("-"):
            hunk.old.append(line[1:])
        elif line.startswith(" "):
            hunk.old.append(line[1:])
            hunk.new.append(line[1:])
        elif line == "":
            continue
    if not patches:
        raise PatchError("patch envelope contains no file sections")
    for p in patches:
        p.hunks = [h for h in p.hunks if h.old or h.new]
    return patches


def _find(lines: list[str], block: list[str], *, loose: bool) -> list[int]:
    if not block:
        return []
    norm = (lambda s: s.rstrip()) if loose else (lambda s: s)
    target = [norm(b) for b in block]
    first = target[0]
    hits = []
    for idx in range(len(lines) - len(block) + 1):
        if norm(lines[idx]) == first and [norm(x) for x in lines[idx : idx + len(block)]] == target:
            hits.append(idx)
    return hits


def apply_hunks(original: str, hunks: list[Hunk], *, path: str = "file") -> str:
    newline = "\r\n" if "\r\n" in original else "\n"
    had_final_newline = original.endswith(("\n", "\r\n")) or original == ""
    lines = original.replace("\r\n", "\n").split("\n")
    if lines and lines[-1] == "" and had_final_newline:
        lines.pop()
    offset = 0
    for n, hunk in enumerate(hunks, 1):
        if not hunk.old:
            # Pure insertion: use the anchor or hint line, else append.
            if hunk.anchor:
                anchor_hits = [i for i, line in enumerate(lines) if line.strip() == hunk.anchor.strip()]
                if len(anchor_hits) != 1:
                    raise PatchError(
                        f"{path}: hunk {n} insertion anchor '{hunk.anchor}' matched {len(anchor_hits)} lines"
                    )
                pos = anchor_hits[0] + 1
            elif hunk.hint_line is not None:
                pos = min(len(lines), max(0, hunk.hint_line - 1 + offset))
            else:
                pos = len(lines)
            lines[pos:pos] = hunk.new
            offset += len(hunk.new)
            continue
        hits = _find(lines, hunk.old, loose=False) or _find(lines, hunk.old, loose=True)
        if not hits:
            preview = "\n".join(hunk.old[:6])
            raise PatchError(
                f"{path}: hunk {n} context not found. Expected lines:\n{preview}\n"
                "Re-read the file and regenerate the patch against its current content."
            )
        if len(hits) > 1:
            chosen = None
            if hunk.anchor:
                anchored = [
                    h
                    for h in hits
                    if any(line.strip() == hunk.anchor.strip() for line in lines[max(0, h - 40) : h])
                ]
                if len(anchored) == 1:
                    chosen = anchored[0]
            if chosen is None and hunk.hint_line is not None:
                target = hunk.hint_line - 1 + offset
                ranked = sorted(hits, key=lambda h: abs(h - target))
                if len(ranked) == 1 or abs(ranked[0] - target) < abs(ranked[1] - target):
                    chosen = ranked[0]
            if chosen is None:
                raise PatchError(
                    f"{path}: hunk {n} matches {len(hits)} locations; include more context to make it unique"
                )
            hits = [chosen]
        start = hits[0]
        lines[start : start + len(hunk.old)] = hunk.new
        offset += len(hunk.new) - len(hunk.old)
    text = newline.join(lines)
    if had_final_newline and lines:
        text += newline
    return text
