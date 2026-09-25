#!/usr/bin/env python3
"""A small but real language server used by Core Main's LSP tests (standard library only).

It speaks the actual wire protocol (``Content-Length`` framed JSON-RPC 2.0 over stdio) and
implements a text-based language service for tiny Python projects: definition, references,
hover, document symbols and push/pull diagnostics. Switches select the behaviours the tests
need, including failure modes:

  --log FILE               append every received message and notable events as JSON lines
  --crash-on METHOD        exit abruptly (code 3, message on stderr) when METHOD arrives
  --crash-times N          ...only for the first N crashes, counted in --state-file across restarts
  --state-file FILE
  --hang-on METHOD         never answer METHOD (a later $/cancelRequest gets RequestCancelled)
  --garbage-on METHOD      answer METHOD with a malformed frame
  --delay-initialize S     sleep S seconds before answering initialize
  --server-requests        send server->client requests and notifications after `initialized`
  --pull-diagnostics       advertise and serve textDocument/diagnostic
  --no-publish             never publish diagnostics
  --sync-none              advertise no text document synchronization
  --location-links         answer definition with LocationLink[]
  --flat-symbols           answer documentSymbol with SymbolInformation[]
  --external-def FILE      definitions of otherwise unknown identifiers point into FILE
  --position-encoding ENC  pick ENC when the client offers it (default: utf-16)
  --ignore-exit            ignore `exit`, stdin EOF and SIGTERM (forces the client's kill fallback)
  --spawn-child            start a long-lived helper process in the server's process group
  --stderr-noise N         write N lines to stderr at startup
  --stdio                  accepted and ignored (built-in server commands pass it)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

WORD = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
DEFINITION = re.compile(
    r"^(?P<indent>[ \t]*)(?P<kw>def|class)[ \t]+(?P<name>[A-Za-z_]\w*)(?P<sig>\([^)]*\))?"
)
LINE_BREAK = re.compile(r"\r\n|\r|\n")


class MethodNotFound(Exception):
    pass


def uri_to_path(uri: str) -> Path:
    return Path(unquote(urlparse(uri).path))


class FakeServer:
    def __init__(self, opts: argparse.Namespace):
        self.opts = opts
        self.stdin = sys.stdin.buffer
        self.stdout = sys.stdout.buffer
        self.docs: dict[str, str] = {}
        self.versions: dict[str, int] = {}
        self.root: Path | None = None
        self.encoding = "utf-16"
        self.shutdown_requested = False
        self.hung: set[Any] = set()

    # ------------------------------------------------------------------ io
    def log(self, entry: dict[str, Any]) -> None:
        if self.opts.log:
            with open(self.opts.log, "a", encoding="utf-8") as fh:
                fh.write(json.dumps({"pid": os.getpid(), **entry}) + "\n")

    def send(self, msg: dict[str, Any]) -> None:
        body = json.dumps(msg).encode("utf-8")
        try:
            self.stdout.write(b"Content-Length: %d\r\n\r\n" % len(body) + body)
            self.stdout.flush()
        except BrokenPipeError:
            os._exit(0)

    def read(self) -> dict[str, Any] | None:
        length = None
        while True:
            line = self.stdin.readline()
            if not line:
                return None
            line = line.strip()
            if not line:
                if length is not None:
                    break
                continue
            name, _, value = line.decode("ascii").partition(":")
            if name.strip().lower() == "content-length":
                length = int(value.strip())
        body = self.stdin.read(length)
        if len(body) < length:
            return None
        return json.loads(body.decode("utf-8"))

    # ------------------------------------------------------------ positions
    def to_units(self, line: str, index: int) -> int:
        prefix = line[:index]
        if self.encoding == "utf-8":
            return len(prefix.encode("utf-8"))
        if self.encoding == "utf-32":
            return len(prefix)
        return len(prefix.encode("utf-16-le")) // 2

    def from_units(self, line: str, units: int) -> int:
        for index in range(len(line) + 1):
            if self.to_units(line, index) >= units:
                return index
        return len(line)

    def rng(self, line_no: int, line: str, start: int, end: int) -> dict[str, Any]:
        return {
            "start": {"line": line_no, "character": self.to_units(line, start)},
            "end": {"line": line_no, "character": self.to_units(line, end)},
        }

    # ---------------------------------------------------------------- files
    def files(self) -> dict[str, str]:
        out: dict[str, str] = {}
        if self.root is not None:
            for dirpath, dirnames, filenames in os.walk(self.root):
                dirnames[:] = sorted(d for d in dirnames if not d.startswith(".") and d != "__pycache__")
                for name in sorted(filenames):
                    if name.endswith(".py"):
                        path = Path(dirpath) / name
                        out[path.as_uri()] = path.read_text(encoding="utf-8", errors="replace")
        out.update(self.docs)
        return dict(sorted(out.items()))

    def text_of(self, uri: str) -> str:
        if uri in self.docs:
            return self.docs[uri]
        return uri_to_path(uri).read_text(encoding="utf-8", errors="replace")

    def word_at(self, uri: str, position: dict[str, Any]) -> str | None:
        lines = LINE_BREAK.split(self.text_of(uri))
        if position["line"] >= len(lines):
            return None
        line = lines[position["line"]]
        index = self.from_units(line, position["character"])
        for m in WORD.finditer(line):
            if m.start() <= index < m.end():
                return m.group()
        return None

    def definitions(self, word: str) -> list[dict[str, Any]]:
        out = []
        for uri, text in self.files().items():
            for n, line in enumerate(LINE_BREAK.split(text)):
                m = DEFINITION.match(line)
                if m and m.group("name") == word:
                    out.append(
                        {"uri": uri, "range": self.rng(n, line, m.start("name"), m.end("name")), "line": line}
                    )
        return out

    def diagnostics(self, text: str) -> list[dict[str, Any]]:
        items = []
        for n, line in enumerate(LINE_BREAK.split(text)):
            for m in re.finditer(r"\bundefined_name\b", line):
                items.append(
                    {
                        "range": self.rng(n, line, m.start(), m.end()),
                        "severity": 1,
                        "code": "E001",
                        "source": "fake",
                        "message": "undefined name 'undefined_name'",
                    }
                )
            if len(line) > 100:
                items.append(
                    {
                        "range": self.rng(n, line, 100, len(line)),
                        "severity": 2,
                        "code": "W501",
                        "source": "fake",
                        "message": f"line too long ({len(line)} > 100)",
                    }
                )
        return items

    def publish(self, uri: str) -> None:
        if self.opts.no_publish:
            return
        self.send(
            {
                "jsonrpc": "2.0",
                "method": "textDocument/publishDiagnostics",
                "params": {
                    "uri": uri,
                    "version": self.versions.get(uri),
                    "diagnostics": self.diagnostics(self.docs[uri]),
                },
            }
        )

    # ------------------------------------------------------------- symbols
    def symbols(self, uri: str) -> list[dict[str, Any]]:
        lines = LINE_BREAK.split(self.text_of(uri))
        found = []
        for n, line in enumerate(lines):
            m = DEFINITION.match(line)
            if m:
                found.append((n, len(m.group("indent")), m))
        roots: list[dict[str, Any]] = []
        stack: list[tuple[int, dict[str, Any]]] = []
        for i, (n, indent, m) in enumerate(found):
            end = len(lines) - 1
            for later_n, later_indent, _ in found[i + 1 :]:
                if later_indent <= indent:
                    end = later_n - 1
                    break
            while stack and stack[-1][0] >= indent:
                stack.pop()
            parent = stack[-1][1] if stack else None
            if m.group("kw") == "class":
                kind = 5
            else:
                kind = 6 if parent is not None and parent["kind"] == 5 else 12
            line_text = lines[n]
            symbol = {
                "name": m.group("name"),
                "detail": m.group("sig") or "",
                "kind": kind,
                "range": {
                    "start": {"line": n, "character": 0},
                    "end": {"line": end, "character": self.to_units(lines[end], len(lines[end]))},
                },
                "selectionRange": self.rng(n, line_text, m.start("name"), m.end("name")),
                "children": [],
                "_container": parent["name"] if parent is not None else None,
            }
            (parent["children"] if parent is not None else roots).append(symbol)
            stack.append((indent, symbol))
        if self.opts.flat_symbols:
            flat: list[dict[str, Any]] = []

            def walk(items: list[dict[str, Any]]) -> None:
                for s in items:
                    info = {
                        "name": s["name"],
                        "kind": s["kind"],
                        "location": {"uri": uri, "range": s["selectionRange"]},
                    }
                    if s["_container"]:
                        info["containerName"] = s["_container"]
                    flat.append(info)
                    walk(s["children"])

            walk(roots)
            return flat

        def clean(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
            return [
                {k: (clean(v) if k == "children" else v) for k, v in s.items() if not k.startswith("_")}
                for s in items
            ]

        return clean(roots)

    # ------------------------------------------------------------ requests
    def handle_request(self, method: str, params: dict[str, Any]) -> Any:
        if method == "initialize":
            return self.initialize(params)
        if method == "shutdown":
            self.shutdown_requested = True
            return None
        if method == "textDocument/definition":
            return self.definition(params)
        if method == "textDocument/references":
            return self.references(params)
        if method == "textDocument/hover":
            return self.hover(params)
        if method == "textDocument/documentSymbol":
            return self.symbols(params["textDocument"]["uri"])
        if method == "textDocument/diagnostic" and self.opts.pull_diagnostics:
            return self.pull(params)
        raise MethodNotFound(method)

    def initialize(self, params: dict[str, Any]) -> dict[str, Any]:
        if self.opts.delay_initialize:
            time.sleep(self.opts.delay_initialize)
        root_uri = params.get("rootUri") or (params.get("workspaceFolders") or [{}])[0].get("uri")
        self.root = uri_to_path(root_uri) if root_uri else None
        offered = ((params.get("capabilities") or {}).get("general") or {}).get("positionEncodings") or [
            "utf-16"
        ]
        if self.opts.position_encoding in offered:
            self.encoding = self.opts.position_encoding
        self.log({"event": "initialize", "encoding": self.encoding, "root": str(self.root)})
        caps: dict[str, Any] = {
            "positionEncoding": self.encoding,
            "definitionProvider": True,
            "referencesProvider": {},
            "hoverProvider": True,
            "documentSymbolProvider": {"label": "fake"},
        }
        if not self.opts.sync_none:
            caps["textDocumentSync"] = {"openClose": True, "change": 1}
        if self.opts.pull_diagnostics:
            caps["diagnosticProvider"] = {
                "identifier": "fake",
                "interFileDependencies": False,
                "workspaceDiagnostics": False,
            }
        return {"capabilities": caps, "serverInfo": {"name": "fake-lsp", "version": "1.0"}}

    def definition(self, params: dict[str, Any]) -> Any:
        word = self.word_at(params["textDocument"]["uri"], params["position"])
        if word is None:
            return None
        found = self.definitions(word)
        if not found and self.opts.external_def:
            path = Path(self.opts.external_def)
            line = path.read_text(encoding="utf-8").splitlines()[0]
            found = [{"uri": path.as_uri(), "range": self.rng(0, line, 0, len(word)), "line": line}]
        if self.opts.location_links:
            return [
                {"targetUri": d["uri"], "targetRange": d["range"], "targetSelectionRange": d["range"]}
                for d in found
            ]
        locations = [{"uri": d["uri"], "range": d["range"]} for d in found]
        return locations[0] if len(locations) == 1 else locations

    def references(self, params: dict[str, Any]) -> list[dict[str, Any]]:
        word = self.word_at(params["textDocument"]["uri"], params["position"])
        if word is None:
            return []
        include_declaration = (params.get("context") or {}).get("includeDeclaration", True)
        out = []
        for uri, text in self.files().items():
            for n, line in enumerate(LINE_BREAK.split(text)):
                definition = DEFINITION.match(line)
                for m in WORD.finditer(line):
                    if m.group() != word:
                        continue
                    if (
                        not include_declaration
                        and definition is not None
                        and definition.group("name") == word
                        and definition.start("name") == m.start()
                    ):
                        continue
                    out.append({"uri": uri, "range": self.rng(n, line, m.start(), m.end())})
        return out

    def hover(self, params: dict[str, Any]) -> Any:
        word = self.word_at(params["textDocument"]["uri"], params["position"])
        found = self.definitions(word) if word else []
        if not found:
            return None
        signature = found[0]["line"].strip().rstrip(":")
        return {
            "contents": {
                "kind": "markdown",
                "value": f"```python\n{signature}\n```\n\nFake documentation for `{word}`.",
            }
        }

    def pull(self, params: dict[str, Any]) -> dict[str, Any]:
        uri = params["textDocument"]["uri"]
        text = self.text_of(uri)
        result_id = f"{self.versions.get(uri, 0)}-{hashlib.sha256(text.encode()).hexdigest()[:8]}"
        if params.get("previousResultId") == result_id:
            return {"kind": "unchanged", "resultId": result_id}
        return {"kind": "full", "resultId": result_id, "items": self.diagnostics(text)}

    # ------------------------------------------------------- notifications
    def handle_notification(self, method: str, params: dict[str, Any]) -> int | None:
        if method == "initialized":
            if self.opts.server_requests:
                self.send_server_requests()
        elif method == "textDocument/didOpen":
            doc = params["textDocument"]
            self.docs[doc["uri"]] = doc["text"]
            self.versions[doc["uri"]] = doc["version"]
            self.publish(doc["uri"])
        elif method == "textDocument/didChange":
            uri = params["textDocument"]["uri"]
            for change in params["contentChanges"]:
                if "range" not in change:
                    self.docs[uri] = change["text"]
            self.versions[uri] = params["textDocument"]["version"]
            self.publish(uri)
        elif method == "textDocument/didClose":
            self.docs.pop(params["textDocument"]["uri"], None)
        elif method == "$/cancelRequest":
            request_id = params.get("id")
            if request_id in self.hung:
                self.hung.discard(request_id)
                self.log({"event": "cancelled", "id": request_id})
                self.send(
                    {"jsonrpc": "2.0", "id": request_id, "error": {"code": -32800, "message": "cancelled"}}
                )
        elif method == "exit":
            if self.opts.ignore_exit:
                self.log({"event": "ignored-exit"})
                return None
            return 0 if self.shutdown_requested else 1
        return None

    def send_server_requests(self) -> None:
        calc = (self.root / "calc.py").as_uri() if self.root else "file:///nonexistent.py"
        edit = {
            "changes": {
                calc: [
                    {
                        "range": {"start": {"line": 0, "character": 0}, "end": {"line": 0, "character": 0}},
                        "newText": "HACKED = True\n",
                    }
                ]
            }
        }
        requests = [
            (
                "cfg-1",
                "workspace/configuration",
                {"items": [{"section": "python"}, {"section": "python.analysis"}]},
            ),
            (
                "reg-1",
                "client/registerCapability",
                {
                    "registrations": [
                        {
                            "id": "watch-1",
                            "method": "workspace/didChangeWatchedFiles",
                            "registerOptions": {"watchers": []},
                        }
                    ]
                },
            ),
            ("progress-1", "window/workDoneProgress/create", {"token": "index-1"}),
            ("folders-1", "workspace/workspaceFolders", None),
            ("edit-1", "workspace/applyEdit", {"label": "inject", "edit": edit}),
            ("unknown-1", "custom/unknownRequest", {}),
        ]
        for request_id, method, params in requests:
            msg: dict[str, Any] = {"jsonrpc": "2.0", "id": request_id, "method": method}
            if params is not None:
                msg["params"] = params
            self.log({"event": "server-request", "id": request_id, "method": method})
            self.send(msg)
        self.send(
            {
                "jsonrpc": "2.0",
                "method": "$/progress",
                "params": {"token": "index-1", "value": {"kind": "begin", "title": "indexing"}},
            }
        )
        self.send(
            {
                "jsonrpc": "2.0",
                "method": "window/logMessage",
                "params": {"type": 1, "message": "fake-lsp: example error log"},
            }
        )
        self.send(
            {
                "jsonrpc": "2.0",
                "method": "$/progress",
                "params": {"token": "index-1", "value": {"kind": "end"}},
            }
        )

    # ---------------------------------------------------------------- loop
    def should_crash(self) -> bool:
        if self.opts.crash_times is None or not self.opts.state_file:
            return True
        state = Path(self.opts.state_file)
        count = int(state.read_text()) if state.exists() else 0
        if count >= self.opts.crash_times:
            return False
        state.write_text(str(count + 1))
        return True

    def linger(self) -> None:
        while True:
            time.sleep(0.5)

    def run(self) -> int:
        if self.opts.ignore_exit:
            signal.signal(signal.SIGTERM, signal.SIG_IGN)
        for n in range(self.opts.stderr_noise):
            sys.stderr.write(f"fake-lsp noise line {n}\n")
        sys.stderr.flush()
        if self.opts.spawn_child:
            child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(300)"])
            self.log({"event": "child", "child_pid": child.pid})
        while True:
            msg = self.read()
            if msg is None:
                self.log({"event": "eof"})
                if self.opts.ignore_exit:
                    self.linger()
                return 0
            self.log({"in": msg})
            method = msg.get("method")
            if method is None:
                continue  # a response to one of our server->client requests (already logged)
            if method == self.opts.crash_on and self.should_crash():
                sys.stderr.write(f"fake-lsp: crashing on {method}\n")
                sys.stderr.flush()
                os._exit(3)
            if method == self.opts.garbage_on:
                self.stdout.write(b"Content-Length: nope\r\n\r\n{}")
                self.stdout.flush()
                continue
            params = msg.get("params") or {}
            if "id" in msg:
                if method == self.opts.hang_on:
                    self.hung.add(msg["id"])
                    continue
                try:
                    result = self.handle_request(method, params)
                except MethodNotFound:
                    self.send(
                        {
                            "jsonrpc": "2.0",
                            "id": msg["id"],
                            "error": {"code": -32601, "message": f"unknown method {method}"},
                        }
                    )
                else:
                    self.send({"jsonrpc": "2.0", "id": msg["id"], "result": result})
            else:
                code = self.handle_notification(method, params)
                if code is not None:
                    self.log({"event": "exit", "code": code})
                    return code


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log")
    parser.add_argument("--crash-on")
    parser.add_argument("--crash-times", type=int)
    parser.add_argument("--state-file")
    parser.add_argument("--hang-on")
    parser.add_argument("--garbage-on")
    parser.add_argument("--delay-initialize", type=float, default=0.0)
    parser.add_argument("--server-requests", action="store_true")
    parser.add_argument("--pull-diagnostics", action="store_true")
    parser.add_argument("--no-publish", action="store_true")
    parser.add_argument("--sync-none", action="store_true")
    parser.add_argument("--location-links", action="store_true")
    parser.add_argument("--flat-symbols", action="store_true")
    parser.add_argument("--external-def")
    parser.add_argument("--position-encoding", default="utf-16")
    parser.add_argument("--ignore-exit", action="store_true")
    parser.add_argument("--spawn-child", action="store_true")
    parser.add_argument("--stderr-noise", type=int, default=0)
    parser.add_argument("--stdio", action="store_true")
    return FakeServer(parser.parse_args()).run()


if __name__ == "__main__":
    sys.exit(main())
