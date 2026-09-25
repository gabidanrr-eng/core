"""Local web fixtures for live browser tests: an allowed app server, an off-limits server that
records every request that reaches it, and an allowed origin with nothing listening."""

from __future__ import annotations

import json
import socket
import threading
from collections.abc import Iterator
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def page(title: str, body: str) -> str:
    return f"<!doctype html><html><head><title>{title}</title></head><body>{body}</body></html>"


@dataclass
class Web:
    app: str
    offlimits: str
    dead: str
    hits: list[str] = field(default_factory=list)

    def config(self, *extra_origins: str, extra: str = "") -> str:
        origins = [self.app, self.dead, "file://*", *extra_origins]
        return f"[browser]\nallowed_origins = {json.dumps(origins)}\n{extra}"


def _serve(handler: type[BaseHTTPRequestHandler]) -> ThreadingHTTPServer:
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def _app_handler(offlimits: str) -> type[BaseHTTPRequestHandler]:
    secure_offlimits = offlimits.replace("http://", "https://")
    pages = {
        "/": page(
            "Home",
            "<h1>Welcome home</h1><p id='status'>idle</p>"
            "<button onclick=\"fetch('/api').then(r => r.json()).then(d => "
            "document.getElementById('status').textContent = d.msg)\">Load data</button>"
            f"<a href='{offlimits}/secret'>Go external</a> <a href='/final' target='_blank'>Open tab</a>"
            "<p id='who'></p><script>document.getElementById('who').textContent = "
            "'visitor: ' + (localStorage.getItem('who') || 'none');</script>"
            "<button onclick=\"localStorage.setItem('who', 'first task')\">Remember me</button>",
        ),
        "/final": page("Final", "<h1>Final destination</h1>"),
        "/late": page("Late", "<script>setTimeout(() => document.body.append('Late content'), 400)</script>"),
        "/noisy": page("Noisy", "<h1>Noisy</h1><script>console.error('noisy failure')</script>"),
        "/ws": page(
            "Socket",
            "<h1>Socket</h1><script>const ws = new WebSocket('"
            + offlimits.replace("http://", "ws://")
            + "/socket'); ws.onerror = () => console.error('websocket failed');</script>",
        ),
        "/subresource-out": page(
            "Pixel", f"<h1>Pixel page</h1><img alt='tracker' src='{offlimits}/pixel.png'>"
        ),
    }
    redirects = {
        "/redirect-local": "/final",
        "/redirect-out": f"{offlimits}/secret",
        "/redirect-secure-out": f"{secure_offlimits}/secret",
    }

    class App(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, format: str, *args: object) -> None:
            return

        def _send(
            self, status: int, body: bytes, ctype: str = "text/html; charset=utf-8", **headers: str
        ) -> None:
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            for name, value in headers.items():
                self.send_header(name, value)
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            path = self.path.split("?", 1)[0]
            if path in redirects:
                self._send(302, b"", Location=redirects[path])
            elif path == "/api":
                self._send(200, json.dumps({"msg": "loaded from api"}).encode(), "application/json")
            elif path == "/broken":
                self._send(500, page("Broken", "<h1>Internal error</h1>").encode())
            elif path in pages:
                self._send(200, pages[path].encode())
            else:
                self._send(404, page("Missing", "<h1>Not found</h1>").encode())

    return App


def _recording_handler(hits: list[str]) -> type[BaseHTTPRequestHandler]:
    class OffLimits(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, format: str, *args: object) -> None:
            return

        def do_GET(self) -> None:
            hits.append(self.path)
            body = b"<h1>TOP SECRET OFF-LIMITS</h1>"
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    return OffLimits


@pytest.fixture
def web() -> Iterator[Web]:
    hits: list[str] = []
    offlimits = _serve(_recording_handler(hits))
    off_url = f"http://127.0.0.1:{offlimits.server_address[1]}"
    app = _serve(_app_handler(off_url))
    try:
        yield Web(
            app=f"http://127.0.0.1:{app.server_address[1]}",
            offlimits=off_url,
            dead=f"http://127.0.0.1:{free_port()}",
            hits=hits,
        )
    finally:
        for server in (app, offlimits):
            server.shutdown()
            server.server_close()
