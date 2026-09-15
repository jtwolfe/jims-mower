"""Stdlib HTTP JSON API + static phone shell for the owner app."""

from __future__ import annotations

import json
import mimetypes
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Optional
from urllib.parse import parse_qs, unquote, urlparse

from jims_mower.app.backend import AppBackend
from jims_mower.yard_profile import YardProfileError, yard_profile_from_dict

UX_A_VIEWER = "/static/ux_a/index.html"


def static_dir() -> Path:
    return Path(__file__).resolve().parent / "static"


def _json_bytes(payload: Any) -> bytes:
    return json.dumps(payload).encode("utf-8")


def make_handler(backend: AppBackend, *, assets: Optional[Path] = None) -> type[BaseHTTPRequestHandler]:
    root = Path(assets) if assets is not None else static_dir()

    class AppHandler(BaseHTTPRequestHandler):
        server_version = "jims-mower-app/0.1"

        def log_message(self, fmt: str, *args: Any) -> None:
            return None

        def _send(self, code: int, body: bytes, content_type: str, *, extra: Optional[dict[str, str]] = None) -> None:
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            if extra:
                for key, value in extra.items():
                    self.send_header(key, value)
            self.end_headers()
            if self.command != "HEAD":
                self.wfile.write(body)

        def _send_json(self, code: int, payload: Any) -> None:
            self._send(code, _json_bytes(payload), "application/json; charset=utf-8")

        def _read_json(self) -> Any:
            length = int(self.headers.get("Content-Length") or 0)
            raw = self.rfile.read(length) if length else b"{}"
            if not raw:
                return {}
            try:
                return json.loads(raw.decode("utf-8"))
            except json.JSONDecodeError as exc:
                raise YardProfileError(f"invalid JSON body: {exc}") from exc

        def _serve_static(self, rel: str) -> None:
            rel = unquote(rel).lstrip("/")
            if not rel or rel == "static":
                rel = "index.html"
            if rel.startswith("static/"):
                rel = rel[len("static/") :]
            dest = (root / rel).resolve()
            try:
                dest.relative_to(root.resolve())
            except ValueError:
                self._send_json(403, {"error": "forbidden"})
                return
            if dest.is_dir():
                dest = dest / "index.html"
            if not dest.is_file():
                # UX-A viewer is optional until that wave lands.
                if rel.startswith("ux_a/"):
                    self._send_json(
                        404,
                        {
                            "error": "UX-A mesh viewer is not installed",
                            "hint": "Deep-link /static/ux_a/index.html when the WAVE UX-A viewer lands; do not add a second three.js stack.",
                            "fallback": "/#/map",
                        },
                    )
                    return
                self._send_json(404, {"error": "not found"})
                return
            ctype = mimetypes.guess_type(dest.name)[0] or "application/octet-stream"
            if dest.suffix in {".js", ".mjs"}:
                ctype = "text/javascript; charset=utf-8"
            elif dest.suffix == ".css":
                ctype = "text/css; charset=utf-8"
            elif dest.suffix == ".html":
                ctype = "text/html; charset=utf-8"
            self._send(200, dest.read_bytes(), ctype)

        def do_HEAD(self) -> None:  # noqa: N802
            self.do_GET()

        def do_GET(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            path = parsed.path.rstrip("/") or "/"
            try:
                if path in {"/", "/app", "/index.html"}:
                    self._serve_static("index.html")
                    return
                if path.startswith("/static/"):
                    self._serve_static(path)
                    return
                if path == "/status":
                    self._send_json(200, backend.status())
                    return
                if path == "/yard":
                    self._send_json(200, backend.get_yard())
                    return
                if path == "/map/mesh":
                    self._send_json(200, backend.mesh())
                    return
                if path == "/map/coverage":
                    self._send_json(200, backend.coverage())
                    return
                if path == "/events":
                    qs = parse_qs(parsed.query)
                    raw_n = (qs.get("n") or ["0"])[0]
                    try:
                        limit = int(raw_n)
                    except ValueError:
                        limit = 0
                    self._sse(limit=limit)
                    return
                if path == "/healthz":
                    self._send_json(200, {"ok": True, "app": "jims-mower-app"})
                    return
                if path == "/viewer":
                    # Prefer UX-A when present; otherwise the 2D map shell.
                    ux_a = (root / "ux_a" / "index.html")
                    target = UX_A_VIEWER if ux_a.is_file() else "/#/map"
                    self.send_response(302)
                    self.send_header("Location", target)
                    self.end_headers()
                    return
                self._send_json(404, {"error": "not found"})
            except YardProfileError as exc:
                self._send_json(400, {"error": str(exc)})
            except Exception as exc:  # pragma: no cover - defensive
                self._send_json(500, {"error": str(exc)})

        def do_PUT(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            path = parsed.path.rstrip("/") or "/"
            try:
                if path != "/yard":
                    self._send_json(404, {"error": "not found"})
                    return
                profile = yard_profile_from_dict(self._read_json())
                self._send_json(200, backend.put_yard(profile))
            except YardProfileError as exc:
                self._send_json(400, {"error": str(exc)})
            except Exception as exc:  # pragma: no cover
                self._send_json(500, {"error": str(exc)})

        def do_POST(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            path = parsed.path.rstrip("/") or "/"
            try:
                if path != "/command":
                    self._send_json(404, {"error": "not found"})
                    return
                body = self._read_json()
                if not isinstance(body, dict):
                    raise YardProfileError("command body must be a mapping")
                cmd = body.get("cmd") or body.get("command")
                reason = str(body.get("reason") or "")
                self._send_json(200, backend.command(str(cmd or ""), reason=reason))
            except YardProfileError as exc:
                self._send_json(400, {"error": str(exc)})
            except Exception as exc:  # pragma: no cover
                self._send_json(500, {"error": str(exc)})

        def _sse(self, *, limit: int = 0) -> None:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "close" if limit > 0 else "keep-alive")
            self.end_headers()
            import time

            n = 0
            try:
                while limit <= 0 or n < limit:
                    backend.tick()
                    payload = json.dumps(backend.status())
                    self.wfile.write(f"data: {payload}\n\n".encode("utf-8"))
                    self.wfile.flush()
                    n += 1
                    time.sleep(0.05 if limit > 0 else 0.4)
            except (BrokenPipeError, ConnectionResetError):
                return

    return AppHandler


def make_server(
    backend: AppBackend,
    *,
    host: str = "127.0.0.1",
    port: int = 8765,
    assets: Optional[Path] = None,
) -> ThreadingHTTPServer:
    handler = make_handler(backend, assets=assets)
    return ThreadingHTTPServer((host, int(port)), handler)


def serve_app(
    backend: AppBackend,
    *,
    host: str = "127.0.0.1",
    port: int = 8765,
    assets: Optional[Path] = None,
) -> None:
    httpd = make_server(backend, host=host, port=port, assets=assets)
    try:
        httpd.serve_forever()
    finally:
        httpd.server_close()
        backend.close()
