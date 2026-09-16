"""Stdlib HTTP JSON API + static phone shell for the owner app."""

from __future__ import annotations

import json
import mimetypes
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Optional
from urllib.parse import parse_qs, unquote, urlparse

from jims_mower.app.backend import AppBackend, viewer_manifest
from jims_mower.viewer import static_dir as viewer_static_dir
from jims_mower.yard_profile import YardProfileError, yard_profile_from_dict


def _live_session(backend: AppBackend) -> Any:
    return getattr(backend, "session", None)


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
                if path in {"/viewer", "/viewer/"}:
                    self._serve_viewer_index()
                    return
                if path.startswith("/viewer/"):
                    self._serve_viewer_asset(path[len("/viewer/") :])
                    return
                if path == "/api/manifest":
                    self._send_json(200, viewer_manifest(backend))
                    return
                if self._handle_live_get(path, parsed):
                    return
                if path == "/api/profile":
                    self._send_json(200, backend.get_yard())
                    return
                if path == "/data/profile.json":
                    self._send_json(200, backend.get_yard())
                    return
                if path in {"/data/yard.json", "/data/yard.mesh.json"}:
                    self._send_json(200, backend.mesh())
                    return
                if path == "/status":
                    self._send_json(200, backend.status())
                    return
                if path == "/yard":
                    self._send_json(200, backend.get_yard())
                    return
                if path == "/yards":
                    listing = getattr(backend, "list_yards", None)
                    if listing is None:
                        self._send_json(200, {"yards": [backend.get_yard()], "active": backend.get_yard().get("name")})
                        return
                    self._send_json(200, listing())
                    return
                if path == "/notifications":
                    notes = getattr(backend, "notifications", None)
                    if notes is None:
                        self._send_json(200, {"items": [], "sms": False, "channel": "in_app"})
                        return
                    self._send_json(200, notes())
                    return
                if path == "/ota":
                    ota = getattr(backend, "ota", None)
                    if ota is None:
                        from jims_mower.ota import ota_status

                        self._send_json(200, ota_status())
                        return
                    self._send_json(200, ota())
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
                self._send_json(404, {"error": "not found"})
            except YardProfileError as exc:
                self._send_json(400, {"error": str(exc)})
            except Exception as exc:  # pragma: no cover - defensive
                self._send_json(500, {"error": str(exc)})

        def do_PUT(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            path = parsed.path.rstrip("/") or "/"
            try:
                if path not in {"/yard", "/api/profile"}:
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
                if path == "/api/profile":
                    profile = yard_profile_from_dict(self._read_json())
                    saved = backend.put_yard(profile)
                    self._send_json(200, {"ok": True, "path": "profile.json", "profile": saved})
                    return
                if path in {"/api/live/control", "/api/live/control/"}:
                    self._live_control()
                    return
                if path in {"/yards/select", "/yards/select/"}:
                    body = self._read_json()
                    if not isinstance(body, dict):
                        raise YardProfileError("select body must be a mapping")
                    name = str(body.get("name") or body.get("yard") or "")
                    select = getattr(backend, "select_yard", None)
                    if select is None:
                        raise YardProfileError("this backend does not switch yards")
                    self._send_json(200, select(name))
                    return
                if path != "/command":
                    self._send_json(404, {"error": "not found"})
                    return
                body = self._read_json()
                if not isinstance(body, dict):
                    raise YardProfileError("command body must be a mapping")
                cmd = body.get("cmd") or body.get("command")
                reason = str(body.get("reason") or "")
                extra = {}
                if body.get("pin") is not None or body.get("code") is not None:
                    extra["pin"] = body.get("pin") if body.get("pin") is not None else body.get("code")
                self._send_json(200, backend.command(str(cmd or ""), reason=reason, **extra))
            except YardProfileError as exc:
                self._send_json(400, {"error": str(exc)})
            except Exception as exc:  # pragma: no cover
                self._send_json(500, {"error": str(exc)})

        def _handle_live_get(self, path: str, parsed: Any) -> bool:
            session = _live_session(backend)
            if session is None:
                return False
            if path in {"/api/live", "/api/live/"}:
                self._sse_live(parsed, session)
                return True
            if path == "/api/live/snapshot":
                self._send_json(200, session.snapshot())
                return True
            if path == "/api/live/observed.png":
                self._send_live_bytes(session.observed_png_bytes(), "image/png")
                return True
            if path == "/api/live/fog.png":
                self._send_live_bytes(session.fog_png_bytes(), "image/png")
                return True
            if path == "/api/live/coverage.png":
                payload = session.coverage_png_bytes() if hasattr(session, "coverage_png_bytes") else b""
                self._send_live_bytes(payload, "image/png")
                return True
            if path == "/api/live/areas.png":
                payload = session.areas_png_bytes() if hasattr(session, "areas_png_bytes") else b""
                self._send_live_bytes(payload, "image/png")
                return True
            if path == "/api/live/observed_mesh.json":
                payload = session.observed_mesh_bytes() if hasattr(session, "observed_mesh_bytes") else b""
                self._send_live_bytes(payload, "application/json")
                return True
            if path.startswith("/api/live/cam/"):
                name = path[len("/api/live/cam/") :].split("?")[0]
                self._send_live_bytes(session.camera_bytes(name), "image/jpeg")
                return True
            return False

        def _send_live_bytes(self, payload: bytes, content_type: str) -> None:
            if not payload:
                self._send_json(404, {"error": "live asset not ready"})
                return
            self._send(200, payload, content_type)

        def _live_control(self) -> None:
            session = _live_session(backend)
            if session is None:
                self._send_json(404, {"error": "no live session"})
                return
            body = self._read_json()
            if not isinstance(body, dict):
                raise YardProfileError("command body must be a mapping")
            cmd = str(body.get("cmd") or body.get("command") or "")
            extra = {k: v for k, v in body.items() if k not in {"cmd", "command"}}
            try:
                if hasattr(backend, "live_control"):
                    result = backend.live_control(cmd, **extra)
                else:
                    result = session.control(cmd, **extra)
            except Exception as exc:  # noqa: BLE001 — owner bar must not 500 the phone
                result = {"ok": False, "error": str(exc)}
            self._send_json(200, result)

        def _sse_live(self, parsed: Any, session: Any) -> None:
            qs = parse_qs(parsed.query)
            try:
                limit = int((qs.get("n") or ["0"])[0])
            except ValueError:
                limit = 0
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "close" if limit > 0 else "keep-alive")
            self.end_headers()
            import time

            n = 0
            try:
                while limit <= 0 or n < limit:
                    payload = json.dumps(session.snapshot())
                    self.wfile.write(f"data: {payload}\n\n".encode("utf-8"))
                    self.wfile.flush()
                    n += 1
                    time.sleep(0.04 if limit > 0 else 0.12)
            except (BrokenPipeError, ConnectionResetError):
                return

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

        def _serve_viewer_index(self) -> None:
            dest = viewer_static_dir() / "index.html"
            html = dest.read_text(encoding="utf-8")
            html = html.replace('href="/style.css"', 'href="/viewer/style.css"')
            html = html.replace('src="/app.js"', 'src="/viewer/app.js"')
            self._send(200, html.encode("utf-8"), "text/html; charset=utf-8")

        def _serve_viewer_asset(self, rel: str) -> None:
            rel = unquote(rel).lstrip("/")
            dest = (viewer_static_dir() / rel).resolve()
            try:
                dest.relative_to(viewer_static_dir().resolve())
            except ValueError:
                self._send_json(403, {"error": "forbidden"})
                return
            if not dest.is_file():
                self._send_json(404, {"error": "not found"})
                return
            ctype = mimetypes.guess_type(dest.name)[0] or "application/octet-stream"
            if dest.suffix in {".js", ".mjs"}:
                ctype = "text/javascript; charset=utf-8"
            elif dest.suffix == ".css":
                ctype = "text/css; charset=utf-8"
            self._send(200, dest.read_bytes(), ctype)

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
    import threading
    import time

    httpd = make_server(backend, host=host, port=port, assets=assets)
    stop = threading.Event()

    def _schedule_tick() -> None:
        while not stop.wait(1.0):
            try:
                backend.tick()
            except Exception:
                time.sleep(0.0)

    ticker = threading.Thread(target=_schedule_tick, daemon=True, name="jims-mower-schedule")
    ticker.start()
    try:
        httpd.serve_forever()
    finally:
        stop.set()
        httpd.server_close()
        backend.close()
