"""Run the read-only Boundless Scroll workbench on localhost."""
import argparse
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import re
from urllib.parse import parse_qs, urlsplit

from workbench_state import MEDIA_TYPES, snapshot


ROOT = Path(__file__).resolve().parent
STATIC = {
    "/": ("workbench_ui/index.html", "text/html; charset=utf-8"),
    "/style.css": ("workbench_ui/style.css", "text/css; charset=utf-8"),
    "/app.js": ("workbench_ui/app.js", "text/javascript; charset=utf-8"),
    "/assets/logo.png": ("assets/logo.png", "image/png"),
}


class WorkbenchServer(ThreadingHTTPServer):
    def __init__(self, port, root, engine):
        super().__init__(("127.0.0.1", port), WorkbenchHandler)
        self.root = root
        self.engine = engine
        self.media = {}


class WorkbenchHandler(BaseHTTPRequestHandler):
    def do_HEAD(self):
        self.do_GET()

    def do_GET(self):
        try:
            self.route()
        except (BrokenPipeError, ConnectionResetError):
            return

    def route(self):
        url = urlsplit(self.path)
        if url.path == "/api/state":
            episode = parse_qs(url.query).get("episode", [None])[0]
            try:
                state = snapshot(self.server.root, episode, self.server.engine, self.server.media)
            except (OSError, ValueError, TypeError, KeyError, AttributeError) as exc:
                self.log_error("Cannot read workbench data: %s", exc)
                self.send_error(HTTPStatus.SERVICE_UNAVAILABLE, "Cannot read workbench data")
                return
            data = json.dumps(state, ensure_ascii=False).encode("utf-8")
            self.headers_for(HTTPStatus.OK, "application/json; charset=utf-8", len(data))
            self.end_headers()
            if self.command != "HEAD":
                self.wfile.write(data)
        elif url.path in STATIC:
            name, mime = STATIC[url.path]
            self.send_file(self.server.root / name, mime)
        elif url.path.startswith("/media/") and (path := self.server.media.get(url.path[7:])):
            self.send_file(path, MEDIA_TYPES[path.suffix.lower()], allow_range=True)
        else:
            self.send_error(HTTPStatus.NOT_FOUND)

    def headers_for(self, status, mime, length):
        self.send_response(status)
        self.send_header("Content-Type", mime)
        self.send_header("Content-Length", str(length))
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Cross-Origin-Resource-Policy", "same-origin")
        self.send_header("X-Content-Type-Options", "nosniff")

    def send_file(self, path, mime, *, allow_range=False):
        try:
            if allow_range and path.resolve() != path:
                raise FileNotFoundError(path)
            source = path.open("rb")
        except OSError:
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        with source:
            size = os.fstat(source.fileno()).st_size
            range_header = self.headers.get("Range") if allow_range else None
            start, end = 0, size - 1
            status = HTTPStatus.OK
            if range_header:
                match = re.fullmatch(r"bytes=(\d{0,19})-(\d{0,19})", range_header)
                if match and any(match.groups()):
                    lower, upper = match.groups()
                    if lower:
                        start = int(lower)
                        end = min(end, int(upper)) if upper else end
                    else:
                        start = max(0, size - int(upper))
                    status = HTTPStatus.PARTIAL_CONTENT
                if status != HTTPStatus.PARTIAL_CONTENT or start > end:
                    self.headers_for(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE, mime, 0)
                    self.send_header("Content-Range", f"bytes */{size}")
                    self.end_headers()
                    return
            length = end - start + 1
            self.headers_for(status, mime, length)
            if allow_range:
                self.send_header("Accept-Ranges", "bytes")
            if status == HTTPStatus.PARTIAL_CONTENT:
                self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            self.end_headers()
            if self.command == "HEAD":
                return
            source.seek(start)
            while length > 0:
                chunk = source.read(min(length, 256 * 1024))
                if not chunk:
                    return
                self.wfile.write(chunk)
                length -= len(chunk)

    def log_request(self, code="-", size="-"):
        if str(code).startswith("5"):
            super().log_request(code, size)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=4762)
    parser.add_argument("--comfy-url", default=os.environ.get("COMFYUI_URL", "http://127.0.0.1:8189"))
    args = parser.parse_args()
    with WorkbenchServer(args.port, ROOT, args.comfy_url) as server:
        print(f"Boundless Scroll workbench: http://127.0.0.1:{server.server_port}/", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
