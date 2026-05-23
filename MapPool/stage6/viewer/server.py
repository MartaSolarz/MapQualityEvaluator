"""HTTP server for stage 6 viewer."""

import json
from http.server import SimpleHTTPRequestHandler
from urllib.parse import parse_qs, urlparse

from .config import IMAGE_DIR, THUMB_DIR
from .data import get_filters_meta, get_maps, get_stats_data, set_human_override, delete_map, reset_map, invalidate_caches
from .templates import render_page, render_stats_content
from .thumbs import make_thumbnail


class ViewerHandler(SimpleHTTPRequestHandler):
    def do_GET(self):
        parsed = urlparse(self.path)

        if parsed.path.startswith("/images/"):
            img_name = parsed.path.split("/")[-1]
            thumb_path = THUMB_DIR / img_name
            if not thumb_path.exists():
                src_path = IMAGE_DIR / img_name
                if not src_path.exists():
                    self.send_error(404)
                    return
                make_thumbnail(src_path, thumb_path)
            self.send_response(200)
            self.send_header("Content-Type", "image/jpeg")
            self.send_header("Cache-Control", "max-age=86400")
            self.end_headers()
            self.wfile.write(thumb_path.read_bytes())
            return

        if parsed.path.startswith("/full/"):
            img_name = parsed.path.split("/")[-1]
            img_path = IMAGE_DIR / img_name
            if img_path.exists():
                self.send_response(200)
                ext = img_path.suffix.lower()
                ct = {".jpg": "image/jpeg", ".jpeg": "image/jpeg",
                      ".png": "image/png", ".gif": "image/gif",
                      ".webp": "image/webp"}.get(ext, "application/octet-stream")
                self.send_header("Content-Type", ct)
                self.end_headers()
                self.wfile.write(img_path.read_bytes())
            else:
                self.send_error(404)
            return

        filters = {k: v[0] for k, v in parse_qs(parsed.query).items()}
        meta = get_filters_meta()

        if parsed.path == "/stats":
            stats = get_stats_data()
            sc = render_stats_content(stats, meta)
            html = render_page([], 0, filters, meta, tab="stats", extra_content=sc)
        elif parsed.path == "/review":
            if not filters.get("review_criterion"):
                filters["review_criterion"] = "D1"
            if not filters.get("review_show"):
                filters["review_show"] = "violations"
            maps, total = get_maps(filters)
            html = render_page(maps, total, filters, meta, tab="review")
        else:
            maps, total = get_maps(filters)
            html = render_page(maps, total, filters, meta, tab="viewer")

        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()
        self.wfile.write(html.encode("utf-8"))

    def do_POST(self):
        parsed = urlparse(self.path)
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length)) if length else {}

        if parsed.path == "/api/review":
            uid = body["uid"]
            criterion = body["criterion"]
            value = body["value"]
            score, violations = set_human_override(uid, criterion, value)
            self._send_json({"ok": True, "score": score, "violations": violations})
            return

        if parsed.path == "/api/map":
            uid = body["uid"]
            action = body["action"]
            if action == "delete":
                delete_map(uid)
                self._send_json({"ok": True})
            elif action == "rerun":
                reset_map(uid)
                self._send_json({"ok": True})
            else:
                self.send_error(400)
            return

        self.send_error(404)

    def _send_json(self, data):
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())

    def log_message(self, format, *args):
        pass
