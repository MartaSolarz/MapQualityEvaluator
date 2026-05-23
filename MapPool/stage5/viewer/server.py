"""HTTP server for the viewer."""

import json
from http.server import HTTPServer, SimpleHTTPRequestHandler
from urllib.parse import parse_qs, urlparse

from .config import IMAGE_DIR, THUMB_DIR
from .data import (
    get_filters_meta, get_candidates, get_stats_data, get_selected_uids,
    add_to_dataset, remove_from_dataset, get_dataset_count, invalidate_caches,
)
from .builder import (
    count_available, preview_batch, add_batch, get_batches,
    undo_batch, get_dataset_summary, export_dataset,
)
from .templates import render_page, render_stats_content
from .templates_builder import render_builder_content, render_history_content
from .thumbs import make_thumbnail


class ViewerHandler(SimpleHTTPRequestHandler):
    def do_GET(self):
        parsed = urlparse(self.path)

        # Serve thumbnail
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

        # Full-size image
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

        params = parse_qs(parsed.query)
        filters = {k: v[0] for k, v in params.items()}
        meta = get_filters_meta()
        selected_uids = get_selected_uids()

        if parsed.path == "/stats":
            stats = get_stats_data(filters)
            sc = render_stats_content(stats, filters, meta)
            html = render_page([], 0, filters, meta, selected_uids, tab="stats", stats_content=sc)

        elif parsed.path == "/builder":
            summary = get_dataset_summary()
            available = count_available({})
            bc = render_builder_content(meta, summary, available)
            html = render_page([], 0, filters, meta, selected_uids, tab="builder", extra_content=bc)

        elif parsed.path == "/history":
            batches = get_batches()
            summary = get_dataset_summary()
            hc = render_history_content(batches, summary)
            html = render_page([], 0, filters, meta, selected_uids, tab="history", extra_content=hc)

        else:
            candidates, total = get_candidates(filters)
            html = render_page(candidates, total, filters, meta, selected_uids, tab="viewer")

        self._send_html(html)

    def do_POST(self):
        parsed = urlparse(self.path)
        body = self._read_json()

        if parsed.path == "/api/dataset":
            uid = body.get("uid")
            action = body.get("action")
            if action == "add":
                add_to_dataset(uid)
            elif action == "remove":
                remove_from_dataset(uid)
            self._send_json({"ok": True, "count": get_dataset_count()})

        elif parsed.path == "/api/builder/count":
            n = count_available(body)
            self._send_json({"count": n})

        elif parsed.path == "/api/builder/preview":
            items = preview_batch(body, n=30)
            self._send_json({"items": items})

        elif parsed.path == "/api/builder/add":
            result = add_batch(body["name"], body["filters"], body["count"])
            self._send_json({"ok": True, **result})

        elif parsed.path == "/api/builder/undo":
            removed = undo_batch(body["batch_id"])
            self._send_json({"ok": True, "removed": removed})

        elif parsed.path == "/api/export":
            from .config import STAGE5_DIR
            path = str(STAGE5_DIR / "dataset_export.csv")
            n = export_dataset(path)
            self._send_json({"ok": True, "count": n, "path": path})

        else:
            self.send_error(404)

    def _read_json(self):
        length = int(self.headers.get("Content-Length", 0))
        return json.loads(self.rfile.read(length)) if length else {}

    def _send_json(self, data):
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())

    def _send_html(self, html):
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()
        self.wfile.write(html.encode("utf-8"))

    def log_message(self, format, *args):
        pass
