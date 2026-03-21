#!/usr/bin/env python3
"""
Simple local viewer for stage5 AI check results.
Opens in browser: shows each image with its AI classification.

Usage: python3 viewer.py [--port 8050]
"""

import sqlite3
import json
import argparse
from pathlib import Path
from http.server import HTTPServer, SimpleHTTPRequestHandler
from urllib.parse import parse_qs, urlparse
import webbrowser

DB_PATH = Path(__file__).parent / "pipeline.db"
IMAGE_DIR = Path(__file__).parent / "images"


def get_candidates(filter_status=None):
    """Fetch candidates that have been through AI check (or downloaded)."""
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row

    where = "WHERE download_status = 'success'"
    if filter_status and filter_status != "all":
        where += f" AND ai_status = '{filter_status}'"

    rows = conn.execute(f"""
        SELECT uid, url, domain, pred_proba, image_width, image_height,
               local_path, ai_status, ai_is_map, ai_is_statistical_map,
               ai_has_quantitative_data, ai_has_admin_units,
               ai_has_choropleth, ai_has_proportional_circles,
               ai_has_graduated_circles, ai_has_structural_circles,
               ai_confidence, ai_description, passes_f3, passes_f4, passes_all
        FROM candidates
        {where}
        ORDER BY
            CASE ai_status WHEN 'pass' THEN 0 WHEN 'fail' THEN 1 ELSE 2 END,
            pred_proba DESC
    """).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def bool_icon(val):
    if val is None:
        return '<span class="grey">—</span>'
    return '<span class="yes">YES</span>' if val else '<span class="no">NO</span>'


def render_card(c):
    uid = c["uid"]
    local_path = c["local_path"]

    # Resolve image URL for the local server
    if local_path:
        img_name = Path(local_path).name
        img_src = f"/images/{img_name}"
    else:
        img_src = ""

    status = c["ai_status"] or "pending"
    status_class = {"pass": "status-pass", "fail": "status-fail",
                    "error": "status-error", "pending": "status-pending"}.get(status, "")

    f3_detail = ""
    if c["ai_status"] and c["ai_status"] != "pending":
        f3_detail = f"""
        <div class="criteria-detail">
            <span>is_map: {bool_icon(c['ai_is_map'])}</span>
            <span>is_statistical: {bool_icon(c['ai_is_statistical_map'])}</span>
            <span>has_quant_data: {bool_icon(c['ai_has_quantitative_data'])}</span>
            <span>has_admin_units: {bool_icon(c['ai_has_admin_units'])}</span>
        </div>"""

    f4_detail = ""
    if c["ai_status"] and c["ai_status"] != "pending":
        f4_detail = f"""
        <div class="criteria-detail">
            <span>choropleth: {bool_icon(c['ai_has_choropleth'])}</span>
            <span>proportional: {bool_icon(c['ai_has_proportional_circles'])}</span>
            <span>graduated: {bool_icon(c['ai_has_graduated_circles'])}</span>
            <span>structural: {bool_icon(c['ai_has_structural_circles'])}</span>
        </div>"""

    desc = c["ai_description"] or ""
    confidence = c["ai_confidence"] or ""

    return f"""
    <div class="card">
        <div class="card-image">
            <img src="{img_src}" alt="{uid}" loading="lazy"
                 onclick="window.open(this.src, '_blank')">
        </div>
        <div class="card-info">
            <div class="card-header">
                <span class="status {status_class}">{status.upper()}</span>
                <span class="confidence">{confidence}</span>
                <span class="resolution">{c['image_width']}×{c['image_height']}px</span>
                <span class="proba">p={c['pred_proba']:.2f}</span>
            </div>
            <div class="description">{desc}</div>
            <div class="criteria-section">
                <div class="criteria-row">
                    <strong>F3</strong> {bool_icon(c['passes_f3'])}
                    {f3_detail}
                </div>
                <div class="criteria-row">
                    <strong>F4</strong> {bool_icon(c['passes_f4'])}
                    {f4_detail}
                </div>
            </div>
            <div class="meta">
                <span class="domain">{c['domain']}</span>
                <span class="uid">{uid[:12]}…</span>
            </div>
        </div>
    </div>"""


def render_page(candidates, filter_status):
    # Count stats
    total = len(candidates)
    cards_html = "\n".join(render_card(c) for c in candidates)

    # Get counts for filter buttons
    conn = sqlite3.connect(str(DB_PATH))
    counts = {}
    for s in ["all", "pass", "fail", "error", "pending"]:
        w = "WHERE download_status = 'success'"
        if s != "all":
            w += f" AND ai_status = '{s}'"
        counts[s] = conn.execute(f"SELECT COUNT(*) FROM candidates {w}").fetchone()[0]
    conn.close()

    active = lambda s: "active" if s == filter_status else ""

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Stage 5 — AI Check Results</title>
<style>
    * {{ margin: 0; padding: 0; box-sizing: border-box; }}
    body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
           background: #f5f5f5; color: #333; padding: 20px; }}
    h1 {{ margin-bottom: 15px; font-size: 1.4em; }}
    .filters {{ margin-bottom: 20px; display: flex; gap: 8px; flex-wrap: wrap; }}
    .filters a {{ padding: 6px 14px; border-radius: 6px; text-decoration: none;
                  background: #e0e0e0; color: #333; font-size: 0.9em; }}
    .filters a.active {{ background: #333; color: #fff; }}
    .filters a:hover {{ background: #ccc; }}
    .grid {{ display: flex; flex-direction: column; gap: 16px; max-width: 1100px; }}
    .card {{ display: flex; gap: 16px; background: #fff; border-radius: 10px;
             padding: 12px; box-shadow: 0 1px 4px rgba(0,0,0,0.08); }}
    .card-image {{ flex-shrink: 0; width: 300px; min-height: 200px;
                   display: flex; align-items: center; justify-content: center;
                   background: #fafafa; border-radius: 6px; overflow: hidden; }}
    .card-image img {{ max-width: 300px; max-height: 350px; cursor: pointer;
                       object-fit: contain; }}
    .card-info {{ flex: 1; display: flex; flex-direction: column; gap: 8px; }}
    .card-header {{ display: flex; gap: 10px; align-items: center; flex-wrap: wrap; }}
    .status {{ padding: 3px 10px; border-radius: 4px; font-weight: 600;
               font-size: 0.85em; }}
    .status-pass {{ background: #d4edda; color: #155724; }}
    .status-fail {{ background: #f8d7da; color: #721c24; }}
    .status-error {{ background: #fff3cd; color: #856404; }}
    .status-pending {{ background: #e2e3e5; color: #383d41; }}
    .confidence {{ font-size: 0.85em; color: #666; }}
    .resolution {{ font-size: 0.85em; color: #666; }}
    .proba {{ font-size: 0.85em; color: #888; }}
    .description {{ font-style: italic; color: #555; font-size: 0.9em; }}
    .criteria-section {{ display: flex; flex-direction: column; gap: 6px; }}
    .criteria-row {{ display: flex; align-items: baseline; gap: 6px; flex-wrap: wrap; }}
    .criteria-row strong {{ min-width: 25px; }}
    .criteria-detail {{ display: flex; gap: 10px; flex-wrap: wrap;
                        font-size: 0.85em; padding-left: 8px;
                        border-left: 2px solid #eee; margin-left: 4px; }}
    .yes {{ color: #28a745; font-weight: 600; }}
    .no {{ color: #dc3545; font-weight: 600; }}
    .grey {{ color: #aaa; }}
    .meta {{ font-size: 0.8em; color: #999; margin-top: auto; }}
    .domain {{ margin-right: 10px; }}
</style>
</head>
<body>
<h1>Stage 5 — AI Check Results ({total} images)</h1>
<div class="filters">
    <a href="/?filter=all" class="{active('all')}">All ({counts['all']})</a>
    <a href="/?filter=pass" class="{active('pass')}">Pass ({counts['pass']})</a>
    <a href="/?filter=fail" class="{active('fail')}">Fail ({counts['fail']})</a>
    <a href="/?filter=error" class="{active('error')}">Error ({counts['error']})</a>
    <a href="/?filter=pending" class="{active('pending')}">Pending ({counts['pending']})</a>
</div>
<div class="grid">
{cards_html}
</div>
</body>
</html>"""


class ViewerHandler(SimpleHTTPRequestHandler):
    def do_GET(self):
        parsed = urlparse(self.path)

        # Serve images
        if parsed.path.startswith("/images/"):
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

        # Main page
        params = parse_qs(parsed.query)
        filter_status = params.get("filter", ["all"])[0]
        candidates = get_candidates(filter_status)
        html = render_page(candidates, filter_status)

        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()
        self.wfile.write(html.encode("utf-8"))

    def log_message(self, format, *args):
        pass  # suppress request logs


def main():
    parser = argparse.ArgumentParser(description="Stage 5 results viewer")
    parser.add_argument("--port", type=int, default=8050)
    args = parser.parse_args()

    url = f"http://localhost:{args.port}"
    print(f"Opening viewer at {url}")
    print("Press Ctrl+C to stop.\n")

    webbrowser.open(url)
    server = HTTPServer(("localhost", args.port), ViewerHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
        server.server_close()


if __name__ == "__main__":
    main()
