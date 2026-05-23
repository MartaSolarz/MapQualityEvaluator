#!/usr/bin/env python3
"""
Local viewer for stage5 AI check results.
Supports filtering, sorting, resolution threshold, and dataset selection.

Usage: python3 viewer.py [--port 8050]
"""

import sqlite3
import json
import argparse
from pathlib import Path
from http.server import HTTPServer, SimpleHTTPRequestHandler
from urllib.parse import parse_qs, urlparse, urlencode
import webbrowser

DB_PATH = Path(__file__).parent / "pipeline.db"
IMAGE_DIR = Path("/Volumes/PHD/phd/mappool_data/stage5_downloaded")

PER_PAGE = 30
THUMB_DIR = Path(__file__).parent / "thumbnails"
THUMB_MAX = 400  # max dimension for thumbnails

# In-memory cache for filter metadata (regenerated every 30s)
import time as _time
_meta_cache = {"data": None, "ts": 0}
_stats_cache = {"key": None, "data": None, "ts": 0}
_META_TTL = 30
_STATS_TTL = 60


def _make_thumbnail(src_path, thumb_path):
    """Generate a JPEG thumbnail, cached on local disk."""
    from PIL import Image
    try:
        img = Image.open(src_path)
        if img.mode not in ("RGB", "L"):
            img = img.convert("RGB")
        img.thumbnail((THUMB_MAX, THUMB_MAX), Image.LANCZOS)
        thumb_path.parent.mkdir(parents=True, exist_ok=True)
        img.save(thumb_path, format="JPEG", quality=75)
    except Exception:
        # Fallback: copy original
        import shutil
        shutil.copy2(src_path, thumb_path)


def _init_dataset_table():
    conn = sqlite3.connect(str(DB_PATH))
    conn.execute("""
        CREATE TABLE IF NOT EXISTS dataset_selected (
            uid TEXT PRIMARY KEY,
            url TEXT,
            domain TEXT,
            pred_proba REAL,
            image_width INTEGER,
            image_height INTEGER,
            local_path TEXT,
            ai_status TEXT,
            ai_is_map INTEGER,
            ai_is_statistical_map INTEGER,
            ai_has_quantitative_data INTEGER,
            ai_has_admin_units INTEGER,
            ai_has_choropleth INTEGER,
            ai_has_proportional_circles INTEGER,
            ai_has_graduated_circles INTEGER,
            ai_has_structural_circles INTEGER,
            ai_confidence TEXT,
            ai_description TEXT,
            ai_map_language TEXT,
            passes_f3 INTEGER,
            passes_f4 INTEGER,
            passes_all INTEGER,
            added_at TEXT DEFAULT (datetime('now'))
        )
    """)
    conn.commit()
    conn.close()


def _get_selected_uids():
    conn = sqlite3.connect(str(DB_PATH))
    uids = {r[0] for r in conn.execute("SELECT uid FROM dataset_selected")}
    conn.close()
    return uids


def _add_to_dataset(uid):
    conn = sqlite3.connect(str(DB_PATH))
    conn.execute("""
        INSERT OR IGNORE INTO dataset_selected
            (uid, url, domain, pred_proba, image_width, image_height, local_path,
             ai_status, ai_is_map, ai_is_statistical_map,
             ai_has_quantitative_data, ai_has_admin_units,
             ai_has_choropleth, ai_has_proportional_circles,
             ai_has_graduated_circles, ai_has_structural_circles,
             ai_confidence, ai_description, ai_map_language,
             passes_f3, passes_f4, passes_all)
        SELECT uid, url, domain, pred_proba, image_width, image_height, local_path,
               ai_status, ai_is_map, ai_is_statistical_map,
               ai_has_quantitative_data, ai_has_admin_units,
               ai_has_choropleth, ai_has_proportional_circles,
               ai_has_graduated_circles, ai_has_structural_circles,
               ai_confidence, ai_description, ai_map_language,
               passes_f3, passes_f4, passes_all
        FROM candidates WHERE uid = ?
    """, (uid,))
    conn.commit()
    conn.close()


def _remove_from_dataset(uid):
    conn = sqlite3.connect(str(DB_PATH))
    conn.execute("DELETE FROM dataset_selected WHERE uid = ?", (uid,))
    conn.commit()
    conn.close()


def get_filters_meta():
    now = _time.time()
    if _meta_cache["data"] and now - _meta_cache["ts"] < _META_TTL:
        return _meta_cache["data"]

    conn = sqlite3.connect(str(DB_PATH))
    meta = {}

    # Status
    meta["status"] = {}
    for r in conn.execute(
        "SELECT ai_status, COUNT(*) FROM candidates WHERE download_status='success' GROUP BY ai_status"
    ):
        meta["status"][r[0] or "null"] = r[1]
    meta["status"]["all"] = sum(meta["status"].values())

    # Language
    meta["language"] = {}
    for r in conn.execute(
        "SELECT ai_map_language, COUNT(*) FROM candidates "
        "WHERE download_status='success' AND ai_map_language IS NOT NULL "
        "GROUP BY ai_map_language ORDER BY COUNT(*) DESC"
    ):
        meta["language"][r[0]] = r[1]

    # Confidence
    meta["confidence"] = {}
    for r in conn.execute(
        "SELECT ai_confidence, COUNT(*) FROM candidates "
        "WHERE download_status='success' AND ai_confidence IS NOT NULL "
        "GROUP BY ai_confidence ORDER BY COUNT(*) DESC"
    ):
        meta["confidence"][r[0]] = r[1]

    # Domains
    meta["domains"] = {}
    for r in conn.execute(
        "SELECT domain, COUNT(*) FROM candidates "
        "WHERE download_status='success' GROUP BY domain ORDER BY COUNT(*) DESC LIMIT 50"
    ):
        meta["domains"][r[0]] = r[1]

    # Dataset selected count
    meta["dataset_count"] = conn.execute("SELECT COUNT(*) FROM dataset_selected").fetchone()[0]

    conn.close()
    _meta_cache["data"] = meta
    _meta_cache["ts"] = now
    return meta


def get_candidates(filters):
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row

    conditions = ["download_status = 'success'"]
    params = []

    # Status
    status = filters.get("status", "all")
    if status and status != "all":
        conditions.append("ai_status = ?")
        params.append(status)

    # Language
    lang = filters.get("lang")
    if lang:
        conditions.append("ai_map_language = ?")
        params.append(lang)

    # Confidence
    confidence = filters.get("confidence")
    if confidence:
        conditions.append("ai_confidence = ?")
        params.append(confidence)

    # Methods (checkboxes -- OR logic)
    methods = []
    if filters.get("m_choro"):
        methods.append("ai_has_choropleth = 1")
    if filters.get("m_prop"):
        methods.append("ai_has_proportional_circles = 1")
    if filters.get("m_grad"):
        methods.append("ai_has_graduated_circles = 1")
    if filters.get("m_struct"):
        methods.append("ai_has_structural_circles = 1")
    if methods:
        conditions.extend(methods)

    # Domain
    domain = filters.get("domain")
    if domain:
        conditions.append("domain = ?")
        params.append(domain)

    # F3 / F4
    f3 = filters.get("f3")
    if f3 == "1":
        conditions.append("passes_f3 = 1")
    elif f3 == "0":
        conditions.append("passes_f3 = 0")

    f4 = filters.get("f4")
    if f4 == "1":
        conditions.append("passes_f4 = 1")
    elif f4 == "0":
        conditions.append("passes_f4 = 0")

    # Min resolution
    min_w = filters.get("min_w")
    if min_w and min_w.isdigit():
        conditions.append("image_width >= ?")
        params.append(int(min_w))
    min_h = filters.get("min_h")
    if min_h and min_h.isdigit():
        conditions.append("image_height >= ?")
        params.append(int(min_h))

    # Search
    search = filters.get("search")
    if search:
        conditions.append("ai_description LIKE ?")
        params.append(f"%{search}%")

    # Show only selected
    only_selected = filters.get("selected")
    if only_selected == "1":
        conditions.append("uid IN (SELECT uid FROM dataset_selected)")

    where = "WHERE " + " AND ".join(conditions)

    # Sort
    sort = filters.get("sort", "status")
    order = {
        "status": "CASE ai_status WHEN 'pass' THEN 0 WHEN 'fail' THEN 1 ELSE 2 END, pred_proba DESC",
        "proba_desc": "pred_proba DESC",
        "proba_asc": "pred_proba ASC",
        "res_desc": "image_width * image_height DESC",
        "res_asc": "image_width * image_height ASC",
        "domain": "domain ASC, pred_proba DESC",
        "lang": "ai_map_language ASC, pred_proba DESC",
    }.get(sort, "pred_proba DESC")

    # Pagination
    page = int(filters.get("page", 1))
    offset = (page - 1) * PER_PAGE

    total = conn.execute(f"SELECT COUNT(*) FROM candidates {where}", params).fetchone()[0]

    rows = conn.execute(f"""
        SELECT uid, url, domain, pred_proba, image_width, image_height,
               local_path, ai_status, ai_is_map, ai_is_statistical_map,
               ai_has_quantitative_data, ai_has_admin_units,
               ai_has_choropleth, ai_has_proportional_circles,
               ai_has_graduated_circles, ai_has_structural_circles,
               ai_confidence, ai_description, ai_map_language,
               passes_f3, passes_f4, passes_all
        FROM candidates
        {where}
        ORDER BY {order}
        LIMIT {PER_PAGE} OFFSET {offset}
    """, params).fetchall()
    conn.close()
    return [dict(r) for r in rows], total


def bool_icon(val):
    if val is None:
        return '<span class="grey">—</span>'
    return '<span class="yes">YES</span>' if val else '<span class="no">NO</span>'


def render_card(c, is_selected):
    uid = c["uid"]
    local_path = c["local_path"]
    img_src = f"/images/{Path(local_path).name}" if local_path else ""

    status = c["ai_status"] or "pending"
    status_class = {"pass": "status-pass", "fail": "status-fail",
                    "error": "status-error", "pending": "status-pending"}.get(status, "")

    lang = c.get("ai_map_language") or ""
    lang_badge = f'<span class="lang-badge">{lang}</span>' if lang else ""

    f3_detail = ""
    if c["ai_status"] and c["ai_status"] != "pending":
        f3_detail = f"""
        <div class="criteria-detail">
            <span>is_map: {bool_icon(c['ai_is_map'])}</span>
            <span>statistical: {bool_icon(c['ai_is_statistical_map'])}</span>
            <span>quant: {bool_icon(c['ai_has_quantitative_data'])}</span>
            <span>admin: {bool_icon(c['ai_has_admin_units'])}</span>
        </div>"""

    f4_detail = ""
    if c["ai_status"] and c["ai_status"] != "pending":
        f4_detail = f"""
        <div class="criteria-detail">
            <span>choro: {bool_icon(c['ai_has_choropleth'])}</span>
            <span>prop: {bool_icon(c['ai_has_proportional_circles'])}</span>
            <span>grad: {bool_icon(c['ai_has_graduated_circles'])}</span>
            <span>struct: {bool_icon(c['ai_has_structural_circles'])}</span>
        </div>"""

    desc = c["ai_description"] or ""
    confidence = c["ai_confidence"] or ""
    w = c["image_width"] or 0
    h = c["image_height"] or 0

    if is_selected:
        ds_btn = f'<button class="ds-btn ds-remove" onclick="dsAction(\'{uid}\', \'remove\')">Remove from dataset</button>'
    else:
        ds_btn = f'<button class="ds-btn ds-add" onclick="dsAction(\'{uid}\', \'add\')">Add to dataset</button>'

    selected_class = " card-selected" if is_selected else ""

    return f"""
    <div class="card{selected_class}" id="card-{uid}">
        <div class="card-image">
            <img src="{img_src}" alt="{uid}" loading="lazy"
                 onclick="window.open('/full/' + this.src.split('/').pop(), '_blank')">
        </div>
        <div class="card-info">
            <div class="card-header">
                <span class="status {status_class}">{status.upper()}</span>
                <span class="confidence">{confidence}</span>
                {lang_badge}
                <span class="resolution">{w}x{h}px</span>
                <span class="proba">p={c['pred_proba']:.2f}</span>
            </div>
            <div class="description">{desc}</div>
            <div class="criteria-section">
                <div class="criteria-row">
                    <strong>F3</strong> {bool_icon(c['passes_f3'])} {f3_detail}
                </div>
                <div class="criteria-row">
                    <strong>F4</strong> {bool_icon(c['passes_f4'])} {f4_detail}
                </div>
            </div>
            <div class="card-bottom">
                <div class="meta">
                    <a href="/?domain={c['domain']}" class="domain">{c['domain']}</a>
                    <span class="uid">{uid[:12]}...</span>
                    <a href="{c['url']}" target="_blank" class="src-link">source</a>
                </div>
                {ds_btn}
            </div>
        </div>
    </div>"""


def get_stats_data(filters):
    """Get aggregated stats for the stats page, respecting active filters."""
    cache_key = json.dumps(filters, sort_keys=True)
    now = _time.time()
    if _stats_cache["key"] == cache_key and _stats_cache["data"] and now - _stats_cache["ts"] < _STATS_TTL:
        return _stats_cache["data"]

    conn = sqlite3.connect(str(DB_PATH))
    conditions = ["download_status = 'success'"]
    params = []

    status = filters.get("status", "all")
    if status and status != "all":
        conditions.append("ai_status = ?")
        params.append(status)
    lang = filters.get("lang")
    if lang:
        conditions.append("ai_map_language = ?")
        params.append(lang)
    confidence = filters.get("confidence")
    if confidence:
        conditions.append("ai_confidence = ?")
        params.append(confidence)
    domain = filters.get("domain")
    if domain:
        conditions.append("domain = ?")
        params.append(domain)
    if filters.get("selected") == "1":
        conditions.append("uid IN (SELECT uid FROM dataset_selected)")
    f3 = filters.get("f3")
    if f3 == "1":
        conditions.append("passes_f3 = 1")
    elif f3 == "0":
        conditions.append("passes_f3 = 0")
    f4 = filters.get("f4")
    if f4 == "1":
        conditions.append("passes_f4 = 1")
    elif f4 == "0":
        conditions.append("passes_f4 = 0")

    where = "WHERE " + " AND ".join(conditions)

    stats = {}
    stats["total"] = conn.execute(f"SELECT COUNT(*) FROM candidates {where}", params).fetchone()[0]

    # Status breakdown
    stats["by_status"] = {}
    for r in conn.execute(f"SELECT ai_status, COUNT(*) FROM candidates {where} GROUP BY ai_status ORDER BY COUNT(*) DESC", params):
        stats["by_status"][r[0] or "null"] = r[1]

    # Language breakdown
    stats["by_language"] = {}
    for r in conn.execute(f"SELECT ai_map_language, COUNT(*) FROM candidates {where} AND ai_map_language IS NOT NULL GROUP BY ai_map_language ORDER BY COUNT(*) DESC", params):
        stats["by_language"][r[0]] = r[1]

    # Confidence
    stats["by_confidence"] = {}
    for r in conn.execute(f"SELECT ai_confidence, COUNT(*) FROM candidates {where} AND ai_confidence IS NOT NULL GROUP BY ai_confidence ORDER BY COUNT(*) DESC", params):
        stats["by_confidence"][r[0]] = r[1]

    # Methods (not mutually exclusive)
    stats["methods"] = {}
    for col, label in [("ai_has_choropleth", "Choropleth"), ("ai_has_proportional_circles", "Proportional"),
                       ("ai_has_graduated_circles", "Graduated"), ("ai_has_structural_circles", "Structural")]:
        n = conn.execute(f"SELECT COUNT(*) FROM candidates {where} AND {col}=1", params).fetchone()[0]
        stats["methods"][label] = n

    # Top 15 domains
    stats["top_domains"] = []
    for r in conn.execute(f"SELECT domain, COUNT(*) FROM candidates {where} GROUP BY domain ORDER BY COUNT(*) DESC LIMIT 15", params):
        stats["top_domains"].append((r[0], r[1]))

    # Resolution distribution
    stats["resolution"] = {}
    for label, cond in [("<400px", "image_max_dim < 400"),
                        ("400-600", "image_max_dim BETWEEN 400 AND 599"),
                        ("600-800", "image_max_dim BETWEEN 600 AND 799"),
                        ("800-1200", "image_max_dim BETWEEN 800 AND 1199"),
                        ("1200+", "image_max_dim >= 1200")]:
        n = conn.execute(f"SELECT COUNT(*) FROM candidates {where} AND {cond}", params).fetchone()[0]
        stats["resolution"][label] = n

    # F3 sub-criteria breakdown (among checked)
    stats["f3_detail"] = {}
    for col, label in [("ai_is_map", "is_map"), ("ai_is_statistical_map", "is_statistical"),
                       ("ai_has_quantitative_data", "has_quant_data"), ("ai_has_admin_units", "has_admin_units")]:
        yes = conn.execute(f"SELECT COUNT(*) FROM candidates {where} AND {col}=1", params).fetchone()[0]
        no = conn.execute(f"SELECT COUNT(*) FROM candidates {where} AND {col}=0", params).fetchone()[0]
        stats["f3_detail"][label] = {"yes": yes, "no": no}

    conn.close()
    _stats_cache["key"] = cache_key
    _stats_cache["data"] = stats
    _stats_cache["ts"] = now
    return stats


def render_stats_content(stats, filters, meta):
    """Render just the stats content (charts + summary cards)."""

    # Chart data
    status_labels = json.dumps(list(stats["by_status"].keys()))
    status_values = json.dumps(list(stats["by_status"].values()))
    status_colors = json.dumps(["#4caf50" if k == "pass" else "#ef5350" if k == "fail" else "#ffc107" if k == "error" else "#9e9e9e" for k in stats["by_status"]])

    lang_data = dict(list(stats["by_language"].items())[:12])
    lang_labels = json.dumps(list(lang_data.keys()))
    lang_values = json.dumps(list(lang_data.values()))

    method_labels = json.dumps(list(stats["methods"].keys()))
    method_values = json.dumps(list(stats["methods"].values()))

    conf_labels = json.dumps(list(stats["by_confidence"].keys()))
    conf_values = json.dumps(list(stats["by_confidence"].values()))
    conf_colors = json.dumps(["#4caf50" if k == "high" else "#ffc107" if k == "medium" else "#ef5350" for k in stats["by_confidence"]])

    res_labels = json.dumps(list(stats["resolution"].keys()))
    res_values = json.dumps(list(stats["resolution"].values()))

    dom_labels = json.dumps([d[0][:25] for d in stats["top_domains"]])
    dom_values = json.dumps([d[1] for d in stats["top_domains"]])

    # F3 detail table
    f3_rows = ""
    for label, vals in stats["f3_detail"].items():
        total = vals["yes"] + vals["no"]
        pct = vals["yes"] / total * 100 if total > 0 else 0
        f3_rows += f"""<tr>
            <td>{label}</td>
            <td style="color:#2e7d32;font-weight:600">{vals["yes"]:,}</td>
            <td style="color:#c62828;font-weight:600">{vals["no"]:,}</td>
            <td>{pct:.1f}%</td>
            <td><div class="bar"><div class="bar-fill" style="width:{pct}%"></div></div></td>
        </tr>"""

    return f"""
<script src="https://cdn.jsdelivr.net/npm/chart.js@4"></script>

<style>
    .stats-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(340px, 1fr)); gap: 20px; }}
    .chart-card {{ background: #fff; border-radius: 12px; padding: 20px;
                   box-shadow: 0 1px 4px rgba(0,0,0,0.06); }}
    .chart-card h3 {{ font-size: 0.9em; font-weight: 600; color: #555; margin-bottom: 14px;
                      text-transform: uppercase; letter-spacing: 0.5px; }}
    .chart-card canvas {{ max-height: 260px; }}
    .chart-card-wide {{ grid-column: 1 / -1; }}
    .summary-cards {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
                      gap: 14px; margin-bottom: 24px; }}
    .summary-card {{ background: #fff; border-radius: 12px; padding: 16px 20px; text-align: center;
                     box-shadow: 0 1px 4px rgba(0,0,0,0.06); }}
    .summary-card .num {{ font-size: 1.8em; font-weight: 700; color: #1a1a2e; }}
    .summary-card .lbl {{ font-size: 0.75em; color: #8892a4; text-transform: uppercase;
                          letter-spacing: 0.5px; margin-top: 4px; }}
    .stats-table {{ width: 100%; border-collapse: collapse; font-size: 0.85em; }}
    .stats-table th {{ text-align: left; padding: 8px 10px; color: #8892a4; font-weight: 600;
          font-size: 0.75em; text-transform: uppercase; border-bottom: 2px solid #f0f2f5; }}
    .stats-table td {{ padding: 8px 10px; border-bottom: 1px solid #f5f6f8; }}
    .bar {{ width: 100%; height: 8px; background: #f0f2f5; border-radius: 4px; overflow: hidden; }}
    .bar-fill {{ height: 100%; background: #5c6bc0; border-radius: 4px; }}
</style>

<div class="summary-cards">
    <div class="summary-card">
        <div class="num">{stats["total"]:,}</div>
        <div class="lbl">Total checked</div>
    </div>
    <div class="summary-card">
        <div class="num" style="color:#2e7d32">{stats["by_status"].get("pass", 0):,}</div>
        <div class="lbl">Pass</div>
    </div>
    <div class="summary-card">
        <div class="num" style="color:#c62828">{stats["by_status"].get("fail", 0):,}</div>
        <div class="lbl">Fail</div>
    </div>
    <div class="summary-card">
        <div class="num">{len(stats["by_language"]):,}</div>
        <div class="lbl">Languages</div>
    </div>
    <div class="summary-card">
        <div class="num" style="color:#5c6bc0">{meta["dataset_count"]:,}</div>
        <div class="lbl">In dataset</div>
    </div>
</div>

<div class="stats-grid">
    <div class="chart-card">
        <h3>Status</h3>
        <canvas id="chartStatus"></canvas>
    </div>
    <div class="chart-card">
        <h3>Confidence</h3>
        <canvas id="chartConf"></canvas>
    </div>
    <div class="chart-card">
        <h3>Methods</h3>
        <canvas id="chartMethods"></canvas>
    </div>
    <div class="chart-card">
        <h3>Language (top 12)</h3>
        <canvas id="chartLang"></canvas>
    </div>
    <div class="chart-card">
        <h3>Resolution (max dimension)</h3>
        <canvas id="chartRes"></canvas>
    </div>
    <div class="chart-card">
        <h3>F3 Sub-criteria</h3>
        <table class="stats-table">
            <tr><th>Criterion</th><th>Yes</th><th>No</th><th>Rate</th><th></th></tr>
            {f3_rows}
        </table>
    </div>
    <div class="chart-card chart-card-wide">
        <h3>Top domains</h3>
        <canvas id="chartDomains" style="max-height:300px"></canvas>
    </div>
</div>

<script>
const doughnutOpts = {{ responsive: true, plugins: {{ legend: {{ position: 'right', labels: {{ font: {{ size: 11 }} }} }} }} }};
const barOpts = {{ responsive: true, indexAxis: 'y', plugins: {{ legend: {{ display: false }} }},
    scales: {{ x: {{ beginAtZero: true }}, y: {{ ticks: {{ font: {{ size: 11 }} }} }} }} }};

new Chart(document.getElementById('chartStatus'), {{
    type: 'doughnut',
    data: {{ labels: {status_labels}, datasets: [{{ data: {status_values}, backgroundColor: {status_colors} }}] }},
    options: doughnutOpts
}});
new Chart(document.getElementById('chartConf'), {{
    type: 'doughnut',
    data: {{ labels: {conf_labels}, datasets: [{{ data: {conf_values}, backgroundColor: {conf_colors} }}] }},
    options: doughnutOpts
}});
new Chart(document.getElementById('chartMethods'), {{
    type: 'bar',
    data: {{ labels: {method_labels}, datasets: [{{ data: {method_values}, backgroundColor: ['#5c6bc0','#42a5f5','#66bb6a','#ffa726'] }}] }},
    options: {{ responsive: true, plugins: {{ legend: {{ display: false }} }}, scales: {{ y: {{ beginAtZero: true }} }} }}
}});
new Chart(document.getElementById('chartLang'), {{
    type: 'bar',
    data: {{ labels: {lang_labels}, datasets: [{{ data: {lang_values}, backgroundColor: '#5c6bc0' }}] }},
    options: barOpts
}});
new Chart(document.getElementById('chartRes'), {{
    type: 'bar',
    data: {{ labels: {res_labels}, datasets: [{{ data: {res_values}, backgroundColor: '#78909c' }}] }},
    options: {{ responsive: true, plugins: {{ legend: {{ display: false }} }}, scales: {{ y: {{ beginAtZero: true }} }} }}
}});
new Chart(document.getElementById('chartDomains'), {{
    type: 'bar',
    data: {{ labels: {dom_labels}, datasets: [{{ data: {dom_values}, backgroundColor: '#5c6bc0' }}] }},
    options: barOpts
}});
</script>"""


def _keep_filters(filters):
    """Build query string preserving shared filters (for tab switching)."""
    shared = {k: v for k, v in filters.items()
              if k in ("status", "lang", "confidence", "domain", "f3", "f4", "selected") and v}
    return "?" + urlencode(shared) if shared else ""


def build_url(filters, **overrides):
    p = {k: v for k, v in filters.items() if v and k != "page"}
    p.update(overrides)
    p = {k: v for k, v in p.items() if v and v != "all" and v != ""}
    return "/?" + urlencode(p) if p else "/"


def render_page(candidates, total, filters, meta, selected_uids, tab="viewer", stats_content=""):
    cards_html = "\n".join(render_card(c, c["uid"] in selected_uids) for c in candidates) if candidates else ""

    page = int(filters.get("page", 1))
    total_pages = max(1, (total + PER_PAGE - 1) // PER_PAGE)

    status = filters.get("status", "all")
    cur_selected = filters.get("selected", "")

    # Language dropdown
    cur_lang = filters.get("lang", "")
    lang_opts = '<option value="">All languages</option>'
    for code, cnt in sorted(meta["language"].items(), key=lambda x: -x[1]):
        sel = "selected" if code == cur_lang else ""
        lang_opts += f'<option value="{code}" {sel}>{code} ({cnt})</option>'

    # Confidence
    cur_conf = filters.get("confidence", "")
    conf_opts = '<option value="">All</option>'
    for c, cnt in meta["confidence"].items():
        sel = "selected" if c == cur_conf else ""
        conf_opts += f'<option value="{c}" {sel}>{c} ({cnt})</option>'

    # Domain
    cur_dom = filters.get("domain", "")
    dom_opts = '<option value="">All domains</option>'
    for d, cnt in meta["domains"].items():
        sel = "selected" if d == cur_dom else ""
        dom_opts += f'<option value="{d}" {sel}>{d} ({cnt})</option>'

    # F3/F4
    cur_f3 = filters.get("f3", "")
    cur_f4 = filters.get("f4", "")

    # Sort
    cur_sort = filters.get("sort", "status")
    sort_opts = ""
    for val, label in [("status", "Status"), ("proba_desc", "Proba high-low"),
                       ("proba_asc", "Proba low-high"), ("res_desc", "Resolution high-low"),
                       ("res_asc", "Resolution low-high"), ("domain", "Domain"), ("lang", "Language")]:
        sel = "selected" if val == cur_sort else ""
        sort_opts += f'<option value="{val}" {sel}>{label}</option>'

    # Methods checkboxes
    m_choro = "checked" if filters.get("m_choro") else ""
    m_prop = "checked" if filters.get("m_prop") else ""
    m_grad = "checked" if filters.get("m_grad") else ""
    m_struct = "checked" if filters.get("m_struct") else ""

    # Resolution
    min_w = filters.get("min_w", "")
    min_h = filters.get("min_h", "")

    # Search
    cur_search = filters.get("search", "")

    # Tab-dependent parts
    is_viewer = tab == "viewer"
    tab_viewer_class = "active" if is_viewer else ""
    tab_stats_class = "active" if not is_viewer else ""
    form_action = "/" if is_viewer else "/stats"
    reset_href = "/" if is_viewer else "/stats"
    ds_href = "/?selected=1" if is_viewer else "/stats?selected=1"
    ds_active = "active" if cur_selected == "1" else ""
    ds_checked = "checked" if cur_selected == "1" else ""
    tab_filters_qs = _keep_filters(filters)

    viewer_row2 = ""
    if is_viewer:
        viewer_row2 = f"""<div class="filter-row">
    <div class="fg">
        <label>Methods</label>
        <div class="methods-group">
            <label><input type="checkbox" name="m_choro" value="1" {m_choro}> Choropleth</label>
            <label><input type="checkbox" name="m_prop" value="1" {m_prop}> Proportional</label>
            <label><input type="checkbox" name="m_grad" value="1" {m_grad}> Graduated</label>
            <label><input type="checkbox" name="m_struct" value="1" {m_struct}> Structural</label>
        </div>
    </div>
    <div class="fg">
        <label>Min width</label>
        <input type="number" name="min_w" value="{min_w}" placeholder="px">
    </div>
    <div class="fg">
        <label>Min height</label>
        <input type="number" name="min_h" value="{min_h}" placeholder="px">
    </div>
    <div class="fg">
        <label>Search description</label>
        <input type="text" name="search" value="{cur_search}" placeholder="e.g. election, GDP...">
    </div>
    </div>"""

    # Pagination
    pagination = ""
    if total_pages > 1:
        parts = []
        if page > 1:
            parts.append(f'<a href="{build_url(filters, page=str(page-1))}">Prev</a>')
        parts.append(f'<span class="page-info">Page {page}/{total_pages} ({total} results)</span>')
        if page < total_pages:
            parts.append(f'<a href="{build_url(filters, page=str(page+1))}">Next</a>')
        pagination = '<div class="pagination">' + " ".join(parts) + '</div>'

    viewer_content = ""
    if is_viewer:
        viewer_content = f"""{pagination}
<div class="grid">
{cards_html}
</div>
{pagination}"""

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Stage 5 — Map Quality Viewer</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
<style>
    * {{ margin: 0; padding: 0; box-sizing: border-box; }}
    body {{ font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
           background: #f0f2f5; color: #1a1a2e; padding: 0; }}

    /* Header */
    .header {{ background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
               padding: 20px 32px; color: #fff; display: flex;
               justify-content: space-between; align-items: center;
               box-shadow: 0 2px 8px rgba(0,0,0,0.15); }}
    .header h1 {{ font-size: 1.25em; font-weight: 600; letter-spacing: -0.02em; }}
    .header-stats {{ display: flex; gap: 20px; font-size: 0.85em; opacity: 0.8; }}
    .header-stats span {{ display: flex; align-items: center; gap: 5px; }}
    .header-stats .dot {{ width: 8px; height: 8px; border-radius: 50%; display: inline-block; }}
    .dot-pass {{ background: #4caf50; }}
    .dot-fail {{ background: #ef5350; }}
    .dot-ds {{ background: #ffd740; }}

    .main {{ max-width: 1200px; margin: 0 auto; padding: 20px 24px; }}

    /* Filter panel */
    .filter-panel {{ margin-bottom: 20px; padding: 18px 20px; background: #fff;
                     border-radius: 0 0 12px 12px; border: 1.5px solid #e1e5eb; border-top: none;
                     box-shadow: 0 2px 6px rgba(0,0,0,0.04);
                     display: flex; flex-direction: column; gap: 14px; }}
    .filter-row {{ display: flex; gap: 12px; align-items: flex-end; flex-wrap: wrap; }}
    .fg {{ display: flex; flex-direction: column; gap: 4px; }}
    .fg label {{ font-size: 0.7em; font-weight: 600; color: #8892a4;
                 text-transform: uppercase; letter-spacing: 0.6px; }}
    .fg select, .fg input {{ padding: 7px 10px; border: 1.5px solid #e1e5eb; border-radius: 8px;
                             font-size: 0.85em; background: #fafbfc; color: #1a1a2e;
                             font-family: inherit; transition: border-color 0.2s; }}
    .fg select:focus, .fg input:focus {{ border-color: #5c6bc0; outline: none;
                                         box-shadow: 0 0 0 3px rgba(92,107,192,0.12); }}
    .fg select {{ min-width: 130px; cursor: pointer; }}
    .fg input[type="text"] {{ width: 170px; }}
    .fg input[type="number"] {{ width: 75px; }}

    .methods-group {{ display: flex; gap: 12px; align-items: center; flex-wrap: wrap; }}
    .methods-group label {{ font-size: 0.82em; color: #444; cursor: pointer;
                            display: flex; align-items: center; gap: 4px;
                            padding: 4px 8px; border-radius: 6px; transition: background 0.15s; }}
    .methods-group label:hover {{ background: #f0f2f5; }}
    .methods-group input {{ accent-color: #5c6bc0; }}

    .filter-actions {{ display: flex; gap: 8px; align-items: center; margin-left: auto; }}
    .btn {{ padding: 7px 18px; border: none; border-radius: 8px; cursor: pointer;
            font-size: 0.85em; font-family: inherit; font-weight: 500;
            transition: all 0.15s; }}
    .btn-apply {{ background: #5c6bc0; color: #fff; }}
    .btn-apply:hover {{ background: #3f51b5; }}
    .btn-reset {{ background: transparent; color: #8892a4; border: 1.5px solid #e1e5eb;
                  text-decoration: none; padding: 7px 18px; border-radius: 8px;
                  font-size: 0.85em; font-family: inherit; }}
    .btn-reset:hover {{ background: #f5f5f5; color: #555; }}
    .tabs {{ display: flex; max-width: 1200px; margin: 0 auto;
             padding: 0 24px; }}
    .tab {{ flex: 1; text-align: center; padding: 14px 24px; text-decoration: none;
            color: #8892a4; font-size: 0.95em; font-weight: 600;
            border: 1.5px solid #e1e5eb; border-bottom: none;
            background: #e4e7ec; margin-top: 16px;
            transition: all 0.15s; }}
    .tab:first-child {{ border-radius: 10px 0 0 0; border-right: none; }}
    .tab:last-child {{ border-radius: 0 10px 0 0; }}
    .tab:hover {{ color: #5c6bc0; background: #f8f9fb; }}
    .tab.active {{ color: #1a1a2e; background: #fff; border-bottom: none;
                   position: relative; z-index: 1; }}

    .ds-checkbox {{ font-size: 0.85em; color: #2e7d32; cursor: pointer;
                    display: flex; align-items: center; gap: 5px;
                    padding: 7px 12px; margin-left: auto;
                    font-weight: 500; white-space: nowrap;
                    background: #f1f8e9; border-radius: 8px; }}
    .ds-checkbox input {{ accent-color: #2e7d32; }}

    /* Pagination */
    .pagination {{ display: flex; gap: 10px; align-items: center; justify-content: center;
                   margin: 16px 0; }}
    .pagination a {{ padding: 6px 16px; border-radius: 8px; text-decoration: none;
                     background: #5c6bc0; color: #fff; font-size: 0.85em;
                     font-weight: 500; transition: background 0.15s; }}
    .pagination a:hover {{ background: #3f51b5; }}
    .page-info {{ font-size: 0.85em; color: #8892a4; }}

    /* Grid */
    .grid {{ display: flex; flex-direction: column; gap: 12px; }}

    /* Card */
    .card {{ display: flex; gap: 16px; background: #fff; border-radius: 12px;
             padding: 14px; box-shadow: 0 1px 3px rgba(0,0,0,0.05);
             border-left: 4px solid transparent; transition: all 0.2s; }}
    .card:hover {{ box-shadow: 0 4px 12px rgba(0,0,0,0.08); }}
    .card-selected {{ border-left-color: #4caf50; background: #f9fdf7; }}
    .card-image {{ flex-shrink: 0; width: 260px; min-height: 170px;
                   display: flex; align-items: center; justify-content: center;
                   background: #f5f6f8; border-radius: 8px; overflow: hidden; }}
    .card-image img {{ max-width: 260px; max-height: 300px; cursor: pointer;
                       object-fit: contain; border-radius: 4px; }}
    .card-info {{ flex: 1; display: flex; flex-direction: column; gap: 8px; }}
    .card-header {{ display: flex; gap: 8px; align-items: center; flex-wrap: wrap; }}

    .status {{ padding: 3px 10px; border-radius: 6px; font-weight: 600;
               font-size: 0.78em; text-transform: uppercase; letter-spacing: 0.3px; }}
    .status-pass {{ background: #e8f5e9; color: #2e7d32; }}
    .status-fail {{ background: #ffebee; color: #c62828; }}
    .status-error {{ background: #fff8e1; color: #f57f17; }}
    .status-pending {{ background: #f5f5f5; color: #757575; }}

    .confidence {{ font-size: 0.78em; color: #8892a4; font-weight: 500; }}
    .lang-badge {{ font-size: 0.72em; padding: 2px 8px; border-radius: 10px;
                   background: #e8eaf6; color: #3949ab; font-weight: 600;
                   text-transform: uppercase; letter-spacing: 0.3px; }}
    .resolution {{ font-size: 0.78em; color: #8892a4; }}
    .proba {{ font-size: 0.78em; color: #b0b8c4;
              background: #f5f6f8; padding: 2px 6px; border-radius: 4px; }}

    .description {{ color: #555; font-size: 0.85em; line-height: 1.45;
                    border-left: 3px solid #e8eaf6; padding-left: 10px; margin: 2px 0; }}

    .criteria-section {{ display: flex; flex-direction: column; gap: 5px; }}
    .criteria-row {{ display: flex; align-items: baseline; gap: 6px; flex-wrap: wrap; }}
    .criteria-row strong {{ min-width: 24px; font-size: 0.85em; color: #555; }}
    .criteria-detail {{ display: flex; gap: 10px; flex-wrap: wrap; font-size: 0.78em;
                        padding-left: 10px; border-left: 2px solid #f0f2f5; margin-left: 4px; }}
    .yes {{ color: #2e7d32; font-weight: 600; }}
    .no {{ color: #c62828; font-weight: 600; }}
    .grey {{ color: #ccc; }}

    .card-bottom {{ display: flex; justify-content: space-between; align-items: center;
                    margin-top: auto; padding-top: 6px; border-top: 1px solid #f5f6f8; }}
    .meta {{ font-size: 0.75em; color: #b0b8c4; display: flex; gap: 10px; align-items: center; }}
    .meta a {{ color: #8892a4; text-decoration: none; transition: color 0.15s; }}
    .meta a:hover {{ color: #5c6bc0; }}

    .ds-btn {{ padding: 5px 14px; border: none; border-radius: 6px; cursor: pointer;
               font-size: 0.8em; font-weight: 500; font-family: inherit;
               transition: all 0.15s; }}
    .ds-add {{ background: #e8f5e9; color: #2e7d32; }}
    .ds-add:hover {{ background: #c8e6c9; }}
    .ds-remove {{ background: #ffebee; color: #c62828; }}
    .ds-remove:hover {{ background: #ffcdd2; }}
</style>
</head>
<body>

<div class="header">
    <h1>MapQualityEvaluator - Stage 5 Viewer</h1>
    <div class="header-stats">
        <span><span class="dot dot-pass"></span> {meta["status"].get("pass", 0)} pass</span>
        <span><span class="dot dot-fail"></span> {meta["status"].get("fail", 0)} fail</span>
        <span><span class="dot dot-ds"></span> {meta["dataset_count"]} in dataset</span>
    </div>
</div>

<div class="tabs">
    <a href="/{tab_filters_qs}" class="tab {tab_viewer_class}">Viewer</a>
    <a href="/stats{tab_filters_qs}" class="tab {tab_stats_class}">Stats</a>
</div>

<div class="main">
<form class="filter-panel" method="GET" action="{form_action}">

    <div class="filter-row">
        <div class="fg">
            <label>Status</label>
            <select name="status">
                {"".join(f'<option value="{s}" {"selected" if s==status else ""}>{s.title()} ({meta["status"].get(s, 0)})</option>' for s in ["all", "pass", "fail", "error", "pending"])}
            </select>
        </div>
        <div class="fg">
            <label>Language</label>
            <select name="lang">{lang_opts}</select>
        </div>
        <div class="fg">
            <label>Confidence</label>
            <select name="confidence">{conf_opts}</select>
        </div>
        <div class="fg">
            <label>Domain</label>
            <select name="domain">{dom_opts}</select>
        </div>
        <label class="ds-checkbox"><input type="checkbox" name="selected" value="1" {ds_checked}> Only selected to dataset ({meta["dataset_count"]})</label>
    </div>

    <div class="filter-row">
        <div class="fg">
            <label>F3</label>
            <select name="f3">
                <option value="">All</option>
                <option value="1" {"selected" if cur_f3=="1" else ""}>Pass</option>
                <option value="0" {"selected" if cur_f3=="0" else ""}>Fail</option>
            </select>
        </div>
        <div class="fg">
            <label>F4</label>
            <select name="f4">
                <option value="">All</option>
                <option value="1" {"selected" if cur_f4=="1" else ""}>Pass</option>
                <option value="0" {"selected" if cur_f4=="0" else ""}>Fail</option>
            </select>
        </div>
        <div class="fg">
            <label>Sort</label>
            <select name="sort">{sort_opts}</select>
        </div>
    </div>

    {viewer_row2}

    <div class="filter-row">
        <div class="filter-actions">
            <button type="submit" class="btn btn-apply">Apply</button>
            <a href="{reset_href}" class="btn-reset">Reset</a>
        </div>
    </div>
</form>

{viewer_content}
{stats_content}
</div>

<script>
function dsAction(uid, action) {{
    fetch('/api/dataset', {{
        method: 'POST',
        headers: {{'Content-Type': 'application/json'}},
        body: JSON.stringify({{uid: uid, action: action}})
    }})
    .then(r => r.json())
    .then(data => {{
        if (data.ok) {{
            const card = document.getElementById('card-' + uid);
            const btn = card.querySelector('.ds-btn');
            if (action === 'add') {{
                card.classList.add('card-selected');
                btn.className = 'ds-btn ds-remove';
                btn.textContent = 'Remove from dataset';
                btn.setAttribute('onclick', "dsAction('" + uid + "', 'remove')");
            }} else {{
                card.classList.remove('card-selected');
                btn.className = 'ds-btn ds-add';
                btn.textContent = 'Add to dataset';
                btn.setAttribute('onclick', "dsAction('" + uid + "', 'add')");
            }}
            // Update dataset count in checkbox label and header
            const dsLabel = document.querySelector('.ds-checkbox');
            if (dsLabel) {{
                const cb = dsLabel.querySelector('input');
                const checked = cb ? cb.checked : false;
                dsLabel.innerHTML = '<input type="checkbox" name="selected" value="1"' + (checked ? ' checked' : '') + ' style="accent-color:#2e7d32"> Only selected (' + data.count + ')';
            }}
            const hdrDs = document.querySelector('.dot-ds');
            if (hdrDs && hdrDs.parentElement) {{
                hdrDs.parentElement.innerHTML = '<span class="dot dot-ds"></span> ' + data.count + ' in dataset';
            }}
        }}
    }});
}}
</script>
</body>
</html>"""


class ViewerHandler(SimpleHTTPRequestHandler):
    def do_GET(self):
        parsed = urlparse(self.path)

        if parsed.path.startswith("/images/"):
            img_name = parsed.path.split("/")[-1]
            thumb_path = THUMB_DIR / img_name
            # Serve thumbnail if exists, otherwise generate it
            if not thumb_path.exists():
                src_path = IMAGE_DIR / img_name
                if not src_path.exists():
                    self.send_error(404)
                    return
                _make_thumbnail(src_path, thumb_path)
            self.send_response(200)
            self.send_header("Content-Type", "image/jpeg")
            self.send_header("Cache-Control", "max-age=86400")
            self.end_headers()
            self.wfile.write(thumb_path.read_bytes())
            return

        # Full-size image (for click-to-open)
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
        selected_uids = _get_selected_uids()

        if parsed.path == "/stats":
            stats = get_stats_data(filters)
            stats_content = render_stats_content(stats, filters, meta)
            html = render_page([], 0, filters, meta, selected_uids, tab="stats", stats_content=stats_content)
        else:
            candidates, total = get_candidates(filters)
            html = render_page(candidates, total, filters, meta, selected_uids, tab="viewer")

        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()
        self.wfile.write(html.encode("utf-8"))

    def do_POST(self):
        parsed = urlparse(self.path)

        if parsed.path == "/api/dataset":
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length))
            uid = body.get("uid")
            action = body.get("action")

            if action == "add":
                _add_to_dataset(uid)
            elif action == "remove":
                _remove_from_dataset(uid)
            _meta_cache["ts"] = 0  # invalidate cache

            conn = sqlite3.connect(str(DB_PATH))
            count = conn.execute("SELECT COUNT(*) FROM dataset_selected").fetchone()[0]
            conn.close()

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"ok": True, "count": count}).encode())
            return

        self.send_error(404)

    def log_message(self, format, *args):
        pass


def main():
    _init_dataset_table()

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
