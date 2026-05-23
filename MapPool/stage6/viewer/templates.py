"""HTML templates for stage 6 viewer."""

import json
from pathlib import Path
from urllib.parse import urlencode

from .config import PER_PAGE

CRITERIA_IDS = ["D1", "D2", "A1", "A2", "A3", "V1", "V2", "L1", "L2", "L3"]
CRITERIA_NAMES = {
    "D1": "Ambiguous variable", "D2": "Inconsistent units",
    "A1": "Absolute data in choro", "A2": "Qualitative symbolization",
    "A3": "Missing classification",
    "V1": "Indistinguishable min/max", "V2": "Overlapping symbols",
    "L1": "Missing/bad title", "L2": "Missing legend", "L3": "Missing source",
}
CRITERIA_WEIGHTS = {
    "D1": 0.87, "D2": 0.51, "A1": 0.76, "A2": 0.59, "A3": 0.74,
    "V1": 0.80, "V2": 0.84, "L1": 0.73, "L2": 0.86, "L3": 0.66,
}


def _score_color(q):
    if q is None:
        return "#9e9e9e"
    if q >= 55:
        return "#2e7d32"
    if q >= 40:
        return "#f9a825"
    if q >= 20:
        return "#ef6c00"
    return "#c62828"


def _badge(cid, vlm_val, human_val):
    final = human_val if human_val is not None else vlm_val
    if final is None:
        cls = "badge-na"
        icon = ""
    elif final:
        cls = "badge-violation"
        icon = ""
    else:
        cls = "badge-ok"
        icon = ""
    override = ' <span class="override-mark">*</span>' if human_val is not None else ""
    name = CRITERIA_NAMES.get(cid, cid)
    w = CRITERIA_WEIGHTS.get(cid, 0)
    status = "violation" if final else "OK" if final is not None else "not checked"
    tooltip = f"{cid}: {name} (w={w:.2f}) — {status}"
    return f'<span class="crit-badge {cls}" data-tooltip="{tooltip}">{cid}{override}</span>'


def render_card(m, review_mode=False, review_cid=None):
    uid = m["uid"]
    path = m.get("local_path", "")
    img_src = f"/images/{Path(path).name}" if path else ""
    q = m.get("quality_score")
    q_str = f"{q:.2f}" if q is not None else "—"
    q_color = _score_color(q)
    q_pct = (q / 60 * 100) if q is not None else 0
    vcount = m.get("violations_count", 0) or 0
    desc = m.get("disq_description") or ""
    lang = m.get("ai_map_language") or ""

    # Criteria badges
    badges = ""
    for cid in CRITERIA_IDS:
        badges += _badge(cid, m.get(f"disq_{cid}"), m.get(f"human_{cid}"))

    # Review buttons (only in review mode)
    review_html = ""
    if review_mode and review_cid:
        vlm_val = m.get(f"disq_{review_cid}")
        human_val = m.get(f"human_{review_cid}")
        current = human_val if human_val is not None else vlm_val
        vlm_label = "VIOLATION" if vlm_val else "OK"
        status_label = "not reviewed"
        if human_val is not None:
            status_label = "confirmed" if human_val == vlm_val else "overridden"

        review_html = f"""
        <div class="review-panel">
            <div class="review-vlm">VLM says: <strong class="{"rv-violation" if vlm_val else "rv-ok"}">{vlm_label}</strong>
                <span class="review-status">({status_label})</span></div>
            <div class="review-actions">
                <button class="rbtn rbtn-agree {"rbtn-active" if human_val is not None and human_val == vlm_val else ""}"
                    onclick="reviewAction('{uid}', '{review_cid}', {vlm_val or 0})">Agree</button>
                <button class="rbtn rbtn-override-ok {"rbtn-active" if human_val == 0 and vlm_val == 1 else ""}"
                    onclick="reviewAction('{uid}', '{review_cid}', 0)">Override -> OK</button>
                <button class="rbtn rbtn-override-viol {"rbtn-active" if human_val == 1 and vlm_val == 0 else ""}"
                    onclick="reviewAction('{uid}', '{review_cid}', 1)">Override -> Violation</button>
            </div>
        </div>"""

    return f"""
    <div class="card" id="card-{uid}">
        <div class="card-image">
            <img src="{img_src}" alt="{uid}" loading="lazy"
                 onclick="window.open('/full/' + this.src.split('/').pop(), '_blank')">
        </div>
        <div class="card-info">
            <div class="card-header">
                <div class="score-bar" title="Quality: {q_str}/60">
                    <div class="score-fill" style="width:{q_pct}%;background:{q_color}"></div>
                    <span class="score-label">{q_str}</span>
                </div>
                <span class="viol-count">{vcount} violation{"s" if vcount != 1 else ""}</span>
                {"<span class='lang-badge'>" + lang + "</span>" if lang else ""}
            </div>
            <div class="badges">{badges}</div>
            <div class="description">{desc}</div>
            {review_html}
            <div class="card-bottom">
                <div class="meta">
                    <span class="domain">{m.get("domain", "")}</span>
                    <span class="uid" onclick="navigator.clipboard.writeText('{uid}').then(()=>this.textContent='copied!')" title="Click to copy full UID" style="cursor:pointer">{uid[:12]}...</span>
                    <a href="{m.get("url", "")}" target="_blank" class="src-link">source</a>
                </div>
                <div class="card-actions">
                    <button class="abtn abtn-rerun" onclick="mapAction('{uid}', 'rerun')" title="Re-run VLM check">Re-check</button>
                    <button class="abtn abtn-delete" onclick="mapAction('{uid}', 'delete')" title="Remove from dataset">Remove</button>
                </div>
            </div>
        </div>
    </div>"""


def _keep_filters(filters):
    shared = {k: v for k, v in filters.items()
              if k in ("status", "lang", "confidence", "domain", "review_criterion", "review_show") and v}
    return "?" + urlencode(shared) if shared else ""


def build_url(filters, **overrides):
    p = {k: v for k, v in filters.items() if v and k != "page"}
    p.update(overrides)
    p = {k: v for k, v in p.items() if v and v != "all" and v != ""}
    return "/?" + urlencode(p) if p else "/"


def render_page(maps_list, total, filters, meta, tab="viewer", extra_content=""):
    cards_html = "\n".join(render_card(m) for m in maps_list) if maps_list and tab == "viewer" else ""

    review_cid = filters.get("review_criterion", "")
    if tab == "review" and maps_list:
        cards_html = "\n".join(render_card(m, review_mode=True, review_cid=review_cid) for m in maps_list)

    page = int(filters.get("page", 1))
    total_pages = max(1, (total + PER_PAGE - 1) // PER_PAGE)

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

    tab_qs = _keep_filters(filters)
    is_viewer = tab == "viewer"
    is_review = tab == "review"
    show_filters = tab in ("viewer", "review")

    # Filter options
    status = filters.get("status", "all")
    cur_lang = filters.get("lang", "")
    cur_conf = filters.get("confidence", "")
    cur_dom = filters.get("domain", "")
    cur_sort = filters.get("sort", "score_asc")
    q_min = filters.get("q_min", "")
    q_max = filters.get("q_max", "")
    cur_search = filters.get("search", "")
    review_show = filters.get("review_show", "violations")

    lang_opts = '<option value="">All</option>'
    for c, n in sorted(meta.get("language", {}).items(), key=lambda x: -x[1]):
        sel = "selected" if c == cur_lang else ""
        lang_opts += f'<option value="{c}" {sel}>{c} ({n})</option>'

    dom_opts = '<option value="">All</option>'
    for d, n in meta.get("domains", {}).items():
        sel = "selected" if d == cur_dom else ""
        dom_opts += f'<option value="{d}" {sel}>{d} ({n})</option>'

    sort_opts = ""
    for val, label in [("score_asc", "Score low-high"), ("score_desc", "Score high-low"),
                       ("violations", "Most violations"), ("domain", "Domain"), ("lang", "Language")]:
        sel = "selected" if val == cur_sort else ""
        sort_opts += f'<option value="{val}" {sel}>{label}</option>'

    # Review tab criterion selector
    review_select = ""
    if is_review:
        crit_opts = ""
        for cid in CRITERIA_IDS:
            sel = "selected" if cid == review_cid else ""
            vn = meta.get("violations", {}).get(cid, 0)
            rn = meta.get("reviewed", {}).get(cid, 0)
            crit_opts += f'<option value="{cid}" {sel}>{cid}: {CRITERIA_NAMES[cid]} ({vn} violations, {rn} reviewed)</option>'

        show_opts = ""
        for val, label in [("violations", "VLM violations"), ("ok", "VLM OK"),
                           ("not_reviewed", "Not yet reviewed"), ("overridden", "Overridden")]:
            sel = "selected" if val == review_show else ""
            show_opts += f'<option value="{val}" {sel}>{label}</option>'

        review_select = f"""
        <div class="filter-row">
            <div class="fg">
                <label>Criterion</label>
                <select name="review_criterion" style="min-width:280px">{crit_opts}</select>
            </div>
            <div class="fg">
                <label>Show</label>
                <select name="review_show">{show_opts}</select>
            </div>
        </div>"""

    # Build filter panel
    filter_panel = ""
    if show_filters:
        form_action = "/" if is_viewer else "/review"
        filter_panel = f"""<form class="filter-panel" method="GET" action="{form_action}">
    {review_select}
    <div class="filter-row">
        <div class="fg"><label>Status</label>
            <select name="status">
                {"".join(f'<option value="{s}" {"selected" if s==status else ""}>{s.title()} ({meta["status"].get(s,0)})</option>' for s in ["all","checked","pending","error"])}
            </select></div>
        <div class="fg"><label>Language</label><select name="lang">{lang_opts}</select></div>
        <div class="fg"><label>Domain</label><select name="domain">{dom_opts}</select></div>
        <div class="fg"><label>Q min</label><input type="number" name="q_min" value="{q_min}" step="1" style="width:65px"></div>
        <div class="fg"><label>Q max</label><input type="number" name="q_max" value="{q_max}" step="1" style="width:65px"></div>
        <div class="fg"><label>Sort</label><select name="sort">{sort_opts}</select></div>
        <div class="fg"><label>Search</label><input type="text" name="search" value="{cur_search}" placeholder="description..." style="width:150px"></div>
        <div class="filter-actions">
            <button type="submit" class="btn btn-apply">Apply</button>
            <a href="{form_action}" class="btn-reset">Reset</a>
        </div>
    </div>
</form>"""

    content = ""
    if tab in ("viewer", "review") and cards_html:
        content = f"{pagination}<div class='grid'>{cards_html}</div>{pagination}"

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Stage 6 -- Disqualifying Criteria</title>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
<script src="https://cdn.jsdelivr.net/npm/chart.js@4"></script>
<style>
    * {{ margin: 0; padding: 0; box-sizing: border-box; }}
    body {{ font-family: 'Inter', -apple-system, sans-serif; background: #f0f2f5; color: #1a1a2e; }}

    .header {{ background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
               padding: 18px 32px; color: #fff; display: flex;
               justify-content: space-between; align-items: center; }}
    .header h1 {{ font-size: 1.2em; font-weight: 600; }}
    .header-stats {{ display: flex; gap: 18px; font-size: 0.85em; opacity: 0.8; }}

    .tabs {{ display: flex; max-width: 1200px; margin: 0 auto; padding: 0 24px; }}
    .tab {{ flex: 1; text-align: center; padding: 12px; text-decoration: none;
            color: #8892a4; font-size: 0.9em; font-weight: 600;
            border: 1.5px solid #e1e5eb; border-bottom: none;
            background: #e4e7ec; margin-top: 14px; transition: all 0.15s; }}
    .tab:first-child {{ border-radius: 10px 0 0 0; }}
    .tab:last-child {{ border-radius: 0 10px 0 0; border-right: 1.5px solid #e1e5eb; }}
    .tab {{ border-right: none; }}
    .tab:hover {{ color: #5c6bc0; background: #f8f9fb; }}
    .tab.active {{ color: #1a1a2e; background: #fff; }}

    .main {{ max-width: 1200px; margin: 0 auto; padding: 20px 24px; }}

    .filter-panel {{ margin-bottom: 18px; padding: 16px 20px; background: #fff;
                     border-radius: 0 0 12px 12px; border: 1.5px solid #e1e5eb; border-top: none;
                     display: flex; flex-direction: column; gap: 12px; }}
    .filter-row {{ display: flex; gap: 10px; align-items: flex-end; flex-wrap: wrap; }}
    .fg {{ display: flex; flex-direction: column; gap: 3px; }}
    .fg label {{ font-size: 0.7em; font-weight: 600; color: #8892a4; text-transform: uppercase; letter-spacing: 0.5px; }}
    .fg select, .fg input {{ padding: 6px 8px; border: 1.5px solid #e1e5eb; border-radius: 6px;
                             font-size: 0.85em; background: #fafbfc; font-family: inherit; }}
    .fg select {{ min-width: 110px; cursor: pointer; }}
    .filter-actions {{ display: flex; gap: 6px; margin-left: auto; }}
    .btn {{ padding: 6px 16px; border: none; border-radius: 6px; cursor: pointer;
            font-size: 0.85em; font-family: inherit; font-weight: 500; }}
    .btn-apply {{ background: #5c6bc0; color: #fff; }}
    .btn-reset {{ background: transparent; color: #8892a4; border: 1.5px solid #e1e5eb;
                  text-decoration: none; padding: 6px 16px; border-radius: 6px; font-size: 0.85em; }}

    .pagination {{ display: flex; gap: 10px; align-items: center; justify-content: center; margin: 14px 0; }}
    .pagination a {{ padding: 5px 14px; border-radius: 6px; text-decoration: none;
                     background: #5c6bc0; color: #fff; font-size: 0.85em; }}
    .page-info {{ font-size: 0.85em; color: #8892a4; }}

    .grid {{ display: flex; flex-direction: column; gap: 12px; }}
    .card {{ display: flex; gap: 14px; background: #fff; border-radius: 10px;
             padding: 14px; box-shadow: 0 1px 3px rgba(0,0,0,0.05); }}
    .card:hover {{ box-shadow: 0 3px 10px rgba(0,0,0,0.08); }}
    .card-image {{ flex-shrink: 0; width: 260px; min-height: 170px;
                   display: flex; align-items: center; justify-content: center;
                   background: #f5f6f8; border-radius: 8px; overflow: hidden; }}
    .card-image img {{ max-width: 260px; max-height: 300px; cursor: pointer; object-fit: contain; }}
    .card-info {{ flex: 1; display: flex; flex-direction: column; gap: 6px; }}
    .card-header {{ display: flex; gap: 10px; align-items: center; flex-wrap: wrap; }}

    .score-bar {{ width: 120px; height: 22px; background: #f0f2f5; border-radius: 6px;
                  overflow: hidden; position: relative; }}
    .score-fill {{ height: 100%; border-radius: 6px; transition: width 0.3s; }}
    .score-label {{ position: absolute; top: 2px; left: 50%; transform: translateX(-50%);
                    font-size: 0.75em; font-weight: 700; color: #fff;
                    text-shadow: 0 1px 2px rgba(0,0,0,0.3); }}
    .viol-count {{ font-size: 0.8em; color: #c62828; font-weight: 500; }}
    .lang-badge {{ font-size: 0.72em; padding: 2px 7px; border-radius: 10px;
                   background: #e8eaf6; color: #3949ab; font-weight: 600; text-transform: uppercase; }}

    .badges {{ display: flex; gap: 4px; flex-wrap: wrap; }}
    .crit-badge {{ font-size: 0.7em; padding: 2px 6px; border-radius: 4px; font-weight: 600;
                   cursor: default; position: relative; }}
    .crit-badge:hover::after {{
        content: attr(data-tooltip);
        position: absolute; bottom: 110%; left: 50%; transform: translateX(-50%);
        background: #1a1a2e; color: #fff; padding: 5px 10px; border-radius: 6px;
        font-size: 1.1em; font-weight: 400; white-space: nowrap; z-index: 100;
        box-shadow: 0 2px 8px rgba(0,0,0,0.2); pointer-events: none;
    }}
    .badge-ok {{ background: #e8f5e9; color: #2e7d32; }}
    .badge-violation {{ background: #ffebee; color: #c62828; }}
    .badge-na {{ background: #f5f5f5; color: #bbb; }}
    .override-mark {{ color: #5c6bc0; font-weight: 700; }}

    .description {{ color: #555; font-size: 0.82em; line-height: 1.4;
                    border-left: 3px solid #e8eaf6; padding-left: 10px; margin: 2px 0; }}
    .card-bottom {{ display: flex; justify-content: space-between; align-items: center;
                    margin-top: auto; padding-top: 6px; border-top: 1px solid #f5f6f8; }}
    .meta {{ font-size: 0.75em; color: #b0b8c4; display: flex; gap: 8px; align-items: center; }}
    .meta a {{ color: #8892a4; text-decoration: none; }}
    .meta a:hover {{ color: #5c6bc0; }}
    .card-actions {{ display: flex; gap: 5px; }}
    .abtn {{ padding: 3px 10px; border: 1.5px solid #e1e5eb; border-radius: 5px;
             background: #fff; cursor: pointer; font-size: 0.75em; font-family: inherit;
             font-weight: 500; transition: all 0.15s; }}
    .abtn-rerun {{ color: #5c6bc0; }}
    .abtn-rerun:hover {{ background: #e8eaf6; border-color: #5c6bc0; }}
    .abtn-delete {{ color: #c62828; }}
    .abtn-delete:hover {{ background: #ffebee; border-color: #c62828; }}

    /* Review panel */
    .review-panel {{ background: #fafbfc; border: 1.5px solid #e1e5eb; border-radius: 8px;
                     padding: 10px 14px; margin-top: 4px; }}
    .review-vlm {{ font-size: 0.85em; margin-bottom: 8px; }}
    .rv-violation {{ color: #c62828; }}
    .rv-ok {{ color: #2e7d32; }}
    .review-status {{ font-size: 0.8em; color: #8892a4; }}
    .review-actions {{ display: flex; gap: 6px; }}
    .rbtn {{ padding: 5px 14px; border: 1.5px solid #e1e5eb; border-radius: 6px;
             background: #fff; cursor: pointer; font-size: 0.8em; font-family: inherit;
             font-weight: 500; transition: all 0.15s; }}
    .rbtn:hover {{ background: #f0f2f5; }}
    .rbtn-active {{ border-color: #5c6bc0; background: #e8eaf6; color: #3949ab; }}
    .rbtn-agree {{ color: #2e7d32; }}
    .rbtn-override-ok {{ color: #2e7d32; }}
    .rbtn-override-viol {{ color: #c62828; }}

    /* Stats */
    .stats-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(340px, 1fr)); gap: 16px; }}
    .chart-card {{ background: #fff; border-radius: 12px; padding: 18px; box-shadow: 0 1px 4px rgba(0,0,0,0.06); }}
    .chart-card h3 {{ font-size: 0.85em; font-weight: 600; color: #555; margin-bottom: 12px;
                      text-transform: uppercase; letter-spacing: 0.5px; }}
    .chart-card canvas {{ max-height: 260px; }}
    .chart-card-wide {{ grid-column: 1 / -1; }}
    .summary-cards {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
                      gap: 12px; margin-bottom: 20px; }}
    .summary-card {{ background: #fff; border-radius: 12px; padding: 14px 18px; text-align: center;
                     box-shadow: 0 1px 4px rgba(0,0,0,0.06); }}
    .summary-card .num {{ font-size: 1.6em; font-weight: 700; }}
    .summary-card .lbl {{ font-size: 0.7em; color: #8892a4; text-transform: uppercase;
                          letter-spacing: 0.5px; margin-top: 3px; }}
    .stats-table {{ width: 100%; border-collapse: collapse; font-size: 0.82em; }}
    .stats-table th {{ text-align: left; padding: 7px 10px; color: #8892a4; font-weight: 600;
                       font-size: 0.7em; text-transform: uppercase; border-bottom: 2px solid #f0f2f5; }}
    .stats-table td {{ padding: 7px 10px; border-bottom: 1px solid #f5f6f8; }}
    .pbar {{ width: 100%; height: 6px; background: #f0f2f5; border-radius: 3px; overflow: hidden; }}
    .pbar-fill {{ height: 100%; background: #5c6bc0; border-radius: 3px; }}
</style>
</head>
<body>

<div class="header">
    <h1>Stage 6 -- Disqualifying Criteria</h1>
    <div class="header-stats">
        <span>{meta.get("checked", 0)} checked</span>
        <span>{meta.get("pending", 0)} pending</span>
        <span>avg Q: {meta.get("avg_score", "—")}</span>
    </div>
</div>

<div class="tabs">
    <a href="/{tab_qs}" class="tab {"active" if tab=="viewer" else ""}">Viewer</a>
    <a href="/review{tab_qs}" class="tab {"active" if tab=="review" else ""}">Review</a>
    <a href="/stats" class="tab {"active" if tab=="stats" else ""}">Stats</a>
</div>

<div class="main">
{filter_panel}
{content}
{extra_content}
</div>

<script>
function mapAction(uid, action) {{
    if (action === 'delete' && !confirm('Remove this map from dataset?')) return;
    if (action === 'rerun' && !confirm('Re-run VLM check on this map?')) return;
    fetch('/api/map', {{
        method: 'POST',
        headers: {{'Content-Type': 'application/json'}},
        body: JSON.stringify({{uid: uid, action: action}})
    }})
    .then(r => r.json())
    .then(data => {{
        if (data.ok) {{
            const card = document.getElementById('card-' + uid);
            if (action === 'delete') {{
                card.style.opacity = '0.3';
                card.style.pointerEvents = 'none';
            }} else if (action === 'rerun') {{
                card.querySelector('.viol-count').textContent = 'Re-check queued';
            }}
        }}
    }});
}}

function reviewAction(uid, cid, value) {{
    fetch('/api/review', {{
        method: 'POST',
        headers: {{'Content-Type': 'application/json'}},
        body: JSON.stringify({{uid: uid, criterion: cid, value: value}})
    }})
    .then(r => r.json())
    .then(data => {{
        if (data.ok) {{
            // Update score bar
            const card = document.getElementById('card-' + uid);
            const scoreLabel = card.querySelector('.score-label');
            const scoreFill = card.querySelector('.score-fill');
            const violCount = card.querySelector('.viol-count');
            if (scoreLabel) scoreLabel.textContent = Math.round(data.score);
            if (scoreFill) {{
                scoreFill.style.width = (data.score / 60 * 100) + '%';
            }}
            if (violCount) violCount.textContent = data.violations + ' violation' + (data.violations !== 1 ? 's' : '');

            // Update review buttons
            const btns = card.querySelectorAll('.rbtn');
            btns.forEach(b => b.classList.remove('rbtn-active'));
            // Mark the clicked button active
            event.target.classList.add('rbtn-active');

            // Update review status text
            const statusEl = card.querySelector('.review-status');
            if (statusEl) statusEl.textContent = '(confirmed)';
        }}
    }});
}}
</script>
</body>
</html>"""


def render_stats_content(stats, meta):
    checked = stats.get("checked", 0)

    # Summary cards
    avg_q = stats.get("avg_score", 0)

    # Violations table
    viol_rows = ""
    for cid in CRITERIA_IDS:
        v = stats["violations"].get(cid, {})
        vlm = v.get("vlm", 0)
        final = v.get("final", 0)
        overrides = v.get("overrides", 0)
        reviewed = v.get("reviewed", 0)
        pct = final / checked * 100 if checked > 0 else 0
        review_pct = reviewed / checked * 100 if checked > 0 else 0
        viol_rows += f"""<tr>
            <td><strong>{cid}</strong></td>
            <td>{CRITERIA_NAMES[cid]}</td>
            <td style="text-align:center">{CRITERIA_WEIGHTS[cid]:.2f}</td>
            <td style="text-align:center;color:#c62828">{final}</td>
            <td style="text-align:center">{pct:.1f}%</td>
            <td style="text-align:center">{overrides}</td>
            <td style="text-align:center">{reviewed}</td>
            <td><div class="pbar"><div class="pbar-fill" style="width:{review_pct}%"></div></div></td>
        </tr>"""

    # Chart data
    score_labels = json.dumps(list(stats.get("score_hist", {}).keys()))
    score_values = json.dumps(list(stats.get("score_hist", {}).values()))
    viol_labels = json.dumps(list(stats.get("viol_dist", {}).keys()))
    viol_values = json.dumps(list(stats.get("viol_dist", {}).values()))

    crit_labels = json.dumps(CRITERIA_IDS)
    crit_final = json.dumps([stats["violations"].get(cid, {}).get("final", 0) for cid in CRITERIA_IDS])

    return f"""
<div class="summary-cards">
    <div class="summary-card"><div class="num">{checked:,}</div><div class="lbl">Checked</div></div>
    <div class="summary-card"><div class="num" style="color:{_score_color(avg_q)}">{avg_q:.1f}</div><div class="lbl">Avg Q score</div></div>
    <div class="summary-card"><div class="num">{meta.get("pending", 0):,}</div><div class="lbl">Pending</div></div>
    <div class="summary-card"><div class="num">{meta.get("errors", 0):,}</div><div class="lbl">Errors</div></div>
</div>

<div class="stats-grid">
    <div class="chart-card">
        <h3>Quality score distribution</h3>
        <canvas id="chartScore"></canvas>
    </div>
    <div class="chart-card">
        <h3>Violations per map</h3>
        <canvas id="chartViolDist"></canvas>
    </div>
    <div class="chart-card chart-card-wide">
        <h3>Violations per criterion</h3>
        <canvas id="chartCrit" style="max-height:250px"></canvas>
    </div>
    <div class="chart-card chart-card-wide">
        <h3>Detailed breakdown</h3>
        <table class="stats-table">
            <tr><th>ID</th><th>Criterion</th><th>Weight</th><th>Violations</th><th>%</th><th>Overrides</th><th>Reviewed</th><th>Review progress</th></tr>
            {viol_rows}
        </table>
    </div>
</div>

<script>
new Chart(document.getElementById('chartScore'), {{
    type: 'bar',
    data: {{ labels: {score_labels}, datasets: [{{ data: {score_values}, backgroundColor: '#5c6bc0' }}] }},
    options: {{ responsive: true, plugins: {{ legend: {{ display: false }} }}, scales: {{ y: {{ beginAtZero: true }} }} }}
}});
new Chart(document.getElementById('chartViolDist'), {{
    type: 'bar',
    data: {{ labels: {viol_labels}, datasets: [{{ data: {viol_values}, backgroundColor: '#ef5350' }}] }},
    options: {{ responsive: true, plugins: {{ legend: {{ display: false }} }}, scales: {{ y: {{ beginAtZero: true }} }} }}
}});
new Chart(document.getElementById('chartCrit'), {{
    type: 'bar',
    data: {{ labels: {crit_labels}, datasets: [{{ data: {crit_final}, backgroundColor: '#ef5350' }}] }},
    options: {{ responsive: true, indexAxis: 'y', plugins: {{ legend: {{ display: false }} }},
        scales: {{ x: {{ beginAtZero: true }} }} }}
}});
</script>"""
