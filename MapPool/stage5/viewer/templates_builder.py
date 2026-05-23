"""HTML templates for Builder and History tabs."""

import json


def render_builder_content(meta, summary, available, preview_cards=""):
    """Render the Builder tab content."""
    # Current dataset composition bars
    total = summary["total"]
    method_bars = ""
    for label, count in summary["by_method"].items():
        pct = count / total * 100 if total > 0 else 0
        method_bars += f"""<div class="comp-row">
            <span class="comp-label">{label}</span>
            <div class="comp-bar"><div class="comp-fill" style="width:{pct}%"></div></div>
            <span class="comp-val">{count:,}</span>
        </div>"""

    lang_bars = ""
    for lang, count in list(summary["by_language"].items())[:8]:
        pct = count / total * 100 if total > 0 else 0
        lang_bars += f"""<div class="comp-row">
            <span class="comp-label">{lang}</span>
            <div class="comp-bar"><div class="comp-fill comp-fill-lang" style="width:{pct}%"></div></div>
            <span class="comp-val">{count:,}</span>
        </div>"""

    return f"""
<style>
    .builder-layout {{ display: grid; grid-template-columns: 1fr 320px; gap: 20px; }}
    .builder-form {{ background: #fff; border-radius: 12px; padding: 24px;
                     box-shadow: 0 1px 4px rgba(0,0,0,0.06); }}
    .builder-form h3 {{ font-size: 0.95em; font-weight: 600; color: #1a1a2e;
                        margin-bottom: 16px; }}
    .builder-sidebar {{ display: flex; flex-direction: column; gap: 16px; }}

    .form-grid {{ display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 14px; }}
    .form-grid .fg {{ display: flex; flex-direction: column; gap: 4px; }}
    .form-grid label {{ font-size: 0.72em; font-weight: 600; color: #8892a4;
                        text-transform: uppercase; letter-spacing: 0.5px; }}
    .form-grid select, .form-grid input {{
        padding: 8px 10px; border: 1.5px solid #e1e5eb; border-radius: 8px;
        font-size: 0.85em; background: #fafbfc; font-family: inherit; }}
    .form-grid input[type="number"] {{ width: 100%; }}

    .methods-row {{ display: flex; gap: 14px; flex-wrap: wrap; margin: 8px 0; }}
    .methods-row label {{ font-size: 0.85em; cursor: pointer;
                          display: flex; align-items: center; gap: 4px;
                          padding: 5px 10px; border-radius: 6px;
                          border: 1.5px solid #e1e5eb; transition: all 0.15s; }}
    .methods-row label:hover {{ background: #f0f2f5; }}
    .methods-row input {{ accent-color: #5c6bc0; }}

    .form-bottom {{ display: flex; justify-content: space-between; align-items: center;
                    margin-top: 18px; padding-top: 14px; border-top: 1px solid #f0f2f5; }}
    .available {{ font-size: 0.85em; color: #8892a4; }}
    .available strong {{ color: #1a1a2e; }}
    .form-actions {{ display: flex; gap: 8px; }}

    .btn-preview {{ background: #f0f2f5; color: #555; padding: 8px 20px; border: none;
                    border-radius: 8px; cursor: pointer; font-size: 0.85em;
                    font-family: inherit; font-weight: 500; }}
    .btn-preview:hover {{ background: #e4e7ec; }}
    .btn-add-batch {{ background: #5c6bc0; color: #fff; padding: 8px 24px; border: none;
                      border-radius: 8px; cursor: pointer; font-size: 0.85em;
                      font-family: inherit; font-weight: 600; }}
    .btn-add-batch:hover {{ background: #3f51b5; }}

    .composition {{ background: #fff; border-radius: 12px; padding: 18px;
                    box-shadow: 0 1px 4px rgba(0,0,0,0.06); }}
    .composition h4 {{ font-size: 0.8em; font-weight: 600; color: #8892a4;
                       text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 10px; }}
    .comp-total {{ font-size: 1.6em; font-weight: 700; color: #1a1a2e; margin-bottom: 12px; }}
    .comp-total small {{ font-size: 0.5em; color: #8892a4; font-weight: 400; }}
    .comp-row {{ display: flex; align-items: center; gap: 8px; margin-bottom: 6px; }}
    .comp-label {{ font-size: 0.78em; color: #555; min-width: 80px; }}
    .comp-bar {{ flex: 1; height: 6px; background: #f0f2f5; border-radius: 3px; overflow: hidden; }}
    .comp-fill {{ height: 100%; background: #5c6bc0; border-radius: 3px; }}
    .comp-fill-lang {{ background: #42a5f5; }}
    .comp-val {{ font-size: 0.78em; color: #8892a4; min-width: 40px; text-align: right; }}

    .preview-section {{ margin-top: 20px; }}
    .preview-section h3 {{ font-size: 0.9em; font-weight: 600; color: #555;
                           margin-bottom: 12px; text-transform: uppercase; letter-spacing: 0.5px; }}
    .preview-grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(180px, 1fr)); gap: 10px; }}
    .preview-thumb {{ background: #fff; border-radius: 8px; overflow: hidden;
                      box-shadow: 0 1px 3px rgba(0,0,0,0.06); text-align: center; }}
    .preview-thumb img {{ width: 100%; height: 140px; object-fit: contain; background: #f5f6f8; }}
    .preview-thumb .pt-info {{ padding: 6px 8px; font-size: 0.72em; color: #8892a4; }}
    .preview-thumb .pt-domain {{ color: #555; }}
</style>

<div class="builder-layout">
    <div class="builder-form">
        <h3>New Batch</h3>
        <form id="batchForm" onsubmit="return false;">
            <div class="form-grid">
                <div class="fg">
                    <label>Batch name</label>
                    <input type="text" id="batchName" value="" placeholder="e.g. choropleth_en_high">
                </div>
                <div class="fg">
                    <label>Count</label>
                    <input type="number" id="batchCount" value="500" min="1" max="50000">
                </div>
                <div class="fg">
                    <label>Language</label>
                    <select id="batchLang">
                        <option value="">All languages</option>
                        {"".join(f'<option value="{code}">{code} ({cnt})</option>' for code, cnt in sorted(meta["language"].items(), key=lambda x: -x[1]))}
                    </select>
                </div>
                <div class="fg">
                    <label>Confidence</label>
                    <select id="batchConf">
                        <option value="">All</option>
                        {"".join(f'<option value="{c}">{c} ({cnt})</option>' for c, cnt in meta["confidence"].items())}
                    </select>
                </div>
                <div class="fg">
                    <label>Min resolution (px)</label>
                    <input type="number" id="batchMinRes" value="400" min="0">
                </div>
                <div class="fg">
                    <label>Max per domain</label>
                    <input type="number" id="batchDomainCap" value="20" min="1">
                </div>
            </div>

            <div style="margin-top:12px">
                <label style="font-size:0.72em;font-weight:600;color:#8892a4;text-transform:uppercase;letter-spacing:0.5px">Methods</label>
                <div class="methods-row">
                    <label><input type="checkbox" id="batchChoro"> Choropleth</label>
                    <label><input type="checkbox" id="batchProp"> Proportional</label>
                    <label><input type="checkbox" id="batchGrad"> Graduated</label>
                    <label><input type="checkbox" id="batchStruct"> Structural</label>
                </div>
            </div>

            <div class="form-bottom">
                <div class="available">
                    Available: <strong id="availableCount">{available:,}</strong> maps
                </div>
                <div class="form-actions">
                    <button class="btn-preview" onclick="builderPreview()">Preview 30</button>
                    <button class="btn-add-batch" onclick="builderAddBatch()">Add batch</button>
                </div>
            </div>
        </form>
    </div>

    <div class="builder-sidebar">
        <div class="composition">
            <h4>Current dataset</h4>
            <div class="comp-total">{total:,} <small>maps</small></div>
            <h4>Methods</h4>
            {method_bars}
            <div style="margin-top:12px"></div>
            <h4>Languages</h4>
            {lang_bars if lang_bars else '<div style="font-size:0.8em;color:#ccc">No data yet</div>'}
        </div>
    </div>
</div>

<div id="previewArea"></div>

<script>
function getBuilderFilters() {{
    return {{
        lang: document.getElementById('batchLang').value,
        confidence: document.getElementById('batchConf').value,
        min_res: document.getElementById('batchMinRes').value,
        domain_cap: document.getElementById('batchDomainCap').value,
        m_choro: document.getElementById('batchChoro').checked ? '1' : '',
        m_prop: document.getElementById('batchProp').checked ? '1' : '',
        m_grad: document.getElementById('batchGrad').checked ? '1' : '',
        m_struct: document.getElementById('batchStruct').checked ? '1' : '',
    }};
}}

// Update available count when filters change
document.querySelectorAll('#batchForm select, #batchForm input').forEach(el => {{
    el.addEventListener('change', () => {{
        fetch('/api/builder/count', {{
            method: 'POST',
            headers: {{'Content-Type': 'application/json'}},
            body: JSON.stringify(getBuilderFilters())
        }})
        .then(r => r.json())
        .then(data => {{
            document.getElementById('availableCount').textContent = data.count.toLocaleString();
        }});
    }});
}});

function builderPreview() {{
    fetch('/api/builder/preview', {{
        method: 'POST',
        headers: {{'Content-Type': 'application/json'}},
        body: JSON.stringify(getBuilderFilters())
    }})
    .then(r => r.json())
    .then(data => {{
        let html = '<div class="preview-section"><h3>Preview (' + data.items.length + ' samples)</h3><div class="preview-grid">';
        data.items.forEach(item => {{
            const img = item.local_path ? '/images/' + item.local_path.split('/').pop() : '';
            html += '<div class="preview-thumb">' +
                '<img src="' + img + '" loading="lazy" onclick="window.open(\\'/full/' + img.split('/').pop() + '\\', \\'_blank\\')">' +
                '<div class="pt-info"><span class="pt-domain">' + item.domain + '</span><br>' +
                (item.ai_map_language || '') + ' | ' + (item.ai_confidence || '') + ' | ' +
                item.image_width + 'x' + item.image_height + '</div></div>';
        }});
        html += '</div></div>';
        document.getElementById('previewArea').innerHTML = html;
    }});
}}

function builderAddBatch() {{
    const name = document.getElementById('batchName').value.trim();
    const count = parseInt(document.getElementById('batchCount').value);
    if (!name) {{ alert('Enter a batch name'); return; }}
    if (!count || count < 1) {{ alert('Enter a valid count'); return; }}

    if (!confirm('Add batch "' + name + '" with ' + count.toLocaleString() + ' maps?')) return;

    fetch('/api/builder/add', {{
        method: 'POST',
        headers: {{'Content-Type': 'application/json'}},
        body: JSON.stringify({{
            name: name,
            count: count,
            filters: getBuilderFilters()
        }})
    }})
    .then(r => r.json())
    .then(data => {{
        if (data.ok) {{
            alert('Batch "' + data.name + '": added ' + data.added + ' maps (requested ' + data.requested + ')');
            window.location.href = '/history';
        }}
    }});
}}
</script>"""


def render_history_content(batches, summary):
    """Render the History tab content."""
    total = summary["total"]

    rows = ""
    for b in batches:
        filters_short = ""
        try:
            f = json.loads(b["filters"]) if b["filters"] else {}
            parts = []
            if f.get("lang"): parts.append(f"lang={f['lang']}")
            if f.get("confidence"): parts.append(f"conf={f['confidence']}")
            for m in ["m_choro", "m_prop", "m_grad", "m_struct"]:
                if f.get(m): parts.append(m.replace("m_", ""))
            if f.get("min_res"): parts.append(f"min_res={f['min_res']}")
            filters_short = ", ".join(parts) if parts else "no filters"
        except (json.JSONDecodeError, TypeError):
            filters_short = "—"

        diff = b["count_added"] - b["current_count"]
        removed_note = f' <span style="color:#c62828">(-{diff} removed)</span>' if diff > 0 else ""

        rows += f"""<tr>
            <td><strong>{b["name"]}</strong></td>
            <td style="font-size:0.8em;color:#8892a4">{filters_short}</td>
            <td style="text-align:center">{b["count_added"]:,}</td>
            <td style="text-align:center">{b["current_count"]:,}{removed_note}</td>
            <td style="font-size:0.8em;color:#8892a4">{b["created_at"][:16]}</td>
            <td>
                <a href="/?batch_id={b["id"]}" class="hist-btn hist-view">View</a>
                <button class="hist-btn hist-undo" onclick="undoBatch({b["id"]}, '{b["name"]}')">Undo</button>
            </td>
        </tr>"""

    return f"""
<style>
    .history-card {{ background: #fff; border-radius: 12px; padding: 24px;
                     box-shadow: 0 1px 4px rgba(0,0,0,0.06); }}
    .history-card h3 {{ font-size: 0.95em; font-weight: 600; color: #1a1a2e;
                        margin-bottom: 16px; }}
    .history-total {{ font-size: 0.9em; color: #8892a4; margin-bottom: 16px; }}
    .history-total strong {{ color: #1a1a2e; font-size: 1.3em; }}
    .progress-bar {{ height: 10px; background: #f0f2f5; border-radius: 5px;
                     overflow: hidden; margin-bottom: 20px; }}
    .progress-fill {{ height: 100%; background: linear-gradient(90deg, #5c6bc0, #42a5f5);
                      border-radius: 5px; transition: width 0.3s; }}

    .hist-table {{ width: 100%; border-collapse: collapse; }}
    .hist-table th {{ text-align: left; padding: 10px 12px; font-size: 0.72em;
                      font-weight: 600; color: #8892a4; text-transform: uppercase;
                      letter-spacing: 0.5px; border-bottom: 2px solid #f0f2f5; }}
    .hist-table td {{ padding: 10px 12px; border-bottom: 1px solid #f5f6f8;
                      font-size: 0.85em; }}
    .hist-table tr:hover {{ background: #fafbfc; }}

    .hist-btn {{ padding: 4px 12px; border-radius: 6px; font-size: 0.78em;
                 text-decoration: none; cursor: pointer; font-family: inherit;
                 font-weight: 500; border: none; margin-right: 4px; }}
    .hist-view {{ background: #e8eaf6; color: #3949ab; }}
    .hist-view:hover {{ background: #c5cae9; }}
    .hist-undo {{ background: #ffebee; color: #c62828; }}
    .hist-undo:hover {{ background: #ffcdd2; }}

    .empty-state {{ text-align: center; padding: 40px; color: #b0b8c4; font-size: 0.9em; }}
</style>

<div class="history-card">
    <h3>Dataset Build History</h3>
    <div class="history-total">
        Total in dataset: <strong>{total:,}</strong> maps
    </div>
    <div class="progress-bar">
        <div class="progress-fill" style="width:{min(total/200, 100)}%"></div>
    </div>

    {"<table class='hist-table'><tr><th>Batch</th><th>Filters</th><th>Added</th><th>Current</th><th>Date</th><th>Actions</th></tr>" + rows + "</table>" if rows else '<div class="empty-state">No batches yet. Go to Builder to create one.</div>'}

    <div style="margin-top:20px;display:flex;gap:10px;justify-content:flex-end">
        <button class="hist-btn" style="background:#e8f5e9;color:#2e7d32;padding:8px 20px;font-size:0.85em" onclick="exportDataset()">Export dataset to CSV</button>
    </div>
</div>

<script>
function exportDataset() {{
    fetch('/api/export', {{ method: 'POST', headers: {{'Content-Type': 'application/json'}}, body: '{{}}' }})
    .then(r => r.json())
    .then(data => {{
        if (data.ok) alert('Exported ' + data.count + ' maps to:\\n' + data.path);
    }});
}}

function undoBatch(id, name) {{
    if (!confirm('Remove all maps from batch "' + name + '"? This cannot be undone.')) return;
    fetch('/api/builder/undo', {{
        method: 'POST',
        headers: {{'Content-Type': 'application/json'}},
        body: JSON.stringify({{batch_id: id}})
    }})
    .then(r => r.json())
    .then(data => {{
        if (data.ok) {{
            alert('Removed ' + data.removed + ' maps from batch "' + name + '"');
            window.location.reload();
        }}
    }});
}}
</script>"""
