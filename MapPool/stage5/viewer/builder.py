"""Dataset builder: batch sampling, history, undo."""

import sqlite3
import json
import random
from datetime import datetime

from .config import DB_PATH


def _conn():
    return sqlite3.connect(str(DB_PATH))


def init_builder_tables():
    conn = _conn()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS dataset_batches (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            filters TEXT,
            count_requested INTEGER,
            count_added INTEGER DEFAULT 0,
            created_at TEXT DEFAULT (datetime('now')),
            status TEXT DEFAULT 'active'
        );

        -- Add batch_id column to dataset_selected if not exists
        CREATE TABLE IF NOT EXISTS dataset_selected (
            uid TEXT PRIMARY KEY,
            url TEXT, domain TEXT, pred_proba REAL,
            image_width INTEGER, image_height INTEGER, local_path TEXT,
            ai_status TEXT,
            ai_is_map INTEGER, ai_is_statistical_map INTEGER,
            ai_has_quantitative_data INTEGER, ai_has_admin_units INTEGER,
            ai_has_choropleth INTEGER, ai_has_proportional_circles INTEGER,
            ai_has_graduated_circles INTEGER, ai_has_structural_circles INTEGER,
            ai_confidence TEXT, ai_description TEXT, ai_map_language TEXT,
            passes_f3 INTEGER, passes_f4 INTEGER, passes_all INTEGER,
            batch_id INTEGER,
            added_at TEXT DEFAULT (datetime('now'))
        );
    """)
    # Add batch_id column if missing (table already existed without it)
    try:
        conn.execute("ALTER TABLE dataset_selected ADD COLUMN batch_id INTEGER")
    except sqlite3.OperationalError:
        pass
    conn.commit()
    conn.close()


def _build_candidate_where(filters):
    """Build WHERE for candidate selection from builder filters."""
    conditions = [
        "download_status = 'success'",
        "ai_status = 'pass'",
        "uid NOT IN (SELECT uid FROM dataset_selected)",
    ]
    params = []

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

    # Methods
    if filters.get("m_choro"):
        conditions.append("ai_has_choropleth = 1")
    if filters.get("m_prop"):
        conditions.append("ai_has_proportional_circles = 1")
    if filters.get("m_grad"):
        conditions.append("ai_has_graduated_circles = 1")
    if filters.get("m_struct"):
        conditions.append("ai_has_structural_circles = 1")

    min_res = filters.get("min_res")
    if min_res and str(min_res).isdigit():
        conditions.append("image_max_dim >= ?")
        params.append(int(min_res))

    domain_cap = filters.get("domain_cap")
    if not domain_cap or not str(domain_cap).isdigit():
        domain_cap = 20

    return conditions, params, int(domain_cap)


def count_available(filters):
    """Count how many candidates match the builder filters."""
    conn = _conn()
    conditions, params, _ = _build_candidate_where(filters)
    where = "WHERE " + " AND ".join(conditions)
    n = conn.execute(f"SELECT COUNT(*) FROM candidates {where}", params).fetchone()[0]
    conn.close()
    return n


def preview_batch(filters, n=30):
    """Get n random samples matching filters (for preview before commit)."""
    conn = _conn()
    conn.row_factory = sqlite3.Row
    conditions, params, domain_cap = _build_candidate_where(filters)
    where = "WHERE " + " AND ".join(conditions)

    rows = conn.execute(f"""
        SELECT uid, url, domain, pred_proba, image_width, image_height,
               local_path, ai_status, ai_is_map, ai_is_statistical_map,
               ai_has_quantitative_data, ai_has_admin_units,
               ai_has_choropleth, ai_has_proportional_circles,
               ai_has_graduated_circles, ai_has_structural_circles,
               ai_confidence, ai_description, ai_map_language,
               passes_f3, passes_f4, passes_all
        FROM candidates {where}
        ORDER BY RANDOM() LIMIT ?
    """, params + [n]).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def add_batch(name, filters, count):
    """Sample `count` candidates matching filters, add to dataset. Returns batch info."""
    conn = _conn()
    conn.row_factory = sqlite3.Row
    conditions, params, domain_cap = _build_candidate_where(filters)
    where = "WHERE " + " AND ".join(conditions)

    # Create batch record
    cur = conn.execute(
        "INSERT INTO dataset_batches (name, filters, count_requested) VALUES (?, ?, ?)",
        (name, json.dumps(filters), count)
    )
    batch_id = cur.lastrowid

    # Select candidates with domain cap
    # Strategy: get all matching, group by domain, cap, then random sample
    all_rows = conn.execute(f"""
        SELECT uid, url, domain, pred_proba, image_width, image_height,
               local_path, ai_status, ai_is_map, ai_is_statistical_map,
               ai_has_quantitative_data, ai_has_admin_units,
               ai_has_choropleth, ai_has_proportional_circles,
               ai_has_graduated_circles, ai_has_structural_circles,
               ai_confidence, ai_description, ai_map_language,
               passes_f3, passes_f4, passes_all
        FROM candidates {where}
        ORDER BY
            CASE ai_confidence WHEN 'high' THEN 0 WHEN 'medium' THEN 1 ELSE 2 END,
            pred_proba DESC
    """, params).fetchall()

    # Apply domain cap
    domain_counts = {}
    eligible = []
    for row in all_rows:
        d = row["domain"]
        domain_counts[d] = domain_counts.get(d, 0) + 1
        if domain_counts[d] <= domain_cap:
            eligible.append(dict(row))

    # Sample from eligible
    if len(eligible) > count:
        selected = random.sample(eligible, count)
    else:
        selected = eligible

    # Insert into dataset_selected
    added = 0
    for r in selected:
        try:
            conn.execute("""
                INSERT OR IGNORE INTO dataset_selected
                    (uid, url, domain, pred_proba, image_width, image_height, local_path,
                     ai_status, ai_is_map, ai_is_statistical_map,
                     ai_has_quantitative_data, ai_has_admin_units,
                     ai_has_choropleth, ai_has_proportional_circles,
                     ai_has_graduated_circles, ai_has_structural_circles,
                     ai_confidence, ai_description, ai_map_language,
                     passes_f3, passes_f4, passes_all, batch_id)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """, (
                r["uid"], r["url"], r["domain"], r["pred_proba"],
                r["image_width"], r["image_height"], r["local_path"],
                r["ai_status"], r["ai_is_map"], r["ai_is_statistical_map"],
                r["ai_has_quantitative_data"], r["ai_has_admin_units"],
                r["ai_has_choropleth"], r["ai_has_proportional_circles"],
                r["ai_has_graduated_circles"], r["ai_has_structural_circles"],
                r["ai_confidence"], r["ai_description"], r["ai_map_language"],
                r["passes_f3"], r["passes_f4"], r["passes_all"], batch_id,
            ))
            added += 1
        except sqlite3.IntegrityError:
            pass

    conn.execute("UPDATE dataset_batches SET count_added=? WHERE id=?", (added, batch_id))
    conn.commit()
    conn.close()

    from .data import invalidate_caches
    invalidate_caches()

    return {"batch_id": batch_id, "name": name, "requested": count, "added": added}


def get_batches():
    """Get all batches with current counts."""
    conn = _conn()
    batches = []
    for row in conn.execute("""
        SELECT b.id, b.name, b.filters, b.count_requested, b.count_added,
               b.created_at, b.status,
               (SELECT COUNT(*) FROM dataset_selected WHERE batch_id = b.id) as current_count
        FROM dataset_batches b
        WHERE b.status = 'active'
        ORDER BY b.id DESC
    """):
        batches.append({
            "id": row[0], "name": row[1], "filters": row[2],
            "count_requested": row[3], "count_added": row[4],
            "created_at": row[5], "status": row[6],
            "current_count": row[7],
        })
    conn.close()
    return batches


def undo_batch(batch_id):
    """Remove all maps from a batch and mark batch as undone."""
    conn = _conn()
    removed = conn.execute(
        "DELETE FROM dataset_selected WHERE batch_id = ?", (batch_id,)
    ).rowcount
    conn.execute(
        "UPDATE dataset_batches SET status='undone' WHERE id=?", (batch_id,)
    )
    conn.commit()
    conn.close()

    from .data import invalidate_caches
    invalidate_caches()

    return removed


def export_dataset(output_path):
    """Export dataset_selected to CSV."""
    conn = _conn()
    import csv
    rows = conn.execute("SELECT * FROM dataset_selected").fetchall()
    cols = [d[0] for d in conn.execute("SELECT * FROM dataset_selected LIMIT 1").description]
    conn.close()

    with open(output_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        w.writerows(rows)
    return len(rows)


def get_dataset_summary():
    """Get current dataset composition summary."""
    conn = _conn()
    summary = {}
    summary["total"] = conn.execute("SELECT COUNT(*) FROM dataset_selected").fetchone()[0]

    summary["by_method"] = {}
    for col, label in [("ai_has_choropleth", "Choropleth"),
                       ("ai_has_proportional_circles", "Proportional"),
                       ("ai_has_graduated_circles", "Graduated"),
                       ("ai_has_structural_circles", "Structural")]:
        n = conn.execute(f"SELECT COUNT(*) FROM dataset_selected WHERE {col}=1").fetchone()[0]
        summary["by_method"][label] = n

    summary["by_language"] = {}
    for r in conn.execute(
        "SELECT ai_map_language, COUNT(*) FROM dataset_selected "
        "WHERE ai_map_language IS NOT NULL GROUP BY ai_map_language ORDER BY COUNT(*) DESC LIMIT 10"
    ):
        summary["by_language"][r[0]] = r[1]

    summary["by_confidence"] = {}
    for r in conn.execute(
        "SELECT ai_confidence, COUNT(*) FROM dataset_selected "
        "WHERE ai_confidence IS NOT NULL GROUP BY ai_confidence ORDER BY COUNT(*) DESC"
    ):
        summary["by_confidence"][r[0]] = r[1]

    conn.close()
    return summary
