"""Database queries for the viewer: filters, candidates, stats, dataset selection."""

import sqlite3
import json
import time

from .config import DB_PATH, PER_PAGE

# In-memory caches
_meta_cache = {"data": None, "ts": 0}
_stats_cache = {"key": None, "data": None, "ts": 0}
_META_TTL = 30
_STATS_TTL = 60


def _conn():
    return sqlite3.connect(str(DB_PATH))


def init_dataset_table():
    conn = _conn()
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


def get_selected_uids():
    conn = _conn()
    uids = {r[0] for r in conn.execute("SELECT uid FROM dataset_selected")}
    conn.close()
    return uids


def add_to_dataset(uid):
    conn = _conn()
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
    invalidate_caches()


def remove_from_dataset(uid):
    conn = _conn()
    conn.execute("DELETE FROM dataset_selected WHERE uid = ?", (uid,))
    conn.commit()
    conn.close()
    invalidate_caches()


def get_dataset_count():
    conn = _conn()
    count = conn.execute("SELECT COUNT(*) FROM dataset_selected").fetchone()[0]
    conn.close()
    return count


def invalidate_caches():
    _meta_cache["ts"] = 0
    _stats_cache["ts"] = 0


def get_filters_meta():
    now = time.time()
    if _meta_cache["data"] and now - _meta_cache["ts"] < _META_TTL:
        return _meta_cache["data"]

    conn = _conn()
    meta = {}

    meta["status"] = {}
    for r in conn.execute(
        "SELECT ai_status, COUNT(*) FROM candidates WHERE download_status='success' GROUP BY ai_status"
    ):
        meta["status"][r[0] or "null"] = r[1]
    meta["status"]["all"] = sum(meta["status"].values())

    meta["language"] = {}
    for r in conn.execute(
        "SELECT ai_map_language, COUNT(*) FROM candidates "
        "WHERE download_status='success' AND ai_map_language IS NOT NULL "
        "GROUP BY ai_map_language ORDER BY COUNT(*) DESC"
    ):
        meta["language"][r[0]] = r[1]

    meta["confidence"] = {}
    for r in conn.execute(
        "SELECT ai_confidence, COUNT(*) FROM candidates "
        "WHERE download_status='success' AND ai_confidence IS NOT NULL "
        "GROUP BY ai_confidence ORDER BY COUNT(*) DESC"
    ):
        meta["confidence"][r[0]] = r[1]

    meta["domains"] = {}
    for r in conn.execute(
        "SELECT domain, COUNT(*) FROM candidates "
        "WHERE download_status='success' GROUP BY domain ORDER BY COUNT(*) DESC LIMIT 50"
    ):
        meta["domains"][r[0]] = r[1]

    meta["dataset_count"] = conn.execute("SELECT COUNT(*) FROM dataset_selected").fetchone()[0]

    conn.close()
    _meta_cache["data"] = meta
    _meta_cache["ts"] = now
    return meta


def _build_where(filters):
    """Build WHERE clause and params from filters dict."""
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

    if filters.get("selected") == "1":
        conditions.append("uid IN (SELECT uid FROM dataset_selected)")

    batch_id = filters.get("batch_id")
    if batch_id and str(batch_id).isdigit():
        conditions.append("uid IN (SELECT uid FROM dataset_selected WHERE batch_id = ?)")
        params.append(int(batch_id))

    return conditions, params


def get_candidates(filters):
    conn = _conn()
    conn.row_factory = sqlite3.Row

    conditions, params = _build_where(filters)

    # Viewer-specific filters
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

    min_w = filters.get("min_w")
    if min_w and min_w.isdigit():
        conditions.append("image_width >= ?")
        params.append(int(min_w))
    min_h = filters.get("min_h")
    if min_h and min_h.isdigit():
        conditions.append("image_height >= ?")
        params.append(int(min_h))

    search = filters.get("search")
    if search:
        conditions.append("ai_description LIKE ?")
        params.append(f"%{search}%")

    where = "WHERE " + " AND ".join(conditions)

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


def get_stats_data(filters):
    cache_key = json.dumps(filters, sort_keys=True)
    now = time.time()
    if _stats_cache["key"] == cache_key and _stats_cache["data"] and now - _stats_cache["ts"] < _STATS_TTL:
        return _stats_cache["data"]

    conn = _conn()
    conditions, params = _build_where(filters)
    where = "WHERE " + " AND ".join(conditions)

    stats = {}
    stats["total"] = conn.execute(f"SELECT COUNT(*) FROM candidates {where}", params).fetchone()[0]

    stats["by_status"] = {}
    for r in conn.execute(f"SELECT ai_status, COUNT(*) FROM candidates {where} GROUP BY ai_status ORDER BY COUNT(*) DESC", params):
        stats["by_status"][r[0] or "null"] = r[1]

    stats["by_language"] = {}
    for r in conn.execute(f"SELECT ai_map_language, COUNT(*) FROM candidates {where} AND ai_map_language IS NOT NULL GROUP BY ai_map_language ORDER BY COUNT(*) DESC", params):
        stats["by_language"][r[0]] = r[1]

    stats["by_confidence"] = {}
    for r in conn.execute(f"SELECT ai_confidence, COUNT(*) FROM candidates {where} AND ai_confidence IS NOT NULL GROUP BY ai_confidence ORDER BY COUNT(*) DESC", params):
        stats["by_confidence"][r[0]] = r[1]

    stats["methods"] = {}
    for col, label in [("ai_has_choropleth", "Choropleth"), ("ai_has_proportional_circles", "Proportional"),
                       ("ai_has_graduated_circles", "Graduated"), ("ai_has_structural_circles", "Structural")]:
        n = conn.execute(f"SELECT COUNT(*) FROM candidates {where} AND {col}=1", params).fetchone()[0]
        stats["methods"][label] = n

    stats["top_domains"] = []
    for r in conn.execute(f"SELECT domain, COUNT(*) FROM candidates {where} GROUP BY domain ORDER BY COUNT(*) DESC LIMIT 15", params):
        stats["top_domains"].append((r[0], r[1]))

    stats["resolution"] = {}
    for label, cond in [("<400px", "image_max_dim < 400"),
                        ("400-600", "image_max_dim BETWEEN 400 AND 599"),
                        ("600-800", "image_max_dim BETWEEN 600 AND 799"),
                        ("800-1200", "image_max_dim BETWEEN 800 AND 1199"),
                        ("1200+", "image_max_dim >= 1200")]:
        n = conn.execute(f"SELECT COUNT(*) FROM candidates {where} AND {cond}", params).fetchone()[0]
        stats["resolution"][label] = n

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
