"""Database queries for stage 6 viewer."""

import sqlite3
import json
import time

from .config import DB_PATH, PER_PAGE

_meta_cache = {"data": None, "ts": 0}
_META_TTL = 30

CRITERIA_IDS = ["D1", "D2", "A1", "A2", "A3", "V1", "V2", "L1", "L2", "L3"]


def _conn():
    return sqlite3.connect(str(DB_PATH))


def get_filters_meta():
    now = time.time()
    if _meta_cache["data"] and now - _meta_cache["ts"] < _META_TTL:
        return _meta_cache["data"]

    conn = _conn()
    meta = {}

    meta["total"] = conn.execute("SELECT COUNT(*) FROM maps").fetchone()[0]
    meta["checked"] = conn.execute("SELECT COUNT(*) FROM maps WHERE disq_status='checked'").fetchone()[0]
    meta["pending"] = conn.execute("SELECT COUNT(*) FROM maps WHERE disq_status='pending'").fetchone()[0]
    meta["errors"] = conn.execute("SELECT COUNT(*) FROM maps WHERE disq_status='error'").fetchone()[0]

    meta["status"] = {"all": meta["total"], "checked": meta["checked"],
                      "pending": meta["pending"], "error": meta["errors"]}

    meta["language"] = {}
    for r in conn.execute(
        "SELECT ai_map_language, COUNT(*) FROM maps WHERE ai_map_language IS NOT NULL "
        "GROUP BY ai_map_language ORDER BY COUNT(*) DESC"
    ):
        meta["language"][r[0]] = r[1]

    meta["confidence"] = {}
    for r in conn.execute(
        "SELECT disq_confidence, COUNT(*) FROM maps WHERE disq_confidence IS NOT NULL "
        "GROUP BY disq_confidence ORDER BY COUNT(*) DESC"
    ):
        meta["confidence"][r[0]] = r[1]

    meta["domains"] = {}
    for r in conn.execute(
        "SELECT domain, COUNT(*) FROM maps GROUP BY domain ORDER BY COUNT(*) DESC LIMIT 50"
    ):
        meta["domains"][r[0]] = r[1]

    # Average quality score
    avg = conn.execute("SELECT AVG(quality_score) FROM maps WHERE disq_status='checked'").fetchone()[0]
    meta["avg_score"] = f"{avg:.2f}" if avg is not None else "—"

    # Per-criterion violation counts
    meta["violations"] = {}
    for cid in CRITERIA_IDS:
        n = conn.execute(f"SELECT COUNT(*) FROM maps WHERE disq_{cid}=1").fetchone()[0]
        meta["violations"][cid] = n

    # Human review progress
    meta["reviewed"] = {}
    for cid in CRITERIA_IDS:
        n = conn.execute(f"SELECT COUNT(*) FROM maps WHERE human_{cid} IS NOT NULL").fetchone()[0]
        meta["reviewed"][cid] = n

    conn.close()
    _meta_cache["data"] = meta
    _meta_cache["ts"] = now
    return meta


def invalidate_caches():
    _meta_cache["ts"] = 0


def get_maps(filters):
    conn = _conn()
    conn.row_factory = sqlite3.Row

    conditions = ["1=1"]
    params = []

    status = filters.get("status", "all")
    if status and status != "all":
        conditions.append("disq_status = ?")
        params.append(status)

    lang = filters.get("lang")
    if lang:
        conditions.append("ai_map_language = ?")
        params.append(lang)

    confidence = filters.get("confidence")
    if confidence:
        conditions.append("disq_confidence = ?")
        params.append(confidence)

    domain = filters.get("domain")
    if domain:
        conditions.append("domain = ?")
        params.append(domain)

    # Filter by specific violation
    for cid in CRITERIA_IDS:
        v = filters.get(f"v_{cid}")
        if v == "1":
            conditions.append(f"COALESCE(human_{cid}, disq_{cid}) = 1")
        elif v == "0":
            conditions.append(f"COALESCE(human_{cid}, disq_{cid}) = 0")

    # Review criterion filter (for Review tab)
    review_cid = filters.get("review_criterion")
    review_show = filters.get("review_show")
    if review_cid and review_cid in CRITERIA_IDS:
        conditions.append("disq_status = 'checked'")
        if review_show == "violations":
            conditions.append(f"disq_{review_cid} = 1")
        elif review_show == "ok":
            conditions.append(f"disq_{review_cid} = 0")
        elif review_show == "not_reviewed":
            conditions.append(f"human_{review_cid} IS NULL")
        elif review_show == "overridden":
            conditions.append(f"human_{review_cid} IS NOT NULL AND human_{review_cid} != disq_{review_cid}")

    # Quality score range
    q_min = filters.get("q_min")
    if q_min and q_min.replace(".", "").isdigit():
        conditions.append("quality_score >= ?")
        params.append(float(q_min))
    q_max = filters.get("q_max")
    if q_max and q_max.replace(".", "").isdigit():
        conditions.append("quality_score <= ?")
        params.append(float(q_max))

    # Method filter
    if filters.get("m_choro"):
        conditions.append("ai_has_choropleth = 1")
    if filters.get("m_circles"):
        conditions.append("(ai_has_proportional_circles = 1 OR ai_has_graduated_circles = 1 OR ai_has_structural_circles = 1)")

    search = filters.get("search")
    if search:
        conditions.append("disq_description LIKE ?")
        params.append(f"%{search}%")

    where = "WHERE " + " AND ".join(conditions)

    sort = filters.get("sort", "score_asc")
    order = {
        "score_asc": "quality_score ASC",
        "score_desc": "quality_score DESC",
        "violations": "violations_count DESC, quality_score ASC",
        "domain": "domain ASC",
        "lang": "ai_map_language ASC",
    }.get(sort, "quality_score ASC")

    page = int(filters.get("page", 1))
    offset = (page - 1) * PER_PAGE

    total = conn.execute(f"SELECT COUNT(*) FROM maps {where}", params).fetchone()[0]
    rows = conn.execute(f"""
        SELECT * FROM maps {where} ORDER BY {order} LIMIT {PER_PAGE} OFFSET {offset}
    """, params).fetchall()
    conn.close()
    return [dict(r) for r in rows], total


def set_human_override(uid, criterion_id, value):
    """Set human override and recompute score."""
    import sys
    sys.path.insert(0, str(DB_PATH.parent))
    from criteria import get_final_violations, compute_quality_score

    conn = _conn()
    conn.row_factory = sqlite3.Row
    from datetime import datetime
    conn.execute(
        f"UPDATE maps SET human_{criterion_id}=?, human_reviewed_at=? WHERE uid=?",
        (value, datetime.now().isoformat(), uid)
    )

    # Recompute score
    row = dict(conn.execute("SELECT * FROM maps WHERE uid=?", (uid,)).fetchone())
    violations = get_final_violations(row)
    q, count, _ = compute_quality_score(violations)
    conn.execute("UPDATE maps SET quality_score=?, violations_count=? WHERE uid=?",
                 (round(q, 2), count, uid))
    conn.commit()
    conn.close()
    invalidate_caches()
    return round(q, 2), count


def delete_map(uid):
    """Remove map from stage6 database."""
    conn = _conn()
    conn.execute("DELETE FROM maps WHERE uid = ?", (uid,))
    conn.commit()
    conn.close()
    invalidate_caches()


def reset_map(uid):
    """Reset map to pending (for re-checking with VLM)."""
    conn = _conn()
    conn.execute("""
        UPDATE maps SET
            disq_status='pending',
            disq_D1=NULL, disq_D2=NULL,
            disq_A1=NULL, disq_A2=NULL, disq_A3=NULL,
            disq_V1=NULL, disq_V2=NULL,
            disq_L1=NULL, disq_L2=NULL, disq_L3=NULL,
            disq_confidence=NULL, disq_description=NULL, disq_raw_response=NULL,
            disq_checked_at=NULL, disq_input_tokens=NULL, disq_output_tokens=NULL,
            human_D1=NULL, human_D2=NULL,
            human_A1=NULL, human_A2=NULL, human_A3=NULL,
            human_V1=NULL, human_V2=NULL,
            human_L1=NULL, human_L2=NULL, human_L3=NULL,
            human_reviewed_at=NULL,
            violations_count=NULL, quality_score=NULL
        WHERE uid = ?
    """, (uid,))
    conn.commit()
    conn.close()
    invalidate_caches()


def get_stats_data():
    conn = _conn()
    stats = {}

    checked = conn.execute("SELECT COUNT(*) FROM maps WHERE disq_status='checked'").fetchone()[0]
    stats["checked"] = checked

    # Violations per criterion
    stats["violations"] = {}
    for cid in CRITERIA_IDS:
        vlm = conn.execute(f"SELECT COUNT(*) FROM maps WHERE disq_{cid}=1").fetchone()[0]
        human_override = conn.execute(
            f"SELECT COUNT(*) FROM maps WHERE human_{cid} IS NOT NULL AND human_{cid} != disq_{cid}"
        ).fetchone()[0]
        final = conn.execute(
            f"SELECT COUNT(*) FROM maps WHERE COALESCE(human_{cid}, disq_{cid})=1"
        ).fetchone()[0]
        reviewed = conn.execute(
            f"SELECT COUNT(*) FROM maps WHERE human_{cid} IS NOT NULL"
        ).fetchone()[0]
        stats["violations"][cid] = {
            "vlm": vlm, "final": final, "overrides": human_override, "reviewed": reviewed
        }

    # Quality score histogram
    stats["score_hist"] = {}
    for lo in range(0, 65, 5):
        hi = lo + 5
        n = conn.execute(
            "SELECT COUNT(*) FROM maps WHERE quality_score >= ? AND quality_score < ? AND disq_status='checked'",
            (lo, hi)
        ).fetchone()[0]
        stats["score_hist"][f"{lo}-{hi}"] = n

    stats["avg_score"] = conn.execute(
        "SELECT AVG(quality_score) FROM maps WHERE disq_status='checked'"
    ).fetchone()[0] or 0

    # Violations count distribution
    stats["viol_dist"] = {}
    for n in range(11):
        c = conn.execute(
            "SELECT COUNT(*) FROM maps WHERE violations_count=? AND disq_status='checked'", (n,)
        ).fetchone()[0]
        if c > 0:
            stats["viol_dist"][str(n)] = c

    conn.close()
    return stats
