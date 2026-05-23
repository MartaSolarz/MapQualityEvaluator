"""
SQLite database for stage 6 pipeline: disqualifying criteria check.
"""

import sqlite3
import json
from datetime import datetime
from pathlib import Path

import config


def get_connection(db_path=None):
    path = db_path or config.DB_PATH
    conn = sqlite3.connect(str(path))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path=None):
    conn = get_connection(db_path)
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS maps (
            uid TEXT PRIMARY KEY,
            url TEXT,
            domain TEXT,
            pred_proba REAL,
            image_width INTEGER,
            image_height INTEGER,
            local_path TEXT,
            ai_map_language TEXT,
            ai_has_choropleth INTEGER,
            ai_has_proportional_circles INTEGER,
            ai_has_graduated_circles INTEGER,
            ai_has_structural_circles INTEGER,

            -- Disqualifying criteria check
            disq_status TEXT DEFAULT 'pending',

            -- VLM results (1=violation, 0=OK, NULL=not checked)
            disq_D1 INTEGER,
            disq_D2 INTEGER,
            disq_A1 INTEGER,
            disq_A2 INTEGER,
            disq_A3 INTEGER,
            disq_V1 INTEGER,
            disq_V2 INTEGER,
            disq_L1 INTEGER,
            disq_L2 INTEGER,
            disq_L3 INTEGER,

            disq_confidence TEXT,
            disq_description TEXT,
            disq_raw_response TEXT,
            disq_checked_at TEXT,
            disq_input_tokens INTEGER,
            disq_output_tokens INTEGER,

            -- Human validation overrides (NULL=not reviewed, 0=OK, 1=violation)
            human_D1 INTEGER,
            human_D2 INTEGER,
            human_A1 INTEGER,
            human_A2 INTEGER,
            human_A3 INTEGER,
            human_V1 INTEGER,
            human_V2 INTEGER,
            human_L1 INTEGER,
            human_L2 INTEGER,
            human_L3 INTEGER,
            human_reviewed_at TEXT,

            -- Computed quality score
            violations_count INTEGER,
            quality_score REAL,

            imported_at TEXT DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS pipeline_runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            step TEXT NOT NULL,
            params TEXT,
            started_at TEXT,
            finished_at TEXT,
            items_processed INTEGER DEFAULT 0,
            items_success INTEGER DEFAULT 0,
            items_failed INTEGER DEFAULT 0,
            notes TEXT
        );

        CREATE INDEX IF NOT EXISTS idx_disq_status ON maps(disq_status);
        CREATE INDEX IF NOT EXISTS idx_quality_score ON maps(quality_score);
        CREATE INDEX IF NOT EXISTS idx_domain ON maps(domain);
        CREATE INDEX IF NOT EXISTS idx_language ON maps(ai_map_language);
    """)
    conn.close()


# --- Pipeline runs ---

def start_run(conn, step, params=None):
    cur = conn.execute(
        "INSERT INTO pipeline_runs (step, params, started_at) VALUES (?, ?, ?)",
        (step, json.dumps(params) if params else None, datetime.now().isoformat())
    )
    conn.commit()
    return cur.lastrowid


def finish_run(conn, run_id, processed=0, success=0, failed=0, notes=None):
    conn.execute(
        """UPDATE pipeline_runs
           SET finished_at=?, items_processed=?, items_success=?, items_failed=?, notes=?
           WHERE id=?""",
        (datetime.now().isoformat(), processed, success, failed, notes, run_id)
    )
    conn.commit()


# --- Map operations ---

def import_from_stage5(conn):
    """Import maps from stage5 dataset_selected table."""
    s5 = sqlite3.connect(str(config.STAGE5_DB))
    s5.row_factory = sqlite3.Row

    rows = s5.execute("""
        SELECT uid, url, domain, pred_proba, image_width, image_height,
               local_path, ai_map_language,
               ai_has_choropleth, ai_has_proportional_circles,
               ai_has_graduated_circles, ai_has_structural_circles
        FROM dataset_selected
    """).fetchall()
    s5.close()

    inserted = 0
    for r in rows:
        try:
            conn.execute("""
                INSERT OR IGNORE INTO maps
                    (uid, url, domain, pred_proba, image_width, image_height,
                     local_path, ai_map_language,
                     ai_has_choropleth, ai_has_proportional_circles,
                     ai_has_graduated_circles, ai_has_structural_circles)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
            """, (
                r["uid"], r["url"], r["domain"], r["pred_proba"],
                r["image_width"], r["image_height"], r["local_path"],
                r["ai_map_language"],
                r["ai_has_choropleth"], r["ai_has_proportional_circles"],
                r["ai_has_graduated_circles"], r["ai_has_structural_circles"],
            ))
            inserted += 1
        except sqlite3.IntegrityError:
            pass
    conn.commit()
    return inserted, len(rows)


def get_pending(conn, limit=None):
    """Get maps pending disqualifying check."""
    q = "SELECT * FROM maps WHERE disq_status = 'pending'"
    if limit:
        q += f" ORDER BY RANDOM() LIMIT {limit}"
    return conn.execute(q).fetchall()


def update_disq_status(conn, uid, status, vlm_response=None,
                       input_tokens=None, output_tokens=None):
    """Update disqualifying criteria check result."""
    if vlm_response is None:
        vlm_response = {}

    def _b(val):
        if val is None:
            return None
        if isinstance(val, bool):
            return 1 if val else 0
        if isinstance(val, str):
            return 1 if val.lower() == "true" else 0
        return int(bool(val))

    conn.execute("""
        UPDATE maps SET
            disq_status=?,
            disq_D1=?, disq_D2=?,
            disq_A1=?, disq_A2=?, disq_A3=?,
            disq_V1=?, disq_V2=?,
            disq_L1=?, disq_L2=?, disq_L3=?,
            disq_confidence=?, disq_description=?,
            disq_raw_response=?,
            disq_checked_at=?,
            disq_input_tokens=?, disq_output_tokens=?
        WHERE uid=?
    """, (
        status,
        _b(vlm_response.get("disq_D1")), _b(vlm_response.get("disq_D2")),
        _b(vlm_response.get("disq_A1")), _b(vlm_response.get("disq_A2")),
        _b(vlm_response.get("disq_A3")),
        _b(vlm_response.get("disq_V1")), _b(vlm_response.get("disq_V2")),
        _b(vlm_response.get("disq_L1")), _b(vlm_response.get("disq_L2")),
        _b(vlm_response.get("disq_L3")),
        vlm_response.get("confidence"),
        vlm_response.get("brief_description"),
        json.dumps(vlm_response),
        datetime.now().isoformat(),
        input_tokens, output_tokens,
        uid,
    ))


def set_human_override(conn, uid, criterion_id, value):
    """Set human override for a specific criterion. value: 0=OK, 1=violation."""
    col = f"human_{criterion_id}"
    conn.execute(
        f"UPDATE maps SET {col}=?, human_reviewed_at=? WHERE uid=?",
        (value, datetime.now().isoformat(), uid)
    )
    conn.commit()


def update_quality_score(conn, uid, score, violations_count):
    conn.execute(
        "UPDATE maps SET quality_score=?, violations_count=? WHERE uid=?",
        (score, violations_count, uid)
    )


def recompute_all_scores(conn):
    """Recompute quality scores for all checked maps (using human overrides where available)."""
    from criteria import get_final_violations, compute_quality_score

    rows = conn.execute("SELECT * FROM maps WHERE disq_status = 'checked'").fetchall()
    for row in rows:
        violations = get_final_violations(dict(row))
        q, count, _ = compute_quality_score(violations)
        update_quality_score(conn, row["uid"], round(q, 2), count)
    conn.commit()
    return len(rows)
