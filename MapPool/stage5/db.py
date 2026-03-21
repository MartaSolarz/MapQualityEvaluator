"""
SQLite database for pipeline state tracking.

Single candidates table tracks the full lifecycle of each image.
Status columns enable resumability — each step processes only 'pending' records.
"""

import sqlite3
import json
from datetime import datetime
from pathlib import Path
from contextlib import contextmanager

import config


def get_connection(db_path=None):
    """Get a SQLite connection with WAL mode for better concurrency."""
    path = db_path or config.DB_PATH
    conn = sqlite3.connect(str(path))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.row_factory = sqlite3.Row
    return conn


@contextmanager
def transaction(conn):
    """Context manager for SQLite transactions."""
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise


def init_db(db_path=None):
    """Create tables and indices if they don't exist."""
    conn = get_connection(db_path)
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS candidates (
            uid TEXT PRIMARY KEY,
            url TEXT NOT NULL,
            domain TEXT NOT NULL,
            pred_proba REAL NOT NULL,
            score REAL,
            stratum TEXT,
            batch_id INTEGER,
            source TEXT DEFAULT 'sample',

            -- Step 2: URL validation
            url_status TEXT DEFAULT 'pending',
            url_http_code INTEGER,
            url_content_type TEXT,
            url_checked_at TEXT,

            -- Step 3: Image download
            download_status TEXT DEFAULT 'pending',
            image_width INTEGER,
            image_height INTEGER,
            image_format TEXT,
            image_size_bytes INTEGER,
            local_path TEXT,
            downloaded_at TEXT,

            -- Step 4: AI formal criteria check
            ai_status TEXT DEFAULT 'pending',
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
            ai_raw_response TEXT,
            ai_checked_at TEXT,
            ai_input_tokens INTEGER,
            ai_output_tokens INTEGER,

            -- Criteria results
            passes_f1 INTEGER,
            passes_f3 INTEGER,
            passes_f4 INTEGER,
            passes_all INTEGER,

            -- Tracking
            created_at TEXT DEFAULT (datetime('now'))
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

        CREATE INDEX IF NOT EXISTS idx_url_status ON candidates(url_status);
        CREATE INDEX IF NOT EXISTS idx_download_status ON candidates(download_status);
        CREATE INDEX IF NOT EXISTS idx_ai_status ON candidates(ai_status);
        CREATE INDEX IF NOT EXISTS idx_domain ON candidates(domain);
        CREATE INDEX IF NOT EXISTS idx_pred_proba ON candidates(pred_proba);
        CREATE INDEX IF NOT EXISTS idx_passes_all ON candidates(passes_all);
        CREATE INDEX IF NOT EXISTS idx_batch_id ON candidates(batch_id);
        CREATE INDEX IF NOT EXISTS idx_source ON candidates(source);
    """)
    conn.close()


# ---------------------------------------------------------------------------
# Pipeline runs (audit trail)
# ---------------------------------------------------------------------------

def start_run(conn, step, params=None):
    """Record the start of a pipeline step. Returns run_id."""
    cur = conn.execute(
        "INSERT INTO pipeline_runs (step, params, started_at) VALUES (?, ?, ?)",
        (step, json.dumps(params) if params else None, datetime.now().isoformat())
    )
    conn.commit()
    return cur.lastrowid


def finish_run(conn, run_id, processed=0, success=0, failed=0, notes=None):
    """Record the end of a pipeline step."""
    conn.execute(
        """UPDATE pipeline_runs
           SET finished_at=?, items_processed=?, items_success=?, items_failed=?, notes=?
           WHERE id=?""",
        (datetime.now().isoformat(), processed, success, failed, notes, run_id)
    )
    conn.commit()


# ---------------------------------------------------------------------------
# Candidate operations
# ---------------------------------------------------------------------------

def insert_candidates(conn, rows, batch_id, source="sample"):
    """
    Insert candidates into DB. Skips duplicates (INSERT OR IGNORE).

    Args:
        rows: list of dicts with keys: uid, url, domain, pred_proba, score, stratum
        batch_id: pipeline_runs.id
        source: 'sample', 'manual', 'domain_add'

    Returns:
        number of newly inserted rows
    """
    before = conn.execute("SELECT COUNT(*) FROM candidates").fetchone()[0]

    with transaction(conn):
        conn.executemany(
            """INSERT OR IGNORE INTO candidates
               (uid, url, domain, pred_proba, score, stratum, batch_id, source)
               VALUES (:uid, :url, :domain, :pred_proba, :score, :stratum, :batch_id, :source)""",
            [{**r, "batch_id": batch_id, "source": source} for r in rows]
        )

    after = conn.execute("SELECT COUNT(*) FROM candidates").fetchone()[0]
    return after - before


def get_pending(conn, step, limit=None):
    """
    Get candidates pending for a given step.

    Steps chain: url_status → download_status → ai_status
    Each step requires the previous to be successful.
    """
    queries = {
        "url_check": (
            "SELECT uid, url FROM candidates WHERE url_status = 'pending'",
        ),
        "download": (
            "SELECT uid, url FROM candidates "
            "WHERE url_status = 'alive' AND download_status = 'pending'",
        ),
        "ai_check": (
            "SELECT uid, local_path, image_width, image_height FROM candidates "
            "WHERE download_status = 'success' AND ai_status = 'pending'",
        ),
    }

    query = queries[step][0]
    if limit:
        query += f" LIMIT {limit}"

    return conn.execute(query).fetchall()


def update_url_status(conn, uid, status, http_code=None, content_type=None):
    """Update URL validation result for a candidate."""
    conn.execute(
        """UPDATE candidates
           SET url_status=?, url_http_code=?, url_content_type=?, url_checked_at=?
           WHERE uid=?""",
        (status, http_code, content_type, datetime.now().isoformat(), uid)
    )


def update_download_status(conn, uid, status, width=None, height=None,
                           fmt=None, size_bytes=None, local_path=None,
                           passes_f1=None):
    """Update download result for a candidate."""
    conn.execute(
        """UPDATE candidates
           SET download_status=?, image_width=?, image_height=?, image_format=?,
               image_size_bytes=?, local_path=?, downloaded_at=?, passes_f1=?
           WHERE uid=?""",
        (status, width, height, fmt, size_bytes, str(local_path) if local_path else None,
         datetime.now().isoformat(), passes_f1, uid)
    )


def update_ai_status(conn, uid, status, ai_response=None, criteria_result=None,
                      input_tokens=None, output_tokens=None):
    """Update AI check result for a candidate."""
    if ai_response is None:
        ai_response = {}
    if criteria_result is None:
        criteria_result = {}

    conn.execute(
        """UPDATE candidates
           SET ai_status=?,
               ai_is_map=?, ai_is_statistical_map=?,
               ai_has_quantitative_data=?, ai_has_admin_units=?,
               ai_has_choropleth=?, ai_has_proportional_circles=?,
               ai_has_graduated_circles=?, ai_has_structural_circles=?,
               ai_confidence=?, ai_description=?, ai_raw_response=?,
               ai_checked_at=?, ai_input_tokens=?, ai_output_tokens=?,
               passes_f3=?, passes_f4=?, passes_all=?
           WHERE uid=?""",
        (
            status,
            _bool_to_int(ai_response.get("is_map")),
            _bool_to_int(ai_response.get("is_statistical_map")),
            _bool_to_int(ai_response.get("has_quantitative_data")),
            _bool_to_int(ai_response.get("has_admin_units")),
            _bool_to_int(ai_response.get("has_choropleth")),
            _bool_to_int(ai_response.get("has_proportional_circles")),
            _bool_to_int(ai_response.get("has_graduated_circles")),
            _bool_to_int(ai_response.get("has_structural_circles")),
            ai_response.get("confidence"),
            ai_response.get("brief_description"),
            json.dumps(ai_response),
            datetime.now().isoformat(),
            input_tokens,
            output_tokens,
            _bool_to_int(criteria_result.get("passes_f3")),
            _bool_to_int(criteria_result.get("passes_f4")),
            _bool_to_int(criteria_result.get("passes_all")),
            uid,
        )
    )


# ---------------------------------------------------------------------------
# Status / statistics
# ---------------------------------------------------------------------------

def get_stats(conn):
    """Get pipeline progress statistics."""
    stats = {}

    stats["total"] = conn.execute(
        "SELECT COUNT(*) FROM candidates"
    ).fetchone()[0]

    stats["url_alive"] = conn.execute(
        "SELECT COUNT(*) FROM candidates WHERE url_status = 'alive'"
    ).fetchone()[0]
    stats["url_dead"] = conn.execute(
        "SELECT COUNT(*) FROM candidates WHERE url_status IN ('dead', 'error')"
    ).fetchone()[0]
    stats["url_pending"] = conn.execute(
        "SELECT COUNT(*) FROM candidates WHERE url_status = 'pending'"
    ).fetchone()[0]

    stats["downloaded"] = conn.execute(
        "SELECT COUNT(*) FROM candidates WHERE download_status = 'success'"
    ).fetchone()[0]
    stats["download_failed"] = conn.execute(
        "SELECT COUNT(*) FROM candidates WHERE download_status IN ('failed', 'not_image')"
    ).fetchone()[0]
    stats["download_pending"] = conn.execute(
        "SELECT COUNT(*) FROM candidates "
        "WHERE url_status = 'alive' AND download_status = 'pending'"
    ).fetchone()[0]

    stats["ai_pass"] = conn.execute(
        "SELECT COUNT(*) FROM candidates WHERE ai_status = 'pass'"
    ).fetchone()[0]
    stats["ai_fail"] = conn.execute(
        "SELECT COUNT(*) FROM candidates WHERE ai_status = 'fail'"
    ).fetchone()[0]
    stats["ai_error"] = conn.execute(
        "SELECT COUNT(*) FROM candidates WHERE ai_status = 'error'"
    ).fetchone()[0]
    stats["ai_pending"] = conn.execute(
        "SELECT COUNT(*) FROM candidates "
        "WHERE download_status = 'success' AND ai_status = 'pending'"
    ).fetchone()[0]

    stats["passes_all"] = conn.execute(
        "SELECT COUNT(*) FROM candidates WHERE passes_all = 1"
    ).fetchone()[0]

    # Per-batch breakdown
    stats["batches"] = conn.execute(
        """SELECT pr.id, pr.step, pr.params, pr.started_at,
                  COUNT(c.uid) as candidates_count
           FROM pipeline_runs pr
           LEFT JOIN candidates c ON c.batch_id = pr.id
           WHERE pr.step LIKE '%sample%' OR pr.step LIKE '%add%'
           GROUP BY pr.id
           ORDER BY pr.id"""
    ).fetchall()

    # Resolution stats (for empirical F1 threshold)
    stats["resolution"] = conn.execute(
        """SELECT
             COUNT(*) as count,
             MIN(image_width) as min_w, MIN(image_height) as min_h,
             MAX(image_width) as max_w, MAX(image_height) as max_h,
             AVG(image_width) as avg_w, AVG(image_height) as avg_h,
             SUM(CASE WHEN image_width < 400 OR image_height < 400 THEN 1 ELSE 0 END) as below_400,
             SUM(CASE WHEN image_width < 600 OR image_height < 600 THEN 1 ELSE 0 END) as below_600,
             SUM(CASE WHEN image_width < 800 OR image_height < 800 THEN 1 ELSE 0 END) as below_800
           FROM candidates WHERE download_status = 'success'"""
    ).fetchone()

    # Method breakdown (among passing maps)
    stats["methods"] = conn.execute(
        """SELECT
             SUM(CASE WHEN ai_has_choropleth = 1 THEN 1 ELSE 0 END) as choropleth,
             SUM(CASE WHEN ai_has_proportional_circles = 1 THEN 1 ELSE 0 END) as proportional,
             SUM(CASE WHEN ai_has_graduated_circles = 1 THEN 1 ELSE 0 END) as graduated,
             SUM(CASE WHEN ai_has_structural_circles = 1 THEN 1 ELSE 0 END) as structural
           FROM candidates WHERE passes_all = 1"""
    ).fetchone()

    return stats


def get_verified_maps(conn):
    """Get all maps that passed all formal criteria."""
    return conn.execute(
        """SELECT uid, url, domain, pred_proba, score, stratum, source,
                  image_width, image_height, image_format, local_path,
                  ai_has_choropleth, ai_has_proportional_circles,
                  ai_has_graduated_circles, ai_has_structural_circles,
                  ai_confidence, ai_description
           FROM candidates
           WHERE passes_all = 1
           ORDER BY pred_proba DESC"""
    ).fetchall()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _bool_to_int(val):
    """Convert bool/None to int for SQLite storage."""
    if val is None:
        return None
    return 1 if val else 0
