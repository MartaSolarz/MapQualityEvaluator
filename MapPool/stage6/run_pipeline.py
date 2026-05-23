#!/usr/bin/env python3
"""
Stage 6 pipeline CLI: disqualifying criteria check.

Usage:
    python3 run_pipeline.py import          Import maps from stage5 dataset
    python3 run_pipeline.py check           Run disqualifying criteria check (bulk)
    python3 run_pipeline.py check --pilot N  Check N random maps (pilot)
    python3 run_pipeline.py compute-scores  Recompute quality scores (after human overrides)
    python3 run_pipeline.py status          Show pipeline progress
    python3 run_pipeline.py export          Export results to CSV
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import config
import db
from criteria import ALL_IDS, WEIGHTS, SUM_WEIGHTS


def cmd_import(args):
    """Import maps from stage5 dataset_selected."""
    print("Stage 6: Import from stage5")
    print(f"  Source: {config.STAGE5_DB}")

    if not config.STAGE5_DB.exists():
        print(f"  ERROR: Stage5 database not found: {config.STAGE5_DB}")
        return

    db.init_db()
    conn = db.get_connection()
    run_id = db.start_run(conn, "import")

    inserted, total = db.import_from_stage5(conn)

    # Verify images exist
    missing = 0
    for row in conn.execute("SELECT uid, local_path FROM maps"):
        if not row["local_path"] or not Path(row["local_path"]).exists():
            missing += 1

    db.finish_run(conn, run_id, processed=total, success=inserted,
                  notes=f"total_in_stage5={total}, inserted={inserted}, missing_images={missing}")
    conn.close()

    print(f"  Found in stage5: {total:,}")
    print(f"  Imported (new):  {inserted:,}")
    if missing:
        print(f"  WARNING: {missing:,} maps have missing image files (is external drive connected?)")
    print("  Done.")


def cmd_check(args):
    """Run disqualifying criteria check."""
    print("Stage 6: Disqualifying Criteria Check")
    db.init_db()
    from disq_check import run_disq_check
    run_disq_check(pilot=args.pilot, model_override=args.model, uids=args.uids)


def cmd_compute_scores(args):
    """Recompute quality scores (e.g. after human overrides)."""
    print("Stage 6: Recomputing quality scores")
    conn = db.get_connection()
    n = db.recompute_all_scores(conn)
    conn.close()
    print(f"  Recomputed scores for {n:,} maps")


def cmd_status(args):
    """Show pipeline status."""
    print("Stage 6: Pipeline Status")
    print("=" * 50)

    conn = db.get_connection()

    # Basic counts
    total = conn.execute("SELECT COUNT(*) FROM maps").fetchone()[0]
    checked = conn.execute("SELECT COUNT(*) FROM maps WHERE disq_status='checked'").fetchone()[0]
    pending = conn.execute("SELECT COUNT(*) FROM maps WHERE disq_status='pending'").fetchone()[0]
    errors = conn.execute("SELECT COUNT(*) FROM maps WHERE disq_status='error'").fetchone()[0]

    print(f"\n  Maps:")
    print(f"    Total:   {total:,}")
    print(f"    Checked: {checked:,}")
    print(f"    Pending: {pending:,}")
    print(f"    Error:   {errors:,}")

    if checked > 0:
        # Violation stats
        print(f"\n  Violations (among {checked:,} checked maps):")
        for cid in ALL_IDS:
            n = conn.execute(f"SELECT COUNT(*) FROM maps WHERE disq_{cid}=1").fetchone()[0]
            pct = n / checked * 100
            print(f"    {cid} (w={WEIGHTS[cid]:.2f}): {n:>5,} ({pct:>5.1f}%)")

        # Quality score distribution
        print(f"\n  Quality score distribution:")
        for lo, hi, label in [(0, 20, "0-20 (poor)"), (20, 40, "20-40 (fair)"),
                               (40, 55, "40-55 (good)"), (55, 60, "55-60 (excellent)"),
                               (60, 61, "60 (perfect)")]:
            n = conn.execute(
                "SELECT COUNT(*) FROM maps WHERE quality_score >= ? AND quality_score < ? AND disq_status='checked'",
                (lo, hi)
            ).fetchone()[0]
            print(f"    {label}: {n:>5,}")

        avg_q = conn.execute("SELECT AVG(quality_score) FROM maps WHERE disq_status='checked'").fetchone()[0]
        print(f"\n  Average quality score: {avg_q:.1f}")

        # Human review progress
        print(f"\n  Human review progress:")
        for cid in ALL_IDS:
            reviewed = conn.execute(f"SELECT COUNT(*) FROM maps WHERE human_{cid} IS NOT NULL").fetchone()[0]
            overridden = conn.execute(
                f"SELECT COUNT(*) FROM maps WHERE human_{cid} IS NOT NULL AND human_{cid} != disq_{cid}"
            ).fetchone()[0]
            print(f"    {cid}: {reviewed:>5,} reviewed, {overridden:>3,} overridden")

    # Pipeline runs
    print(f"\n  Recent runs:")
    for r in conn.execute("SELECT * FROM pipeline_runs ORDER BY id DESC LIMIT 5").fetchall():
        print(f"    [{r['step']}] {r['started_at'][:16]} -> {r['items_processed']} processed, {r['notes'] or ''}")

    conn.close()


def cmd_export(args):
    """Export results to CSV."""
    print("Stage 6: Export")
    conn = db.get_connection()
    import csv

    output = config.BASE_DIR / "disq_results.csv"
    rows = conn.execute("SELECT * FROM maps WHERE disq_status='checked' ORDER BY quality_score").fetchall()
    cols = [d[0] for d in conn.execute("SELECT * FROM maps LIMIT 1").description]

    with open(output, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        w.writerows(rows)

    conn.close()
    print(f"  Exported {len(rows):,} maps to {output}")


def main():
    parser = argparse.ArgumentParser(
        description="Stage 6: Disqualifying Criteria Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("import", help="Import maps from stage5 dataset")

    p_check = sub.add_parser("check", help="Run disqualifying criteria check")
    p_check.add_argument("--pilot", type=int, help="Check N random maps only")
    p_check.add_argument("--model", type=str, help="Override model")
    p_check.add_argument("--uids", nargs="+", help="Check specific map UIDs (space-separated)")

    sub.add_parser("compute-scores", help="Recompute quality scores")
    sub.add_parser("status", help="Show pipeline status")
    sub.add_parser("export", help="Export results to CSV")

    args = parser.parse_args()

    if args.command == "import":
        cmd_import(args)
    elif args.command == "check":
        cmd_check(args)
    elif args.command == "compute-scores":
        cmd_compute_scores(args)
    elif args.command == "status":
        cmd_status(args)
    elif args.command == "export":
        cmd_export(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
