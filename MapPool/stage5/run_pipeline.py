#!/usr/bin/env python3
"""
Stage 5 Pipeline CLI — Statistical Map Selection & Verification.

Usage:
    python3 run_pipeline.py sample [options]        # Step 1: Smart sampling
    python3 run_pipeline.py validate-urls            # Step 2: URL validation
    python3 run_pipeline.py download                 # Step 3: Download images
    python3 run_pipeline.py ai-check [options]       # Step 4: AI formal criteria
    python3 run_pipeline.py add-urls <file>           # Add URLs manually
    python3 run_pipeline.py add-from-domain <domain>  # Add from specific domain
    python3 run_pipeline.py status                    # Pipeline progress
    python3 run_pipeline.py export                    # Export verified maps

After each add/sample, run: validate-urls → download → ai-check
Only new (pending) records are processed — safe to rerun.
"""

import argparse
import sys
from pathlib import Path

# Ensure stage5 is in path for local imports
sys.path.insert(0, str(Path(__file__).parent))

import config
import db
import pipeline
import ai_check


def cmd_sample(args):
    """Step 1: Smart sampling from predictions_50M (for testing)."""
    print("=" * 60)
    print("STEP 1: Smart Sampling (test subset)")
    print("=" * 60)

    exclude = args.exclude_domains.split(",") if args.exclude_domains else None

    db.init_db()
    inserted, skipped = pipeline.smart_sample(
        pred_proba_min=args.pred_proba_min,
        domain_cap=args.domain_cap,
        sample_target=args.sample_target,
        exclude_domains=exclude,
        seed=args.seed,
    )

    print(f"\nDone. Run 'validate-urls' next.")


def cmd_load_all(args):
    """Step 1: Load ALL candidates above threshold (full run)."""
    print("=" * 60)
    print("STEP 1: Load ALL candidates")
    print("=" * 60)

    exclude = args.exclude_domains.split(",") if args.exclude_domains else None

    db.init_db()
    inserted, skipped = pipeline.load_all(
        pred_proba_min=args.pred_proba_min,
        exclude_domains=exclude,
    )

    print(f"\nDone. Run 'validate-urls' next.")


def cmd_validate_urls(args):
    """Step 2: Validate URLs with parallel HEAD requests."""
    print("=" * 60)
    print("STEP 2: URL Validation")
    print("=" * 60)

    db.init_db()
    pipeline.validate_urls()

    print(f"\nDone. Run 'download' next.")


def cmd_download(args):
    """Step 3: Download images + F1 resolution check."""
    print("=" * 60)
    print("STEP 3: Image Download + F1 Check")
    print("=" * 60)

    db.init_db()
    pipeline.download_images()

    print(f"\nDone. Run 'ai-check' next.")


def cmd_ai_check(args):
    """Step 4: AI formal criteria check (Claude Vision)."""
    print("=" * 60)
    print("STEP 4: AI Formal Criteria Check")
    print("=" * 60)

    if args.pilot:
        print(f"  PILOT MODE: {args.pilot} images with Sonnet")
    else:
        print("  BULK MODE: all pending with Haiku")

    db.init_db()
    ai_check.run_ai_check(pilot=args.pilot, model_override=args.model)


def cmd_recheck(args):
    """Re-run AI check on specific UIDs (to test prompt changes)."""
    print("=" * 60)
    print("RECHECK — re-run AI on specific images")
    print("=" * 60)

    uids = [u.strip() for u in args.uids.split(",") if u.strip()]
    model = args.model or config.CLAUDE_MODEL_PILOT  # default Sonnet

    print(f"  UIDs: {len(uids)}")
    print(f"  Model: {model}")

    db.init_db()
    conn = db.get_connection()

    # Fetch candidates
    placeholders = ",".join("?" for _ in uids)
    rows = conn.execute(
        f"SELECT uid, local_path, image_width, image_height FROM candidates "
        f"WHERE uid IN ({placeholders})",
        uids,
    ).fetchall()

    if not rows:
        print("  No matching candidates found.")
        conn.close()
        return

    found_uids = {r["uid"] for r in rows}
    for u in uids:
        if u not in found_uids:
            print(f"  WARNING: uid '{u}' not found in DB")

    # Reset ai_status to pending for these UIDs
    conn.executemany(
        "UPDATE candidates SET ai_status='pending' WHERE uid=?",
        [(r["uid"],) for r in rows],
    )
    conn.commit()
    conn.close()

    # Run AI check — it will pick up these pending ones
    ai_check.run_ai_check(pilot=len(rows), model_override=model)

    # Show results
    conn = db.get_connection()
    for uid in uids:
        r = conn.execute(
            "SELECT uid, ai_status, ai_description, ai_is_map, ai_is_statistical_map, "
            "ai_has_quantitative_data, ai_has_admin_units, ai_has_choropleth, "
            "ai_has_proportional_circles, ai_has_graduated_circles, "
            "ai_has_structural_circles, ai_confidence, passes_f3, passes_f4, passes_all "
            "FROM candidates WHERE uid=?", (uid,)
        ).fetchone()
        if not r:
            continue
        print(f"\n  ── {r['uid'][:16]}… ──")
        print(f"  Status:      {r['ai_status'].upper()}")
        print(f"  Description: {r['ai_description']}")
        print(f"  F3: is_map={bool(r['ai_is_map'])} stat={bool(r['ai_is_statistical_map'])} "
              f"quant={bool(r['ai_has_quantitative_data'])} admin={bool(r['ai_has_admin_units'])} "
              f"→ {'PASS' if r['passes_f3'] else 'FAIL'}")
        print(f"  F4: choro={bool(r['ai_has_choropleth'])} prop={bool(r['ai_has_proportional_circles'])} "
              f"grad={bool(r['ai_has_graduated_circles'])} struct={bool(r['ai_has_structural_circles'])} "
              f"→ {'PASS' if r['passes_f4'] else 'FAIL'}")
        print(f"  Confidence:  {r['ai_confidence']}")
        print(f"  OVERALL:     {'PASS' if r['passes_all'] else 'FAIL'}")
    conn.close()


def cmd_add_urls(args):
    """Add URLs from a text file."""
    print("=" * 60)
    print("ADD URLs from file")
    print("=" * 60)

    filepath = Path(args.file)
    if not filepath.exists():
        print(f"  ERROR: File not found: {filepath}")
        return

    db.init_db()
    pipeline.add_urls_from_file(filepath)

    print(f"\nDone. Run 'validate-urls' → 'download' → 'ai-check' next.")


def cmd_add_from_domain(args):
    """Add more maps from a specific domain."""
    print("=" * 60)
    print(f"ADD from domain: {args.domain}")
    print("=" * 60)

    db.init_db()
    pipeline.add_from_domain(
        args.domain,
        limit=args.limit,
        pred_proba_min=args.pred_proba_min,
    )

    print(f"\nDone. Run 'validate-urls' → 'download' → 'ai-check' next.")


def cmd_status(args):
    """Show pipeline progress."""
    db.init_db()
    conn = db.get_connection()
    stats = db.get_stats(conn)
    conn.close()

    print("=" * 60)
    print("PIPELINE STATUS")
    print("=" * 60)

    total = stats["total"]
    if total == 0:
        print("\n  No candidates yet. Run 'sample' first.")
        return

    print(f"\n  Candidates:         {total:>8,}")
    print(f"  ─────────────────────────────")

    # URL validation
    url_done = stats["url_alive"] + stats["url_dead"]
    print(f"  URL alive:          {stats['url_alive']:>8,}  "
          f"({stats['url_alive']/total*100:.1f}%)" if total else "")
    print(f"  URL dead/error:     {stats['url_dead']:>8,}")
    print(f"  URL pending:        {stats['url_pending']:>8,}")
    print(f"  ─────────────────────────────")

    # Download
    if stats['downloaded']:
        print(f"  Downloaded:         {stats['downloaded']:>8,}")
    print(f"  Download failed:    {stats['download_failed']:>8,}")
    print(f"  Download pending:   {stats['download_pending']:>8,}")
    print(f"  ─────────────────────────────")

    # Resolution stats (F1 — metadata only, threshold TBD)
    res = stats.get("resolution")
    if res and res["count"] and res["count"] > 0:
        print(f"  Resolution (downloaded images):")
        print(f"    Range W:  {res['min_w']}–{res['max_w']} px  (avg {res['avg_w']:.0f})")
        print(f"    Range H:  {res['min_h']}–{res['max_h']} px  (avg {res['avg_h']:.0f})")
        print(f"    < 400px:  {res['below_400']:,}")
        print(f"    < 600px:  {res['below_600']:,}")
        print(f"    < 800px:  {res['below_800']:,}")
        print(f"  ─────────────────────────────")

    # AI check
    print(f"  AI pass (F3+F4):    {stats['ai_pass']:>8,}")
    print(f"  AI fail:            {stats['ai_fail']:>8,}")
    print(f"  AI error:           {stats['ai_error']:>8,}")
    print(f"  AI pending:         {stats['ai_pending']:>8,}")
    print(f"  ─────────────────────────────")

    # Final
    print(f"  VERIFIED MAPS:      {stats['passes_all']:>8,}")

    # Methods breakdown
    if stats["methods"] and stats["passes_all"] > 0:
        m = stats["methods"]
        print(f"\n  Methods (among verified):")
        print(f"    Choropleth:           {m['choropleth'] or 0:,}")
        print(f"    Proportional circles: {m['proportional'] or 0:,}")
        print(f"    Graduated circles:    {m['graduated'] or 0:,}")
        print(f"    Structural circles:   {m['structural'] or 0:,}")

    # Batches
    if stats["batches"]:
        print(f"\n  Batches:")
        for b in stats["batches"]:
            params = b["params"] or ""
            print(f"    #{b['id']} {b['step']}: {b['candidates_count']:,} candidates "
                  f"({b['started_at'][:10] if b['started_at'] else '?'})")


def cmd_export(args):
    """Export verified maps to parquet."""
    import pandas as pd

    db.init_db()
    conn = db.get_connection()
    rows = db.get_verified_maps(conn)
    conn.close()

    if not rows:
        print("  No verified maps to export.")
        return

    df = pd.DataFrame([dict(r) for r in rows])

    export_dir = Path(config.EXPORT_DIR)
    export_dir.mkdir(parents=True, exist_ok=True)
    output_path = export_dir / "verified_maps.parquet"

    df.to_parquet(output_path, index=False)
    print(f"  Exported {len(df):,} verified maps → {output_path}")
    print(f"  Columns: {list(df.columns)}")

    # Summary
    print(f"\n  Methods:")
    if "ai_has_choropleth" in df.columns:
        print(f"    Choropleth:           {df['ai_has_choropleth'].sum():,}")
        print(f"    Proportional circles: {df['ai_has_proportional_circles'].sum():,}")
        print(f"    Graduated circles:    {df['ai_has_graduated_circles'].sum():,}")
        print(f"    Structural circles:   {df['ai_has_structural_circles'].sum():,}")

    print(f"  Domains: {df['domain'].nunique():,} unique")
    print(f"  pred_proba: {df['pred_proba'].min():.3f} – {df['pred_proba'].max():.3f}")


def main():
    parser = argparse.ArgumentParser(
        description="Stage 5: Statistical Map Selection Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    subparsers = parser.add_subparsers(dest="command", help="Pipeline step")

    # sample (test subset)
    p_sample = subparsers.add_parser("sample",
        help="Stratified sample for testing (use load-all for full run)")
    p_sample.add_argument("--pred-proba-min", type=float, default=config.PRED_PROBA_MIN,
                          help=f"Min pred_proba (default: {config.PRED_PROBA_MIN})")
    p_sample.add_argument("--domain-cap", type=int, default=config.DOMAIN_CAP,
                          help=f"Max per domain (default: {config.DOMAIN_CAP})")
    p_sample.add_argument("--sample-target", type=int, default=config.SAMPLE_TARGET,
                          help=f"Target sample size (default: {config.SAMPLE_TARGET})")
    p_sample.add_argument("--exclude-domains", type=str, default=None,
                          help="Comma-separated domains to exclude")
    p_sample.add_argument("--seed", type=int, default=None,
                          help="Random seed (default: current timestamp = new sample each run)")
    p_sample.set_defaults(func=cmd_sample)

    # load-all (full run)
    p_all = subparsers.add_parser("load-all",
        help="Load ALL candidates above threshold (full run)")
    p_all.add_argument("--pred-proba-min", type=float, default=config.PRED_PROBA_MIN,
                       help=f"Min pred_proba (default: {config.PRED_PROBA_MIN})")
    p_all.add_argument("--exclude-domains", type=str, default=None,
                       help="Comma-separated domains to exclude")
    p_all.set_defaults(func=cmd_load_all)

    # validate-urls
    p_urls = subparsers.add_parser("validate-urls", help="Check URL availability")
    p_urls.set_defaults(func=cmd_validate_urls)

    # download
    p_dl = subparsers.add_parser("download", help="Download images + F1 check")
    p_dl.set_defaults(func=cmd_download)

    # ai-check
    p_ai = subparsers.add_parser("ai-check", help="AI formal criteria check")
    p_ai.add_argument("--pilot", type=int, default=None,
                      help="Pilot mode: check N images with Sonnet")
    p_ai.add_argument("--model", type=str, default=None,
                      help="Override model (e.g., claude-sonnet-4-20250514)")
    p_ai.set_defaults(func=cmd_ai_check)

    # recheck
    p_re = subparsers.add_parser("recheck",
        help="Re-run AI check on specific UIDs (test prompt changes)")
    p_re.add_argument("uids", type=str,
                      help="Comma-separated UIDs (or single UID) to re-check")
    p_re.add_argument("--model", type=str, default=None,
                      help="Model to use (default: Sonnet)")
    p_re.set_defaults(func=cmd_recheck)

    # add-urls
    p_add = subparsers.add_parser("add-urls", help="Add URLs from file")
    p_add.add_argument("file", type=str, help="Text file with URLs (one per line)")
    p_add.set_defaults(func=cmd_add_urls)

    # add-from-domain
    p_dom = subparsers.add_parser("add-from-domain", help="Add maps from a domain")
    p_dom.add_argument("domain", type=str, help="Domain name (e.g. upload.wikimedia.org)")
    p_dom.add_argument("--limit", type=int, default=200,
                       help="Max maps to add (default: 200)")
    p_dom.add_argument("--pred-proba-min", type=float, default=config.PRED_PROBA_MIN,
                       help=f"Min pred_proba (default: {config.PRED_PROBA_MIN})")
    p_dom.set_defaults(func=cmd_add_from_domain)

    # status
    p_status = subparsers.add_parser("status", help="Show pipeline progress")
    p_status.set_defaults(func=cmd_status)

    # export
    p_export = subparsers.add_parser("export", help="Export verified maps to parquet")
    p_export.set_defaults(func=cmd_export)

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    args.func(args)


if __name__ == "__main__":
    main()
