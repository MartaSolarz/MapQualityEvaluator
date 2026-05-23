"""
Clean the 1.68M candidate pool:
1. Normalize URLs (strip http/https, www., trailing slash)
2. Deduplicate by normalized URL (keep highest pred_proba)
3. Exclude blacklisted domains

Input:  ../stage4/results/predictions_50M.parquet (1,675,572 records)
Output: predictions_clean.parquet (deduplicated, exclusions applied)

Usage: python3 clean_pool.py
"""

import polars as pl
from pathlib import Path
import config

INPUT = Path(__file__).parent.parent / "stage4" / "results" / "predictions_50M.parquet"
OUTPUT = Path(__file__).parent / "predictions_clean.parquet"


def main():
    print(f"Loading {INPUT}...")
    df = pl.read_parquet(INPUT)
    print(f"  Raw records: {len(df):,}")
    print()

    # 1. Normalize URLs
    print("Step 1: Normalizing URLs...")
    df = df.with_columns(
        pl.col("url")
          .str.replace(r"^https?://", "")
          .str.replace(r"^www\.", "")
          .str.strip_chars("/")
          .alias("url_norm")
    )

    # Extract domain from normalized URL
    df = df.with_columns(
        pl.col("url_norm").str.extract(r"^([^/]+)", 1).alias("domain")
    )
    print(f"  Unique raw URLs:        {df.select('url').n_unique():,}")
    print(f"  Unique normalized URLs: {df.select('url_norm').n_unique():,}")
    print()

    # 2. Deduplicate by normalized URL (keep highest pred_proba)
    print("Step 2: Deduplicating by normalized URL...")
    before = len(df)
    df = df.sort("pred_proba", descending=True).unique(subset=["url_norm"], keep="first")
    removed_dedup = before - len(df)
    print(f"  Removed duplicates: {removed_dedup:,}")
    print(f"  Remaining: {len(df):,}")
    print()

    # 3. Exclude blacklisted domains
    print(f"Step 3: Excluding domains: {config.EXCLUDE_DOMAINS}")
    before = len(df)
    for domain in config.EXCLUDE_DOMAINS:
        n = df.filter(pl.col("domain") == domain).height
        print(f"  {domain}: {n:,} images")
    df = df.filter(~pl.col("domain").is_in(config.EXCLUDE_DOMAINS))
    removed_excl = before - len(df)
    print(f"  Removed by exclusion: {removed_excl:,}")
    print(f"  Remaining: {len(df):,}")
    print()

    # Summary stats
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Input:                {before + removed_dedup:,}")
    print(f"  Removed (dedup):      {removed_dedup:,}")
    print(f"  Removed (exclusions): {removed_excl:,}")
    print(f"  Output:               {len(df):,}")
    print(f"  Unique domains:       {df.select('domain').n_unique():,}")
    print()

    # pred_proba distribution
    print("  pred_proba distribution:")
    for lo, hi in [(0.50, 0.60), (0.60, 0.70), (0.70, 0.80), (0.80, 0.90), (0.90, 1.01)]:
        n = df.filter((pl.col("pred_proba") >= lo) & (pl.col("pred_proba") < hi)).height
        print(f"    [{lo:.2f}, {hi:.2f}): {n:>10,}  ({n/len(df)*100:.1f}%)")
    print()

    # Top 10 domains
    top = df.group_by("domain").agg(pl.len().alias("n")).sort("n", descending=True).head(10)
    print("  Top 10 domains:")
    for row in top.iter_rows(named=True):
        print(f"    {row['domain']:<40} {row['n']:>7,}")
    print()

    # Save
    df = df.select(["uid", "url", "url_norm", "domain", "pred_proba", "score"])
    df.write_parquet(OUTPUT)
    size_mb = OUTPUT.stat().st_size / 1024 / 1024
    print(f"Saved to: {OUTPUT}")
    print(f"Size: {size_mb:.1f} MB")


if __name__ == "__main__":
    main()
