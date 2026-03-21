#!/usr/bin/env python3
"""
Analiza wyników z large scale inference

Użycie:
    python3 analyze_predictions.py results/predictions_50M.parquet
"""

import pandas as pd
import numpy as np
import argparse
import json
from pathlib import Path
import matplotlib.pyplot as plt


def load_predictions(file_path):
    """Wczytaj predictions"""
    print(f"📂 Wczytywanie: {file_path}")
    df = pd.read_parquet(file_path)
    print(f"   Loaded: {len(df):,} predictions")
    return df


def basic_stats(df):
    """Podstawowe statystyki"""
    print("\n" + "="*80)
    print("📊 PODSTAWOWE STATYSTYKI")
    print("="*80)
    
    print(f"\nTotal predictions: {len(df):,}")
    print(f"Kolumny: {list(df.columns)}")
    
    print(f"\n📈 Pred_proba statistics:")
    print(f"   Min:    {df['pred_proba'].min():.4f}")
    print(f"   Q25:    {df['pred_proba'].quantile(0.25):.4f}")
    print(f"   Median: {df['pred_proba'].median():.4f}")
    print(f"   Mean:   {df['pred_proba'].mean():.4f}")
    print(f"   Q75:    {df['pred_proba'].quantile(0.75):.4f}")
    print(f"   Max:    {df['pred_proba'].max():.4f}")
    print(f"   Std:    {df['pred_proba'].std():.4f}")
    
    if 'score' in df.columns:
        print(f"\n📊 Dictionary score statistics:")
        print(f"   Min:    {df['score'].min():.3f}")
        print(f"   Median: {df['score'].median():.3f}")
        print(f"   Mean:   {df['score'].mean():.3f}")
        print(f"   Max:    {df['score'].max():.3f}")


def confidence_breakdown(df):
    """Rozkład wg confidence"""
    print("\n" + "="*80)
    print("🎯 CONFIDENCE BREAKDOWN")
    print("="*80)
    
    bins = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    labels = ['0.5-0.6', '0.6-0.7', '0.7-0.8', '0.8-0.9', '0.9-1.0']
    
    df['confidence_bin'] = pd.cut(df['pred_proba'], bins=bins, labels=labels, include_lowest=True)
    
    breakdown = df['confidence_bin'].value_counts().sort_index()
    
    print("\nRozkład pred_proba:")
    print(f"{'Bin':<12} {'Count':>12} {'Percentage':>12}")
    print("-" * 40)
    
    for label in labels:
        count = breakdown.get(label, 0)
        pct = count / len(df) * 100
        print(f"{label:<12} {count:>12,} {pct:>11.1f}%")
    
    print(f"\n{'HIGH (>= 0.8)':<12} {df[df['pred_proba'] >= 0.8].shape[0]:>12,} {df[df['pred_proba'] >= 0.8].shape[0]/len(df)*100:>11.1f}%")
    print(f"{'MEDIUM (0.6-0.8)':<12} {df[(df['pred_proba'] >= 0.6) & (df['pred_proba'] < 0.8)].shape[0]:>12,} {df[(df['pred_proba'] >= 0.6) & (df['pred_proba'] < 0.8)].shape[0]/len(df)*100:>11.1f}%")
    print(f"{'LOW (0.5-0.6)':<12} {df[(df['pred_proba'] >= 0.5) & (df['pred_proba'] < 0.6)].shape[0]:>12,} {df[(df['pred_proba'] >= 0.5) & (df['pred_proba'] < 0.6)].shape[0]/len(df)*100:>11.1f}%")


def top_predictions(df, n=20):
    """Top N predictions"""
    print("\n" + "="*80)
    print(f"🏆 TOP {n} PREDICTIONS (highest confidence)")
    print("="*80)
    
    top_df = df.nlargest(n, 'pred_proba')
    
    print(f"\n{'Rank':<6} {'UID':<35} {'Pred_proba':<12} {'Score':<10}")
    print("-" * 70)
    
    for idx, (i, row) in enumerate(top_df.iterrows(), 1):
        uid_short = row['uid'][:32] + "..."
        score_str = f"{row['score']:.2f}" if 'score' in row else "N/A"
        print(f"{idx:<6} {uid_short:<35} {row['pred_proba']:<12.4f} {score_str:<10}")


def score_vs_proba_analysis(df):
    """Analiza score vs pred_proba"""
    if 'score' not in df.columns:
        return
    
    print("\n" + "="*80)
    print("📊 SCORE vs PRED_PROBA CORRELATION")
    print("="*80)
    
    correlation = df[['score', 'pred_proba']].corr().iloc[0, 1]
    print(f"\nCorrelation: {correlation:.3f}")
    
    # Breakdown by score bins
    score_bins = [-1, 0, 1, 2, 5, 10, 100]
    df['score_bin'] = pd.cut(df['score'], bins=score_bins, include_lowest=True)
    
    print(f"\nMean pred_proba by score bin:")
    print(f"{'Score bin':<15} {'Count':>12} {'Mean pred_proba':>18}")
    print("-" * 50)
    
    for score_bin in df['score_bin'].cat.categories:
        bin_df = df[df['score_bin'] == score_bin]
        if len(bin_df) > 0:
            print(f"{str(score_bin):<15} {len(bin_df):>12,} {bin_df['pred_proba'].mean():>18.4f}")


def save_high_confidence(df, output_path, threshold=0.8):
    """Zapisz high confidence predictions"""
    print("\n" + "="*80)
    print(f"💾 SAVING HIGH CONFIDENCE (>= {threshold})")
    print("="*80)
    
    df_high = df[df['pred_proba'] >= threshold].copy()
    
    print(f"\nHigh confidence predictions: {len(df_high):,} ({len(df_high)/len(df)*100:.1f}%)")
    
    if len(df_high) > 0:
        output_file = output_path.parent / f"{output_path.stem}_high_conf.parquet"
        df_high.to_parquet(output_file, index=False)
        
        file_size_mb = output_file.stat().st_size / (1024 * 1024)
        
        print(f"✅ Saved: {output_file}")
        print(f"   Size: {file_size_mb:.2f} MB")
        print(f"   Predictions: {len(df_high):,}")
        print(f"   Mean pred_proba: {df_high['pred_proba'].mean():.4f}")


def load_stats(predictions_path):
    """Wczytaj stats file"""
    stats_file = predictions_path.parent / f"{predictions_path.stem}_stats.json"
    
    if not stats_file.exists():
        return None
    
    with open(stats_file) as f:
        return json.load(f)


def print_processing_stats(stats):
    """Wyświetl statystyki przetwarzania"""
    if not stats:
        return
    
    print("\n" + "="*80)
    print("⚙️  PROCESSING STATISTICS")
    print("="*80)
    
    print(f"\n📁 Files:")
    print(f"   Total:      {stats['files']['total']}")
    print(f"   Successful: {stats['files']['successful']} ✅")
    print(f"   Errors:     {stats['files']['errors']}")
    print(f"   Skipped:    {stats['files']['skipped']}")
    
    print(f"\n📊 Samples:")
    print(f"   Total processed: {stats['samples']['total']:,}")
    print(f"   Positive (saved): {stats['samples']['positive']:,}")
    print(f"   Positive rate: {stats['samples']['positive_rate']:.2%}")
    
    print(f"\n⏱️  Performance:")
    print(f"   Elapsed time: {stats['elapsed_time_seconds']:.1f}s ({stats['elapsed_time_seconds']/3600:.2f}h)")
    print(f"   Throughput: {stats['throughput']['samples_per_second']:.0f} samples/sec")
    print(f"   Files/min: {stats['throughput']['files_per_minute']:.1f}")


def main():
    parser = argparse.ArgumentParser(description='Analyze large scale predictions')
    parser.add_argument('predictions_file', type=str,
                       help='Path to predictions parquet file')
    parser.add_argument('--save-high-conf', action='store_true',
                       help='Save high confidence (>= 0.8) to separate file')
    parser.add_argument('--high-conf-threshold', type=float, default=0.8,
                       help='Threshold for high confidence (default: 0.8)')
    parser.add_argument('--top-n', type=int, default=20,
                       help='Number of top predictions to show (default: 20)')
    
    args = parser.parse_args()
    
    predictions_path = Path(args.predictions_file)
    
    if not predictions_path.exists():
        print(f"❌ File not found: {predictions_path}")
        return
    
    print("\n" + "="*80)
    print("📊 PREDICTIONS ANALYSIS")
    print("="*80)
    print(f"\nFile: {predictions_path}")
    
    # Load data
    df = load_predictions(predictions_path)
    
    # Load processing stats
    stats = load_stats(predictions_path)
    
    # Analyses
    basic_stats(df)
    confidence_breakdown(df)
    top_predictions(df, n=args.top_n)
    score_vs_proba_analysis(df)
    
    # Processing stats
    print_processing_stats(stats)
    
    # Save high confidence
    if args.save_high_conf:
        save_high_confidence(df, predictions_path, threshold=args.high_conf_threshold)
    
    print("\n" + "="*80)
    print("✅ ANALYSIS COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()
