#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Predict na 50M embeddings - OPTIMIZED VERSION

Zapisuje TYLKO próbki z pred_proba > 0.5 (threshold_save)
Output: uid, url, score, pred_proba

Expected output size: ~1.5M próbek × 100 bytes = ~150 MB (vs 2GB!)
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from tqdm import tqdm
import multiprocessing as mp
from functools import partial
import time
import json
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')


def predict_chunk(chunk_id, embeddings_dir, source_data_dir, model_path,
                  threshold_save=0.5, output_dir='predictions_50M'):
    """
    Process one chunk of embeddings

    Args:
        chunk_id: ID chunka do przetworzenia
        embeddings_dir: Folder z embeddingami
        source_data_dir: Folder z metadata (urls, scores)
        model_path: Ścieżka do modelu
        threshold_save: Minimum proba żeby zapisać (default: 0.5)
        output_dir: Folder output

    Returns:
        dict ze statystykami chunka
    """

    try:
        # Load model (każdy worker ładuje osobno)
        model_pkg = joblib.load(model_path)
        model = model_pkg['model']
        scaler = model_pkg['scaler']

        # Load embeddings chunk
        embedding_file = embeddings_dir / f'chunk_{chunk_id:04d}.parquet'
        if not embedding_file.exists():
            return {
                'chunk_id': chunk_id,
                'status': 'missing',
                'error': f'File not found: {embedding_file}'
            }

        df_embeddings = pd.read_parquet(embedding_file)

        # Extract embeddings matrix
        X = np.stack(df_embeddings['l14_img'].values)

        # Scale if needed
        if scaler:
            X = scaler.transform(X)

        # Predict probabilities
        proba = model.predict_proba(X)[:, 1]

        # Filter: tylko pred_proba > threshold_save
        mask = proba > threshold_save
        n_total = len(proba)
        n_saved = mask.sum()

        if n_saved > 0:
            # Get UIDs dla filtered samples
            uids_filtered = df_embeddings.loc[mask, 'uid'].values
            proba_filtered = proba[mask]

            # Load metadata (urls, scores) - TYLKO dla filtered UIDs
            metadata_file = source_data_dir / f'chunk_{chunk_id:04d}.parquet'

            if metadata_file.exists():
                df_metadata = pd.read_parquet(metadata_file)

                # Filter metadata do tylko tych UIDs które passed threshold
                df_metadata_filtered = df_metadata[df_metadata['uid'].isin(uids_filtered)]

                # Merge predictions
                df_results = df_metadata_filtered[['uid', 'url', 'score']].copy()

                # Add predictions (match by uid)
                uid_to_proba = dict(zip(uids_filtered, proba_filtered))
                df_results['pred_proba'] = df_results['uid'].map(uid_to_proba)

            else:
                # Fallback: brak metadata
                df_results = pd.DataFrame({
                    'uid': uids_filtered,
                    'url': None,
                    'score': None,
                    'pred_proba': proba_filtered
                })

            # Sort by pred_proba (highest first)
            df_results = df_results.sort_values('pred_proba', ascending=False)

            # Save chunk results
            output_file = Path(output_dir) / f'chunk_{chunk_id:04d}.parquet'
            df_results.to_parquet(output_file, index=False)

        else:
            # Nic nie passed threshold - nie zapisuj pliku
            pass

        return {
            'chunk_id': chunk_id,
            'status': 'success',
            'n_total': n_total,
            'n_saved': n_saved,
            'save_rate': n_saved / n_total if n_total > 0 else 0,
            'mean_proba': float(np.mean(proba)),
            'max_proba': float(np.max(proba)),
            'processing_time': 0  # Will be set by caller
        }

    except Exception as e:
        return {
            'chunk_id': chunk_id,
            'status': 'error',
            'error': str(e)
        }


def combine_results(output_dir, n_chunks, threshold_final=0.6):
    """
    Combine all chunk results into final files

    Args:
        output_dir: Folder z chunk results
        n_chunks: Liczba chunków
        threshold_final: Threshold dla final classification (default: 0.6)

    Returns:
        DataFrame z combined results
    """

    print("\n" + "=" * 80)
    print("🔗 COMBINING CHUNK RESULTS")
    print("=" * 80)

    all_results = []
    chunks_found = 0

    for chunk_id in tqdm(range(n_chunks), desc="Loading chunks"):
        chunk_file = Path(output_dir) / f'chunk_{chunk_id:04d}.parquet'

        if chunk_file.exists():
            df_chunk = pd.read_parquet(chunk_file)
            all_results.append(df_chunk)
            chunks_found += 1

    print(f"✅ Loaded {chunks_found}/{n_chunks} chunks")

    if not all_results:
        print("❌ No results to combine!")
        return None

    # Combine all chunks
    df_combined = pd.concat(all_results, ignore_index=True)

    # Sort by pred_proba (highest first)
    df_combined = df_combined.sort_values('pred_proba', ascending=False)

    print(f"\n📊 Combined statistics:")
    print(f"   Total samples (proba > 0.5): {len(df_combined):,}")
    print(f"   Mean probability: {df_combined['pred_proba'].mean():.4f}")
    print(f"   Median probability: {df_combined['pred_proba'].median():.4f}")

    # Save full combined results
    output_file = Path(output_dir) / 'all_predictions.parquet'
    df_combined.to_parquet(output_file, index=False)
    print(f"   ✅ Saved: {output_file}")

    # Create filtered versions by confidence
    print(f"\n📂 Creating confidence tiers...")

    # High confidence (≥0.8)
    df_high = df_combined[df_combined['pred_proba'] >= 0.8].copy()
    df_high['confidence_tier'] = 'HIGH'
    if len(df_high) > 0:
        output_file = Path(output_dir) / 'high_confidence_maps.parquet'
        df_high.to_parquet(output_file, index=False)
        print(f"   🏆 High (≥0.8): {len(df_high):,} maps → {output_file.name}")

    # Medium confidence (0.6-0.8)
    df_medium = df_combined[
        (df_combined['pred_proba'] >= threshold_final) &
        (df_combined['pred_proba'] < 0.8)
        ].copy()
    df_medium['confidence_tier'] = 'MEDIUM'
    if len(df_medium) > 0:
        output_file = Path(output_dir) / 'medium_confidence_maps.parquet'
        df_medium.to_parquet(output_file, index=False)
        print(f"   🟡 Medium (0.6-0.8): {len(df_medium):,} maps → {output_file.name}")

    # Combined high+medium
    df_final = pd.concat([df_high, df_medium], ignore_index=True)
    df_final = df_final.sort_values('pred_proba', ascending=False)
    if len(df_final) > 0:
        output_file = Path(output_dir) / 'final_detected_maps.parquet'
        df_final.to_parquet(output_file, index=False)
        print(f"   ✅ Final (≥{threshold_final}): {len(df_final):,} maps → {output_file.name}")

    # Low confidence (0.5-0.6) - for reference
    df_low = df_combined[
        (df_combined['pred_proba'] >= 0.5) &
        (df_combined['pred_proba'] < threshold_final)
        ].copy()
    if len(df_low) > 0:
        output_file = Path(output_dir) / 'low_confidence_maps.parquet'
        df_low.to_parquet(output_file, index=False)
        print(f"   🔵 Low (0.5-0.6): {len(df_low):,} maps → {output_file.name}")

    return df_combined


def main():
    """Main prediction pipeline"""

    # ========================================
    # CONFIGURATION
    # ========================================

    # ZMIEŃ TE ŚCIEŻKI:
    EMBEDDINGS_DIR = Path('/path/to/50M/embeddings')  # Folder z l14_img embeddings
    SOURCE_DATA_DIR = Path('/path/to/50M/metadata')  # Folder z uid, url, score
    MODEL_PATH = Path('models/model_final.pkl')  # Twój final model
    OUTPUT_DIR = Path('predictions_50M')  # Output folder

    # Processing parameters
    N_CHUNKS = 1000  # Liczba chunków (50M / 50k per chunk)
    N_WORKERS = mp.cpu_count() - 1  # Use all cores minus 1
    THRESHOLD_SAVE = 0.5  # Minimum proba do zapisania
    THRESHOLD_FINAL = 0.6  # Threshold dla final classification

    # ========================================
    # VALIDATION
    # ========================================

    print("=" * 80)
    print("🚀 50M PREDICTION PIPELINE - OPTIMIZED")
    print("=" * 80)

    print(f"\n📋 Configuration:")
    print(f"   Embeddings dir:  {EMBEDDINGS_DIR}")
    print(f"   Source data dir: {SOURCE_DATA_DIR}")
    print(f"   Model:           {MODEL_PATH}")
    print(f"   Output dir:      {OUTPUT_DIR}")
    print(f"   Chunks:          {N_CHUNKS:,}")
    print(f"   Workers:         {N_WORKERS}")
    print(f"   Save threshold:  ≥{THRESHOLD_SAVE}")
    print(f"   Final threshold: ≥{THRESHOLD_FINAL}")

    # Check paths
    if not EMBEDDINGS_DIR.exists():
        print(f"\n❌ Embeddings dir not found: {EMBEDDINGS_DIR}")
        return

    if not MODEL_PATH.exists():
        print(f"\n❌ Model not found: {MODEL_PATH}")
        return

    # Create output dir
    OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

    # ========================================
    # LOAD MODEL (test)
    # ========================================

    print(f"\n📦 Loading model...")
    try:
        model_pkg = joblib.load(MODEL_PATH)
        print(f"   ✅ Model: {model_pkg['name']}")
        print(f"   ✅ Optimal threshold: {model_pkg.get('optimal_threshold', 'N/A'):.4f}")
    except Exception as e:
        print(f"   ❌ Error loading model: {e}")
        return

    # ========================================
    # PARALLEL PROCESSING
    # ========================================

    print(f"\n🔮 Processing {N_CHUNKS:,} chunks with {N_WORKERS} workers...")
    print(f"   Saving only samples with pred_proba > {THRESHOLD_SAVE}")

    start_time = time.time()

    # Prepare function with fixed parameters
    predict_fn = partial(
        predict_chunk,
        embeddings_dir=EMBEDDINGS_DIR,
        source_data_dir=SOURCE_DATA_DIR,
        model_path=MODEL_PATH,
        threshold_save=THRESHOLD_SAVE,
        output_dir=OUTPUT_DIR
    )

    # Process chunks in parallel
    with mp.Pool(N_WORKERS) as pool:
        results = list(tqdm(
            pool.imap(predict_fn, range(N_CHUNKS)),
            total=N_CHUNKS,
            desc="   Processing"
        ))

    processing_time = time.time() - start_time

    # ========================================
    # STATISTICS FROM CHUNKS
    # ========================================

    print(f"\n📊 Processing statistics:")

    successful = [r for r in results if r.get('status') == 'success']
    failed = [r for r in results if r.get('status') == 'error']
    missing = [r for r in results if r.get('status') == 'missing']

    print(f"   Successful: {len(successful)}/{N_CHUNKS}")
    if failed:
        print(f"   ❌ Failed: {len(failed)}")
        for r in failed[:5]:  # Show first 5 errors
            print(f"      Chunk {r['chunk_id']}: {r.get('error', 'Unknown')}")
    if missing:
        print(f"   ⚠️  Missing: {len(missing)}")

    if successful:
        total_samples = sum(r['n_total'] for r in successful)
        total_saved = sum(r['n_saved'] for r in successful)
        avg_save_rate = np.mean([r['save_rate'] for r in successful])
        avg_mean_proba = np.mean([r['mean_proba'] for r in successful])

        print(f"\n   Total samples processed: {total_samples:,}")
        print(f"   Samples saved (>{THRESHOLD_SAVE}): {total_saved:,} ({avg_save_rate:.2%})")
        print(f"   Average probability: {avg_mean_proba:.4f}")
        print(f"   Processing time: {processing_time:.1f}s ({processing_time / 60:.1f} min)")
        print(f"   Throughput: {total_samples / processing_time:,.0f} samples/sec")

    # ========================================
    # COMBINE RESULTS
    # ========================================

    df_combined = combine_results(OUTPUT_DIR, N_CHUNKS, THRESHOLD_FINAL)

    if df_combined is None:
        print("\n❌ No results to combine!")
        return

    # ========================================
    # FINAL STATISTICS & REPORT
    # ========================================

    print("\n" + "=" * 80)
    print("🎉 FINAL RESULTS")
    print("=" * 80)

    n_high = (df_combined['pred_proba'] >= 0.8).sum()
    n_medium = ((df_combined['pred_proba'] >= THRESHOLD_FINAL) &
                (df_combined['pred_proba'] < 0.8)).sum()
    n_low = ((df_combined['pred_proba'] >= THRESHOLD_SAVE) &
             (df_combined['pred_proba'] < THRESHOLD_FINAL)).sum()

    print(f"\n📊 Detection summary:")
    print(f"   Total candidates (>{THRESHOLD_SAVE}): {len(df_combined):,}")
    print(f"\n   High confidence (≥0.8):     {n_high:>10,}  (~95% precision)")
    print(f"   Medium confidence (0.6-0.8): {n_medium:>10,}  (~82% precision)")
    print(f"   {'─' * 60}")
    print(f"   FINAL MAPS (≥{THRESHOLD_FINAL}):        {n_high + n_medium:>10,}")
    print(f"\n   Low confidence (0.5-0.6):    {n_low:>10,}  (~65% precision)")

    # Expected true maps
    expected_true = int(n_high * 0.95 + n_medium * 0.82)
    print(f"\n   ✨ Expected TRUE maps:       {expected_true:>10,}")

    # File sizes
    print(f"\n💾 Output files:")
    for file in sorted(OUTPUT_DIR.glob('*.parquet')):
        if file.name.startswith('chunk_'):
            continue
        size_mb = file.stat().st_size / (1024 * 1024)
        print(f"   {file.name:40s} {size_mb:>8.1f} MB")

    # Save summary statistics
    summary = {
        'timestamp': datetime.now().isoformat(),
        'configuration': {
            'embeddings_dir': str(EMBEDDINGS_DIR),
            'model': str(MODEL_PATH),
            'n_chunks': N_CHUNKS,
            'n_workers': N_WORKERS,
            'threshold_save': THRESHOLD_SAVE,
            'threshold_final': THRESHOLD_FINAL
        },
        'processing': {
            'total_time_seconds': processing_time,
            'chunks_successful': len(successful),
            'chunks_failed': len(failed),
            'chunks_missing': len(missing),
            'samples_processed': total_samples if successful else 0,
            'throughput_samples_per_sec': total_samples / processing_time if successful else 0
        },
        'results': {
            'total_candidates': len(df_combined),
            'high_confidence': n_high,
            'medium_confidence': n_medium,
            'low_confidence': n_low,
            'final_maps': n_high + n_medium,
            'expected_true_maps': expected_true
        },
        'statistics': {
            'mean_proba': float(df_combined['pred_proba'].mean()),
            'median_proba': float(df_combined['pred_proba'].median()),
            'min_proba': float(df_combined['pred_proba'].min()),
            'max_proba': float(df_combined['pred_proba'].max())
        }
    }

    summary_file = OUTPUT_DIR / 'prediction_summary.json'
    with open(summary_file, 'w') as f:
        json.dump(summary, indent=2, fp=f)

    print(f"\n📄 Summary saved: {summary_file}")

    print("\n" + "=" * 80)
    print("✅ PIPELINE COMPLETE!")
    print("=" * 80)

    total_time = time.time() - start_time
    print(f"\nTotal runtime: {total_time:.1f}s ({total_time / 60:.1f} min)")
    print(f"\n🎯 Next steps:")
    print(f"   1. Review: {OUTPUT_DIR / 'final_detected_maps.parquet'}")
    print(f"   2. High quality: {OUTPUT_DIR / 'high_confidence_maps.parquet'}")
    print(f"   3. Check summary: {summary_file}")


if __name__ == '__main__':
    main()