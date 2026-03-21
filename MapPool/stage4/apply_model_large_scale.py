#!/usr/bin/env python3
"""
Aplikacja modelu na dużą skalę (50M+ próbek)

Optymalizacje:
- STREAMING: pliki parquet czytane w chunkach (NIE wczytuje całości do RAM!)
- Wielowątkowe przetwarzanie plików
- Batch processing (memory efficient)
- Streaming write (nie trzyma wszystkiego w pamięci)
- Progress tracking
- Resume capability (pomija już przetworzone pliki)
- Filtrowanie pred_proba >= threshold

Wymagania RAM (dla plików z embeddingami):
- Model: ~1 GB
- Batch processing: batch_size × 768 floats × 4 bytes
  - batch_size=5000 → ~15 MB na batch
  - batch_size=10000 → ~30 MB na batch
- Bezpieczny margin: ~500 MB na worker
- Total: Model + (workers × batch_memory) + 500MB margin

Przykład:
- 1 worker, batch_size=10000: ~1.5 GB RAM
- 2 workers, batch_size=5000: ~2.5 GB RAM

Użycie:
    python3 apply_model_large_scale.py \
        --input-dir /path/to/parquet_files \
        --output-file predictions_positive.parquet \
        --threshold 0.5 \
        --batch-size 10000 \
        --workers 1
"""

import pandas as pd
import numpy as np
import joblib
import argparse
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import warnings
import gc
import json
import pyarrow.parquet as pq

warnings.filterwarnings('ignore')


def load_model(model_path):
    """Wczytaj model"""
    print(f"📦 Wczytywanie modelu: {model_path}")
    model_package = joblib.load(model_path)
    
    print(f"   Model: {model_package['name']}")
    print(f"   Iteration: {model_package.get('iteration', 'baseline')}")
    print(f"   Optimal threshold: {model_package['optimal_threshold']:.4f}")
    
    return model_package


def process_single_file(args_tuple):
    """
    Przetwórz pojedynczy plik parquet
    
    Args:
        args_tuple: (file_path, model_path, threshold, batch_size, output_dir, file_idx)
    
    Returns:
        dict: Statystyki przetwarzania
    """
    file_path, model_path, threshold, batch_size, output_dir, file_idx = args_tuple
    
    try:
        # Nazwa pliku wyjściowego
        output_file = output_dir / f"predictions_{file_path.stem}.parquet"
        
        # Skip jeśli już przetworzony
        if output_file.exists():
            return {
                'file': file_path.name,
                'status': 'skipped',
                'reason': 'already_processed'
            }
        
        # Otwórz parquet file (bez wczytywania do RAM!)
        parquet_file = pq.ParquetFile(file_path)
        total_samples = parquet_file.metadata.num_rows
        
        print(f"   [Worker {file_idx}] Processing {file_path.name} ({total_samples:,} samples)...")
        print(f"   [Worker {file_idx}] Using STREAMING mode (memory efficient)")
        
        # Wczytaj model (każdy worker ładuje sam - unika pickle issues)
        model_package = joblib.load(model_path)
        model = model_package['model']
        scaler = model_package['scaler']
        
        print(f"   [Worker {file_idx}] Model loaded, streaming batches...")
        
        # Określ kolumny do wczytania (zawsze próbujemy uid, url, l14_img, score)
        columns_to_read = ['uid', 'url', 'l14_img', 'score']
        
        # Przetwarzaj w STREAMING batches
        predictions = []
        batch_idx = 0
        n_batches = (total_samples + batch_size - 1) // batch_size
        
        # Iteruj przez batche BEZ wczytywania całego pliku!
        for batch in parquet_file.iter_batches(batch_size=batch_size, columns=columns_to_read):
            # Progress co 5 batchy
            if batch_idx % 5 == 0:
                progress_pct = (batch_idx / n_batches) * 100
                print(f"   [Worker {file_idx}] Batch {batch_idx+1}/{n_batches} ({progress_pct:.0f}%)")
            
            # Konwertuj pyarrow batch do pandas (tylko ten batch!)
            df_batch = batch.to_pandas()
            
            # Przygotuj X
            X_batch = np.stack(df_batch['l14_img'].values)
            
            # Preprocessing (skalowanie)
            if scaler:
                X_batch = scaler.transform(X_batch)
            
            # Predict
            y_pred_proba = model.predict_proba(X_batch)[:, 1]
            
            # Filter: tylko pred_proba >= threshold
            mask = y_pred_proba >= threshold
            
            if mask.sum() > 0:
                # Zbierz tylko potrzebne kolumny
                batch_results = pd.DataFrame({
                    'uid': df_batch['uid'].values[mask],
                    'url': df_batch['url'].values[mask],
                    'pred_proba': y_pred_proba[mask]
                })
                
                # Dodaj score jeśli istnieje
                if 'score' in df_batch.columns:
                    batch_results['score'] = df_batch['score'].values[mask]
                
                predictions.append(batch_results)
            
            # Clear memory (KRYTYCZNE dla streaming!)
            del X_batch, y_pred_proba, mask, df_batch, batch
            gc.collect()
            
            batch_idx += 1
        
        # Zapisz wyniki
        if predictions:
            df_predictions = pd.concat(predictions, ignore_index=True)
            df_predictions.to_parquet(output_file, index=False)
            n_positive = len(df_predictions)
        else:
            # Plik pusty - brak pozytywnych predykcji
            n_positive = 0
        
        positive_rate = n_positive / total_samples if total_samples > 0 else 0
        print(f"   [Worker {file_idx}] ✅ Done: {n_positive:,}/{total_samples:,} ({positive_rate:.2%}) positive")
        
        # Statystyki
        return {
            'file': file_path.name,
            'status': 'success',
            'total': total_samples,
            'positive': n_positive,
            'positive_rate': positive_rate,
            'output': str(output_file)
        }
    
    except Exception as e:
        error_msg = f"{type(e).__name__}: {str(e)}"
        print(f"   [Worker {file_idx}] ❌ ERROR: {error_msg}")
        import traceback
        traceback.print_exc()  # Full traceback dla debugowania
        return {
            'file': file_path.name,
            'status': 'error',
            'reason': error_msg,
            'total': 0
        }


def find_parquet_files(input_dir):
    """Znajdź wszystkie pliki parquet w katalogu"""
    input_path = Path(input_dir)
    
    if not input_path.exists():
        raise FileNotFoundError(f"Katalog nie istnieje: {input_dir}")
    
    parquet_files = sorted(input_path.glob("*.parquet"))
    
    if not parquet_files:
        raise FileNotFoundError(f"Brak plików .parquet w: {input_dir}")
    
    return parquet_files


def merge_prediction_files(temp_dir, output_file, keep_temp=False):
    """
    Połącz wszystkie prediction files w jeden
    
    Args:
        temp_dir: katalog z plikami predictions_*.parquet
        output_file: ścieżka do finalnego pliku
        keep_temp: czy zachować pliki tymczasowe
    """
    print(f"\n🔗 Łączenie wyników...")
    
    temp_path = Path(temp_dir)
    prediction_files = sorted(temp_path.glob("predictions_*.parquet"))
    
    if not prediction_files:
        print("   ⚠️  Brak plików do połączenia (wszystkie pliki miały 0 pozytywnych?)")
        return None
    
    print(f"   Znaleziono {len(prediction_files)} plików do połączenia")
    
    # Wczytaj i połącz
    dfs = []
    for pred_file in tqdm(prediction_files, desc="   Łączenie"):
        try:
            df = pd.read_parquet(pred_file)
            if len(df) > 0:
                dfs.append(df)
        except Exception as e:
            print(f"   ⚠️  Błąd wczytywania {pred_file.name}: {e}")
    
    if not dfs:
        print("   ⚠️  Wszystkie pliki były puste")
        return None
    
    # Połącz
    df_final = pd.concat(dfs, ignore_index=True)
    
    # Usuń duplikaty (na wszelki wypadek)
    initial_len = len(df_final)
    df_final = df_final.drop_duplicates(subset='uid', keep='first')
    duplicates_removed = initial_len - len(df_final)
    
    if duplicates_removed > 0:
        print(f"   ⚠️  Usunięto {duplicates_removed} duplikatów")
    
    # Sortuj po pred_proba (malejąco)
    df_final = df_final.sort_values('pred_proba', ascending=False).reset_index(drop=True)
    
    # Zapisz
    df_final.to_parquet(output_file, index=False)
    
    print(f"   ✅ Zapisano: {output_file}")
    print(f"   Total predictions: {len(df_final):,}")
    
    # Usuń pliki tymczasowe jeśli requested
    if not keep_temp:
        print(f"\n🗑️  Usuwanie plików tymczasowych...")
        for pred_file in prediction_files:
            pred_file.unlink()
        print(f"   ✅ Usunięto {len(prediction_files)} plików")
    
    return df_final


def save_statistics(stats_list, output_file, elapsed_time):
    """Zapisz statystyki przetwarzania"""
    stats_file = output_file.parent / f"{output_file.stem}_stats.json"
    
    # Agreguj statystyki
    total_files = len(stats_list)
    successful = sum(1 for s in stats_list if s['status'] == 'success')
    errors = sum(1 for s in stats_list if s['status'] == 'error')
    skipped = sum(1 for s in stats_list if s['status'] == 'skipped')
    
    total_samples = sum(s.get('total', 0) for s in stats_list if 'total' in s)
    total_positive = sum(s.get('positive', 0) for s in stats_list if 'positive' in s)
    
    overall_positive_rate = total_positive / total_samples if total_samples > 0 else 0
    
    statistics = {
        'timestamp': datetime.now().isoformat(),
        'elapsed_time_seconds': elapsed_time,
        'files': {
            'total': total_files,
            'successful': successful,
            'errors': errors,
            'skipped': skipped
        },
        'samples': {
            'total': total_samples,
            'positive': total_positive,
            'positive_rate': overall_positive_rate
        },
        'throughput': {
            'samples_per_second': total_samples / elapsed_time if elapsed_time > 0 else 0,
            'files_per_minute': (total_files / elapsed_time) * 60 if elapsed_time > 0 else 0
        },
        'per_file_stats': stats_list
    }
    
    with open(stats_file, 'w') as f:
        json.dump(statistics, f, indent=2)
    
    print(f"\n📊 Statystyki zapisane: {stats_file}")
    
    return statistics


def print_summary(stats):
    """Wyświetl podsumowanie"""
    print("\n" + "="*80)
    print("📊 PODSUMOWANIE PRZETWARZANIA")
    print("="*80)
    
    print(f"\n📁 Pliki:")
    print(f"   Total:      {stats['files']['total']}")
    print(f"   Successful: {stats['files']['successful']} ✅")
    print(f"   Errors:     {stats['files']['errors']} {'❌' if stats['files']['errors'] > 0 else ''}")
    print(f"   Skipped:    {stats['files']['skipped']}")
    
    print(f"\n📊 Próbki:")
    print(f"   Total processed: {stats['samples']['total']:,}")
    print(f"   Positive (>= threshold): {stats['samples']['positive']:,}")
    print(f"   Positive rate: {stats['samples']['positive_rate']:.2%}")
    
    print(f"\n⏱️  Performance:")
    print(f"   Elapsed time: {stats['elapsed_time_seconds']:.1f}s ({stats['elapsed_time_seconds']/60:.1f} min)")
    print(f"   Throughput: {stats['throughput']['samples_per_second']:.0f} samples/sec")
    print(f"   Files/min: {stats['throughput']['files_per_minute']:.1f}")
    
    print("\n" + "="*80)


def main():
    parser = argparse.ArgumentParser(description='Apply model large scale (50M+ samples)')
    
    parser.add_argument('--input-dir', type=str, required=True,
                       help='Katalog z plikami parquet do przetworzenia')
    parser.add_argument('--output-file', type=str, required=True,
                       help='Ścieżka do pliku wyjściowego (finalne predykcje)')
    parser.add_argument('--model', type=str, default='models/model_latest.pkl',
                       help='Ścieżka do modelu (default: models/model_latest.pkl)')
    parser.add_argument('--threshold', type=float, default=0.5,
                       help='Threshold pred_proba (zapisz tylko >= threshold) (default: 0.5)')
    parser.add_argument('--batch-size', type=int, default=10000,
                       help='Batch size dla predykcji (default: 10000)')
    parser.add_argument('--workers', type=int, default=4,
                       help='Liczba workerów (parallel files) (default: 4)')
    parser.add_argument('--temp-dir', type=str, default=None,
                       help='Katalog tymczasowy (default: obok output-file)')
    parser.add_argument('--keep-temp', action='store_true',
                       help='Zachowaj pliki tymczasowe')
    parser.add_argument('--resume', action='store_true',
                       help='Wznów przetwarzanie (pomija już przetworzone pliki)')
    
    args = parser.parse_args()
    
    print("\n" + "="*80)
    print("🚀 LARGE SCALE MODEL APPLICATION")
    print("="*80)
    
    # Przygotuj ścieżki
    input_dir = Path(args.input_dir)
    output_file = Path(args.output_file)
    model_path = Path(args.model)
    
    # Temp dir
    if args.temp_dir:
        temp_dir = Path(args.temp_dir)
    else:
        temp_dir = output_file.parent / f"{output_file.stem}_temp"
    
    temp_dir.mkdir(parents=True, exist_ok=True)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"\n📂 Konfiguracja:")
    print(f"   Input dir:    {input_dir}")
    print(f"   Output file:  {output_file}")
    print(f"   Model:        {model_path}")
    print(f"   Threshold:    {args.threshold}")
    print(f"   Batch size:   {args.batch_size:,}")
    print(f"   Workers:      {args.workers}")
    print(f"   Temp dir:     {temp_dir}")
    print(f"   Resume:       {args.resume}")
    
    # Znajdź pliki
    print(f"\n🔍 Szukanie plików parquet...")
    parquet_files = find_parquet_files(input_dir)
    print(f"   Znaleziono: {len(parquet_files)} plików")
    
    # Oszacuj rozmiar
    total_size_mb = sum(f.stat().st_size for f in parquet_files) / (1024 * 1024)
    print(f"   Total size: {total_size_mb:.1f} MB ({total_size_mb/1024:.1f} GB)")
    
    # Wczytaj model info (tylko do wyświetlenia)
    model_package = load_model(model_path)
    
    # Przygotuj argumenty dla workerów (przekaż model_path, nie model_package!)
    # Każdy worker załaduje model sam (unikamy problemu z pickle/serialization)
    tasks = [
        (file_path, model_path, args.threshold, args.batch_size, temp_dir, idx)
        for idx, file_path in enumerate(parquet_files)
    ]
    
    # Start przetwarzania
    print(f"\n⚙️  Przetwarzanie ({args.workers} workers)...")
    print(f"   ℹ️  Każdy worker załaduje model niezależnie")
    start_time = datetime.now()
    
    stats_list = []
    
    # ThreadPoolExecutor zamiast ProcessPoolExecutor (lepsze dla I/O + shared memory)
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        # Submit wszystkie tasks
        futures = {executor.submit(process_single_file, task): task for task in tasks}
        
        # Progress bar
        with tqdm(total=len(parquet_files), desc="Files processed") as pbar:
            for future in as_completed(futures):
                result = future.result()
                stats_list.append(result)
                
                # Update progress
                status_emoji = {
                    'success': '✅',
                    'error': '❌',
                    'skipped': '⏭️'
                }.get(result['status'], '❓')
                
                pbar.set_postfix_str(f"{result['file']} {status_emoji}")
                pbar.update(1)
                
                # Print error reason jeśli error
                if result['status'] == 'error' and 'reason' in result:
                    print(f"\n❌ {result['file']}: {result['reason']}")
    
    elapsed_time = (datetime.now() - start_time).total_seconds()
    
    # Połącz wyniki
    print(f"\n🔗 Łączenie wyników z {len(parquet_files)} plików...")
    df_final = merge_prediction_files(temp_dir, output_file, keep_temp=args.keep_temp)
    
    # Zapisz statystyki
    statistics = save_statistics(stats_list, output_file, elapsed_time)
    
    # Podsumowanie
    print_summary(statistics)
    
    if df_final is not None:
        print(f"\n✅ GOTOWE!")
        print(f"   Wyniki: {output_file}")
        print(f"   Predictions: {len(df_final):,}")
        print(f"   Positive rate: {statistics['samples']['positive_rate']:.2%}")
        print(f"   Elapsed: {elapsed_time:.1f}s ({elapsed_time/60:.1f} min)")
    else:
        print(f"\n⚠️  Brak pozytywnych predykcji (wszystkie < {args.threshold})")
    
    print("\n" + "="*80)


if __name__ == "__main__":
    main()
