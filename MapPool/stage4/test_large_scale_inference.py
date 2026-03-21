#!/usr/bin/env python3
"""
Test large scale inference na małym zbiorze (pool 196k)

Użycie:
    python3 test_large_scale_inference.py
"""

import pandas as pd
import subprocess
from pathlib import Path
import json


def run_test():
    """Test na pool (196k próbek)"""
    
    print("="*80)
    print("🧪 TEST LARGE SCALE INFERENCE")
    print("="*80)
    
    # Ścieżki
    base_dir = Path(__file__).parent.parent
    pool_file = base_dir / "stage3" / "data" / "pool.parquet"
    test_output = base_dir / "results" / "test_predictions.parquet"
    
    # Sprawdź czy pool istnieje
    if not pool_file.exists():
        print(f"\n❌ Błąd: Brak pliku pool: {pool_file}")
        print("   Uruchom najpierw prepare_train_data.py")
        return False
    
    # Sprawdź rozmiar pool
    df_pool = pd.read_parquet(pool_file)
    print(f"\n📊 Pool data:")
    print(f"   Próbek: {len(df_pool):,}")
    print(f"   Kolumny: {list(df_pool.columns)}")
    
    if 'l14_img' not in df_pool.columns:
        print(f"\n❌ Błąd: Brak kolumny l14_img (embeddingi)")
        return False
    
    # Stwórz temp directory z jednym plikiem
    temp_input_dir = base_dir / "test_input_temp"
    temp_input_dir.mkdir(exist_ok=True)
    
    # Kopiuj pool do temp (jako test input)
    temp_pool_file = temp_input_dir / "pool.parquet"
    
    print(f"\n📁 Przygotowanie test input...")
    print(f"   Kopiuję {pool_file} → {temp_pool_file}")
    
    # Kopiuj używając symlink (szybsze) lub copy
    if temp_pool_file.exists():
        temp_pool_file.unlink()
    
    import shutil
    shutil.copy(pool_file, temp_pool_file)
    
    # Uruchom inference
    print(f"\n🚀 Uruchamianie inference...")
    
    cmd = [
        "python3",
        str(base_dir / "stage4" / "apply_model_large_scale.py"),
        "--input-dir", str(temp_input_dir),
        "--model", str(base_dir / "stage4" / "models" / "model_final.pkl"),
        "--output-file", str(test_output),
        "--threshold", "0.5",
        "--workers", "2",
        "--batch-size", "10000"
    ]
    
    print(f"   Command: {' '.join(cmd)}")
    
    result = subprocess.run(cmd, capture_output=False, text=True)
    
    if result.returncode != 0:
        print(f"\n❌ Inference failed with code {result.returncode}")
        return False
    
    # Sprawdź wyniki
    print(f"\n📊 Analiza wyników...")
    
    if not test_output.exists():
        print(f"❌ Brak pliku wyjściowego: {test_output}")
        return False
    
    df_pred = pd.read_parquet(test_output)
    
    print(f"\n✅ Test SUKCES!")
    print(f"\n📈 Wyniki:")
    print(f"   Input próbek:    {len(df_pool):,}")
    print(f"   Output próbek:   {len(df_pred):,}")
    print(f"   Positive rate:   {len(df_pred)/len(df_pool):.2%}")
    print(f"   Pred_proba range: {df_pred['pred_proba'].min():.3f} - {df_pred['pred_proba'].max():.3f}")
    print(f"   Mean pred_proba:  {df_pred['pred_proba'].mean():.3f}")
    
    # Sprawdź kolumny
    print(f"\n   Kolumny output: {list(df_pred.columns)}")
    expected_cols = ['uid', 'url', 'pred_proba']
    missing = set(expected_cols) - set(df_pred.columns)
    if missing:
        print(f"   ⚠️  Brakujące kolumny: {missing}")
    else:
        print(f"   ✅ Wszystkie wymagane kolumny obecne")
    
    # Sprawdź czy są embeddngi (nie powinno być!)
    if 'l14_img' in df_pred.columns:
        print(f"   ⚠️  WARNING: Output zawiera embeddingi (l14_img) - plik będzie duży!")
    else:
        print(f"   ✅ Output NIE zawiera embeddingów (dobry rozmiar)")
    
    # Rozmiar pliku
    file_size_mb = test_output.stat().st_size / (1024 * 1024)
    print(f"\n   File size: {file_size_mb:.2f} MB")
    
    # Estimate dla 50M
    estimated_50M_mb = (file_size_mb / len(df_pred)) * 50_000_000 * (len(df_pred)/len(df_pool))
    print(f"   Estimated size for 50M: {estimated_50M_mb:.0f} MB ({estimated_50M_mb/1024:.1f} GB)")
    
    # Sprawdź stats file
    stats_file = test_output.parent / f"{test_output.stem}_stats.json"
    if stats_file.exists():
        with open(stats_file) as f:
            stats = json.load(f)
        
        print(f"\n📊 Performance stats:")
        print(f"   Elapsed time: {stats['elapsed_time_seconds']:.1f}s")
        print(f"   Throughput: {stats['throughput']['samples_per_second']:.0f} samples/s")
        
        # Estimate dla 50M
        estimated_time_50M = 50_000_000 / stats['throughput']['samples_per_second']
        print(f"\n   Estimated time for 50M: {estimated_time_50M/3600:.1f}h ({estimated_time_50M/60:.0f} min)")
    
    # Cleanup
    print(f"\n🗑️  Cleaning up temp files...")
    temp_pool_file.unlink()
    temp_input_dir.rmdir()
    
    print(f"\n✅ Test zakończony pomyślnie!")
    print(f"\n📁 Pliki wyjściowe:")
    print(f"   Predictions: {test_output}")
    print(f"   Stats: {stats_file}")
    
    print(f"\n🚀 Gotowy do uruchomienia na 50M!")
    
    return True


if __name__ == "__main__":
    success = run_test()
    
    if success:
        print("\n" + "="*80)
        print("✅ TEST PASSED - możesz uruchomić na 50M")
        print("="*80)
        
        print("\n📝 Przykładowa komenda dla 50M:")
        print("""
python3 apply_model_large_scale.py \\
    --input-dir /path/to/volume/50M_parquet_files \\
    --output-file results/predictions_50M_positive.parquet \\
    --threshold 0.5 \\
    --workers 8 \\
    --batch-size 10000
""")
    else:
        print("\n" + "="*80)
        print("❌ TEST FAILED - napraw błędy przed uruchomieniem na 50M")
        print("="*80)
