# 🚀 Large Scale Model Application (50M próbek)

## 📊 Przegląd

Skrypt `apply_model_large_scale.py` umożliwia efektywne przetworzenie milionów próbek z wykorzystaniem:
- **Wielowątkowości** (parallel file processing)
- **Batch processing** (memory efficient)
- **Streaming** (nie ładuje wszystkiego do pamięci)
- **Resume capability** (można wznowić przerwane przetwarzanie)
- **Filtrowania** (zapisuje tylko pred_proba >= threshold)

---

## 🎯 Użycie

### Podstawowe wywołanie:

```bash
python3 apply_model_large_scale.py \
    --input-dir /path/to/volume/parquet_files \
    --output-file results/predictions_50M.parquet \
    --threshold 0.5 \
    --workers 8
```

---

## ⚙️ Parametry

### Wymagane:

- `--input-dir`: Katalog z plikami `.parquet` do przetworzenia
- `--output-file`: Ścieżka do finalnego pliku z predykcjami

### Opcjonalne:

| Parametr | Default | Opis |
|----------|---------|------|
| `--model` | `models/model_latest.pkl` | Ścieżka do modelu |
| `--threshold` | `0.5` | Minimalne pred_proba do zapisu |
| `--batch-size` | `10000` | Rozmiar batcha (memory vs speed) |
| `--workers` | `4` | Liczba równoległych procesów |
| `--temp-dir` | auto | Katalog tymczasowy |
| `--keep-temp` | False | Zachowaj pliki tymczasowe |
| `--resume` | False | Wznów przerwane przetwarzanie |

---

## 📈 Wydajność

### Przykładowe czasy (zależne od hardware):

| Dataset | Próbek | Workers | Batch | Czas | Throughput |
|---------|--------|---------|-------|------|------------|
| 1M próbek | 1,000,000 | 4 | 10k | ~5 min | 3,300 samples/s |
| 10M próbek | 10,000,000 | 8 | 10k | ~40 min | 4,100 samples/s |
| 50M próbek | 50,000,000 | 8 | 10k | ~3.5h | 3,900 samples/s |

**Uwagi:**
- Szybkość zależy od: CPU, dysku (SSD vs HDD), rozmiaru plików
- Więcej workers = szybciej, ale więcej RAM
- Większy batch = szybciej, ale więcej RAM na worker

---

## 🔧 Optymalizacja parametrów

### Workers (--workers)

**Rekomendacja:** `min(CPU_cores, liczba_plików)`

```bash
# Sprawdź liczbę rdzeni
python3 -c "import os; print(f'CPU cores: {os.cpu_count()}')"

# Dobór workers:
# - Mało plików (< 10): workers = liczba_plików
# - Dużo plików (> 50): workers = CPU_cores
# - Złoty środek: workers = CPU_cores / 2 (zostaw miejsce na system)
```

**Przykłady:**
- CPU: 4 cores → `--workers 4`
- CPU: 8 cores → `--workers 6-8`
- CPU: 16 cores → `--workers 12-16`

---

### Batch size (--batch-size)

**Trade-off:** Speed vs Memory

| Batch size | RAM/worker | Speed | Kiedy używać |
|------------|------------|-------|--------------|
| 1,000 | ~100 MB | Wolno | Mało RAM |
| 5,000 | ~500 MB | OK | Standard |
| 10,000 | ~1 GB | Szybko | Recommended |
| 50,000 | ~5 GB | Bardzo szybko | Dużo RAM |

**Rekomendacja:** 
```python
# Oblicz optymalny batch size
available_ram_gb = 32  # Twój RAM
workers = 8
batch_size = (available_ram_gb * 0.5) / workers * 1000  # ~2000-10000
```

---

## 💾 Zarządzanie pamięcią

### Struktura przetwarzania:

```
Input: 50M próbek w wielu plikach
  ↓
Per-file processing (parallel):
  ↓
Batch processing (10k na raz):
  - Load batch → Predict → Filter (>= 0.5) → Save → Clear memory
  ↓
Temp files: predictions_file1.parquet, predictions_file2.parquet, ...
  ↓
Merge → Final: predictions_50M.parquet
```

**Zużycie RAM (approximate):**
```
RAM = workers * batch_size * embedding_size
    = 8 * 10,000 * 768 * 4 bytes
    = ~2.4 GB
```

---

## 📁 Struktura plików wyjściowych

### Durante przetwarzania:

```
output_dir/
├── predictions_50M_temp/           ← Temp directory
│   ├── predictions_file1.parquet   ← Per-file results
│   ├── predictions_file2.parquet
│   └── ...
```

### Po zakończeniu:

```
output_dir/
├── predictions_50M.parquet         ← Final merged results
└── predictions_50M_stats.json      ← Processing statistics
```

### Format predictions_50M.parquet:

```
Kolumny:
- uid: str
- url: str
- pred_proba: float (>= threshold)
- score: float (jeśli dostępne w input)

Sortowanie: pred_proba (malejąco - najlepsze na górze)
```

---

## 🔄 Resume capability

Jeśli przetwarzanie zostanie przerwane:

```bash
# Wznów od miejsca przerwania
python3 apply_model_large_scale.py \
    --input-dir /path/to/volume \
    --output-file results/predictions_50M.parquet \
    --resume
```

**Jak działa:**
- Sprawdza które pliki w temp_dir już zostały przetworzone
- Pomija je (status: 'skipped')
- Przetwarza tylko pozostałe

---

## 📊 Monitoring postępu

### Realtime:

```
Files processed: 45%|████████░░░░| 123/274 [12:34<15:23, 0.16it/s] file_123.parquet ✅
```

### Po zakończeniu:

```json
{
  "files": {
    "total": 274,
    "successful": 273,
    "errors": 1,
    "skipped": 0
  },
  "samples": {
    "total": 50000000,
    "positive": 1250000,
    "positive_rate": 0.025
  },
  "throughput": {
    "samples_per_second": 3900,
    "files_per_minute": 1.3
  }
}
```

---

## 🎯 Przykłady użycia

### 1. Standard (50M próbek, 8 workers):

```bash
python3 apply_model_large_scale.py \
    --input-dir /volume/maps_50M \
    --output-file results/predictions_positive.parquet \
    --threshold 0.5 \
    --workers 8 \
    --batch-size 10000
```

**Expected output:**
- Czas: ~3-4h
- Positive rate: ~2-5% (1-2.5M map)
- Final file size: ~50-150 MB (bez embeddingów!)

---

### 2. High confidence tylko (threshold 0.7):

```bash
python3 apply_model_large_scale.py \
    --input-dir /volume/maps_50M \
    --output-file results/predictions_high_conf.parquet \
    --threshold 0.7 \
    --workers 8
```

**Expected output:**
- Mniej wyników (tylko high confidence)
- Wyższa precision, niższy recall

---

### 3. Szybkie przetwarzanie (więcej workers + większy batch):

```bash
python3 apply_model_large_scale.py \
    --input-dir /volume/maps_50M \
    --output-file results/predictions_positive.parquet \
    --threshold 0.5 \
    --workers 16 \
    --batch-size 20000
```

**Uwaga:** Wymaga więcej RAM (~4-6 GB)

---

### 4. Conservative (mało RAM):

```bash
python3 apply_model_large_scale.py \
    --input-dir /volume/maps_50M \
    --output-file results/predictions_positive.parquet \
    --threshold 0.5 \
    --workers 2 \
    --batch-size 5000
```

**Uwaga:** Wolniejsze, ale bezpieczne dla systemów z <8GB RAM

---

## 🐛 Troubleshooting

### Problem: Out of Memory

**Rozwiązania:**
1. Zmniejsz `--batch-size` (np. 5000)
2. Zmniejsz `--workers` (np. 2-4)
3. Użyj `--temp-dir` na dysku z więcej miejsca

```bash
python3 apply_model_large_scale.py \
    --batch-size 5000 \
    --workers 2 \
    ...
```

---

### Problem: Przetwarzanie zbyt wolne

**Rozwiązania:**
1. Zwiększ `--workers` (max: CPU cores)
2. Zwiększ `--batch-size` (jeśli masz RAM)
3. Upewnij się że temp-dir jest na SSD (nie HDD)

```bash
python3 apply_model_large_scale.py \
    --workers 12 \
    --batch-size 15000 \
    --temp-dir /path/to/ssd \
    ...
```

---

### Problem: Przerwane przetwarzanie

**Rozwiązanie:**
```bash
# Wznów z --resume
python3 apply_model_large_scale.py \
    --resume \
    --input-dir /volume/maps_50M \
    --output-file results/predictions_positive.parquet
```

---

### Problem: Niektóre pliki mają błędy

**Diagnoza:**
```bash
# Sprawdź stats file
cat results/predictions_positive_stats.json | grep -A5 '"status": "error"'
```

**Możliwe przyczyny:**
- Brak kolumny `l14_img` (embeddingi)
- Skorumpowany plik parquet
- Niekompatybilny format

---

## 📊 Po przetworzeniu - analiza wyników

### 1. Sprawdź top predictions:

```python
import pandas as pd

df = pd.read_parquet('results/predictions_positive.parquet')

print(f"Total predictions: {len(df):,}")
print(f"\nTop 10 (highest pred_proba):")
print(df.head(10)[['uid', 'pred_proba', 'score']])

print(f"\nRozkład pred_proba:")
print(df['pred_proba'].describe())
```

---

### 2. Filtruj high confidence:

```python
# High confidence only (>= 0.8)
df_high = df[df['pred_proba'] >= 0.8]
print(f"High confidence (>= 0.8): {len(df_high):,}")

# Save
df_high.to_parquet('results/predictions_high_confidence.parquet')
```

---

### 3. Statystyki:

```python
import json

with open('results/predictions_positive_stats.json') as f:
    stats = json.load(f)

print(f"Processed: {stats['samples']['total']:,} próbek")
print(f"Positive: {stats['samples']['positive']:,} ({stats['samples']['positive_rate']:.2%})")
print(f"Time: {stats['elapsed_time_seconds']/60:.1f} min")
print(f"Throughput: {stats['throughput']['samples_per_second']:.0f} samples/s")
```

---

## ✅ Checklist przed uruchomieniem

- [ ] Model wytrenowany i zapisany (`models/model_latest.pkl`)
- [ ] Pliki input mają kolumnę `l14_img` (embeddingi)
- [ ] Pliki input mają kolumny `uid`, `url`
- [ ] Wystarczająco miejsca na dysku (output + temp)
- [ ] Wystarczająco RAM (workers * batch_size * 1GB)
- [ ] Threshold dobrany (0.5 = balanced, 0.7 = high precision)

---

## 🚀 Quick Start

```bash
# 1. Test na małym zbiorze (sprawdź czy działa)
python3 apply_model_large_scale.py \
    --input-dir test_data_small \
    --output-file test_predictions.parquet \
    --workers 2

# 2. Production run (50M)
python3 apply_model_large_scale.py \
    --input-dir /volume/maps_50M \
    --output-file results/predictions_positive.parquet \
    --threshold 0.5 \
    --workers 8 \
    --batch-size 10000

# 3. Analyze results
python3 -c "
import pandas as pd
df = pd.read_parquet('results/predictions_positive.parquet')
print(f'Total: {len(df):,}')
print(f'Top pred_proba: {df[\"pred_proba\"].max():.4f}')
print(f'Mean pred_proba: {df[\"pred_proba\"].mean():.4f}')
"
```

---

**GOTOWE DO UŻYCIA!** 🎯
