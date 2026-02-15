"""
Skrypt służy do łączenia wielu plików Parquet z katalogu w jeden duży plik Parquet. Skrypt iteruje po wszystkich plikach `part*.parquet` w katalogu, łączy je w jeden DataFrame, a następnie zapisuje do nowego pliku `all.parquet`. Skrypt jest zoptymalizowany pod kątem pamięci, przetwarzając dane w mniejszych "chunkach" i zapisując je etapami. W przypadku dużych zbiorów danych, ten podejście pozwala uniknąć problemów z pamięcią RAM.

Użycie: podmieniń PART_NAME
"""
import time
import pyarrow as pa
import pyarrow.parquet as pq
import pandas as pd
import os
import numpy as np

PART_NAME = 'part_7'
directory = '/Volumes/PHD/phd/data/selected/above_value_minus1/'
OUTPUT_PATH = '/Volumes/PHD/phd/data/selected/above_value_minus1/all.parquet'
CHUNK_SIZE = 500

startTime = time.time()

print("🔍 Szukanie plików .parquet w katalogu:", directory)
all_files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.parquet')]
print(f"✅ Znaleziono {len(all_files)} plików .parquet.")

# Podziel na chunki
chunks = [all_files[i:i + CHUNK_SIZE] for i in range(0, len(all_files), CHUNK_SIZE)]

writer = None
schema = None

for chunk_idx, chunk_files in enumerate(chunks):
    print(f"🔄 Przetwarzanie chunk {chunk_idx + 1}/{len(chunks)} ({len(chunk_files)} plików)...")

    df_list = [pd.read_parquet(file) for file in chunk_files]
    combined_df = pd.concat(df_list, ignore_index=True)

    # Filtrowanie
    # df_selected = combined_df[combined_df['score'] >= -1]
    #
    # if len(df_selected) == 0:
    #     print(f"⚠️ Chunk {chunk_idx + 1} - brak danych po filtrowaniu")
    #     continue

    # Przygotuj dane
    emb_dim = len(combined_df['l14_img'].iloc[0])

    if schema is None:
        schema = pa.schema([
            ('uid', pa.string()),
            ('url', pa.string()),
            ('score', pa.float64()),
            ('pos_keys', pa.string()),
            ('neg_keys', pa.string()),
            ('l14_img', pa.list_(pa.float32(), emb_dim))
        ])

    # Konwersja embeddingów
    embeddings_np = np.stack(combined_df['l14_img'].values).astype(np.float32)
    embeddings_array = pa.FixedSizeListArray.from_arrays(
        embeddings_np.flatten(),
        list_size=emb_dim
    )

    table = pa.table({
        'uid': combined_df['uid'],
        'url': combined_df['url'],
        'score': combined_df['score'],
        'pos_keys': combined_df['pos_keys'],
        'neg_keys': combined_df['neg_keys'],
        'l14_img': embeddings_array
    }, schema=schema)

    # Zapisz (append mode)
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    if writer is None:
        writer = pq.ParquetWriter(OUTPUT_PATH, schema, compression='gzip')

    writer.write_table(table)

    print(f"✅ Chunk {chunk_idx + 1} zapisany ({len(combined_df)} wierszy)")

    # Zwolnij pamięć
    del df_list, combined_df, combined_df, table, embeddings_array, embeddings_np

if writer:
    writer.close()

print(f"✅ Gotowe! Czas całkowity: {time.time() - startTime:.2f} sekund.")