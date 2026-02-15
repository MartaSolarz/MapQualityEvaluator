"""
W tym skrypcie zapisujemy wszystkie dane z katalogu `above_value_minus1` do jednego pliku Parquet, ale bez kolumny `l14_img`, która zawiera embeddingi. Skrypt iteruje po wszystkich plikach Parquet w katalogu, usuwa kolumnę z embeddingiem i zapisuje pozostałe dane do nowego pliku `all_no_embeddings.parquet`. Dzięki temu możemy zaoszczędzić miejsce na dysku, jeśli embeddingi nie są potrzebne do dalszej analizy.
"""
import time
import pyarrow as pa
import pyarrow.parquet as pq
import os

directory = '/Volumes/PHD/phd/data/selected/above_value_minus1/'
OUTPUT_PATH = '/Volumes/PHD/phd/data/selected/above_value_minus1/all_no_embeddings.parquet'

startTime = time.time()

print("🔍 Szukanie plików .parquet w katalogu:", directory)
all_files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.parquet')]
print(f"✅ Znaleziono {len(all_files)} plików .parquet.")

writer = None
schema = None

for idx, file_path in enumerate(all_files, 1):
    print(f"\n📂 Przetwarzanie pliku {idx}/{len(all_files)}: {file_path}")

    parquet_file = pq.ParquetFile(file_path)

    for batch_idx, batch in enumerate(parquet_file.iter_batches(batch_size=50_000)):
        table = pa.Table.from_batches([batch])

        # 🔥 Usuń kolumnę z embeddingiem
        if 'l14_img' in table.schema.names:
            table = table.drop(['l14_img'])

        if schema is None:
            schema = table.schema
            os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
            writer = pq.ParquetWriter(OUTPUT_PATH, schema, compression='gzip')

        writer.write_table(table)

        if batch_idx % 10 == 0:
            print(f"   ↳ zapisano batch {batch_idx}")

if writer:
    writer.close()

print(f"\n✅ Gotowe! Czas całkowity: {time.time() - startTime:.2f} s")
