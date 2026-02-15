"""
Skrypt pobiera embeddingi dla 200k próbek z pliku `sample_200k.parquet` i zapisuje je do nowego pliku `sample_200k_with_embeddings.parquet`. Skrypt iteruje po wszystkich plikach `part*.parquet` w katalogu, filtrując tylko te rekordy, których `uid` znajduje się w zbiorze `sample_200k`. Znalezione rekordy są łączone i zapisywane do nowego pliku Parquet. Skrypt zatrzymuje się, gdy znajdzie wszystkie potrzebne `uid` lub przetworzy wszystkie pliki.
"""
import polars as pl
from pathlib import Path

sampled_df = pl.read_parquet("../data_all/above_value_minus1/sample_200k.parquet")
sampled_uids_set = set(sampled_df.select("uid").to_series().to_list())

data_dir = Path("/Volumes/PHD/phd/data/selected/above_value_minus1/")
parquet_files = sorted(data_dir.glob("part*.parquet"))
output_file = "../data_all/above_value_minus1/sample_200k_with_embeddings.parquet"

all_dfs = []
found_uids = set()

for file in parquet_files:
    if len(found_uids) >= len(sampled_uids_set):
        print(f"Found all {len(sampled_uids_set)} UIDs, stopping early")
        break

    print(f"Processing {file.name} ... (found {len(found_uids)}/{len(sampled_uids_set)})")

    df = (
        pl.scan_parquet(file)
        .select(["uid", "l14_img"])
        .filter(pl.col("uid").is_in(sampled_uids_set))
        .collect()
    )

    if len(df) > 0:
        all_dfs.append(df)
        found_uids.update(df["uid"].to_list())


if all_dfs:
    final_df = pl.concat(all_dfs)
    final_df.write_parquet(output_file)
    print(f"Saved {len(final_df)} records to {output_file}")
else:
    print("No matching records found")