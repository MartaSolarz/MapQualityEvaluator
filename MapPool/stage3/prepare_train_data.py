"""
Przygotowanie danych treningowych dla Active Learning (Stage 3) - Nowe podejście

Dane wejściowe:
- baseline.parquet: 4k annotowanych próbek (YES/NO)
- sample_200k_with_embeddings.parquet: embeddingi CLIP L14 (768-dim)
- sample_200k.parquet: metadane (score, pos_keys, neg_keys, score_bin)

Dane wyjściowe:
- train.parquet: ~3.2k próbek treningowych (80% z annotacji)
- test.parquet: ~800 próbek testowych (20% z annotacji) - STAŁY między iteracjami
- pool.parquet: ~196k próbek bez adnotacji (do active learning)

UWAGA: Test set jest tworzony raz i pozostaje STAŁY przez wszystkie iteracje AL.
Użyj RANDOM_SEED=42 dla powtarzalności.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
from datetime import datetime
from sklearn.model_selection import train_test_split

# Stały seed dla powtarzalności
RANDOM_SEED = 42

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data_all" / "above_value_minus1"
BASELINE_DIR = Path(__file__).parent / "data" / "baseline"
OUTPUT_DIR = Path(__file__).parent / "data"

BASELINE_FILE = BASELINE_DIR / "baseline.parquet"
EMBEDDINGS_FILE = DATA_DIR / "sample_200k_with_embeddings.parquet"
METADATA_FILE = DATA_DIR / "sample_200k.parquet"

TRAIN_OUTPUT = OUTPUT_DIR / "train.parquet"
TEST_OUTPUT = OUTPUT_DIR / "test.parquet"
POOL_OUTPUT = OUTPUT_DIR / "pool.parquet"
STATS_OUTPUT = OUTPUT_DIR / "data_preparation_stats.json"


def load_annotations():
    """Wczytaj adnotacje z baseline.parquet"""
    print("📂 Wczytywanie adnotacji z baseline...")
    df = pd.read_parquet(BASELINE_FILE)
    
    print(f"   Liczba adnotacji: {len(df)}")
    print(f"   Kolumny: {list(df.columns)}")
    print(f"   Rozkład labelek:")
    print(df['label'].value_counts().to_string())
    
    # Mapuj na wartości binarne
    df['label'] = df['label'].map({'YES': 1, 'NO': 0})
    
    print(f"\n   YES (1): {(df['label'] == 1).sum()}")
    print(f"   NO (0): {(df['label'] == 0).sum()}")
    print(f"   Class balance: {(df['label'] == 1).sum() / len(df):.2%}")
    
    return df


def load_embeddings():
    """Wczytaj embeddingi CLIP L14"""
    print("\n🔢 Wczytywanie embeddingów...")
    df = pd.read_parquet(EMBEDDINGS_FILE)
    
    print(f"   Liczba próbek z embeddingami: {len(df)}")
    print(f"   Wymiar embeddingu: {df.iloc[0]['l14_img'].shape}")
    
    return df


def load_metadata():
    """Wczytaj metadane (score, keywords)"""
    print("\n📊 Wczytywanie metadanych...")
    df = pd.read_parquet(METADATA_FILE)
    
    print(f"   Liczba próbek z metadanymi: {len(df)}")
    print(f"   Kolumny: {list(df.columns)}")
    
    return df


def merge_data(annotations, embeddings, metadata):
    """Połącz wszystkie dane po uid"""
    print("\n🔗 Łączenie danych...")
    
    # Merge adnotacji z embeddingami
    annotated_data = annotations.merge(embeddings, on='uid', how='inner')
    print(f"   Po merge z embeddingami: {len(annotated_data)} próbek")
    
    # Merge z metadanymi - tylko score
    metadata_subset = metadata[['uid', 'score']].copy()
    annotated_data = annotated_data.merge(metadata_subset, on='uid', how='left')
    print(f"   Po merge z metadanymi: {len(annotated_data)} próbek")
    
    # Sprawdź czy są braki
    missing_embeddings = len(annotations) - len(annotated_data)
    if missing_embeddings > 0:
        print(f"   ⚠️  Brak embeddingów dla {missing_embeddings} próbek")
    
    return annotated_data


def split_train_test(annotated_data, test_size=0.2):
    """
    Podziel annotowane dane na train i test (stratyfikowany podział)
    
    UWAGA: Test set będzie STAŁY przez wszystkie iteracje Active Learning!
    """
    print(f"\n✂️  Podział train/test (test_size={test_size}, stratified)...")
    
    # Stratyfikowany podział - zachowaj proporcje klas
    train_data, test_data = train_test_split(
        annotated_data,
        test_size=test_size,
        random_state=RANDOM_SEED,
        stratify=annotated_data['label']
    )
    
    print(f"   Train: {len(train_data)} próbek")
    print(f"      YES (1): {(train_data['label'] == 1).sum()} ({(train_data['label'] == 1).sum()/len(train_data):.2%})")
    print(f"      NO (0): {(train_data['label'] == 0).sum()} ({(train_data['label'] == 0).sum()/len(train_data):.2%})")
    
    print(f"   Test: {len(test_data)} próbek")
    print(f"      YES (1): {(test_data['label'] == 1).sum()} ({(test_data['label'] == 1).sum()/len(test_data):.2%})")
    print(f"      NO (0): {(test_data['label'] == 0).sum()} ({(test_data['label'] == 0).sum()/len(test_data):.2%})")
    
    print(f"\n   ⚠️  UWAGA: Test set jest STAŁY! Zapisz UIDs dla weryfikacji.")
    
    return train_data, test_data


def create_pool_data(embeddings, metadata, annotated_uids):
    """Utwórz pool danych (wszystkie próbki bez adnotacji)"""
    print("\n🎱 Tworzenie pool danych...")
    
    # Merge embeddings z metadata
    pool_data = embeddings.merge(metadata, on='uid', how='inner')
    
    # Usuń próbki, które są już annotowane (train + test)
    pool_data = pool_data[~pool_data['uid'].isin(annotated_uids)].copy()
    
    print(f"   Próbek w pool: {len(pool_data)}")
    print(f"   Statystyki score:")
    if 'score' in pool_data.columns:
        print(f"      Mean: {pool_data['score'].mean():.3f}")
        print(f"      Std: {pool_data['score'].std():.3f}")
        print(f"      Min: {pool_data['score'].min():.3f}, Max: {pool_data['score'].max():.3f}")
    
    return pool_data


def save_data(train_data, test_data, pool_data):
    """Zapisz dane do plików parquet"""
    print("\n💾 Zapisywanie danych...")
    
    # Uporządkuj kolumny dla train i test - uid, label, embeddingi, url, score
    labeled_cols = ['uid', 'label', 'l14_img', 'url', 'score']
    labeled_cols = [c for c in labeled_cols if c in train_data.columns]
    train_data = train_data[labeled_cols].copy()
    test_data = test_data[labeled_cols].copy()
    
    # Uporządkuj kolumny dla pool - uid, embeddingi, url, score (bez label)
    pool_cols = ['uid', 'l14_img', 'url', 'score']
    pool_cols = [c for c in pool_cols if c in pool_data.columns]
    pool_data = pool_data[pool_cols].copy()
    
    # Zapisz
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    train_data.to_parquet(TRAIN_OUTPUT, index=False)
    test_data.to_parquet(TEST_OUTPUT, index=False)
    pool_data.to_parquet(POOL_OUTPUT, index=False)
    
    print(f"   ✅ Train data: {TRAIN_OUTPUT}")
    print(f"      Rozmiar: {len(train_data)} próbek")
    print(f"   ✅ Test data: {TEST_OUTPUT}")
    print(f"      Rozmiar: {len(test_data)} próbek")
    print(f"   ✅ Pool data: {POOL_OUTPUT}")
    print(f"      Rozmiar: {len(pool_data)} próbek")
    
    # Zapisz również UIDs test setu do osobnego pliku dla weryfikacji
    test_uids_file = OUTPUT_DIR / "test_uids.txt"
    with open(test_uids_file, 'w') as f:
        for uid in test_data['uid']:
            f.write(f"{uid}\n")
    print(f"   ✅ Test UIDs: {test_uids_file}")
    
    return train_data, test_data, pool_data


def save_statistics(train_data, test_data, pool_data):
    """Zapisz statystyki przygotowania danych"""
    stats = {
        'timestamp': datetime.now().isoformat(),
        'random_seed': RANDOM_SEED,
        'train': {
            'size': len(train_data),
            'label_distribution': {
                'YES (1)': int((train_data['label'] == 1).sum()),
                'NO (0)': int((train_data['label'] == 0).sum())
            },
            'class_balance': float((train_data['label'] == 1).sum() / len(train_data)),
            'embedding_dim': int(train_data.iloc[0]['l14_img'].shape[0]),
            'columns': list(train_data.columns)
        },
        'test': {
            'size': len(test_data),
            'label_distribution': {
                'YES (1)': int((test_data['label'] == 1).sum()),
                'NO (0)': int((test_data['label'] == 0).sum())
            },
            'class_balance': float((test_data['label'] == 1).sum() / len(test_data)),
            'embedding_dim': int(test_data.iloc[0]['l14_img'].shape[0]),
            'columns': list(test_data.columns)
        },
        'pool': {
            'size': len(pool_data),
            'embedding_dim': int(pool_data.iloc[0]['l14_img'].shape[0]),
            'columns': list(pool_data.columns)
        }
    }
    
    # Dodaj rozkład score jeśli istnieje
    for dataset_name, dataset in [('train', train_data), ('test', test_data), ('pool', pool_data)]:
        if 'score' in dataset.columns:
            stats[dataset_name]['score_stats'] = {
                'mean': float(dataset['score'].mean()),
                'std': float(dataset['score'].std()),
                'min': float(dataset['score'].min()),
                'max': float(dataset['score'].max())
            }
    
    # Zapisz
    with open(STATS_OUTPUT, 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"\n📈 Statystyki zapisane: {STATS_OUTPUT}")
    
    return stats


def display_summary(stats):
    """Wyświetl podsumowanie"""
    print("\n" + "="*60)
    print("📋 PODSUMOWANIE PRZYGOTOWANIA DANYCH")
    print("="*60)
    
    print(f"\n🎯 DANE TRENINGOWE:")
    print(f"   Liczba próbek: {stats['train']['size']}")
    print(f"   YES (1): {stats['train']['label_distribution']['YES (1)']} ({stats['train']['class_balance']:.2%})")
    print(f"   NO (0): {stats['train']['label_distribution']['NO (0)']} ({1-stats['train']['class_balance']:.2%})")
    print(f"   Wymiar embeddingu: {stats['train']['embedding_dim']}")
    
    print(f"\n🧪 DANE TESTOWE (STAŁY ZBIÓR):")
    print(f"   Liczba próbek: {stats['test']['size']}")
    print(f"   YES (1): {stats['test']['label_distribution']['YES (1)']} ({stats['test']['class_balance']:.2%})")
    print(f"   NO (0): {stats['test']['label_distribution']['NO (0)']} ({1-stats['test']['class_balance']:.2%})")
    print(f"   Wymiar embeddingu: {stats['test']['embedding_dim']}")
    
    print(f"\n🎱 POOL DANYCH (do Active Learning):")
    print(f"   Liczba próbek: {stats['pool']['size']}")
    print(f"   Wymiar embeddingu: {stats['pool']['embedding_dim']}")
    
    print(f"\n📌 UWAGA:")
    print(f"   - Test set jest STAŁY (random_seed={stats['random_seed']})")
    print(f"   - Używaj tego samego test setu we wszystkich iteracjach AL")
    print(f"   - UIDs test setu zapisane w: test_uids.txt")
    
    print(f"\n✅ Gotowe do treningu modelu baseline i Active Learning!")
    print("="*60)


def main():
    """Główna funkcja przygotowania danych"""
    print("\n" + "="*60)
    print("🚀 PRZYGOTOWANIE DANYCH DO ACTIVE LEARNING (NOWE PODEJŚCIE)")
    print("="*60)
    
    # 1. Wczytaj dane
    annotations = load_annotations()
    embeddings = load_embeddings()
    metadata = load_metadata()
    
    # 2. Połącz annotacje z embeddingami i metadanymi
    annotated_data = merge_data(annotations, embeddings, metadata)
    
    # 3. Podziel na train i test (STAŁY test set!)
    train_data, test_data = split_train_test(annotated_data, test_size=0.2)
    
    # 4. Przygotuj pool data (usuń wszystkie annotowane: train + test)
    all_annotated_uids = pd.concat([train_data['uid'], test_data['uid']])
    pool_data = create_pool_data(embeddings, metadata, all_annotated_uids)
    
    # 5. Zapisz dane
    train_data, test_data, pool_data = save_data(train_data, test_data, pool_data)
    
    # 6. Zapisz statystyki
    stats = save_statistics(train_data, test_data, pool_data)
    
    # 7. Wyświetl podsumowanie
    display_summary(stats)


if __name__ == "__main__":
    main()
