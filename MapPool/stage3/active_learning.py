"""
Active Learning - wybór najbardziej niepewnych próbek do adnotacji

Strategie uncertainty sampling:
1. Entropy - maksymalna entropia rozkładu prawdopodobieństw
2. Margin - minimalna różnica między top-2 klasami
3. Least Confident - minimalne prawdopodobieństwo najlepszej klasy

Dla binary classification wszystkie trzy są równoważne, ale entropy jest najbardziej intuicyjna.
"""

import pandas as pd
import numpy as np
import json
import joblib
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Ścieżki
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
ITERATIONS_DIR = BASE_DIR / "iterations"

POOL_FILE = DATA_DIR / "pool.parquet"
TRAIN_FILE = DATA_DIR / "train.parquet"  # Do sprawdzania distribution shift
MODEL_LATEST = MODELS_DIR / "model_latest.pkl"  # Z retrain
BASELINE_MODEL = MODELS_DIR / "baseline_model_latest.pkl"  # Z baseline


def load_model():
    """
    Wczytaj najlepszy model
    
    Priorytet:
    1. model_latest.pkl (z retrain - jeśli istnieje)
    2. baseline_model_latest.pkl (z baseline - fallback)
    """
    print("📦 Wczytywanie modelu...")
    
    # Sprawdź czy jest model z retrain
    if MODEL_LATEST.exists():
        model_file = MODEL_LATEST
        print(f"   ✅ Znaleziono model z retrain: {model_file.name}")
    elif BASELINE_MODEL.exists():
        model_file = BASELINE_MODEL
        print(f"   ⚠️  Używam baseline model (brak retrain): {model_file.name}")
    else:
        raise FileNotFoundError(f"Brak modelu! Sprawdź: {MODEL_LATEST} lub {BASELINE_MODEL}")
    
    model_package = joblib.load(model_file)
    
    print(f"   Model: {model_package['name']}")
    print(f"   SMOTE: {model_package['use_smote']}")
    print(f"   Timestamp: {model_package['timestamp']}")
    
    # Sprawdź czy to model z iteration
    if 'iteration' in model_package:
        print(f"   🔄 Model z iteration: {model_package['iteration']}")
        print(f"   F1: {model_package['results']['f1']:.4f}")
        print(f"   Recall: {model_package['results']['recall']:.4f}")
    else:
        print(f"   📊 Baseline model")
        print(f"   Baseline F1: {model_package['results']['f1']:.4f}")
        print(f"   Baseline Recall: {model_package['results']['recall']:.4f}")
    
    print(f"   Optimal threshold: {model_package['optimal_threshold']:.4f}")
    
    return model_package


def load_pool_data():
    """Wczytaj pool danych (próbki bez adnotacji)"""
    print("\n📂 Wczytywanie pool danych...")
    
    df = pd.read_parquet(POOL_FILE)
    
    print(f"   Próbek w pool: {len(df):,}")
    print(f"   Kolumny: {list(df.columns)}")
    
    # Przygotuj X
    X = np.stack(df['l14_img'].values)
    print(f"   X shape: {X.shape}")
    
    return df, X


def predict_probabilities(model_package, X, batch_size=10000):
    """
    Przewiduj prawdopodobieństwa dla wszystkich próbek w pool
    
    batch_size: liczba próbek procesowanych naraz (dla zarządzania pamięcią)
    """
    print("\n🔮 Predykcja prawdopodobieństw...")
    
    model = model_package['model']
    scaler = model_package['scaler']
    
    # Skalowanie jeśli potrzebne
    if scaler:
        print("   Skalowanie danych...")
        X_processed = scaler.transform(X)
    else:
        X_processed = X
    
    # Predykcja w batch'ach
    n_samples = len(X_processed)
    n_batches = (n_samples + batch_size - 1) // batch_size
    
    print(f"   Próbek: {n_samples:,}")
    print(f"   Batch size: {batch_size:,}")
    print(f"   Liczba batchy: {n_batches}")
    
    y_pred_proba = []
    
    for i in tqdm(range(n_batches), desc="   Predykcja"):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, n_samples)
        
        batch_proba = model.predict_proba(X_processed[start_idx:end_idx])[:, 1]
        y_pred_proba.append(batch_proba)
    
    y_pred_proba = np.concatenate(y_pred_proba)
    
    print(f"\n   ✅ Predykcja zakończona")
    print(f"   Prawdopodobieństwa: min={y_pred_proba.min():.4f}, max={y_pred_proba.max():.4f}, mean={y_pred_proba.mean():.4f}")
    
    return y_pred_proba


def calculate_uncertainty_entropy(y_pred_proba):
    """
    Oblicz uncertainty używając entropii
    
    Entropy = -p*log(p) - (1-p)*log(1-p)
    
    Maksymalna entropia (max uncertainty) gdy p = 0.5
    Minimalna entropia (min uncertainty) gdy p = 0 lub p = 1
    """
    # Zabezpieczenie przed log(0)
    epsilon = 1e-10
    p = np.clip(y_pred_proba, epsilon, 1 - epsilon)
    
    # Binary entropy
    entropy = -p * np.log2(p) - (1 - p) * np.log2(1 - p)
    
    return entropy


def calculate_uncertainty_margin(y_pred_proba):
    """
    Oblicz uncertainty używając margin
    
    Margin = 1 - |p - 0.5| * 2
    
    Dla binary classification równoważne z entropy
    """
    # Odległość od decision boundary (0.5)
    margin = 1 - np.abs(y_pred_proba - 0.5) * 2
    
    return margin


def calculate_uncertainty_least_confident(y_pred_proba):
    """
    Oblicz uncertainty używając least confident
    
    Uncertainty = 1 - max(p, 1-p)
    """
    # Prawdopodobieństwo najlepszej klasy
    max_prob = np.maximum(y_pred_proba, 1 - y_pred_proba)
    
    # Uncertainty = 1 - confidence
    uncertainty = 1 - max_prob
    
    return uncertainty


def select_samples_to_annotate(df, y_pred_proba, n_samples=600, strategy='entropy', invalid_rate=0.38):
    """
    Wybierz n najbardziej niepewnych próbek (z buforem na invalid URLs)
    
    ⚠️  UWAGA: Pure uncertainty - wybiera tylko próbki z proba ≈ 0.5!
    Dla lepszych wyników użyj select_samples_stratified()
    
    strategy: 'entropy', 'margin', lub 'least_confident'
    invalid_rate: oczekiwany % invalid URLs (default: 38%)
    
    Wybieramy więcej próbek niż n_samples, żeby po odfiltrowaniu invalid
    zostało ~n_samples valid annotations
    """
    # Oblicz ile próbek wybrać z buforem
    n_with_buffer = int(n_samples / (1 - invalid_rate))
    
    print(f"\n🎯 Wybór próbek do adnotacji (PURE UNCERTAINTY)...")
    print(f"   ⚠️  UWAGA: Pure uncertainty może wybierać tylko edge cases!")
    print(f"   Strategia: {strategy}")
    print(f"   Target valid annotations: {n_samples}")
    print(f"   Oczekiwany % invalid URLs: {invalid_rate:.1%}")
    print(f"   Wybieramy z buforem: {n_with_buffer} próbek")
    print(f"   (spodziewamy się ~{int(n_with_buffer * (1-invalid_rate))} valid)")
    
    # Oblicz uncertainty
    if strategy == 'entropy':
        uncertainty = calculate_uncertainty_entropy(y_pred_proba)
    elif strategy == 'margin':
        uncertainty = calculate_uncertainty_margin(y_pred_proba)
    elif strategy == 'least_confident':
        uncertainty = calculate_uncertainty_least_confident(y_pred_proba)
    else:
        raise ValueError(f"Nieznana strategia: {strategy}")
    
    # Dodaj do DataFrame
    df_with_uncertainty = df.copy()
    df_with_uncertainty['pred_proba'] = y_pred_proba
    df_with_uncertainty['uncertainty'] = uncertainty
    
    # Sortuj po uncertainty (malejąco)
    df_sorted = df_with_uncertainty.sort_values('uncertainty', ascending=False)
    
    # Wybierz top N (z buforem)
    df_selected = df_sorted.head(n_with_buffer).copy()
    
    # Statystyki wybranych próbek
    print(f"\n   📊 Statystyki wybranych próbek:")
    print(f"      Uncertainty: min={df_selected['uncertainty'].min():.4f}, max={df_selected['uncertainty'].max():.4f}, mean={df_selected['uncertainty'].mean():.4f}")
    print(f"      Pred proba: min={df_selected['pred_proba'].min():.4f}, max={df_selected['pred_proba'].max():.4f}, mean={df_selected['pred_proba'].mean():.4f}")
    
    # Rozkład prawdopodobieństw
    bins = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    hist, _ = np.histogram(df_selected['pred_proba'], bins=bins)
    
    print(f"\n      Rozkład pred_proba (wybrane próbki):")
    for i in range(len(bins) - 1):
        pct = hist[i] / len(df_selected) * 100
        print(f"         [{bins[i]:.1f}-{bins[i+1]:.1f}): {hist[i]:4d} próbek ({pct:5.1f}%)")
    
    # Statystyki całego pool
    print(f"\n   📊 Statystyki całego pool (dla porównania):")
    print(f"      Uncertainty: mean={uncertainty.mean():.4f}, std={uncertainty.std():.4f}")
    print(f"      Pred proba: mean={y_pred_proba.mean():.4f}, std={y_pred_proba.std():.4f}")
    
    # Ile "potencjalnych pozytywnych" (proba > 0.5)?
    n_potential_positive = (df_selected['pred_proba'] > 0.5).sum()
    print(f"\n      Potencjalne YES (proba > 0.5): {n_potential_positive} ({n_potential_positive/len(df_selected):.1%})")
    
    return df_selected, df_with_uncertainty


def select_samples_stratified(df, y_pred_proba, n_samples=600, strategy='entropy', 
                               invalid_rate=0.38, positive_ratio=0.4):
    """
    🔴 REKOMENDOWANA: Stratified Uncertainty Sampling 
    
    Wybiera próbki z OBUĘ stron decision boundary:
    - positive_ratio próbek z proba > 0.5 (potential positives)
    - (1 - positive_ratio) próbek z proba ≤ 0.5 (potential negatives)
    
    To zapobiega wybieraniu TYLKO edge cases (proba ≈ 0.5) i pomaga
    modelowi zachować zdolność rozpoznawania obu klas.
    
    Args:
        df: DataFrame z pool
        y_pred_proba: prawdopodobieństwa
        n_samples: target liczba VALID annotations
        strategy: 'entropy', 'margin', 'least_confident'
        invalid_rate: oczekiwany % invalid URLs
        positive_ratio: jaki % wybranych próbek ma być "potential positives" (default: 0.4)
    """
    n_with_buffer = int(n_samples / (1 - invalid_rate))
    
    print(f"\n🎯 Wybór próbek do adnotacji (STRATIFIED UNCERTAINTY)...")
    print(f"   ✅ Stratyfikacja zapobiega wyborowi TYLKO edge cases")
    print(f"   Strategia: {strategy} (stratified)")
    print(f"   Target valid annotations: {n_samples}")
    print(f"   Positive ratio: {positive_ratio:.1%}")
    print(f"   Oczekiwany % invalid URLs: {invalid_rate:.1%}")
    print(f"   Wybieramy z buforem: {n_with_buffer} próbek")
    
    # Oblicz uncertainty
    if strategy == 'entropy':
        uncertainty = calculate_uncertainty_entropy(y_pred_proba)
    elif strategy == 'margin':
        uncertainty = calculate_uncertainty_margin(y_pred_proba)
    elif strategy == 'least_confident':
        uncertainty = calculate_uncertainty_least_confident(y_pred_proba)
    else:
        raise ValueError(f"Nieznana strategia: {strategy}")
    
    # Dodaj do DataFrame
    df_with_uncertainty = df.copy()
    df_with_uncertainty['pred_proba'] = y_pred_proba
    df_with_uncertainty['uncertainty'] = uncertainty
    
    # STRATYFIKACJA: Podziel na potential positive/negative
    threshold = 0.5
    df_potential_pos = df_with_uncertainty[df_with_uncertainty['pred_proba'] > threshold].copy()
    df_potential_neg = df_with_uncertainty[df_with_uncertainty['pred_proba'] <= threshold].copy()
    
    # Oblicz ile z każdej grupy
    n_pos = int(n_with_buffer * positive_ratio)
    n_neg = n_with_buffer - n_pos
    
    print(f"\n   📊 Stratyfikacja:")
    print(f"      Potential positives (proba > {threshold}): {len(df_potential_pos):,} dostępnych")
    print(f"      Potential negatives (proba ≤ {threshold}): {len(df_potential_neg):,} dostępnych")
    print(f"      Wybieramy: {n_pos} positives + {n_neg} negatives")
    
    # Sortuj każdą grupę po uncertainty
    df_pos_sorted = df_potential_pos.sort_values('uncertainty', ascending=False)
    df_neg_sorted = df_potential_neg.sort_values('uncertainty', ascending=False)
    
    # Wybierz top N z każdej grupy (z fallback jeśli za mało)
    n_pos_actual = min(n_pos, len(df_pos_sorted))
    n_neg_actual = min(n_neg, len(df_neg_sorted))
    
    if n_pos_actual < n_pos:
        print(f"      ⚠️  Za mało potential positives! Wybrano {n_pos_actual}/{n_pos}")
        # Dokup z negatives
        n_neg_actual = min(n_with_buffer - n_pos_actual, len(df_neg_sorted))
    
    df_selected_pos = df_pos_sorted.head(n_pos_actual)
    df_selected_neg = df_neg_sorted.head(n_neg_actual)
    
    # Połącz
    df_selected = pd.concat([df_selected_pos, df_selected_neg], ignore_index=True)
    
    # Shuffle (żeby nie było widać stratyfikacji w interface)
    df_selected = df_selected.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Statystyki wybranych próbek
    print(f"\n   📊 Statystyki wybranych próbek:")
    print(f"      Total: {len(df_selected)}")
    print(f"      Actual positives: {n_pos_actual} ({n_pos_actual/len(df_selected):.1%})")
    print(f"      Actual negatives: {n_neg_actual} ({n_neg_actual/len(df_selected):.1%})")
    print(f"      Uncertainty: min={df_selected['uncertainty'].min():.4f}, max={df_selected['uncertainty'].max():.4f}, mean={df_selected['uncertainty'].mean():.4f}")
    print(f"      Pred proba: min={df_selected['pred_proba'].min():.4f}, max={df_selected['pred_proba'].max():.4f}, mean={df_selected['pred_proba'].mean():.4f}")
    
    # Rozkład prawdopodobieństw
    bins = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    hist, _ = np.histogram(df_selected['pred_proba'], bins=bins)
    
    print(f"\n      Rozkład pred_proba (wybrane próbki):")
    for i in range(len(bins) - 1):
        pct = hist[i] / len(df_selected) * 100
        print(f"         [{bins[i]:.1f}-{bins[i+1]:.1f}): {hist[i]:4d} próbek ({pct:5.1f}%)")
    
    # WARNING jeśli bardzo niezbalansowane
    potential_yes_ratio = (df_selected['pred_proba'] > 0.5).sum() / len(df_selected)
    if potential_yes_ratio < 0.2:
        print(f"\n      ⚠️  WARNING: Tylko {potential_yes_ratio:.1%} potential positives!")
        print(f"         Model może stać się zbyt konserwatywny w następnej iteracji")
    elif potential_yes_ratio > 0.8:
        print(f"\n      ⚠️  WARNING: Aż {potential_yes_ratio:.1%} potential positives!")
        print(f"         Model może stracić zdolność odrzucania negatywów")
    
    return df_selected, df_with_uncertainty


def analyze_selection(df_selected, df_all, train_score_mean=None):
    """
    Analiza wybranych próbek + sprawdzenie distribution shift
    
    Args:
        df_selected: wybrane próbki
        df_all: cały pool
        train_score_mean: średni score z train set (dla porównania distribution shift)
    """
    print(f"\n🔍 Analiza wyboru...")
    
    # Porównanie ze score słownikowym (jeśli dostępne)
    if 'score' in df_selected.columns:
        selected_score = df_selected['score'].mean()
        pool_score = df_all['score'].mean()
        
        print(f"\n   📊 Dictionary score:")
        print(f"      Wybrane próbki: {selected_score:.3f}")
        print(f"      Cały pool:      {pool_score:.3f}")
        
        if train_score_mean is not None:
            print(f"      Train set:      {train_score_mean:.3f}")
            
            # 🔴 KLUCZOWE: Distribution shift warning
            score_diff_train = selected_score - train_score_mean
            print(f"\n      Różnica (selected - train): {score_diff_train:+.3f}")
            
            if abs(score_diff_train) > 1.0:
                print(f"\n      🔴 WARNING: DUŻY DISTRIBUTION SHIFT!")
                print(f"         Selected score znacznie różni się od train")
                print(f"         Model może być overconfident lub underconfident")
                
                if score_diff_train < -1.0:
                    print(f"\n         → Selected ma NIŻSZY score (diff: {score_diff_train:.3f})")
                    print(f"         → Model trenowany na score={train_score_mean:.3f}")
                    print(f"         → Teraz wybiera próbki ze score={selected_score:.3f}")
                    print(f"         → RYZYKO: Model może być OVERCONFIDENT")
                    print(f"         → ROZWIĄZANIE: Rozważ dodanie 'easy positives'")
                    print(f"            (wysokie score + wysokie proba)")
                else:
                    print(f"\n         → Selected ma WYŻSZY score (diff: {score_diff_train:+.3f})")
                    print(f"         → To DOBRZE - szukamy map w bogatszym terenie!")
            elif abs(score_diff_train) > 0.5:
                print(f"      ⚠️  Moderate distribution shift (różnica: {score_diff_train:+.3f})")
                print(f"         Monitor performance po tej iteracji")
            else:
                print(f"      ✅ Distribution shift OK (różnica: {score_diff_train:+.3f})")
        
        # Czy wybrane próbki mają wyższy/niższy score od pool?
        score_diff_pool = selected_score - pool_score
        print(f"\n      Różnica (selected - pool): {score_diff_pool:+.3f}")
        if score_diff_pool > 0:
            print(f"         → Wybrane próbki mają WYŻSZY score (potencjalnie więcej map)")
        elif score_diff_pool < -0.1:
            print(f"         → Wybrane próbki mają NIŻSZY score")
        else:
            print(f"         → Wybrane próbki reprezentatywne dla pool")


def save_iteration(df_selected, iteration_num, strategy, model_package, stats):
    """
    Zapisz wybraną iterację
    """
    print(f"\n💾 Zapisywanie iteracji {iteration_num}...")
    
    # Utwórz katalog iteracji
    iteration_dir = ITERATIONS_DIR / f"iteration_{iteration_num}"
    iteration_dir.mkdir(parents=True, exist_ok=True)
    
    # Przygotuj dane do adnotacji (bez uncertainty scores - żeby nie wpływać na annotator)
    to_annotate = df_selected[['uid', 'url', 'pred_proba']].copy()
    to_annotate_file = iteration_dir / "to_annotate.parquet"
    to_annotate.to_parquet(to_annotate_file, index=False)
    
    print(f"   ✅ Próbki do adnotacji: {to_annotate_file}")
    print(f"      Liczba próbek: {len(to_annotate)}")
    
    # Zapisz pełne dane (z uncertainty) dla analizy
    full_data_file = iteration_dir / "to_annotate_full.parquet"
    df_selected.to_parquet(full_data_file, index=False)
    
    print(f"   ✅ Pełne dane (z uncertainty): {full_data_file}")
    
    # Zapisz metadata
    metadata = {
        'iteration': iteration_num,
        'timestamp': datetime.now().isoformat(),
        'n_samples': len(df_selected),
        'strategy': strategy,
        'model': {
            'name': model_package['name'],
            'use_smote': model_package['use_smote'],
            'baseline_f1': model_package['results']['f1'],
            'baseline_recall': model_package['results']['recall'],
            'optimal_threshold': model_package['optimal_threshold']
        },
        'stats': stats
    }
    
    metadata_file = iteration_dir / "metadata.json"
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"   ✅ Metadata: {metadata_file}")
    
    return iteration_dir


def create_annotation_instructions(iteration_dir, n_samples):
    """
    Utwórz plik z instrukcjami adnotacji
    """
    iteration_num = int(iteration_dir.name.split('_')[1])
    
    instructions = f"""# Instrukcje adnotacji - Iteracja {iteration_num}

## 📋 Zadanie
Zaadnotuj {n_samples} **VALID** próbek jako:
- **YES** - mapa statystyczna
- **NO** - nie jest mapą statystyczną
- **INVALID** - URL nie działa lub błędny obraz (NIE LICZĄ SIĘ do targetu)

## 🎯 Cel
Active Learning wybrał te próbki jako **najbardziej niepewne** dla modelu.
Są to przypadki graniczne, które pomogą modelowi nauczyć się lepiej.

## 🔧 Jak adnotować?

### POLECANY: Użyj interface stage3 (automatycznie pomija invalid)

```bash
cd MapPool/stage3
streamlit run annotation_interface.py -- --iteration {iteration_num} --target {n_samples}
```

**Automatyczne funkcje:**
- ✅ HEAD request sprawdza URL przed wyświetleniem
- ✅ Automatycznie pomija invalid URLs (NIE liczą się do targetu!)
- ✅ Zapisuje progress po każdej adnotacji
- ✅ Resume: możesz przerwać i wrócić później
- ✅ Pokazuje model prediction dla każdej próbki

## 📊 Co zobaczysz?

- **Więcej map** niż w pierwszej rundzie (model teraz szuka trudnych przypadków)
- **Case'y graniczne:** mapy niskiej jakości, częściowe mapy, edge cases
- **Model prediction hint:** interfejs pokazuje co model myśli o próbce
- **Niepewne przypadki:** prawdopodobieństwa blisko 0.5

## ✅ Po adnotacji

Interface automatycznie zapisze wyniki do:
- `{iteration_dir.name}/annotated.parquet` (gdy osiągniesz target)
- `{iteration_dir.name}/annotations_progress.jsonl` (progress)

Następnie możesz:
1. Użyć `retrain_model.py` (TODO) żeby retrenować model
2. Uruchomić kolejną iterację active learning

## 🎲 Statystyki

Interface wybierze więcej próbek niż {n_samples} (buffer na invalid URLs).
Oczekiwany % invalid: ~38%, więc wybranych zostanie ~{int(n_samples / 0.62)} próbek.
Ale adnotujesz tylko do osiągnięcia {n_samples} VALID!
"""
    
    instructions_file = iteration_dir / "INSTRUCTIONS.md"
    with open(instructions_file, 'w') as f:
        f.write(instructions)
    
    print(f"   ✅ Instrukcje: {instructions_file}")


def main(n_samples=600, strategy='entropy', iteration_num=1, invalid_rate=0.38,
         positive_ratio=0.4, use_stratified=True):
    """
    Główna funkcja active learning
    
    Args:
        n_samples: target liczba VALID annotations (interface pominie invalid)
        strategy: strategia uncertainty ('entropy', 'margin', 'least_confident')
        iteration_num: numer iteracji
        invalid_rate: oczekiwany % invalid URLs dla bufora (default: 38%)
        positive_ratio: dla stratified - jaki % potential positives (default: 0.4)
        use_stratified: czy użyć stratyfikacji (default: True - REKOMENDOWANE!)
    """
    print("\n" + "="*80)
    print("🎯 ACTIVE LEARNING - WYBÓR PRÓBEK DO ADNOTACJI")
    print("="*80)
    
    # 1. Wczytaj train score dla sprawdzenia distribution shift
    train_score_mean = None
    if TRAIN_FILE.exists():
        print("\n📊 Wczytywanie train statistics...")
        df_train = pd.read_parquet(TRAIN_FILE)
        if 'score' in df_train.columns:
            train_score_mean = df_train['score'].mean()
            print(f"   Train mean score: {train_score_mean:.3f}")
    
    # 2. Wczytaj model
    model_package = load_model()
    
    # 3. Wczytaj pool danych
    df_pool, X_pool = load_pool_data()
    
    # 4. Przewiduj prawdopodobieństwa
    y_pred_proba = predict_probabilities(model_package, X_pool)
    
    # 5. Wybierz próbki do adnotacji
    if use_stratified:
        print(f"\n✅ Używam STRATIFIED uncertainty sampling (REKOMENDOWANE)")
        df_selected, df_all = select_samples_stratified(
            df_pool, y_pred_proba, 
            n_samples=n_samples, 
            strategy=strategy,
            invalid_rate=invalid_rate,
            positive_ratio=positive_ratio
        )
    else:
        print(f"\n⚠️  Używam PURE uncertainty sampling (może wybierać tylko edge cases!)")
        df_selected, df_all = select_samples_to_annotate(
            df_pool, y_pred_proba, 
            n_samples=n_samples, 
            strategy=strategy,
            invalid_rate=invalid_rate
        )
    
    # 6. Analiza wyboru (z train score dla sprawdzenia distribution shift)
    analyze_selection(df_selected, df_all, train_score_mean=train_score_mean)
    
    # 7. Zapisz iterację
    stats = {
        'n_samples_selected': len(df_selected),
        'uncertainty_mean': float(df_selected['uncertainty'].mean()),
        'uncertainty_std': float(df_selected['uncertainty'].std()),
        'pred_proba_mean': float(df_selected['pred_proba'].mean()),
        'pred_proba_std': float(df_selected['pred_proba'].std()),
        'potential_positive_count': int((df_selected['pred_proba'] > 0.5).sum()),
        'potential_positive_ratio': float((df_selected['pred_proba'] > 0.5).mean()),
        'stratified': use_stratified,
        'positive_ratio': positive_ratio if use_stratified else None
    }
    
    if 'score' in df_selected.columns:
        stats['selected_score_mean'] = float(df_selected['score'].mean())
        stats['pool_score_mean'] = float(df_all['score'].mean())
        if train_score_mean is not None:
            stats['train_score_mean'] = float(train_score_mean)
            stats['distribution_shift'] = float(df_selected['score'].mean() - train_score_mean)
    
    iteration_dir = save_iteration(df_selected, iteration_num, strategy, model_package, stats)
    
    # 7. Utwórz instrukcje
    create_annotation_instructions(iteration_dir, n_samples)
    
    # Podsumowanie
    print("\n" + "="*80)
    print("✅ ACTIVE LEARNING - GOTOWE!")
    print("="*80)
    print(f"\n📁 Katalog iteracji: {iteration_dir}")
    print(f"📝 Próbek do adnotacji: {n_samples}")
    print(f"🎲 Strategia: {strategy}")
    
    print(f"\n🔮 Predykcje modelu dla wybranych próbek:")
    print(f"   Potencjalne YES (proba > 0.5): {stats['potential_positive_count']} ({stats['potential_positive_ratio']:.1%})")
    print(f"   Średnie prawdopodobieństwo: {stats['pred_proba_mean']:.3f}")
    
    print(f"\n🎯 Następne kroki:")
    print(f"   1. Zaadnotuj próbki w: {iteration_dir}/to_annotate.parquet")
    print(f"   2. Zapisz wyniki jako: {iteration_dir}/annotated.parquet")
    print(f"   3. Użyj annotation interface z stage2 lub zaadnotuj ręcznie")
    print(f"   4. Po adnotacji uruchom retrain (TODO: implement)")
    
    print("\n" + "="*80)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Active Learning - wybór próbek do adnotacji')
    parser.add_argument('--n-samples', type=int, default=600, 
                       help='Target liczba VALID annotations - interface automatycznie pominie invalid (default: 600)')
    parser.add_argument('--strategy', type=str, default='entropy', 
                       choices=['entropy', 'margin', 'least_confident'],
                       help='Strategia uncertainty sampling (default: entropy)')
    parser.add_argument('--iteration', type=int, default=1, help='Numer iteracji (default: 1)')
    parser.add_argument('--invalid-rate', type=float, default=0.38,
                       help='Oczekiwany %% invalid URLs dla bufora (default: 0.38 = 38%%)')
    parser.add_argument('--positive-ratio', type=float, default=0.4,
                       help='Dla stratified: jaki %% potential positives (default: 0.4 = 40%%)')
    parser.add_argument('--no-stratified', action='store_true',
                       help='Wyłącz stratyfikację (użyj pure uncertainty) - NIE REKOMENDOWANE!')
    
    args = parser.parse_args()
    
    main(
        n_samples=args.n_samples,
        strategy=args.strategy,
        iteration_num=args.iteration,
        invalid_rate=args.invalid_rate,
        positive_ratio=args.positive_ratio,
        use_stratified=not args.no_stratified
    )
