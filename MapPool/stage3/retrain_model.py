# -*- coding: utf-8 -*-
"""
Retrain modelu po active learning iteration

Łączy:
- Baseline train data (3000 próbek)
- Iteration N annotations (np. 601 próbek)
= Extended train data (np. 3601 próbek)

Trenuje wszystkie modele i porównuje z baseline
"""

import pandas as pd
import numpy as np
import json
import joblib
from pathlib import Path
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    confusion_matrix, roc_auc_score, average_precision_score,
    precision_recall_curve, f1_score, precision_score, 
    recall_score, accuracy_score
)
import warnings
warnings.filterwarnings('ignore')

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("⚠️  XGBoost not available - skipping XGBoost models")

try:
    from imblearn.over_sampling import SMOTE
    SMOTE_AVAILABLE = True
except ImportError:
    SMOTE_AVAILABLE = False
    print("⚠️  imblearn not available - skipping SMOTE variants")

# Ścieżki
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
ITERATIONS_DIR = BASE_DIR / "iterations"
RESULTS_DIR = BASE_DIR / "results"

BASELINE_TRAIN = DATA_DIR / "train.parquet"
BASELINE_VAL_UIDS = DATA_DIR / "baseline_val_uids.json"
EMBEDDINGS_FILE = BASE_DIR.parent / "data_all" / "above_value_minus1" / "sample_200k_with_embeddings.parquet"


def load_baseline_train():
    """Wczytaj baseline train data"""
    print("📂 Wczytywanie baseline train data...")
    df = pd.read_parquet(BASELINE_TRAIN)
    print(f"   Próbek: {len(df)}")
    print(f"   YES: {(df['label'] == 1).sum()} ({(df['label'] == 1).mean():.2%})")
    print(f"   NO: {(df['label'] == 0).sum()} ({(df['label'] == 0).mean():.2%})")
    return df


def load_iteration_annotations(iteration_num):
    """Wczytaj adnotacje z konkretnej iteracji"""
    print(f"\n📂 Wczytywanie iteration_{iteration_num} annotations...")
    
    iteration_dir = ITERATIONS_DIR / f"iteration_{iteration_num}"
    
    # Sprawdź różne możliwe pliki
    possible_files = [
        iteration_dir / "annotated.parquet",
        iteration_dir / "annotated_partial.parquet",
        iteration_dir / "annotations_progress.jsonl"
    ]
    
    df = None
    for file_path in possible_files:
        if file_path.exists():
            print(f"   Znaleziono: {file_path.name}")
            
            if file_path.suffix == '.parquet':
                df = pd.read_parquet(file_path)
            elif file_path.suffix == '.jsonl':
                # Wczytaj JSONL i konwertuj do DataFrame
                records = []
                with open(file_path, 'r') as f:
                    for line in f:
                        if line.strip():
                            records.append(json.loads(line))
                df = pd.DataFrame(records)
            
            break
    
    if df is None:
        raise FileNotFoundError(f"Nie znaleziono pliku z adnotacjami w {iteration_dir}")
    
    # Filtruj tylko YES/NO (bez INVALID)
    df = df[df['label'].isin(['YES', 'NO'])].copy()
    
    # Mapuj na 0/1
    df['label'] = df['label'].map({'YES': 1, 'NO': 0})
    
    print(f"   Valid annotations: {len(df)}")
    print(f"   YES: {(df['label'] == 1).sum()} ({(df['label'] == 1).mean():.2%})")
    print(f"   NO: {(df['label'] == 0).sum()} ({(df['label'] == 0).mean():.2%})")
    
    return df


def merge_with_embeddings(df):
    """Merge annotations z embeddingami"""
    print(f"\n🔗 Merge z embeddingami...")
    
    # Wczytaj embeddingi tylko dla potrzebnych uidów
    print(f"   Wczytywanie embeddingów...")
    df_embeddings = pd.read_parquet(EMBEDDINGS_FILE)
    
    # Merge
    df_merged = df.merge(df_embeddings, on='uid', how='inner')
    
    print(f"   Po merge: {len(df_merged)} próbek")
    
    missing = len(df) - len(df_merged)
    if missing > 0:
        print(f"   ⚠️  Brak embeddingów dla {missing} próbek")
    
    return df_merged


def combine_datasets(baseline_df, iteration_df):
    """Połącz baseline i iteration data"""
    print(f"\n🔀 Łączenie datasets...")
    
    # Sprawdź czy są duplikaty
    baseline_uids = set(baseline_df['uid'])
    iteration_uids = set(iteration_df['uid'])
    
    duplicates = baseline_uids & iteration_uids
    if duplicates:
        print(f"   ⚠️  Znaleziono {len(duplicates)} duplikatów - usuwam z iteration")
        iteration_df = iteration_df[~iteration_df['uid'].isin(duplicates)].copy()
    
    # Połącz
    combined_df = pd.concat([baseline_df, iteration_df], ignore_index=True)
    
    print(f"   Baseline: {len(baseline_df)} próbek")
    print(f"   Iteration: {len(iteration_df)} próbek")
    print(f"   Combined: {len(combined_df)} próbek")
    print(f"   YES: {(combined_df['label'] == 1).sum()} ({(combined_df['label'] == 1).mean():.2%})")
    print(f"   NO: {(combined_df['label'] == 0).sum()} ({(combined_df['label'] == 0).mean():.2%})")
    
    return combined_df


def prepare_train_data(df):
    """Przygotuj X, y"""
    X = np.stack(df['l14_img'].values)
    y = df['label'].values
    uids = df['uid'].values
    
    return X, y, uids


def create_fixed_train_val_split(X, y, uids):
    """
    Split używając STAŁEGO validation set z baseline
    
    To zapewnia fair comparison - testujemy na tych samych danych co baseline!
    """
    print(f"\n🔀 Podział train/val (STAŁY validation set z baseline)...")
    
    # Wczytaj baseline val UIDs
    if not BASELINE_VAL_UIDS.exists():
        raise FileNotFoundError(f"Brak pliku z baseline val UIDs: {BASELINE_VAL_UIDS}")
    
    with open(BASELINE_VAL_UIDS, 'r') as f:
        baseline_val_uids = set(json.load(f))
    
    print(f"   Baseline val UIDs: {len(baseline_val_uids)}")
    
    # Split na podstawie UIDs
    val_mask = np.array([uid in baseline_val_uids for uid in uids])
    train_mask = ~val_mask
    
    X_train = X[train_mask]
    y_train = y[train_mask]
    uid_train = uids[train_mask]
    
    X_val = X[val_mask]
    y_val = y[val_mask]
    uid_val = uids[val_mask]
    
    print(f"   Train: {len(y_train)} próbek (baseline_train + iteration)")
    print(f"      YES: {(y_train == 1).sum()} ({(y_train == 1).mean():.2%})")
    print(f"      NO: {(y_train == 0).sum()} ({(y_train == 0).mean():.2%})")
    
    print(f"   Val: {len(y_val)} próbek (STAŁY - taki sam jak baseline!)")
    print(f"      YES: {(y_val == 1).sum()} ({(y_val == 1).mean():.2%})")
    print(f"      NO: {(y_val == 0).sum()} ({(y_val == 0).mean():.2%})")
    
    # Sprawdź czy val set się zgadza z baseline
    expected_val_yes = 14  # z baseline
    if (y_val == 1).sum() != expected_val_yes:
        print(f"   ⚠️  UWAGA: Val set nie zgadza się z baseline!")
        print(f"      Oczekiwano {expected_val_yes} YES, mam {(y_val == 1).sum()}")
    else:
        print(f"   ✅ Val set zgodny z baseline!")
    
    return X_train, X_val, y_train, y_val, uid_train, uid_val


def get_models():
    """Modele do trenowania (wszystkie z baseline dla porównania)"""
    models = {}
    
    # 1. Logistic Regression - baseline winner
    models['LogisticRegression'] = {
        'model': LogisticRegression(
            class_weight='balanced',
            max_iter=1000,
            random_state=42,
            n_jobs=-1
        ),
        'scale': True,
        'description': 'Logistic Regression z class_weight=balanced'
    }
    
    # 2. Logistic Regression L1
    models['LogisticRegression_L1'] = {
        'model': LogisticRegression(
            class_weight='balanced',
            penalty='l1',
            solver='liblinear',
            max_iter=1000,
            random_state=42
        ),
        'scale': True,
        'description': 'Logistic Regression L1 (sparse)'
    }
    
    # 3. Random Forest
    models['RandomForest'] = {
        'model': RandomForestClassifier(
            n_estimators=200,
            class_weight='balanced',
            max_depth=20,
            min_samples_split=10,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        ),
        'scale': False,
        'description': 'Random Forest z class_weight=balanced'
    }
    
    # 4. Random Forest Subsample
    models['RandomForest_Subsample'] = {
        'model': RandomForestClassifier(
            n_estimators=200,
            class_weight='balanced_subsample',
            max_depth=20,
            min_samples_split=10,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        ),
        'scale': False,
        'description': 'Random Forest z balanced_subsample'
    }
    
    # 5. XGBoost
    if XGBOOST_AVAILABLE:
        models['XGBoost'] = {
            'model': xgb.XGBClassifier(
                n_estimators=200,
                max_depth=8,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1,
                eval_metric='logloss'
            ),
            'scale': False,
            'description': 'XGBoost z scale_pos_weight'
        }
    
    # 6. SVM RBF
    models['SVM_RBF'] = {
        'model': SVC(
            kernel='rbf',
            class_weight='balanced',
            probability=True,
            random_state=42
        ),
        'scale': True,
        'description': 'SVM z RBF kernel'
    }
    
    return models


def apply_smote(X_train, y_train, sampling_strategy=0.3):
    """SMOTE oversampling"""
    if not SMOTE_AVAILABLE:
        print("   ⚠️  SMOTE nie jest dostępny - zwracam oryginalne dane")
        return X_train, y_train
    
    smote = SMOTE(
        sampling_strategy=sampling_strategy,
        random_state=42,
        k_neighbors=min(3, (y_train == 1).sum() - 1)
    )
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
    return X_resampled, y_resampled


def train_and_evaluate_model(name, model_config, X_train, y_train, X_val, y_val, use_smote=False):
    """Trenuj i ewaluuj model"""
    print(f"\n{'='*60}")
    print(f"🤖 Model: {name}")
    if use_smote:
        print(f"   + SMOTE oversampling")
    print(f"{'='*60}")
    
    # Preprocessing
    scaler = None
    X_train_processed = X_train.copy()
    X_val_processed = X_val.copy()
    
    if model_config['scale']:
        scaler = StandardScaler()
        X_train_processed = scaler.fit_transform(X_train)
        X_val_processed = scaler.transform(X_val)
    
    # SMOTE
    if use_smote:
        X_train_processed, y_train_processed = apply_smote(X_train_processed, y_train)
    else:
        y_train_processed = y_train
    
    # XGBoost scale_pos_weight
    model = model_config['model']
    if 'XGBoost' in name:
        scale_pos_weight = (y_train_processed == 0).sum() / (y_train_processed == 1).sum()
        model.set_params(scale_pos_weight=scale_pos_weight)
    
    # Train
    print("   Trenowanie...")
    model.fit(X_train_processed, y_train_processed)
    
    # Predict
    y_val_pred_proba = model.predict_proba(X_val_processed)[:, 1]
    y_val_pred = model.predict(X_val_processed)
    
    # Metrics
    results = calculate_metrics(y_val, y_val_pred, y_val_pred_proba)
    
    # Optimal threshold
    optimal_threshold, optimal_f1 = find_optimal_threshold(y_val, y_val_pred_proba)
    results['optimal_threshold'] = optimal_threshold
    results['optimal_f1'] = optimal_f1
    
    print_results(results, optimal_threshold)
    
    # Bootstrap CI
    bootstrap_stats = analyze_validation_stability(y_val, y_val_pred_proba, name)
    results['bootstrap'] = bootstrap_stats
    
    return {
        'model': model,
        'scaler': scaler,
        'results': results,
        'name': name,
        'use_smote': use_smote
    }


def calculate_metrics(y_true, y_pred, y_pred_proba):
    """Oblicz metryki"""
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    roc_auc = roc_auc_score(y_true, y_pred_proba) if len(np.unique(y_true)) > 1 else 0.0
    pr_auc = average_precision_score(y_true, y_pred_proba) if len(np.unique(y_true)) > 1 else 0.0
    
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': roc_auc,
        'pr_auc': pr_auc,
        'specificity': specificity,
        'confusion_matrix': {
            'TP': int(tp), 'FP': int(fp),
            'TN': int(tn), 'FN': int(fn)
        }
    }


def find_optimal_threshold(y_true, y_pred_proba):
    """Znajdź optymalny threshold"""
    precision_vals, recall_vals, thresholds = precision_recall_curve(y_true, y_pred_proba)
    f1_scores = 2 * (precision_vals * recall_vals) / (precision_vals + recall_vals + 1e-10)
    optimal_idx = np.argmax(f1_scores)
    optimal_threshold = thresholds[optimal_idx] if optimal_idx < len(thresholds) else 0.5
    optimal_f1 = f1_scores[optimal_idx]
    return optimal_threshold, optimal_f1


def print_results(results, optimal_threshold=None):
    """Wyświetl wyniki"""
    print(f"\n📊 Wyniki (threshold=0.5):")
    print(f"   Accuracy:  {results['accuracy']:.4f}")
    print(f"   Precision: {results['precision']:.4f}")
    print(f"   Recall:    {results['recall']:.4f}")
    print(f"   F1-score:  {results['f1']:.4f}")
    print(f"   ROC-AUC:   {results['roc_auc']:.4f}")
    print(f"   PR-AUC:    {results['pr_auc']:.4f}")
    
    cm = results['confusion_matrix']
    print(f"\n   Confusion Matrix:")
    print(f"      TP: {cm['TP']:4d}  FP: {cm['FP']:4d}")
    print(f"      FN: {cm['FN']:4d}  TN: {cm['TN']:4d}")
    
    if optimal_threshold:
        print(f"\n🎯 Optymalny threshold: {optimal_threshold:.4f}")
        print(f"   F1-score: {results['optimal_f1']:.4f}")


def analyze_validation_stability(y_val, y_pred_proba, model_name, n_bootstrap=1000):
    """Bootstrap confidence intervals dla metryk"""
    print(f"\n🔬 Bootstrap analiza stabilności ({model_name}, n={n_bootstrap}):")
    
    f1_scores = []
    precision_scores = []
    recall_scores = []
    
    for _ in range(n_bootstrap):
        indices = np.random.choice(len(y_val), len(y_val), replace=True)
        y_boot = y_val[indices]
        y_pred_boot = (y_pred_proba[indices] >= 0.5).astype(int)
        
        f1_scores.append(f1_score(y_boot, y_pred_boot, zero_division=0))
        precision_scores.append(precision_score(y_boot, y_pred_boot, zero_division=0))
        recall_scores.append(recall_score(y_boot, y_pred_boot, zero_division=0))
    
    f1_scores = np.array(f1_scores)
    precision_scores = np.array(precision_scores)
    recall_scores = np.array(recall_scores)
    
    print(f"   F1:        {f1_scores.mean():.4f} ± {f1_scores.std():.4f}")
    print(f"              95% CI: [{np.percentile(f1_scores, 2.5):.4f}, {np.percentile(f1_scores, 97.5):.4f}]")
    print(f"   Precision: {precision_scores.mean():.4f} ± {precision_scores.std():.4f}")
    print(f"   Recall:    {recall_scores.mean():.4f} ± {recall_scores.std():.4f}")
    
    if f1_scores.std() > 0.1:
        print(f"   ⚠️  UWAGA: Duża niestabilność (std > 0.1)!")
    
    return {
        'f1_mean': float(f1_scores.mean()),
        'f1_std': float(f1_scores.std()),
        'f1_ci_lower': float(np.percentile(f1_scores, 2.5)),
        'f1_ci_upper': float(np.percentile(f1_scores, 97.5))
    }


def track_all_models_over_iterations(iteration_num):
    """Śledź WSZYSTKIE modele przez iteracje"""
    print("\n" + "="*80)
    print("📈 TRENDY WSZYSTKICH MODELI PRZEZ ITERACJE")
    print("="*80)
    
    baseline_files = sorted(RESULTS_DIR.glob("baseline_results_*.json"))
    if not baseline_files:
        print("⚠️  Brak baseline results")
        return None
    
    with open(baseline_files[-1], 'r') as f:
        baseline_data = json.load(f)
    
    history = {}
    
    for model_data in baseline_data['all_models']:
        model_name = model_data['Model']
        history[model_name] = [{
            'iteration': 0,
            'f1': model_data['F1'],
            'recall': model_data['Recall'],
            'precision': model_data['Precision'],
            'pr_auc': model_data['PR-AUC']
        }]
    
    for i in range(1, iteration_num + 1):
        files = list(RESULTS_DIR.glob(f"iteration_{i}_results_*.json"))
        if files:
            with open(files[-1], 'r') as f:
                data = json.load(f)
            
            for model_data in data['all_models']:
                model_name = model_data['Model']
                if model_name not in history:
                    history[model_name] = []
                
                history[model_name].append({
                    'iteration': i,
                    'f1': model_data['F1'],
                    'recall': model_data['Recall'],
                    'precision': model_data['Precision'],
                    'pr_auc': model_data['PR-AUC']
                })
    
    print(f"\n📊 Porównanie PER-MODEL (baseline → iteration_{iteration_num}):\n")
    print(f"{'Model':<30} {'Baseline':<12} {'Iter_{}':<12} {'Zmiana':<15} {'Trend'}".format(iteration_num))
    print("-" * 85)
    
    trends = {}
    for model_name, iterations in sorted(history.items()):
        if len(iterations) >= 2:
            baseline_f1 = iterations[0]['f1']
            latest_f1 = iterations[-1]['f1']
            change = latest_f1 - baseline_f1
            change_pct = (change / baseline_f1 * 100) if baseline_f1 > 0 else 0
            
            arrow = "📈" if change > 0 else "📉" if change < 0 else "➡️"
            print(f"{model_name:<30} {baseline_f1:<12.4f} {latest_f1:<12.4f} {change:+.4f} ({change_pct:+5.1f}%) {arrow}")
            
            trends[model_name] = {
                'baseline_f1': baseline_f1,
                'latest_f1': latest_f1,
                'change': change
            }
    
    print(f"\n🏆 TOP 3 - Największa poprawa:")
    sorted_trends = sorted(trends.items(), key=lambda x: x[1]['change'], reverse=True)
    for model_name, trend in sorted_trends[:3]:
        print(f"   {model_name}: {trend['change']:+.4f}")
    
    return trends


def compare_same_models_baseline_vs_iteration(all_results, iteration_num):
    """Porównaj TEN SAM model: baseline vs iteration"""
    print("\n" + "="*80)
    print("🔍 SZCZEGÓŁOWE PORÓWNANIE: Ten sam model baseline vs iteration")
    print("="*80)
    
    baseline_files = sorted(RESULTS_DIR.glob("baseline_results_*.json"))
    if not baseline_files:
        print("⚠️  Brak baseline results")
        return None
    
    with open(baseline_files[-1], 'r') as f:
        baseline_data = json.load(f)
    
    baseline_map = {}
    for model_data in baseline_data['all_models']:
        baseline_map[model_data['Model']] = model_data
    
    print(f"\n{'Model':<35} {'Baseline F1':<15} {'Iteration F1':<15} {'Zmiana'}")
    print("-" * 80)
    
    comparisons = []
    for result in all_results:
        model_name = result['name'] + (' + SMOTE' if result['use_smote'] else '')
        
        if model_name in baseline_map:
            baseline_f1 = baseline_map[model_name]['F1']
            iter_f1 = result['results']['f1']
            change = iter_f1 - baseline_f1
            change_pct = (change / baseline_f1 * 100) if baseline_f1 > 0 else 0
            
            arrow = "📈" if change > 0 else "📉" if change < 0 else "➡️"
            print(f"{model_name:<35} {baseline_f1:<15.4f} {iter_f1:<15.4f} {arrow} {change:+.4f} ({change_pct:+.1f}%)")
            
            comparisons.append({
                'model': model_name,
                'baseline_f1': baseline_f1,
                'iteration_f1': iter_f1,
                'change': change
            })
    
    return comparisons


def compare_with_baseline(new_results, iteration_num):
    """Porównaj z baseline - TERAZ FAIR (ten sam validation set!)"""
    print("\n" + "="*80)
    print("📊 PORÓWNANIE Z BASELINE (na tym samym validation set!)")
    print("="*80)
    
    # Wczytaj baseline results
    baseline_files = sorted(RESULTS_DIR.glob("baseline_results_*.json"))
    if not baseline_files:
        print("⚠️  Brak baseline results do porównania")
        return
    
    with open(baseline_files[-1], 'r') as f:
        baseline_data = json.load(f)
    
    baseline_metrics = baseline_data['best_model']['metrics']
    
    # Znajdź najlepszy nowy model
    best_new = max(new_results, key=lambda x: x['results']['f1'])
    new_metrics = best_new['results']
    
    # Porównanie
    print(f"\n{'Metryka':<15} {'Baseline':<12} {'Iteration_{iteration_num}':<15} {'Zmiana':<15}")
    print("-" * 60)
    
    metrics_to_compare = ['f1', 'precision', 'recall', 'roc_auc', 'pr_auc']
    
    improvements = []
    for metric in metrics_to_compare:
        baseline_val = baseline_metrics[metric]
        new_val = new_metrics[metric]
        change = new_val - baseline_val
        change_pct = (change / baseline_val * 100) if baseline_val > 0 else 0
        
        arrow = "📈" if change > 0 else "📉" if change < 0 else "➡️"
        
        print(f"{metric.upper():<15} {baseline_val:<12.4f} {new_val:<15.4f} {arrow} {change:+.4f} ({change_pct:+.1f}%)")
        
        if change > 0:
            improvements.append(metric)
    
    # Podsumowanie
    print("\n" + "="*80)
    print("💡 INTERPRETACJA (validation set jest IDENTYCZNY z baseline):")
    print("="*80)
    
    f1_improved = new_metrics['f1'] > baseline_metrics['f1']
    recall_improved = new_metrics['recall'] > baseline_metrics['recall']
    pr_auc_improved = new_metrics['pr_auc'] > baseline_metrics['pr_auc']
    
    if f1_improved and recall_improved:
        print("🎉 MODEL SIĘ ZNACZNIE POPRAWIŁ!")
        print(f"   ✅ F1 wzrósł: {baseline_metrics['f1']:.4f} → {new_metrics['f1']:.4f}")
        print(f"   ✅ Recall wzrósł: {baseline_metrics['recall']:.4f} → {new_metrics['recall']:.4f}")
        print(f"\n   🎯 ZALECENIE: Użyj tego modelu i uruchom iteration_{iteration_num + 1}")
    elif recall_improved:
        print("✅ MODEL SIĘ POPRAWIŁ!")
        print(f"   ✅ Recall wzrósł: {baseline_metrics['recall']:.4f} → {new_metrics['recall']:.4f}")
        print(f"   → Model znajduje WIĘCEJ map!")
        if not f1_improved:
            print(f"   ⚠️  F1 lekko spadł ale to trade-off - więcej recall za cenę precision")
        print(f"\n   🎯 ZALECENIE: To dobra poprawa - uruchom iteration_{iteration_num + 1}")
    elif pr_auc_improved:
        print("✅ MODEL SIĘ LEKKO POPRAWIŁ!")
        print(f"   ✅ PR-AUC wzrósł: {baseline_metrics['pr_auc']:.4f} → {new_metrics['pr_auc']:.4f}")
        print(f"   → Ogólna jakość predykcji lepsza dla imbalanced data")
        print(f"\n   🎯 ZALECENIE: Uruchom iteration_{iteration_num + 1} - może kolejna iteracja da większy boost")
    elif len(improvements) > 0:
        print(f"✅ CZĘŚCIOWA POPRAWA!")
        print(f"   Poprawione metryki: {', '.join(improvements)}")
        print(f"\n   🎯 ZALECENIE: Uruchom iteration_{iteration_num + 1}")
    else:
        print("⚠️  Model nie pokazał poprawy")
        print(f"\n   Możliwe przyczyny:")
        print(f"   1. Za mało nowych danych ({iteration_num * 600} próbek)")
        print(f"   2. Nowe próbki były zbyt podobne do baseline")
        print(f"   3. Model potrzebuje więcej iteracji (spróbuj iteration_{iteration_num + 1})")
        print(f"   4. Być może osiągnęliśmy plateau dla tego modelu")
        print(f"\n   🎯 ZALECENIE: Spróbuj jeszcze iteration_{iteration_num + 1} lub rozważ inne podejście")
    
    return best_new


def save_model(best_result, iteration_num, X_train, y_train):
    """Zapisz model"""
    print(f"\n💾 Zapisywanie modelu iteration_{iteration_num}...")
    
    name = best_result['name']
    use_smote = best_result['use_smote']
    model = best_result['model']
    scaler = best_result['scaler']
    
    # Retrain na całym train set
    X_train_processed = X_train.copy()
    if scaler:
        X_train_processed = scaler.fit_transform(X_train)
    
    if use_smote:
        X_train_processed, y_train_processed = apply_smote(X_train_processed, y_train)
    else:
        y_train_processed = y_train
    
    if 'XGBoost' in name:
        scale_pos_weight = (y_train_processed == 0).sum() / (y_train_processed == 1).sum()
        model.set_params(scale_pos_weight=scale_pos_weight)
    
    model.fit(X_train_processed, y_train_processed)
    
    # Zapisz
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_filename = MODELS_DIR / f"model_iteration_{iteration_num}_{timestamp}.pkl"
    
    model_package = {
        'model': model,
        'scaler': scaler,
        'name': name,
        'use_smote': use_smote,
        'results': best_result['results'],
        'timestamp': timestamp,
        'iteration': iteration_num,
        'optimal_threshold': best_result['results']['optimal_threshold']
    }
    
    joblib.dump(model_package, model_filename)
    print(f"   ✅ Model zapisany: {model_filename}")
    
    # Update latest link
    latest_link = MODELS_DIR / "model_latest.pkl"
    joblib.dump(model_package, latest_link)
    print(f"   ✅ Latest link: {latest_link}")
    
    return model_filename


def save_results(all_results, best_result, iteration_num, baseline_comparison=None):
    """Zapisz wyniki iteration do results/ (taka sama struktura jak baseline)"""
    print(f"\n📄 Zapisywanie wyników iteration_{iteration_num}...")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = RESULTS_DIR / f"iteration_{iteration_num}_results_{timestamp}.json"
    
    # Przygotuj dane - taka sama struktura jak baseline_results
    results_data = {
        'iteration': iteration_num,
        'timestamp': timestamp,
        'best_model': {
            'name': best_result['name'],
            'use_smote': best_result['use_smote'],
            'metrics': best_result['results']
        },
        'all_models': []
    }
    
    # Dodaj wszystkie modele w formacie jak w baseline
    for result in all_results:
        metrics = result['results']
        bootstrap = metrics.get('bootstrap', {})
        
        model_data = {
            'Model': result['name'] + (' + SMOTE' if result['use_smote'] else ''),
            'F1': metrics['f1'],
            'Precision': metrics['precision'],
            'Recall': metrics['recall'],
            'Accuracy': metrics.get('accuracy', metrics['f1']),
            'ROC-AUC': metrics['roc_auc'],
            'PR-AUC': metrics['pr_auc'],
            'Optimal F1': metrics.get('optimal_f1', metrics['f1']),
            'Bootstrap_CI': {
                'f1_mean': bootstrap.get('f1_mean'),
                'f1_std': bootstrap.get('f1_std'),
                'f1_ci_lower': bootstrap.get('f1_ci_lower'),
                'f1_ci_upper': bootstrap.get('f1_ci_upper')
            } if bootstrap else None
        }
        results_data['all_models'].append(model_data)
    
    # Dodaj porównanie z baseline jeśli dostępne
    if baseline_comparison:
        results_data['baseline_comparison'] = baseline_comparison
    
    # Zapisz
    with open(results_file, 'w') as f:
        json.dump(results_data, f, indent=2)
    
    print(f"   ✅ Wyniki zapisane: {results_file}")
    print(f"   📊 Zapisano {len(all_results)} modeli")
    
    return results_file

def main(iteration_num=1):
    """Główna funkcja retrain"""
    print("\n" + "="*80)
    print(f"🔄 RETRAIN MODELU - ITERATION {iteration_num}")
    print("="*80)
    
    # 1. Wczytaj baseline train
    baseline_df = load_baseline_train()
    
    # 2. Wczytaj iteration annotations
    iteration_df = load_iteration_annotations(iteration_num)
    
    # 3. Merge iteration z embeddingami
    iteration_df = merge_with_embeddings(iteration_df)
    
    # 4. Połącz datasets
    combined_df = combine_datasets(baseline_df, iteration_df)
    
    # 5. Przygotuj dane
    X, y, uids = prepare_train_data(combined_df)
    
    # 6. Train/val split (STAŁY validation set!)
    X_train, X_val, y_train, y_val, uid_train, uid_val = create_fixed_train_val_split(X, y, uids)
    
    # 7. Trenuj modele
    models = get_models()
    all_results = []
    
    for name, model_config in models.items():
        # Bez SMOTE
        result = train_and_evaluate_model(
            name, model_config, X_train, y_train, X_val, y_val, use_smote=False
        )
        all_results.append(result)
        
        # Z SMOTE (tylko jeśli dostępny i nie SVM - SVM jest zbyt wolny)
        if SMOTE_AVAILABLE and name != 'SVM_RBF':
            result_smote = train_and_evaluate_model(
                name, model_config, X_train, y_train, X_val, y_val, use_smote=True
            )
            all_results.append(result_smote)
    
    # 8. NOWA ANALIZA: Per-model comparison
    comparisons = compare_same_models_baseline_vs_iteration(all_results, iteration_num)
    
    # 9. NOWA ANALIZA: Tracking przez iteracje
    trends = track_all_models_over_iterations(iteration_num)
    
    # 10. Porównaj z baseline (stary sposób)
    best_result = compare_with_baseline(all_results, iteration_num)
    
    # 11. Zapisz wyniki do results/
    if best_result:
        # Przygotuj porównanie dla JSON
        baseline_files = sorted(RESULTS_DIR.glob("baseline_results_*.json"))
        baseline_comparison = None
        if baseline_files:
            with open(baseline_files[-1], 'r') as f:
                baseline_data = json.load(f)
            baseline_metrics = baseline_data['best_model']['metrics']
            
            baseline_comparison = {
                'baseline_f1': baseline_metrics['f1'],
                'baseline_recall': baseline_metrics['recall'],
                'baseline_precision': baseline_metrics['precision'],
                'iteration_f1': best_result['results']['f1'],
                'iteration_recall': best_result['results']['recall'],
                'iteration_precision': best_result['results']['precision'],
                'f1_change': best_result['results']['f1'] - baseline_metrics['f1'],
                'recall_change': best_result['results']['recall'] - baseline_metrics['recall'],
                'precision_change': best_result['results']['precision'] - baseline_metrics['precision']
            }
        
        results_file = save_results(all_results, best_result, iteration_num, baseline_comparison)
    
    # 12. Zapisz model
    if best_result:
        model_filename = save_model(best_result, iteration_num, X_train, y_train)
    
    print("\n" + "="*80)
    print(f"✅ RETRAIN ZAKOŃCZONY - ITERATION {iteration_num}")
    print("="*80)
    
    print(f"\n🎯 Następne kroki:")
    print(f"   1. Sprawdź czy model się poprawił (powyżej)")
    print(f"   2. Jeśli TAK: uruchom iteration_{iteration_num + 1}")
    print(f"      python active_learning.py --iteration {iteration_num + 1}")
    print(f"   3. Jeśli NIE: przeanalizuj dlaczego i rozważ inne podejście")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Retrain model po active learning iteration')
    parser.add_argument('--iteration', type=int, default=1, help='Numer iteracji do retrain (default: 1)')
    
    args = parser.parse_args()
    
    main(iteration_num=args.iteration)
