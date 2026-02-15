"""
Baseline Model dla Active Learning - Stage 3 (Nowe podejście)

Problem: Niezbalansowane klasy (YES: 18.46%, NO: 81.54%, imbalance ratio 1:4.4)
Rozwiązanie: Class weighting, SMOTE, odpowiednie metryki, threshold tuning

Modele do przetestowania:
1. Logistic Regression (szybki, interpretowalny)
2. Random Forest (odporny na imbalance)
3. XGBoost (najczęściej najlepszy performance)
4. SVM z RBF kernel (dobry dla high-dimensional embeddings)

Dane:
- train.parquet: 3,201 próbek (80% annotacji)
- test.parquet: 801 próbek (20% annotacji) - STAŁY przez wszystkie iteracje AL

Metryki: Precision, Recall, F1, ROC-AUC, PR-AUC (nie accuracy!)
"""

import pandas as pd
import numpy as np
import json
import joblib
from pathlib import Path
from datetime import datetime
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, confusion_matrix, 
    roc_auc_score, average_precision_score,
    precision_recall_curve, roc_curve,
    f1_score, precision_score, recall_score,
    accuracy_score
)
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline
import warnings
warnings.filterwarnings('ignore')

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("⚠️  XGBoost nie jest zainstalowany. Zainstaluj: pip install xgboost")

# Ścieżki
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"

TRAIN_FILE = DATA_DIR / "train.parquet"
TEST_FILE = DATA_DIR / "test.parquet"


def setup_directories():
    """Utwórz katalogi na modele i wyniki"""
    MODELS_DIR.mkdir(exist_ok=True)
    RESULTS_DIR.mkdir(exist_ok=True)


def load_train_data():
    """Wczytaj dane treningowe (train.parquet)"""
    print("📂 Wczytywanie danych treningowych (train.parquet)...")
    df = pd.read_parquet(TRAIN_FILE)
    
    print(f"   Próbek: {len(df)}")
    print(f"   YES (1): {(df['label'] == 1).sum()} ({(df['label'] == 1).mean():.2%})")
    print(f"   NO (0): {(df['label'] == 0).sum()} ({(df['label'] == 0).mean():.2%})")
    print(f"   Imbalance ratio: {(df['label'] == 0).sum() / (df['label'] == 1).sum():.1f}:1")
    
    # Przygotuj X i y
    X = np.stack(df['l14_img'].values)
    y = df['label'].values
    uids = df['uid'].values
    
    print(f"   X shape: {X.shape}")
    print(f"   y shape: {y.shape}")
    
    return X, y, uids, df


def load_test_data():
    """Wczytaj dane testowe (test.parquet) - STAŁY zbiór testowy"""
    print("\n🧪 Wczytywanie danych testowych (test.parquet - STAŁY)...")
    df = pd.read_parquet(TEST_FILE)
    
    print(f"   Próbek: {len(df)}")
    print(f"   YES (1): {(df['label'] == 1).sum()} ({(df['label'] == 1).mean():.2%})")
    print(f"   NO (0): {(df['label'] == 0).sum()} ({(df['label'] == 0).mean():.2%})")
    
    # Przygotuj X i y
    X = np.stack(df['l14_img'].values)
    y = df['label'].values
    uids = df['uid'].values
    
    print(f"   X shape: {X.shape}")
    print(f"   y shape: {y.shape}")
    
    return X, y, uids, df




def get_models():
    """
    Zdefiniuj modele do przetestowania
    
    Każdy model ma class weighting lub scale_pos_weight
    dla obsługi niezbalansowanych danych
    """
    models = {}
    
    # 1. Logistic Regression - szybki baseline
    models['LogisticRegression'] = {
        'model': LogisticRegression(
            class_weight='balanced',  # Automatyczne ważenie klas
            max_iter=1000,
            random_state=42,
            n_jobs=-1
        ),
        'scale': True,  # Skalowanie dla LR
        'description': 'Logistic Regression z class_weight=balanced'
    }
    
    # 2. Logistic Regression z L1 (sparse)
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
    
    # 3. Random Forest - odporny na imbalance
    models['RandomForest'] = {
        'model': RandomForestClassifier(
            n_estimators=200,
            class_weight='balanced',  # Ważenie klas
            max_depth=20,
            min_samples_split=10,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        ),
        'scale': False,  # RF nie potrzebuje skalowania
        'description': 'Random Forest z class_weight=balanced'
    }
    
    # 4. Random Forest z balanced_subsample
    models['RandomForest_Subsample'] = {
        'model': RandomForestClassifier(
            n_estimators=200,
            class_weight='balanced_subsample',  # Balansowanie w każdym drzewie
            max_depth=20,
            min_samples_split=10,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        ),
        'scale': False,
        'description': 'Random Forest z balanced_subsample'
    }
    
    # 5. XGBoost - często najlepszy
    if XGBOOST_AVAILABLE:
        # Oblicz scale_pos_weight (ratio negative/positive)
        # Będzie uzupełnione podczas treningu
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
    
    # 6. SVM z RBF - dobry dla embeddings
    models['SVM_RBF'] = {
        'model': SVC(
            kernel='rbf',
            class_weight='balanced',
            probability=True,  # Dla predict_proba
            random_state=42
        ),
        'scale': True,  # SVM wymaga skalowania
        'description': 'SVM z RBF kernel'
    }
    
    return models


def apply_smote(X_train, y_train, sampling_strategy=0.3):
    """
    Zastosuj SMOTE oversampling
    
    sampling_strategy=0.3 oznacza: po SMOTE będziemy mieć 30% próbek pozytywnych
    (nie 50% bo to może prowadzić do overfittingu)
    """
    print(f"\n🔄 Stosowanie SMOTE (sampling_strategy={sampling_strategy})...")
    print(f"   Przed SMOTE:")
    print(f"      Próbek: {len(y_train)}")
    print(f"      YES: {(y_train == 1).sum()}")
    print(f"      NO: {(y_train == 0).sum()}")
    
    smote = SMOTE(
        sampling_strategy=sampling_strategy,
        random_state=42,
        k_neighbors=3  # Mniej sąsiadów bo mamy mało próbek pozytywnych
    )
    
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
    
    print(f"   Po SMOTE:")
    print(f"      Próbek: {len(y_resampled)}")
    print(f"      YES: {(y_resampled == 1).sum()} ({(y_resampled == 1).mean():.2%})")
    print(f"      NO: {(y_resampled == 0).sum()} ({(y_resampled == 0).mean():.2%})")
    
    return X_resampled, y_resampled


def train_and_evaluate_model(name, model_config, X_train, y_train, X_test, y_test, use_smote=False):
    """
    Trenuj na train set i ewaluuj na test set
    """
    print(f"\n{'='*60}")
    print(f"🤖 Model: {name}")
    print(f"   {model_config['description']}")
    if use_smote:
        print(f"   + SMOTE oversampling")
    print(f"{'='*60}")
    
    # Skalowanie jeśli potrzebne
    scaler = None
    X_train_processed = X_train.copy()
    X_test_processed = X_test.copy()
    
    if model_config['scale']:
        print("   Skalowanie danych...")
        scaler = StandardScaler()
        X_train_processed = scaler.fit_transform(X_train)
        X_test_processed = scaler.transform(X_test)
    
    # SMOTE jeśli enabled
    if use_smote:
        X_train_processed, y_train_processed = apply_smote(X_train_processed, y_train, sampling_strategy=0.3)
    else:
        y_train_processed = y_train
    
    # XGBoost - ustaw scale_pos_weight
    model = model_config['model']
    if 'XGBoost' in name:
        scale_pos_weight = (y_train_processed == 0).sum() / (y_train_processed == 1).sum()
        model.set_params(scale_pos_weight=scale_pos_weight)
        print(f"   XGBoost scale_pos_weight: {scale_pos_weight:.2f}")
    
    # Trenowanie
    print("   Trenowanie...")
    model.fit(X_train_processed, y_train_processed)
    
    # Predykcje na test set
    y_test_pred_proba = model.predict_proba(X_test_processed)[:, 1]
    y_test_pred = model.predict(X_test_processed)
    
    # Metryki
    results = calculate_metrics(y_test, y_test_pred, y_test_pred_proba)
    
    # Znajdź optymalny threshold
    optimal_threshold, optimal_f1 = find_optimal_threshold(y_test, y_test_pred_proba)
    results['optimal_threshold'] = optimal_threshold
    results['optimal_f1'] = optimal_f1
    
    # Predykcje z optymalnym threshold
    y_test_pred_optimal = (y_test_pred_proba >= optimal_threshold).astype(int)
    results['metrics_optimal_threshold'] = calculate_metrics(y_test, y_test_pred_optimal, y_test_pred_proba)
    
    print_results(results, optimal_threshold)
    
    return {
        'model': model,
        'scaler': scaler,
        'results': results,
        'name': name,
        'use_smote': use_smote
    }


def calculate_metrics(y_true, y_pred, y_pred_proba):
    """
    Oblicz wszystkie ważne metryki dla imbalanced data
    """
    # Podstawowe metryki
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    # ROC-AUC i PR-AUC
    roc_auc = roc_auc_score(y_true, y_pred_proba) if len(np.unique(y_true)) > 1 else 0.0
    pr_auc = average_precision_score(y_true, y_pred_proba) if len(np.unique(y_true)) > 1 else 0.0
    
    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    # Specificity (True Negative Rate)
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
    """
    Znajdź optymalny threshold maksymalizujący F1-score
    
    Dla imbalanced data threshold 0.5 często nie jest optymalny
    """
    precision_vals, recall_vals, thresholds = precision_recall_curve(y_true, y_pred_proba)
    
    # Oblicz F1 dla każdego threshold
    f1_scores = 2 * (precision_vals * recall_vals) / (precision_vals + recall_vals + 1e-10)
    
    # Znajdź maximum
    optimal_idx = np.argmax(f1_scores)
    optimal_threshold = thresholds[optimal_idx] if optimal_idx < len(thresholds) else 0.5
    optimal_f1 = f1_scores[optimal_idx]
    
    return optimal_threshold, optimal_f1


def print_results(results, optimal_threshold=None):
    """Wyświetl wyniki w czytelnej formie"""
    print(f"\n📊 Wyniki (threshold=0.5):")
    print(f"   Accuracy:  {results['accuracy']:.4f} ⚠️  (mylące dla imbalanced!)")
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
        print(f"   F1-score z optymalnym threshold: {results['optimal_f1']:.4f}")


def compare_models(all_results):
    """Porównaj wszystkie modele i wybierz najlepszy"""
    print("\n" + "="*80)
    print("📈 PORÓWNANIE WSZYSTKICH MODELI")
    print("="*80)
    
    # Tabela wyników
    comparison = []
    for result in all_results:
        metrics = result['results']
        comparison.append({
            'Model': result['name'] + (' + SMOTE' if result['use_smote'] else ''),
            'F1': metrics['f1'],
            'Precision': metrics['precision'],
            'Recall': metrics['recall'],
            'Accuracy': metrics['accuracy'],
            'ROC-AUC': metrics['roc_auc'],
            'PR-AUC': metrics['pr_auc'],
            'Optimal F1': metrics['optimal_f1']
        })
    
    df_comparison = pd.DataFrame(comparison)
    df_comparison = df_comparison.sort_values('F1', ascending=False)
    
    print("\nRanking według F1-score (threshold=0.5):")
    print(df_comparison.to_string(index=False))
    
    # Najlepszy model
    best_idx = df_comparison['F1'].idxmax()
    best_model_name = df_comparison.loc[best_idx, 'Model']
    
    print(f"\n🏆 NAJLEPSZY MODEL: {best_model_name}")
    print(f"   F1-score: {df_comparison.loc[best_idx, 'F1']:.4f}")
    print(f"   Precision: {df_comparison.loc[best_idx, 'Precision']:.4f}")
    print(f"   Recall: {df_comparison.loc[best_idx, 'Recall']:.4f}")
    print(f"   PR-AUC: {df_comparison.loc[best_idx, 'PR-AUC']:.4f}")
    
    # Znajdź odpowiedni model w all_results
    best_result = None
    for result in all_results:
        model_name = result['name'] + (' + SMOTE' if result['use_smote'] else '')
        if model_name == best_model_name:
            best_result = result
            break
    
    return best_result, df_comparison


def save_model(best_result):
    """
    Zapisz najlepszy model (już wytrenowany na całym train set)
    """
    print(f"\n💾 Zapisywanie najlepszego modelu...")
    
    name = best_result['name']
    use_smote = best_result['use_smote']
    model = best_result['model']
    scaler = best_result['scaler']
    
    # Model jest już wytrenowany na całym train set
    print(f"   Model już wytrenowany na całym train set")
    
    # Zapisz
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_filename = MODELS_DIR / f"baseline_model_{name}_{timestamp}.pkl"
    
    model_package = {
        'model': model,
        'scaler': scaler,
        'name': name,
        'use_smote': use_smote,
        'results': best_result['results'],
        'timestamp': timestamp,
        'optimal_threshold': best_result['results']['optimal_threshold']
    }
    
    joblib.dump(model_package, model_filename)
    print(f"   ✅ Model zapisany: {model_filename}")
    
    # Zapisz też "latest" link
    latest_link = MODELS_DIR / "baseline_model_latest.pkl"
    joblib.dump(model_package, latest_link)
    print(f"   ✅ Latest link: {latest_link}")
    
    return model_filename


def save_results(df_comparison, best_result):
    """Zapisz wyniki do JSON"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = RESULTS_DIR / f"baseline_results_{timestamp}.json"
    
    results = {
        'timestamp': timestamp,
        'best_model': {
            'name': best_result['name'],
            'use_smote': best_result['use_smote'],
            'metrics': best_result['results']
        },
        'all_models': df_comparison.to_dict(orient='records')
    }
    
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n📄 Wyniki zapisane: {results_file}")


def main():
    """Główna funkcja trenowania baseline model"""
    print("\n" + "="*80)
    print("🚀 BASELINE MODEL - TRENOWANIE (NOWE PODEJŚCIE)")
    print("="*80)
    
    setup_directories()
    
    # 1. Wczytaj dane train i test
    X_train, y_train, uid_train, df_train = load_train_data()
    X_test, y_test, uid_test, df_test = load_test_data()
    
    print("\n📊 Podsumowanie danych:")
    print(f"   Train: {len(y_train)} próbek")
    print(f"   Test:  {len(y_test)} próbek (STAŁY przez wszystkie iteracje AL)")
    
    # 2. Przygotuj modele
    models = get_models()
    print(f"\n🔧 Modeli do przetestowania: {len(models)}")
    
    # 3. Trenuj wszystkie modele (z i bez SMOTE)
    all_results = []
    
    for name, model_config in models.items():
        # Bez SMOTE
        result = train_and_evaluate_model(
            name, model_config, X_train, y_train, X_test, y_test, use_smote=False
        )
        all_results.append(result)
        
        # Z SMOTE (tylko dla wybranych modeli - nie dla bardzo dużych)
        if name not in ['SVM_RBF']:  # SVM zbyt wolny z SMOTE
            result_smote = train_and_evaluate_model(
                name, model_config, X_train, y_train, X_test, y_test, use_smote=True
            )
            all_results.append(result_smote)
    
    # 4. Porównaj modele
    best_result, df_comparison = compare_models(all_results)
    
    # 5. Zapisz najlepszy model
    model_filename = save_model(best_result)
    
    # 6. Zapisz wyniki
    save_results(df_comparison, best_result)
    
    print("\n" + "="*80)
    print("✅ BASELINE MODEL - GOTOWE!")
    print("="*80)
    print(f"\n📦 Model zapisany: {model_filename}")
    print(f"🧪 Test set: 801 próbek (STAŁY przez wszystkie iteracje)")
    print(f"🎯 Następny krok: active_learning.py")


if __name__ == "__main__":
    main()
