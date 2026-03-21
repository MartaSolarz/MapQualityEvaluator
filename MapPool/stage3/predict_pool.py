# predict_pool.py
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from tqdm import tqdm

# Load model
model_package = joblib.load('models/model_final.pkl')
model = model_package['model']
scaler = model_package['scaler']
threshold = 0.6  # Conservative

# Load pool
df_pool = pd.read_parquet('data/pool.parquet')
X_pool = np.stack(df_pool['l14_img'].values)

print(f"Pool size: {len(df_pool):,}")

# Predict in batches
batch_size = 10000
predictions = []
probabilities = []

for i in tqdm(range(0, len(X_pool), batch_size)):
    batch = X_pool[i:i + batch_size]

    if scaler:
        batch = scaler.transform(batch)

    proba = model.predict_proba(batch)[:, 1]
    pred = (proba >= threshold).astype(int)

    predictions.extend(pred)
    probabilities.extend(proba)

# Add to dataframe
df_pool['pred_proba'] = probabilities
df_pool['predicted_label'] = predictions

# Save
df_pool.to_parquet('data/pool_with_predictions.parquet')

# Statistics
print(f"\nPrediction statistics:")
print(f"Total samples: {len(df_pool):,}")
print(f"Predicted YES: {sum(predictions):,} ({sum(predictions) / len(predictions):.2%})")
print(f"Predicted NO: {len(predictions) - sum(predictions):,}")
print(f"Mean probability: {np.mean(probabilities):.4f}")
print(f"Median probability: {np.median(probabilities):.4f}")

# Podziel predykcje na grupy:

# HIGH CONFIDENCE (proba >= 0.8)
df_high = df_pool[df_pool['pred_proba'] >= 0.8]
print(f"High confidence: {len(df_high):,}")
# Expected: ~1,000-2,000 samples
# Precision: ~95%+ (prawie wszystko to mapy)

# MEDIUM CONFIDENCE (0.6 <= proba < 0.8)
df_medium = df_pool[(df_pool['pred_proba'] >= 0.6) & (df_pool['pred_proba'] < 0.8)]
print(f"Medium confidence: {len(df_medium):,}")
# Expected: ~3,000-5,000 samples
# Precision: ~80-85%

# LOW POSITIVE (0.5 <= proba < 0.6)
df_low = df_pool[(df_pool['pred_proba'] >= 0.5) & (df_pool['pred_proba'] < 0.6)]
print(f"Low confidence: {len(df_low):,}")
# Expected: ~2,000-4,000 samples
# Precision: ~60-70% (dużo FP)

# NEGATIVE (proba < 0.5)
df_negative = df_pool[df_pool['pred_proba'] < 0.5]
print(f"Negative: {len(df_negative):,}")
# Expected: ~185,000-190,000
# Precision: <20% maps (mostly correct negatives)

# Sprawdź precision na próbkach:

# 1. Validate HIGH confidence (100 samples)
sample_high = df_high.sample(min(100, len(df_high)))
# Zaadnotuj ręcznie → oczekiwana precision: 95%+
sample_high.to_parquet('data/validation_samples/sample_high.parquet')

# 2. Validate MEDIUM confidence (100 samples)
sample_medium = df_medium.sample(min(100, len(df_medium)))
# Zaadnotuj ręcznie → oczekiwana precision: 80-85%
sample_medium.to_parquet('data/validation_samples/sample_medium.parquet')

# 3. Validate NEGATIVES (100 samples)
sample_negative = df_negative.sample(100)
# Zaadnotuj ręcznie → oczekiwany % map: <5%
sample_negative.to_parquet('data/validation_samples/sample_negative.parquet')

# Jeśli validation OK → proceed to 50M
# Jeśli precision < expected → rozważ retrain lub adjust threshold

