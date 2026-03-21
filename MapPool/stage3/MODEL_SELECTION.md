ą# Final Model Card - Statistical Map Classifier

## Model Details
- **Model type:** SVM with RBF kernel
- **Framework:** scikit-learn
- **Training data:** 3,801 annotated samples
  - Baseline: 3,201 samples (591 YES, 18.5%)
  - Iteration 1: 600 samples (284 YES, 47.3%)
- **Features:** CLIP ViT-L/14 image embeddings (768-dim)

## Performance Metrics
- **F1 Score:** 0.850 (default) / 0.885 (optimal threshold)
- **Precision:** 0.763
- **Recall:** 0.959
- **ROC-AUC:** 0.980
- **Optimal threshold:** 0.457

## Confusion Matrix (test set, 801 samples)
- True Positives: 142
- False Positives: 44
- True Negatives: 609
- False Negatives: 6

## Use Cases
- Primary: Filter 196k pool for statistical maps
- Secondary: Scale to 50M image corpus