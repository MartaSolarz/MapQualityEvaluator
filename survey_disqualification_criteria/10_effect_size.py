"""
10_effect_size.py
==================
"Robustness analyses" section, chapter 6 of the Stan_prac.pdf document:
Effect size (Cohen's d) and tests against the scale midpoint (= 3).

For each criterion computes:
- Cohen's d = (mean - 3) / SD
  One-sample version of d (effect relative to test value = Likert scale midpoint).
  d > 0 means expert ratings are shifted above the scale midpoint.
  |d| > 0.5 — moderate, |d| > 0.8 — large, |d| > 1.2 — very large.
- One-sample t-test (parametric)
- Wilcoxon signed-rank test for the difference rating - 3 (non-parametric, robust to
  non-normality of distribution)

Run:
    python 10_effect_size.py
"""
import numpy as np
from scipy import stats
from data_loader import load_ratings, CRITERIA, CRIT_IDS


def cohen_d_one_sample(values, mu0=3.0):
    """Cohen's d for one sample against test value mu0."""
    mean = values.mean()
    sd = values.std(ddof=1)
    return (mean - mu0) / sd


def effect_size_label(d):
    """Cohen's convention for effect size."""
    ad = abs(d)
    if ad >= 1.2: return "Very large"
    if ad >= 0.8: return "Large"
    if ad >= 0.5: return "Moderate"
    if ad >= 0.2: return "Small"
    return "Negligible"


def main():
    ratings, _ = load_ratings()
    MIDPOINT = 3.0

    print("="*100)
    print(f"Effect size (Cohen's d) and significance tests against scale midpoint = {MIDPOINT}")
    print("="*100)

    header = (f"{'ID':<4} {'Mean':>6} {'SD':>5} {'d':>7} "
              f"{'t':>7} {'p(t)':>9} {'p(W)':>9}  Effect size")
    print(header)
    print('-' * len(header))

    results = []
    for i, crit in enumerate(CRITERIA):
        col = ratings[:, i]
        mean = col.mean()
        sd = col.std(ddof=1)
        d = cohen_d_one_sample(col, MIDPOINT)
        t_stat, t_p = stats.ttest_1samp(col, MIDPOINT)
        try:
            w_stat, w_p = stats.wilcoxon(col - MIDPOINT)
        except Exception:
            w_stat, w_p = (float('nan'), float('nan'))

        results.append({
            'id': crit['id'], 'mean': float(mean), 'sd': float(sd),
            'd': float(d), 't': float(t_stat),
            'p_t': float(t_p), 'p_wilcoxon': float(w_p),
        })
        p_t_str = "<0.0001" if t_p < 0.0001 else f"{t_p:.4f}"
        p_w_str = "<0.0001" if w_p < 0.0001 else f"{w_p:.4f}"
        marker = "" if t_p < 0.05 else " *"  # not significant
        print(f"{crit['id']:<4} {mean:>6.2f} {sd:>5.2f} {d:>7.3f} "
              f"{t_stat:>7.3f} {p_t_str:>9} {p_w_str:>9}  {effect_size_label(d)}{marker}")

    print('-' * len(header))
    print("* = deviation from scale midpoint is NOT statistically significant (p >= 0.05)")

    print("\nColumn explanations:")
    print("  Mean — mean of ratings")
    print("  SD   — standard deviation (sample)")
    print("  d    — Cohen's d (one-sample) = (mean - 3) / SD")
    print("  t    — one-sample t-test statistic")
    print("  p(t) — p-value from t-test (parametric)")
    print("  p(W) — p-value from Wilcoxon test (non-parametric, on difference rating - 3)")

    print("\nCohen's thresholds for effect size (|d|):")
    print("  < 0.2 — negligible")
    print("  0.2 - 0.5 — small")
    print("  0.5 - 0.8 — moderate")
    print("  0.8 - 1.2 — large")
    print("  > 1.2 — very large")


if __name__ == "__main__":
    main()
