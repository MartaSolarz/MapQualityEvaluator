"""
11_respondent_diag.py
======================
"Robustness analyses" section, chapter 7 of the Stan_prac.pdf document:
Respondent-level diagnostics — detecting systematic biases.

Examines for each respondent:
- Mean of all their 10 ratings (personal mean)
- SD of their ratings (within-person spread)
- Number of unique scale values used (1..5)
- Number of responses of 1 and 5 (extreme values)

Identifies respondents with low response diversity:
- Personal SD < 0.5  -> low-diversity ratings
- Number of unique values <= 2 -> using only 2 scale points

Run:
    python 11_respondent_diag.py
"""
import numpy as np
import pandas as pd
from data_loader import load_ratings, COL_COUNTRY


def main():
    ratings, df = load_ratings()
    countries = df.iloc[:, COL_COUNTRY].astype(str).str.strip().tolist()
    N = ratings.shape[0]

    print("="*90)
    print("Respondent-level diagnostics")
    print("="*90)

    diag = []
    for j in range(N):
        r = ratings[j]
        diag.append({
            'idx': j,
            'country': countries[j],
            'mean': r.mean(),
            'sd': r.std(ddof=1),
            'n_unique': len(np.unique(r)),
            'n_5': int((r == 5).sum()),
            'n_1': int((r == 1).sum()),
        })

    header = (f"{'#':>3} {'Country':<25} {'Mean':>6} {'SD':>6} "
              f"{'#unique':>8} {'n=5':>4} {'n=1':>4}")
    print(header)
    print('-' * len(header))
    for d in diag:
        flag = ""
        if d['sd'] < 0.5 or d['n_unique'] <= 2:
            flag = "  <- low diversity"
        print(f"{d['idx']:>3} {d['country'][:25]:<25} {d['mean']:>6.2f} {d['sd']:>6.2f} "
              f"{d['n_unique']:>8} {d['n_5']:>4} {d['n_1']:>4}{flag}")

    # Personal mean statistics
    pers_means = [d['mean'] for d in diag]
    print(f"\n--- Personal mean distribution ---")
    print(f"  N:        {len(pers_means)}")
    print(f"  Min:      {min(pers_means):.2f}")
    print(f"  Max:      {max(pers_means):.2f}")
    print(f"  Mean:     {np.mean(pers_means):.2f}")
    print(f"  SD:       {np.std(pers_means, ddof=1):.2f}")

    # Suspicious respondents
    suspicious = [d for d in diag if d['sd'] < 0.5 or d['n_unique'] <= 2]
    print(f"\n--- Respondents with low response diversity ---")
    print(f"  Criteria: personal SD < 0.5  OR  number of unique values <= 2")
    print(f"  Count: {len(suspicious)}")
    if suspicious:
        for s in suspicious:
            print(f"    #{s['idx']} ({s['country']}): mean = {s['mean']:.2f}, "
                  f"SD = {s['sd']:.2f}, unique = {s['n_unique']}, "
                  f"n_5 = {s['n_5']}, n_1 = {s['n_1']}")

    print("\nInterpretation:")
    print("  Acquiescence bias = systematic inflation of ratings (high personal mean)")
    print("  Extreme response bias = tendency toward extreme values (1, 5)")
    print("  Low diversity (e.g., only 4 and 5) may signal lack of reflection on the question.")
    print("\n  Conclusion: see text in Stan_prac.pdf document, chapter 7.")


if __name__ == "__main__":
    main()
