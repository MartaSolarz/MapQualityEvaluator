"""
09_friedman.py
===============
"Robustness analyses" section, chapter 5 of the Stan_prac.pdf document:
Friedman test — whether criteria differ significantly.

The Friedman test is a non-parametric equivalent of repeated measures ANOVA.
It tests the null hypothesis that all 10 criteria are rated equally severely
(the medians of the rating distributions are equal).

If H0 is rejected (p < 0.05), it means that criteria differ significantly —
which justifies the Stage 1 model architecture based on differentiated weights.

Run:
    python 09_friedman.py
"""
import numpy as np
from scipy import stats
from data_loader import load_ratings, CRITERIA, CRIT_IDS


def main():
    ratings, _ = load_ratings()

    print("="*70)
    print("Friedman test — whether criteria differ in rating severity")
    print("="*70)

    # scipy.stats.friedmanchisquare takes rating series — one series per criterion.
    # Each series is the ratings of all respondents for that criterion.
    series = [ratings[:, i] for i in range(10)]
    chi2, p = stats.friedmanchisquare(*series)

    print(f"\n  H0: all 10 criteria have equal median ratings")
    print(f"  H1: at least one criterion differs from the rest")
    print()
    print(f"  Chi-square statistic: {chi2:.4f}")
    print(f"  Degrees of freedom (df):  9  (k - 1, where k = 10 criteria)")
    if p < 1e-6:
        p_str = "< 0.000001"
    else:
        p_str = f"{p:.6f}"
    print(f"  p-value:                  {p_str}")

    print()
    if p < 0.05:
        print("  Conclusion: H0 rejected — criteria differ significantly in rating severity.")
        print("  Implication: the Stage 1 model architecture with individual weights w_i")
        print("               is empirically justified.")
    else:
        print("  Conclusion: no grounds to reject H0.")

    # Additional information: mean ranks per criterion
    print("\n--- Mean ranks per criterion (ascending) ---")
    # For each respondent rank the criteria (ranks 1..10 within a row)
    # Higher rank = higher rating.
    N = ratings.shape[0]
    ranks_per_resp = np.zeros_like(ratings)
    for j in range(N):
        # method='average' handles tied values correctly
        ranks_per_resp[j] = stats.rankdata(ratings[j], method='average')
    mean_ranks = ranks_per_resp.mean(axis=0)

    order = np.argsort(mean_ranks)
    print(f"\n{'ID':<4} {'Mean rank':>10}  Criterion")
    print('-' * 70)
    for i in order:
        print(f"{CRIT_IDS[i]:<4} {mean_ranks[i]:>10.3f}  {CRITERIA[i]['name_short_en']}")


if __name__ == "__main__":
    main()
