"""
04_agreement.py
================
Section 4 of the Stan_prac.pdf document: Expert agreement analysis.

Computes:
- Krippendorff's alpha (ordinal) for all 10 criteria combined
- Coefficient of variation CV = SD/mean for each criterion
- Criterion ranking by agreement (CV ascending)

Requires the `krippendorff` package (pip install krippendorff).

Run:
    python 04_agreement.py
"""
import numpy as np
import krippendorff
from data_loader import load_ratings, CRITERIA, CRIT_IDS
from importlib import import_module


def compute_alpha(ratings, level='ordinal'):
    """
    Compute Krippendorff's alpha for the ratings matrix.

    Args:
        ratings: np.ndarray (N, 10) — rows: respondents, columns: criteria
        level: 'ordinal' | 'interval' | 'nominal' | 'ratio'

    Returns:
        float — alpha value

    Note: The `krippendorff` package expects reliability_data format, where
    rows are CODERS (or coding units) and columns are UNITS
    (things being rated). Convention: rows = criteria, cols = respondents.
    We transpose ratings (N x 10) -> (10 x N).
    """
    reliability = ratings.T  # 10 rows (criteria) x N columns (respondents)
    return krippendorff.alpha(reliability_data=reliability,
                               level_of_measurement=level)


def main():
    ratings, _ = load_ratings()

    # ----- KRIPPENDORFF'S ALPHA -----
    print("="*70)
    print("Krippendorff's alpha (expert agreement for all 10 criteria)")
    print("="*70)

    alpha_ord = compute_alpha(ratings, 'ordinal')
    alpha_int = compute_alpha(ratings, 'interval')
    alpha_nom = compute_alpha(ratings, 'nominal')

    print(f"\n  alpha (ordinal):  {alpha_ord:.4f}  <- used in the document")
    print(f"  alpha (interval): {alpha_int:.4f}")
    print(f"  alpha (nominal):  {alpha_nom:.4f}")

    print("\nInterpretation thresholds (Krippendorff, 2004):")
    print("  alpha >= 0.800 — conclusive inferences")
    print("  alpha >= 0.667 — tentative inferences acceptable")
    print("  alpha <  0.667 — low agreement")
    print(f"\n  Result alpha = {alpha_ord:.4f} falls below the 0.667 threshold.")

    # ----- COEFFICIENT OF VARIATION CV -----
    print("\n" + "="*70)
    print("Coefficient of variation CV per criterion (sorted ascending)")
    print("="*70)
    print("(Lower CV = greater expert agreement for a given criterion)")

    mod02 = import_module('02_per_criterion')
    stats = mod02.compute_per_criterion(ratings)

    sorted_by_cv = sorted(stats, key=lambda s: s['cv'])

    header = f"{'#':>3} {'ID':<4} {'Mean':>6} {'SD':>6} {'CV':>6}  Criterion"
    print(header)
    print('-' * 90)
    for rank, s in enumerate(sorted_by_cv, start=1):
        print(f"{rank:>3} {s['id']:<4} {s['mean']:>6.2f} {s['sd']:>6.2f} "
              f"{s['cv']:>6.2f}  {s['name_en']}")

    print(f"\nHighest agreement: {sorted_by_cv[0]['id']} "
          f"(CV = {sorted_by_cv[0]['cv']:.2f})")
    print(f"Lowest agreement: {sorted_by_cv[-1]['id']} "
          f"(CV = {sorted_by_cv[-1]['cv']:.2f})")


if __name__ == "__main__":
    main()
