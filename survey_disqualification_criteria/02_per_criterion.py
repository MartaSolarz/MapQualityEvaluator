"""
02_per_criterion.py
====================
Section 2 of the Stan_prac.pdf document: Results per criterion.

For each of the 10 criteria computes:
- Mean of expert ratings
- Median
- Standard deviation (SD, sample n-1)
- Min, Max
- Response distribution (n=1, n=2, ..., n=5)
- Disqualification weight w_i = (mean - 1) / 4
- Coefficient of variation CV = SD / mean

Prints a summary table consistent with the table in section 2 of the document and
the sum of weights Sigma w_i needed for normalization in the Stage 1 model.

Run:
    python 02_per_criterion.py
"""
import numpy as np
from data_loader import load_ratings, CRITERIA, CRIT_IDS


def compute_per_criterion(ratings):
    """
    Compute the full set of statistics for each criterion.

    Args:
        ratings: np.ndarray (N, 10) — ratings matrix

    Returns:
        list of dict — one dict per criterion with fields:
            id, mean, median, sd, min, max, dist (distribution 1..5),
            w (weight), cv (coefficient of variation), n
    """
    results = []
    for i, crit in enumerate(CRITERIA):
        col = ratings[:, i]
        col = col[~np.isnan(col)]  # remove NaN if any
        mean = col.mean()
        median = float(np.median(col))
        sd = col.std(ddof=1)
        mn = int(col.min())
        mx = int(col.max())
        dist = {k: int((col == k).sum()) for k in range(1, 6)}
        w = (mean - 1) / 4
        cv = sd / mean if mean else None
        results.append({
            'id': crit['id'],
            'name_en': crit['name_en'],
            'category': crit['category'],
            'n': len(col),
            'mean': mean,
            'median': median,
            'sd': sd,
            'min': mn,
            'max': mx,
            'dist': dist,
            'w': w,
            'cv': cv,
        })
    return results


def main():
    ratings, _ = load_ratings()
    stats = compute_per_criterion(ratings)

    # Table header consistent with the table in section 2 of the document
    header = (f"{'ID':<4} {'Mean':>6} {'Med':>5} {'SD':>5} {'Min':>4} {'Max':>4} "
              f"{'n=1':>4} {'n=2':>4} {'n=3':>4} {'n=4':>4} {'n=5':>4} "
              f"{'w_i':>6} {'CV':>6}")
    print(header)
    print('-' * len(header))

    sum_w = 0.0
    for s in stats:
        print(f"{s['id']:<4} {s['mean']:>6.2f} {s['median']:>5.2f} {s['sd']:>5.2f} "
              f"{s['min']:>4} {s['max']:>4} "
              f"{s['dist'][1]:>4} {s['dist'][2]:>4} {s['dist'][3]:>4} "
              f"{s['dist'][4]:>4} {s['dist'][5]:>4} "
              f"{s['w']:>6.2f} {s['cv']:>6.2f}")
        sum_w += s['w']

    print('-' * len(header))
    print(f"\nSum of all criterion weights: Sigma w_i = {sum_w:.2f}")
    print("(This value serves as the denominator in the penalty normalization P_norm = P / Sigma w_i "
          "in the Stage 1 model.)")

    print("\nCriterion names:")
    for s in stats:
        print(f"  {s['id']}: {s['name_en']}  [{s['category']}]")


if __name__ == "__main__":
    main()
