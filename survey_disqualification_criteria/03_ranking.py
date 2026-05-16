"""
03_ranking.py
==============
Section 3 of the Stan_prac.pdf document: Criterion ranking + category comparison.

Computes:
- Criterion ranking by descending mean rating
- Category comparison: mean of means within each of 4 categories
  (DATA, DATA ANALYSIS & TRANSFORM, VISUALIZATION, LAYOUT)

Run:
    python 03_ranking.py
"""
import numpy as np
from data_loader import load_ratings, CRITERIA, CRIT_IDS
from importlib import import_module


def main():
    ratings, _ = load_ratings()

    # Reuse computation from module 02
    mod02 = import_module('02_per_criterion')
    stats = mod02.compute_per_criterion(ratings)

    # ---------- RANKING ----------
    print("="*70)
    print("Criterion ranking (descending by mean rating)")
    print("="*70)
    ranking = sorted(stats, key=lambda x: -x['mean'])

    header = f"{'#':>3} {'ID':<4} {'Mean':>6} {'Category':<28} Criterion"
    print(header)
    print('-' * 100)
    for rank, s in enumerate(ranking, start=1):
        print(f"{rank:>3} {s['id']:<4} {s['mean']:>6.2f} {s['category']:<28} {s['name_en']}")

    # ---------- CATEGORIES ----------
    print("\n" + "="*70)
    print("Category comparison (mean of means)")
    print("="*70)

    cat_means = {}
    for s in stats:
        cat_means.setdefault(s['category'], []).append(s['mean'])

    # Sort categories descending by mean of means
    cat_sorted = sorted(
        cat_means.items(),
        key=lambda kv: -np.mean(kv[1])
    )

    header = f"{'#':>3} {'Category':<28} {'Criteria':<14} {'N':>3} {'Mean':>9}"
    print(header)
    print('-' * 65)
    cat_results = []
    for rank, (cat, means) in enumerate(cat_sorted, start=1):
        ids_in_cat = [c['id'] for c in CRITERIA if c['category'] == cat]
        avg = float(np.mean(means))
        cat_results.append({'rank': rank, 'category': cat, 'ids': ids_in_cat, 'mean': avg})
        print(f"{rank:>3} {cat:<28} {', '.join(ids_in_cat):<14} "
              f"{len(means):>3} {avg:>9.2f}")

    # Highest minus lowest
    high = cat_results[0]['mean']
    low = cat_results[-1]['mean']
    print(f"\nDifference between the highest and lowest rated category:")
    print(f"  {cat_results[0]['category']} - {cat_results[-1]['category']} = "
          f"{high:.2f} - {low:.2f} = {high - low:.2f}")


if __name__ == "__main__":
    main()
