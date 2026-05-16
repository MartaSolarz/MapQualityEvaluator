"""
06_bootstrap.py
================
"Robustness analyses" section, chapter 1 of the Stan_prac.pdf document:
Bootstrap 95% CI for disqualification weights and for Krippendorff's alpha.

Method: in each iteration we sample with replacement N = 25 respondents from the original sample
and compute weights (and alpha). After all iterations we determine the 95% CI as the 2.5th and 97.5th
percentile of the bootstrap distribution.

Number of iterations:
- 10,000 for weights (fast, pure arithmetic of means)
- 1,000 for alpha (slower, each iteration is a full Krippendorff alpha computation)

Seed: 42 (deterministic replication).

Run:
    python 06_bootstrap.py
"""
import numpy as np
import krippendorff
from data_loader import load_ratings, CRITERIA, CRIT_IDS


def bootstrap_weights(ratings, n_iter=10000, seed=42):
    """
    Bootstrap 95% CI for weights w_i = (mean - 1) / 4.

    Returns:
        dict {crit_id: {'w': float, 'se': float, 'ci_lo': float, 'ci_hi': float}}

    Note: we use the older API np.random.seed + np.random.choice to
    maintain consistency with the values in Stan_prac.pdf. The newer generator
    (np.random.default_rng) would yield numbers differing by +/-0.01 due to
    a different random number sequence.
    """
    np.random.seed(seed)
    N = ratings.shape[0]
    boot_means = np.zeros((n_iter, 10))
    for b in range(n_iter):
        idx = np.random.choice(N, N, replace=True)
        boot_means[b] = ratings[idx].mean(axis=0)
    boot_w = (boot_means - 1) / 4

    results = {}
    for i, crit in enumerate(CRITERIA):
        w_est = (ratings[:, i].mean() - 1) / 4
        results[crit['id']] = {
            'w':     float(w_est),
            'se':    float(boot_w[:, i].std()),
            'ci_lo': float(np.percentile(boot_w[:, i], 2.5)),
            'ci_hi': float(np.percentile(boot_w[:, i], 97.5)),
        }
    return results


def bootstrap_alpha(ratings, n_iter=1000, seed=42, level='ordinal'):
    """
    Bootstrap 95% CI for Krippendorff's alpha.

    Returns:
        dict {'value', 'mean_boot', 'sd_boot', 'ci_lo', 'ci_hi'}

    Note: we use the older API (np.random.seed + np.random.choice) to
    maintain consistency with the values in Stan_prac.pdf.
    """
    np.random.seed(seed)
    N = ratings.shape[0]
    alphas = []
    for b in range(n_iter):
        idx = np.random.choice(N, N, replace=True)
        rel_b = ratings[idx].T  # 10 rows x N columns
        try:
            a = krippendorff.alpha(reliability_data=rel_b, level_of_measurement=level)
            alphas.append(a)
        except Exception:
            pass
    alphas = np.array(alphas)
    alpha_orig = krippendorff.alpha(reliability_data=ratings.T,
                                     level_of_measurement=level)
    return {
        'value':     float(alpha_orig),
        'mean_boot': float(alphas.mean()),
        'sd_boot':   float(alphas.std()),
        'ci_lo':     float(np.percentile(alphas, 2.5)),
        'ci_hi':     float(np.percentile(alphas, 97.5)),
        'n_iter':    int(len(alphas)),
    }


def main():
    ratings, _ = load_ratings()

    print("="*70)
    print("Bootstrap 95% CI for disqualification weights w_i")
    print("(n_iter = 10,000, seed = 42)")
    print("="*70)

    results = bootstrap_weights(ratings, n_iter=10000, seed=42)

    header = f"{'ID':<4} {'w_i':>6} {'SE':>6} {'CI lower':>10} {'CI upper':>10} {'CI width':>14}"
    print(header)
    print('-' * len(header))
    for cid in CRIT_IDS:
        r = results[cid]
        width = r['ci_hi'] - r['ci_lo']
        print(f"{cid:<4} {r['w']:>6.3f} {r['se']:>6.3f} "
              f"{r['ci_lo']:>10.3f} {r['ci_hi']:>10.3f} {width:>14.3f}")

    widths = [results[c]['ci_hi'] - results[c]['ci_lo'] for c in CRIT_IDS]
    print(f"\nCI width range: {min(widths):.3f} - {max(widths):.3f}")

    print("\n" + "="*70)
    print("Bootstrap 95% CI for Krippendorff's alpha (ordinal)")
    print("(n_iter = 1000, seed = 42)")
    print("="*70)

    alpha_res = bootstrap_alpha(ratings, n_iter=1000, seed=42)
    print(f"\n  Original alpha:         {alpha_res['value']:.4f}")
    print(f"  Bootstrap mean:         {alpha_res['mean_boot']:.4f}")
    print(f"  Bootstrap SD:           {alpha_res['sd_boot']:.4f}")
    print(f"  95% CI:                 [{alpha_res['ci_lo']:.4f}, {alpha_res['ci_hi']:.4f}]")
    print(f"  Successful iterations:  {alpha_res['n_iter']}")


if __name__ == "__main__":
    main()
