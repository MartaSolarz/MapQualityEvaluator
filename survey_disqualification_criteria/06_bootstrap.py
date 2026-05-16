"""
06_bootstrap.py
================
Sekcja "Analizy odporności", rozdz. 1 dokumentu Stan_prac.pdf:
Bootstrap 95% CI dla wag dyskwalifikujących oraz dla Krippendorff's alpha.

Metoda: w każdej iteracji losujemy ze zwracaniem N = 25 respondentów z oryginalnej próby
i obliczamy wagi (oraz α). Po wszystkich iteracjach wyznaczamy 95% CI jako 2.5. i 97.5.
percentyl rozkładu bootstrapowego.

Liczby iteracji:
- 10 000 dla wag (szybkie, czysta arytmetyka średnich)
- 1 000 dla α (wolniejsze, każda iteracja to pełne wyliczenie Krippendorff α)

Seed: 42 (deterministyczna replikacja).

Uruchomienie:
    python 06_bootstrap.py
"""
import numpy as np
import krippendorff
from data_loader import load_ratings, CRITERIA, CRIT_IDS


def bootstrap_weights(ratings, n_iter=10000, seed=42):
    """
    Bootstrap 95% CI dla wag w_i = (mean - 1) / 4.

    Returns:
        dict {crit_id: {'w': float, 'se': float, 'ci_lo': float, 'ci_hi': float}}

    Uwaga: używamy starszego API np.random.seed + np.random.choice dla
    zachowania zgodności z wartościami w Stan_prac.pdf. Nowszy generator
    (np.random.default_rng) dawałby liczby różniące się o ±0.01 z powodu
    innej sekwencji liczb losowych.
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
    Bootstrap 95% CI dla Krippendorff's alpha.

    Returns:
        dict {'value', 'mean_boot', 'sd_boot', 'ci_lo', 'ci_hi'}

    Uwaga: starsze API (np.random.seed + np.random.choice) używamy dla
    zachowania zgodności z wartościami w Stan_prac.pdf.
    """
    np.random.seed(seed)
    N = ratings.shape[0]
    alphas = []
    for b in range(n_iter):
        idx = np.random.choice(N, N, replace=True)
        rel_b = ratings[idx].T  # 10 wierszy × N kolumn
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
    print("Bootstrap 95% CI dla wag dyskwalifikujących w_i")
    print("(n_iter = 10 000, seed = 42)")
    print("="*70)

    results = bootstrap_weights(ratings, n_iter=10000, seed=42)

    header = f"{'ID':<4} {'w_i':>6} {'SE':>6} {'CI dolne':>10} {'CI górne':>10} {'Szerokość CI':>14}"
    print(header)
    print('-' * len(header))
    for cid in CRIT_IDS:
        r = results[cid]
        width = r['ci_hi'] - r['ci_lo']
        print(f"{cid:<4} {r['w']:>6.3f} {r['se']:>6.3f} "
              f"{r['ci_lo']:>10.3f} {r['ci_hi']:>10.3f} {width:>14.3f}")

    widths = [results[c]['ci_hi'] - results[c]['ci_lo'] for c in CRIT_IDS]
    print(f"\nZakres szerokości CI: {min(widths):.3f} – {max(widths):.3f}")

    print("\n" + "="*70)
    print("Bootstrap 95% CI dla Krippendorff's alpha (ordinal)")
    print("(n_iter = 1000, seed = 42)")
    print("="*70)

    alpha_res = bootstrap_alpha(ratings, n_iter=1000, seed=42)
    print(f"\n  Oryginalna α:           {alpha_res['value']:.4f}")
    print(f"  Średnia bootstrapowa:   {alpha_res['mean_boot']:.4f}")
    print(f"  SD bootstrap:           {alpha_res['sd_boot']:.4f}")
    print(f"  95% CI:                 [{alpha_res['ci_lo']:.4f}, {alpha_res['ci_hi']:.4f}]")
    print(f"  Liczba udanych iteracji: {alpha_res['n_iter']}")


if __name__ == "__main__":
    main()