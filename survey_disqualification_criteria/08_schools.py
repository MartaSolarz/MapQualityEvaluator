"""
08_schools.py
==============
Sekcja "Analizy odporności", rozdz. 3-4 dokumentu Stan_prac.pdf:
Porównanie między szkołami kartograficznymi.

Klasyfikacja "first-listed" (zob. data_loader.classify_school):
    UK       — pierwszy kraj: UK lub United Kingdom         (n = 10)
    CZ       — pierwszy kraj: Czechia / Czech Republic      (n = 5)
    DE-speak — pierwszy kraj: Germany lub Switzerland       (n = 5)
    Other    — pozostałe (USA, Francja, Ghana, Nigeria)     (n = 5)

Oblicza:
- 3.2 Średnie ocen między szkołami (M ± SD per kryterium per szkoła)
- 4.1 Korelację rang Spearmana między każdą szkołą a rankingiem ogólnym
- 4.2 Korelację rang Spearmana między parami szkół
- 4.3 Współczynnik konkordancji Kendalla W

Uruchomienie:
    python 08_schools.py
"""
import numpy as np
from itertools import combinations
from scipy import stats
from data_loader import load_ratings, get_schools, CRITERIA, CRIT_IDS, CRIT_INDEX


def school_means_table(ratings, schools, school_list):
    """
    Oblicz średnią i SD każdej szkoły dla każdego kryterium.

    Returns:
        dict {school: {'n': int,
                       'means': {crit_id: float},
                       'sds':   {crit_id: float},
                       'overall_mean': float,
                       'overall_sd':   float}}
    """
    schools_arr = np.array(schools)
    result = {}
    for sch in school_list:
        mask = schools_arr == sch
        n_sch = int(mask.sum())
        result[sch] = {
            'n': n_sch,
            'means': {crit['id']: float(ratings[mask, i].mean())
                      for i, crit in enumerate(CRITERIA)},
            'sds':   {crit['id']: float(ratings[mask, i].std(ddof=1))
                      for i, crit in enumerate(CRITERIA)},
            'overall_mean': float(ratings[mask].mean()),
            'overall_sd':   float(ratings[mask].std(ddof=1)),
        }
    return result


def school_ranks(school_means):
    """
    Dla każdej szkoły utwórz ranking kryteriów (1 = najwyższa średnia).

    Returns:
        dict {school: {crit_id: rank}}
    """
    ranks = {}
    for sch, info in school_means.items():
        sorted_crit = sorted(CRIT_IDS, key=lambda c: -info['means'][c])
        ranks[sch] = {c: sorted_crit.index(c) + 1 for c in CRIT_IDS}
    return ranks


def overall_ranking(ratings):
    """Ranking ogólny: 1 = kryterium o najwyższej średniej."""
    overall = ratings.mean(axis=0)
    sorted_crit = [CRIT_IDS[i] for i in np.argsort(-overall)]
    return {c: sorted_crit.index(c) + 1 for c in CRIT_IDS}


def kendall_w(rank_matrix):
    """
    Współczynnik konkordancji Kendalla W dla macierzy rang.

    Args:
        rank_matrix: np.ndarray (k_judges, n_items)

    Returns:
        dict {'W', 'chi2', 'p', 'k', 'n_items'}
    """
    k, n = rank_matrix.shape
    R = rank_matrix.sum(axis=0)
    mean_R = R.mean()
    S = ((R - mean_R) ** 2).sum()
    W = 12 * S / (k**2 * (n**3 - n))
    chi2 = k * (n - 1) * W
    p = 1 - stats.chi2.cdf(chi2, df=n - 1)
    return {'W': float(W), 'chi2': float(chi2), 'p': float(p),
            'k': k, 'n_items': n}


def main():
    ratings, df = load_ratings()
    schools = get_schools(df)
    school_list = ['UK', 'CZ', 'DE-speak', 'Other']

    # ----- 3.2 ŚREDNIE OCEN MIĘDZY SZKOŁAMI -----
    print("="*90)
    print("3.2. Średnie ocen między szkołami (format: M (SD))")
    print("="*90)
    means_table = school_means_table(ratings, schools, school_list)

    header = f"{'ID':<4} {'Kryterium':<30}"
    for sch in school_list:
        n = means_table[sch]['n']
        header += f" {sch + ' (n=' + str(n) + ')':>14}"
    print(header)
    print('-' * len(header))

    for crit in CRITERIA:
        cid = crit['id']
        row = f"{cid:<4} {crit['name_short_pl'][:30]:<30}"
        for sch in school_list:
            m = means_table[sch]['means'][cid]
            sd = means_table[sch]['sds'][cid]
            row += f" {m:>5.2f} ({sd:>4.2f})"
        print(row)

    print('-' * len(header))
    row = f"{'—':<4} {'Średnia całościowa':<30}"
    for sch in school_list:
        m = means_table[sch]['overall_mean']
        sd = means_table[sch]['overall_sd']
        row += f" {m:>5.2f} ({sd:>4.2f})"
    print(row)

    # ----- 4.1 KORELACJA RANG Z RANKINGIEM OGÓLNYM -----
    print("\n" + "="*70)
    print("4.1. Korelacja rang Spearmana między szkołą a rankingiem ogólnym")
    print("="*70)
    ranks = school_ranks(means_table)
    overall = overall_ranking(ratings)
    overall_vec = [overall[c] for c in CRIT_IDS]

    print(f"{'Szkoła':<12} {'ρ (Spearman)':>14} {'p':>10}  Interpretacja")
    print('-' * 60)
    for sch in school_list:
        sch_vec = [ranks[sch][c] for c in CRIT_IDS]
        rho, p = stats.spearmanr(sch_vec, overall_vec)
        if p < 0.01 and rho > 0.8:
            interp = "Bardzo silna"
        elif p < 0.05 and rho > 0.6:
            interp = "Silna"
        elif p < 0.1:
            interp = "Umiarkowana"
        else:
            interp = "Słaba / brak"
        print(f"{sch:<12} {rho:>14.4f} {p:>10.4f}  {interp}")

    # ----- 4.2 KORELACJA RANG MIĘDZY PARAMI SZKÓŁ -----
    print("\n" + "="*70)
    print("4.2. Korelacja rang Spearmana między parami szkół")
    print("="*70)
    print(f"{'Para szkół':<24} {'ρ (Spearman)':>14} {'p':>10}  Interpretacja")
    print('-' * 70)
    for s1, s2 in combinations(school_list, 2):
        v1 = [ranks[s1][c] for c in CRIT_IDS]
        v2 = [ranks[s2][c] for c in CRIT_IDS]
        rho, p = stats.spearmanr(v1, v2)
        if p < 0.01 and rho > 0.8:
            interp = "Bardzo silna"
        elif p < 0.05 and rho > 0.6:
            interp = "Silna"
        elif p < 0.1:
            interp = "Umiarkowana"
        else:
            interp = "Słaba / brak"
        print(f"{s1 + ' vs ' + s2:<24} {rho:>14.4f} {p:>10.4f}  {interp}")

    # ----- 4.3 KENDALL'S W -----
    print("\n" + "="*70)
    print("4.3. Kendall's W — globalny współczynnik konkordancji")
    print("="*70)
    rank_matrix = np.array([[ranks[s][c] for c in CRIT_IDS] for s in school_list])
    kw = kendall_w(rank_matrix)
    print(f"  W       = {kw['W']:.4f}")
    print(f"  χ²      = {kw['chi2']:.4f}  (df = {kw['n_items'] - 1})")
    print(f"  p       = {kw['p']:.4f}")
    print(f"  k       = {kw['k']} (liczba szkół jako sędziów)")
    print(f"  n_items = {kw['n_items']} (liczba kryteriów)")
    print("\n  Interpretacja:")
    print("    W bliskie 1 — silny konsensus rangowy między szkołami")
    print("    W bliskie 0 — brak konsensusu")
    print("    p < 0.05    — można odrzucić H₀ o całkowitym braku konkordancji")


if __name__ == "__main__":
    main()