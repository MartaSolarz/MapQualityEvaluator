"""
08_schools.py
==============
"Robustness analyses" section, chapters 3-4 of the Stan_prac.pdf document:
Comparison between cartographic schools.

Classification "first-listed" (see data_loader.classify_school):
    UK       — first country: UK or United Kingdom         (n = 10)
    CZ       — first country: Czechia / Czech Republic      (n = 5)
    DE-speak — first country: Germany or Switzerland       (n = 5)
    Other    — remaining (USA, France, Ghana, Nigeria)     (n = 5)

Computes:
- 3.2 Mean ratings between schools (M +/- SD per criterion per school)
- 4.1 Spearman rank correlation between each school and the overall ranking
- 4.2 Spearman rank correlation between pairs of schools
- 4.3 Kendall's coefficient of concordance W

Run:
    python 08_schools.py
"""
import numpy as np
from itertools import combinations
from scipy import stats
from data_loader import load_ratings, get_schools, CRITERIA, CRIT_IDS, CRIT_INDEX


def school_means_table(ratings, schools, school_list):
    """
    Compute mean and SD for each school for each criterion.

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
    For each school create a criterion ranking (1 = highest mean).

    Returns:
        dict {school: {crit_id: rank}}
    """
    ranks = {}
    for sch, info in school_means.items():
        sorted_crit = sorted(CRIT_IDS, key=lambda c: -info['means'][c])
        ranks[sch] = {c: sorted_crit.index(c) + 1 for c in CRIT_IDS}
    return ranks


def overall_ranking(ratings):
    """Overall ranking: 1 = criterion with the highest mean."""
    overall = ratings.mean(axis=0)
    sorted_crit = [CRIT_IDS[i] for i in np.argsort(-overall)]
    return {c: sorted_crit.index(c) + 1 for c in CRIT_IDS}


def kendall_w(rank_matrix):
    """
    Kendall's coefficient of concordance W for a rank matrix.

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

    # ----- 3.2 MEAN RATINGS BETWEEN SCHOOLS -----
    print("="*90)
    print("3.2. Mean ratings between schools (format: M (SD))")
    print("="*90)
    means_table = school_means_table(ratings, schools, school_list)

    header = f"{'ID':<4} {'Criterion':<30}"
    for sch in school_list:
        n = means_table[sch]['n']
        header += f" {sch + ' (n=' + str(n) + ')':>14}"
    print(header)
    print('-' * len(header))

    for crit in CRITERIA:
        cid = crit['id']
        row = f"{cid:<4} {crit['name_short_en'][:30]:<30}"
        for sch in school_list:
            m = means_table[sch]['means'][cid]
            sd = means_table[sch]['sds'][cid]
            row += f" {m:>5.2f} ({sd:>4.2f})"
        print(row)

    print('-' * len(header))
    row = f"{'--':<4} {'Overall mean':<30}"
    for sch in school_list:
        m = means_table[sch]['overall_mean']
        sd = means_table[sch]['overall_sd']
        row += f" {m:>5.2f} ({sd:>4.2f})"
    print(row)

    # ----- 4.1 RANK CORRELATION WITH OVERALL RANKING -----
    print("\n" + "="*70)
    print("4.1. Spearman rank correlation between school and overall ranking")
    print("="*70)
    ranks = school_ranks(means_table)
    overall = overall_ranking(ratings)
    overall_vec = [overall[c] for c in CRIT_IDS]

    print(f"{'School':<12} {'rho (Spearman)':>14} {'p':>10}  Interpretation")
    print('-' * 60)
    for sch in school_list:
        sch_vec = [ranks[sch][c] for c in CRIT_IDS]
        rho, p = stats.spearmanr(sch_vec, overall_vec)
        if p < 0.01 and rho > 0.8:
            interp = "Very strong"
        elif p < 0.05 and rho > 0.6:
            interp = "Strong"
        elif p < 0.1:
            interp = "Moderate"
        else:
            interp = "Weak / none"
        print(f"{sch:<12} {rho:>14.4f} {p:>10.4f}  {interp}")

    # ----- 4.2 RANK CORRELATION BETWEEN PAIRS OF SCHOOLS -----
    print("\n" + "="*70)
    print("4.2. Spearman rank correlation between pairs of schools")
    print("="*70)
    print(f"{'School pair':<24} {'rho (Spearman)':>14} {'p':>10}  Interpretation")
    print('-' * 70)
    for s1, s2 in combinations(school_list, 2):
        v1 = [ranks[s1][c] for c in CRIT_IDS]
        v2 = [ranks[s2][c] for c in CRIT_IDS]
        rho, p = stats.spearmanr(v1, v2)
        if p < 0.01 and rho > 0.8:
            interp = "Very strong"
        elif p < 0.05 and rho > 0.6:
            interp = "Strong"
        elif p < 0.1:
            interp = "Moderate"
        else:
            interp = "Weak / none"
        print(f"{s1 + ' vs ' + s2:<24} {rho:>14.4f} {p:>10.4f}  {interp}")

    # ----- 4.3 KENDALL'S W -----
    print("\n" + "="*70)
    print("4.3. Kendall's W — global coefficient of concordance")
    print("="*70)
    rank_matrix = np.array([[ranks[s][c] for c in CRIT_IDS] for s in school_list])
    kw = kendall_w(rank_matrix)
    print(f"  W       = {kw['W']:.4f}")
    print(f"  chi2    = {kw['chi2']:.4f}  (df = {kw['n_items'] - 1})")
    print(f"  p       = {kw['p']:.4f}")
    print(f"  k       = {kw['k']} (number of schools as judges)")
    print(f"  n_items = {kw['n_items']} (number of criteria)")
    print("\n  Interpretation:")
    print("    W close to 1 — strong rank consensus between schools")
    print("    W close to 0 — no consensus")
    print("    p < 0.05    — can reject H0 of complete lack of concordance")


if __name__ == "__main__":
    main()
