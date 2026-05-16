"""
07_jackknife.py
================
"Robustness analyses" section, chapter 2 of the Stan_prac.pdf document:
Jackknife (leave-one-out) — impact of a single respondent on weights.

Method: for each of the 10 criteria we compute the weight w_i N times, each time
omitting one respondent from the sample. This allows identifying whether any
respondent disproportionately influences the obtained weight (outlying observation).

For each criterion we report:
- w_i (full) — weight from the full sample N = 25
- jk_min, jk_max — range of jackknife weights (after omitting one respondent)
- max_change — maximum change |w_i_jk - w_i_full|
- most_influential_resp — index of the most influential respondent

Run:
    python 07_jackknife.py
"""
import numpy as np
from data_loader import load_ratings, CRITERIA, CRIT_IDS, COL_COUNTRY


def jackknife_weights(ratings):
    """
    Leave-one-out procedure for each criterion.

    Returns:
        dict {crit_id: {'w_full', 'jk_min', 'jk_max', 'max_change',
                        'most_influential_resp'}}
    """
    N = ratings.shape[0]
    results = {}
    for i, crit in enumerate(CRITERIA):
        col = ratings[:, i]
        w_full = (col.mean() - 1) / 4

        jk_weights = np.zeros(N)
        for j in range(N):
            mask = np.ones(N, dtype=bool)
            mask[j] = False
            jk_weights[j] = (col[mask].mean() - 1) / 4

        changes = np.abs(jk_weights - w_full)
        most_inf = int(np.argmax(changes))

        results[crit['id']] = {
            'w_full': float(w_full),
            'jk_min': float(jk_weights.min()),
            'jk_max': float(jk_weights.max()),
            'max_change': float(changes.max()),
            'most_influential_resp': most_inf,
        }
    return results


def main():
    ratings, df = load_ratings()
    countries = df.iloc[:, COL_COUNTRY].astype(str).str.strip().tolist()
    jk = jackknife_weights(ratings)

    print("="*90)
    print("Jackknife (leave-one-out) — impact of a single respondent on weights w_i")
    print("="*90)
    header = (f"{'ID':<4} {'w (full)':>10} {'jk_min':>8} {'jk_max':>8} "
              f"{'Max change':>14}  {'Influential respondent':<30}")
    print(header)
    print('-' * len(header))

    for cid in CRIT_IDS:
        r = jk[cid]
        resp_idx = r['most_influential_resp']
        resp_country = countries[resp_idx] if resp_idx < len(countries) else '—'
        print(f"{cid:<4} {r['w_full']:>10.3f} {r['jk_min']:>8.3f} {r['jk_max']:>8.3f} "
              f"{r['max_change']:>14.3f}  #{resp_idx} ({resp_country})")

    max_changes = [jk[c]['max_change'] for c in CRIT_IDS]
    print(f"\nRange of maximum changes: {min(max_changes):.3f} - {max(max_changes):.3f}")
    print("(Very small values indicate that no single respondent")
    print(" significantly changes the obtained disqualification weight.)")


if __name__ == "__main__":
    main()
