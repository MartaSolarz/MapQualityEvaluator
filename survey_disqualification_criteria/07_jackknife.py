"""
07_jackknife.py
================
Sekcja "Analizy odporności", rozdz. 2 dokumentu Stan_prac.pdf:
Jackknife (leave-one-out) — wpływ pojedynczego respondenta na wagi.

Metoda: dla każdego z 10 kryteriów obliczamy wagę w_i N razy, za każdym razem
pomijając jednego respondenta z próby. Pozwala to zidentyfikować, czy któryś
respondent dysproporcjonalnie wpływa na uzyskaną wagę (odstająca obserwacja).

Dla każdego kryterium podajemy:
- w_i (pełne) — waga z pełnej próby N = 25
- jk_min, jk_max — zakres wag z jackknife (po pominięciu jednego respondenta)
- max_change — maksymalna zmiana |w_i_jk - w_i_full|
- most_influential_resp — indeks respondenta o największym wpływie

Uruchomienie:
    python 07_jackknife.py
"""
import numpy as np
from data_loader import load_ratings, CRITERIA, CRIT_IDS, COL_COUNTRY


def jackknife_weights(ratings):
    """
    Procedura leave-one-out dla każdego kryterium.

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
    print("Jackknife (leave-one-out) — wpływ pojedynczego respondenta na wagi w_i")
    print("="*90)
    header = (f"{'ID':<4} {'w (pełne)':>10} {'jk_min':>8} {'jk_max':>8} "
              f"{'Maks. zmiana':>14}  {'Wpływowy respondent':<30}")
    print(header)
    print('-' * len(header))

    for cid in CRIT_IDS:
        r = jk[cid]
        resp_idx = r['most_influential_resp']
        resp_country = countries[resp_idx] if resp_idx < len(countries) else '—'
        print(f"{cid:<4} {r['w_full']:>10.3f} {r['jk_min']:>8.3f} {r['jk_max']:>8.3f} "
              f"{r['max_change']:>14.3f}  #{resp_idx} ({resp_country})")

    max_changes = [jk[c]['max_change'] for c in CRIT_IDS]
    print(f"\nZakres maksymalnych zmian: {min(max_changes):.3f} – {max(max_changes):.3f}")
    print("(Bardzo małe wartości oznaczają, że żaden pojedynczy respondent")
    print(" nie zmienia istotnie uzyskanej wagi dyskwalifikującej.)")


if __name__ == "__main__":
    main()