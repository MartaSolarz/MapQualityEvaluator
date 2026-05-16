"""
02_per_criterion.py
====================
Sekcja 2 dokumentu Stan_prac.pdf: Wyniki per kryterium.

Dla każdego z 10 kryteriów oblicza:
- Średnią (mean) ocen ekspertów
- Medianę
- Odchylenie standardowe (SD, próbkowe n-1)
- Min, Max
- Rozkład odpowiedzi (n=1, n=2, ..., n=5)
- Wagę dyskwalifikującą w_i = (mean - 1) / 4
- Współczynnik zmienności CV = SD / mean

Drukuje tabelę zbiorczą zgodną z tabelą w sekcji 2 dokumentu oraz
sumę wag Σ w_i potrzebną do normalizacji w modelu Stage 1.

Uruchomienie:
    python 02_per_criterion.py
"""
import numpy as np
from data_loader import load_ratings, CRITERIA, CRIT_IDS


def compute_per_criterion(ratings):
    """
    Oblicz pełen zestaw statystyk dla każdego kryterium.

    Args:
        ratings: np.ndarray (N, 10) — macierz ocen

    Returns:
        list of dict — jeden dict per kryterium z polami:
            id, mean, median, sd, min, max, dist (rozkład 1..5),
            w (waga), cv (współczynnik zmienności), n
    """
    results = []
    for i, crit in enumerate(CRITERIA):
        col = ratings[:, i]
        col = col[~np.isnan(col)]  # usuń NaN, gdyby były
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
            'name_pl': crit['name_pl'],
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

    # Nagłówek tabeli zgodny z tabelą w sekcji 2 dokumentu
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
    print(f"\nSuma wag wszystkich kryteriów: Σ w_i = {sum_w:.2f}")
    print("(Wartość ta stanowi mianownik w normalizacji kary P_norm = P / Σ w_i "
          "w modelu Stage 1.)")

    print("\nNazwy kryteriów:")
    for s in stats:
        print(f"  {s['id']}: {s['name_pl']}  [{s['category']}]")


if __name__ == "__main__":
    main()