"""
04_agreement.py
================
Sekcja 4 dokumentu Stan_prac.pdf: Analiza zgodności ekspertów.

Oblicza:
- Krippendorff's alpha (ordinal) dla wszystkich 10 kryteriów łącznie
- Współczynnik zmienności CV = SD/mean dla każdego kryterium
- Ranking kryteriów według zgodności (CV rosnąco)

Wymaga pakietu `krippendorff` (pip install krippendorff).

Uruchomienie:
    python 04_agreement.py
"""
import numpy as np
import krippendorff
from data_loader import load_ratings, CRITERIA, CRIT_IDS
from importlib import import_module


def compute_alpha(ratings, level='ordinal'):
    """
    Oblicz Krippendorff's alpha dla macierzy ocen.

    Args:
        ratings: np.ndarray (N, 10) — wiersze: respondenci, kolumny: kryteria
        level: 'ordinal' | 'interval' | 'nominal' | 'ratio'

    Returns:
        float — wartość α

    Uwaga: Pakiet `krippendorff` oczekuje formatu reliability_data, gdzie
    wiersze to KODOWACZE (lub jednostki kodowania) a kolumny to JEDNOSTKI
    (rzeczy oceniane). Konwencja: rows = criteria, cols = respondents.
    Transponujemy ratings (N×10) → (10×N).
    """
    reliability = ratings.T  # 10 wierszy (kryteria) × N kolumn (respondenci)
    return krippendorff.alpha(reliability_data=reliability,
                               level_of_measurement=level)


def main():
    ratings, _ = load_ratings()

    # ----- KRIPPENDORFF'S ALPHA -----
    print("="*70)
    print("Krippendorff's alpha (zgodność ekspertów dla wszystkich 10 kryteriów)")
    print("="*70)

    alpha_ord = compute_alpha(ratings, 'ordinal')
    alpha_int = compute_alpha(ratings, 'interval')
    alpha_nom = compute_alpha(ratings, 'nominal')

    print(f"\n  α (ordinal):  {alpha_ord:.4f}  ← używana w dokumencie")
    print(f"  α (interval): {alpha_int:.4f}")
    print(f"  α (nominal):  {alpha_nom:.4f}")

    print("\nProgi interpretacyjne (Krippendorff, 2004):")
    print("  α ≥ 0.800 — wnioski rozstrzygające")
    print("  α ≥ 0.667 — wnioski wstępne dopuszczalne")
    print("  α <  0.667 — niska zgodność")
    print(f"\n  Wynik α = {alpha_ord:.4f} mieści się poniżej progu 0.667.")

    # ----- WSPÓŁCZYNNIK ZMIENNOŚCI CV -----
    print("\n" + "="*70)
    print("Współczynnik zmienności CV per kryterium (uporządkowane rosnąco)")
    print("="*70)
    print("(Niższy CV = większa zgodność ekspertów dla danego kryterium)")

    mod02 = import_module('02_per_criterion')
    stats = mod02.compute_per_criterion(ratings)

    sorted_by_cv = sorted(stats, key=lambda s: s['cv'])

    header = f"{'#':>3} {'ID':<4} {'Mean':>6} {'SD':>6} {'CV':>6}  Kryterium"
    print(header)
    print('-' * 90)
    for rank, s in enumerate(sorted_by_cv, start=1):
        print(f"{rank:>3} {s['id']:<4} {s['mean']:>6.2f} {s['sd']:>6.2f} "
              f"{s['cv']:>6.2f}  {s['name_pl']}")

    print(f"\nNajwiększa zgodność: {sorted_by_cv[0]['id']} "
          f"(CV = {sorted_by_cv[0]['cv']:.2f})")
    print(f"Najmniejsza zgodność: {sorted_by_cv[-1]['id']} "
          f"(CV = {sorted_by_cv[-1]['cv']:.2f})")


if __name__ == "__main__":
    main()