"""
09_friedman.py
===============
Sekcja "Analizy odporności", rozdz. 5 dokumentu Stan_prac.pdf:
Test Friedmana — czy kryteria różnią się istotnie.

Test Friedmana jest nieparametrycznym odpowiednikiem analizy wariancji z
powtarzanymi pomiarami. Testuje hipotezę zerową, że wszystkie 10 kryteriów
są oceniane równie surowo (mediany rozkładów ocen są równe).

Jeśli H₀ zostanie odrzucona (p < 0.05), oznacza to, że kryteria różnią się
istotnie — co uzasadnia architekturę modelu Stage 1 opartego na zróżnicowanych
wagach.

Uruchomienie:
    python 09_friedman.py
"""
import numpy as np
from scipy import stats
from data_loader import load_ratings, CRITERIA, CRIT_IDS


def main():
    ratings, _ = load_ratings()

    print("="*70)
    print("Test Friedmana — czy kryteria różnią się w surowości oceny")
    print("="*70)

    # scipy.stats.friedmanchisquare przyjmuje serie ocen — jedną serię per kryterium.
    # Każda seria to oceny wszystkich respondentów dla tego kryterium.
    series = [ratings[:, i] for i in range(10)]
    chi2, p = stats.friedmanchisquare(*series)

    print(f"\n  H₀: wszystkie 10 kryteriów ma równą medianę ocen")
    print(f"  H₁: przynajmniej jedno kryterium różni się od pozostałych")
    print()
    print(f"  Statystyka chi-square: {chi2:.4f}")
    print(f"  Stopnie swobody (df):   9  (k - 1, gdzie k = 10 kryteriów)")
    if p < 1e-6:
        p_str = "< 0.000001"
    else:
        p_str = f"{p:.6f}"
    print(f"  Wartość p:              {p_str}")

    print()
    if p < 0.05:
        print("  Wniosek: H₀ odrzucona — kryteria różnią się istotnie w surowości oceny.")
        print("  Implikacja: architektura modelu Stage 1 z indywidualnymi wagami w_i")
        print("              jest empirycznie uzasadniona.")
    else:
        print("  Wniosek: brak podstaw do odrzucenia H₀.")

    # Dodatkowa informacja: średnie rangi per kryterium
    print("\n--- Średnie rangi per kryterium (rosnąco) ---")
    # Dla każdego respondenta uszereguj kryteria (rangi 1..10 wewnątrz wiersza)
    # Wyższa ranga = wyższa ocena.
    N = ratings.shape[0]
    ranks_per_resp = np.zeros_like(ratings)
    for j in range(N):
        # method='average' obsługuje tied values poprawnie
        ranks_per_resp[j] = stats.rankdata(ratings[j], method='average')
    mean_ranks = ranks_per_resp.mean(axis=0)

    order = np.argsort(mean_ranks)
    print(f"\n{'ID':<4} {'Mean rank':>10}  Kryterium")
    print('-' * 70)
    for i in order:
        print(f"{CRIT_IDS[i]:<4} {mean_ranks[i]:>10.3f}  {CRITERIA[i]['name_short_pl']}")


if __name__ == "__main__":
    main()