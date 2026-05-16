"""
05_correlations.py
===================
Sekcja 5.3 dokumentu Stan_prac.pdf: Korelacja doświadczenia z surowością ocen.

Bada, czy długość doświadczenia respondenta z mapami (oraz jego samoocena ekspercka)
koreluje ze średnią oceną, jaką respondent przyznaje kryteriom — interpretowaną jako
"surowość" osądu.

Używa współczynnika korelacji rang Spearmana (ρ), odpowiedniego dla danych porządkowych.

Uruchomienie:
    python 05_correlations.py
"""
import numpy as np
from scipy import stats
from data_loader import (
    load_ratings, parse_years,
    COL_YEARS_MAPS, COL_YEARS_STAT, COL_SELF_EXPERT,
)


def main():
    ratings, df = load_ratings()
    N = len(df)

    # Surowość respondenta = średnia ze wszystkich jego ocen (pomija NaN-y)
    respondent_mean = ratings.mean(axis=1)

    # Lata doświadczenia z mapami (po sprowadzeniu wartości tekstowych do liczb)
    years_maps = np.array([parse_years(v) for v in df.iloc[:, COL_YEARS_MAPS]],
                          dtype=object)
    years_stat = np.array([parse_years(v) for v in df.iloc[:, COL_YEARS_STAT]],
                          dtype=object)
    self_expert = df.iloc[:, COL_SELF_EXPERT].values

    def spearman(x_raw, y_raw, label):
        # Wyfiltruj None / NaN parami
        pairs = [(x, y) for x, y in zip(x_raw, y_raw)
                 if x is not None and not (isinstance(x, float) and np.isnan(x))
                 and y is not None and not (isinstance(y, float) and np.isnan(y))]
        if len(pairs) < 3:
            print(f"  {label}: zbyt mało danych (N = {len(pairs)})")
            return
        xs, ys = zip(*pairs)
        rho, p = stats.spearmanr(xs, ys)
        sig = "" if p >= 0.05 else "  ← istotne (p < 0.05)"
        print(f"  {label:<55} N = {len(pairs):>2}  ρ = {rho:>7.4f}  p = {p:>6.4f}{sig}")

    print("="*80)
    print("Korelacja Spearmana między cechami respondenta a 'surowością' jego ocen")
    print("(Surowość = średnia ze wszystkich 10 ocen przyznanych przez respondenta)")
    print("="*80)
    print()

    spearman(years_maps,  respondent_mean, "Lata doświadczenia z mapami (ogółem)")
    spearman(years_stat,  respondent_mean, "Lata doświadczenia z mapami statystycznymi")
    spearman(self_expert, respondent_mean, "Samoocena ekspercka (1–5)")

    print("\nInterpretacja:")
    print("  Ujemne ρ — bardziej doświadczeni eksperci są MNIEJ surowi")
    print("  Dodatnie ρ — bardziej doświadczeni eksperci są BARDZIEJ surowi")
    print("  Istotność: p < 0.05 oznacza, że obserwowana korelacja jest mało prawdopodobna")
    print("             przy hipotezie zerowej o braku związku.")


if __name__ == "__main__":
    main()