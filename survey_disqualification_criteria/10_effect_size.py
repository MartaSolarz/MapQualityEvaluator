"""
10_effect_size.py
==================
Sekcja "Analizy odporności", rozdz. 6 dokumentu Stan_prac.pdf:
Effect size (Cohen's d) i testy względem środka skali (= 3).

Dla każdego kryterium oblicza:
- Cohen's d = (mean - 3) / SD
  Jednopróbkowa wersja d (efekt względem wartości testowej = środek skali Likerta).
  d > 0 oznacza, że oceny ekspertów są przesunięte powyżej środka skali.
  |d| > 0.5 — umiarkowany, |d| > 0.8 — duży, |d| > 1.2 — bardzo duży.
- Jednopróbkowy test t (parametryczny)
- Test rangowy Wilcoxona dla różnicy rating - 3 (nieparametryczny, odporny na
  nienormalność rozkładu)

Uruchomienie:
    python 10_effect_size.py
"""
import numpy as np
from scipy import stats
from data_loader import load_ratings, CRITERIA, CRIT_IDS


def cohen_d_one_sample(values, mu0=3.0):
    """Cohen's d dla jednej próby względem wartości testowej mu0."""
    mean = values.mean()
    sd = values.std(ddof=1)
    return (mean - mu0) / sd


def effect_size_label(d):
    """Konwencja Cohena dla wielkości efektu."""
    ad = abs(d)
    if ad >= 1.2: return "Bardzo duży"
    if ad >= 0.8: return "Duży"
    if ad >= 0.5: return "Umiarkowany"
    if ad >= 0.2: return "Mały"
    return "Pomijalny"


def main():
    ratings, _ = load_ratings()
    MIDPOINT = 3.0

    print("="*100)
    print(f"Effect size (Cohen's d) i testy istotności względem środka skali = {MIDPOINT}")
    print("="*100)

    header = (f"{'ID':<4} {'Mean':>6} {'SD':>5} {'d':>7} "
              f"{'t':>7} {'p(t)':>9} {'p(W)':>9}  Wielkość efektu")
    print(header)
    print('-' * len(header))

    results = []
    for i, crit in enumerate(CRITERIA):
        col = ratings[:, i]
        mean = col.mean()
        sd = col.std(ddof=1)
        d = cohen_d_one_sample(col, MIDPOINT)
        t_stat, t_p = stats.ttest_1samp(col, MIDPOINT)
        try:
            w_stat, w_p = stats.wilcoxon(col - MIDPOINT)
        except Exception:
            w_stat, w_p = (float('nan'), float('nan'))

        results.append({
            'id': crit['id'], 'mean': float(mean), 'sd': float(sd),
            'd': float(d), 't': float(t_stat),
            'p_t': float(t_p), 'p_wilcoxon': float(w_p),
        })
        p_t_str = "<0.0001" if t_p < 0.0001 else f"{t_p:.4f}"
        p_w_str = "<0.0001" if w_p < 0.0001 else f"{w_p:.4f}"
        marker = "" if t_p < 0.05 else " *"  # nieistotne
        print(f"{crit['id']:<4} {mean:>6.2f} {sd:>5.2f} {d:>7.3f} "
              f"{t_stat:>7.3f} {p_t_str:>9} {p_w_str:>9}  {effect_size_label(d)}{marker}")

    print('-' * len(header))
    print("* = odbieganie od środka skali NIE jest statystycznie istotne (p ≥ 0.05)")

    print("\nObjaśnienia kolumn:")
    print("  Mean — średnia ocen")
    print("  SD   — odchylenie standardowe (próbkowe)")
    print("  d    — Cohen's d (jednopróbkowy) = (mean - 3) / SD")
    print("  t    — statystyka jednopróbkowego testu t")
    print("  p(t) — p-value z testu t (parametryczny)")
    print("  p(W) — p-value z testu Wilcoxona (nieparametryczny, na różnicy rating - 3)")

    print("\nProgi Cohena dla wielkości efektu (|d|):")
    print("  < 0.2 — pomijalny")
    print("  0.2 – 0.5 — mały")
    print("  0.5 – 0.8 — umiarkowany")
    print("  0.8 – 1.2 — duży")
    print("  > 1.2 — bardzo duży")


if __name__ == "__main__":
    main()