"""
11_respondent_diag.py
======================
Sekcja "Analizy odporności", rozdz. 7 dokumentu Stan_prac.pdf:
Diagnostyka poziomu respondenta — wykrywanie systematycznych obciążeń.

Bada dla każdego respondenta:
- Średnią ze wszystkich 10 jego ocen (mean personalne)
- SD jego ocen (rozrzut wewnątrzosobniczy)
- Liczbę unikalnych wartości skali, których użył (1..5)
- Liczbę odpowiedzi 1 i 5 (skrajne wartości)

Identyfikuje respondentów o niskiej różnorodności odpowiedzi:
- SD personalna < 0.5  → mało zróżnicowane oceny
- liczba unikalnych wartości ≤ 2 → korzystanie tylko z 2 punktów skali

Uruchomienie:
    python 11_respondent_diag.py
"""
import numpy as np
import pandas as pd
from data_loader import load_ratings, COL_COUNTRY


def main():
    ratings, df = load_ratings()
    countries = df.iloc[:, COL_COUNTRY].astype(str).str.strip().tolist()
    N = ratings.shape[0]

    print("="*90)
    print("Diagnostyka poziomu respondenta")
    print("="*90)

    diag = []
    for j in range(N):
        r = ratings[j]
        diag.append({
            'idx': j,
            'country': countries[j],
            'mean': r.mean(),
            'sd': r.std(ddof=1),
            'n_unique': len(np.unique(r)),
            'n_5': int((r == 5).sum()),
            'n_1': int((r == 1).sum()),
        })

    header = (f"{'#':>3} {'Kraj':<25} {'Mean':>6} {'SD':>6} "
              f"{'#unique':>8} {'n=5':>4} {'n=1':>4}")
    print(header)
    print('-' * len(header))
    for d in diag:
        flag = ""
        if d['sd'] < 0.5 or d['n_unique'] <= 2:
            flag = "  ← niska różnorodność"
        print(f"{d['idx']:>3} {d['country'][:25]:<25} {d['mean']:>6.2f} {d['sd']:>6.2f} "
              f"{d['n_unique']:>8} {d['n_5']:>4} {d['n_1']:>4}{flag}")

    # Statystyki średnich personalnych
    pers_means = [d['mean'] for d in diag]
    print(f"\n--- Rozkład średnich personalnych ---")
    print(f"  N:        {len(pers_means)}")
    print(f"  Min:      {min(pers_means):.2f}")
    print(f"  Max:      {max(pers_means):.2f}")
    print(f"  Mean:     {np.mean(pers_means):.2f}")
    print(f"  SD:       {np.std(pers_means, ddof=1):.2f}")

    # Respondenci podejrzani
    suspicious = [d for d in diag if d['sd'] < 0.5 or d['n_unique'] <= 2]
    print(f"\n--- Respondenci o niskiej różnorodności odpowiedzi ---")
    print(f"  Kryteria: SD personalna < 0.5  LUB  liczba unikalnych wartości ≤ 2")
    print(f"  Liczba: {len(suspicious)}")
    if suspicious:
        for s in suspicious:
            print(f"    #{s['idx']} ({s['country']}): mean = {s['mean']:.2f}, "
                  f"SD = {s['sd']:.2f}, unique = {s['n_unique']}, "
                  f"n_5 = {s['n_5']}, n_1 = {s['n_1']}")

    print("\nInterpretacja:")
    print("  Acquiescence bias = systematyczne zawyżanie ocen (mean osobnicza wysoka)")
    print("  Extreme response bias = skłonność do skrajnych wartości (1, 5)")
    print("  Niska różnorodność (np. tylko 4 i 5) może sygnalizować brak refleksji nad pytaniem.")
    print("\n  Wniosek: zob. tekst dokumentu Stan_prac.pdf, rozdział 7.")


if __name__ == "__main__":
    main()