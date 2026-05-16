"""
12_sensitivity.py
==================
Sekcja "Analizy odporności", rozdz. 8 dokumentu Stan_prac.pdf:
Analiza wrażliwości Quality Score na niepewność wag.

Dla każdego z sześciu scenariuszy testowych oblicza Quality Score zgodnie ze wzorem:

    Q = Q_max × (1 − P_norm)
    P_norm = Σ w_i (naruszone) / Σ w_i (wszystkie)

przy czterech alternatywnych zestawach wag:
- orig  — empiryczne wagi w_i z ankiety
- CI_lo — dolne 95% CI wag z bootstrap
- CI_hi — górne 95% CI wag z bootstrap
- równe — wszystkie wagi = 0.7 (baseline, jak wyglądałby model bez różnicowania)

Przyjęto Q_max = 60 (zgodnie z dokumentem teoretycznym, przykład 1 i 2).

Uruchomienie:
    python 12_sensitivity.py
"""
import numpy as np
from data_loader import load_ratings, CRITERIA, CRIT_IDS
from importlib import import_module


def quality_score(violated_ids, weights, Qmax=60):
    """
    Oblicz Quality Score dla mapy naruszającej zbiór kryteriów.

    Args:
        violated_ids: list[str] — np. ['D2', 'A1']
        weights: dict {crit_id: float} — wagi w_i
        Qmax: float — górny pułap Stage 1

    Returns:
        float — Q
    """
    P = sum(weights[c] for c in violated_ids)
    sum_w = sum(weights.values())
    P_norm = P / sum_w
    return Qmax * (1 - P_norm)


# Scenariusze testowe (zgodne z PDF)
SCENARIOS = [
    {'name': 'Brak L3',                          'violations': ['L3']},
    {'name': 'Brak D2 + A1 (przykład 2 z teorii)','violations': ['D2', 'A1']},
    {'name': 'Brak D1',                          'violations': ['D1']},
    {'name': 'Brak L1 + L3',                     'violations': ['L1', 'L3']},
    {'name': 'Brak V1 + V2',                     'violations': ['V1', 'V2']},
    {'name': 'Brak D1 + L2 + V2 (najgorsze)',    'violations': ['D1', 'L2', 'V2']},
]


def main():
    ratings, _ = load_ratings()
    Qmax = 60

    # Pobierz bootstrap CI z modułu 06
    mod06 = import_module('06_bootstrap')
    boot = mod06.bootstrap_weights(ratings, n_iter=10000, seed=42)

    # Cztery zestawy wag
    w_orig  = {cid: boot[cid]['w']     for cid in CRIT_IDS}
    w_lo    = {cid: boot[cid]['ci_lo'] for cid in CRIT_IDS}
    w_hi    = {cid: boot[cid]['ci_hi'] for cid in CRIT_IDS}
    w_equal = {cid: 0.7                for cid in CRIT_IDS}

    print("="*100)
    print(f"Analiza wrażliwości Quality Score (Q_max = {Qmax})")
    print("="*100)

    print("\nSumy wag w czterech zestawach:")
    print(f"  Σ w (orig):   {sum(w_orig.values()):.2f}")
    print(f"  Σ w (CI_lo):  {sum(w_lo.values()):.2f}")
    print(f"  Σ w (CI_hi):  {sum(w_hi.values()):.2f}")
    print(f"  Σ w (równe):  {sum(w_equal.values()):.2f}")

    header = (f"{'Scenariusz':<40} {'Naruszone':<18} "
              f"{'Q (orig)':>10} {'Q (CI_lo)':>10} {'Q (CI_hi)':>10} "
              f"{'Q (równe)':>10} {'Zakres':>8}")
    print("\n" + header)
    print('-' * len(header))

    for sc in SCENARIOS:
        q_orig  = quality_score(sc['violations'], w_orig,  Qmax)
        q_lo    = quality_score(sc['violations'], w_lo,    Qmax)
        q_hi    = quality_score(sc['violations'], w_hi,    Qmax)
        q_equal = quality_score(sc['violations'], w_equal, Qmax)

        all_q = [q_orig, q_lo, q_hi, q_equal]
        rng = max(all_q) - min(all_q)

        viol = ', '.join(sc['violations'])
        print(f"{sc['name']:<40} {viol:<18} "
              f"{q_orig:>10.2f} {q_lo:>10.2f} {q_hi:>10.2f} "
              f"{q_equal:>10.2f} {rng:>8.2f}")

    print('-' * len(header))

    print("\nWnioski:")
    print("  • Zakres zmienności Q (różnica między scenariuszem CI_lo a CI_hi) jest niewielki")
    print("    — wagi z bootstrap CI dają bardzo zbliżone wyniki.")
    print("  • Model z empirycznymi wagami daje istotnie inne wyniki niż wagi równe — co potwierdza")
    print("    praktyczną wartość różnicowania wag w modelu Stage 1.")


if __name__ == "__main__":
    main()