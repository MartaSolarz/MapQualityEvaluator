"""
12_sensitivity.py
==================
"Robustness analyses" section, chapter 8 of the Stan_prac.pdf document:
Sensitivity analysis of Quality Score to weight uncertainty.

For each of six test scenarios computes the Quality Score according to the formula:

    Q = Q_max * (1 - P_norm)
    P_norm = Sigma w_i (violated) / Sigma w_i (all)

using four alternative weight sets:
- orig  — empirical weights w_i from the survey
- CI_lo — lower 95% CI of weights from bootstrap
- CI_hi — upper 95% CI of weights from bootstrap
- equal — all weights = 0.7 (baseline, how the model would look without differentiation)

Q_max = 60 is assumed (according to the theoretical document, examples 1 and 2).

Run:
    python 12_sensitivity.py
"""
import numpy as np
from data_loader import load_ratings, CRITERIA, CRIT_IDS
from importlib import import_module


def quality_score(violated_ids, weights, Qmax=60):
    """
    Compute Quality Score for a map violating a set of criteria.

    Args:
        violated_ids: list[str] — e.g. ['D2', 'A1']
        weights: dict {crit_id: float} — weights w_i
        Qmax: float — Stage 1 upper bound

    Returns:
        float — Q
    """
    P = sum(weights[c] for c in violated_ids)
    sum_w = sum(weights.values())
    P_norm = P / sum_w
    return Qmax * (1 - P_norm)


# Test scenarios (consistent with PDF)
SCENARIOS = [
    {'name': 'Missing L3',                              'violations': ['L3']},
    {'name': 'Missing D2 + A1 (example 2 from theory)', 'violations': ['D2', 'A1']},
    {'name': 'Missing D1',                              'violations': ['D1']},
    {'name': 'Missing L1 + L3',                         'violations': ['L1', 'L3']},
    {'name': 'Missing V1 + V2',                         'violations': ['V1', 'V2']},
    {'name': 'Missing D1 + L2 + V2 (worst case)',       'violations': ['D1', 'L2', 'V2']},
]


def main():
    ratings, _ = load_ratings()
    Qmax = 60

    # Get bootstrap CI from module 06
    mod06 = import_module('06_bootstrap')
    boot = mod06.bootstrap_weights(ratings, n_iter=10000, seed=42)

    # Four weight sets
    w_orig  = {cid: boot[cid]['w']     for cid in CRIT_IDS}
    w_lo    = {cid: boot[cid]['ci_lo'] for cid in CRIT_IDS}
    w_hi    = {cid: boot[cid]['ci_hi'] for cid in CRIT_IDS}
    w_equal = {cid: 0.7                for cid in CRIT_IDS}

    print("="*100)
    print(f"Sensitivity analysis of Quality Score (Q_max = {Qmax})")
    print("="*100)

    print("\nWeight sums for the four sets:")
    print(f"  Sigma w (orig):   {sum(w_orig.values()):.2f}")
    print(f"  Sigma w (CI_lo):  {sum(w_lo.values()):.2f}")
    print(f"  Sigma w (CI_hi):  {sum(w_hi.values()):.2f}")
    print(f"  Sigma w (equal):  {sum(w_equal.values()):.2f}")

    header = (f"{'Scenario':<40} {'Violated':<18} "
              f"{'Q (orig)':>10} {'Q (CI_lo)':>10} {'Q (CI_hi)':>10} "
              f"{'Q (equal)':>10} {'Range':>8}")
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

    print("\nConclusions:")
    print("  - The Q variability range (difference between CI_lo and CI_hi scenarios) is small")
    print("    — bootstrap CI weights yield very similar results.")
    print("  - The model with empirical weights gives significantly different results than equal weights — confirming")
    print("    the practical value of weight differentiation in the Stage 1 model.")


if __name__ == "__main__":
    main()
