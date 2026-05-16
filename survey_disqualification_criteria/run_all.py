"""
run_all.py
==========
Runs all analytical modules sequentially and prints their results.
Allows reproducing all numbers from the Stan_prac.pdf document with a single command.

Run:
    python run_all.py
"""
import subprocess
import sys
import os

MODULES = [
    ('01_participants.py',     'Section 1: Respondent characteristics'),
    ('02_per_criterion.py',    'Section 2: Results per criterion'),
    ('03_ranking.py',          'Section 3: Ranking and category comparison'),
    ('04_agreement.py',        'Section 4: Expert agreement (Krippendorff alpha, CV)'),
    ('05_correlations.py',     'Section 5.3: Correlations with experience'),
    ('06_bootstrap.py',        'Robustness 1: Bootstrap 95% CI'),
    ('07_jackknife.py',        'Robustness 2: Jackknife (leave-one-out)'),
    ('08_schools.py',          'Robustness 3-4: Inter-school comparison'),
    ('09_friedman.py',         'Robustness 5: Friedman test'),
    ('10_effect_size.py',      'Robustness 6: Effect size (Cohen\'s d)'),
    ('11_respondent_diag.py',  'Robustness 7: Respondent diagnostics'),
    ('12_sensitivity.py',      'Robustness 8: Quality Score sensitivity analysis'),
]


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    for fname, title in MODULES:
        print("\n\n" + "#" * 100)
        print(f"# {title}")
        print(f"# File: {fname}")
        print("#" * 100)
        path = os.path.join(here, fname)
        result = subprocess.run([sys.executable, path], capture_output=True, text=True)
        print(result.stdout, end='')
        if result.returncode != 0:
            print(f"\n[!] ERROR in {fname}:")
            print(result.stderr)
            sys.exit(1)


if __name__ == "__main__":
    main()
