"""
run_all.py
==========
Uruchamia wszystkie moduły analityczne po kolei i drukuje ich wyniki.
Pozwala odtworzyć wszystkie liczby z dokumentu Stan_prac.pdf jednym poleceniem.

Uruchomienie:
    python run_all.py
"""
import subprocess
import sys
import os

MODULES = [
    ('01_participants.py',     'Sekcja 1: Charakterystyka respondentów'),
    ('02_per_criterion.py',    'Sekcja 2: Wyniki per kryterium'),
    ('03_ranking.py',          'Sekcja 3: Ranking i porównanie kategorii'),
    ('04_agreement.py',        'Sekcja 4: Zgodność ekspertów (Krippendorff α, CV)'),
    ('05_correlations.py',     'Sekcja 5.3: Korelacje z doświadczeniem'),
    ('06_bootstrap.py',        'Odporność 1: Bootstrap 95% CI'),
    ('07_jackknife.py',        'Odporność 2: Jackknife (leave-one-out)'),
    ('08_schools.py',          'Odporność 3-4: Porównanie międzyszkolne'),
    ('09_friedman.py',         'Odporność 5: Test Friedmana'),
    ('10_effect_size.py',      'Odporność 6: Effect size (Cohen\'s d)'),
    ('11_respondent_diag.py',  'Odporność 7: Diagnostyka respondenta'),
    ('12_sensitivity.py',      'Odporność 8: Analiza wrażliwości Quality Score'),
]


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    for fname, title in MODULES:
        print("\n\n" + "#" * 100)
        print(f"# {title}")
        print(f"# Plik: {fname}")
        print("#" * 100)
        path = os.path.join(here, fname)
        result = subprocess.run([sys.executable, path], capture_output=True, text=True)
        print(result.stdout, end='')
        if result.returncode != 0:
            print(f"\n[!] BŁĄD w {fname}:")
            print(result.stderr)
            sys.exit(1)


if __name__ == "__main__":
    main()