"""
01_participants.py
==================
Sekcja 1 dokumentu Stan_prac.pdf: Charakterystyka respondentów.

Oblicza i drukuje:
- 1.1 Rozkład płci
- 1.2 Rozkład krajów (znormalizowany)
- 1.3 Rozkład poziomu wykształcenia
- 1.4 Typ instytucji
- 1.5 Praca z mapami i mapami statystycznymi
- 1.6 Lata doświadczenia (mean, median, SD, min, max)
- 1.7 Samoocena poziomu eksperckiego

Uruchomienie:
    python 01_participants.py

Wszystkie liczby zaokrąglono do 2 miejsc po przecinku.
"""
import pandas as pd
import numpy as np
from data_loader import (
    load_ratings, normalize_country, parse_years,
    COL_GENDER, COL_COUNTRY, COL_INSTITUTION, COL_EDUCATION,
    COL_WORKS_MAPS, COL_YEARS_MAPS, COL_WORKS_STAT, COL_YEARS_STAT, COL_SELF_EXPERT,
)


def section_header(title):
    print(f"\n{'='*70}")
    print(title)
    print('='*70)


def print_freq_table(series, title, N):
    """Wydrukuj tabelę częstości z liczbą i procentem."""
    print(f"\n--- {title} ---")
    counts = series.value_counts()
    for k, v in counts.items():
        pct = (v / N) * 100
        print(f"  {str(k):<55} {v:>4}  ({pct:>5.2f}%)")


def classify_institution(inst):
    """Klasyfikacja typu instytucji na podstawie nazwy."""
    if pd.isna(inst):
        return 'Nie podano'
    s = str(inst).lower()
    if 'leibniz' in s and 'university' not in s:
        return 'Instytut badawczy'
    university_markers = [
        'university', 'tum', 'tu dresden', 'ensg', 'charles', 'leeds',
        'chicago', 'oregon', 'east anglia', 'edinburgh', 'manchester',
        'cambridge', 'bournemouth', 'westminster', 'munich', 'open university',
        'heriot', 'ucla', 'ghana', 'wien', 'tu '
    ]
    if any(m in s for m in university_markers):
        return 'Uniwersytet / uczelnia wyższa'
    return 'Inne'


def main():
    ratings, df = load_ratings()
    N = len(df)
    print(f"Łączna liczba respondentów: N = {N}")

    # ----- 1.1 PŁEĆ -----
    section_header("1.1. Płeć")
    print_freq_table(df.iloc[:, COL_GENDER], "Rozkład płci", N)

    # ----- 1.2 KRAJ -----
    section_header("1.2. Kraj afiliacji")
    countries_norm = [normalize_country(c) for c in df.iloc[:, COL_COUNTRY]]
    # W dokumencie używamy formy "raw" (np. 'UK', 'Germany'), ale z połączeniami wariantów.
    # Tutaj pokazujemy oba rzuty.
    print("\n--- Rozkład krajów (wersja znormalizowana) ---")
    counts_norm = pd.Series(countries_norm).value_counts()
    for k, v in counts_norm.items():
        pct = (v / N) * 100
        print(f"  {k:<25} {v:>4}  ({pct:>5.2f}%)")

    print("\n--- Rozkład krajów (wersje wzbogacone — wpisy wielokrajowe rozdzielone) ---")
    print("(Uwaga: dokument zachowuje wpisy złożone jako odrębne kategorie.)")
    raw = df.iloc[:, COL_COUNTRY].astype(str).str.strip()
    # Zastąp różne formy "UK" jednolitym zapisem 'UK', a Czech variants → 'Czechia'
    def display_form(c):
        s = str(c).strip()
        if s.upper() == 'UK' or s == 'Uk':
            return 'UK'
        if s in ('Czechia', 'Czech Republic', 'CZ'):
            return 'Czechia'
        return s
    display = raw.apply(display_form)
    counts_display = display.value_counts()
    for k, v in counts_display.items():
        pct = (v / N) * 100
        print(f"  {k:<45} {v:>4}  ({pct:>5.2f}%)")

    # ----- 1.3 WYKSZTAŁCENIE -----
    section_header("1.3. Poziom wykształcenia")
    print_freq_table(df.iloc[:, COL_EDUCATION], "Wykształcenie", N)

    # ----- 1.4 INSTYTUCJE -----
    section_header("1.4. Typ instytucji")
    institutions = [classify_institution(i) for i in df.iloc[:, COL_INSTITUTION]]
    print_freq_table(pd.Series(institutions), "Typ instytucji", N)

    # ----- 1.5 PRACA Z MAPAMI -----
    section_header("1.5. Praca z mapami i mapami statystycznymi")
    print("\n--- Praca z mapami (ogółem) ---")
    maps_counts = df.iloc[:, COL_WORKS_MAPS].value_counts()
    for k, v in maps_counts.items():
        pct = (v / N) * 100
        print(f"  {str(k):<25} {v:>4}  ({pct:>5.2f}%)")

    print("\n--- Praca z mapami statystycznymi ---")
    stat_counts = df.iloc[:, COL_WORKS_STAT].value_counts()
    for k, v in stat_counts.items():
        pct = (v / N) * 100
        print(f"  {str(k):<25} {v:>4}  ({pct:>5.2f}%)")

    # ----- 1.6 LATA DOŚWIADCZENIA -----
    section_header("1.6. Lata doświadczenia")

    years_maps = [parse_years(v) for v in df.iloc[:, COL_YEARS_MAPS]]
    years_maps_clean = [y for y in years_maps if y is not None]

    years_stat = [parse_years(v) for v in df.iloc[:, COL_YEARS_STAT]]
    years_stat_clean = [y for y in years_stat if y is not None]

    def summarize(name, values, N_total):
        print(f"\n--- {name} ---")
        if not values:
            print("  Brak danych")
            return
        print(f"  N (liczba udzielonych odp): {len(values)}/{N_total}")
        print(f"  Średnia (mean): {np.mean(values):.2f}")
        print(f"  Mediana:        {np.median(values):.2f}")
        print(f"  Odch. std (SD): {np.std(values, ddof=1):.2f}")
        print(f"  Min:            {min(values):.2f}")
        print(f"  Max:            {max(values):.2f}")

    summarize("Mapy ogółem", years_maps_clean, N)
    summarize("Mapy statystyczne", years_stat_clean, N)

    print("\n--- Wartości tekstowe (sprowadzone do liczb) ---")
    text_to_number = []
    for raw, parsed in zip(df.iloc[:, COL_YEARS_MAPS], years_maps):
        if not pd.isna(raw) and not isinstance(raw, (int, float)):
            text_to_number.append((raw, parsed))
    for raw, parsed in zip(df.iloc[:, COL_YEARS_STAT], years_stat):
        if not pd.isna(raw) and not isinstance(raw, (int, float)):
            text_to_number.append((raw, parsed))
    if text_to_number:
        # Usuń duplikaty zachowując kolejność
        seen = set()
        for raw, parsed in text_to_number:
            key = str(raw).strip()
            if key not in seen:
                seen.add(key)
                print(f"  '{raw}'  →  {parsed}")
    else:
        print("  (brak wartości tekstowych)")

    # ----- 1.7 SAMOOCENA EKSPERCKA -----
    section_header("1.7. Samoocena poziomu eksperckiego")
    expertise = df.iloc[:, COL_SELF_EXPERT].dropna().tolist()
    print(f"\n  N: {len(expertise)}")
    print(f"  Średnia: {np.mean(expertise):.2f}")
    print(f"  Mediana: {np.median(expertise):.2f}")
    print(f"  SD:      {np.std(expertise, ddof=1):.2f}")
    print(f"\n--- Rozkład ocen ---")
    for lvl in [1, 2, 3, 4, 5]:
        cnt = sum(1 for e in expertise if e == lvl)
        pct = (cnt / N) * 100
        print(f"  Ocena {lvl}: {cnt:>4}  ({pct:>5.2f}%)")


if __name__ == "__main__":
    main()