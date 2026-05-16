"""
01_participants.py
==================
Section 1 of the Stan_prac.pdf document: Respondent characteristics.

Computes and prints:
- 1.1 Gender distribution
- 1.2 Country distribution (normalized)
- 1.3 Education level distribution
- 1.4 Institution type
- 1.5 Working with maps and statistical maps
- 1.6 Years of experience (mean, median, SD, min, max)
- 1.7 Self-assessed expertise level

Run:
    python 01_participants.py

All numbers rounded to 2 decimal places.
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
    """Print a frequency table with count and percentage."""
    print(f"\n--- {title} ---")
    counts = series.value_counts()
    for k, v in counts.items():
        pct = (v / N) * 100
        print(f"  {str(k):<55} {v:>4}  ({pct:>5.2f}%)")


def classify_institution(inst):
    """Classify institution type based on its name."""
    if pd.isna(inst):
        return 'Not provided'
    s = str(inst).lower()
    if 'leibniz' in s and 'university' not in s:
        return 'Research institute'
    university_markers = [
        'university', 'tum', 'tu dresden', 'ensg', 'charles', 'leeds',
        'chicago', 'oregon', 'east anglia', 'edinburgh', 'manchester',
        'cambridge', 'bournemouth', 'westminster', 'munich', 'open university',
        'heriot', 'ucla', 'ghana', 'wien', 'tu '
    ]
    if any(m in s for m in university_markers):
        return 'University / higher education'
    return 'Other'


def main():
    ratings, df = load_ratings()
    N = len(df)
    print(f"Total number of respondents: N = {N}")

    # ----- 1.1 GENDER -----
    section_header("1.1. Gender")
    print_freq_table(df.iloc[:, COL_GENDER], "Gender distribution", N)

    # ----- 1.2 COUNTRY -----
    section_header("1.2. Country of affiliation")
    countries_norm = [normalize_country(c) for c in df.iloc[:, COL_COUNTRY]]
    # In the document we use the "raw" form (e.g. 'UK', 'Germany'), but with variant merging.
    # Here we show both projections.
    print("\n--- Country distribution (normalized version) ---")
    counts_norm = pd.Series(countries_norm).value_counts()
    for k, v in counts_norm.items():
        pct = (v / N) * 100
        print(f"  {k:<25} {v:>4}  ({pct:>5.2f}%)")

    print("\n--- Country distribution (enriched version — multi-country entries separated) ---")
    print("(Note: the document preserves composite entries as separate categories.)")
    raw = df.iloc[:, COL_COUNTRY].astype(str).str.strip()
    # Replace various forms of "UK" with a uniform spelling 'UK', and Czech variants -> 'Czechia'
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

    # ----- 1.3 EDUCATION -----
    section_header("1.3. Education level")
    print_freq_table(df.iloc[:, COL_EDUCATION], "Education", N)

    # ----- 1.4 INSTITUTIONS -----
    section_header("1.4. Institution type")
    institutions = [classify_institution(i) for i in df.iloc[:, COL_INSTITUTION]]
    print_freq_table(pd.Series(institutions), "Institution type", N)

    # ----- 1.5 WORKING WITH MAPS -----
    section_header("1.5. Working with maps and statistical maps")
    print("\n--- Working with maps (overall) ---")
    maps_counts = df.iloc[:, COL_WORKS_MAPS].value_counts()
    for k, v in maps_counts.items():
        pct = (v / N) * 100
        print(f"  {str(k):<25} {v:>4}  ({pct:>5.2f}%)")

    print("\n--- Working with statistical maps ---")
    stat_counts = df.iloc[:, COL_WORKS_STAT].value_counts()
    for k, v in stat_counts.items():
        pct = (v / N) * 100
        print(f"  {str(k):<25} {v:>4}  ({pct:>5.2f}%)")

    # ----- 1.6 YEARS OF EXPERIENCE -----
    section_header("1.6. Years of experience")

    years_maps = [parse_years(v) for v in df.iloc[:, COL_YEARS_MAPS]]
    years_maps_clean = [y for y in years_maps if y is not None]

    years_stat = [parse_years(v) for v in df.iloc[:, COL_YEARS_STAT]]
    years_stat_clean = [y for y in years_stat if y is not None]

    def summarize(name, values, N_total):
        print(f"\n--- {name} ---")
        if not values:
            print("  No data")
            return
        print(f"  N (number of responses): {len(values)}/{N_total}")
        print(f"  Mean:           {np.mean(values):.2f}")
        print(f"  Median:         {np.median(values):.2f}")
        print(f"  Std. dev (SD):  {np.std(values, ddof=1):.2f}")
        print(f"  Min:            {min(values):.2f}")
        print(f"  Max:            {max(values):.2f}")

    summarize("Maps overall", years_maps_clean, N)
    summarize("Statistical maps", years_stat_clean, N)

    print("\n--- Text values (converted to numbers) ---")
    text_to_number = []
    for raw, parsed in zip(df.iloc[:, COL_YEARS_MAPS], years_maps):
        if not pd.isna(raw) and not isinstance(raw, (int, float)):
            text_to_number.append((raw, parsed))
    for raw, parsed in zip(df.iloc[:, COL_YEARS_STAT], years_stat):
        if not pd.isna(raw) and not isinstance(raw, (int, float)):
            text_to_number.append((raw, parsed))
    if text_to_number:
        # Remove duplicates preserving order
        seen = set()
        for raw, parsed in text_to_number:
            key = str(raw).strip()
            if key not in seen:
                seen.add(key)
                print(f"  '{raw}'  ->  {parsed}")
    else:
        print("  (no text values)")

    # ----- 1.7 SELF-ASSESSED EXPERTISE -----
    section_header("1.7. Self-assessed expertise level")
    expertise = df.iloc[:, COL_SELF_EXPERT].dropna().tolist()
    print(f"\n  N: {len(expertise)}")
    print(f"  Mean:   {np.mean(expertise):.2f}")
    print(f"  Median: {np.median(expertise):.2f}")
    print(f"  SD:     {np.std(expertise, ddof=1):.2f}")
    print(f"\n--- Score distribution ---")
    for lvl in [1, 2, 3, 4, 5]:
        cnt = sum(1 for e in expertise if e == lvl)
        pct = (cnt / N) * 100
        print(f"  Score {lvl}: {cnt:>4}  ({pct:>5.2f}%)")


if __name__ == "__main__":
    main()
