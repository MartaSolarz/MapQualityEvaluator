"""
data_loader.py
==============
Shared data loading module for all analytical scripts.

Contains:
- definition of 10 disqualification criteria (ID, name, category, Excel column)
- load_ratings() function — loads the Excel file and returns a 25x10 ratings matrix + metadata
- normalize_country() function — merges country name variants (UK / United Kingdom / Uk -> UK)
- classify_school() function — classifies a respondent into a cartographic school (UK/CZ/DE-speak/Other)
  according to the "first-listed" rule (the first listed country determines the school)
- parse_years() function — converts text values ("A year and still counting", "Five years",
  "4 years") to numbers

All other modules import this file, ensuring that loading and classification
are consistent across analyses.
"""
import pandas as pd
import numpy as np
import re

# Default path to the responses file
DEFAULT_PATH = 'survey_responses.xlsx'

# Definition of 10 disqualification criteria
# col_idx: column number in the Excel file (0-indexed)
# id: criterion identifier
# name_en: English name
# category: category according to the theoretical model
CRITERIA = [
    {'id': 'D1', 'col_idx': 2,  'category': 'DATA',
     'name_en': 'Ambiguity of the statistical variable (thematic data)',
     'name_short_en': 'Variable ambiguity'},
    {'id': 'D2', 'col_idx': 3,  'category': 'DATA',
     'name_en': 'Lack of consistency in the selection of division units across the entire map',
     'name_short_en': 'Inconsistent units'},
    {'id': 'A1', 'col_idx': 4,  'category': 'DATA ANALYSIS & TRANSFORM',
     'name_en': 'Absolute data used in a choropleth map',
     'name_short_en': 'Absolute data in choropleth'},
    {'id': 'A2', 'col_idx': 5,  'category': 'DATA ANALYSIS & TRANSFORM',
     'name_en': 'Applying symbolization appropriate for qualitative data to quantitative data',
     'name_short_en': 'Qualitative symbolization for quantitative data'},
    {'id': 'A3', 'col_idx': 6,  'category': 'DATA ANALYSIS & TRANSFORM',
     'name_en': 'Lack of information enabling understanding of the data classification method',
     'name_short_en': 'No classification info'},
    {'id': 'V1', 'col_idx': 7,  'category': 'VISUALIZATION',
     'name_en': 'Lack of distinguishability between minimum and maximum values',
     'name_short_en': 'No min-max distinguishability'},
    {'id': 'V2', 'col_idx': 8,  'category': 'VISUALIZATION',
     'name_en': 'Overlapping diagrams preventing readability',
     'name_short_en': 'Overlapping diagrams'},
    {'id': 'L1', 'col_idx': 9,  'category': 'LAYOUT',
     'name_en': 'Missing map title',
     'name_short_en': 'Missing map title'},
    {'id': 'L2', 'col_idx': 10, 'category': 'LAYOUT',
     'name_en': 'Missing legend',
     'name_short_en': 'Missing legend'},
    {'id': 'L3', 'col_idx': 11, 'category': 'LAYOUT',
     'name_en': 'Missing data source information',
     'name_short_en': 'Missing data source'},
]

# Short lists used as keys
CRIT_IDS = [c['id'] for c in CRITERIA]
CRIT_INDEX = {c['id']: i for i, c in enumerate(CRITERIA)}

# Demographic columns (0-indexed)
COL_TIMESTAMP   = 0
COL_CONSENT     = 1
COL_GENDER      = 12
COL_COUNTRY     = 13
COL_INSTITUTION = 14
COL_EDUCATION   = 15
COL_WORKS_MAPS  = 16
COL_YEARS_MAPS  = 17
COL_WORKS_STAT  = 18
COL_YEARS_STAT  = 19
COL_SELF_EXPERT = 20


def load_ratings(path=DEFAULT_PATH):
    """
    Load the Excel file and return the ratings matrix + raw data frame.

    Returns:
        ratings: np.ndarray of shape (N, 10) — respondent ratings for each of 10 criteria
        df: pd.DataFrame — full data frame (for extracting demographic columns)
    """
    df = pd.read_excel(path)
    n = len(df)
    ratings = np.zeros((n, 10))
    for i, crit in enumerate(CRITERIA):
        ratings[:, i] = df.iloc[:, crit['col_idx']].values
    return ratings, df


def normalize_country(c):
    """
    Normalize various country name variants to a canonical form.

    Examples:
        'Uk', 'UK', 'United Kingdom', 'United Kingdom ' -> 'United Kingdom'
        'CZ', 'Czechia', 'Czech Republic' -> 'Czechia'
    """
    if pd.isna(c):
        return 'Unknown'
    c = str(c).strip()
    if c in ('CZ', 'Czechia', 'Czech Republic'):
        return 'Czechia'
    if c.upper() in ('UK', 'UNITED KINGDOM'):
        return 'United Kingdom'
    if 'United Kingdom' in c and ',' in c:
        return 'Multi-UK'
    if c.upper() == 'USA':
        return 'USA'
    if c.upper() == 'FRANCE':
        return 'France'
    return c


def classify_school(country_raw):
    """
    Classify a respondent into a cartographic school according to the 'first-listed' rule.

    Rule: if a respondent listed several countries (separated by comma or slash),
    the first one decides (interpreted as the primary professional affiliation).

    Categories:
        UK       — first country: UK or United Kingdom
        CZ       — first country: Czechia, Czech Republic, or CZ
        DE-speak — first country: Germany, Switzerland, or Austria (DACH)
        Other    — remaining (USA, France, Ghana, Nigeria, ...)
    """
    if pd.isna(country_raw):
        return 'Other'
    s = str(country_raw).strip()
    # Select the first country (split by comma or slash)
    first = s.split(',')[0].split('/')[0].strip().lower()
    if first in ('uk', 'united kingdom'):
        return 'UK'
    if first in ('cz', 'czechia', 'czech republic'):
        return 'CZ'
    if first in ('germany', 'switzerland', 'austria'):
        return 'DE-speak'
    return 'Other'


def parse_years(val):
    """
    Convert text experience values to numbers.

    Mapping rules:
        'A year and still counting' -> 1
        'Five years', 'Ten years', etc. -> corresponding numbers
        '4 years', '12.5' -> corresponding number
        Numeric value -> unchanged
        NaN -> None
    """
    if pd.isna(val):
        return None
    if isinstance(val, (int, float)):
        return float(val)
    s = str(val).lower().strip()
    if 'a year' in s or 'one year' in s:
        return 1.0
    if 'two' in s:
        return 2.0
    if 'three' in s:
        return 3.0
    if 'four' in s:
        return 4.0
    if 'five' in s:
        return 5.0
    if 'six' in s:
        return 6.0
    if 'seven' in s:
        return 7.0
    if 'eight' in s:
        return 8.0
    if 'nine' in s:
        return 9.0
    if 'ten' in s:
        return 10.0
    m = re.search(r'(\d+(\.\d+)?)', s)
    if m:
        return float(m.group(1))
    return None


def get_schools(df):
    """Return a list of assigned schools for each respondent."""
    return [classify_school(c) for c in df.iloc[:, COL_COUNTRY]]


def get_countries_normalized(df):
    """Return a list of normalized countries for each respondent."""
    return [normalize_country(c) for c in df.iloc[:, COL_COUNTRY]]


if __name__ == "__main__":
    # Smoke-test: verify that loading works correctly
    ratings, df = load_ratings()
    print(f"Number of respondents (N): {len(df)}")
    print(f"Ratings matrix shape: {ratings.shape}")
    print(f"List of criterion identifiers: {CRIT_IDS}")
    print(f"\nSchool classification:")
    schools = get_schools(df)
    print(pd.Series(schools).value_counts().to_dict())
