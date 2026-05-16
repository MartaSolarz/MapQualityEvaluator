"""
data_loader.py
==============
Wspólny moduł wczytywania danych dla wszystkich skryptów analitycznych.

Zawiera:
- definicję 10 kryteriów dyskwalifikujących (ID, nazwa, kategoria, kolumna w pliku Excel)
- funkcję load_ratings() — wczytuje plik Excel i zwraca macierz ocen 25×10 + metadane
- funkcję normalize_country() — łączy warianty zapisu krajów (UK / United Kingdom / Uk → UK)
- funkcję classify_school() — klasyfikuje respondenta do szkoły kartograficznej (UK/CZ/DE-speak/Other)
  zgodnie z regułą "first-listed" (pierwszy wymieniony kraj określa szkołę)
- funkcję parse_years() — sprowadza wartości tekstowe ("A year and still counting", "Five years",
  "4 years") do liczb

Wszystkie pozostałe moduły importują ten plik, dzięki czemu wczytywanie i klasyfikacja
są spójne między analizami.
"""
import pandas as pd
import numpy as np
import re

# Domyślna ścieżka do pliku z odpowiedziami
DEFAULT_PATH = 'survey_responses.xlsx'

# Definicja 10 kryteriów dyskwalifikujących
# col_idx: numer kolumny w pliku Excel (0-indeksowany)
# id: identyfikator kryterium
# name_pl: polska nazwa
# category: kategoria zgodna z modelem teoretycznym
CRITERIA = [
    {'id': 'D1', 'col_idx': 2,  'category': 'DATA',
     'name_pl': 'Niejednoznaczność zmiennej statystycznej (danych tematycznych)',
     'name_short_pl': 'Niejednoznaczność zmiennej'},
    {'id': 'D2', 'col_idx': 3,  'category': 'DATA',
     'name_pl': 'Brak spójności doboru jednostek podziału na całej mapie',
     'name_short_pl': 'Brak spójności jednostek'},
    {'id': 'A1', 'col_idx': 4,  'category': 'DATA ANALYSIS & TRANSFORM',
     'name_pl': 'Dane bezwzględne użyte w kartogramie',
     'name_short_pl': 'Dane bezwzględne w kartogramie'},
    {'id': 'A2', 'col_idx': 5,  'category': 'DATA ANALYSIS & TRANSFORM',
     'name_pl': 'Zastosowanie symbolizacji właściwej dla danych jakościowych do danych ilościowych',
     'name_short_pl': 'Symbolika jakościowa dla ilościowych'},
    {'id': 'A3', 'col_idx': 6,  'category': 'DATA ANALYSIS & TRANSFORM',
     'name_pl': 'Brak informacji umożliwiającej zrozumienie sposobu klasyfikacji danych',
     'name_short_pl': 'Brak inf. o klasyfikacji'},
    {'id': 'V1', 'col_idx': 7,  'category': 'VISUALIZATION',
     'name_pl': 'Brak rozróżnialności wartości minimalnych i maksymalnych',
     'name_short_pl': 'Brak rozróżnialności min-max'},
    {'id': 'V2', 'col_idx': 8,  'category': 'VISUALIZATION',
     'name_pl': 'Nakładanie się diagramów uniemożliwiające odczyt',
     'name_short_pl': 'Nakładanie się diagramów'},
    {'id': 'L1', 'col_idx': 9,  'category': 'LAYOUT',
     'name_pl': 'Brak tytułu mapy',
     'name_short_pl': 'Brak tytułu mapy'},
    {'id': 'L2', 'col_idx': 10, 'category': 'LAYOUT',
     'name_pl': 'Brak legendy',
     'name_short_pl': 'Brak legendy'},
    {'id': 'L3', 'col_idx': 11, 'category': 'LAYOUT',
     'name_pl': 'Brak informacji o źródle danych',
     'name_short_pl': 'Brak źródła danych'},
]

# Krótkie listy używane jako klucze
CRIT_IDS = [c['id'] for c in CRITERIA]
CRIT_INDEX = {c['id']: i for i, c in enumerate(CRITERIA)}

# Kolumny demograficzne (0-indeksowane)
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
    Wczytaj plik Excel i zwróć macierz ocen + ramkę z surowymi danymi.

    Returns:
        ratings: np.ndarray kształtu (N, 10) — oceny respondentów dla każdego z 10 kryteriów
        df: pd.DataFrame — pełna ramka danych (do wyciągania kolumn demograficznych)
    """
    df = pd.read_excel(path)
    n = len(df)
    ratings = np.zeros((n, 10))
    for i, crit in enumerate(CRITERIA):
        ratings[:, i] = df.iloc[:, crit['col_idx']].values
    return ratings, df


def normalize_country(c):
    """
    Sprowadź różne warianty zapisu kraju do formy kanonicznej.

    Przykłady:
        'Uk', 'UK', 'United Kingdom', 'United Kingdom ' → 'United Kingdom'
        'CZ', 'Czechia', 'Czech Republic' → 'Czechia'
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
    Sklasyfikuj respondenta do szkoły kartograficznej zgodnie z regułą 'first-listed'.

    Reguła: jeśli respondent wymienił kilka krajów (oddzielone przecinkiem lub slashem),
    decyduje pierwszy z nich (interpretowany jako podstawowa afiliacja zawodowa).

    Kategorie:
        UK       — pierwszy kraj: UK lub United Kingdom
        CZ       — pierwszy kraj: Czechia, Czech Republic lub CZ
        DE-speak — pierwszy kraj: Germany, Switzerland lub Austria (DACH)
        Other    — pozostałe (USA, Francja, Ghana, Nigeria, …)
    """
    if pd.isna(country_raw):
        return 'Other'
    s = str(country_raw).strip()
    # Wybierz pierwszy kraj (split po przecinku lub slash-u)
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
    Sprowadź tekstową wartość lat doświadczenia do liczby.

    Reguły mapowania:
        'A year and still counting' → 1
        'Five years', 'Ten years', itd. → odpowiednie liczby
        '4 years', '12.5' → odpowiednia liczba
        Wartość liczbowa → bez zmian
        NaN → None
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
    """Zwróć listę przypisanych szkół dla każdego respondenta."""
    return [classify_school(c) for c in df.iloc[:, COL_COUNTRY]]


def get_countries_normalized(df):
    """Zwróć listę znormalizowanych krajów dla każdego respondenta."""
    return [normalize_country(c) for c in df.iloc[:, COL_COUNTRY]]


if __name__ == "__main__":
    # Smoke-test: czy wczytanie działa poprawnie
    ratings, df = load_ratings()
    print(f"Liczba respondentów (N): {len(df)}")
    print(f"Kształt macierzy ocen: {ratings.shape}")
    print(f"Lista identyfikatorów kryteriów: {CRIT_IDS}")
    print(f"\nKlasyfikacja szkół:")
    schools = get_schools(df)
    print(pd.Series(schools).value_counts().to_dict())