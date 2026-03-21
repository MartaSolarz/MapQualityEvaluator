# Stage 5: Selekcja i weryfikacja map statystycznych

## Cel
Przefiltrować ~1.67M kandydatów z `predictions_50M.parquet` do kilku tysięcy zweryfikowanych map statystycznych spełniających kryteria formalne (F1-F4), gotowych do oceny jakości.

## Pipeline

```
predictions_50M.parquet (1.67M)
  → [1] Smart Sampling (per-domain cap, stratyfikacja po pred_proba) → ~20-25k
  → [2] URL Validation (parallel HEAD requests) → ~11-13k alive
  → [3] Download + record resolution (F1 metadata) → ~10-12k images
  → [4] AI Formal Criteria check (Claude Vision: F3+F4) → ~5-7k pass
  → [5] Zapis lokalny → ręczny upload GDrive / rclone
```

Pipeline jest **iteracyjny** — można wielokrotnie doładowywać nowe mapy.
Każdy krok jest **resumowalny** — SQLite śledzi status per-rekord.

## Architektura plików

```
MapPool/stage5/
├── config.py              # Parametry pipeline (progi, ścieżki, workers)
├── criteria.py            # Definicje kryteriów formalnych (F1-F4) — dataclass
├── criteria_prompt.md     # Czytelny template prompta AI (edytowalny osobno)
├── db.py                  # SQLite: schemat, helpers, queries statusów
├── pipeline.py            # Główny skrypt: sampling, URL validation, download
├── ai_check.py            # Claude Vision API: ładuje prompt + criteria, ewaluuje
├── run_pipeline.py        # CLI entry point (argparse)
└── PLAN.md                # Ten plik
```

## System kryteriów formalnych

Kryteria żyją **osobno od logiki pipeline** — łatwa modyfikacja bez ruszania kodu:

### `criteria.py` — CO sprawdzamy
```python
@dataclass
class Criterion:
    id: str              # "F1", "F3", "F4"
    name: str            # nazwa
    description: str     # pełny opis z Podejscie_teoretyczne.md
    check_type: str      # "automated" | "ai_vision"
    response_fields: list  # pola JSON oczekiwane od AI
    pass_condition: str  # wyrażenie logiczne (eval)

FORMAL_CRITERIA = [
    Criterion(id="F1", ..., check_type="metadata_only",
              pass_condition=""),  # threshold TBD empirically
    Criterion(id="F3", ..., check_type="ai_vision",
              response_fields=["is_map", "is_statistical_map", "has_quantitative_data", "has_admin_units"],
              pass_condition="is_map and is_statistical_map and has_quantitative_data and has_admin_units"),
    Criterion(id="F4", ..., check_type="ai_vision",
              response_fields=["has_choropleth", "has_proportional_circles", "has_graduated_circles", "has_structural_circles"],
              pass_condition="has_choropleth or has_proportional_circles or has_graduated_circles or has_structural_circles"),
]
```

### `criteria_prompt.md` — JAK pytamy AI
Czytelny Markdown z opisem zadania, definicjami F3/F4, przykładami pozytywnych/negatywnych, schematem JSON odpowiedzi. Edytowalny bez ruszania Pythona.

### Jak to współgra
```
criteria.py          → definiuje pola odpowiedzi AI + warunki pass/fail
criteria_prompt.md   → definiuje treść prompta (czytelny tekst)
ai_check.py          → łączy oba: ładuje prompt, wysyła do API, parsuje, ewaluuje
```

Modyfikacja:
- Zmiana progu F1 (np. 300px) → edycja pass_condition w criteria.py
- Dodanie kryterium F5 → nowy Criterion + sekcja w prompt.md
- Zmiana sposobu pytania AI → edycja criteria_prompt.md
- Zero zmian w ai_check.py

## Konfiguracja (config.py)

### Sampling
- `PRED_PROBA_MIN = 0.70` (zaczynamy wyżej, obniżamy jeśli za mało map)
- `DOMAIN_CAP = 50` (max map per domena — zapobiega dominacji)
- Stratyfikacja po pred_proba: low/med-low/med-high/high
- `SAMPLE_TARGET = 25000`

### Techniczne
- URL validation: timeout 8s, 30 workers
- Download: timeout 15s, 15 workers
- AI: model claude-haiku-4-5-20251001, resize 1024px, direct API (5 workers)
- Pilot: Sonnet na 100 obrazach → weryfikacja accuracy → bulk Haiku

## SQLite schema

### Tabela `candidates`
```
uid TEXT PRIMARY KEY
url TEXT NOT NULL
domain TEXT NOT NULL
pred_proba REAL NOT NULL
score REAL
stratum TEXT               -- 'high', 'medium_high', 'medium_low', 'low'
batch_id INTEGER           -- FK → pipeline_runs.id
source TEXT                -- 'sample', 'manual', 'domain_add'

-- Krok 2: URL validation
url_status TEXT DEFAULT 'pending'   -- pending/alive/dead/error
url_http_code INTEGER
url_checked_at TEXT

-- Krok 3: Download
download_status TEXT DEFAULT 'pending'  -- pending/success/failed/too_small/not_image
image_width INTEGER
image_height INTEGER
image_format TEXT
local_path TEXT

-- Krok 4: AI check
ai_status TEXT DEFAULT 'pending'        -- pending/pass/fail/error
ai_is_map INTEGER                       -- 0/1
ai_is_statistical_map INTEGER
ai_has_quantitative_data INTEGER
ai_has_admin_units INTEGER
ai_has_choropleth INTEGER
ai_has_proportional_circles INTEGER
ai_has_graduated_circles INTEGER
ai_has_structural_circles INTEGER
ai_confidence TEXT                      -- high/medium/low
ai_description TEXT
ai_raw_response TEXT

-- Wyniki
passes_f1 INTEGER                       -- 0/1 (obliczone z evaluate_criteria)
passes_f3 INTEGER
passes_f4 INTEGER
passes_all INTEGER

-- Tracking
created_at TEXT DEFAULT (datetime('now'))
```

### Tabela `pipeline_runs`
```
id INTEGER PRIMARY KEY AUTOINCREMENT
step TEXT                -- 'sample', 'url_check', 'download', 'ai_check'
params TEXT              -- JSON z parametrami tego runu
started_at TEXT
finished_at TEXT
items_processed INTEGER
items_success INTEGER
```

## CLI

```bash
# --- Standardowy przebieg ---
python3 run_pipeline.py sample                      # Krok 1
python3 run_pipeline.py validate-urls               # Krok 2
python3 run_pipeline.py download                    # Krok 3
python3 run_pipeline.py ai-check --pilot 100        # Pilot: Sonnet, 100 obrazów
python3 run_pipeline.py ai-check                    # Pełny: Haiku
python3 run_pipeline.py status                      # Postęp
python3 run_pipeline.py export                      # → verified_maps.parquet

# --- Doładowywanie ---
python3 run_pipeline.py sample --pred-proba-min 0.5 --domain-cap 100
    # Nowi kandydaci dodani (INSERT OR IGNORE), nowy batch_id

python3 run_pipeline.py add-urls urls.txt
    # Ręczne URL-e (jeden per linia)

python3 run_pipeline.py add-from-domain "ourworldindata.org" --limit 200
    # Więcej map z domeny dającej dobre wyniki

python3 run_pipeline.py sample --exclude-domains namespedia.com,ville-data.com

# Po doładowaniu: validate-urls → download → ai-check (przetwarza TYLKO nowe)
```

## Iteracyjny workflow — przykład

```
1. Pierwszy run:
   sample → validate-urls → download → ai-check --pilot 100
   → pilot OK (accuracy >90%) → ai-check (pełny) → export
   → wynik: 5200 map

2. Za mało kartodiagramów kołowych:
   add-from-domain "app.23degrees.io" --limit 300
   → validate-urls → download → ai-check → export
   → wynik: 5400 map, lepszy rozkład metod

3. Ręczne znaleziska:
   add-urls znalezione_mapy.txt
   → validate-urls → download → ai-check
```

## Szacunki

### Czas implementacji: ~8-10h
| Komponent | Czas |
|-----------|------|
| config.py + criteria.py + criteria_prompt.md | 1.5h |
| db.py (schemat, helpers) | 1.5h |
| pipeline.py (sampling, URL check, download) | 2.5h |
| ai_check.py (Claude API, parsing, ewaluacja) | 1.5h |
| run_pipeline.py (CLI, subkomendy) | 1h |
| Testing + debugging | 1.5h |

### Czas uruchomienia: ~2-4h
| Krok | Czas |
|------|------|
| Sampling | 2-3 min |
| URL validation (25k, 30 workers) | 15-20 min |
| Download (13k, 15 workers) | 30-45 min |
| AI check pilot (100, Sonnet) | 5 min |
| AI check full (12k, Haiku, 5 workers) | 40-60 min |

### Koszt AI: ~$15-20
- Pilot 100 obrazów × Sonnet: ~$1
- Bulk ~12k obrazów × Haiku: ~$12-15
- Ewentualne powtórki/doładowania: ~$3-5

## Prereqs
1. **Konto Anthropic** → console.anthropic.com → wygenerować ANTHROPIC_API_KEY
2. **Google Drive** → wybrać folder docelowy, po pipeline `rclone sync` lub ręczny upload
3. `pip install anthropic` (reszta zależności już w requirements.txt)

## Krytyczna ocena planu

### Mocne strony
- Separacja kryteriów od kodu → łatwa modyfikacja F1-F4
- Per-domain cap → jedyny sposób na różnorodność z tego datasetu
- Iteracyjne doładowywanie → nie trzeba z góry przewidywać idealnych parametrów
- SQLite resumowalność → nie tracisz pracy

### Ryzyka
- **Haiku accuracy**: może nie wystarczyć do rozróżniania niuansów wizualnych (kartodiagram kołowy vs pie chart). Mitygacja: pilot 100 z Sonnet, porównanie.
- **pred_proba 0.70**: może odciąć ciekawe mapy niskiej jakości. Mitygacja: iteracyjne doładowywanie z niższym progiem.
- **~5-7k map w pierwszym runie**: może być za dużo lub za mało. Dopiero po runie będzie wiadomo.
- **Martwe URL-e** (~46%): strata połowy kandydatów. Nie do uniknięcia.
