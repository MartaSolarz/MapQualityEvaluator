=============================================
  Stage 5: EuroCarto Abstract Pipeline Run
=============================================

[CHECK] Sprawdzam wymagania...
[OK] Wszystko gotowe.

=============================================
  KROK 0: Reset 14 errorów z poprzedniego pilotu
=============================================
  Zresetowano 14 errorów do pending.

=============================================
  KROK 1: Sampling 2000 nowych kandydatów
=============================================
============================================================
STEP 1: Smart Sampling (test subset)
============================================================
  Random seed: 42
Loading /Users/martasolarz/phd/MapQualityEvaluator/MapPool/stage4/results/predictions_50M.parquet...
  Loaded: 1,675,572 candidates
  After pred_proba >= 0.7: 1,343,830
/Users/martasolarz/phd/MapQualityEvaluator/MapPool/stage5/pipeline.py:125: FutureWarning: DataFrameGroupBy.apply operated on the grouping columns. This behavior is deprecated, and in a future version of pandas the grouping columns will be excluded from the operation. Either pass `include_groups=False` to exclude the groupings or explicitly select the grouping columns after groupby to silence this warning.
  .apply(lambda g: g.sample(n=min(len(g), domain_cap), random_state=seed))
  After domain cap (50): 470,916
  Stratum [0.70, 0.80): 400 sampled from 84,110
  Stratum [0.80, 0.90): 700 sampled from 110,855
  Stratum [0.90, 1.01): 900 sampled from 275,951
  Total sampled: 2,000

  Inserted: 1,900 new candidates
  Skipped:  100 duplicates

Done. Run 'validate-urls' next.

=============================================
  KROK 2: Walidacja URL (~15-20 min)
=============================================
============================================================
STEP 2: URL Validation
============================================================
  URLs to validate: 1,900
URL validation: 100%|██████████████████████████████████████████████████████████████████████████████████████████████| 1900/1900 [02:58<00:00, 10.63it/s]

  Alive: 1,240 (65.3%)
  Dead:  660 (34.7%)

Done. Run 'download' next.

=============================================
  KROK 3: Pobieranie obrazów (~10 min)
=============================================
============================================================
STEP 3: Image Download + F1 Check
============================================================
  Images to download: 1,240
Downloading: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████| 1240/1240 [01:43<00:00, 11.97it/s]

  Downloaded: 1,199 (96.7%)
  Failed:     41 (3.3%)

Done. Run 'ai-check' next.

=============================================
  KROK 4a: AI check pilot (50 obrazów, Sonnet)
=============================================
============================================================
STEP 4: AI Formal Criteria Check
============================================================
  PILOT MODE: 50 images with Sonnet
  Images to check: 50
  Model: claude-sonnet-4-20250514
AI check (pilot):  62%|███████████████████████████████████████████████████████████▌                                    | 31/50 [02:53<01:44,  5.51s/it]/Users/martasolarz/phd/MapQualityEvaluator/venv/lib/python3.11/site-packages/PIL/Image.py:945: UserWarning: Palette images with Transparency expressed in bytes should be converted to RGBA images
  warnings.warn(
AI check (pilot): 100%|████████████████████████████████████████████████████████████████████████████████████████████████| 50/50 [05:39<00:00,  6.79s/it]

  Results:
    Pass:   37
    Fail:   11
    Error:  2
  Tokens:
    Input:  196,624
    Output: 7,387
  Estimated cost: $0.70

=============================================
  CHECKPOINT: Przejrzyj wyniki pilotu
=============================================
============================================================
PIPELINE STATUS
============================================================

  Candidates:            2,100
  ─────────────────────────────
  URL alive:             1,366  (65.0%)
  URL dead/error:          734
  URL pending:               0
  ─────────────────────────────
  Downloaded:            1,323
  Download failed:          43
  Download pending:          0
  ─────────────────────────────
  Resolution (downloaded images):
    Range W:  1–7000 px  (avg 778)
    Range H:  1–5001 px  (avg 568)
    < 400px:  484
    < 600px:  914
    < 800px:  1,139
  ─────────────────────────────
  AI pass (F3+F4):         126
  AI fail:                  32
  AI error:                  2
  AI pending:            1,163
  ─────────────────────────────
  VERIFIED MAPS:           126

  Methods (among verified):
    Choropleth:           112
    Proportional circles: 17
    Graduated circles:    3
    Structural circles:   2

  Batches:
    #1 sample: 100 candidates (2026-03-08)
    #10 sample: 0 candidates (2026-03-08)
    #11 sample: 100 candidates (2026-03-08)
    #15 sample: 1,900 candidates (2026-03-15)

Uruchom viewer: python3 viewer.py
Przejrzyj ~20 wyników pass i ~10 fail w przeglądarce.

Kontynuować bulk AI check? (y/n) y
=============================================
  KROK 4b: AI check bulk (Haiku, ~20-30 min)
=============================================
============================================================
STEP 4: AI Formal Criteria Check
============================================================
  BULK MODE: all pending with Haiku
  Images to check: 1,163
  Model: claude-haiku-4-5-20251001
AI check (bulk):   4%|███▊                                                                                         | 48/1163 [03:14<2:17:58,  7.42s/it]/Users/martasolarz/phd/MapQualityEvaluator/venv/lib/python3.11/site-packages/PIL/Image.py:945: UserWarning: Palette images with Transparency expressed in bytes should be converted to RGBA images
  warnings.warn(
AI check (bulk): 100%|███████████████████████████████████████████████████████████████████████████████████████████| 1163/1163 [1:46:49<00:00,  5.51s/it]

  Results:
    Pass:   866
    Fail:   107
    Error:  190
  Tokens:
    Input:  3,908,033
    Output: 151,437
  Estimated cost: $4.67

=============================================
  WYNIK KOŃCOWY
=============================================
============================================================
PIPELINE STATUS
============================================================

  Candidates:            2,100
  ─────────────────────────────
  URL alive:             1,366  (65.0%)
  URL dead/error:          734
  URL pending:               0
  ─────────────────────────────
  Downloaded:            1,323
  Download failed:          43
  Download pending:          0
  ─────────────────────────────
  Resolution (downloaded images):
    Range W:  1–7000 px  (avg 778)
    Range H:  1–5001 px  (avg 568)
    < 400px:  484
    < 600px:  914
    < 800px:  1,139
  ─────────────────────────────
  AI pass (F3+F4):         992
  AI fail:                 139
  AI error:                192
  AI pending:                0
  ─────────────────────────────
  VERIFIED MAPS:           992

  Methods (among verified):
    Choropleth:           902
    Proportional circles: 142
    Graduated circles:    34
    Structural circles:   23

  Batches:
    #1 sample: 100 candidates (2026-03-08)
    #10 sample: 0 candidates (2026-03-08)
    #11 sample: 100 candidates (2026-03-08)
    #15 sample: 1,900 candidates (2026-03-15)

=============================================
  EKSPORT
=============================================
  Exported 992 verified maps → /Users/martasolarz/phd/MapQualityEvaluator/MapPool/stage5/results/verified_maps.parquet
  Columns: ['uid', 'url', 'domain', 'pred_proba', 'score', 'stratum', 'source', 'image_width', 'image_height', 'image_format', 'local_path', 'ai_has_choropleth', 'ai_has_proportional_circles', 'ai_has_graduated_circles', 'ai_has_structural_circles', 'ai_confidence', 'ai_description']

  Methods:
    Choropleth:           902
    Proportional circles: 142
    Graduated circles:    34
    Structural circles:   23
  Domains: 967 unique
  pred_proba: 0.701 – 1.000

=============================================
  STATYSTYKI DO ABSTRAKTU
=============================================
==================================================
LICZBY DO ABSTRAKTU:
==================================================
Kandydaci:          2,100
URL alive:          1,366 (65.0%)
URL dead:           734 (35.0%)
Pobrane:            1,323
AI pass:            992
AI fail:            139
AI error:           192
VERIFIED:           992 (75.0% of downloaded)

Metody (wśród verified):
  Choropleth:       902
  Proportional:     142
  Graduated:        34
  Structural:       23
  Unikalne domeny:  967

Powody odrzucenia (F3 sub-criteria):
  Nie mapa:             10
  Nie stat. map:        122
  Nie ilościowe:        0
  Nie admin units:      0

