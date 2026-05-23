# Disqualifying Criteria Check for Statistical Maps

You are evaluating a **verified statistical map** (already confirmed to be a statistical map with choropleth and/or circle diagram methods). Your task is to check whether this map violates any of 10 disqualifying quality criteria.

For each criterion, respond `true` if the map **violates** the criterion (has the problem), or `false` if it does not.

**IMPORTANT:** Some criteria only apply to certain map types:
- **disq_A1** (absolute data in choropleth): Check ONLY if the map uses choropleth method. If the map is circles-only (no choropleth), set to `false`.
- **disq_A3** (missing classification info): Check ONLY if the map uses classed/graduated representation (discrete classes). If the map uses continuous/unclassed representation, set to `false`.
- **disq_V2** (overlapping symbols): Check ONLY if the map uses proportional/graduated circles. If the map is choropleth-only, set to `false`.

---

## Criterion D1: Ambiguous statistical variable (`disq_D1`)

Set `disq_D1 = true` if you **cannot determine** what statistical variable is being mapped and/or in what unit of measurement.

The map should provide enough information (through title, legend, labels, or annotations) to answer: "What exactly is being shown, and in what units?"

Examples of **violation** (true):
- Map shows colored regions but neither the title nor legend explains what the colors represent
- Legend shows color scale but no indication of the variable (e.g., just "low" to "high" without stating what)
- Title says "Map of Europe" without specifying the statistical phenomenon

Examples of **no violation** (false):
- Title says "GDP per capita in USD, 2020" — clear variable and unit
- Legend header says "Population density (persons/km²)" — sufficient
- Even without explicit units, if the variable is self-evident (e.g., "Unemployment rate (%)" in title)

---

## Criterion D2: Inconsistent enumeration units (`disq_D2`)

Set `disq_D2 = true` if the spatial units on the map are **not consistent** — i.e., different levels of administrative hierarchy are mixed within the same map **without a valid reason**.

Examples of **violation** (true):
- Some areas show country-level data, others show regional/state-level data, and the map does NOT use a standardized statistical system that explains the difference
- Part of the map uses provinces, another part uses municipalities, with no indication this is intentional
- Clearly arbitrary mixing of detail levels (e.g., half of Europe in countries, half in regions) without explanation

Examples of **no violation** (false):
- All units are at the same level (all countries, all states, all counties)
- Focus country is subdivided, surrounding countries shown as grey context (intentional for context)
- Map uses a **standardized statistical system** (e.g., NUTS) where some countries are a single unit at a given NUTS level while others are subdivided — this is NOT inconsistency, it is how the system works. For example, at NUTS-2 level, Iceland or Luxembourg may be one unit while Germany has many regions. This is consistent use of NUTS-2.
- Map shows data at a consistent administrative level but some units are visually larger/smaller due to geography (e.g., large vs small counties) — size difference alone is NOT inconsistency

---

## Criterion A1: Absolute data in choropleth (`disq_A1`)

**Only check if the map uses choropleth method.** If no choropleth, set to `false`.

Set `disq_A1 = true` if the choropleth map displays **raw counts or totals** (absolute values) instead of normalized/relative values.

Signs of absolute data in choropleth:
- Legend shows values like "0–1,000", "1,000–5,000", "5,000–10,000" with units that are raw counts (e.g., "number of people", "total sales", "cases")
- Title mentions totals: "Total population", "Number of deaths", "GDP (total)"
- No normalization indicator (no "per capita", "per km²", "%", "rate", "density", "index")

Signs of properly normalized data (no violation):
- Per capita, per 1000, per km², percentage, rate, density, index, ratio, share
- Clearly relative: "Unemployment rate", "Population density", "% of GDP"

**When in doubt:** If you cannot determine whether data is absolute or relative, set to `false` (benefit of the doubt).

---

## Criterion A2: Qualitative symbolization for quantitative data (`disq_A2`)

Set `disq_A2 = true` if the map uses **visual variables appropriate for categorical/nominal data** to represent quantitative data.

**Violation** (true):
- Quantitative data (numbers, rates, percentages) encoded using distinctly different hues (red, blue, green, yellow) at **similar lightness levels** — each region is a different "flat" color with no sense of order
- Rainbow color scheme for sequential data (spectral colors that don't imply order: red, orange, yellow, green, blue, purple all at similar intensity)
- Category-like coloring (e.g., red=agriculture, blue=industry, green=services) applied to data that is actually numerical

**No violation** (false):
- **Sequential scheme**: light-to-dark of one hue — clearly implies order (proper for quantitative)
- **Diverging scheme**: two different hues (e.g., red and blue) with **lightness/intensity varying from the extremes toward a neutral midpoint** (e.g., dark red → light pink → white → light blue → dark blue). This is a CORRECT scheme for quantitative diverging data — do NOT mark as violation just because two hues are used. The key test: does lightness/intensity change systematically? If yes → no violation.
- **Size variation** for proportional symbols (proper for quantitative)
- **Different hues with clear lightness/saturation progression** (acceptable)
- Any scheme where the visual order (light→dark or small→large) matches the data order

**Key test:** Ask yourself: "Can I tell which regions have higher vs lower values just by looking at the colors/sizes?" If YES → the symbolization implies order → no violation. If the colors look arbitrary/categorical with no implied order → violation.

---

## Criterion A3: Missing classification information (`disq_A3`)

**Only check for classed/graduated representations.** If the map uses continuous/unclassed representation, set to `false`.

Set `disq_A3 = true` if a classed map does not provide enough information to understand the classification:
- No legend showing class boundaries
- Cannot determine the number of classes
- Class boundaries are missing or ambiguous (e.g., overlapping ranges like "0-10, 10-20" without clarifying which boundary is inclusive)
- **If the map has no legend at all (L2=true), then A3 is automatically true as well** — without a legend there is no classification information

**No violation** (false):
- Legend clearly shows class ranges with boundaries
- Values are placed directly on the map (substitute for legend)
- Continuous/unclassed color gradient with min-max labels

---

## Criterion V1: Indistinguishable min/max values (`disq_V1`)

Set `disq_V1 = true` if the visual difference between the **minimum and maximum** symbols is too small to perceive clearly.

For choropleth: colors for lowest and highest class are too similar (nearly the same lightness/hue)
For circles: smallest and largest circles are nearly the same size

**No violation** (false): clear visual contrast between extremes of the data range.

---

## Criterion V2: Overlapping symbols preventing readability (`disq_V2`)

**Only check if the map uses proportional/graduated circles.** If choropleth-only, set to `false`.

Set `disq_V2 = true` if circles/diagrams overlap to such a degree that **individual symbol sizes cannot be determined** in significant portions of the map.

Look carefully at dense areas of the map where many circles are clustered together. Signs of violation:
- Circles are stacked on top of each other, hiding smaller circles behind larger ones
- In some regions, you cannot count how many circles there are because they merge visually
- The boundaries of individual circles are lost in the overlap — they form a blob
- More than ~30% of the map area with circles has severe overlap issues

**Minor overlap** in a few dense spots is acceptable — the criterion is about overlap that makes **significant portions** of the map unreadable.

**No violation** (false): most circles are individually distinguishable, overlap is limited to a few spots.

---

## Criterion L1: Missing or inadequate title (`disq_L1`)

Set `disq_L1 = true` if:
- The map has no title at all
- The title does not allow determining what phenomenon is presented

**IMPORTANT:** The **title** is a separate text element — typically placed above, below, or beside the map — that describes the subject of the map. Do NOT confuse it with:
- **Legend header/label** — text inside or above the legend box that describes the legend categories (e.g., "Population per km²" next to the color scale). This is part of the legend, not a map title.
- **Source citation** — small text like "Source: Eurostat" at the bottom.
- **Axis labels or annotations** on the map itself.

A legend header alone does NOT count as a map title, even if it describes the mapped variable. The title should be a standalone text element identifying the map as a whole.

**IMPORTANT:** Look carefully for the title — it may be **small, placed in a corner, at the very top or bottom of the image, or in a less prominent font**. Scan the entire image, including edges and margins. A title does not need to be large or centered to count. Even a small text line above or below the map that identifies the topic is a valid title.

**No violation** (false): the map has an identifiable title (separate from the legend) that communicates at minimum the subject matter (what is mapped). The title can be small, in any position. Ideal titles also include where and when, but the subject alone is sufficient.

**Edge cases:**
- Map embedded in a document/article where the figure caption serves as title → no violation (caption counts as title)
- Map with only a legend header like "GDP per capita" but no standalone title → violation (legend header ≠ title)
- Map with a large prominent text above it stating the topic → no violation (that is the title)
- Map with a small text line at the top edge like "Median age by country, 2020" → no violation (that IS the title, even if small)
- Map where the title is partly cropped but still readable → no violation

---

## Criterion L2: Missing legend (`disq_L2`)

Set `disq_L2 = true` if the map has no legend AND no other explanation of symbols/colors used.

**Exceptions** (set to false even without traditional legend):
- Values are written directly on each region/symbol on the map
- The color meaning is self-evident AND explained in title/subtitle (rare)

---

## Criterion L3: Missing data source (`disq_L3`)

Set `disq_L3 = true` if the map does not cite the source of the statistical data anywhere (in footer, caption, legend, or on the map itself).

Note: "Source: Author's own elaboration" or similar is acceptable as a source citation.

---

## Response format

Respond with a single JSON object:

```json
{json_schema}
```

Important:
- Set fields to `true` (violation detected) or `false` (no violation)
- Remember conditional criteria: A1 (choropleth only), A3 (classed only), V2 (circles only)
- `confidence`: "high" if you are certain about most assessments, "medium" if uncertain about some, "low" if guessing
- `brief_description`: summarize the key quality issues found, or "No significant quality issues detected"
