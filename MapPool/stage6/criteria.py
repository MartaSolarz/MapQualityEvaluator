"""
Disqualifying criteria definitions for statistical map quality assessment.

10 criteria (D1, D2, A1-A3, V1-V2, L1-L3) with expert weights from survey.
Each violation lowers the quality score but does not eliminate the map.
"""

from dataclasses import dataclass
import json


@dataclass
class DisqualifyingCriterion:
    id: str             # e.g. "D1", "A1", "L2"
    category: str       # DATA, DATA_ANALYSIS, VISUALIZATION, LAYOUT
    name: str           # short Polish name
    name_en: str        # short English name (for prompts)
    description: str    # what constitutes a violation
    weight: float       # expert weight from survey (0-1)
    response_field: str # JSON field name in VLM response
    condition: str      # when this criterion applies ("always", "choropleth", "graduated", "circles")


CRITERIA = [
    DisqualifyingCriterion(
        id="D1", category="DATA",
        name="Niejednoznaczność zmiennej statystycznej",
        name_en="Ambiguous statistical variable",
        description=(
            "Cannot determine what is being mapped or in what unit of measure. "
            "The map lacks sufficient information (in title, legend, labels, or description) "
            "to identify the statistical variable and its measurement unit."
        ),
        weight=0.87, response_field="disq_D1", condition="always",  # mean=4.48
    ),
    DisqualifyingCriterion(
        id="D2", category="DATA",
        name="Brak spójności jednostek podziału",
        name_en="Inconsistent enumeration units",
        description=(
            "The spatial units on the map are not consistent — e.g., some areas show "
            "country-level data while others show regional-level data within the same map. "
            "The level of spatial detail should be uniform across the entire map."
        ),
        weight=0.51, response_field="disq_D2", condition="always",  # mean=3.04
    ),
    DisqualifyingCriterion(
        id="A1", category="DATA_ANALYSIS",
        name="Dane bezwzględne w kartogramie",
        name_en="Absolute data in choropleth",
        description=(
            "A choropleth map displays raw counts or totals (absolute values) instead of "
            "normalized/relative values (rates, percentages, densities, per capita). "
            "This is a critical cartographic error because larger areas appear to have "
            "higher values simply due to their size."
        ),
        weight=0.80, response_field="disq_A1", condition="choropleth",  # mean=4.20
    ),
    DisqualifyingCriterion(
        id="A2", category="DATA_ANALYSIS",
        name="Symbolizacja jakościowa dla danych ilościowych",
        name_en="Qualitative symbolization for quantitative data",
        description=(
            "The map uses a visual variable appropriate for qualitative/nominal data "
            "(e.g., different hues without lightness progression) to represent quantitative "
            "or ordinal data. Quantitative data should use ordered visual variables "
            "(lightness, size) not categorical ones (hue, shape)."
        ),
        weight=0.59, response_field="disq_A2", condition="always",  # mean=3.36
    ),
    DisqualifyingCriterion(
        id="A3", category="DATA_ANALYSIS",
        name="Brak informacji o klasyfikacji danych",
        name_en="Missing classification information",
        description=(
            "A classed/graduated map does not provide information about how data was "
            "classified — missing class boundaries, number of classes cannot be determined "
            "from the legend. Applies only to classed choropleth and graduated symbols "
            "(not to unclassed/continuous representations)."
        ),
        weight=0.69, response_field="disq_A3", condition="graduated",  # mean=3.76
    ),
    DisqualifyingCriterion(
        id="V1", category="VISUALIZATION",
        name="Brak rozróżnialności wartości min/max",
        name_en="Indistinguishable min/max values",
        description=(
            "The visual difference between the symbol for the minimum value and the "
            "symbol for the maximum value is too small to perceive. Colors are too similar, "
            "or circle sizes are nearly identical. The reader cannot tell apart the "
            "extremes of the data range."
        ),
        weight=0.73, response_field="disq_V1", condition="always",  # mean=3.92
    ),
    DisqualifyingCriterion(
        id="V2", category="VISUALIZATION",
        name="Nakładanie się diagramów uniemożliwiające odczyt",
        name_en="Overlapping symbols preventing readability",
        description=(
            "Proportional or graduated symbols overlap to such a degree that individual "
            "symbol sizes cannot be determined. The map becomes unreadable due to "
            "excessive visual clutter from overlapping circles/diagrams."
        ),
        weight=0.84, response_field="disq_V2", condition="circles",  # mean=4.36
    ),
    DisqualifyingCriterion(
        id="L1", category="LAYOUT",
        name="Brak tytułu lub tytuł nieadekwatny",
        name_en="Missing or inadequate title",
        description=(
            "The map has no title, or the title does not allow the reader to determine "
            "what phenomenon is being presented. A good title should communicate "
            "the subject (what), location (where), and time period (when)."
        ),
        weight=0.64, response_field="disq_L1", condition="always",  # mean=3.56
    ),
    DisqualifyingCriterion(
        id="L2", category="LAYOUT",
        name="Brak legendy",
        name_en="Missing legend",
        description=(
            "The map does not contain a legend or any other explanation of the symbols/colors "
            "used. Without a legend, the reader cannot interpret what the colors, shades, "
            "or symbol sizes represent. Note: if values are placed directly on the map "
            "(e.g., numbers in each region), this may substitute for a legend."
        ),
        weight=0.86, response_field="disq_L2", condition="always",  # mean=4.44
    ),
    DisqualifyingCriterion(
        id="L3", category="LAYOUT",
        name="Brak informacji o źródle danych",
        name_en="Missing data source information",
        description=(
            "The map does not cite the source of the statistical data. "
            "For a statistical map, the origin of the data is essential for credibility "
            "and verification."
        ),
        weight=0.65, response_field="disq_L3", condition="always",  # mean=3.60
    ),
]

# Pre-computed
CRITERIA_BY_ID = {c.id: c for c in CRITERIA}
WEIGHTS = {c.id: c.weight for c in CRITERIA}
SUM_WEIGHTS = sum(WEIGHTS.values())
ALL_IDS = [c.id for c in CRITERIA]
RESPONSE_FIELDS = [c.response_field for c in CRITERIA]


def build_json_schema():
    """Build JSON schema string for the VLM prompt."""
    schema = {}
    for c in CRITERIA:
        schema[c.response_field] = "true/false"
    schema["confidence"] = "high/medium/low"
    schema["brief_description"] = "one-sentence summary of quality issues found (or 'no issues')"
    return json.dumps(schema, indent=2)


def compute_quality_score(violations, use_human=True):
    """
    Compute quality score Q = Q_MAX * (1 - P_norm).

    Args:
        violations: dict with keys D1..L3, values True (violation) / False (OK) / None (not checked)
        use_human: if True, human overrides take priority

    Returns:
        (quality_score, violations_count, penalty_norm)
    """
    import config
    count = 0
    penalty = 0.0
    for cid in ALL_IDS:
        v = violations.get(cid)
        if v is None:
            continue
        if v:
            count += 1
            penalty += WEIGHTS[cid]

    p_norm = penalty / SUM_WEIGHTS if SUM_WEIGHTS > 0 else 0
    q = config.Q_MAX * (1 - p_norm)
    return q, count, p_norm


def get_final_violations(row):
    """Get final violation flags: human override takes priority over VLM."""
    violations = {}
    for cid in ALL_IDS:
        human = row.get(f"human_{cid}")
        vlm = row.get(f"disq_{cid}")
        violations[cid] = bool(human) if human is not None else bool(vlm) if vlm is not None else None
    return violations
