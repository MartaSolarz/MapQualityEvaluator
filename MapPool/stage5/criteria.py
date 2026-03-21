"""
Formal criteria definitions for statistical map selection.

Criteria are defined here as data — separate from pipeline logic.
To modify what's checked, edit FORMAL_CRITERIA or criteria_prompt.md.
ai_check.py reads both and handles the API calls.

Based on: Podejscie_teoretyczne.md, section "WYMAGANIA FORMALNE"
"""

from dataclasses import dataclass, field
import json


@dataclass
class Criterion:
    """Single formal criterion for map inclusion."""
    id: str               # e.g. "F1", "F3", "F4"
    name: str             # short Polish name
    name_en: str          # short English name (for prompts)
    description: str      # full description from Podejscie_teoretyczne.md
    check_type: str       # "automated" or "ai_vision"
    response_fields: list = field(default_factory=list)  # JSON fields expected from AI
    pass_condition: str = ""  # logical expression evaluated on response + image meta


# ---------------------------------------------------------------------------
# Formal criteria (F1–F4)
# F2 is satisfied by design (all images from MapPool = public internet sources)
# ---------------------------------------------------------------------------

FORMAL_CRITERIA = [
    Criterion(
        id="F1",
        name="Wystarczająca jakość techniczna obrazu",
        name_en="Sufficient image resolution",
        description=(
            "Obraz mapy musi być zapisany w rozdzielczości i jakości "
            "umożliwiającej jednoznaczny odczyt treści kartograficznej "
            "(symboli, legendy, granic jednostek, etykiet)."
        ),
        check_type="metadata_only",  # recorded but does not gate passes_all
        response_fields=[],
        pass_condition="",  # threshold TBD empirically after data collection
    ),
    Criterion(
        id="F3",
        name="Zgodność z przedmiotem analizy",
        name_en="Statistical map with quantitative data on admin units",
        description=(
            "Obraz musi być mapą statystyczną, która przedstawia zjawisko "
            "ilościowe (dane statystyczne) odniesione do jednostek odniesienia "
            "przestrzennego w postaci jednostek podziału administracyjnego/"
            "statystycznego (np. kraje/regiony/diecezje/obwody statystyczne)."
        ),
        check_type="ai_vision",
        response_fields=[
            "is_map",
            "is_statistical_map",
            "has_quantitative_data",
            "has_admin_units",
        ],
        pass_condition=(
            "is_map and is_statistical_map and "
            "has_quantitative_data and has_admin_units"
        ),
    ),
    Criterion(
        id="F4",
        name="Ograniczenie do wybranych metod prezentacji",
        name_en="Uses choropleth and/or proportional circle symbols",
        description=(
            "Mapa musi być wykonana co najmniej jedną z metod: kartogram "
            "(choropleth) i/lub kartodiagram kołowy (proportional/graduated "
            "symbols w kształcie koła — jednorodne lub strukturalne). "
            "Dopuszczamy dodatkową metodę nie prezentującą danych ilościowych."
        ),
        check_type="ai_vision",
        response_fields=[
            "has_choropleth",
            "has_proportional_circles",
            "has_graduated_circles",
            "has_structural_circles",
        ],
        pass_condition=(
            "has_choropleth or has_proportional_circles or "
            "has_graduated_circles or has_structural_circles"
        ),
    ),
]


def get_ai_criteria():
    """Return only criteria that need AI vision check."""
    return [c for c in FORMAL_CRITERIA if c.check_type == "ai_vision"]


def get_all_ai_response_fields():
    """Collect all response fields expected from AI across all criteria."""
    fields = []
    for c in get_ai_criteria():
        fields.extend(c.response_fields)
    return fields


def build_json_schema():
    """Build the JSON schema string for the AI prompt."""
    schema = {}
    for field_name in get_all_ai_response_fields():
        schema[field_name] = "true/false"
    schema["confidence"] = "high/medium/low"
    schema["brief_description"] = "one-sentence description of what the image shows"
    return json.dumps(schema, indent=2)


def evaluate_criterion(criterion: Criterion, context: dict) -> dict:
    """
    Evaluate a single criterion against a context dict.

    Args:
        criterion: the Criterion to check
        context: dict with all available fields (AI response + image metadata)

    Returns:
        {"criterion_id": str, "passed": bool, "reason": str}
    """
    if not criterion.pass_condition:
        return {"criterion_id": criterion.id, "passed": True, "reason": "no condition"}

    try:
        # Only allow boolean fields and numeric comparisons
        allowed = {f: context.get(f) for f in
                   criterion.response_fields + ["image_width", "image_height"]}
        passed = bool(eval(criterion.pass_condition, {"__builtins__": {}}, allowed))
    except Exception as e:
        return {
            "criterion_id": criterion.id,
            "passed": False,
            "reason": f"evaluation error: {e}",
        }

    reason = "OK" if passed else f"failed: {criterion.pass_condition}"
    return {"criterion_id": criterion.id, "passed": passed, "reason": reason}


def evaluate_all_criteria(ai_response: dict, image_meta: dict) -> dict:
    """
    Evaluate all formal criteria.

    Args:
        ai_response: parsed JSON from Claude Vision (or empty dict for automated-only)
        image_meta: dict with image_width, image_height, etc.

    Returns:
        {
            "passes_f1": bool, "passes_f3": bool, "passes_f4": bool,
            "passes_all": bool,
            "details": [{"criterion_id": ..., "passed": ..., "reason": ...}, ...]
        }
    """
    context = {**image_meta, **ai_response}
    results = []

    for criterion in FORMAL_CRITERIA:
        result = evaluate_criterion(criterion, context)
        results.append(result)

    passes = {r["criterion_id"]: r["passed"] for r in results}

    # F1 is metadata-only (threshold TBD) — does not affect passes_all
    passes_f3 = passes.get("F3", False)
    passes_f4 = passes.get("F4", False)

    return {
        "passes_f1": passes.get("F1", True),  # always True until threshold set
        "passes_f3": passes_f3,
        "passes_f4": passes_f4,
        "passes_all": passes_f3 and passes_f4,
        "details": results,
    }
