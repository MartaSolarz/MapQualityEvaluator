"""
Disqualifying criteria check using Claude Vision API.

Sends map images to Claude with the disqualifying criteria prompt,
parses structured JSON response, computes quality scores.
"""

import json
import base64
import time
from io import BytesIO
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from PIL import Image
from tqdm import tqdm

import config
import db
from criteria import build_json_schema, compute_quality_score, get_final_violations


def load_prompt_template():
    template = config.PROMPT_TEMPLATE_PATH.read_text(encoding="utf-8")
    schema = build_json_schema()
    return template.replace("{json_schema}", schema)


def _prepare_image_base64(image_path, max_dim=None):
    max_dim = max_dim or config.CLAUDE_MAX_IMAGE_DIM
    img = Image.open(image_path)
    if img.mode not in ("RGB", "L"):
        img = img.convert("RGB")
    if max(img.size) > max_dim:
        ratio = max_dim / max(img.size)
        new_size = (int(img.width * ratio), int(img.height * ratio))
        img = img.resize(new_size, Image.LANCZOS)
    buffer = BytesIO()
    img.save(buffer, format="JPEG", quality=85)
    b64 = base64.standard_b64encode(buffer.getvalue()).decode("utf-8")
    return b64, "image/jpeg"


def _build_cached_system(system_msg, prompt):
    block = {"type": "text", "text": system_msg + "\n\n" + prompt}
    if config.CLAUDE_USE_CACHE:
        block["cache_control"] = {"type": "ephemeral"}
    return [block]


def _extract_json(text):
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:])
        if "```" in text:
            text = text[:text.rfind("```")].strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    start = text.find("{")
    if start == -1:
        raise json.JSONDecodeError("No JSON object found", text, 0)
    for end in range(len(text), start, -1):
        if text[end - 1] == "}":
            try:
                return json.loads(text[start:end])
            except json.JSONDecodeError:
                continue
    for suffix in ['"}', '"}']:
        try:
            return json.loads(text[start:] + suffix)
        except json.JSONDecodeError:
            continue
    raise json.JSONDecodeError("Could not extract valid JSON", text, 0)


def _call_api(client, model, image_path, system_cached, map_methods=None):
    b64, media_type = _prepare_image_base64(image_path)

    # Build user message listing exactly which criteria to check
    criteria_list = "D1 (ambiguous variable), D2 (inconsistent units), A2 (qualitative symbolization), V1 (indistinguishable min/max), L1 (title), L2 (legend), L3 (data source)"

    has_choro = map_methods and map_methods.get("choropleth")
    has_circles = map_methods and (map_methods.get("proportional") or map_methods.get("graduated") or map_methods.get("structural"))

    if has_choro and has_circles:
        criteria_list += ", A1 (absolute data in choropleth), A3 (classification info), V2 (overlapping circles)"
        method_desc = "This map uses BOTH choropleth AND circle diagrams."
    elif has_choro:
        criteria_list += ", A1 (absolute data in choropleth), A3 (classification info)"
        method_desc = "This map uses choropleth. Set disq_V2=false (no circles)."
    elif has_circles:
        criteria_list += ", A3 (classification info), V2 (overlapping circles)"
        method_desc = "This map uses circle diagrams. Set disq_A1=false (no choropleth)."
    else:
        method_desc = ""

    user_text = f"{method_desc} Check each of these criteria independently: {criteria_list}. Respond with JSON only."

    message = client.messages.create(
        model=model,
        max_tokens=config.CLAUDE_MAX_TOKENS,
        system=system_cached,
        messages=[{
            "role": "user",
            "content": [
                {"type": "image", "source": {"type": "base64", "media_type": media_type, "data": b64}},
                {"type": "text", "text": user_text},
            ],
        }],
    )
    response_text = message.content[0].text.strip()
    parsed = _extract_json(response_text)
    cache_read = getattr(message.usage, "cache_read_input_tokens", 0) or 0
    cache_creation = getattr(message.usage, "cache_creation_input_tokens", 0) or 0
    return parsed, message.usage.input_tokens, message.usage.output_tokens, cache_read, cache_creation


def _check_single(client, model, uid, image_path, system_cached, map_methods=None):
    for attempt in range(config.CLAUDE_RETRY_ATTEMPTS):
        try:
            response, in_tok, out_tok, _, _ = _call_api(client, model, image_path, system_cached, map_methods)
            return (uid, "checked", response, in_tok, out_tok)
        except json.JSONDecodeError:
            if attempt < config.CLAUDE_RETRY_ATTEMPTS - 1:
                time.sleep(config.CLAUDE_RETRY_DELAY)
                continue
            return (uid, "error", {"error": "JSON parse failed"}, 0, 0)
        except Exception as e:
            error_str = str(e)
            if any(x in error_str.lower() for x in ["rate_limit", "429", "overloaded", "529"]):
                time.sleep(config.CLAUDE_RETRY_DELAY * (attempt + 1))
                continue
            if attempt < config.CLAUDE_RETRY_ATTEMPTS - 1:
                time.sleep(config.CLAUDE_RETRY_DELAY)
                continue
            return (uid, "error", {"error": error_str}, 0, 0)
    return (uid, "error", {"error": "max retries"}, 0, 0)


def run_disq_check(pilot=None, model_override=None, uids=None):
    try:
        import anthropic
    except ImportError:
        print("ERROR: 'anthropic' package not installed.")
        return

    if not config.ANTHROPIC_API_KEY:
        print("ERROR: ANTHROPIC_API_KEY not set.")
        return

    client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)

    if model_override:
        model = model_override
    elif pilot:
        model = config.CLAUDE_MODEL_PILOT
    else:
        model = config.CLAUDE_MODEL_BULK

    prompt = load_prompt_template()
    system_msg = (
        "You are a rigorous cartographic quality assessment expert. "
        "You evaluate verified statistical maps against disqualifying criteria "
        "that identify cartographic errors and omissions. "
        "Be thorough and precise. Respond ONLY with the requested JSON structure."
    )
    system_cached = _build_cached_system(system_msg, prompt)

    conn = db.get_connection()
    if uids:
        # Reset status so they get re-checked with current prompt
        placeholders = ",".join("?" for _ in uids)
        conn.execute(f"""
            UPDATE maps SET disq_status='pending',
                disq_D1=NULL, disq_D2=NULL, disq_A1=NULL, disq_A2=NULL, disq_A3=NULL,
                disq_V1=NULL, disq_V2=NULL, disq_L1=NULL, disq_L2=NULL, disq_L3=NULL,
                disq_confidence=NULL, disq_description=NULL, disq_raw_response=NULL,
                quality_score=NULL, violations_count=NULL
            WHERE uid IN ({placeholders})
        """, uids)
        conn.commit()
        pending = conn.execute(
            f"SELECT * FROM maps WHERE uid IN ({placeholders})", uids
        ).fetchall()
        print(f"  Reset {len(pending)} maps for re-check")
    else:
        pending = db.get_pending(conn, limit=pilot)
    print(f"  Maps to check: {len(pending):,}")
    print(f"  Model: {model}")
    print(f"  Prompt caching: enabled")

    if not pending:
        print("  Nothing to do.")
        conn.close()
        return

    mode = "pilot" if pilot else "bulk"
    run_id = db.start_run(conn, f"disq_check_{mode}", params={
        "model": model, "count": len(pending), "caching": True,
    })

    checked = 0
    errors = 0
    total_in = 0
    total_out = 0
    total_cache_read = 0

    with ThreadPoolExecutor(max_workers=config.CLAUDE_WORKERS) as executor:
        futures = {}
        for row in pending:
            path = row["local_path"]
            if not path or not Path(path).exists():
                db.update_disq_status(conn, row["uid"], "error",
                                      vlm_response={"error": "image not found"})
                errors += 1
                continue
            map_methods = {
                "choropleth": row["ai_has_choropleth"],
                "proportional": row["ai_has_proportional_circles"],
                "graduated": row["ai_has_graduated_circles"],
                "structural": row["ai_has_structural_circles"],
            }
            future = executor.submit(
                _check_single, client, model, row["uid"], path, system_cached, map_methods
            )
            futures[future] = row["uid"]

        for future in tqdm(as_completed(futures), total=len(futures),
                           desc=f"Disq check ({mode})"):
            uid, status, response, in_tok, out_tok = future.result()

            db.update_disq_status(conn, uid, status,
                                  vlm_response=response,
                                  input_tokens=in_tok, output_tokens=out_tok)

            # Compute quality score immediately
            if status == "checked":
                row_data = dict(conn.execute("SELECT * FROM maps WHERE uid=?", (uid,)).fetchone())
                violations = get_final_violations(row_data)
                q, count, _ = compute_quality_score(violations)
                db.update_quality_score(conn, uid, round(q, 2), count)
                checked += 1
            else:
                errors += 1

            total_in += in_tok
            total_out += out_tok
            conn.commit()

    cost_in = total_in / 1_000_000 * 3.0  # Sonnet input rate
    cost_out = total_out / 1_000_000 * 15.0  # Sonnet output rate
    est_cost = cost_in + cost_out

    db.finish_run(conn, run_id, processed=checked + errors, success=checked, failed=errors,
                  notes=f"checked={checked}, error={errors}, "
                        f"tokens_in={total_in}, tokens_out={total_out}, "
                        f"est_cost=${est_cost:.2f}")
    conn.close()

    print(f"\n  Results:")
    print(f"    Checked: {checked:,}")
    print(f"    Error:   {errors:,}")
    print(f"  Tokens:")
    print(f"    Input:  {total_in:,}")
    print(f"    Output: {total_out:,}")
    print(f"  Estimated cost: ${est_cost:.2f}")
