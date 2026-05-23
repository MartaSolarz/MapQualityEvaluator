"""
Step 4: AI Formal Criteria Check using Claude Vision API.

Reads prompt from criteria_prompt.md and field definitions from criteria.py.
Sends images to Claude, parses structured JSON response, evaluates pass/fail.

Supports:
  - --pilot N: run on N images with Sonnet (test accuracy before bulk)
  - bulk mode: run on all pending with Haiku
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
from criteria import build_json_schema, evaluate_all_criteria


def load_prompt_template():
    """Load the prompt from criteria_prompt.md and insert JSON schema."""
    template = config.PROMPT_TEMPLATE_PATH.read_text(encoding="utf-8")
    schema = build_json_schema()
    return template.replace("{json_schema}", schema)


def _prepare_image_base64(image_path, max_dim=None):
    """
    Load image, resize if needed, encode as base64 JPEG.
    Returns (base64_string, media_type).
    """
    max_dim = max_dim or config.CLAUDE_MAX_IMAGE_DIM
    img = Image.open(image_path)

    # Convert to RGB if necessary (e.g., RGBA, palette mode)
    if img.mode not in ("RGB", "L"):
        img = img.convert("RGB")

    # Resize if too large
    if max(img.size) > max_dim:
        ratio = max_dim / max(img.size)
        new_size = (int(img.width * ratio), int(img.height * ratio))
        img = img.resize(new_size, Image.LANCZOS)

    buffer = BytesIO()
    img.save(buffer, format="JPEG", quality=85)
    b64 = base64.standard_b64encode(buffer.getvalue()).decode("utf-8")
    return b64, "image/jpeg"


def _build_cached_system(system_msg, prompt):
    """
    Build system message with prompt caching.
    System msg + full prompt are combined into a single cached block.
    Haiku requires >= 4096 tokens for caching; Sonnet >= 1024.
    """
    return [
        {
            "type": "text",
            "text": system_msg + "\n\n" + prompt,
            "cache_control": {"type": "ephemeral"},
        }
    ]


def _call_claude_api(client, model, image_path, system_cached):
    """
    Send one image to Claude Vision API with prompt caching.
    Returns (parsed_response_dict, input_tokens, output_tokens, cache_read, cache_creation).
    Raises on API or parse error.
    """
    b64, media_type = _prepare_image_base64(image_path)

    message = client.messages.create(
        model=model,
        max_tokens=config.CLAUDE_MAX_TOKENS,
        system=system_cached,
        messages=[{
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": media_type,
                        "data": b64,
                    },
                },
                {
                    "type": "text",
                    "text": "Evaluate this image according to the criteria above. Respond with JSON only.",
                },
            ],
        }],
    )

    response_text = message.content[0].text.strip()

    parsed = _extract_json(response_text)

    cache_read = getattr(message.usage, "cache_read_input_tokens", 0) or 0
    cache_creation = getattr(message.usage, "cache_creation_input_tokens", 0) or 0

    return parsed, message.usage.input_tokens, message.usage.output_tokens, cache_read, cache_creation


def _extract_json(text):
    """
    Robustly extract JSON from VLM response.
    Handles: markdown fences, trailing text after JSON, truncated descriptions.
    """
    text = text.strip()

    # Strip markdown code fences if present
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:])
        # Find closing fence
        if "```" in text:
            text = text[:text.rfind("```")].strip()

    # Try direct parse first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Find first { and try to find matching }
    start = text.find("{")
    if start == -1:
        raise json.JSONDecodeError("No JSON object found", text, 0)

    # Try progressively shorter substrings from the end
    for end in range(len(text), start, -1):
        if text[end - 1] == "}":
            try:
                return json.loads(text[start:end])
            except json.JSONDecodeError:
                continue

    # Last resort: truncated brief_description -- try to close it
    # Common pattern: {"key": "value", ..., "brief_description": "Some text that got cut
    truncated = text[start:]
    # Try adding closing quote + }
    for suffix in ['"}', '"}']:
        try:
            return json.loads(truncated + suffix)
        except json.JSONDecodeError:
            continue

    raise json.JSONDecodeError("Could not extract valid JSON", text, 0)


def _check_single_image(client, model, uid, image_path, image_width,
                         image_height, system_cached):
    """
    Check one image against formal criteria.
    Returns (uid, ai_status, ai_response, criteria_result, input_tokens, output_tokens, cache_read, cache_creation).
    """
    for attempt in range(config.CLAUDE_RETRY_ATTEMPTS):
        try:
            ai_response, in_tok, out_tok, cache_read, cache_creation = _call_claude_api(
                client, model, image_path, system_cached
            )

            # Evaluate criteria
            image_meta = {"image_width": image_width, "image_height": image_height}
            criteria_result = evaluate_all_criteria(ai_response, image_meta)

            status = "pass" if criteria_result["passes_all"] else "fail"
            return (uid, status, ai_response, criteria_result, in_tok, out_tok, cache_read, cache_creation)

        except json.JSONDecodeError as e:
            if attempt < config.CLAUDE_RETRY_ATTEMPTS - 1:
                time.sleep(config.CLAUDE_RETRY_DELAY)
                continue
            return (uid, "error", {"error": "JSON parse failed", "detail": str(e)}, {}, 0, 0, 0, 0)

        except Exception as e:
            error_str = str(e)
            # Rate limit — wait and retry
            if "rate_limit" in error_str.lower() or "429" in error_str:
                wait = config.CLAUDE_RETRY_DELAY * (attempt + 1)
                time.sleep(wait)
                continue
            # Overloaded — wait and retry
            if "overloaded" in error_str.lower() or "529" in error_str:
                wait = config.CLAUDE_RETRY_DELAY * (attempt + 1)
                time.sleep(wait)
                continue

            if attempt < config.CLAUDE_RETRY_ATTEMPTS - 1:
                time.sleep(config.CLAUDE_RETRY_DELAY)
                continue
            return (uid, "error", {"error": error_str}, {}, 0, 0, 0, 0)

    return (uid, "error", {"error": "max retries"}, {}, 0, 0, 0, 0)


def run_ai_check(pilot=None, model_override=None):
    """
    Run AI formal criteria check on pending images.

    Args:
        pilot: if set, process only this many images (with Sonnet for accuracy)
        model_override: force a specific model
    """
    try:
        import anthropic
    except ImportError:
        print("ERROR: 'anthropic' package not installed.")
        print("  Run: pip install anthropic")
        return

    if not config.ANTHROPIC_API_KEY:
        print("ERROR: ANTHROPIC_API_KEY not set.")
        print("  Export it: export ANTHROPIC_API_KEY='your-key-here'")
        return

    client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)

    # Select model
    if model_override:
        model = model_override
    elif pilot:
        model = config.CLAUDE_MODEL_PILOT  # Sonnet for pilot
    else:
        model = config.CLAUDE_MODEL_BULK   # Haiku for bulk

    # Load prompt and build cached system message
    prompt = load_prompt_template()
    system_msg = (
        "You are a rigorous cartographic analysis expert with deep knowledge of "
        "thematic cartography. You evaluate images for inclusion in a research "
        "dataset of statistical maps. Be strict — when in doubt, reject. "
        "Respond ONLY with the requested JSON structure."
    )
    system_cached = _build_cached_system(system_msg, prompt)

    # Get pending images
    conn = db.get_connection()
    pending = db.get_pending(conn, "ai_check", limit=pilot)
    print(f"  Images to check: {len(pending):,}")
    print(f"  Model: {model}")
    print(f"  Prompt caching: enabled (system + prompt in cached block)")

    if not pending:
        print("  Nothing to do.")
        conn.close()
        return

    mode = "pilot" if pilot else "bulk"
    run_id = db.start_run(conn, f"ai_check_{mode}", params={
        "model": model, "count": len(pending), "caching": True,
    })

    passed = 0
    failed_criteria = 0
    errors = 0
    total_input_tokens = 0
    total_output_tokens = 0
    total_cache_read = 0
    total_cache_creation = 0

    # Process with controlled concurrency
    with ThreadPoolExecutor(max_workers=config.CLAUDE_WORKERS) as executor:
        futures = {}
        for row in pending:
            image_path = row["local_path"]
            if not image_path or not Path(image_path).exists():
                db.update_ai_status(conn, row["uid"], "error",
                                    ai_response={"error": "image file not found"})
                errors += 1
                continue

            future = executor.submit(
                _check_single_image, client, model,
                row["uid"], image_path,
                row["image_width"], row["image_height"],
                system_cached
            )
            futures[future] = row["uid"]

        for future in tqdm(as_completed(futures), total=len(futures),
                           desc=f"AI check ({mode})"):
            uid, status, ai_response, criteria_result, in_tok, out_tok, cache_read, cache_creation = future.result()

            db.update_ai_status(
                conn, uid, status,
                ai_response=ai_response,
                criteria_result=criteria_result,
                input_tokens=in_tok,
                output_tokens=out_tok,
            )
            conn.commit()

            total_input_tokens += in_tok
            total_output_tokens += out_tok
            total_cache_read += cache_read
            total_cache_creation += cache_creation

            if status == "pass":
                passed += 1
            elif status == "fail":
                failed_criteria += 1
            else:
                errors += 1

    total = passed + failed_criteria + errors
    cost_in = total_input_tokens / 1_000_000
    cost_out = total_output_tokens / 1_000_000
    cost_cache_read = total_cache_read / 1_000_000
    cost_cache_write = total_cache_creation / 1_000_000

    # Cost estimate with caching
    if "haiku" in model:
        base_rate_in, base_rate_out = 1.0, 5.0
    else:
        base_rate_in, base_rate_out = 3.0, 15.0

    est_cost = (
        cost_in * base_rate_in +           # uncached input (image + short user text)
        cost_out * base_rate_out +          # output
        cost_cache_read * base_rate_in * 0.1 +   # cached reads (90% discount)
        cost_cache_write * base_rate_in * 1.25    # cache writes (25% premium)
    )

    cache_pct = total_cache_read / (total_cache_read + total_cache_creation + total_input_tokens) * 100 if (total_cache_read + total_cache_creation + total_input_tokens) > 0 else 0

    db.finish_run(conn, run_id, processed=total, success=passed,
                  failed=failed_criteria + errors,
                  notes=f"pass={passed}, fail={failed_criteria}, error={errors}, "
                        f"tokens_in={total_input_tokens}, tokens_out={total_output_tokens}, "
                        f"cache_read={total_cache_read}, cache_write={total_cache_creation}, "
                        f"est_cost=${est_cost:.2f}")
    conn.close()

    print(f"\n  Results:")
    print(f"    Pass:   {passed:,}")
    print(f"    Fail:   {failed_criteria:,}")
    print(f"    Error:  {errors:,}")
    print(f"  Tokens:")
    print(f"    Input:       {total_input_tokens:,}")
    print(f"    Output:      {total_output_tokens:,}")
    print(f"    Cache read:  {total_cache_read:,} ({cache_pct:.1f}% cache hit)")
    print(f"    Cache write: {total_cache_creation:,}")
    print(f"  Estimated cost: ${est_cost:.2f}")
