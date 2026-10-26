"""Small shared helpers (no third-party imports so they are trivially testable)."""

from __future__ import annotations

import json


def extract_json(text: str) -> dict:
    """Extract the first JSON object from an LLM response.

    Raises ValueError if no valid JSON object can be found.
    """
    if not isinstance(text, str):
        raise ValueError("LLM response was not text.")

    start = text.find("{")
    end = text.rfind("}") + 1

    if start == -1 or end <= start:
        raise ValueError("No JSON object found in LLM response.")

    try:
        obj = json.loads(text[start:end])
    except json.JSONDecodeError as exc:
        raise ValueError("The LLM did not return valid JSON.") from exc

    if not isinstance(obj, dict):
        raise ValueError("Expected a JSON object.")

    return obj
