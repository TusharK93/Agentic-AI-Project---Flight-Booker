"""
Layered guardrails for the travel planner.

Design: cheap deterministic checks first (this file), LLM topic classifier second
(supervisor). Nothing here calls an LLM, so it is fast, free and unit-testable.

Layers
------
1. INPUT   screen_input()          length, prompt-injection, PII redaction
2. TOOL    tooling.check_tool_call per-agent tool allowlist, arg validation, call cap
           sanitize_tool_output()  indirect prompt-injection in web/tool results
3. OUTPUT  screen_output()         secret/card redaction, fake "booked" claims,
                                   flight numbers not present in tool data,
                                   missing "estimate" disclaimer
4. HUMAN   human_approval node     (already in graph.py)
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any

MAX_INPUT_CHARS = 2000
MAX_TOOL_OUTPUT_CHARS = 20000

INJECTION_PLACEHOLDER = "[removed: suspected prompt injection]"


# ============================================================
# Events
# ============================================================

@dataclass
class GuardrailEvent:
    stage: str      # input | tool | output | routing
    name: str       # e.g. prompt_injection, pii_redaction
    action: str     # block | redact | replace | warn
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ============================================================
# PII / secrets
# ============================================================

_CARD_RE = re.compile(r"(?<!\d)(?:\d[ -]?){12,18}\d(?!\d)")
_AADHAAR_RE = re.compile(r"(?<!\d)[2-9]\d{3}[ -]?\d{4}[ -]?\d{4}(?!\d)")
_PAN_RE = re.compile(r"\b[A-Z]{5}\d{4}[A-Z]\b")
_CVV_RE = re.compile(r"(?i)\b(?:cvv2?|cvc)\s*[:#-]?\s*\d{3,4}\b")
_SECRET_RES = [
    re.compile(r"\b(?:sk|gsk|tvly|lsv2)[-_][A-Za-z0-9_-]{16,}\b"),
    re.compile(r"\bAIza[0-9A-Za-z_-]{30,}\b"),
    re.compile(
        r"(?i)\b(?:api[_-]?key|secret|token|passw(?:or)?d|pwd)\s*[:=]\s*\S+"
    ),
]
_EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
_PHONE_RE = re.compile(
    r"(?<![\w.])(?:\+\d{1,3}[ -]?)?[6-9]\d{9}(?!\d)"
    r"|(?<![\w.])\+\d{1,3}[ -]?\d{2,4}[ -]?\d{3,4}[ -]?\d{3,4}(?!\d)"
)


def _luhn_ok(digits: str) -> bool:
    total, alt = 0, False
    for ch in reversed(digits):
        d = int(ch)
        if alt:
            d *= 2
            if d > 9:
                d -= 9
        total += d
        alt = not alt
    return total % 10 == 0


def _redact_cards(text: str) -> tuple[str, bool]:
    hit = False

    def repl(match: re.Match) -> str:
        nonlocal hit
        digits = re.sub(r"\D", "", match.group())
        if 13 <= len(digits) <= 19 and _luhn_ok(digits):
            hit = True
            return "[REDACTED_CARD]"
        return match.group()

    return _CARD_RE.sub(repl, text), hit


def redact_pii(text: str, *, include_contact: bool = True) -> tuple[str, list[str]]:
    """Redact sensitive data. Returns (clean_text, categories_found).

    include_contact=False leaves emails / phone numbers alone (used for tool
    output, where venues legitimately publish contact details).
    """
    found: list[str] = []

    for pattern in _SECRET_RES:
        text, n = pattern.subn("[REDACTED_SECRET]", text)
        if n and "secret" not in found:
            found.append("secret")

    text, n = _CVV_RE.subn("[REDACTED_CVV]", text)
    if n:
        found.append("cvv")

    text, hit = _redact_cards(text)
    if hit:
        found.append("card_number")

    text, n = _AADHAAR_RE.subn("[REDACTED_ID]", text)
    if n:
        found.append("aadhaar")

    text, n = _PAN_RE.subn("[REDACTED_ID]", text)
    if n:
        found.append("pan")

    if include_contact:
        text, n = _EMAIL_RE.subn("[REDACTED_EMAIL]", text)
        if n:
            found.append("email")
        text, n = _PHONE_RE.subn("[REDACTED_PHONE]", text)
        if n:
            found.append("phone")

    return text, found


# ============================================================
# Prompt injection
# ============================================================

_INJECTION_PATTERNS = [
    r"ignore\s+(?:all\s+|any\s+|the\s+|your\s+)?(?:previous|prior|above|earlier|preceding)\s+(?:instructions?|prompts?|rules?|directions?)",
    r"disregard\s+(?:all\s+|any\s+|the\s+|your\s+)?(?:previous|prior|above|earlier|system|instructions?|rules?)",
    r"forget\s+(?:all\s+|your\s+|the\s+)?(?:previous\s+|prior\s+|above\s+)?(?:instructions?|rules?|prompts?)",
    r"(?:reveal|show|print|repeat|display|leak|output)\s+(?:me\s+)?(?:your|the)\s+(?:system|hidden|initial|original|developer)\s+(?:prompt|instructions?|message)",
    r"you\s+are\s+now\s+(?:in\s+)?(?:dan|developer\s+mode|jailbreak\w*|unrestricted)",
    r"\b(?:act|behave)\s+as\s+(?:an?\s+)?(?:dan|unrestricted|jailbroken|unfiltered)",
    r"\bdo\s+anything\s+now\b",
    r"\bjailbreak\b",
    r"override\s+(?:your|the|all)\s+(?:safety|guardrails?|rules|instructions|restrictions)",
    r"(?:bypass|disable|turn\s+off)\s+(?:your\s+|the\s+)?(?:safety|guardrails?|filters?|restrictions)",
    r"<\s*/?\s*(?:system|assistant)\s*>",
    r"\[\s*/?\s*(?:system|inst)\s*\]",
]
_INJECTION_RES = [re.compile(p, re.IGNORECASE) for p in _INJECTION_PATTERNS]


def detect_prompt_injection(text: str) -> list[str]:
    return [p.pattern for p in _INJECTION_RES if p.search(text or "")]


def sanitize_tool_output(text: str) -> tuple[str, list[GuardrailEvent]]:
    """Strip injection attempts hidden in web/tool results (indirect injection)
    and cap the size. Tool output is untrusted data, never instructions."""
    events: list[GuardrailEvent] = []
    removed = 0

    for pattern in _INJECTION_RES:
        text, n = pattern.subn(INJECTION_PLACEHOLDER, text)
        removed += n

    if removed:
        events.append(
            GuardrailEvent(
                "tool",
                "indirect_prompt_injection",
                "replace",
                f"removed {removed} suspected instruction(s) from tool output",
            )
        )

    if len(text) > MAX_TOOL_OUTPUT_CHARS:
        text = text[:MAX_TOOL_OUTPUT_CHARS] + "\n[truncated]"
        events.append(
            GuardrailEvent("tool", "tool_output_truncated", "warn", "output capped")
        )

    return text, events


# ============================================================
# Topic fallback (used only if the LLM classifier fails)
# ============================================================

_TRAVEL_TERMS = re.compile(
    r"\b(trip|travel\w*|flights?|fly\w*|airlines?|airports?|airfare|tickets?|"
    r"hotels?|hostels?|resorts?|stay|accommodation|lodging|visa|itinerary|"
    r"vacation|holiday|honeymoon|tour|sightseeing|destination|cruise|train|"
    r"weather|climate|forecast|pack|budget|luggage|booking|book|layover|passport)\b",
    re.IGNORECASE,
)


def looks_travel_related(text: str) -> bool:
    return bool(_TRAVEL_TERMS.search(text or ""))


REFUSAL_TOPIC = (
    "I can only help with travel planning: flights, hotels, weather, budgets "
    "and itineraries. Please ask me something about a trip."
)


# ============================================================
# INPUT guardrail
# ============================================================

@dataclass
class InputScreen:
    allowed: bool
    sanitized: str
    reason: str = ""
    events: list[GuardrailEvent] = field(default_factory=list)


def screen_input(text: str, *, max_chars: int = MAX_INPUT_CHARS) -> InputScreen:
    """Deterministic input checks. PII is redacted (request continues);
    empty, oversized or injection-style input is blocked."""
    raw = (text or "").strip()

    if not raw:
        return InputScreen(
            False,
            "",
            "Please enter a travel request.",
            [GuardrailEvent("input", "empty_input", "block")],
        )

    if len(raw) > max_chars:
        return InputScreen(
            False,
            raw[:max_chars],
            f"That request is too long (limit {max_chars} characters).",
            [GuardrailEvent("input", "input_too_long", "block", f"{len(raw)} chars")],
        )

    if detect_prompt_injection(raw):
        return InputScreen(
            False,
            raw,
            "That request looks like an attempt to override my instructions, "
            "so I can't process it. " + REFUSAL_TOPIC,
            [GuardrailEvent("input", "prompt_injection", "block")],
        )

    clean, categories = redact_pii(raw)
    events: list[GuardrailEvent] = []
    if categories:
        events.append(
            GuardrailEvent(
                "input",
                "pii_redaction",
                "redact",
                "redacted: " + ", ".join(categories),
            )
        )

    return InputScreen(True, clean, "", events)


# ============================================================
# OUTPUT guardrail
# ============================================================

# Flight-number-like token: 2-letter airline code + 2-4 digits (EK 512, 6E-2031)
_FLIGHT_NO_RE = re.compile(r"\b(?:[A-Z]{2}|[A-Z]\d|\d[A-Z])[ -]?\d{2,4}\b")
_NOT_AIRLINES = {
    "RS", "PM", "NO", "ID", "US", "UK", "EU", "UN", "BC", "ST", "ND", "RD",
    "TH", "KM", "KG", "MM", "CM", "MB", "GB",
}
UNVERIFIED_FLIGHT = "[unverified flight no.]"

_BOOKING_CLAIM_RE = re.compile(
    r"(?i)\b(?:"
    r"(?:your|the)\s+(?:flights?|tickets?|hotel|room|trip)s?\s+(?:has|have|is|are)\s+(?:been\s+)?(?:booked|confirmed|reserved|issued)"
    r"|i(?:'ve|\s+have)\s+(?:booked|reserved|purchased)"
    r"|booking\s+(?:is\s+)?(?:confirmed|complete[d]?)"
    r"|(?:successfully|has\s+been)\s+booked"
    r"|ticket\s+(?:is\s+)?issued"
    r")"
)
BOOKING_REPLACEMENT = "[nothing has been booked: this app only provides planning guidance]"

_ESTIMATE_RE = re.compile(
    r"(?i)estimat|could not be confirmed|not confirmed|unverified|approximate|indicative"
)
DISCLAIMER = (
    "\n\n_Note: live availability and fares could not be confirmed, "
    "so figures above are estimates._"
)


def _norm_flight(token: str) -> str:
    return re.sub(r"[ -]", "", token).upper()


def extract_flight_numbers(text: str) -> set[str]:
    found = set()
    for match in _FLIGHT_NO_RE.finditer(text or ""):
        token = _norm_flight(match.group())
        letters, digits = token[:2], token[2:]
        if letters in _NOT_AIRLINES:
            continue
        if len(digits) == 4 and 1900 <= int(digits) <= 2100:  # looks like a year
            continue
        found.add(token)
    return found


def find_ungrounded_flight_numbers(text: str, corpus: str) -> list[str]:
    """Flight numbers mentioned in `text` that never appear in tool data."""
    mentioned = extract_flight_numbers(text)
    if not mentioned:
        return []
    norm_corpus = re.sub(r"[ -]", "", (corpus or "").upper())
    return sorted(token for token in mentioned if token not in norm_corpus)


@dataclass
class OutputScreen:
    text: str
    events: list[GuardrailEvent] = field(default_factory=list)

    @property
    def clean(self) -> bool:
        return not self.events


def screen_output(
    text: str,
    *,
    grounding_corpus: str = "",
    check_flights: bool = False,
    live_data_ok: bool = True,
) -> OutputScreen:
    """Post-generation checks. Never raises; returns a corrected text + events."""
    events: list[GuardrailEvent] = []
    text = text or ""

    # 1. Secrets / card numbers / IDs must never leave the system.
    text, categories = redact_pii(text, include_contact=False)
    if categories:
        events.append(
            GuardrailEvent(
                "output", "sensitive_data_redacted", "redact", ", ".join(categories)
            )
        )

    # 2. The app cannot book anything. Never claim it did.
    text, n = _BOOKING_CLAIM_RE.subn(BOOKING_REPLACEMENT, text)
    if n:
        events.append(
            GuardrailEvent(
                "output",
                "unverified_booking_claim",
                "replace",
                f"{n} claim(s) that something was booked/confirmed",
            )
        )

    # 3. Hallucination guard: flight numbers must exist in the MCP data.
    if check_flights:
        bad = find_ungrounded_flight_numbers(text, grounding_corpus)
        if bad:
            bad_set = set(bad)
            text = _FLIGHT_NO_RE.sub(
                lambda m: UNVERIFIED_FLIGHT
                if _norm_flight(m.group()) in bad_set
                else m.group(),
                text,
            )
            events.append(
                GuardrailEvent(
                    "output",
                    "ungrounded_flight_number",
                    "replace",
                    "not in tool data: " + ", ".join(bad),
                )
            )

    # 4. If live data was unavailable, the answer must say it is an estimate.
    if not live_data_ok and not _ESTIMATE_RE.search(text):
        text += DISCLAIMER
        events.append(
            GuardrailEvent(
                "output", "missing_estimate_disclaimer", "warn", "disclaimer appended"
            )
        )

    return OutputScreen(text, events)


def events_to_dicts(events: list[GuardrailEvent]) -> list[dict[str, Any]]:
    return [e.to_dict() for e in events]
