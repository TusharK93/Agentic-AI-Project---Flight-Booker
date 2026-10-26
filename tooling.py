"""
Guarded MCP tool runner.

Every tool call in the graph goes through run_tool(), which:

  * enforces a per-agent ALLOWLIST (hotel_agent can never call the flight API)
  * validates / sanitises arguments (IATA format, city names, query PII)
  * caps the number of tool calls per run
  * applies a timeout so a hung MCP subprocess can't freeze the app
  * strips prompt-injection text from the result (tool output = untrusted data)
  * records telemetry -> state["tool_calls"], used by the tool-selection and
    tool-success evaluators and visible in LangSmith.

This module imports only stdlib + guardrails; the real MCP client is resolved
lazily so it can be unit-tested with a fake `caller`.
"""

from __future__ import annotations

import asyncio
import re
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

from guardrails import GuardrailEvent, redact_pii, sanitize_tool_output

MAX_TOOL_CALLS_PER_RUN = 12
TOOL_TIMEOUT_S = 60.0


@dataclass(frozen=True)
class ToolSpec:
    server: str
    mcp_tool: str


# logical tool name -> MCP server / tool
TOOL_REGISTRY: dict[str, ToolSpec] = {
    "future_flights": ToolSpec("aviationstack", "future_flights_arrival_departure_schedule"),
    "tavily_search": ToolSpec("tavily", "tavily_search"),
    "current_weather": ToolSpec("weather", "get_current_weather"),
    "forecast": ToolSpec("weather", "get_forecast"),
}

# Which agent may call which tools. budget/itinerary agents use no tools.
AGENT_TOOL_ALLOWLIST: dict[str, set[str]] = {
    "flight_agent": {"future_flights"},
    "hotel_agent": {"tavily_search"},
    "weather_agent": {"current_weather", "forecast"},
}


class ToolGuardError(Exception):
    """Raised when a tool call violates policy."""


_IATA_RE = re.compile(r"^[A-Z]{3}$")
_CITY_RE = re.compile(r"^[^\W\d_][\w\s.,'’()-]{1,79}$", re.UNICODE)


def check_tool_call(
    agent: str,
    tool: str,
    args: dict[str, Any],
    calls_so_far: int = 0,
) -> dict[str, Any]:
    """Validate a tool call. Returns (possibly sanitised) args or raises ToolGuardError."""
    if tool not in TOOL_REGISTRY:
        raise ToolGuardError(f"unknown tool '{tool}'")

    if tool not in AGENT_TOOL_ALLOWLIST.get(agent, set()):
        raise ToolGuardError(f"agent '{agent}' is not allowed to call '{tool}'")

    if calls_so_far >= MAX_TOOL_CALLS_PER_RUN:
        raise ToolGuardError(f"tool-call cap reached ({MAX_TOOL_CALLS_PER_RUN})")

    clean = dict(args)

    if tool == "future_flights":
        code = str(clean.get("airport_iata_code", "")).upper().strip()
        if not _IATA_RE.fullmatch(code):
            raise ToolGuardError(f"invalid IATA code '{clean.get('airport_iata_code')}'")
        clean["airport_iata_code"] = code
        if clean.get("schedule_type") not in {"arrival", "departure"}:
            raise ToolGuardError("schedule_type must be 'arrival' or 'departure'")
        count = clean.get("number_of_flights", 5)
        if not isinstance(count, int) or not 1 <= count <= 50:
            raise ToolGuardError("number_of_flights must be 1-50")

    elif tool == "tavily_search":
        query = str(clean.get("query", "")).strip()
        if not query:
            raise ToolGuardError("empty search query")
        query, _ = redact_pii(query)
        clean["query"] = query[:500]
        max_results = clean.get("max_results", 5)
        if not isinstance(max_results, int) or not 1 <= max_results <= 10:
            raise ToolGuardError("max_results must be 1-10")

    elif tool in {"current_weather", "forecast"}:
        city = str(clean.get("city", "")).strip()
        if not _CITY_RE.fullmatch(city):
            raise ToolGuardError(f"invalid city '{city[:40]}'")
        clean["city"] = city

    return clean


@dataclass
class ToolOutcome:
    ok: bool
    data: str = ""
    error: str = ""
    record: dict[str, Any] = field(default_factory=dict)
    events: list[GuardrailEvent] = field(default_factory=list)


_API_ERROR_RE = re.compile(
    r"(?i)function_access_restricted|usage_limit_reached|invalid_access_key"
    r"|rate_limit_reached|[\"']error[\"']\s*:"
)


def looks_like_api_error(text: str) -> bool:
    """MCP tools often return API failures as a normal string payload."""
    return bool(_API_ERROR_RE.search(text[:2000]))


def skipped_record(agent: str, tool: str, reason: str) -> dict[str, Any]:
    """Telemetry for a tool the agent *wanted* but could not run (e.g. no IATA code)."""
    spec = TOOL_REGISTRY.get(tool)
    return {
        "agent": agent,
        "tool": tool,
        "mcp_tool": spec.mcp_tool if spec else tool,
        "ok": False,
        "skipped": True,
        "blocked": False,
        "latency_ms": 0,
        "error": reason,
    }


def _default_caller(tool: str) -> Callable[..., Awaitable[Any]]:
    import mcp_client  # lazy: pulls in langchain_mcp_adapters

    return getattr(mcp_client, tool)


async def run_tool(
    agent: str,
    tool: str,
    args: dict[str, Any],
    *,
    calls_so_far: int = 0,
    caller: Callable[..., Awaitable[Any]] | None = None,
    timeout: float = TOOL_TIMEOUT_S,
) -> ToolOutcome:
    spec = TOOL_REGISTRY.get(tool)
    record: dict[str, Any] = {
        "agent": agent,
        "tool": tool,
        "mcp_tool": spec.mcp_tool if spec else tool,
        "ok": False,
        "skipped": False,
        "blocked": False,
        "latency_ms": 0,
        "error": "",
    }
    started = time.perf_counter()

    try:
        clean_args = check_tool_call(agent, tool, args, calls_so_far)
        fn = caller or _default_caller(tool)
        raw = await asyncio.wait_for(fn(**clean_args), timeout=timeout)

        text = raw if isinstance(raw, str) else str(raw)
        text, events = sanitize_tool_output(text)
        record["latency_ms"] = int((time.perf_counter() - started) * 1000)

        if looks_like_api_error(text):
            record["error"] = "tool returned an API error payload"
            return ToolOutcome(False, text, record["error"], record, events)

        record["ok"] = True
        return ToolOutcome(True, text, "", record, events)

    except ToolGuardError as exc:
        record["blocked"] = True
        record["error"] = f"blocked by tool guardrail: {exc}"
        event = GuardrailEvent("tool", "tool_call_blocked", "block", f"{tool}: {exc}")
        events = [event]

    except asyncio.TimeoutError:
        record["error"] = f"timeout after {timeout:.0f}s"
        events = [GuardrailEvent("tool", "tool_timeout", "warn", f"{tool}")]

    except Exception as exc:  # MCP/network/API errors must not crash the graph
        record["error"] = f"{type(exc).__name__}: {exc}"[:500]
        events = []

    record["latency_ms"] = int((time.perf_counter() - started) * 1000)
    return ToolOutcome(False, "", record["error"], record, events)
