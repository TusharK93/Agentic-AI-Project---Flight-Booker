"""
LLM gateway: one entry point for every model call in the app.

Providers
---------
  groq    openai/gpt-oss-20b   (GROQ_MODEL)
  gemini  gemini-3.8-flash     (GEMINI_MODEL)

What the gateway adds on top of a plain chat model
--------------------------------------------------
  * ROUTES      "fast" / "quality" / "judge" each have a provider order
                (config.GATEWAY_ROUTES) - cheap tasks on Groq, long-form
                synthesis on Gemini, evaluation judge on a different family.
  * FAILOVER    if a provider errors, times out, is rate-limited or returns an
                empty answer, the next provider in the route is tried.
  * RETRIES     transient errors (429/5xx/timeouts) are retried with backoff
                before failing over; auth/404/400 errors fail over immediately.
  * BREAKER     a provider that fails N times in a row is skipped for a
                cool-down period instead of being hammered.
  * NORMALISING Gemini 3 can return content as a list of blocks; the gateway
                always hands agents plain text.
  * TELEMETRY   every provider attempt is a normal LangChain LLM run tagged
                with route/provider/attempt/fallback metadata, so LangSmith
                shows exactly which model served each step and which failed.
"""

from __future__ import annotations

import logging
import random
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable

import config

log = logging.getLogger("gateway")


# ============================================================
# Helpers
# ============================================================

def to_text(content: Any) -> str:
    """Normalise a chat-model `.content` (str or list of blocks) to plain text."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") in (None, "text", "output_text"):
                parts.append(block.get("text", ""))
            # thinking / reasoning / tool blocks are intentionally dropped
        return "".join(parts)
    return str(content)


def _is_retryable(exc: Exception) -> bool:
    code = getattr(exc, "status_code", None) or getattr(
        getattr(exc, "response", None), "status_code", None
    )
    if isinstance(code, int):
        return code in (408, 409, 425, 429, 500, 502, 503, 504)

    blob = f"{type(exc).__name__} {exc}".lower()
    return any(
        key in blob
        for key in (
            "rate limit", "ratelimit", "429", "timeout", "timed out", "temporarily",
            "overloaded", "unavailable", "503", "502", "504", "connection",
            "resource exhausted", "resourceexhausted", "deadline",
        )
    )


class EmptyResponse(RuntimeError):
    pass


class AllProvidersFailed(RuntimeError):
    def __init__(self, route: str, errors: list[str]):
        self.route = route
        self.errors = errors
        super().__init__(
            f"All LLM providers failed for route '{route}': " + " | ".join(errors)
        )


# ============================================================
# Providers
# ============================================================

@dataclass
class Provider:
    name: str
    model: str
    api_key: str | None
    build: Callable[[], Any]
    _client: Any = field(default=None, repr=False)

    def available(self) -> bool:
        return bool(self.api_key)

    def client(self) -> Any:
        if self._client is None:
            self._client = self.build()
        return self._client


def _build_groq() -> Any:
    from langchain_groq import ChatGroq

    return ChatGroq(
        model=config.GROQ_MODEL,
        api_key=config.GROQ_API_KEY,
        max_tokens=config.LLM_MAX_TOKENS,
        reasoning_effort=config.GROQ_REASONING_EFFORT,
        timeout=config.GATEWAY_TIMEOUT_S,
        max_retries=0,  # the gateway owns retries
    )


def _build_gemini() -> Any:
    from langchain_google_genai import ChatGoogleGenerativeAI

    # Note: temperature is intentionally left at Google's default for Gemini 3.
    return ChatGoogleGenerativeAI(
        model=config.GEMINI_MODEL,
        google_api_key=config.GOOGLE_API_KEY,
        max_output_tokens=config.GEMINI_MAX_OUTPUT_TOKENS,
        timeout=config.GATEWAY_TIMEOUT_S,
        max_retries=0,
    )


def default_providers() -> dict[str, Provider]:
    return {
        "groq": Provider("groq", config.GROQ_MODEL, config.GROQ_API_KEY, _build_groq),
        "gemini": Provider("gemini", config.GEMINI_MODEL, config.GOOGLE_API_KEY, _build_gemini),
    }


# ============================================================
# Circuit breaker + stats (process-wide, shared by all routes)
# ============================================================

class CircuitBreaker:
    def __init__(self, threshold: int, cooldown_s: float, clock=time.monotonic):
        self.threshold = threshold
        self.cooldown_s = cooldown_s
        self._clock = clock
        self._failures: dict[str, int] = defaultdict(int)
        self._open_until: dict[str, float] = {}
        self._lock = threading.Lock()

    def allow(self, name: str) -> bool:
        with self._lock:
            until = self._open_until.get(name)
            if until is None:
                return True
            if self._clock() >= until:  # half-open: allow one probe
                del self._open_until[name]
                self._failures[name] = self.threshold - 1
                return True
            return False

    def success(self, name: str) -> None:
        with self._lock:
            self._failures[name] = 0
            self._open_until.pop(name, None)

    def failure(self, name: str) -> None:
        with self._lock:
            self._failures[name] += 1
            if self._failures[name] >= self.threshold:
                self._open_until[name] = self._clock() + self.cooldown_s


_BREAKER = CircuitBreaker(
    config.GATEWAY_BREAKER_THRESHOLD, config.GATEWAY_BREAKER_COOLDOWN_S
)

_STATS: dict[str, dict[str, float]] = defaultdict(
    lambda: {"calls": 0, "ok": 0, "errors": 0, "served_as_fallback": 0, "latency_ms": 0.0}
)
_STATS_LOCK = threading.Lock()


def get_stats() -> dict[str, dict[str, float]]:
    with _STATS_LOCK:
        return {k: dict(v) for k, v in _STATS.items()}


def reset_stats() -> None:
    with _STATS_LOCK:
        _STATS.clear()


# ============================================================
# Gateway
# ============================================================

class LLMGateway:
    def __init__(
        self,
        route: str,
        order: list[str] | None = None,
        providers: dict[str, Provider] | None = None,
        *,
        max_retries: int | None = None,
        backoff_s: float | None = None,
        breaker: CircuitBreaker | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ):
        self.route = route
        self.order = order if order is not None else config.GATEWAY_ROUTES.get(
            route, config.GATEWAY_ROUTES["fast"]
        )
        self.providers = providers if providers is not None else default_providers()
        self.max_retries = config.GATEWAY_MAX_RETRIES if max_retries is None else max_retries
        self.backoff_s = config.GATEWAY_BACKOFF_S if backoff_s is None else backoff_s
        self.breaker = breaker or _BREAKER
        self._sleep = sleep

    # ---- core --------------------------------------------------------

    def invoke_with_meta(self, messages: Any) -> tuple[Any, dict[str, Any]]:
        """Call providers in route order. Returns (model_message, meta)."""
        errors: list[str] = []

        for name in self.order:
            provider = self.providers.get(name)

            if provider is None or not provider.available():
                errors.append(f"{name}: not configured")
                continue

            if not self.breaker.allow(name):
                errors.append(f"{name}: circuit open")
                continue

            for attempt in range(self.max_retries + 1):
                started = time.perf_counter()
                fell_back = bool(errors)
                try:
                    message = provider.client().invoke(
                        messages,
                        config={
                            "run_name": f"{name}:{provider.model}",
                            "tags": [f"gateway:{self.route}", f"provider:{name}"],
                            "metadata": {
                                "gateway_route": self.route,
                                "gateway_provider": name,
                                "gateway_model": provider.model,
                                "gateway_attempt": attempt + 1,
                                "gateway_fallback": fell_back,
                                "gateway_prior_errors": errors[-3:],
                            },
                        },
                    )

                    if not to_text(getattr(message, "content", "")).strip():
                        raise EmptyResponse(f"{name} returned an empty response")

                    latency_ms = (time.perf_counter() - started) * 1000
                    self.breaker.success(name)
                    self._record(name, ok=True, latency_ms=latency_ms, fallback=fell_back)

                    usage = getattr(message, "usage_metadata", None) or {}
                    meta = {
                        "route": self.route,
                        "provider": name,
                        "model": provider.model,
                        "attempts": attempt + 1,
                        "fallback": fell_back,
                        "latency_ms": round(latency_ms),
                        "errors": errors,
                        "input_tokens": usage.get("input_tokens"),
                        "output_tokens": usage.get("output_tokens"),
                    }
                    return message, meta

                except Exception as exc:
                    latency_ms = (time.perf_counter() - started) * 1000
                    self._record(name, ok=False, latency_ms=latency_ms, fallback=False)
                    summary = f"{name}: {type(exc).__name__}: {str(exc)[:200]}"

                    if isinstance(exc, EmptyResponse) or not _is_retryable(exc):
                        errors.append(summary)
                        self.breaker.failure(name)
                        log.warning("gateway[%s] %s -> failing over", self.route, summary)
                        break

                    if attempt < self.max_retries:
                        delay = self.backoff_s * (2 ** attempt) * (0.75 + random.random() / 2)
                        log.warning(
                            "gateway[%s] %s (transient) -> retry in %.2fs",
                            self.route, summary, delay,
                        )
                        self._sleep(delay)
                        continue

                    errors.append(summary)
                    self.breaker.failure(name)
                    log.warning("gateway[%s] %s -> retries exhausted, failing over", self.route, summary)

        raise AllProvidersFailed(self.route, errors)

    def invoke(self, messages: Any) -> Any:
        return self.invoke_with_meta(messages)[0]

    def text(self, system: str, prompt: str) -> str:
        """Convenience used by agents: (system, prompt) -> plain text."""
        if len(prompt) > config.GATEWAY_MAX_PROMPT_CHARS:
            prompt = prompt[: config.GATEWAY_MAX_PROMPT_CHARS] + "\n[truncated by gateway]"
        message, _ = self.invoke_with_meta([("system", system), ("human", prompt)])
        return to_text(message.content)

    # ---- stats -------------------------------------------------------

    @staticmethod
    def _record(name: str, *, ok: bool, latency_ms: float, fallback: bool) -> None:
        with _STATS_LOCK:
            s = _STATS[name]
            s["calls"] += 1
            s["latency_ms"] += latency_ms
            if ok:
                s["ok"] += 1
                if fallback:
                    s["served_as_fallback"] += 1
            else:
                s["errors"] += 1


_GATEWAYS: dict[str, LLMGateway] = {}


def get_gateway(route: str = "fast") -> LLMGateway:
    if route not in _GATEWAYS:
        _GATEWAYS[route] = LLMGateway(route)
    return _GATEWAYS[route]
