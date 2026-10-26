"""
Single place that runs the LangGraph app.

  run_graph()          used by Streamlit. Gives every invocation a known LangSmith
                       run_id and scores it (online evaluation).
  run_pipeline_once()  used by the offline evaluation. Runs the full graph for one
                       query with an in-memory checkpointer and auto-approves the
                       human-in-the-loop step.
"""

from __future__ import annotations

import time
import uuid
from typing import Any

from langchain_core.messages import HumanMessage

from config import DATABASE_URL
from evaluation.online import evaluate_and_log
from graph import build_graph
from guardrails import GuardrailEvent, redact_pii

_MEMORY_SAVER = None


def _memory_saver():
    """Process-wide in-memory checkpointer (used when DATABASE_URL is not set).
    Module-level so the human-approval resume works across Streamlit reruns."""
    global _MEMORY_SAVER
    if _MEMORY_SAVER is None:
        from langgraph.checkpoint.memory import MemorySaver

        _MEMORY_SAVER = MemorySaver()
    return _MEMORY_SAVER


def prepare_query(query: str) -> tuple[str, list[dict[str, Any]]]:
    """Redact PII *before* the query enters the graph, so card numbers etc. never
    reach LangSmith traces or the checkpoint database. (Blocking decisions are
    made by the supervisor so blocked attempts are still traced.)"""
    clean, categories = redact_pii((query or "").strip())
    events = []
    if categories:
        events.append(
            GuardrailEvent(
                "input", "pii_redaction", "redact", "redacted: " + ", ".join(categories)
            ).to_dict()
        )
    return clean, events


def initial_state(query: str, user_id: str = "demo_user") -> dict[str, Any]:
    clean, events = prepare_query(query)
    return {
        "messages": [HumanMessage(content=clean)],
        "user_id": user_id,
        "user_query": clean,
        "flight_results": "",
        "hotel_results": "",
        "weather_results": "",
        "budget_results": "",
        "itinerary": "",
        "final_response": "",
        "llm_calls": 0,
        "blocked": False,
        "guardrail_events": events,
        "tool_calls": [],
    }


async def run_graph(
    input_data: Any,
    run_cfg: dict[str, Any],
    *,
    user_id: str = "",
    query: str = "",
):
    """Run (or resume) the graph.

    Returns (result_state, run_id, phase, online_eval_results).
    Each call is one LangSmith root run with a fresh run_id (a resume is a new run).
    """
    graph_builder = build_graph()
    run_id = uuid.uuid4()

    traced_cfg = {
        **run_cfg,
        "run_id": run_id,
        "run_name": "travel_planner",
        "tags": ["travel-planner"],
        "metadata": {
            "app": "flight-booker",
            "user_id": user_id,
            "thread_id": run_cfg.get("configurable", {}).get("thread_id"),
        },
    }

    started = time.perf_counter()

    if DATABASE_URL:
        from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

        async with AsyncPostgresSaver.from_conn_string(DATABASE_URL) as checkpointer:
            await checkpointer.setup()
            app = graph_builder.compile(checkpointer=checkpointer)
            result = await app.ainvoke(input_data, config=traced_cfg)
    else:
        app = graph_builder.compile(checkpointer=_memory_saver())
        result = await app.ainvoke(input_data, config=traced_cfg)

    latency_s = time.perf_counter() - started
    phase = "draft" if "__interrupt__" in result else "final"

    evals = evaluate_and_log(
        result,
        str(run_id),
        phase=phase,
        query=result.get("user_query", query),
        latency_s=latency_s,
    )

    return result, str(run_id), phase, evals


# ------------------------------------------------------------------
# Offline evaluation target
# ------------------------------------------------------------------

_OUTPUT_KEYS = (
    "user_query", "blocked", "block_reason", "selected_agents", "trip_constraints",
    "flight_results", "flight_raw", "hotel_results", "weather_results",
    "budget_results", "itinerary", "approved", "final_response", "llm_calls",
    "guardrail_events", "tool_calls",
)


def _serialisable(state: dict[str, Any]) -> dict[str, Any]:
    return {k: state.get(k) for k in _OUTPUT_KEYS if k in state}


async def run_pipeline_once(query: str, *, auto_approve: bool = True) -> dict[str, Any]:
    """Run the whole graph for one query (no Streamlit, no Postgres)."""
    from langgraph.checkpoint.memory import MemorySaver
    from langgraph.types import Command

    app = build_graph().compile(checkpointer=MemorySaver())
    cfg = {"configurable": {"thread_id": f"eval_{uuid.uuid4().hex[:10]}"}}

    try:
        result = await app.ainvoke(initial_state(query, "eval"), config=cfg)

        if "__interrupt__" in result and auto_approve:
            result = await app.ainvoke(
                Command(resume={"approved": True, "feedback": ""}), config=cfg
            )

        return _serialisable(result)

    except Exception as exc:  # a crash is a failed task, not a crashed eval run
        return {"error": f"{type(exc).__name__}: {exc}", "user_query": query}
