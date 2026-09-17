import json
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.types import interrupt

from config import get_llm
from mcp_client import (
    current_weather,
    forecast,
    list_airlines,
    list_airports,
    tavily_search,
)
from state import TravelState


llm = get_llm()


# ============================================================
# LLM HELPER
# ============================================================

def _llm_text(system: str, prompt: str) -> str:
    response = llm.invoke(
        [
            SystemMessage(content=system),
            HumanMessage(content=prompt),
        ]
    )

    return response.content


# ============================================================
# JSON HELPER
# ============================================================

def _json_from_llm(text: str) -> dict:
    print("\n========== RAW LLM RESPONSE ==========")
    print(text)
    print("======================================\n")

    try:
        start = text.index("{")
        end = text.rindex("}") + 1

        json_text = text[start:end]

        print("\n========== EXTRACTED JSON ==========")
        print(json_text)
        print("====================================\n")

        return json.loads(json_text)

    except (ValueError, json.JSONDecodeError) as e:
        print("\n========== JSON PARSING ERROR ==========")
        print("Error:", repr(e))
        print("LLM response:", text)
        print("========================================\n")

        raise ValueError(
            "The LLM did not return valid JSON."
        ) from e


# ============================================================
# SUPERVISOR AGENT
# ============================================================

def supervisor_agent(state: TravelState):
    query = state["user_query"]

    print("\n")
    print("================================================")
    print("              SUPERVISOR AGENT")
    print("================================================")
    print("User Query:")
    print(query)
    print("================================================\n")

    # --------------------------------------------------------
    # INPUT GUARDRAIL
    # --------------------------------------------------------

    guardrail_prompt = f"""
Determine whether the following request is a valid travel
planning request.

A valid request can include things such as:
- flights
- hotels
- destinations
- tourism
- sightseeing
- weather
- itineraries
- budgets
- transportation
- trip planning
- vacation planning
- business travel

Return only JSON in this exact format:

{{
    "allowed": true,
    "reason": ""
}}

User request:

{query}
"""

    guardrail_raw = _llm_text(
        "You are an input validation guardrail. Return strict JSON only.",
        guardrail_prompt,
    )

    print("\n========== GUARDRAIL RAW RESPONSE ==========")
    print(guardrail_raw)
    print("============================================\n")

    guardrail_result = _json_from_llm(guardrail_raw)

    print("\n========== GUARDRAIL PARSED RESPONSE ==========")
    print(json.dumps(guardrail_result, indent=2))
    print("================================================\n")

    # --------------------------------------------------------
    # GUARDRAIL REJECTION
    # --------------------------------------------------------

    if not guardrail_result.get("allowed", False):

        reason = guardrail_result.get(
            "reason",
            "Request rejected by input guardrail.",
        )

        print("\n========== GUARDRAIL BLOCKED REQUEST ==========")
        print(reason)
        print("================================================\n")

        return {
            "selected_agents": [],
            "trip_constraints": {},
            "supervisor_reasoning": reason,
            "final_response": reason,
            "messages": [
                AIMessage(
                    content=f"Guardrail blocked request: {reason}"
                )
            ],
            "llm_calls": state.get("llm_calls", 0) + 1,
        }

    # --------------------------------------------------------
    # SUPERVISOR ROUTING
    # --------------------------------------------------------

    prompt = f"""
You are the supervisor of a real-world multi-agent
travel planning system.

Decide which specialist agents are needed for this
user request.

Available agents:

- flight_agent:
  Use when flights, airports, airlines, routes,
  or airfare guidance are needed.

- hotel_agent:
  Use when hotels, stays, neighborhoods,
  or accommodation are needed.

- weather_agent:
  Use when weather, climate, season, packing,
  or forecast is useful.

- budget_agent:
  Use when budget, affordability, cost,
  or price constraints are mentioned.

- itinerary_agent:
  Almost always needed to produce the travel plan.

Return ONLY valid JSON with this schema:

{{
    "selected_agents": [
        "flight_agent",
        "hotel_agent",
        "weather_agent",
        "budget_agent",
        "itinerary_agent"
    ],
    "trip_constraints": {{
        "destination": "",
        "origin": "",
        "duration": "",
        "budget": "",
        "travel_style": "",
        "special_preferences": []
    }},
    "reasoning": ""
}}

User request:

{query}
"""

    raw = _llm_text(
        "You route work to specialist agents. Return strict JSON only.",
        prompt,
    )

    print("\n========== SUPERVISOR RAW RESPONSE ==========")
    print(raw)
    print("==============================================\n")

    parsed = _json_from_llm(raw)

    print("\n========== SUPERVISOR PARSED JSON ==========")
    print(json.dumps(parsed, indent=2))
    print("=============================================\n")

    selected = parsed.get("selected_agents", [])

    trip_constraints = parsed.get(
        "trip_constraints",
        {},
    )

    reasoning = parsed.get(
        "reasoning",
        "",
    )

    print("\n========== SUPERVISOR DECISION ==========")
    print("Selected Agents:", selected)
    print("Trip Constraints:")
    print(json.dumps(trip_constraints, indent=2))
    print("Reasoning:", reasoning)
    print("=========================================\n")

    return {
        "selected_agents": selected,
        "trip_constraints": trip_constraints,
        "supervisor_reasoning": reasoning,
        "messages": [
            AIMessage(
                content="Supervisor created the agent plan."
            )
        ],
        "llm_calls": state.get("llm_calls", 0) + 1,
    }


# ============================================================
# FLIGHT AGENT
# ============================================================

async def flight_agent(state: TravelState):

    query = state["user_query"]
    constraints = state.get("trip_constraints", {})

    destination = constraints.get("destination", "")
    origin = constraints.get("origin", "")

    print("\n")
    print("================================================")
    print("                FLIGHT AGENT")
    print("================================================")
    print("Query:", query)
    print("Origin:", origin)
    print("Destination:", destination)
    print("Constraints:", constraints)
    print("================================================\n")

    # --------------------------------------------------------
    # CALL AVIATIONSTACK MCP
    # --------------------------------------------------------

    airports = await list_airports(
        destination,
        limit=10,
    )

    airlines = await list_airlines(
        "",
        limit=10,
    )

    print("\n========== AIRPORT MCP DATA ==========")
    print(airports)
    print("======================================\n")

    print("\n========== AIRLINE MCP DATA ==========")
    print(airlines)
    print("======================================\n")

    # --------------------------------------------------------
    # LLM FLIGHT ANALYSIS
    # --------------------------------------------------------

    prompt = f"""
Create practical flight guidance for this trip.

User request:

{query}

Trip constraints:

{constraints}

Origin:

{origin}

Destination:

{destination}

Airport MCP data:

{str(airports)[:3000]}

Airline MCP data:

{str(airlines)[:3000]}

Include:

1. Likely departure airport(s)
2. Likely arrival airport(s)
3. Relevant airlines
4. Estimated flight duration
5. Estimated fare range
6. Peak-season warning if relevant
7. Booking advice

Do not pretend that a specific flight or fare is
live/available unless the provided MCP data explicitly
contains that information.

Clearly distinguish estimates from confirmed information.
"""

    result = _llm_text(
        "You are a flight planning specialist.",
        prompt,
    )

    print("\n========== FLIGHT AGENT OUTPUT ==========")
    print(result)
    print("=========================================\n")

    return {
        "flight_results": result,
        "messages": [
            AIMessage(
                content="Flight agent completed."
            )
        ],
        "llm_calls": state.get("llm_calls", 0) + 1,
    }


# ============================================================
# HOTEL AGENT
# ============================================================

async def hotel_agent(state: TravelState):

    query = (
        f"Best hotels and areas to stay for: "
        f"{state['user_query']}"
    )

    print("\n")
    print("================================================")
    print("                 HOTEL AGENT")
    print("================================================")
    print("Search Query:")
    print(query)
    print("================================================\n")

    # --------------------------------------------------------
    # CALL TAVILY MCP
    # --------------------------------------------------------

    result = await tavily_search(query)

    print("\n========== HOTEL SEARCH RESULT ==========")
    print(result)
    print("=========================================\n")

    return {
        "hotel_results": str(result),
        "messages": [
            AIMessage(
                content="Hotel agent completed."
            )
        ],
        "llm_calls": state.get("llm_calls", 0) + 1,
    }


# ============================================================
# WEATHER AGENT
# ============================================================

async def weather_agent(state: TravelState):

    constraints = state.get(
        "trip_constraints",
        {},
    )

    city = constraints.get(
        "destination",
        "",
    )

    print("\n")
    print("================================================")
    print("                WEATHER AGENT")
    print("================================================")
    print("City:", city)
    print("================================================\n")

    # --------------------------------------------------------
    # CALL WEATHER MCP
    # --------------------------------------------------------

    weather_data = await current_weather(city)

    forecast_data = await forecast(city)

    print("\n========== CURRENT WEATHER ==========")
    print(weather_data)
    print("=====================================\n")

    print("\n========== WEATHER FORECAST ==========")
    print(forecast_data)
    print("======================================\n")

    # --------------------------------------------------------
    # FORMAT WEATHER RESULT
    # --------------------------------------------------------

    result = f"""
Current weather:

{weather_data}

Forecast:

{forecast_data}
"""

    print("\n========== WEATHER AGENT OUTPUT ==========")
    print(result)
    print("==========================================\n")

    return {
        "weather_results": result,
        "messages": [
            AIMessage(
                content="Weather agent completed."
            )
        ],
        "llm_calls": state.get("llm_calls", 0) + 1,
    }


# ============================================================
# BUDGET AGENT
# ============================================================

def budget_agent(state: TravelState):

    print("\n")
    print("================================================")
    print("                 BUDGET AGENT")
    print("================================================")

    print("\nTrip Constraints:")
    print(state.get("trip_constraints"))

    print("\nFlight Results:")
    print(state.get("flight_results"))

    print("\nHotel Results:")
    print(state.get("hotel_results"))

    print("\nWeather Results:")
    print(state.get("weather_results"))

    print("================================================\n")

    prompt = f"""
Analyze whether this trip plan is realistic for the
user's budget.

User request:

{state['user_query']}

Trip constraints:

{state.get('trip_constraints', {})}

Flight results:

{state.get('flight_results', '')}

Hotel results:

{state.get('hotel_results', '')}

Weather results:

{state.get('weather_results', '')}

Return a concise budget assessment containing:

1. Estimated cost categories
2. Risk areas
3. Money-saving suggestions
4. Whether the plan seems feasible

Clearly identify estimates and avoid presenting
unverified prices as confirmed prices.
"""

    result = _llm_text(
        "You are a practical travel budget analyst.",
        prompt,
    )

    print("\n========== BUDGET AGENT OUTPUT ==========")
    print(result)
    print("=========================================\n")

    return {
        "budget_results": result,
        "messages": [
            AIMessage(
                content="Budget agent completed."
            )
        ],
        "llm_calls": state.get("llm_calls", 0) + 1,
    }


# ============================================================
# ITINERARY AGENT
# ============================================================

def itinerary_agent(state: TravelState):

    print("\n")
    print("================================================")
    print("               ITINERARY AGENT")
    print("================================================")

    print("\nTrip Constraints:")
    print(state.get("trip_constraints"))

    print("\nFlight Results:")
    print(state.get("flight_results"))

    print("\nHotel Results:")
    print(state.get("hotel_results"))

    print("\nWeather Results:")
    print(state.get("weather_results"))

    print("\nBudget Results:")
    print(state.get("budget_results"))

    print("================================================\n")

    prompt = f"""
Create a clear draft travel itinerary.

User request:

{state['user_query']}

Trip constraints:

{state.get('trip_constraints', {})}

Flight results:

{state.get('flight_results', '')}

Hotel results:

{state.get('hotel_results', '')}

Weather results:

{state.get('weather_results', '')}

Budget results:

{state.get('budget_results', '')}

Make the output:

- structured
- practical
- easy to read
- realistic
- ready for human review

If information is estimated or uncertain, clearly
label it as an estimate.
"""

    result = _llm_text(
        "You are an expert itinerary planner.",
        prompt,
    )

    print("\n========== ITINERARY OUTPUT ==========")
    print(result)
    print("======================================\n")

    approval_request = f"""
Please review this draft travel plan.

{result}

Reply with approval or feedback.
"""

    return {
        "itinerary": result,
        "approval_request": approval_request,
        "messages": [
            AIMessage(
                content="Draft itinerary created for human review."
            )
        ],
        "llm_calls": state.get("llm_calls", 0) + 1,
    }


# ============================================================
# HUMAN APPROVAL AGENT
# ============================================================

def human_approval_agent(state: TravelState):

    print("\n")
    print("================================================")
    print("             HUMAN APPROVAL STEP")
    print("================================================\n")

    feedback = interrupt(
        {
            "question": "Do you approve this itinerary?",

            "draft_itinerary": state.get(
                "itinerary",
                "",
            ),

            "approval_request": state.get(
                "approval_request",
                "",
            ),

            "expected_response": {
                "approved": True,
                "feedback": "Optional feedback for revision",
            },
        }
    )

    approved = feedback.get(
        "approved",
        False,
    )

    human_feedback = feedback.get(
        "feedback",
        "",
    )

    print("\n========== HUMAN APPROVAL ==========")
    print("Approved:", approved)
    print("Feedback:", human_feedback)
    print("====================================\n")

    return {
        "approved": approved,
        "human_feedback": human_feedback,
        "messages": [
            AIMessage(
                content="Human approval step completed."
            )
        ],
    }


# ============================================================
# FINAL RESPONSE AGENT
# ============================================================

def final_response_agent(state: TravelState):

    print("\n")
    print("================================================")
    print("             FINAL RESPONSE AGENT")
    print("================================================")

    print("Approved:", state.get("approved"))
    print("Feedback:", state.get("human_feedback"))

    print("================================================\n")

    # --------------------------------------------------------
    # APPROVED
    # --------------------------------------------------------

    if state.get("approved", False):

        prompt = f"""
The human approved this draft itinerary.

Produce the final polished travel plan.

User request:

{state['user_query']}

Draft itinerary:

{state.get('itinerary', '')}

Budget notes:

{state.get('budget_results', '')}

Produce a concise but useful final answer.

Do not invent live flight availability, hotel availability,
or exact prices that were not verified.
Clearly label estimates.
"""

    # --------------------------------------------------------
    # NOT APPROVED
    # --------------------------------------------------------

    else:

        prompt = f"""
The human did not approve the draft.

Original user request:

{state['user_query']}

Draft itinerary:

{state.get('itinerary', '')}

Human feedback:

{state.get('human_feedback', '')}

Budget notes:

{state.get('budget_results', '')}

Revise the travel plan according to the human feedback.

Produce a polished revised plan.

Do not invent live flight availability, hotel availability,
or exact prices that were not verified.
Clearly label estimates.
"""

    result = _llm_text(
        "You produce final user-ready travel plans.",
        prompt,
    )

    print("\n========== FINAL RESPONSE ==========")
    print(result)
    print("====================================\n")

    return {
        "final_response": result,
        "messages": [
            AIMessage(
                content=result
            )
        ],
        "llm_calls": state.get("llm_calls", 0) + 1,
    }