import asyncio
import uuid

import streamlit as st

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.types import Command

from config import DATABASE_URL
from graph import build_graph


st.set_page_config(
    page_title="Real-World Multi-Agent Travel Planner",
    layout="wide",
)

st.title("Real-World Multi-Agent Travel Planner")


# ============================================================
# Async graph execution
# ============================================================

async def run_graph(input_data, config):
    """
    Build and execute the LangGraph application.

    The AsyncPostgresSaver is created inside the same async
    event loop that executes app.ainvoke().
    """

    graph_builder = build_graph()

    if DATABASE_URL:
        async with AsyncPostgresSaver.from_conn_string(
            DATABASE_URL
        ) as checkpointer:

            await checkpointer.setup()

            app = graph_builder.compile(
                checkpointer=checkpointer
            )

            result = await app.ainvoke(
                input_data,
                config=config,
            )

            return result

    # --------------------------------------------------------
    # No PostgreSQL configured
    # --------------------------------------------------------

    app = graph_builder.compile()

    result = await app.ainvoke(
        input_data,
        config=config,
    )

    return result


# ============================================================
# Sidebar
# ============================================================

with st.sidebar:

    st.subheader("Session")

    user_id = st.text_input(
        "User ID",
        value="demo_user",
    )

    if "thread_id" not in st.session_state:
        st.session_state.thread_id = (
            f"{user_id}_{uuid.uuid4().hex[:8]}"
        )

    if st.button("New Thread"):

        st.session_state.thread_id = (
            f"{user_id}_{uuid.uuid4().hex[:8]}"
        )

        st.session_state.pop(
            "waiting_for_approval",
            None,
        )

        st.session_state.pop(
            "latest_result",
            None,
        )

    st.caption(
        f"Thread: {st.session_state.thread_id}"
    )


# ============================================================
# User request
# ============================================================

query = st.text_area(
    "Travel request",
    placeholder=(
        "Plan a 7-day Japan trip under Rs. 2 lakh. "
        "I prefer budget hotels and no overnight flights."
    ),
    height=110,
)


config = {
    "configurable": {
        "thread_id": st.session_state.thread_id
    }
}


# ============================================================
# Create Draft Plan
# ============================================================

if st.button(
    "Create Draft Plan",
    type="primary",
):

    if not query.strip():

        st.warning(
            "Enter a travel request first."
        )

    else:

        with st.spinner(
            "Agents are planning..."
        ):

            try:

                result = asyncio.run(
                    run_graph(
                        {
                            "messages": [
                                HumanMessage(
                                    content=query
                                )
                            ],
                            "user_id": user_id,
                            "user_query": query,
                            "flight_results": "",
                            "hotel_results": "",
                            "weather_results": "",
                            "budget_results": "",
                            "itinerary": "",
                            "final_response": "",
                            "llm_calls": 0,
                        },
                        config,
                    )
                )

                st.session_state.latest_result = result

                st.session_state.waiting_for_approval = (
                    "__interrupt__" in result
                )

                st.rerun()

            except Exception as e:

                st.error(
                    f"Error while creating travel plan: {e}"
                )

                st.exception(e)


# ============================================================
# Display latest result
# ============================================================

result = st.session_state.get(
    "latest_result"
)


if result:

    # --------------------------------------------------------
    # Supervisor
    # --------------------------------------------------------

    st.subheader("Supervisor Plan")

    st.write(
        result.get(
            "supervisor_reasoning",
            "",
        )
    )

    st.write(
        "Selected agents:",
        result.get(
            "selected_agents",
            [],
        ),
    )

    # --------------------------------------------------------
    # Agent results
    # --------------------------------------------------------

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("Flight")

        flight_result = result.get(
            "flight_results",
            "",
        )

        if flight_result:
            st.markdown(
                str(flight_result)
            )
        else:
            st.info(
                "Flight agent did not return a result."
            )

        st.subheader("Weather")

        weather_result = result.get(
            "weather_results",
            "",
        )

        if weather_result:
            st.markdown(
                str(weather_result)
            )
        else:
            st.info(
                "Weather agent did not return a result."
            )

    with col2:

        st.subheader("Hotels")

        hotel_result = result.get(
            "hotel_results",
            "",
        )

        if hotel_result:
            st.markdown(
                str(hotel_result)
            )
        else:
            st.info(
                "Hotel agent did not return a result."
            )

        st.subheader("Budget")

        budget_result = result.get(
            "budget_results",
            "",
        )

        if budget_result:
            st.markdown(
                str(budget_result)
            )
        else:
            st.info(
                "Budget agent did not return a result."
            )

    # --------------------------------------------------------
    # Draft itinerary
    # --------------------------------------------------------

    st.subheader("Draft Itinerary")

    draft = ""

    if "__interrupt__" in result:

        interrupts = result.get(
            "__interrupt__",
            [],
        )

        if interrupts:

            interrupt_value = (
                interrupts[0].value
            )

            if isinstance(
                interrupt_value,
                dict,
            ):

                draft = interrupt_value.get(
                    "draft_itinerary",
                    "",
                )

    else:

        draft = result.get(
            "itinerary",
            "",
        )

    if draft:

        st.markdown(
            str(draft)
        )

    else:

        st.info(
            "Draft itinerary is not available yet."
        )


# ============================================================
# Human Approval
# ============================================================

if st.session_state.get(
    "waiting_for_approval",
    False,
):

    st.divider()

    st.subheader(
        "Human Approval"
    )

    approved = st.radio(
        "Approve this draft?",
        [
            "Yes",
            "No, revise it",
        ],
        horizontal=True,
    )

    feedback = st.text_area(
        "Feedback",
        disabled=(
            approved == "Yes"
        ),
        placeholder=(
            "Tell the itinerary agent what "
            "you want changed..."
        ),
    )

    if st.button(
        "Submit Approval"
    ):

        with st.spinner(
            "Creating final response..."
        ):

            try:

                final_result = asyncio.run(
                    run_graph(
                        Command(
                            resume={
                                "approved": (
                                    approved == "Yes"
                                ),
                                "feedback": feedback,
                            }
                        ),
                        config,
                    )
                )

                st.session_state.latest_result = (
                    final_result
                )

                st.session_state.waiting_for_approval = (
                    "__interrupt__"
                    in final_result
                )

                st.rerun()

            except Exception as e:

                st.error(
                    f"Error while submitting approval: {e}"
                )

                st.exception(e)


# ============================================================
# Final Travel Plan
# ============================================================

final_result = st.session_state.get(
    "latest_result"
)


if (
    final_result
    and final_result.get(
        "final_response"
    )
):

    st.divider()

    st.subheader(
        "Final Travel Plan"
    )

    st.markdown(
        final_result[
            "final_response"
        ]
    )