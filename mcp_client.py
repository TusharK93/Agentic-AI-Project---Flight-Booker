import os
import sys
import shutil
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langchain_mcp_adapters.client import MultiServerMCPClient

load_dotenv()

# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

AVIATIONSTACK_DIR = BASE_DIR / "aviationstack-mcp"
AVIATIONSTACK_VENV_PYTHON = (
    AVIATIONSTACK_DIR / ".venv" / "Scripts" / "python.exe"
)

WEATHER_SERVER = BASE_DIR / "weather_mcp_server.py"


# ============================================================
# ENVIRONMENT
# ============================================================

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
AVIATION_STACK_API_KEY = os.getenv("AVIATION_STACK_API_KEY")
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")


# Always preserve the normal Windows environment.
# This is important for PATH, Node.js, etc.
MCP_ENV = os.environ.copy()

if TAVILY_API_KEY:
    MCP_ENV["TAVILY_API_KEY"] = TAVILY_API_KEY

if AVIATION_STACK_API_KEY:
    MCP_ENV["AVIATION_STACK_API_KEY"] = AVIATION_STACK_API_KEY

if OPENWEATHER_API_KEY:
    MCP_ENV["OPENWEATHER_API_KEY"] = OPENWEATHER_API_KEY


# ============================================================
# FIND NPX
# ============================================================

def find_npx() -> str:
    """
    Find npx on Windows/Linux/macOS.

    Tavily MCP is launched through npx.
    """

    candidates = [
        shutil.which("npx"),
        shutil.which("npx.cmd"),
        shutil.which("npx.exe"),
    ]

    for candidate in candidates:
        if candidate and Path(candidate).exists():
            return candidate

    # Common Windows Node.js installation locations.
    common_paths = [
        Path(os.environ.get("ProgramFiles", "")) / "nodejs" / "npx.cmd",
        Path(os.environ.get("ProgramFiles(x86)", "")) / "nodejs" / "npx.cmd",
        Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "nodejs" / "npx.cmd",
    ]

    for candidate in common_paths:
        if candidate.exists():
            return str(candidate)

    raise FileNotFoundError(
        "npx was not found. Node.js/npm must be installed because "
        "the Tavily MCP server is launched with npx."
    )


# ============================================================
# AVIATIONSTACK SERVER
# ============================================================

def get_aviationstack_server() -> dict[str, Any]:
    """
    Prefer the Python executable inside aviationstack-mcp/.venv.

    This avoids depending on 'uv' being available in PATH.
    """

    if AVIATIONSTACK_VENV_PYTHON.exists():
        command = str(AVIATIONSTACK_VENV_PYTHON)

        return {
            "transport": "stdio",
            "command": command,
            "args": [
                "-m",
                "aviationstack_mcp",
                "mcp",
                "run",
            ],
            "cwd": str(AVIATIONSTACK_DIR),
            "env": MCP_ENV,
        }

    # Fallback: use uv if the nested venv doesn't exist.
    uv_candidates = [
        shutil.which("uv"),
        shutil.which("uv.exe"),
    ]

    uv_path = next(
        (path for path in uv_candidates if path),
        None,
    )

    if uv_path:
        return {
            "transport": "stdio",
            "command": uv_path,
            "args": [
                "run",
                "-m",
                "aviationstack_mcp",
                "mcp",
                "run",
            ],
            "cwd": str(AVIATIONSTACK_DIR),
            "env": MCP_ENV,
        }

    raise FileNotFoundError(
        f"AviationStack MCP Python executable was not found:\n"
        f"{AVIATIONSTACK_VENV_PYTHON}\n\n"
        f"Run this from the project root:\n"
        f"cd aviationstack-mcp\n"
        f"uv sync"
    )


# ============================================================
# WEATHER SERVER
# ============================================================

def get_weather_server() -> dict[str, Any]:
    """
    Run weather_mcp_server.py using the same Python environment
    that is running Streamlit.
    """

    if not WEATHER_SERVER.exists():
        raise FileNotFoundError(
            f"Weather MCP server not found:\n{WEATHER_SERVER}"
        )

    return {
        "transport": "stdio",
        "command": sys.executable,
        "args": [
            str(WEATHER_SERVER),
        ],
        "cwd": str(BASE_DIR),
        "env": MCP_ENV,
    }


# ============================================================
# TAVILY SERVER
# ============================================================

def get_tavily_server() -> dict[str, Any]:
    """
    Tavily MCP server.
    """

    npx_path = find_npx()

    return {
        "transport": "stdio",
        "command": npx_path,
        "args": [
            "-y",
            "tavily-mcp@latest",
        ],
        "env": MCP_ENV,
        "cwd": str(BASE_DIR),
    }


# ============================================================
# SERVER FACTORY
# ============================================================

def get_server_config(server_name: str) -> dict[str, Any]:
    if server_name == "tavily":
        return get_tavily_server()

    if server_name == "aviationstack":
        return get_aviationstack_server()

    if server_name == "weather":
        return get_weather_server()

    raise ValueError(
        f"Unknown MCP server: {server_name}"
    )


# ============================================================
# GENERIC MCP TOOL CALL
# ============================================================

async def call_tool(
    server_name: str,
    tool_name: str,
    arguments: dict[str, Any] | None = None,
):
    """
    Call only ONE MCP server.

    This is important.

    We do NOT load Tavily + AviationStack + Weather together.
    If one server is unavailable, it will not break the others.
    """

    server_config = get_server_config(server_name)

    print(
        f"\n[MCP] Server: {server_name}"
        f"\n[MCP] Command: {server_config['command']}"
        f"\n[MCP] Args: {server_config.get('args', [])}"
    )

    client = MultiServerMCPClient({
        server_name: server_config
    })

    tools = await client.get_tools()

    tool = next(
        (
            item
            for item in tools
            if item.name == tool_name
        ),
        None,
    )

    if tool is None:
        available_tools = [
            item.name
            for item in tools
        ]

        raise RuntimeError(
            f"MCP tool '{tool_name}' was not found "
            f"on server '{server_name}'.\n"
            f"Available tools: {available_tools}"
        )

    if arguments is None:
        arguments = {}

    return await tool.ainvoke(arguments)


# ============================================================
# TAVILY
# ============================================================

async def tavily_search(
    query: str,
    max_results: int = 5,
):
    return await call_tool(
        "tavily",
        "tavily_search",
        {
            "query": query,
            "max_results": max_results,
        },
    )


# ============================================================
# WEATHER
# ============================================================

async def current_weather(city: str):
    return await call_tool(
        "weather",
        "get_current_weather",
        {
            "city": city,
        },
    )


async def forecast(city: str):
    return await call_tool(
        "weather",
        "get_forecast",
        {
            "city": city,
        },
    )


# ============================================================
# AVIATIONSTACK
# ============================================================

async def future_flights(
    airport_iata_code: str,
    schedule_type: str,
    airline_iata: str = "",
    date: str = "",
    number_of_flights: int = 5,
):
    return await call_tool(
        "aviationstack",
        "future_flights_arrival_departure_schedule",
        {
            "airport_iata_code": airport_iata_code,
            "schedule_type": schedule_type,
            "airline_iata": airline_iata,
            "date": date,
            "number_of_flights": number_of_flights,
        },
    )


async def flight_schedule(
    airport_iata_code: str,
    schedule_type: str,
    airline_name: str = "",
    number_of_flights: int = 5,
):
    return await call_tool(
        "aviationstack",
        "flight_arrival_departure_schedule",
        {
            "airport_iata_code": airport_iata_code,
            "schedule_type": schedule_type,
            "airline_name": airline_name,
            "number_of_flights": number_of_flights,
        },
    )


async def flights_with_airline(
    airline_name: str,
    number_of_flights: int,
    flight_status: str = "",
):
    return await call_tool(
        "aviationstack",
        "flights_with_airline",
        {
            "airline_name": airline_name,
            "number_of_flights": number_of_flights,
            "flight_status": flight_status,
        },
    )


async def historical_flights(
    flight_date: str,
    number_of_flights: int,
    airline_iata: str = "",
    dep_iata: str = "",
    arr_iata: str = "",
):
    return await call_tool(
        "aviationstack",
        "historical_flights_by_date",
        {
            "flight_date": flight_date,
            "number_of_flights": number_of_flights,
            "airline_iata": airline_iata,
            "dep_iata": dep_iata,
            "arr_iata": arr_iata,
        },
    )


async def flight_status(
    flight_iata: str,
    flight_date: str = "",
):
    return await call_tool(
        "aviationstack",
        "get_flight_status",
        {
            "flight_iata": flight_iata,
            "flight_date": flight_date,
        },
    )


# ============================================================
# DEBUG
# ============================================================

def print_mcp_configuration():
    print("\n========== MCP CONFIGURATION ==========")

    print(f"BASE_DIR:")
    print(f"  {BASE_DIR}")

    print("\nAviationStack directory:")
    print(f"  {AVIATIONSTACK_DIR}")

    print("\nAviationStack Python:")
    print(f"  {AVIATIONSTACK_VENV_PYTHON}")
    print(f"  Exists: {AVIATIONSTACK_VENV_PYTHON.exists()}")

    print("\nWeather server:")
    print(f"  {WEATHER_SERVER}")
    print(f"  Exists: {WEATHER_SERVER.exists()}")

    print("\nCurrent Python:")
    print(f"  {sys.executable}")

    try:
        npx = find_npx()
        print("\nnpx:")
        print(f"  {npx}")
    except FileNotFoundError:
        print("\nnpx:")
        print("  NOT FOUND")

    print("========================================\n")


if __name__ == "__main__":
    print_mcp_configuration()