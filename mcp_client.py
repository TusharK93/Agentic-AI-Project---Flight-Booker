import os
import sys
from pathlib import Path

from langchain_mcp_adapters.client import MultiServerMCPClient

from config import (
    AVIATION_STACK_API_KEY,
    OPENWEATHER_API_KEY,
    TAVILY_API_KEY,
)


# ============================================================
# PATHS
# ============================================================
BASE_DIR = Path(__file__).resolve().parent

AVIATIONSTACK_MCP_DIR = BASE_DIR / "aviationstack-mcp"
WEATHER_MCP_SERVER = BASE_DIR / "weather_mcp_server.py"


# ============================================================
# DEBUG
# ============================================================

print("\n========== MCP PATH CHECK ==========")
print("BASE_DIR:", BASE_DIR)
print("WEATHER_MCP_SERVER:", WEATHER_MCP_SERVER)
print("WEATHER_MCP_SERVER exists:", WEATHER_MCP_SERVER.exists())
print("====================================\n")


# ============================================================
# MCP SERVERS
# ============================================================

mcp_servers = {

    # --------------------------------------------------------
    # Tavily
    # --------------------------------------------------------

    "tavily": {
        "transport": "streamable_http",
        "url": (
            f"https://mcp.tavily.com/mcp/"
            f"?tavilyApiKey={TAVILY_API_KEY}"
        ),
    },

    # --------------------------------------------------------
    # AviationStack
    # --------------------------------------------------------

   "aviationstack": {
    "transport": "stdio",

    "command": "uv",

    "args": [
        "run",
        "-m",
        "aviationstack_mcp",
        "mcp",
        "run",
    ],

    "env": {
        **os.environ,
        "AVIATION_STACK_API_KEY": AVIATION_STACK_API_KEY,
    },

    "cwd": str(AVIATIONSTACK_MCP_DIR),
},

    # --------------------------------------------------------
    # Weather
    # --------------------------------------------------------

    "weather": {
        "transport": "stdio",

        "command": sys.executable,

        "args": [
            str(WEATHER_MCP_SERVER),
        ],

        "env": {
            **os.environ,
            "OPENWEATHER_API_KEY": OPENWEATHER_API_KEY,
        },
    },
}


# ============================================================
# CLIENT
# ============================================================

client = MultiServerMCPClient(mcp_servers)


# ============================================================
# TOOL CACHE
# ============================================================

_tools_cache = None


async def get_tools():
    global _tools_cache

    if _tools_cache is None:

        try:
            print("\n========== LOADING MCP TOOLS ==========")

            _tools_cache = await client.get_tools()

            print("MCP tools loaded successfully:")

            for tool in _tools_cache:
                print(" -", tool.name)

            print("=======================================\n")

        except Exception as e:

            print("\n========== MCP TOOL LOAD ERROR ==========")
            print("Exception type:", type(e))
            print("Exception:", repr(e))

            if hasattr(e, "exceptions"):

                print("\nSUB EXCEPTIONS:")

                for i, sub in enumerate(e.exceptions):
                    print(f"\n--- Exception {i + 1} ---")
                    print("Type:", type(sub))
                    print("Error:", repr(sub))

            print("=========================================\n")

            raise

    return _tools_cache


# ============================================================
# GENERIC TOOL CALL
# ============================================================

async def call_tool(tool_name: str, args: dict = None):

    tools = await get_tools()

    tool = next(
        (
            tool
            for tool in tools
            if tool.name == tool_name
        ),
        None,
    )

    if tool is None:
        raise ValueError(
            f"Tool '{tool_name}' not found. "
            f"Available tools: {[t.name for t in tools]}"
        )

    return await tool.ainvoke(args or {})


# ============================================================
# TAVILY
# ============================================================

async def tavily_search(query: str):

    return await call_tool(
        "tavily_search",
        {
            "query": query,
        },
    )


# ============================================================
# AVIATIONSTACK
# ============================================================

async def list_airports(
    search: str = "",
    limit: int = 10,
):

    return await call_tool(
        "list_airports",
        {
            "search": search,
            "limit": limit,
            "offset": 0,
        },
    )


async def list_airlines(
    search: str = "",
    limit: int = 10,
):

    return await call_tool(
        "list_airlines",
        {
            "search": search,
            "limit": limit,
            "offset": 0,
        },
    )


# ============================================================
# WEATHER
# ============================================================

async def current_weather(city: str):

    return await call_tool(
        "get_current_weather",
        {
            "city": city,
        },
    )


async def forecast(city: str):

    return await call_tool(
        "get_forecast",
        {
            "city": city,
        },
    )