"""Weather agent using Ollama Cloud and tools from a remote MCP server."""
from google.adk import Agent
from google.adk.models.lite_llm import LiteLlm
from google.adk.tools.mcp_tool.mcp_toolset import McpToolset, StreamableHTTPConnectionParams
import logging
import os
from dotenv import load_dotenv

# `adk web` does not guarantee that it loads this package's .env file.  Load it
# before reading the settings so the same configuration works with the CLI and
# direct Python imports.
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Mặc định: server chạy local. Deploy lên Cloud Run thì đặt MCP_SERVER_URL trong .env
# (ví dụ: https://<service>.run.app/mcp). verify_setup.py đọc cùng biến này.
MCP_SERVER_URL = os.getenv("MCP_SERVER_URL", "http://localhost:8085/mcp")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gpt-oss:120b")
# LiteLLM appends `/api/chat` for the ollama_chat provider.
OLLAMA_API_BASE = os.getenv("OLLAMA_API_BASE", "https://ollama.com")
OLLAMA_API_KEY = os.getenv("OLLAMA_API_KEY")

if not OLLAMA_API_KEY:
    raise RuntimeError(
        "OLLAMA_API_KEY is not configured. Add it to mcp-client/.env "
        "(never commit that file)."
    )

logger.info(f"🌐 Initializing weather agent with remote MCP server")
logger.info(f"📡 MCP Server: {MCP_SERVER_URL}")
logger.info(f"🤖 Ollama model: {OLLAMA_MODEL}")

# McpToolset opens the connection lazily when the agent needs a tool.  Do not
# silently create a tool-less agent: a failed connection should remain visible
# to the user instead of returning made-up weather data.
connection_params = StreamableHTTPConnectionParams(
    url=MCP_SERVER_URL,
    timeout=30.0,
)
weather_tools = McpToolset(connection_params=connection_params)

root_agent = Agent(
    name="weather_agent",
    model=LiteLlm(
        model=f"ollama_chat/{OLLAMA_MODEL}",
        api_base=OLLAMA_API_BASE,
        api_key=OLLAMA_API_KEY,
        # Ollama's chat endpoint accepts the OpenAI-style tool schema; discard
        # unrelated provider-specific generation options sent by the UI.
        drop_params=True,
    ),
    tools=[weather_tools],
    instruction=(
        "You are a weather assistant. Use the MCP weather tools for every "
        "weather or forecast question. Never invent weather data."
    ),
)

logger.info("✅ Weather agent initialized with Ollama Cloud and MCP tools")

