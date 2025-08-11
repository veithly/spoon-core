import os
import sys
import asyncio
import logging
from typing import Dict, Any

# Ensure the toolkit is in the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../spoon-toolkit')))

from spoon_ai.agents.spoon_react_mcp import SpoonReactMCP
from spoon_ai.tools.mcp_tool import MCPTool
from spoon_ai.tools.tool_manager import ToolManager
from spoon_ai.chat import ChatBot
from spoon_toolkits.crypto.crypto_powerdata.tools import CryptoPowerDataCEXTool

# Set up basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SpoonMacroAnalysisAgent(SpoonReactMCP):
    """
    An agent that performs macroeconomic analysis on cryptocurrencies by combining
    price/indicator data with the latest news.
    """
    name: str = "SpoonMacroAnalysisAgent"
    system_prompt: str = (
        '''You are a cryptocurrency market analyst. Your task is to provide a comprehensive
        macroeconomic analysis of a given token.

        To do this, you will perform the following steps:
        1. Use the `crypto_power_data_cex` tool to get the latest candlestick data and
           technical indicators (like EMA, RSI, MACD) for the token.
        2. Use the `tavily_search` tool to find the latest news and market sentiment
           related to the token. You MUST call the tool with the `query` parameter.
           For example: `tavily_search(query='latest news about NEO token')`
        3. Synthesize the quantitative data from step 1 and the qualitative information
           from step 2 to form a holistic analysis.
        4. Present the final analysis, including a summary of the data, key news insights,
           and your overall assessment of the token's market position.'''
    )

    def __init__(self, **kwargs):
        """Initializes the agent and its tool manager."""
        super().__init__(**kwargs)
        self.avaliable_tools = ToolManager([])

    def _get_mcp_config(self) -> Dict[str, Any]:
        """Returns the simplified, essential MCP server configuration."""
        tavily_key = os.getenv("TAVILY_API_KEY", "")
        if not tavily_key or tavily_key == "your-tavily-api-key-here":
            raise ValueError("TAVILY_API_KEY is not set or contains placeholder value. Please set a valid API key from https://tavily.com")

        return {
            "command": "npx",
            "args": ["--yes", "tavily-mcp"],
            "env": {
                "TAVILY_API_KEY": tavily_key
            }
        }

    async def initialize(self):
        """Initializes the agent by creating and loading all necessary tools."""
        logger.info("Initializing agent and loading tools...")

        tavily_key = os.getenv("TAVILY_API_KEY", "")
        if not tavily_key or tavily_key == "your-tavily-api-key-here":
            raise ValueError("TAVILY_API_KEY is not set or contains placeholder value. Please set a valid API key from https://tavily.com")

        tavily_tool = MCPTool(
            name="tavily_search",
            description="Performs a web search using the Tavily API.",
            mcp_config={
                "command": "npx",
                "args": ["--yes", "tavily-mcp"],
                "env": {
                    "TAVILY_API_KEY": tavily_key
                },
                "tool_name_mapping": {
                    "tavily_search": "tavily-search"
                }
            }
        )

        # 2. Create and load the Crypto Power Data CEX tool
        crypto_tool = CryptoPowerDataCEXTool()

        # 3. Load the MCP tool into the agent's tool manager
        self.avaliable_tools = ToolManager([tavily_tool, crypto_tool])

        logger.info("Agent initialized successfully with all tools.")
        logger.info(f"Available tools: {list(self.avaliable_tools.tool_map.keys())}")

async def main():
    """A minimal demonstration of the macro analysis agent."""
    print("--- SpoonOS Macro Analysis Agent Demo ---")

    # Check for necessary API keys
    tavily_key = os.getenv("TAVILY_API_KEY")
    if not tavily_key or tavily_key == "your-tavily-api-key-here":
        logger.error("TAVILY_API_KEY is not set or contains placeholder value. Please set a valid API key.")
        print("Error: TAVILY_API_KEY is required. Please:")
        print("1. Get an API key from https://tavily.com")
        print("2. Set TAVILY_API_KEY in your .env file")
        return

    # 1. Create the Agent instance
    agent = SpoonMacroAnalysisAgent(llm=ChatBot(llm_provider="openai", model_name="gpt-4.1"))
    print("Agent instance created.")

    # 2. Initialize the agent and its tools
    await agent.initialize()

    # 3. Define and run the analysis query
    query = "Perform a macro analysis of the NEO token."
    # query = "What is the latest news about NEO token?"
    print(f"\nRunning query: {query}")
    print("-" * 30)

    # Run the agent to get the analysis
    response = await agent.run(query)

    print("\n--- Analysis Complete ---")
    print(response)

if __name__ == "__main__":
    asyncio.run(main())
