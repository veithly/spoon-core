"""
🔍 Tavily Search Agent with MCP stdio Integration

This agent demonstrates how to integrate with the tavily-mcp server using stdio transport.
It provides intelligent web search capabilities powered by Tavily's search API.

📌 Requirements:
- You must have a valid Tavily API key set in environment variable TAVILY_API_KEY
- The tavily-mcp package should be available via npx

📚 For more info, see:
- https://github.com/modelcontextprotocol/servers/tree/main/src/tavily

💡 Example usage:
    > Search for the latest news about AI developments
    > Find information about Python programming best practices
    > What are the current trends in machine learning?
"""

from spoon_ai.agents.spoon_react import SpoonReactAI
from spoon_ai.agents.mcp_client_mixin import MCPClientMixin
from fastmcp.client.transports import StdioTransport
from spoon_ai.tools.tool_manager import ToolManager
from pydantic import Field
from spoon_ai.chat import ChatBot
import os
import asyncio
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TavilySearchAgent(SpoonReactAI, MCPClientMixin):
    """
    An AI assistant specialized in web search using Tavily's search API via MCP.
    Provides intelligent search capabilities with context-aware results.
    """

    name: str = "TavilySearchAgent"
    description: str = (
        "An AI assistant specialized in web search using Tavily's search API. "
        "Supports intelligent web search with context-aware results, news search, "
        "and comprehensive information retrieval from the web. "
        "Use this agent when you need to search for current information, "
        "news, research topics, or any web-based content."
    )

    system_prompt: str = """
    You are TavilySearchAgent, an intelligent web search assistant powered by Tavily's search API.
    You can perform comprehensive web searches and provide relevant, up-to-date information.

    Available capabilities:
    - General web search with context-aware results
    - News and current events search
    - Research and information gathering
    - Real-time information retrieval

    When the user asks for information that requires web search:
    1. Use the tavily_search tool for general web searches
    2. Provide comprehensive and relevant results
    3. Summarize key findings when appropriate
    4. If the search doesn't yield sufficient results, try refining the search query

    Always be helpful and provide accurate, up-to-date information based on the search results.
    If you're unsure about something, indicate that and suggest alternative search approaches.
    """

    next_step_prompt: str = (
        "Based on the search results, decide what to do next. "
        "If the results are incomplete or need clarification, consider refining the search query "
        "or asking for more specific information."
    )

    max_steps: int = 5
    avaliable_tools: ToolManager = Field(default_factory=lambda: ToolManager([]))

    def __init__(self, **kwargs):
        """
        Initialize the Tavily Search Agent with stdio MCP transport.

        Args:
            **kwargs: Additional arguments passed to parent classes
        """
        super().__init__(**kwargs)

        # Create stdio transport for tavily-mcp
        # This will run: npx -y tavily-mcp
        stdio_transport = StdioTransport(
            command="npx",
            args=["-y", "tavily-mcp"],
            env={
                "TAVILY_API_KEY": os.getenv("TAVILY_API_KEY", "your-api-key-here")
            }
        )

        MCPClientMixin.__init__(self, mcp_transport=stdio_transport)

    async def initialize(self):
        """Initialize the agent and verify MCP connection."""
        try:
            # Test the connection by listing available tools
            tools = await self.list_mcp_tools()
            logger.info(f"Successfully connected to Tavily MCP server. Available tools: {[tool.name for tool in tools]}")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize Tavily Search Agent: {e}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
            return False

    async def search_web(self, query: str, max_results: int = 5) -> str:
        """
        Perform a web search using Tavily's search API.

        Args:
            query: Search query string
            max_results: Maximum number of results to return

        Returns:
            Search results as formatted string
        """
        try:
            result = await self.call_mcp_tool(
                "tavily_search",
                query=query,
                max_results=max_results
            )
            return result
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return f"Search failed: {str(e)}"


async def main():
    """
    Main function demonstrating the Tavily Search Agent usage.
    """
    # Set API key directly for demo purposes
    api_key = "your-api-key-here"
    os.environ["TAVILY_API_KEY"] = api_key

    # Check if API key is set
    if not api_key or api_key == "your-api-key-here":
        print("⚠️  Warning: TAVILY_API_KEY environment variable not set or using placeholder value.")
        print("Please set your Tavily API key:")
        print("export TAVILY_API_KEY=your_actual_api_key")
        print("\nContinuing with demo (may fail without valid API key)...")

    # Create the Tavily Search Agent with new LLM architecture
    print("🔧 Setting up LLM architecture...")

    # Create agent with LLM manager architecture (only option now)
    search_agent = TavilySearchAgent(
        llm=ChatBot(
            llm_provider="openai",
            model_name="anthropic/claude-sonnet-4"
        )
    )
    print("✓ Using LLM manager architecture")

    print("=== Tavily Search Agent Demo ===")
    print("Initializing agent...")

    # Initialize the agent
    if not await search_agent.initialize():
        print("❌ Failed to initialize agent. Please check your setup.")
        return

    print("✅ Agent initialized successfully!")

    # Example searches
    test_queries = [
        "What are the latest developments in artificial intelligence in 2024?",
        "Python programming best practices for beginners",
        "Current trends in renewable energy technology"
    ]

    for i, query in enumerate(test_queries, 1):
        print(f"\n--- Test Query {i} ---")
        print(f"Query: {query}")

        # Reset agent state for each query
        search_agent.clear()

        try:
            # Run the search
            response = await search_agent.run(query)
            print(f"Response: {response}")
        except Exception as e:
            print(f"❌ Error during search: {e}")

        print("-" * 50)

    print("\n=== Demo completed ===")


if __name__ == "__main__":
    asyncio.run(main())
