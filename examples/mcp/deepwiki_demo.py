#!/usr/bin/env python3
"""
Dual DeepWiki MCP Demo

This demo showcases both SSE and StreamableHttp MCP connections to DeepWiki.
It demonstrates how to use the same MCP server with different transport methods.
"""

import asyncio
import sys
import os
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from spoon_ai.agents.spoon_react_mcp import SpoonReactMCP
from spoon_ai.tools.mcp_tool import MCPTool
from spoon_ai.config import ConfigManager
from spoon_ai.llm.manager import LLMManager
from spoon_ai.chat import ChatBot
import logging

# Configure logging - reduce output
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

class DualDeepWikiAgent:
    """Agent for testing both SSE and StreamableHttp DeepWiki MCP connections"""

    def __init__(self):
        self.agent = None
        self.llm_manager = None
        self.chatbot = None
        self.sse_tool = None
        self.http_tool = None

    async def initialize(self):
        """Initialize the agent with both SSE and HTTP MCP tools"""
        try:
            # Initialize LLM Manager
            self.llm_manager = LLMManager()

            # Initialize ChatBot
            self.chatbot = ChatBot(llm_manager=self.llm_manager)

            # Create SSE MCP tool configuration
            sse_mcp_config = {
                "name": "deepwiki_sse",
                "type": "mcp",
                "description": "DeepWiki SSE MCP tool for repository analysis",
                "enabled": True,
                "mcp_server": {
                    "url": "https://mcp.deepwiki.com/sse",
                    "transport": "sse",
                    "timeout": 30,
                    "headers": {
                        "User-Agent": "SpoonOS-SSE-MCP/1.0",
                        "Accept": "text/event-stream"
                    }
                }
            }

            # Create HTTP MCP tool configuration using StreamableHttpTransport
            http_mcp_config = {
                "name": "deepwiki_http",
                "type": "mcp",
                "description": "DeepWiki HTTP MCP tool for repository analysis",
                "enabled": True,
                "mcp_server": {
                    "url": "https://mcp.deepwiki.com/mcp",
                    "transport": "http",
                    "timeout": 30,
                    "headers": {
                        "User-Agent": "SpoonOS-HTTP-MCP/1.0",
                        "Accept": "application/json, text/event-stream"
                    }
                }
            }

            # Create SSE MCP tool
            self.sse_tool = MCPTool(
                name=sse_mcp_config["name"],
                description=sse_mcp_config["description"],
                mcp_config=sse_mcp_config["mcp_server"]
            )

            # Create HTTP MCP tool
            self.http_tool = MCPTool(
                name=http_mcp_config["name"],
                description=http_mcp_config["description"],
                mcp_config=http_mcp_config["mcp_server"]
            )

            # Initialize both tools
            await self.sse_tool.ensure_parameters_loaded()
            await self.http_tool.ensure_parameters_loaded()

            # Create agent with both MCP tools
            self.agent = SpoonReactMCP(
                name="dual_deepwiki_agent",
                llm_manager=self.llm_manager,
                tools=[self.sse_tool, self.http_tool]
            )

            return True

        except Exception as e:
            logger.error(f"❌ Failed to initialize Dual DeepWiki Agent: {e}")
            return False



    async def query_sse(self, question: str):
        """Query using SSE MCP tool"""
        try:
            result = await self.sse_tool.execute(repoName="XSpoonAi/spoon-core")
            return result
        except Exception as e:
            return f"Error: {e}"

    async def query_http(self, question: str):
        """Query using HTTP MCP tool"""
        try:
            result = await self.http_tool.execute(repoName="XSpoonAi/spoon-core")
            return result
        except Exception as e:
            return f"Error: {e}"

    async def query_agent(self, question: str):
        """Query the agent with a question (will use available tools)"""
        try:
            result = await self.agent.run(question)
            return result
        except Exception as e:
            return f"Error: {e}"

    async def cleanup(self):
        """Cleanup resources"""
        try:
            if self.sse_tool:
                await self.sse_tool.cleanup()
            if self.http_tool:
                await self.http_tool.cleanup()
        except Exception as e:
            logger.error(f"❌ Cleanup failed: {e}")



async def main():
    """Main demo function"""
    print("🚀 Dual DeepWiki MCP Demo")
    print("=" * 50)
    print("This demo showcases both SSE and StreamableHttp MCP connections")
    print("to the same DeepWiki server with the same query.")
    print()

    # Initialize agent
    print("🤖 Initializing Dual DeepWiki Agent...")
    agent = DualDeepWikiAgent()

    if not await agent.initialize():
        print("❌ Agent initialization failed. Stopping demo.")
        return

    # Demo queries
    print("\n💬 Demo Queries")
    print("Testing the same query with different transport methods...")

    query = "What is XSpoonAi/spoon-core project about?"

    print(f"\n🔍 Query: {query}")
    print("-" * 50)

    # Test SSE query
    print("\n📡 Testing SSE MCP...")
    sse_result = await agent.query_sse(query)
    print(f"SSE Result:\n{sse_result}")

    print("\n" + "="*50)

    # Test HTTP query
    print("\n🌐 Testing HTTP MCP...")
    http_result = await agent.query_http(query)
    print(f"HTTP Result:\n{http_result}")

    print("\n" + "="*50)

    # Test agent query (will use available tools)
    print("\n🤖 Testing Agent with both tools...")
    agent_result = await agent.query_agent(query)
    print(f"Agent Result:\n{agent_result}")

    # Interactive demo
    print("\n💬 Interactive Demo")
    print("Ask questions about GitHub repositories (type 'quit' to exit):")
    print("Commands:")
    print("  'sse <question>' - Use SSE transport")
    print("  'http <question>' - Use HTTP transport")
    print("  '<question>' - Use agent (both tools)")
    print("  'quit' - Exit")

    while True:
        try:
            user_input = input("\n🤔 Your input: ").strip()

            if user_input.lower() in ['quit', 'exit', 'q']:
                break

            if not user_input:
                continue

            if user_input.lower().startswith('sse '):
                question = user_input[4:].strip()
                print("🔄 Processing with SSE...")
                result = await agent.query_sse(question)
                print(f"\n📝 SSE Answer:\n{result}")
            elif user_input.lower().startswith('http '):
                question = user_input[5:].strip()
                print("🔄 Processing with HTTP...")
                result = await agent.query_http(question)
                print(f"\n📝 HTTP Answer:\n{result}")
            else:
                print("🔄 Processing with Agent...")
                result = await agent.query_agent(user_input)
                print(f"\n📝 Agent Answer:\n{result}")

        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"❌ Error: {e}")

    # Cleanup
    print("\n🧹 Cleaning up...")
    await agent.cleanup()
    print("✅ Demo completed!")

if __name__ == "__main__":
    asyncio.run(main())
