from spoon_ai.mcp import FastMCPClient, MCPAgentAdapter
from spoon_ai.agents.base import BaseAgent
from spoon_ai.chat import ChatBot
import asyncio
from pathlib import Path

# Specify server script path
server_script_path = str(Path(__file__).parent / "server_example.py")

# Create a simple agent
agent = BaseAgent(
    name="my-agent",
    description="Test agent",
    llm=ChatBot()
)

# Create adapter
adapter = MCPAgentAdapter(agent)

async def main():
    # Connect to server using FastMCPClient
    async with FastMCPClient(
        server_path=server_script_path,
        transport_type="stdio"
    ) as client:
        # Register agent message handler
        client.register_agent_message_handler(agent.name, adapter.handle_mcp_message)
        
        # List available tools
        tools = await client.list_tools()
        print(f"Available tools: {tools}")
        
        # Call echo_tool
        result = await client.call_tool("echo_tool", {"message": "Hello, world!"})
        print(f"Tool result: {result}")
        
        # Read resource
        status = await client.read_resource("status://system")
        print(f"System status: {status}")

# Run main function
asyncio.run(main())