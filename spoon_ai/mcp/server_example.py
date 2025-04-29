"""
A simple FastMCP server example for SpoonAI integration.
"""

from fastmcp import FastMCP

# Create a FastMCP server
mcp = FastMCP("SpoonAI Demo Server")

# Define a simple echo tool
@mcp.tool()
def echo_tool(message: str) -> str:
    """Echo the input message."""
    return f"Server echo: {message}"

# Define a simple resource
@mcp.resource("status://system")
def get_system_status() -> str:
    """Get the system status."""
    return "System is operational"

# Define a simple template resource
@mcp.resource("agent://{agent_name}/info")
def get_agent_info(agent_name: str) -> str:
    """Get information about an agent."""
    return f"Info for agent: {agent_name}"

# Define a simple prompt
@mcp.prompt()
def echo_prompt(message: str) -> dict:
    """Create an echo prompt."""
    return {
        "messages": [
            {"role": "user", "content": {"type": "text", "text": f"Please echo: {message}"}}
        ]
    }

if __name__ == "__main__":
    # Run the server
    mcp.run() 