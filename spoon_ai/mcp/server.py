#!/usr/bin/env python
"""
SpoonAI FastMCP Server

A deployable FastMCP server that integrates SpoonAI agents.
"""

import os
import sys
import asyncio
import logging
from typing import Dict, Any, List, Optional

from fastmcp import FastMCP

# Add parent directory to path if running as a script
if __name__ == "__main__" and __package__ is None:
    parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    sys.path.insert(0, parent_dir)
    __package__ = "spoon_ai.mcp"

from spoon_ai.agents.base import BaseAgent
from spoon_ai.chat import ChatBot
from spoon_ai.schema import AgentState
from spoon_ai.mcp.adapter import MCPAgentAdapter

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Dictionary to store available agents
AGENTS: Dict[str, BaseAgent] = {}

# Create a FastMCP server
mcp = FastMCP("SpoonAI Agent Server")

# Define agent-related resources and tools
@mcp.resource("agents://list")
async def list_available_agents() -> Dict[str, Any]:
    """List all available agents."""
    return {
        "agents": [
            {
                "name": agent.name,
                "description": agent.description,
                "state": agent.state.value
            }
            for agent in AGENTS.values()
        ]
    }

@mcp.resource("agent://{agent_name}/state")
async def get_agent_state(agent_name: str) -> str:
    """Get the state of a specific agent."""
    if agent_name not in AGENTS:
        return f"Agent '{agent_name}' not found"
    return AGENTS[agent_name].state.value

@mcp.resource("agent://{agent_name}/info")
async def get_agent_info(agent_name: str) -> Dict[str, Any]:
    """Get information about a specific agent."""
    if agent_name not in AGENTS:
        return {"error": f"Agent '{agent_name}' not found"}
    
    agent = AGENTS[agent_name]
    return {
        "name": agent.name,
        "description": agent.description,
        "state": agent.state.value,
        "max_steps": agent.max_steps,
        "current_step": agent.current_step
    }

@mcp.tool()
async def send_message_to_agent(agent_name: str, message: str, metadata: Optional[Dict[str, Any]] = None) -> str:
    """
    Send a message to a specific agent.
    
    Args:
        agent_name: Name of the agent to send the message to
        message: Content of the message
        metadata: Optional metadata to include with the message
    
    Returns:
        Agent's response
    """
    if agent_name not in AGENTS:
        return f"Error: Agent '{agent_name}' not found"
    
    agent = AGENTS[agent_name]
    agent_message = {
        "content": message,
        "sender": "user",
        "metadata": metadata or {},
        "topic": "general"
    }
    
    try:
        response = await agent.process_mcp_message(
            content=message,
            sender="user",
            message=agent_message,
            agent_id=agent.name
        )
        
        # Handle both streaming and non-streaming responses
        if asyncio.iscoroutine(response) or hasattr(response, "__aiter__"):
            # For streaming responses, collect all chunks
            chunks = []
            async for chunk in response:
                chunks.append(chunk)
            return "".join(chunks)
        else:
            # For non-streaming responses
            return str(response)
    except Exception as e:
        logger.error(f"Error processing message for agent {agent_name}: {e}", exc_info=True)
        return f"Error processing message: {str(e)}"

@mcp.tool()
async def register_agent(agent_name: str, agent_description: Optional[str] = None) -> str:
    """
    Register a new agent with the server.
    
    Args:
        agent_name: Name for the new agent
        agent_description: Optional description for the agent
        
    Returns:
        Result of the registration
    """
    if agent_name in AGENTS:
        return f"Error: Agent '{agent_name}' already exists"
    
    try:
        # Create a new agent
        agent = BaseAgent(
            name=agent_name,
            description=agent_description or f"Agent {agent_name}",
            llm=ChatBot(),
        )
        
        # Store the agent
        AGENTS[agent_name] = agent
        return f"Agent '{agent_name}' registered successfully"
    except Exception as e:
        logger.error(f"Error registering agent {agent_name}: {e}", exc_info=True)
        return f"Error registering agent: {str(e)}"

@mcp.tool()
async def unregister_agent(agent_name: str) -> str:
    """
    Unregister an agent from the server.
    
    Args:
        agent_name: Name of the agent to unregister
        
    Returns:
        Result of the unregistration
    """
    if agent_name not in AGENTS:
        return f"Error: Agent '{agent_name}' not found"
    
    try:
        # Remove the agent
        del AGENTS[agent_name]
        return f"Agent '{agent_name}' unregistered successfully"
    except Exception as e:
        logger.error(f"Error unregistering agent {agent_name}: {e}", exc_info=True)
        return f"Error unregistering agent: {str(e)}"

@mcp.tool()
async def run_agent(agent_name: str, request: Optional[str] = None) -> str:
    """
    Run an agent with an optional request.
    
    Args:
        agent_name: Name of the agent to run
        request: Optional initial request to the agent
        
    Returns:
        Result of the agent run
    """
    if agent_name not in AGENTS:
        return f"Error: Agent '{agent_name}' not found"
    
    agent = AGENTS[agent_name]
    if agent.state != AgentState.IDLE:
        return f"Error: Agent '{agent_name}' is not in IDLE state (current state: {agent.state.value})"
    
    try:
        result = await agent.run(request=request)
        return result
    except Exception as e:
        logger.error(f"Error running agent {agent_name}: {e}", exc_info=True)
        return f"Error running agent: {str(e)}"

def initialize_default_agents():
    """Initialize some default agents for demonstration purposes."""
    # Create a default echo agent
    echo_agent = BaseAgent(
        name="echo",
        description="A simple echo agent",
        llm=ChatBot(),
    )
    
    # Override the step method to make it an echo agent
    async def echo_step(self):
        messages = self.memory.get_messages()
        if not messages:
            return "No messages to process."
        
        last_message = messages[-1]
        if last_message.role.value == "user":
            response = f"Echo: {last_message.content}"
            self.add_message("assistant", response)
            return response
        
        return "No user message to echo."
    
    # Use monkey patching to add the step method
    echo_agent.step = echo_step.__get__(echo_agent)
    
    # Register the agent
    AGENTS[echo_agent.name] = echo_agent
    logger.info(f"Initialized default agent: {echo_agent.name}")

if __name__ == "__main__":
    # Initialize default agents
    initialize_default_agents()
    
    # Log available agents
    logger.info(f"Server starting with {len(AGENTS)} agent(s):")
    for agent_name in AGENTS:
        logger.info(f"  - {agent_name}")
    
    # Run the server
    logger.info("Starting FastMCP server...")
    mcp.run() 