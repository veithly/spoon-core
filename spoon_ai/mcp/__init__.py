"""
MCP (Model Context Protocol) integration for SpoonAI using FastMCP.
"""

from spoon_ai.agents.base import BaseAgent
from .client import FastMCPClient
from .adapter import MCPAgentAdapter

__version__ = "0.1.0"
__all__ = ["FastMCPClient", "MCPAgentAdapter"] 