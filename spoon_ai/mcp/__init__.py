"""
MCP (Message Connectivity Protocol) client module for SpoonAI.
Provides a client for message-based communication between agents and external services.
"""

from .base import MCPClient
from .client import SpoonMCPClient, MCPMessage
from .config import MCPConfig
from .agent_adapter import MCPAgentAdapter
from .exceptions import (
    MCPError, 
    MCPConnectionError, 
    MCPAuthenticationError, 
    MCPMessageError, 
    MCPSubscriptionError
)

__all__ = [
    'MCPClient',
    'SpoonMCPClient',
    'MCPMessage',
    'MCPConfig',
    'MCPAgentAdapter',
    'MCPError',
    'MCPConnectionError',
    'MCPAuthenticationError',
    'MCPMessageError',
    'MCPSubscriptionError',
]
