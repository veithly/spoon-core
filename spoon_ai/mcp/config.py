import os
from typing import Optional
from pydantic import BaseModel, Field

class MCPConfig(BaseModel):
    """
    Configuration for MCP client
    """
    server_url: str = Field(
        default="ws://localhost:8765",
        description="WebSocket URL for the MCP server"
    )
    auth_token: Optional[str] = Field(
        default=None,
        description="Authentication token for the server"
    )
    auto_reconnect: bool = Field(
        default=True,
        description="Whether to automatically reconnect if disconnected"
    )
    reconnect_delay: float = Field(
        default=5.0,
        description="Delay in seconds before attempting to reconnect"
    )
    heartbeat_interval: float = Field(
        default=30.0,
        description="Interval in seconds for sending heartbeat messages"
    )
    connection_timeout: float = Field(
        default=10.0,
        description="Timeout in seconds for connection attempts"
    )
    
    @classmethod
    def from_env(cls):
        """
        Create a configuration instance from environment variables
        
        Returns:
            MCPConfig: Configuration instance
        """
        return cls(
            server_url=os.environ.get("MCP_SERVER_URL", "ws://localhost:8765"),
            auth_token=os.environ.get("MCP_AUTH_TOKEN"),
            auto_reconnect=os.environ.get("MCP_AUTO_RECONNECT", "true").lower() == "true",
            reconnect_delay=float(os.environ.get("MCP_RECONNECT_DELAY", "5.0")),
            heartbeat_interval=float(os.environ.get("MCP_HEARTBEAT_INTERVAL", "30.0")),
            connection_timeout=float(os.environ.get("MCP_CONNECTION_TIMEOUT", "10.0"))
        )
