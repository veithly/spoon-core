"""Pydantic models for unified tool configuration."""

from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field, field_validator, model_validator

from .errors import ValidationError


class MCPServerConfig(BaseModel):
    """Configuration for MCP server."""

    # For stdio-based servers
    command: Optional[str] = Field(None, description="Command to start MCP server")
    args: List[str] = Field(default_factory=list, description="Command arguments")
    env: Dict[str, str] = Field(default_factory=dict, description="Environment variables")
    cwd: Optional[str] = Field(None, description="Working directory")

    # For URL-based servers (SSE/WebSocket)
    url: Optional[str] = Field(None, description="Remote MCP server URL (http(s) for SSE, ws(s) for WebSocket)")
    headers: Dict[str, str] = Field(default_factory=dict, description="Optional HTTP headers for URL transports")
    reconnect_interval: Optional[int] = Field(None, description="Reconnect interval (seconds) for SSE/WebSocket")

    # Common options
    disabled: bool = Field(False, description="Server disabled flag")
    autoApprove: List[str] = Field(default_factory=list, description="Auto-approved tool names")
    transport: Literal["auto", "npx", "python", "uvx", "sse", "websocket"] = Field(
        "auto", description="Transport method"
    )
    timeout: int = Field(30, description="Connection timeout in seconds")
    retry_attempts: int = Field(3, description="Number of retry attempts")

    @field_validator('timeout')
    @classmethod
    def validate_timeout(cls, v):
        if v <= 0:
            raise ValidationError("Timeout must be positive")
        return v

    @field_validator('retry_attempts')
    @classmethod
    def validate_retry_attempts(cls, v):
        if v < 0:
            raise ValidationError("Retry attempts must be non-negative")
        return v

    @model_validator(mode='after')
    def validate_transport_config(self):
        """Ensure required fields exist for the chosen transport type."""
        # Require either url or command
        if not self.url and not self.command:
            raise ValidationError("MCP server config must include either 'url' (SSE/WebSocket) or 'command' (stdio)")

        if self.transport in ["sse", "websocket"] or (self.url is not None):
            if not self.url:
                raise ValidationError("URL-based transports require 'url' to be set")
        else:
            if not self.command:
                raise ValidationError("Stdio transports require 'command' to be set")

        return self


class ToolConfig(BaseModel):
    """Configuration for a tool."""

    name: str = Field(..., description="Unique tool identifier")
    type: Literal["mcp", "builtin", "external"] = Field(..., description="Tool type")
    description: Optional[str] = Field(None, description="Tool description")
    enabled: bool = Field(True, description="Whether tool is enabled")
    mcp_server: Optional[MCPServerConfig] = Field(None, description="MCP server configuration")
    config: Dict[str, Any] = Field(default_factory=dict, description="Tool-specific settings")
    env: Dict[str, str] = Field(default_factory=dict, description="Environment variables for this tool")

    @model_validator(mode='after')
    def validate_mcp_config(self):
        """Validate MCP-specific configuration."""
        if self.type == "mcp" and not self.mcp_server:
            raise ValidationError("MCP tools must have mcp_server configuration")

        if self.type != "mcp" and self.mcp_server:
            raise ValidationError("Only MCP tools can have mcp_server configuration")

        return self

    @field_validator('name')
    @classmethod
    def validate_name(cls, v):
        if not v or not v.strip():
            raise ValidationError("Tool name cannot be empty")
        return v.strip()


class AgentConfig(BaseModel):
    """Configuration for an agent."""

    class_name: str = Field(..., alias="class", description="Agent class name")
    aliases: List[str] = Field(default_factory=list, description="Agent aliases")
    description: Optional[str] = Field(None, description="Agent description")
    config: Dict[str, Any] = Field(default_factory=dict, description="Agent-specific configuration")
    tools: List[ToolConfig] = Field(default_factory=list, description="Tool configurations")

    # Legacy fields for backward compatibility
    mcp_servers: Optional[List[str]] = Field(None, description="Legacy MCP servers list")

    @field_validator('class_name')
    @classmethod
    def validate_class_name(cls, v):
        if not v or not v.strip():
            raise ValidationError("Agent class name cannot be empty")
        return v.strip()

    @field_validator('tools')
    @classmethod
    def validate_unique_tool_names(cls, v):
        """Ensure tool names are unique within an agent."""
        names = [tool.name for tool in v]
        if len(names) != len(set(names)):
            duplicates = [name for name in names if names.count(name) > 1]
            raise ValidationError(f"Duplicate tool names found: {duplicates}")
        return v

    model_config = {"populate_by_name": True}


class SpoonConfig(BaseModel):
    """Complete SpoonAI configuration."""

    api_keys: Dict[str, str] = Field(default_factory=dict)
    providers: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    llm_settings: Dict[str, Any] = Field(default_factory=dict)
    default_agent: Optional[str] = None
    agents: Dict[str, AgentConfig] = Field(default_factory=dict)

    # Other configuration fields
    RPC_URL: Optional[str] = None
    SCAN_URL: Optional[str] = None
    CHAIN_ID: Optional[str] = None

    @field_validator('agents')
    @classmethod
    def validate_unique_agent_names(cls, v):
        """Ensure agent names and aliases don't conflict."""
        all_names = set()
        for name, agent in v.items():
            if name in all_names:
                raise ValidationError(f"Duplicate agent name: {name}")
            all_names.add(name)

            for alias in agent.aliases:
                if alias in all_names:
                    raise ValidationError(f"Agent alias '{alias}' conflicts with existing name")
                all_names.add(alias)

        return v