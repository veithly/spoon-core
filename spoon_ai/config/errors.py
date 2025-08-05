"""Configuration error classes."""


class ConfigurationError(Exception):
    """Base class for configuration errors."""
    pass


class ToolConfigurationError(ConfigurationError):
    """Tool-specific configuration error."""
    
    def __init__(self, tool_name: str, message: str):
        self.tool_name = tool_name
        super().__init__(f"Tool '{tool_name}': {message}")


class MCPServerError(ConfigurationError):
    """MCP server configuration or runtime error."""
    
    def __init__(self, server_id: str, message: str):
        self.server_id = server_id
        super().__init__(f"MCP Server '{server_id}': {message}")


class ValidationError(ConfigurationError):
    """Configuration validation error."""
    pass


class MigrationError(ConfigurationError):
    """Configuration migration error."""
    pass