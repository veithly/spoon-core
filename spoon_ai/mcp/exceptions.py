class MCPError(Exception):
    """Base exception for MCP client errors"""
    pass


class MCPConnectionError(MCPError):
    """Exception raised for connection errors"""
    pass


class MCPAuthenticationError(MCPError):
    """Exception raised for authentication errors"""
    pass


class MCPMessageError(MCPError):
    """Exception raised for message sending/receiving errors"""
    pass


class MCPSubscriptionError(MCPError):
    """Exception raised for subscription errors"""
    pass
