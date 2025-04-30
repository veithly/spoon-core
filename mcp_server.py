from fastmcp import FastMCP

# Create a named server
mcp = FastMCP("My App")

@mcp.tool()
def echo(message: str) -> str:
    """回显用户的输入信息"""
    return f"服务器收到消息: {message}"


if __name__ == "__main__":
    mcp.run(port=8765, transport="sse")