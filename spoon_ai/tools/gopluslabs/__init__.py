from fastmcp import FastMCP

mcp_server = FastMCP("GoPlusLabsServer")

if __name__ == "__main__":
    mcp_server.run(host='0.0.0.0', port=8000)