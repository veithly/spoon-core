# MCP Mode Usage: Stdio, HTTP & SSE Tool Integration in SpoonOS

This guide explains **how to configure and use MCP tools in SpoonOS**, focusing on the three main integration methods:
**Stdio-based MCP tools** (recommended for most modern tools), **HTTP-based MCP tools** (for cloud services), and **SSE-based MCP tools** (for legacy/custom servers).

---

## 1. What is MCP?

**MCP (Model Context Protocol)** is a unified protocol for integrating external tools and services (like Tavily, Brave, GitHub, custom Web3 APIs) into SpoonOS agents.
MCP tools can be provided by:
- **Stdio tools**: Tools that communicate via standard input/output (Stdio), typically started as a subprocess by SpoonOS.
- **HTTP tools**: Tools that communicate over HTTP requests, typically cloud-hosted services or APIs.
- **SSE tools**: Tools running as local/remote SSE servers, communicating via Server-Sent Events over HTTP.

---

## 2. Recommended: Stdio MCP Tools

### What are Stdio MCP Tools?

- **Stdio tools** are MCP tools that run as a subprocess and communicate with SpoonOS via stdin/stdout.
- No need to run or configure any server manually.
- Just declare the tool and its command in your agent's config, and SpoonOS will auto-start and manage the tool process.

### How to Use Stdio MCP Tools

**Step 1: Add the tool to your agent config**

```json
{
  "agents": {
    "my_agent": {
      "class": "SpoonReactMCP",
      "tools": [
        {
          "name": "tavily_search",
          "type": "mcp",
          "mcp_server": {
            "command": "npx",
            "args": ["-y", "tavily-mcp"],
            "transport": "stdio"
          }
        },
        {
          "name": "github_tools",
          "type": "mcp",
          "mcp_server": {
            "command": "npx",
            "args": ["-y", "@modelcontextprotocol/server-github"],
            "transport": "stdio"
          }
        }
      ]
    }
  }
}
```

- `"transport": "stdio"` tells SpoonOS to use Stdio for communication.
- SpoonOS will auto-start the process and manage its lifecycle.

**Step 2: Use the agent as normal**
SpoonOS will handle all Stdio tool connections automatically.

---

## 3. HTTP MCP Tools (Cloud Services)

### What are HTTP MCP Tools?

- **HTTP tools** are MCP tools that run as remote HTTP services, typically cloud-hosted APIs.
- No need to run or configure any server manually.
- Just declare the tool and its HTTP endpoint in your agent's config, and SpoonOS will communicate via HTTP.

### How to Use HTTP MCP Tools

**Step 1: Add the tool to your agent config**

**Example: Context7 Documentation Service**
```json
{
  "agents": {
    "docs_agent": {
      "class": "SpoonReactMCP",
      "tools": [
        {
          "name": "context7_docs",
          "type": "mcp",
          "mcp_server": {
            "url": "https://mcp.context7.com/mcp",
            "transport": "http",
            "timeout": 30,
            "headers": {
              "User-Agent": "SpoonOS-Agent/1.0"
            }
          }
        }
      ]
    }
  }
}
```

- `"transport": "http"` tells SpoonOS to use HTTP for communication.
- `"url"` specifies the HTTP MCP server endpoint.
- Optional `"headers"` can be added for authentication or user agent identification.

**Step 2: Use the agent as normal**
SpoonOS will handle all HTTP tool connections automatically.

---

## 4. Advanced: SSE MCP Tools (Custom/Legacy)

### What are SSE Tools?

- **SSE tools** are MCP tools you run yourself, typically for custom or in-development integrations.
- You must start the SSE server and specify its endpoint in your config.

### How to Use SSE Tools

**Step 1: Set up your SSE MCP server**

For **custom SSE servers**:
```bash
python my_mcp_server.py
# or
npx -y @modelcontextprotocol/server-github
```

For **cloud-based SSE services** like Firecrawl:
- No server setup needed
- Use the provided SSE endpoint with your API key

**Step 2: Add the tool to your agent config**

**Example: Firecrawl SSE Service**
```json
{
  "agents": {
    "scraping_agent": {
      "class": "SpoonReactMCP",
      "tools": [
        {
          "name": "firecrawl_scraper",
          "type": "mcp",
          "mcp_server": {
            "url": "https://mcp.firecrawl.dev/{FIRECRAWL_API_KEY}/sse",
            "transport": "sse",
            "timeout": 60,
            "reconnect_interval": 5,
            "headers": {
              "Accept": "text/event-stream",
              "Cache-Control": "no-cache"
            }
          }
        }
      ]
    }
  }
}
```

- `"transport": "sse"` is used for SSE tools.
- `"endpoint"` specifies the SSE server URL.
- You are responsible for starting and managing the SSE server process.

**Step 3: Use the agent as normal**
SpoonOS will connect to your SSE server and expose the tool to the agent.

---

## 5. Configuration Summary Table

| Tool Type   | Config `"type"` | Transport   | How to Use/Connect                | Example Use Case                |
|-------------|-----------------|-------------|-----------------------------------|---------------------------------|
| Stdio MCP   | `"mcp"`         | `"stdio"`   | Auto-managed subprocess           | Tavily, GitHub, Brave, etc.     |
| HTTP MCP    | `"mcp"`         | `"http"`    | HTTP requests to remote service   | Context7, cloud APIs, etc.      |
| SSE MCP     | `"mcp"`         | `"sse"`     | Cloud or custom SSE server        | Firecrawl, custom Web3, etc.    |
| Built-in    | `"builtin"`     | N/A         | Provided by SpoonOS               | Crypto tools, price data, etc.  |

---

## 6. Example: Mixed Agent Config

```json
{
  "agents": {
    "research_agent": {
      "class": "SpoonReactMCP",
      "tools": [
        {
          "name": "tavily_search",
          "type": "mcp",
          "mcp_server": {
            "command": "npx",
            "args": ["-y", "tavily-mcp"],
            "transport": "stdio"
          }
        },
        {
          "name": "context7_docs",
          "type": "mcp",
          "mcp_server": {
            "url": "https://mcp.context7.com/mcp",
            "transport": "http",
            "timeout": 30,
            "headers": {
              "User-Agent": "SpoonOS-Agent/1.0"
            }
          }
        },
        {
          "name": "firecrawl_scraper",
          "type": "mcp",
          "mcp_server": {
            "url": "https://mcp.firecrawl.dev/{FIRECRAWL_API_KEY}/sse",
            "transport": "sse",
            "timeout": 60,
            "headers": {
              "Accept": "text/event-stream",
              "Cache-Control": "no-cache"
            }
          }
        },
        { "name": "crypto_tools", "type": "builtin" }
      ]
    }
  }
}
```

---

## 7. Best Practices

- **Prefer Stdio MCP tools** for most use cases—no server management, always up to date, auto-restart.
- Use **HTTP MCP tools** for cloud services and external APIs that provide MCP endpoints.
- Use **SSE tools** only for custom or in-development integrations with real-time streaming needs.
- You can mix Stdio, HTTP, SSE, and built-in tools in a single agent.
- All tool configuration is managed in `config.json`—no need to modify code for tool integration.

---

## 8. Troubleshooting

**Stdio Tools:**
- If a Stdio tool is not available, check the command and args in your config.
- Ensure Node.js is installed for npx-based tools.

**HTTP Tools:**
- Check network connectivity and ensure the HTTP server is accessible.
- Verify the URL endpoint is correct and responding.
- Check timeout settings if requests are taking too long.
- Verify headers are set correctly for authentication or API requirements.
- Ensure the HTTP MCP service is properly implementing the MCP protocol.

**SSE Tools:**
- For custom SSE servers, ensure the server is running and the endpoint is correct.
- For cloud-based services like Firecrawl, verify your API key is set correctly.
- Check SSE connection issues with timeout and reconnect settings.
- Verify headers are set correctly for SSE connections (Accept: text/event-stream).

**General:**
- Use the CLI or logs to see which tools are loaded for each agent.
- Check environment variables (e.g., `FIRECRAWL_API_KEY`, `TAVILY_API_KEY`).

---

## Next Steps

- [Agent Development Guide](./agent.md)
- [Configuration Guide](./configuration.md)
- [MCP Protocol Documentation](https://modelcontextprotocol.io/)