# MCP Mode Usage: Stdio, SSE & WebSocket MCP Tool Integration in SpoonOS

This guide explains **how to configure and use MCP tools in SpoonOS** using the supported methods:
**Stdio-based MCP tools** (recommended), **HTTP-based MCP tools** (http/https), and **SSE-based MCP tools** (http/https event streams).

---

## 1. What is MCP?

**MCP (Model Context Protocol)** is a unified protocol for integrating external tools and services (like Tavily, Brave, GitHub, custom Web3 APIs) into SpoonOS agents.

MCP tools can be provided by:

- **Stdio tools**: Tools that communicate via standard input/output (Stdio), typically started as a subprocess by SpoonOS.
- **HTTP tools**: Tools that communicate over HTTP requests, typically cloud-hosted services or APIs.
- **SSE tools**: Tools running as local/remote SSE servers, communicating via Server-Sent Events over HTTP.

---

## 2. Stdio MCP Tools

### What are Stdio MCP Tools?

- **Stdio tools** are MCP tools that run as a subprocess and communicate with SpoonOS via stdin/stdout.
- No need to run or configure any server manually.
- Just declare the tool and its command in your agent's config, and SpoonOS will auto-start and manage the tool process.
- Perfect for local CLI tools, npm packages, and command-line utilities.

### Complete Stdio MCP Configuration Example

#### Step 1: Agent Configuration (config.json)

```json
{
  "default_agent": "stdio_agent",
  "agents": {
    "stdio_agent": {
      "class": "SpoonReactMCP",
      "description": "Agent with Stdio-based MCP tools for local operations",
      "aliases": ["stdio", "local_tools"],
      "tools": [
        {
          "name": "tavily_search",
          "type": "mcp",
          "description": "Web search tool via Tavily API",
          "mcp_server": {
            "command": "npx",
            "args": ["-y", "tavily-mcp"],
            "env": { "TAVILY_API_KEY": "${TAVILY_API_KEY}" },
            "transport": "stdio"
          }
        },
        {
          "name": "github_tools",
          "type": "mcp",
          "description": "GitHub repository operations",
          "mcp_server": {
            "command": "npx",
            "args": ["-y", "@modelcontextprotocol/server-github"],
            "env": { "GITHUB_PERSONAL_ACCESS_TOKEN": "${GITHUB_TOKEN}" },
            "transport": "stdio"
          }
        },
        {
          "name": "filesystem_tools",
          "type": "mcp",
          "description": "Local file system operations",
          "mcp_server": {
            "command": "npx",
            "args": ["-y", "@modelcontextprotocol/server-filesystem", "/path/to/workspace"],
            "transport": "stdio"
          }
        }
      ]
    }
  }
}
```

#### Step 2: Python Code Usage Example

```python
#!/usr/bin/env python3
"""Stdio MCP Tools Agent Example"""

import asyncio
import os
from spoon_ai.agents.spoon_react_mcp import SpoonReactMCP
from spoon_ai.tools.mcp_tool import MCPTool
from spoon_ai.llm.manager import LLMManager

class StdioMCPAgent:
    def __init__(self):
        self.agent = None
        self.llm_manager = None
        self.tools = []

    async def initialize(self):
        """Initialize the agent with Stdio MCP tools"""
        self.llm_manager = LLMManager()

        # Configure multiple Stdio MCP tools
        tool_configs = [
            {
                "name": "tavily_search",
                "description": "Web search and research tool",
                "command": "npx",
                "args": ["-y", "tavily-mcp"],
                "env": {"TAVILY_API_KEY": os.getenv("TAVILY_API_KEY", "")},
            },
            {
                "name": "github_tools",
                "description": "GitHub repository operations",
                "command": "npx",
                "args": ["-y", "@modelcontextprotocol/server-github"],
                "env": {"GITHUB_PERSONAL_ACCESS_TOKEN": os.getenv("GITHUB_TOKEN", "")},
            },
            {
                "name": "filesystem_tools",
                "description": "Local file system operations",
                "command": "npx",
                "args": ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"],
            }
        ]

        # Create MCP tool instances
        for config in tool_configs:
            tool = MCPTool(
                name=config["name"],
                description=config["description"],
                mcp_config={
                    "command": config["command"],
                    "args": config["args"],
                    "env": config["env"],
                    "transport": "stdio"
                }
            )

            # Load tool parameters
            await tool.ensure_parameters_loaded()
            self.tools.append(tool)

        # Create the agent
        self.agent = SpoonReactMCP(
            name="stdio_mcp_agent",
            llm_manager=self.llm_manager,
            tools=self.tools
        )

        return True

    async def perform_web_search(self, query: str):
        """Perform web search using Tavily"""
        return await self.agent.run(f"Search the web for: {query}")

    async def analyze_repository(self, repo_url: str):
        """Analyze a GitHub repository"""
        return await self.agent.run(f"Analyze this GitHub repository: {repo_url}")

    async def file_operations_demo(self):
        """Demonstrate file system operations"""
        return await self.agent.run("List files in the current directory and show their details")

    async def multi_tool_workflow(self):
        """Demonstrate using multiple tools in sequence"""
        # Step 1: Search for information
        search_result = await self.perform_web_search("Python async best practices")

        # Step 2: Create a file with the findings
        file_result = await self.agent.run(
            f"Create a summary file with the following research findings: {search_result[:500]}"
        )

        return f"Search completed and summary saved: {file_result}"

    async def cleanup(self):
        """Clean up tool connections"""
        for tool in self.tools:
            await tool.cleanup()

async def main():
    """Main demonstration function"""
    print("🔧 Stdio MCP Tools Agent Demo")
    print("=" * 50)

    # Check environment variables
    if not os.getenv("TAVILY_API_KEY"):
        print("⚠️  Warning: TAVILY_API_KEY not set. Web search may not work.")
    if not os.getenv("GITHUB_TOKEN"):
        print("⚠️  Warning: GITHUB_TOKEN not set. GitHub operations may not work.")

    # Initialize agent
    agent = StdioMCPAgent()
    await agent.initialize()

    # Demo scenarios
    scenarios = [
        ("Web Search", lambda: agent.perform_web_search("Latest developments in AI")),
        ("Repository Analysis", lambda: agent.analyze_repository("https://github.com/microsoft/vscode")),
        ("File Operations", lambda: agent.file_operations_demo()),
        ("Multi-Tool Workflow", lambda: agent.multi_tool_workflow())
    ]

    for i, (name, func) in enumerate(scenarios, 1):
        print(f"\n{i}. {name}")
        print("-" * 30)

        try:
            result = await func()
            print(f"✓ Success: {result[:200]}..." if len(str(result)) > 200 else f"✓ Success: {result}")
        except Exception as e:
            print(f"✗ Error: {e}")

    # Cleanup
    await agent.cleanup()
    print("\n" + "=" * 50)
    print("Demo completed!")

if __name__ == "__main__":
    asyncio.run(main())
```

#### Step 3: Environment Setup

Before running the example, set up your environment:

```bash
# Install required packages
pip install spoon-ai

# Set environment variables
export TAVILY_API_KEY="your-tavily-api-key"
export GITHUB_TOKEN="your-github-personal-access-token"

# Ensure Node.js is installed for npx commands
node --version
npm --version
```

#### Step 4: Running the Example

```bash
# Make the script executable and run it
chmod +x stdio_mcp_demo.py
python stdio_mcp_demo.py
```

### Stdio MCP Tools Features

#### Process Management
- **Automatic Startup**: SpoonOS automatically starts the subprocess when needed
- **Lifecycle Management**: Processes are properly managed and cleaned up
- **Error Handling**: Failed processes are restarted automatically
- **Resource Monitoring**: CPU and memory usage is monitored

#### Environment Variables
```json
{
  "mcp_server": {
    "env": {
      "API_KEY": "${MY_API_KEY}",
      "DEBUG": "true",
      "LOG_LEVEL": "info"
    }
  }
}
```

#### Command Arguments
```json
{
  "mcp_server": {
    "command": "python",
    "args": ["-m", "my_mcp_server", "--port", "8080", "--debug"]
  }
}
```

---

## 3. HTTP Stream MCP Tools (StreamableHttp Transport)

### What are HTTP Stream MCP Tools?

- **HTTP Stream tools** use the `StreamableHttpTransport` from fastmcp library
- Communicate via bidirectional HTTP/HTTPS with streaming capabilities
- Ideal for cloud-hosted MCP servers that support modern HTTP streaming
- More efficient than traditional HTTP for real-time data transfer

### Complete HTTP Stream Configuration Example

#### Step 1: Agent Configuration (config.json)

```json
{
  "default_agent": "deepwiki_http_agent",
  "agents": {
    "deepwiki_http_agent": {
      "class": "SpoonReactMCP",
      "description": "DeepWiki agent using HTTP Stream transport for repository analysis",
      "aliases": ["deepwiki_http", "docs_http"],
      "tools": [
        {
          "name": "deepwiki_mcp",
          "type": "mcp",
          "description": "Get documentation for GitHub repos",
          "mcp_server": {
            "url": "https://mcp.deepwiki.com/mcp",
            "transport": "http",
            "timeout": 30,
            "headers": {
              "User-Agent": "SpoonOS-HTTP-MCP/1.0",
              "Accept": "application/json, text/event-stream",
              "Authorization": "Bearer YOUR_API_TOKEN"
            }
          }
        }
      ]
    }
  }
}
```

#### Step 2: Python Code Usage Example

```python
#!/usr/bin/env python3
"""HTTP Stream DeepWiki MCP Agent Example"""

import asyncio
from spoon_ai.agents.spoon_react_mcp import SpoonReactMCP
from spoon_ai.tools.mcp_tool import MCPTool
from spoon_ai.llm.manager import LLMManager

class HTTPDeepWikiAgent:
    def __init__(self):
        self.agent = None
        self.llm_manager = None
        self.wiki_tool = None

    async def initialize(self):
        self.llm_manager = LLMManager()

        # HTTP Stream MCP tool config
        http_config = {
            "name": "deepwiki_http",
            "type": "mcp",
            "description": "DeepWiki HTTP Stream MCP tool",
            "enabled": True,
            "mcp_server": {
                "url": "https://mcp.deepwiki.com/mcp",
                "transport": "http",
                "timeout": 30,
                "headers": {
                    "User-Agent": "SpoonOS-HTTP-MCP/1.0",
                    "Accept": "application/json, text/event-stream"
                }
            }
        }

        self.wiki_tool = MCPTool(
            name=http_config["name"],
            description=http_config["description"],
            mcp_config=http_config["mcp_server"]
        )

        await self.wiki_tool.ensure_parameters_loaded()

        self.agent = SpoonReactMCP(
            name="http_deepwiki_agent",
            llm_manager=self.llm_manager,
            tools=[self.wiki_tool]
        )

        return True

    async def query_repository(self, repo_name: str, question: str = None):
        structure = await self.wiki_tool.execute(repoName=repo_name)

        if question:
            answer = await self.wiki_tool.execute(repoName=repo_name, question=question)
            return f"Structure:\n{structure}\n\nAnswer:\n{answer}"
        return f"Structure:\n{structure}"

    async def cleanup(self):
        if self.wiki_tool:
            await self.wiki_tool.cleanup()

async def main():
    agent = HTTPDeepWikiAgent()
    await agent.initialize()

    # Query repository
    result = await agent.query_repository("XSpoonAi/spoon-core")
    print(result)

    await agent.cleanup()

if __name__ == "__main__":
    asyncio.run(main())
```

---

## 4. SSE MCP Tools (Server-Sent Events Transport)

### What are SSE MCP Tools?

- **SSE tools** use Server-Sent Events for real-time communication
- Communicate via HTTP/HTTPS with event streaming capabilities
- Ideal for MCP servers that use traditional SSE endpoints
- Good for legacy systems or specific SSE implementations

### Complete SSE Configuration Example

#### Step 1: Agent Configuration (config.json)

```json
{
  "default_agent": "deepwiki_sse_agent",
  "agents": {
    "deepwiki_sse_agent": {
      "class": "SpoonReactMCP",
      "description": "DeepWiki agent using SSE transport for repository analysis",
      "aliases": ["deepwiki_sse", "docs_sse"],
      "tools": [
        {
          "name": "deepwiki_mcp",
          "type": "mcp",
          "description": "Read specific wiki content sections",
          "mcp_server": {
            "url": "https://mcp.deepwiki.com/sse",
            "transport": "sse",
            "timeout": 30,
            "headers": {
              "User-Agent": "SpoonOS-SSE-MCP/1.0",
              "Accept": "text/event-stream"
            }
          }
        }
      ]
    }
  }
}
```

#### Step 2: Python Code Usage Example

```python
#!/usr/bin/env python3
"""SSE DeepWiki MCP Agent Example"""

import asyncio
from spoon_ai.agents.spoon_react_mcp import SpoonReactMCP
from spoon_ai.tools.mcp_tool import MCPTool
from spoon_ai.llm.manager import LLMManager

class SSEDeepWikiAgent:
    def __init__(self):
        self.agent = None
        self.llm_manager = None
        self.wiki_tool = None

    async def initialize(self):
        self.llm_manager = LLMManager()

        # SSE MCP tool config
        sse_config = {
            "name": "deepwiki_sse",
            "type": "mcp",
            "description": "DeepWiki SSE MCP tool",
            "enabled": True,
            "mcp_server": {
                "url": "https://mcp.deepwiki.com/sse",
                "transport": "sse",
                "timeout": 30,
                "headers": {
                    "User-Agent": "SpoonOS-SSE-MCP/1.0",
                    "Accept": "text/event-stream"
                }
            }
        }

        self.wiki_tool = MCPTool(
            name=sse_config["name"],
            description=sse_config["description"],
            mcp_config=sse_config["mcp_server"]
        )

        await self.wiki_tool.ensure_parameters_loaded()

        self.agent = SpoonReactMCP(
            name="sse_deepwiki_agent",
            llm_manager=self.llm_manager,
            tools=[self.wiki_tool]
        )

        return True

    async def query_repository(self, repo_name: str, question: str = None):
        structure = await self.wiki_tool.execute(repoName=repo_name)

        if question:
            answer = await self.wiki_tool.execute(repoName=repo_name, question=question)
            return f"Structure:\n{structure}\n\nAnswer:\n{answer}"
        return f"Structure:\n{structure}"

    async def cleanup(self):
        if self.wiki_tool:
            await self.wiki_tool.cleanup()

async def main():
    agent = SSEDeepWikiAgent()
    await agent.initialize()

    # Query repository
    result = await agent.query_repository("XSpoonAi/spoon-core")
    print(result)

    await agent.cleanup()

if __name__ == "__main__":
    asyncio.run(main())
```

## 5. Configuration Summary

| Transport Type | Config `"transport"` | Protocol | Best For | Example Endpoint |
|----------------|---------------------|----------|----------|------------------|
| Stdio          | `"stdio"`          | Subprocess | Local CLI tools | `npx tavily-mcp` |
| HTTP Stream    | `"http"`           | StreamableHttp | Cloud APIs | `https://api.example.com/mcp` |
| SSE            | `"sse"`            | Server-Sent Events | Event streams | `https://api.example.com/sse` |
| Built-in       | N/A                | N/A | Native tools | N/A |

## 6. Advanced Configuration Examples

### Example: Production Environment Configuration

```json
{
  "default_agent": "production_agent",
  "agents": {
    "production_agent": {
      "class": "SpoonReactMCP",
      "description": "Production-ready agent with MCP tools",
      "tools": [
        {
          "name": "web_search",
          "type": "mcp",
          "mcp_server": {
            "command": "uvx",
            "args": ["mcp-server-brave-search"],
            "env": { "BRAVE_API_KEY": "${BRAVE_API_KEY}" },
            "transport": "stdio"
          }
        },
        {
          "name": "documentation",
          "type": "mcp",
          "mcp_server": {
            "url": "https://api.deepwiki.com/mcp",
            "transport": "http",
            "timeout": 60,
            "headers": {
              "Authorization": "Bearer ${API_KEY}",
              "User-Agent": "SpoonOS/1.0"
            }
          }
        }
      ]
    }
  }
}
```

---

## 7. Best Practices

### Transport Selection Guidelines

- **Prefer Stdio MCP tools** where possible—they're easy to manage and reliable.
- Use **HTTP Stream** for cloud-hosted APIs that support modern streaming.
- Use **SSE** for legacy systems or specific event streaming requirements.
- You can mix different transport types in a single agent for maximum flexibility.
- All tool configuration is managed in `config.json`—no need to modify code for tool integration.

### Configuration Best Practices

#### Headers and Authentication

```json
{
  "mcp_server": {
    "headers": {
      "User-Agent": "SpoonOS/1.0",
      "Authorization": "Bearer ${API_KEY}",
      "Accept": "application/json, text/event-stream"
    }
  }
}
```

#### Timeout and Retry Configuration

```json
{
  "mcp_server": {
    "timeout": 30,
    "retry_attempts": 3
  }
}
```

#### Environment Variables

```json
{
  "mcp_server": {
    "env": {
      "API_KEY": "${MY_API_KEY}"
    }
  }
}
```

### Performance Optimization

- Use appropriate timeouts (30-60 seconds for HTTP, 300+ seconds for SSE)
- Configure retry attempts (2-3 retries recommended)
- Enable connection pooling for high-traffic scenarios
- Monitor rate limits and implement backoff strategies

---

## 8. Troubleshooting

### Stdio Transport Issues

**Common Problems:**

- **Tool not found**: Check the `command` and `args` in your config
- **Node.js missing**: Ensure Node.js is installed for `npx`-based tools
- **Python missing**: Ensure Python is installed for `python`-based tools
- **Package not installed**: Run `npx -y package-name` or `pip install package-name`

**Debug Commands:**

```bash
npx tavily-mcp --help
uvx mcp-server-brave-search --help
```

### HTTP Stream Transport Issues

**Common Problems:**

- **Connection timeout**: Check network connectivity and firewall
- **SSL certificate errors**: Verify server certificate or use `verify_ssl: false`
- **Authentication failed**: Check API keys and authorization headers
- **Rate limiting**: Implement retry logic with exponential backoff

**Network Debugging:**
```bash
curl -I https://api.example.com/mcp
curl -H "Authorization: Bearer YOUR_TOKEN" https://api.example.com/mcp
```

### SSE Transport Issues

**Common Problems:**
- **Connection failed**: Verify SSE endpoint supports GET requests
- **405 Method Not Allowed**: Some servers require POST for SSE connections
- **Event parsing errors**: Check SSE event format compliance
- **Connection drops**: Implement reconnection logic

**SSE Testing:**
```bash
curl -H "Accept: text/event-stream" https://api.example.com/sse
```

### General Debugging

**Environment Variables:**
```bash
echo $TAVILY_API_KEY $BRAVE_API_KEY $OPENAI_API_KEY
```

**Configuration Validation:**
```json
{
  "mcp_server": {
    "timeout": 30,
    "headers": {
      "User-Agent": "SpoonOS/1.0",
      "Authorization": "Bearer ${API_KEY}"
    }
  }
}
```

### Performance Issues

**High-throughput Configuration:**
```json
{
  "mcp_server": {
    "connection_pool_size": 10,
    "keep_alive": true,
    "timeout": 60,
    "retry_attempts": 3
  }
}
```

---

## 9. Real-world Examples: Tavily Search Integration

### What is Tavily Search?

**Tavily Search** is a powerful web search API that provides high-quality, structured search results. It's one of the most popular MCP tools for web research and information gathering.

### Complete Tavily Search Configuration

#### Step 1: Environment Setup

First, get your Tavily API key from [Tavily](https://tavily.com) and set it as an environment variable:

```bash
export TAVILY_API_KEY="your-tavily-api-key-here"
```

#### Step 2: Agent Configuration (config.json)

```json
{
  "default_agent": "research_agent",
  "agents": {
    "research_agent": {
      "class": "SpoonReactMCP",
      "description": "AI research agent with web search capabilities",
      "aliases": ["research", "search"],
      "tools": [
        {
          "name": "tavily_search",
          "type": "mcp",
          "description": "Web search and research tool powered by Tavily",
          "mcp_server": {
            "command": "npx",
            "args": ["-y", "tavily-mcp"],
            "env": { "TAVILY_API_KEY": "${TAVILY_API_KEY}" },
            "transport": "stdio"
          }
        }
      ]
    }
  }
}
```

#### Step 3: Usage Examples

**Basic Web Search:**
```python
#!/usr/bin/env python3
"""Tavily Search Agent Example"""

import asyncio
from spoon_ai.agents.spoon_react_mcp import SpoonReactMCP
from spoon_ai.tools.mcp_tool import MCPTool
from spoon_ai.llm.manager import LLMManager

async def create_research_agent():
    """Create an agent with Tavily search capabilities"""

    # Configure Tavily Search tool
    search_tool = MCPTool(
        name="tavily_search",
        description="Web search and research tool",
        mcp_config={
            "command": "npx",
            "args": ["-y", "tavily-mcp"],
            "env": {"TAVILY_API_KEY": "your-api-key"},
            "transport": "stdio"
        }
    )

    # Create agent
    agent = SpoonReactMCP(
        name="research_agent",
        llm_manager=LLMManager(),
        tools=[search_tool]
    )

    return agent

async def main():
    print("🔍 Tavily Search Agent Demo")
    print("=" * 40)

    agent = await create_research_agent()

    # Example queries
    queries = [
        "What are the latest developments in AI safety research?",
        "Compare React vs Vue.js for modern web development",
        "What are the best practices for Python async programming?"
    ]

    for i, query in enumerate(queries, 1):
        print(f"\n📋 Query {i}: {query}")
        print("-" * 50)

        try:
            result = await agent.run(query)
            print(f"Result: {result[:300]}..." if len(result) > 300 else f"Result: {result}")
        except Exception as e:
            print(f"Error: {e}")

        print()

    # Cleanup
    await search_tool.cleanup()

if __name__ == "__main__":
    asyncio.run(main())
```

**Advanced Research Queries:**
```python
async def advanced_research():
    agent = await create_research_agent()

    # Complex research queries
    complex_queries = [
        "Find recent papers on transformer architecture improvements from 2023-2024",
        "Compare cloud costs: AWS vs Google Cloud vs Azure for ML workloads",
        "What are the current trends in decentralized finance (DeFi) protocols?"
    ]

    for query in complex_queries:
        print(f"\n🔬 Researching: {query}")

        result = await agent.run(
            f"Please provide a comprehensive analysis of: {query}. "
            "Include key findings, statistics, and expert opinions."
        )

        print(f"Analysis: {result[:500]}...")
        print("-" * 80)

asyncio.run(advanced_research())
```

### Tavily Search Features

#### Search Configuration Options

**Basic Search** - Standard web search
```json
{
  "mcp_server": {
    "command": "npx",
    "args": ["-y", "tavily-mcp"],
    "env": { "TAVILY_API_KEY": "${TAVILY_API_KEY}" },
    "transport": "stdio"
  }
}
```

**Advanced Search** - With custom parameters
```json
{
  "mcp_server": {
    "command": "npx",
    "args": ["-y", "tavily-mcp"],
    "env": {
      "TAVILY_API_KEY": "${TAVILY_API_KEY}",
      "TAVILY_SEARCH_DEPTH": "advanced",
      "TAVILY_MAX_RESULTS": "10"
    },
    "transport": "stdio"
  }
}
```

#### Search Result Structure

Tavily Search typically returns structured results like:
```json
{
  "title": "Latest AI Developments",
  "url": "https://example.com/ai-news",
  "content": "Comprehensive article about recent AI advancements...",
  "score": 0.95,
  "published_date": "2024-01-15"
}
```

### Multi-Tool Research Agent

#### Configuration with Multiple Search Tools

```json
{
  "default_agent": "multi_search_agent",
  "agents": {
    "multi_search_agent": {
      "class": "SpoonReactMCP",
      "description": "Advanced research agent with multiple search capabilities",
      "tools": [
        {
          "name": "tavily_search",
          "type": "mcp",
          "description": "Primary web search tool",
          "mcp_server": {
            "command": "npx",
            "args": ["-y", "tavily-mcp"],
            "env": { "TAVILY_API_KEY": "${TAVILY_API_KEY}" },
            "transport": "stdio"
          }
        },
        {
          "name": "deepwiki_docs",
          "type": "mcp",
          "description": "Documentation search",
          "mcp_server": {
            "url": "https://mcp.deepwiki.com/mcp",
            "transport": "http",
            "timeout": 30
          }
        }
      ]
    }
  }
}
```

#### Multi-Tool Research Example

```python
async def comprehensive_research():
    """Use multiple tools for comprehensive research"""

    # Create multi-tool agent
    agent = SpoonReactMCP(
        name="comprehensive_research",
        llm_manager=LLMManager(),
        tools=[
            MCPTool(name="tavily", mcp_config={"command": "npx", "args": ["-y", "tavily-mcp"], "transport": "stdio"}),
            MCPTool(name="deepwiki", mcp_config={"url": "https://mcp.deepwiki.com/mcp", "transport": "http"})
        ]
    )

    query = "Analyze the current state of quantum computing and its potential impact on cryptography"

    result = await agent.run(
        f"Conduct comprehensive research on: {query}. "
        "Use web search for recent developments and documentation search for technical details."
    )

    print(f"Comprehensive Analysis:\n{result}")

asyncio.run(comprehensive_research())
```

### Tavily Search Best Practices

#### Query Optimization
- Use specific, focused queries rather than broad questions
- Include relevant keywords and context
- Specify date ranges for time-sensitive topics

#### Rate Limiting
- Tavily has rate limits; implement retry logic
- Use environment variables for API key management
- Monitor usage to avoid hitting limits

#### Result Processing
- Parse structured results for better analysis
- Combine multiple search queries for comprehensive coverage
- Filter and rank results based on relevance scores

---

## 10. Troubleshooting

### Tavily Search Issues

**API Key Problems:**
```bash
# Check if API key is set
echo $TAVILY_API_KEY

# Test API key validity
curl -H "Authorization: Bearer $TAVILY_API_KEY" https://api.tavily.com/search
```

**Installation Issues:**
```bash
# Install Tavily MCP
npm install -g tavily-mcp

# Test installation
npx tavily-mcp --help
```

**Rate Limiting:**
- Tavily has rate limits based on your plan
- Implement exponential backoff for retries
- Monitor usage in your Tavily dashboard

### Stdio Transport Issues

**Common Problems:**
- **Tool not found**: Check the `command` and `args` in your config
- **Node.js missing**: Ensure Node.js is installed for `npx`-based tools
- **Python missing**: Ensure Python is installed for `python`-based tools
- **Package not installed**: Run `npx -y package-name` or `pip install package-name`

**Debug Commands:**
```bash
npx tavily-mcp --help
uvx mcp-server-brave-search --help
```

### HTTP Stream Transport Issues

**Common Problems:**
- **Connection timeout**: Check network connectivity and firewall
- **SSL certificate errors**: Verify server certificate or use `verify_ssl: false`
- **Authentication failed**: Check API keys and authorization headers
- **Rate limiting**: Implement retry logic with exponential backoff

**Network Debugging:**
```bash
curl -I https://api.example.com/mcp
curl -H "Authorization: Bearer YOUR_TOKEN" https://api.example.com/mcp
```

### SSE Transport Issues

**Common Problems:**
- **Connection failed**: Verify SSE endpoint supports GET requests
- **405 Method Not Allowed**: Some servers require POST for SSE connections
- **Event parsing errors**: Check SSE event format compliance
- **Connection drops**: Implement reconnection logic

**SSE Testing:**
```bash
curl -H "Accept: text/event-stream" https://api.example.com/sse
```

### General Debugging

**Environment Variables:**
```bash
echo $TAVILY_API_KEY $BRAVE_API_KEY $OPENAI_API_KEY
```

**Configuration Validation:**
```json
{
  "mcp_server": {
    "timeout": 30,
    "headers": {
      "User-Agent": "SpoonOS/1.0",
      "Authorization": "Bearer ${API_KEY}"
    }
  }
}
```

---

## 11. Next Steps

### Learn More

- [Agent Development Guide](./agent.md)
- [Configuration Guide](./configuration.md)
- [CLI Usage Guide](./cli.md)
- [MCP Protocol Documentation](https://modelcontextprotocol.io/)
