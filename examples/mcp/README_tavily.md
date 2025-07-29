# Tavily Search Agent with MCP stdio Integration

This example demonstrates how to integrate the Tavily search API with SpoonAI using the MCP (Model Context Protocol) stdio transport mode.

## Overview

The Tavily Search Agent provides intelligent web search capabilities by connecting to the `tavily-mcp` server using stdio transport. This allows for real-time web search and information retrieval within your AI agent workflows.

## Features

- 🔍 **Intelligent Web Search**: Powered by Tavily's advanced search API
- 🔄 **Real-time Results**: Get up-to-date information from the web
- 🎯 **Context-aware**: Search results are optimized for AI consumption
- 📡 **MCP Integration**: Uses stdio transport for reliable communication
- 🛠️ **Easy Setup**: Simple configuration and deployment

## Prerequisites

1. **Tavily API Key**: Get your free API key from [Tavily](https://tavily.com/)
2. **Node.js**: Required for running the tavily-mcp server via npx
3. **SpoonAI**: Make sure you have the spoon-ai package installed

## Installation

1. **Install Node.js dependencies** (if not already available):
   ```bash
   # The tavily-mcp package will be automatically installed via npx
   # No manual installation required
   ```

2. **Set up your Tavily API key**:
   ```bash
   export TAVILY_API_KEY=your_actual_tavily_api_key_here
   ```

3. **Install Python dependencies**:
   ```bash
   pip install spoon-ai fastmcp
   ```

## Configuration

### For Claude Desktop (or other MCP clients)

Add the following configuration to your MCP client config file (e.g., `claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "tavily-mcp": {
      "command": "npx",
      "args": ["-y", "tavily-mcp"],
      "env": {
        "TAVILY_API_KEY": "your-api-key-here"
      },
      "disabled": false,
      "autoApprove": []
    }
  }
}
```

### For SpoonAI Agent

The agent automatically configures the stdio transport. Just ensure your API key is set in the environment.

## Usage

### Running the Example

1. **Set your API key**:
   ```bash
   export TAVILY_API_KEY=your_actual_api_key
   ```

2. **Run the agent**:
   ```bash
   cd spoon-core/examples/mcp
   python tavily_search_agent.py
   ```

### Example Queries

The agent can handle various types of search queries:

- **Current Events**: "What are the latest developments in AI?"
- **Research**: "Find information about quantum computing breakthroughs"
- **How-to Guides**: "Best practices for Python web development"
- **News**: "Recent news about renewable energy"

### Programmatic Usage

```python
from tavily_search_agent import TavilySearchAgent
from spoon_ai.chat import ChatBot

# Create the agent
agent = TavilySearchAgent(
    llm=ChatBot(
        llm_provider="openai",
        model_name="gpt-4",
        base_url="your_llm_endpoint"
    )
)

# Initialize and run a search
await agent.initialize()
result = await agent.run("Search for the latest AI research papers")
print(result)
```

## Architecture

```
[User Query]
     ↓
[TavilySearchAgent] 🧠
     ↓ stdio transport
[tavily-mcp server] (npx)
     ↓ HTTP API
[Tavily Search API]
     ↓
[Web Search Results]
```

## Key Components

1. **TavilySearchAgent**: Main agent class that inherits from SpoonReactAI and MCPClientMixin
2. **NpxStdioTransport**: Handles communication with the tavily-mcp server via stdio
3. **tavily-mcp**: Node.js MCP server that interfaces with Tavily's API

## Troubleshooting

### Common Issues

1. **API Key Not Set**:
   ```
   Error: TAVILY_API_KEY environment variable not set
   ```
   **Solution**: Set your API key: `export TAVILY_API_KEY=your_key`

2. **npx Command Not Found**:
   ```
   Error: npx command not found
   ```
   **Solution**: Install Node.js from [nodejs.org](https://nodejs.org/)

3. **Connection Timeout**:
   ```
   Error: Failed to connect to MCP server
   ```
   **Solution**: Check your internet connection and API key validity

### Debug Mode

Enable debug logging to troubleshoot issues:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Advanced Configuration

### Custom Search Parameters

You can customize search behavior by modifying the agent:

```python
# Custom search with specific parameters
result = await agent.call_mcp_tool(
    "tavily_search",
    query="your search query",
    max_results=10,
    search_depth="advanced",
    include_domains=["example.com"],
    exclude_domains=["spam.com"]
)
```

### Environment Variables

- `TAVILY_API_KEY`: Your Tavily API key (required)
- `TAVILY_MAX_RESULTS`: Default maximum results (optional, default: 5)
- `TAVILY_TIMEOUT`: Request timeout in seconds (optional, default: 30)

## Related Examples

- `SpoonThirdWebagent.py`: Blockchain data querying with MCP
- `my_agent_demo.py`: Basic agent with custom tools

## Support

For issues and questions:
- Check the [Tavily documentation](https://docs.tavily.com/)
- Review the [MCP specification](https://modelcontextprotocol.io/)
- Open an issue in the SpoonAI repository

## License

This example is part of the SpoonAI project and follows the same license terms.
