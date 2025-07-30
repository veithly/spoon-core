# Configuration Examples

This directory contains example configuration files demonstrating various SpoonAI agent setups.

## Files

### `config_with_mcp.json`

A comprehensive configuration example showing:

- **Multiple Agent Types**: Search, crypto, development, and general-purpose agents
- **MCP Server Integration**: Tavily for web search and GitHub for code management
- **Tool Set Configuration**: Built-in and MCP-based tool sets
- **Agent Specialization**: Different agents optimized for specific tasks

## Usage

1. Copy the desired configuration file to your working directory as `config.json`
2. Update API keys and environment variables
3. Install required MCP servers:
   ```bash
   # For Tavily search
   npm install -g tavily-mcp
   
   # For GitHub integration
   npm install -g @modelcontextprotocol/server-github
   ```
4. Set environment variables:
   ```bash
   export TAVILY_API_KEY="your-tavily-api-key"
   export GITHUB_PERSONAL_ACCESS_TOKEN="your-github-token"
   ```
5. Start SpoonAI and load your desired agent

## Agent Types

### Search Agent (`search_agent`)
- **Purpose**: Web search and information retrieval
- **Tools**: Tavily web search + crypto tools
- **Best for**: Research, fact-checking, current events

### Crypto Agent (`crypto_agent`)
- **Purpose**: Cryptocurrency and blockchain operations
- **Tools**: Built-in crypto tools only
- **Best for**: Trading analysis, DeFi operations, token research

### Development Agent (`dev_agent`)
- **Purpose**: Software development and code management
- **Tools**: GitHub integration + crypto tools
- **Best for**: Code review, repository management, development tasks

### General Agent (`general_agent`)
- **Purpose**: Multi-purpose agent with all capabilities
- **Tools**: All available tool sets
- **Best for**: Complex tasks requiring multiple tool types

## MCP Server Configuration

### Tavily Search
```json
{
  "tavily-mcp": {
    "command": "npx",
    "args": ["-y", "tavily-mcp"],
    "env": {
      "TAVILY_API_KEY": "your-api-key"
    }
  }
}
```

### GitHub Integration
```json
{
  "github-mcp": {
    "command": "npx",
    "args": ["-y", "@modelcontextprotocol/server-github"],
    "env": {
      "GITHUB_PERSONAL_ACCESS_TOKEN": "your-token"
    }
  }
}
```

## Tool Set Types

### Built-in Tools
- **crypto_tools**: Cryptocurrency analysis and trading tools
- **Type**: `builtin`
- **Configuration**: Enable/disable only

### MCP Server Tools
- **web_search**: Web search via Tavily
- **github_tools**: GitHub repository management
- **Type**: `mcp_server`
- **Configuration**: Requires MCP server setup

## Best Practices

1. **Start Simple**: Begin with basic agents and add complexity gradually
2. **Test MCP Servers**: Verify MCP server connectivity before deployment
3. **Secure API Keys**: Use environment variables for sensitive data
4. **Agent Specialization**: Create specialized agents for specific workflows
5. **Tool Optimization**: Only enable tools needed for each agent's purpose

## Troubleshooting

### MCP Server Issues
- Verify Node.js and npm are installed
- Check network connectivity
- Ensure API keys are valid
- Review server logs for errors

### Agent Loading Problems
- Validate JSON syntax
- Check agent class names
- Verify tool set references
- Review configuration file path

## Advanced Configuration

For more advanced configurations, see the main documentation at `doc/agent_configuration.md`.
