# Agent Configuration Guide

This guide explains how to use configuration files to define and manage SpoonAI agents, implementing a configuration-driven agent loading system.

## Overview

SpoonAI supports defining custom agents through the `config.json` file, allowing users to:

- Define custom agent configurations
- Set agent aliases
- Configure agent parameters
- Dynamically load and switch agents
- Integrate MCP servers and additional tools

## Configuration File Structure

### Basic Configuration Example

```json
{
  "api_keys": {
    "openai": "your-openai-api-key",
    "anthropic": "your-anthropic-api-key",
    "deepseek": "your-deepseek-api-key"
  },
  "base_url": "https://openrouter.ai/api/v1",
  "default_agent": "spoon_react",
  "llm_providers": {
    "openai": {
      "api_key": "your-openai-key",
      "model": "gpt-4.1",
      "max_tokens": 4096,
      "temperature": 0.3,
      "timeout": 30,
      "retry_attempts": 3
    },
    "anthropic": {
      "api_key": "your-anthropic-key",
      "model": "claude-sonnet-4-20250514",
      "max_tokens": 4096,
      "temperature": 0.3,
      "timeout": 30,
      "retry_attempts": 3
    },
    "gemini": {
      "api_key": "your-gemini-key",
      "model": "gemini-2.5-pro",
      "max_tokens": 4096,
      "temperature": 0.3
    }
  },
  "llm_settings": {
    "default_provider": "openai",
    "fallback_chain": ["openai", "anthropic", "gemini"],
    "enable_monitoring": true,
    "enable_caching": true,
    "enable_debug_logging": false,
    "max_concurrent_requests": 10
  },
  "agents": {
    "custom_react": {
      "class": "SpoonReactAI",
      "aliases": ["custom", "my_react"],
      "description": "Custom configured SpoonReact agent",
      "config": {
        "max_steps": 15,
        "tool_choice": "auto",
        "llm_provider": "openai"
      }
    },
    "search_agent": {
      "class": "SpoonReactMCP",
      "aliases": ["search", "tavily"],
      "description": "Search agent with Tavily MCP integration",
      "config": {
        "max_steps": 20,
        "tool_choice": "auto",
        "llm_provider": "anthropic"
      },
      "mcp_servers": ["tavily-mcp"],
      "tools": ["web_search", "crypto_tools"]
    }
  },
  "mcp_servers": {
    "tavily-mcp": {
      "command": "npx",
      "args": ["-y", "tavily-mcp"],
      "env": {
        "TAVILY_API_KEY": "your-api-key-here"
      },
      "disabled": false,
      "autoApprove": []
    }
  },
  "tool_sets": {
    "web_search": {
      "type": "mcp_server",
      "server": "tavily-mcp"
    },
    "crypto_tools": {
      "type": "builtin",
      "enabled": true
    }
  }
}
```

## Agent Definition

### Agent Configuration Fields

Each agent definition contains the following fields:

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `class` | string | Yes | Agent class name (`SpoonReactAI` or `SpoonReactMCP`) |
| `aliases` | array | No | List of agent aliases |
| `description` | string | No | Agent description |
| `config` | object | No | Agent-specific configuration parameters |
| `mcp_servers` | array | No | List of MCP server names to connect |
| `tools` | array | No | List of tool sets to enable |

### Supported Agent Classes

- **SpoonReactAI**: Basic ReAct agent
- **SpoonReactMCP**: Advanced agent with MCP protocol support

### Agent Configuration Parameters

The `config` field can contain the following parameters:

- `max_steps`: Maximum execution steps
- `tool_choice`: Tool selection strategy (`auto`, `required`, `none`)
- `llm_provider`: Specific LLM provider for this agent (`openai`, `anthropic`, `gemini`)
- `llm_model`: Override model for this agent
- `llm_temperature`: Override temperature for this agent
- `enable_fallback`: Enable/disable fallback for this agent
- Other agent-specific parameters

## LLM Provider Configuration

### Provider Definition

LLM providers are configured in the `llm_providers` section:

```json
{
  "llm_providers": {
    "provider_name": {
      "api_key": "your-api-key",
      "model": "model-name",
      "max_tokens": 4096,
      "temperature": 0.3,
      "timeout": 30,
      "retry_attempts": 3,
      "base_url": "custom-endpoint",
      "custom_headers": {}
    }
  }
}
```

### Provider Configuration Fields

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `api_key` | string | Yes | Provider API key |
| `model` | string | No | Default model name |
| `max_tokens` | integer | No | Maximum tokens per request |
| `temperature` | float | No | Response randomness (0.0-1.0) |
| `timeout` | integer | No | Request timeout in seconds |
| `retry_attempts` | integer | No | Number of retry attempts |
| `base_url` | string | No | Custom API endpoint |
| `custom_headers` | object | No | Additional HTTP headers |

### Global LLM Settings

Configure global LLM behavior in the `llm_settings` section:

```json
{
  "llm_settings": {
    "default_provider": "openai",
    "fallback_chain": ["openai", "anthropic", "gemini"],
    "enable_monitoring": true,
    "enable_caching": true,
    "enable_debug_logging": false,
    "max_concurrent_requests": 10,
    "cache_ttl": 3600,
    "health_check_interval": 300
  }
}
```

### LLM Settings Fields

| Field | Type | Description | Default |
|-------|------|-------------|---------|
| `default_provider` | string | Default LLM provider | `"openai"` |
| `fallback_chain` | array | Provider fallback order | `[]` |
| `enable_monitoring` | boolean | Enable request monitoring | `true` |
| `enable_caching` | boolean | Enable response caching | `true` |
| `enable_debug_logging` | boolean | Enable debug logs | `false` |
| `max_concurrent_requests` | integer | Max concurrent requests | `10` |
| `cache_ttl` | integer | Cache time-to-live (seconds) | `3600` |
| `health_check_interval` | integer | Health check interval (seconds) | `300` |

### Agent-Specific LLM Configuration

Agents can override global LLM settings:

```json
{
  "agents": {
    "analytical_agent": {
      "class": "SpoonReactAI",
      "description": "Agent optimized for analytical tasks",
      "config": {
        "llm_provider": "openai",
        "llm_model": "gpt-4.1",
        "llm_temperature": 0.1,
        "max_steps": 10
      }
    },
    "creative_agent": {
      "class": "SpoonReactAI", 
      "description": "Agent optimized for creative tasks",
      "config": {
        "llm_provider": "anthropic",
        "llm_model": "claude-3-opus-20240229",
        "llm_temperature": 0.8,
        "max_steps": 15
      }
    }
  }
}
```

## MCP Server Configuration

### MCP Server Definition

MCP servers are defined in the `mcp_servers` section:

```json
{
  "mcp_servers": {
    "server_name": {
      "command": "command_to_run",
      "args": ["arg1", "arg2"],
      "env": {
        "ENV_VAR": "value"
      },
      "disabled": false,
      "autoApprove": []
    }
  }
}
```

### MCP Server Fields

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `command` | string | Yes | Command to start the MCP server |
| `args` | array | No | Command line arguments |
| `env` | object | No | Environment variables |
| `disabled` | boolean | No | Whether the server is disabled |
| `autoApprove` | array | No | Auto-approve patterns |

## Tool Set Configuration

### Tool Set Definition

Tool sets are defined in the `tool_sets` section:

```json
{
  "tool_sets": {
    "tool_set_name": {
      "type": "mcp_server|builtin",
      "server": "server_name",
      "enabled": true
    }
  }
}
```

### Tool Set Types

- **mcp_server**: Tools provided by MCP servers
- **builtin**: Built-in tool collections (e.g., crypto_tools)

## Usage

### 1. List Available Agents

```bash
> list-agents
Available agents:
  react (aliases: spoon_react): A smart ai agent in neo blockchain
  spoon_react (aliases: react): A smart ai agent in neo blockchain
  spoon_react_mcp: SpoonReact agent with MCP support
  custom_react (aliases: custom, my_react): Custom configured SpoonReact agent
  search_agent (aliases: search, tavily): Search agent with Tavily MCP integration

Currently loaded agents:
  spoon_react: A smart ai agent in neo blockchain
```

### 2. Load Agents

Using agent name:

```bash
> load-agent search_agent
Loaded agent: spoon_react
```

Using alias:

```bash
> load-agent search
Loaded agent: spoon_react
```

### 3. Error Handling

If an agent doesn't exist, the system will display all available agents:

```bash
> load-agent nonexistent
Agent nonexistent not found
Available agents:
  react (aliases: spoon_react): A smart ai agent in neo blockchain
  search_agent (aliases: search, tavily): Search agent with Tavily MCP integration
```

## Built-in Agents

The system includes the following built-in agents:

| Name | Aliases | Class | Description |
|------|---------|-------|-------------|
| `react` | `spoon_react` | SpoonReactAI | Basic intelligent agent |
| `spoon_react` | `react` | SpoonReactAI | Basic intelligent agent |
| `spoon_react_mcp` | - | SpoonReactMCP | MCP-enabled agent |

## Configuration Management

### View Current Configuration

```bash
> config
Current configuration:
API Keys:
  openai: sk-o...xxxx
  anthropic: Not set
  deepseek: Not set
base_url: https://openrouter.ai/api/v1
default_agent: spoon_react
agents: [object Object]
```

### Modify Configuration

```bash
> config default_agent search_agent
Set default_agent = search_agent
```

## Advanced Usage

### Dynamic Agent Addition

You can dynamically add new agents through configuration commands:

```bash
> config agents.new_agent.class SpoonReactAI
> config agents.new_agent.description "New custom agent"
```

### Configuration File Location

The configuration file is located at `config.json` in the current working directory.

### Configuration Priority

1. Custom agents in configuration file
2. Built-in agent definitions
3. Environment variables

## Best Practices

1. **Use Descriptive Names**: Choose clear names for agents
2. **Set Useful Aliases**: Provide short aliases for quick access
3. **Add Detailed Descriptions**: Help users understand agent purposes
4. **Configure Parameters Appropriately**: Adjust agent parameters based on use cases
5. **Backup Configuration Files**: Regularly backup important configurations
6. **Secure API Keys**: Store sensitive API keys in environment variables
7. **Test MCP Connections**: Verify MCP server connectivity before deployment

## Troubleshooting

### Configuration File Not Found

If the configuration file doesn't exist, the system will use default configuration. Create a `config.json` file to customize settings.

### Agent Loading Failed

Check:

- Agent class name is correct
- Configuration file syntax is valid
- Required fields are present
- MCP servers are accessible (if using MCP agents)

### Alias Conflicts

If multiple agents use the same alias, the system will use the first matching agent.

### MCP Server Connection Issues

Check:

- MCP server command is correct
- Required environment variables are set
- Network connectivity is available
- Server dependencies are installed

### LLM Provider Configuration Issues

#### Provider Configuration Errors

**Issue:** Agent fails to initialize with LLM provider

**Solutions:**
1. Verify provider is configured in `llm_providers` section
2. Check API key is valid and has correct format
3. Ensure model name is supported by provider

```json
{
  "llm_providers": {
    "openai": {
      "api_key": "sk-your-valid-key",
      "model": "gpt-4.1"  // Ensure model exists
    }
  }
}
```

#### Agent-Specific LLM Issues

**Issue:** Agent uses wrong LLM provider or model

**Solutions:**
1. Set explicit `llm_provider` in agent config
2. Override model settings per agent
3. Check fallback chain configuration

```json
{
  "agents": {
    "my_agent": {
      "config": {
        "llm_provider": "anthropic",  // Explicit provider
        "llm_model": "claude-sonnet-4-20250514",  // Explicit model
        "enable_fallback": false  // Disable fallback if needed
      }
    }
  }
}
```

#### Fallback Chain Issues

**Issue:** Fallback not working as expected

**Solutions:**
1. Verify all providers in chain are configured
2. Check provider health status
3. Enable debug logging to trace fallback behavior

```json
{
  "llm_settings": {
    "fallback_chain": ["openai", "anthropic"],  // Ensure both are configured
    "enable_debug_logging": true  // Enable for troubleshooting
  }
}
```

#### Performance Issues

**Issue:** Slow LLM responses or timeouts

**Solutions:**
1. Adjust timeout settings
2. Reduce max_tokens if appropriate
3. Enable caching for repeated requests

```json
{
  "llm_providers": {
    "openai": {
      "timeout": 60,  // Increase timeout
      "max_tokens": 2048  // Reduce if needed
    }
  },
  "llm_settings": {
    "enable_caching": true,
    "cache_ttl": 3600
  }
}
```

### Debugging LLM Configuration

#### Enable Debug Logging

```json
{
  "llm_settings": {
    "enable_debug_logging": true
  }
}
```

#### Test Provider Configuration

```bash
# Test individual providers
> test-provider openai
> test-provider anthropic

# Check provider statistics
> provider-stats openai

# View configuration
> config llm_providers
```

#### Validate Configuration File

```bash
# Check JSON syntax
python -c "import json; print('Valid JSON' if json.load(open('config.json')) else 'Invalid JSON')"

# Validate required fields
> config validate
```

## Example Configurations

### Development Environment Configuration

```json
{
  "default_agent": "dev_agent",
  "agents": {
    "dev_agent": {
      "class": "SpoonReactAI",
      "aliases": ["dev", "development"],
      "description": "Development environment agent",
      "config": {
        "max_steps": 5,
        "tool_choice": "auto"
      },
      "tools": ["crypto_tools"]
    }
  },
  "tool_sets": {
    "crypto_tools": {
      "type": "builtin",
      "enabled": true
    }
  }
}
```

### Production Environment with MCP Integration

```json
{
  "default_agent": "prod_agent",
  "agents": {
    "prod_agent": {
      "class": "SpoonReactMCP",
      "aliases": ["prod", "production"],
      "description": "Production environment agent with search capabilities",
      "config": {
        "max_steps": 10,
        "tool_choice": "required"
      },
      "mcp_servers": ["tavily-mcp"],
      "tools": ["web_search", "crypto_tools"]
    }
  },
  "mcp_servers": {
    "tavily-mcp": {
      "command": "npx",
      "args": ["-y", "tavily-mcp"],
      "env": {
        "TAVILY_API_KEY": "your-tavily-api-key"
      },
      "disabled": false,
      "autoApprove": []
    }
  },
  "tool_sets": {
    "web_search": {
      "type": "mcp_server",
      "server": "tavily-mcp"
    },
    "crypto_tools": {
      "type": "builtin",
      "enabled": true
    }
  }
}
```

### Multi-Agent Configuration

```json
{
  "default_agent": "search_agent",
  "agents": {
    "search_agent": {
      "class": "SpoonReactMCP",
      "aliases": ["search", "tavily"],
      "description": "Specialized search agent",
      "mcp_servers": ["tavily-mcp"],
      "tools": ["web_search"]
    },
    "crypto_agent": {
      "class": "SpoonReactAI",
      "aliases": ["crypto", "trading"],
      "description": "Cryptocurrency trading agent",
      "tools": ["crypto_tools"]
    },
    "general_agent": {
      "class": "SpoonReactMCP",
      "aliases": ["general", "all"],
      "description": "General purpose agent with all tools",
      "mcp_servers": ["tavily-mcp"],
      "tools": ["web_search", "crypto_tools"]
    }
  },
  "mcp_servers": {
    "tavily-mcp": {
      "command": "npx",
      "args": ["-y", "tavily-mcp"],
      "env": {
        "TAVILY_API_KEY": "your-tavily-api-key"
      }
    }
  },
  "tool_sets": {
    "web_search": {
      "type": "mcp_server",
      "server": "tavily-mcp"
    },
    "crypto_tools": {
      "type": "builtin",
      "enabled": true
    }
  }
}
```
