# 🔐 Configuration Guide for SpoonOS

This guide covers how to configure SpoonOS with the unified tool configuration system, including API keys, MCP tools, built-in tools, and environment variables.

---

## Overview

SpoonOS uses a **unified tool configuration system** that consolidates all tool settings directly within agent configurations. This approach provides:

### Key Benefits
- **Single Configuration Point**: All tool settings in one place per agent
- **Embedded MCP Servers**: MCP server configuration included directly in tool definitions
- **Automatic Lifecycle Management**: MCP servers start/stop automatically based on tool usage
- **Type Safety**: Strong validation for all configuration options
- **Environment Variable Integration**: Tool-specific environment variables can be configured directly in tool definitions
- **Priority-based Configuration**: Tool-level env vars override system env vars for flexible configuration management

---

## 1. 🧾 Method: .env File (Recommended)

Create a `.env` file in the project root:

```bash
cp .env.example .env
```

Edit the file and fill in your credentials:

```bash
# LLM APIs
OPENAI_API_KEY=sk-your-openai-key
ANTHROPIC_API_KEY=sk-your-anthropic-key
DEEPSEEK_API_KEY=your-deepseek-key

# Blockchain
PRIVATE_KEY=your-wallet-private-key
RPC_URL=https://mainnet.rpc
CHAIN_ID=12345

# Tool-specific environment variables
TAVILY_API_KEY=your-tavily-api-key
BRAVE_API_KEY=your-brave-search-key
GITHUB_TOKEN=your-github-token

# Built-in tool environment variables
OKX_API_KEY=your_okx_api_key
OKX_SECRET_KEY=your_okx_secret_key
OKX_API_PASSPHRASE=your_okx_api_passphrase
OKX_PROJECT_ID=your_okx_project_id
CHAINBASE_API_KEY=your_chainbase_api_key
THIRDWEB_CLIENT_ID=your_thirdweb_client_id
BITQUERY_API_KEY=your_bitquery_api_key
RPC_URL=https://eth.llamarpc.com
```

Then load it at the top of your Python entry file (e.g. main.py):

```python
from dotenv import load_dotenv
load_dotenv(override=True)
```

## 2. 💻 Method: Shell Environment Variables

**Linux/macOS:**

```bash
# Set environment variables in your shell
export OPENAI_API_KEY="sk-your-openai-api-key-here"
export ANTHROPIC_API_KEY="sk-ant-your-anthropic-api-key-here"
export DEEPSEEK_API_KEY="your-deepseek-api-key-here"
export PRIVATE_KEY="your-wallet-private-key-here"

# Make them persistent by adding to your shell profile
echo 'export OPENAI_API_KEY="sk-your-openai-api-key-here"' >> ~/.bashrc
echo 'export ANTHROPIC_API_KEY="sk-ant-your-anthropic-api-key-here"' >> ~/.bashrc
source ~/.bashrc
```

**Windows (PowerShell):**

```powershell
# Set environment variables
$env:OPENAI_API_KEY="sk-your-openai-api-key-here"
$env:ANTHROPIC_API_KEY="sk-ant-your-anthropic-api-key-here"
$env:DEEPSEEK_API_KEY="your-deepseek-api-key-here"
$env:PRIVATE_KEY="your-wallet-private-key-here"

# Make them persistent
[Environment]::SetEnvironmentVariable("OPENAI_API_KEY", "sk-your-openai-api-key-here", "User")
[Environment]::SetEnvironmentVariable("ANTHROPIC_API_KEY", "sk-ant-your-anthropic-api-key-here", "User")
```

## 3. 🧪 Method: CLI Configuration Commands

After starting the CLI, use the `config` command:

```bash
# Start the CLI
python main.py

# Configure API keys using the CLI
> config api_key openai sk-your-openai-api-key-here
✅ OpenAI API key configured successfully

> config api_key anthropic sk-ant-your-anthropic-api-key-here
✅ Anthropic API key configured successfully

> config api_key deepseek your-deepseek-api-key-here
✅ DeepSeek API key configured successfully

# Configure wallet private key
> config PRIVATE_KEY your-wallet-private-key-here
✅ Private key configured successfully

# View current configuration (keys are masked for security)
> config
Current configuration:
API Keys:
  openai: sk-12...ab34
  anthropic: sk-an...xy89
  deepseek: ****...****
PRIVATE_KEY: 0x12...ab34
```

## 4. 📁 Method: Unified Configuration File

The unified configuration format consolidates all settings in `config.json`:

### Complete Configuration Example

```json
{
  "api_keys": {
    "openai": "sk-your-openai-api-key-here",
    "anthropic": "sk-ant-your-anthropic-api-key-here",
    "deepseek": "your-deepseek-api-key-here"
  },
  "default_agent": "web_researcher",
  "providers": {
    "openai": {
      "api_key": "sk-your-openai-key",
      "model": "gpt-4.1",
      "max_tokens": 4096,
      "temperature": 0.3,
      "timeout": 30,
      "retry_attempts": 3
    },
    "anthropic": {
      "api_key": "sk-ant-your-key",
      "model": "claude-3-5-sonnet-20241022",
      "max_tokens": 4096,
      "temperature": 0.3,
      "timeout": 30,
      "retry_attempts": 3
    }
  },
  "llm_settings": {
    "default_provider": "openai",
    "fallback_chain": ["openai", "anthropic"],
    "enable_monitoring": true,
    "enable_caching": true,
    "enable_debug_logging": false,
    "max_concurrent_requests": 10
  },
  "agents": {
    "web_researcher": {
      "class": "SpoonReactMCP",
      "description": "Agent with web search and analysis capabilities",
      "aliases": ["researcher", "web"],
      "config": {
        "max_steps": 10,
        "tool_choice": "auto"
      },
      "tools": [
        {
          "name": "web_search",
          "type": "mcp",
          "description": "Web search capabilities via Tavily API",
          "enabled": true,
          "mcp_server": {
            "command": "npx",
            "args": ["-y", "@tavily/mcp-server"],
            "env": {
              "TAVILY_API_KEY": "your-tavily-api-key-here"
            },
            "disabled": false,
            "autoApprove": ["search", "get_content"],
            "timeout": 30,
            "retry_attempts": 3
          },
          "config": {
            "max_results": 10,
            "include_raw_content": true
          }
        },
        {
          "name": "crypto_powerdata_cex",
          "type": "builtin",
          "description": "Crypto PowerData CEX market data tool",
          "enabled": true,
          "env": {
            "OKX_API_KEY": "your_okx_api_key",
            "OKX_SECRET_KEY": "your_okx_secret_key",
            "OKX_API_PASSPHRASE": "your_okx_api_passphrase",
            "OKX_PROJECT_ID": "your_okx_project_id"
          },
          "config": {
            "timeout": 30,
            "max_retries": 3
          }
        }
      ]
    },
    "trading_bot": {
      "class": "SpoonReactAI",
      "description": "Automated trading and DeFi operations",
      "aliases": ["trader", "bot"],
      "config": {
        "max_steps": 15,
        "tool_choice": "auto"
      },
      "tools": [
        {
          "name": "github_tools",
          "type": "mcp",
          "description": "GitHub repository analysis and management",
          "enabled": true,
          "mcp_server": {
            "command": "uvx",
            "args": ["github-mcp-server"],
            "env": {
              "GITHUB_TOKEN": "your-github-token-here"
            },
            "autoApprove": ["list_repos", "get_file"]
          }
        },
        {
          "name": "get_token_price",
          "type": "builtin",
          "description": "Get current token prices",
          "enabled": true,
          "env": {
            "RPC_URL": "https://eth.llamarpc.com",
            "BITQUERY_API_KEY": "your_bitquery_api_key"
          },
          "config": {
            "timeout": 30,
            "max_retries": 3
          }
        }
      ]
    }
  }
}
```

### Environment Variables Configuration Example

Here's a complete example showing how to use environment variables in tool configurations:

```json
{
  "api_keys": {
    "OPENAI_API_KEY": "sk-your-openai-key",
    "ANTHROPIC_API_KEY": "sk-your-anthropic-key"
  },
  "default_agent": "trading_agent",
  "agents": {
    "trading_agent": {
      "class": "SpoonReactAI",
      "description": "Trading agent with environment variable configuration",
      "tools": [
        {
          "name": "crypto_powerdata_cex",
          "type": "builtin",
          "description": "Crypto PowerData CEX with environment variables",
          "enabled": true,
          "env": {
            "POWERDATA_API_KEY": "your-powerdata-api-key-here",
            "POWERDATA_BASE_URL": "https://api.cryptopowerdata.com",
            "DEBUG_MODE": "false"
          },
          "config": {
            "max_retries": 3,
            "rate_limit": 100
          }
        },
        {
          "name": "tavily_search",
          "type": "mcp",
          "description": "Tavily with dual environment variables",
          "enabled": true,
          "env": {
            "SEARCH_TIMEOUT": "60",
            "DEBUG_SEARCH": "true"
          },
          "mcp_server": {
            "command": "npx",
            "args": ["-y", "@tavily/mcp-server"],
            "env": {
              "TAVILY_API_KEY": "your-tavily-api-key-here",
              "NODE_ENV": "production"
            },
            "autoApprove": ["search", "get_content"]
          }
        }
      ]
    }
  }
}
```

**Key Features:**
- **Tool-level environment variables**: Set at the tool level (`env` field)
- **Server-level environment variables**: Set in MCP server configuration
- **Environment isolation**: Each tool has its own environment scope
- **Override capability**: Tool-level env vars override server-level ones

## 5. 🔧 Tool Configuration Details

### MCP Tools Configuration

MCP (Model Context Protocol) tools require an embedded server configuration. Here's a basic example:

#### Environment Variables for MCP Tools

MCP tools support environment variables at two levels:

1. **Tool-level environment variables** (`env` at tool level): Applied to the entire tool context
2. **Server-level environment variables** (`env` in `mcp_server`): Applied specifically to the MCP server process

Tool-level environment variables are merged with server-level ones, with tool-level taking precedence.

#### Web Search Tool (Tavily)
```json
{
  "name": "tavily-search",
  "type": "mcp",
  "description": "Web search and content retrieval via Tavily",
  "enabled": true,
  "mcp_server": {
    "command": "npx",
    "args": ["--yes", "tavily-mcp"],
    "env": {
      "TAVILY_API_KEY": "your-tavily-api-key-here"
    },
    "autoApprove": ["tavily-search"],
    "timeout": 30,
    "retry_attempts": 3
  },
  "config": {
    "max_results": 10,
    "include_raw_content": true
  }
}
```

### Built-in Tools Configuration

Built-in tools are part of the SpoonOS toolkit and don't require external servers. For a complete list of all available built-in tools, see the [Built-in Tools Reference](./builtin_tools.md).

#### Environment Variables for Tools

You can configure environment variables directly in tool configurations instead of relying only on system environment variables:

```json
{
  "name": "crypto_powerdata_cex",
  "type": "builtin",
  "description": "Crypto PowerData CEX market data",
  "enabled": true,
  "env": {
    "OKX_API_KEY": "your_okx_api_key",
    "OKX_SECRET_KEY": "your_okx_secret_key",
    "OKX_API_PASSPHRASE": "your_okx_api_passphrase",
    "OKX_PROJECT_ID": "your_okx_project_id"
  },
  "config": {
    "timeout": 30,
    "max_retries": 3
  }
}
```

**Key Benefits:**
- **Tool-specific Environment**: Each tool can have its own environment variables
- **Configuration Isolation**: Environment variables are scoped to individual tools
- **Override System Variables**: Tool-level env vars take precedence over system environment
- **Dynamic Configuration**: No need to restart the application to change environment variables

### Mixed Configuration Example

A simple agent with both MCP and builtin tools:

```json
{
  "research_agent": {
    "class": "SpoonReactMCP",
    "description": "Research agent with web search and crypto data",
    "aliases": ["researcher"],
    "config": {
      "max_steps": 10,
      "tool_choice": "auto"
    },
    "tools": [
      {
        "name": "tavily-search",
        "type": "mcp",
        "description": "Web search via Tavily",
        "enabled": true,
        "mcp_server": {
          "command": "npx",
          "args": ["--yes", "tavily-mcp"],
          "env": {"TAVILY_API_KEY": "your-tavily-key"}
        }
      },
      {
        "name": "crypto_powerdata_cex",
        "type": "builtin",
        "description": "Cryptocurrency market data",
        "enabled": true,
        "env": {
          "OKX_API_KEY": "your_okx_api_key",
          "OKX_SECRET_KEY": "your_okx_secret_key"
        }
      }
    ]
  }
}
```

## 6. 🔍 Verification & Testing

### Check Environment Variables

```bash
# Verify environment variables are set
echo $OPENAI_API_KEY
echo $ANTHROPIC_API_KEY
echo $DEEPSEEK_API_KEY

# Test with a simple Python script
python -c "import os; print('OpenAI:', 'SET' if os.getenv('OPENAI_API_KEY') else 'NOT SET')"
```

### Test API Connectivity

```bash
# Start CLI and test
python main.py

# Test agent loading
> load-agent web_researcher
✅ Loaded agent: web_researcher

# Test chat functionality
> action chat
> Hello, can you respond to test the API connection?
```

### Test Tool Configuration

```bash
# Test configuration validation
python -c "
from spoon_ai.config.manager import ConfigManager
manager = ConfigManager()
config = manager.load_config()
issues = manager.validate_configuration()
print('✅ Valid configuration' if not issues else f'❌ Issues: {issues}')
"

# Test tool loading
python -c "
from spoon_ai.config.manager import ConfigManager
import asyncio

async def test_tools():
    manager = ConfigManager()
    tools = await manager.load_agent_tools('web_researcher')
    print(f'✅ Loaded {len(tools)} tools successfully')

asyncio.run(test_tools())
"
```

## 7. 🔒 Security Best Practices

### 🚨 Critical Security Guidelines

1. **Never commit API keys to version control**

   ```bash
   # Ensure .env is in .gitignore
   echo ".env" >> .gitignore
   ```

2. **Use environment variables in production**

   - Avoid hardcoding keys in source code
   - Use secure environment variable management in deployment

3. **Wallet private key security**

   - **NEVER share your private key with anyone**
   - Store in secure environment variables only
   - Consider using hardware wallets for production

4. **API key rotation**
   - Regularly rotate API keys (monthly recommended)
   - Monitor API usage for unusual activity
   - Use API key restrictions when available

### 🛡️ Additional Security Measures

```bash
# Set restrictive file permissions for .env
chmod 600 .env

# Use a dedicated wallet for testing with minimal funds
# Never use your main wallet's private key

# Monitor API usage regularly
# Set up billing alerts on API provider dashboards
```

## 8. 🏗️ LLM Provider Configuration

SpoonOS supports multiple LLM providers through a **flexible configuration system** with smart fallback logic. You can configure providers with minimal settings or full customization.

### 🎯 Configuration Priority System

The LLM configuration follows a clear priority hierarchy:

```
1. providers[provider_name] section (highest priority)
2. api_keys[provider_name] (for API key fallback)
3. Environment variables (provider-specific)
4. Default values (built-in provider defaults)
```

### 🔧 Flexible Configuration Options

#### Option 1: Minimal Configuration (Recommended)
Only configure API keys, everything else uses smart defaults:

```json
{
  "api_keys": {
    "anthropic": "sk-ant-your-key",
    "openai": "sk-your-openai-key",
    "openrouter": "sk-or-your-key"
  }
}
```

#### Option 2: Mixed Configuration
API keys in `api_keys`, custom settings in `providers`:

```json
{
  "api_keys": {
    "anthropic": "sk-ant-your-key",
    "openai": "sk-your-openai-key"
  },
  "providers": {
    "openai": {
      "model": "gpt-4.1",
      "temperature": 0.5
    },
    "anthropic": {
      "temperature": 0.1
    }
  }
}
```

#### Option 3: Full Provider Configuration
Complete configuration in `providers` section:

```json
{
  "providers": {
    "openai": {
      "api_key": "sk-your-openai-key",
      "model": "gpt-4.1",
      "temperature": 0.3
    }
  }
}
```

### 📋 Configuration Options Reference

| Option | Type | Required | Description | Default |
|--------|------|----------|-------------|---------|
| `api_key` | string | **Yes** | Provider API key | None |
| `model` | string | No | Model name to use | Provider-specific default |
| `max_tokens` | integer | No | Maximum tokens per request | 4096 |
| `temperature` | float | No | Response randomness (0.0-1.0) | Provider-specific |
| `timeout` | integer | No | Request timeout in seconds | 30 |
| `retry_attempts` | integer | No | Number of retry attempts | 3 |
| `base_url` | string | No | Custom API endpoint | Provider-specific |
| `custom_headers` | object | No | Additional HTTP headers | {} |
| `extra_params` | object | No | Additional provider parameters | {} |

### 🎨 Provider-Specific Defaults

Each provider has optimized default values. You only need to specify what you want to override:

#### OpenAI
**Default Values:**
- Model: `gpt-4.1`
- Base URL: `https://api.openai.com/v1`
- Temperature: `0.3`
- Max Tokens: `4096`

**Minimal Configuration:**
```json
{
  "api_keys": {
    "openai": "sk-your-openai-key"
  }
}
```

**Custom Configuration:**
```json
{
  "api_keys": {
    "openai": "sk-your-openai-key"
  },
  "providers": {
    "openai": {
      "model": "gpt-4.1",
      "temperature": 0.5
    }
  }
}
```

#### Anthropic
**Default Values:**
- Model: `claude-sonnet-4-20250514`
- Base URL: `https://api.anthropic.com`
- Temperature: `0.1` (optimized for Claude)
- Max Tokens: `4096`

**Minimal Configuration:**
```json
{
  "api_keys": {
    "anthropic": "sk-ant-your-key"
  }
}
```

#### OpenRouter
**Default Values:**
- Model: `anthropic/claude-sonnet-4`
- Base URL: `https://openrouter.ai/api/v1`
- Temperature: `0.3`
- Custom Headers: Includes referer and title

**Minimal Configuration:**
```json
{
  "api_keys": {
    "openrouter": "sk-or-your-key"
  }
}
```

#### DeepSeek
**Default Values:**
- Model: `deepseek-reasoner`
- Base URL: `https://api.deepseek.com/v1`
- Temperature: `0.2` (optimized for reasoning)
- Max Tokens: `65536` (large context support)

#### Gemini
**Default Values:**
- Model: `gemini-2.0-flash-exp`
- Base URL: `https://generativelanguage.googleapis.com/v1beta`
- Temperature: `0.1` (optimized for Gemini)
- Max Tokens: `4096`

### 🌐 Global LLM Settings

Configure global LLM behavior and fallback chains:

```json
{
  "llm_settings": {
    "default_provider": "anthropic",
    "fallback_chain": ["anthropic", "openrouter", "openai"],
    "enable_monitoring": true,
    "enable_caching": true,
    "enable_debug_logging": false,
    "max_concurrent_requests": 20
  }
}
```

**LLM Settings Options:**

| Option | Type | Required | Description | Default |
|--------|------|----------|-------------|---------|
| `default_provider` | string | No | Primary provider to use | Auto-detected |
| `fallback_chain` | array | No | Provider fallback order | Auto-generated |
| `enable_monitoring` | boolean | No | Enable usage monitoring | true |
| `enable_caching` | boolean | No | Enable response caching | true |
| `enable_debug_logging` | boolean | No | Enable debug logs | false |
| `max_concurrent_requests` | integer | No | Max parallel requests | 20 |

### 📝 Complete Configuration Examples

#### Example 1: Minimal Setup (Recommended for beginners)
```json
{
  "api_keys": {
    "anthropic": "sk-ant-your-key",
    "openrouter": "sk-or-your-key"
  },
  "llm_settings": {
    "default_provider": "anthropic",
    "fallback_chain": ["anthropic", "openrouter"]
  }
}
```

#### Example 2: Mixed Configuration (Recommended for most users)
```json
{
  "api_keys": {
    "anthropic": "sk-ant-your-key",
    "openai": "sk-your-openai-key",
    "openrouter": "sk-or-your-key"
  },
  "providers": {
    "openai": {
      "model": "gpt-4.1",
      "temperature": 0.5
    },
    "openrouter": {
      "model": "anthropic/claude-sonnet-4"
    }
  },
  "llm_settings": {
    "default_provider": "anthropic",
    "fallback_chain": ["anthropic", "openai", "openrouter"]
  }
}
```

#### Example 3: Advanced Configuration (Power users)
```json
{
  "providers": {
    "anthropic": {
      "api_key": "sk-ant-your-key",
      "model": "claude-sonnet-4-20250514",
      "temperature": 0.1,
      "max_tokens": 4096,
      "timeout": 60
    },
    "openrouter": {
      "api_key": "sk-or-your-key",
      "model": "anthropic/claude-sonnet-4",
      "temperature": 0.3,
      "custom_headers": {
        "HTTP-Referer": "https://your-app.com",
        "X-Title": "Your App Name"
      }
    }
  },
  "llm_settings": {
    "default_provider": "anthropic",
    "fallback_chain": ["anthropic", "openrouter"],
    "enable_monitoring": true,
    "max_concurrent_requests": 10
  }
}
```

### Environment Variable Mapping

You can also configure providers using environment variables:

```bash
# OpenAI
OPENAI_API_KEY=sk-your-key
OPENAI_MODEL=gpt-4.1
OPENAI_MAX_TOKENS=4096
OPENAI_TEMPERATURE=0.3

# Anthropic
ANTHROPIC_API_KEY=sk-ant-your-key
ANTHROPIC_MODEL=claude-3-5-sonnet-20241022
ANTHROPIC_MAX_TOKENS=4096

# Gemini
GEMINI_API_KEY=your-key
GEMINI_MODEL=gemini-2.0-flash-exp
```

## 9. 🔧 Troubleshooting Configuration Issues

### Common Configuration Problems

#### 1. Missing Environment Variables

**Error:** `Tool 'web_search' missing environment variable TAVILY_API_KEY`

**Solution:**
- Add the required environment variable to your `.env` file
- Or set it in the tool's MCP server configuration
- Check that the variable name matches exactly

```json
{
  "mcp_server": {
    "env": {
      "TAVILY_API_KEY": "your-actual-api-key-here"
    }
  }
}
```

#### 2. MCP Server Startup Failures

**Error:** `MCP Server failed to start: Command not found`

**Solutions:**
- Ensure the command is installed and available in PATH
- For `npx` commands: `npm install -g npm`
- For `uvx` commands: Install uv package manager
- Check the command and arguments are correct

```bash
# Test MCP server manually
npx -y @tavily/mcp-server

# Or for uvx
uvx github-mcp-server
```

#### 3. Tool Configuration Validation Errors

**Error:** `MCP tools must have mcp_server configuration`

**Solution:**
- Ensure MCP tools have the `mcp_server` section
- Check that the tool type is set to "mcp"
- Verify all required fields are present

```json
{
  "name": "my_tool",
  "type": "mcp",  // Must be "mcp" for MCP tools
  "enabled": true,
  "mcp_server": {  // Required for MCP tools
    "command": "npx",
    "args": ["-y", "my-mcp-server"]
  }
}
```

#### 4. Duplicate Tool Names

**Error:** `Duplicate tool names found: ['web_search']`

**Solution:**
- Ensure tool names are unique within each agent
- Use descriptive, unique names for tools
- Consider prefixing tools with their purpose

```json
{
  "tools": [
    {"name": "tavily_web_search", "type": "mcp"},
    {"name": "brave_web_search", "type": "mcp"}
  ]
}
```

#### 5. Agent Class Not Found

**Error:** `Agent class 'MyCustomAgent' not found`

**Solution:**
- Verify the agent class name is correct
- Use supported agent classes: `SpoonReactAI`, `SpoonReactMCP`
- Check for typos in the class name

```json
{
  "agents": {
    "my_agent": {
      "class": "SpoonReactMCP"  // Use exact class name
    }
  }
}
```

### Environment Variable Requirements

Different tools require specific environment variables:

#### Web Search Tools
```bash
# Tavily
TAVILY_API_KEY=your-tavily-api-key

# Brave Search
BRAVE_API_KEY=your-brave-search-key

# SerpAPI
SERPAPI_API_KEY=your-serpapi-key
```

#### Development Tools
```bash
# GitHub
GITHUB_TOKEN=your-github-personal-access-token

# GitLab
GITLAB_TOKEN=your-gitlab-access-token
```

#### Database Tools
```bash
# PostgreSQL
DATABASE_URL=postgresql://user:password@localhost:5432/database

# MongoDB
MONGODB_URI=mongodb://localhost:27017/database
```

#### Blockchain Tools
```bash
# Ethereum
ETHEREUM_RPC_URL=https://mainnet.infura.io/v3/your-project-id
PRIVATE_KEY=your-wallet-private-key

# Solana
SOLANA_RPC_URL=https://api.mainnet-beta.solana.com
```

### Debug Mode

Enable debug logging for detailed troubleshooting:

```json
{
  "llm_settings": {
    "enable_debug_logging": true
  }
}
```

Or via environment variable:
```bash
export SPOON_DEBUG=true
python main.py
```

## 10. 📋 Configuration Schema Reference

### Agent Configuration Schema

```json
{
  "agent_name": {
    "class": "SpoonReactAI | SpoonReactMCP",
    "description": "string (optional)",
    "aliases": ["array", "of", "strings"] (optional),
    "config": {
      "max_steps": "integer (default: 10)",
      "tool_choice": "string (default: 'auto')"
    },
    "tools": [
      // Array of tool configurations (see below)
    ]
  }
}
```

### Tool Configuration Schema

#### Built-in Tool
```json
{
  "name": "string (required, unique per agent)",
  "type": "builtin",
  "description": "string (optional)",
  "enabled": "boolean (default: true)",
  "env": {
    "KEY": "value"
  } (optional),
  "config": {
    // Tool-specific configuration options
  }
}
```

#### MCP Tool
```json
{
  "name": "string (required, unique per agent)",
  "type": "mcp",
  "description": "string (optional)",
  "enabled": "boolean (default: true)",
  "env": {
    "KEY": "value"
  } (optional),
  "mcp_server": {
    "command": "string (required)",
    "args": ["array", "of", "strings"] (optional),
    "env": {
      "KEY": "value"
    } (optional),
    "cwd": "string (optional)",
    "disabled": "boolean (default: false)",
    "autoApprove": ["array", "of", "tool", "names"] (optional),
    "timeout": "integer (default: 30)",
    "retry_attempts": "integer (default: 3)"
  },
  "config": {
    // Tool-specific configuration options
  } (optional)
}
```

## ✅ Next Steps

After configuration, continue to:

- 🤖 [Agent Configuration Guide](./agent_configuration.md)
- 🧠 [Start the CLI](./cli.md)
- 🔧 [Tool Development Guide](./tools.md)
- 📊 [Monitoring and Debugging](./monitoring.md)

---

## 📚 Additional Resources

- [Configuration Examples Repository](../examples/)
- [Tool Configuration Templates](../examples/config/)
- [MCP Server Documentation](https://modelcontextprotocol.io/)
- [SpoonOS API Reference](./api.md)