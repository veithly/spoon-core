# 🔐 Configuration Guide for SpoonOS

This guide covers how to configure SpoonOS with API keys, private keys, RPC endpoints, and other environment variables.

---

## 1.🧾 Method: .env File (Recommended)

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
```

Then load it at the top of your Python entry file (e.g. main.py):

```python
from dotenv import load_dotenv
load_dotenv(override=True)
```

## 2.💻 Method: Shell Environment Variables

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

## 3.🧪 Method: CLI Configuration Commands

After starting the CLI, use the `config` command:

```bash
# Start the MCP server with all available tools
python -m spoon_ai.tools.mcp_tools_collection

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

## 4 📁 Method: Configuration File

The CLI creates a configuration file at `config.json` in the project root directory:

```json
{
  "api_keys": {
    "openai": "sk-your-openai-api-key-here",
    "anthropic": "sk-ant-your-anthropic-api-key-here",
    "deepseek": "your-deepseek-api-key-here"
  },
  "base_url": "your_base_url_here",
  "default_agent": "default",
  "llm_providers": {
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
  }
}
```

## 5. 🔍 Verification & Testing

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
# Start the MCP server with all available tools
python -m spoon_ai.tools.mcp_tools_collection

# Start CLI and test
python main.py

# start chat and test
> action chat
> Hello, can you respond to test the API connection?
```

## 6 🔒 Security Best Practices

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

## 🏗️ LLM Provider Configuration

SpoonOS supports multiple LLM providers through a unified configuration system. You can configure providers individually and set up fallback chains for high availability.

### Provider Configuration Options

Each provider supports the following configuration options:

| Option | Type | Description | Default |
|--------|------|-------------|---------|
| `api_key` | string | Provider API key | Required |
| `model` | string | Model name to use | Provider default |
| `max_tokens` | integer | Maximum tokens per request | 4096 |
| `temperature` | float | Response randomness (0.0-1.0) | 0.3 |
| `timeout` | integer | Request timeout in seconds | 30 |
| `retry_attempts` | integer | Number of retry attempts | 3 |
| `base_url` | string | Custom API endpoint | Provider default |
| `custom_headers` | object | Additional HTTP headers | {} |

### Provider-Specific Configuration

#### OpenAI Configuration
```json
{
  "llm_providers": {
    "openai": {
      "api_key": "sk-your-openai-key",
      "model": "gpt-4.1",
      "max_tokens": 4096,
      "temperature": 0.3,
      "timeout": 30,
      "retry_attempts": 3,
      "base_url": "https://api.openai.com/v1"
    }
  }
}
```

#### Anthropic Configuration
```json
{
  "llm_providers": {
    "anthropic": {
      "api_key": "sk-ant-your-key",
      "model": "claude-sonnet-4-20250514",
      "max_tokens": 4096,
      "temperature": 0.3,
      "timeout": 30,
      "retry_attempts": 3
    }
  }
}
```

#### Gemini Configuration
```json
{
  "llm_providers": {
    "gemini": {
      "api_key": "your-gemini-key",
      "model": "gemini-2.5-pro",
      "max_tokens": 4096,
      "temperature": 0.3,
      "timeout": 30
    }
  }
}
```

### Global LLM Settings

Configure global LLM behavior:

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

### Fallback Configuration

Set up automatic fallback between providers:

```json
{
  "llm_settings": {
    "fallback_chain": ["openai", "anthropic", "gemini"],
    "fallback_on_errors": ["rate_limit", "timeout", "authentication"],
    "fallback_delay": 1.0
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
ANTHROPIC_MODEL=claude-sonnet-4-20250514
ANTHROPIC_MAX_TOKENS=4096

# Gemini
GEMINI_API_KEY=your-key
GEMINI_MODEL=gemini-2.5-pro
```

### CLI Provider Commands

Manage providers through the CLI:

```bash
# List available providers
> config providers

# Configure a provider
> config provider openai api_key sk-your-new-key
> config provider openai model gpt-4.1-turbo

# Set default provider
> config llm_settings default_provider anthropic

# Configure fallback chain
> config llm_settings fallback_chain openai,anthropic,gemini

# Test provider connectivity
> test-provider openai
> test-provider all
```

### Provider Switching

Switch providers dynamically:

```python
from spoon_ai.llm import LLMManager

llm_manager = LLMManager()

# Use specific provider
response = await llm_manager.chat(
    messages=[{"role": "user", "content": "Hello"}],
    provider="anthropic"
)

# Use default provider (with fallback)
response = await llm_manager.chat(
    messages=[{"role": "user", "content": "Hello"}]
)
```

## 🔧 Troubleshooting LLM Provider Issues

### Common Provider Configuration Issues

#### 1. Provider Not Found
**Error:** `Provider 'openai' not found`

**Solutions:**
- Check provider name spelling in configuration
- Ensure provider is properly configured in `llm_providers` section
- Verify provider module is imported

```bash
# Check available providers
> list-providers

# Test provider configuration
> test-provider openai
```

#### 2. API Key Authentication Errors
**Error:** `AuthenticationError: Invalid API key`

**Solutions:**
- Verify API key is correct and active
- Check API key format (OpenAI: `sk-...`, Anthropic: `sk-ant-...`)
- Ensure API key has sufficient permissions

```bash
# Update API key
> config provider openai api_key sk-your-new-key

# Test authentication
> test-provider openai
```

#### 3. Rate Limit Errors
**Error:** `RateLimitError: Rate limit exceeded`

**Solutions:**
- Configure retry attempts and delays
- Set up fallback providers
- Reduce concurrent requests

```json
{
  "llm_providers": {
    "openai": {
      "retry_attempts": 5,
      "retry_delay": 2.0
    }
  },
  "llm_settings": {
    "fallback_chain": ["openai", "anthropic"],
    "max_concurrent_requests": 5
  }
}
```

#### 4. Model Not Available
**Error:** `ModelNotFoundError: Model 'gpt-5' not available`

**Solutions:**
- Check available models for provider
- Use correct model names
- Update to supported model versions

```bash
# Check provider capabilities
> provider-status openai

# Update model
> config provider openai model gpt-4.1-turbo
```

#### 5. Network Connectivity Issues
**Error:** `NetworkError: Connection timeout`

**Solutions:**
- Check internet connectivity
- Verify firewall settings
- Increase timeout values

```json
{
  "llm_providers": {
    "openai": {
      "timeout": 60,
      "retry_attempts": 3
    }
  }
}
```

#### 6. Configuration Validation Errors
**Error:** `ConfigurationError: Invalid configuration format`

**Solutions:**
- Validate JSON syntax
- Check required fields are present
- Verify data types match expected format

```bash
# Validate configuration
python -c "import json; json.load(open('config.json'))"

# Reset to default configuration
> config reset
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

```bash
# Enable debug mode via CLI
> config llm_settings enable_debug_logging true

# View debug logs
> show-logs llm
```

### Health Monitoring

Monitor provider health:

```bash
# Check all providers
> provider-status

# Monitor specific provider
> provider-stats openai

# Test connectivity
> test-provider all
```

## ✅ Next Steps

After configuration, continue to:

- 🤖 [Set up OpenRouter LLM models](./openrouter.md)
- 🧠 [Start the CLI or develop your custom agents](./cli.md)
- 🧩 [Learn to build your own agent](./agent.md)
- 🌐 [Integrate Web3 tools with MCP](./mcp_mode_usage.md)
