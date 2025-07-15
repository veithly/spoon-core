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
  "default_agent": "default"
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

## ✅ Next Steps

After configuration, continue to:

- 🤖 [Set up OpenRouter LLM models](./openrouter.md)
- 🧠 [Start the CLI or develop your custom agents](./cli.md)
- 🧩 [Learn to build your own agent](./agent.md)
- 🌐 [Integrate Web3 tools with MCP](./mcp_mode_usage.md)
