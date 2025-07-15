# 🌐 OpenRouter Configuration Guide

OpenRouter provides an OpenAI-compatible API interface that enables you to access multiple LLM providers (e.g., OpenAI, Anthropic, Meta, Mistral) through a **single API key**.

This guide explains how to configure and use OpenRouter in your SpoonOS projects.

---

## 1. 🔑 Get Your OpenRouter API Key

- Visit [OpenRouter Platform](https://openrouter.ai/keys)
- Sign up or log in
- Create an API key from the dashboard

---

## 2. ⚙️ Set Environment Variables

Use the `OPENAI_API_KEY` environment variable to store your OpenRouter key.

```bash
# Linux/macOS
export OPENAI_API_KEY="sk-or-your-openrouter-api-key-here"

# Windows PowerShell
$env:OPENAI_API_KEY="sk-or-your-openrouter-api-key-here"
```

📌 Important:
Even though you're using OpenRouter, the variable name must be OPENAI_API_KEY for compatibility with most OpenAI clients.

## 3. 🧠 Use OpenRouter in Your SpoonAI Agent

You can use OpenRouter in SpoonReactAI or any agent that accepts a ChatBot-style LLM interface.

```python
from spoon_ai.chat import ChatBot
from spoon_ai.agents import SpoonReactAI

# Configuring OpenRouter-powered agent
openrouter_agent = SpoonReactAI(
    llm=ChatBot(
        model_name="anthropic/claude-sonnet-4",     # Model name from OpenRouter
        llm_provider="openai",                      # MUST be "openai"
        base_url="https://openrouter.ai/api/v1"     # OpenRouter API endpoint
    )
)
```

## 4. 📌 Key Note

- base_url must be set to: https://openrouter.ai/api/v1

- llm_provider must be "openai" — even for Anthropic, Meta, or Mistral models
  (because OpenRouter uses the OpenAI-compatible format)

- The API key is automatically read from OPENAI_API_KEY

## 5. 🧪 Example model_name Values

- `openai/gpt-4o` - GPT-4o model
- `openai/gpt-4.1` - GPT-4.1 model
- `anthropic/claude-sonnet-4` - Claude 4 sonnet model
- `anthropic/claude-opus-4` - Claude 4 opus model
- `deepseek/deepseek-r1` - DeepSeek R1 model
- For more models, see [OpenRouter Models List](https://openrouter.ai/models)

## ✅ Next Steps

Once OpenRouter is set up:

🧩 [Learn to build your own agent](./agent.md)
🌐 [Integrate Web3 tools with MCP](./mcp_mode_usage.md)
