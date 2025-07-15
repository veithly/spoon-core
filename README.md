# 🚀 SpoonOS Core Developer Framework(SCDF)

<div align="center">
  <img src="logo/spoon.gif" alt="SpoonAI Logo" width="200"/>
  <p><strong>Core developer framework of SpoonOS ——Agentic OS for the sentient economy. Next-Generation AI Agent Framework | Powerful Interactive CLI | Web3 infrastructure optimized Support</strong></p>
</div>

## 📘 How to Use This README

This README is your guide to getting started with the **SpoonOS Core Developer Framework (SCDF)**. It walks you through everything you need—from understanding core capabilities to actually running your own agents.

Here's how to navigate it:

- [✨ Features](#features): Start here to understand what SpoonOS can do. This section gives you a high-level overview of its agentic, composable, and interoperable architecture.

- [🔧 Installation](#installation): As of **June 2025**, SpoonOS currently supports **Python only**. This section tells you which Python version to use and how to set up a virtual environment.

- [🔐 Environment & API Key Config](#environment-variables-and-api-key-Configuration): Learn how to configure the API keys for various LLMs (e.g., OpenAI, Claude, deepseek). We also provide configuration methods for Web3 infrastructure such as chains, RPC endpoints, databases, and blockchain explorers.

- [🚀 Quick Start](#quick-start): Once your environment is ready, start calling our **MCP server**, which bundles a wide range of tools. Other servers are also available.

- [🛠️ CLI Tools](#cli-tools): This section shows how to use the CLI to run LLM-powered tasks with ease.

- [🧩 Agent Framework](#agent-framework): Learn how to create your own agents, register custom tools, and extend SpoonOS with minimal setup.

- [🔌 API Integration](#api-integration): Plug in external APIs to enhance your agent workflows.

- [🤝 Contributing](#contributing): Want to get involved? Check here for contribution guidelines.

- [📄 License](#license): Standard license information.

By the end of this README, you'll not only understand what SCDF is—but you'll be ready to build and run your own AI agents and will gain ideas on scenarios what SCDF could empower. **Have fun!**

## Features

SpoonOS is a living, evolving agentic operating system. Its SCDF is purpose-built to meet the growing demands of Web3 developers — offering a complete toolkit for building sentient, composable, and interoperable AI agents.

- **🧠 ReAct Intelligent Agent** - Advanced agent architecture combining reasoning and action
- **🔧 Custom Tool Ecosystem** - Modular tool system for easily extending agent capabilities
- **💬 Multi-Model Support** - Compatible with major large language models including OpenAI, Anthropic, DeepSeek, and more Web3 fine-tuned LLM
- **🌐 Web3-Native Interoperability** - Enables AI agents to communicate and coordinate across ecosystems via DID and ZKML-powered interoperability protocols.
- **📡 Scalable Data Access** - Supports structured and unstructured data via MCP integration
- **💻 Interactive CLI** - Feature-rich command line interface
- **🔄 State Management** - Comprehensive session history and state persistence
- **🔗Composable Agent Logic** - Create agents that can sense, reason, plan, and execute modularly — enabling use cases across DeFi, creator economy, and more
- **🚀 Easy to Use** - Well-designed API for rapid development and integration

## ⚙️ Quick Installation

### Prerequisites

- Python 3.10+
- pip package manager (or uv as a faster alternative)

```bash
# Clone the repo
$ git clone https://github.com/XSpoonAi/spoon-core.git
$ cd spoon-core

# Create a virtual environment
$ python -m venv spoon-env
$ source spoon-env/bin/activate  # For macOS/Linux

# Install dependencies
$ pip install -r requirements.txt
```

Prefer faster install? See docs/installation.md for uv-based setup.

## 🔐 API Key & Environment Setup

Create a .env file in the root directory:

```bash
cp .env.example .env
```

Fill in your keys:

```bash
OPENAI_API_KEY=sk-your-openai-key
ANTHROPIC_API_KEY=sk-your-claude-key
DEEPSEEK_API_KEY=your-deepseek-key
PRIVATE_KEY=your-wallet-private-key
RPC_URL=https://mainnet.rpc
CHAIN_ID=12345
```

Then in your Python entry file:

```bash
from dotenv import load_dotenv
load_dotenv(override=True)
```


For advanced config methods (CLI setup, config.json, PowerShell), see docs/configuration.md.


## Using OpenRouter (Multi-LLM Gateway)

```python
from spoon_ai.chat import ChatBot
from spoon_ai.agents import SpoonReactAI

# Using OpenAI's GPT-4
openai_agent = SpoonReactAI(
    llm=ChatBot(model_name="gpt-4", llm_provider="openai")
)

# Using Anthropic's Claude
claude_agent = SpoonReactAI(
    llm=ChatBot(model_name="claude-3-7-sonnet-20250219", llm_provider="anthropic")
)


# Using OpenRouter (OpenAI-compatible API)
# Uses OPENAI_API_KEY environment variable with your OpenRouter API key
openrouter_agent = SpoonReactAI(
    llm=ChatBot(
        model_name="anthropic/claude-sonnet-4",     # Model name from OpenRouter
        llm_provider="openai",                      # MUST be "openai"
        base_url="https://openrouter.ai/api/v1"     # OpenRouter API endpoint
)
)
```

## 🚀 Run the CLI

### Start the MCP Server

```bash
# Start the MCP server with all available tools
python -m spoon_ai.tools.mcp_tools_collection

# The server will start and display:
# MCP Server running on stdio transport
# Available tools: [list of tools]
```

### Start the CLI

```bash
python main.py
```


Try chatting with your agent:


```bash
> action chat
> Hello, Spoon!
```

## 🧩 Build Your Own Agent

### 1. Define Your Own Tool

```python
from spoon_ai.tools.base import BaseTool

class MyCustomTool(BaseTool):
    name: str = "my_tool"
    description: str = "Description of what this tool does"
    parameters: dict = {
        "type": "object",
        "properties": {
            "param1": {"type": "string", "description": "Parameter description"}
        },
        "required": ["param1"]
    }

    async def execute(self, param1: str) -> str:
        # Tool implementation
        return f"Result: {param1}"

```

### 2. Define Your Own Agent

```python
from spoon_ai.agents import ToolCallAgent
from spoon_ai.tools import ToolManager

class MyAgent(ToolCallAgent):
    name: str = "my_agent"
    description: str = "Agent description"
    system_prompt: str = "You are a helpful assistant..."
    max_steps: int = 5

    available_tools: ToolManager = Field(
        default_factory=lambda: ToolManager([MyCustomTool()])
    )
```

#### 3. Run the Agent and Interact via Prompt

```python
import asyncio

async def main():
    agent = MyCustomAgent(llm=ChatBot())
    result = await agent.run("Say hello to Scarlett")
    print("Result:", result)

if __name__ == "__main__":
    asyncio.run(main())
```


Register your own tools, override run(), or extend with MCP integrations. See docs/agents.md or docs/mcp_mode_usage.md


## 📁 Repository Structure

```yaml
spoon-core/
├── README.md
├── .env.example
├── requirements.txt
├── main.py
│
├── examples/             # 🧪 Examples
│   ├── agents/           # 🧠 Agent demos (GitHub,Weather)
│   └── mcp/              # 🔌 Tool server examples
│
├── spoon_ai/             # 🍴 Core agent framework
│
├── docs/                 # 📚 Documentation
│   ├── installation.md
│   ├── configuration.md
│   ├── openrouter.md
│   ├── cli.md
│   ├── agents.md
│   └── mcp_mode_usage.md
```
