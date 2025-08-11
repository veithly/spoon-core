
# 🤖 Agent Development Guide

This guide provides a comprehensive walkthrough for developing and configuring agents in the SpoonOS Core Developer Framework (SCDF). We will use a practical example, the `SpoonMacroAnalysisAgent`, to illustrate key concepts, including agent definition, tool integration, and execution.

For details on the configuration file system, see [`configuration.md`](./configuration.md).

---

## 1. Core Concepts: Agent and Tools

In SpoonOS, an **Agent** is an autonomous entity that leverages **Tools** to achieve specific goals. Our example, `SpoonMacroAnalysisAgent`, is designed to perform macroeconomic analysis on cryptocurrencies by combining market data with the latest news.

### Agent Architecture

`SpoonMacroAnalysisAgent` inherits from `SpoonReactMCP`, a powerful base class that provides:
- **ReAct Loop**: A reasoning and acting cycle for intelligent decision-making.
- **MCP Support**: Built-in capabilities to connect with external tools via the Model Context Protocol (MCP).

### Tool Integration

Agents can use two main types of tools:

1.  **Built-in Tools**: Python classes that are part of the core framework or custom toolkits. In our example, `CryptoPowerDataCEXTool` is a built-in tool that provides cryptocurrency market data.
2.  **MCP Tools**: External tools accessed via the MCP protocol. `tavily_search`, used for web searches, is an example of an MCP tool.

---

## 2. Building an Agent: A Step-by-Step Example

Let's break down the implementation of `SpoonMacroAnalysisAgent` from the `spoon_search_agent.py` example.

### Step 1: Define the Agent Class

First, we define the agent by inheriting from `SpoonReactMCP` and setting its core properties:

```python
class SpoonMacroAnalysisAgent(SpoonReactMCP):
    name: str = "SpoonMacroAnalysisAgent"
    system_prompt: str = (
        '''You are a cryptocurrency market analyst. Your task is to provide a comprehensive
        macroeconomic analysis of a given token.

        To do this, you will perform the following steps:
        1. Use the `crypto_power_data_cex` tool to get the latest candlestick data and
           technical indicators (like EMA, RSI, MACD) for the token.
        2. Use the `tavily_search` tool to find the latest news and market sentiment
           related to the token. You MUST call the tool with the `query` parameter.
        3. Synthesize the quantitative data and qualitative information to form a holistic analysis.
        4. Present the final analysis, including a summary of the data, key news insights,
           and your overall assessment of the token's market position.'''
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.avaliable_tools = ToolManager([])
```

- **`name`**: A unique identifier for the agent.
- **`system_prompt`**: A directive that guides the AI model, explaining its role, the tools available, and how to use them.
- **`__init__`**: Initializes the agent and its `ToolManager`, which will hold the tools.

### Step 2: Configure and Load Tools

In the `initialize` method, we set up and load the necessary tools.

#### Built-in Tool Configuration

Loading a built-in tool is straightforward. Simply import and instantiate it:

```python
from spoon_toolkits.crypto.crypto_powerdata.tools import CryptoPowerDataCEXTool

# Inside the initialize method:
crypto_tool = CryptoPowerDataCEXTool()
```

#### MCP Tool Configuration

Configuring MCP tools is now simpler and more direct. Instead of a separate method, we define the configuration right where the tool is instantiated, using the `mcp_config` parameter:

```python
from spoon_ai.tools.mcp_tool import MCPTool

# Inside the initialize method:
tavily_key = os.getenv("TAVILY_API_KEY", "")
if not tavily_key or tavily_key == "your-tavily-api-key-here":
    raise ValueError("TAVILY_API_KEY is not set or is a placeholder.")

tavily_tool = MCPTool(
    name="tavily_search",
    description="Performs a web search using the Tavily API.",
    mcp_config={
        "command": "npx",
        "args": ["--yes", "tavily-mcp"],
        "env": {
            "TAVILY_API_KEY": tavily_key
        },
        "tool_name_mapping": {
            "tavily_search": "tavily-search"
        }
    }
)
```

This approach:
- Is more intuitive and places the configuration next to its usage.
- Eliminates the need for a separate `_get_mcp_config` helper method.

#### Add Tools to the ToolManager

Finally, we add both tools to the agent's `ToolManager`:

```python
# Inside the initialize method:
self.avaliable_tools = ToolManager([tavily_tool, crypto_tool])
logger.info(f"Available tools: {list(self.avaliable_tools.tool_map.keys())}")
```

This makes the tools available for the agent to use.

### Step 3: Execute the Agent

The `main` function orchestrates the agent's execution:

```python
async def main():
    # 1. Create the Agent instance
    agent = SpoonMacroAnalysisAgent(llm=ChatBot(llm_provider="openai", model_name="gpt-4.1"))

    # 2. Initialize the agent and its tools
    await agent.initialize()

    # 3. Define and run the analysis query
    query = "Perform a macro analysis of the NEO token."
    response = await agent.run(query)

    print(f"\n--- Analysis Complete ---\n{response}")

if __name__ == "__main__":
    asyncio.run(main())
```

This function:
1.  Instantiates the `SpoonMacroAnalysisAgent`.
2.  Calls `agent.initialize()` to load the tools.
3.  Defines a `query` and calls `agent.run()` to start the analysis.

---

## 3. Full Code Example

Here is the complete, refactored code for `spoon_search_agent.py`:

```python
import os
import sys
import asyncio
import logging
from typing import Dict, Any

# Ensure the toolkit is in the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../spoon-toolkit')))

from spoon_ai.agents.spoon_react_mcp import SpoonReactMCP
from spoon_ai.tools.mcp_tool import MCPTool
from spoon_ai.tools.tool_manager import ToolManager
from spoon_ai.chat import ChatBot
from spoon_toolkits.crypto.crypto_powerdata.tools import CryptoPowerDataCEXTool

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SpoonMacroAnalysisAgent(SpoonReactMCP):
    name: str = "SpoonMacroAnalysisAgent"
    system_prompt: str = (
        '''You are a cryptocurrency market analyst. Your task is to provide a comprehensive
        macroeconomic analysis of a given token.

        To do this, you will perform the following steps:
        1. Use the `crypto_power_data_cex` tool to get the latest candlestick data and
           technical indicators (like EMA, RSI, MACD) for the token.
        2. Use the `tavily_search` tool to find the latest news and market sentiment
           related to the token. You MUST call the tool with the `query` parameter.
        3. Synthesize the quantitative data and qualitative information to form a holistic analysis.
        4. Present the final analysis, including a summary of the data, key news insights,
           and your overall assessment of the token's market position.'''
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.avaliable_tools = ToolManager([])

    async def initialize(self):
        logger.info("Initializing agent and loading tools...")
        
        tavily_key = os.getenv("TAVILY_API_KEY", "")
        if not tavily_key or tavily_key == "your-tavily-api-key-here":
            raise ValueError("TAVILY_API_KEY is not set or is a placeholder.")

        tavily_tool = MCPTool(
            name="tavily_search",
            description="Performs a web search using the Tavily API.",
            mcp_config={
                "command": "npx",
                "args": ["--yes", "tavily-mcp"],
                "env": {
                    "TAVILY_API_KEY": tavily_key
                }
            }
        )
        
        crypto_tool = CryptoPowerDataCEXTool()
        self.avaliable_tools = ToolManager([tavily_tool, crypto_tool])
        logger.info(f"Available tools: {list(self.avaliable_tools.tool_map.keys())}")

async def main():
    print("--- SpoonOS Macro Analysis Agent Demo ---")
    tavily_key = os.getenv("TAVILY_API_KEY")
    if not tavily_key or tavily_key == "your-tavily-api-key-here":
        logger.error("TAVILY_API_KEY is not set or contains a placeholder. Please set a valid API key.")
        return

    agent = SpoonMacroAnalysisAgent(llm=ChatBot(llm_provider="openai", model_name="gpt-4.1"))
    print("Agent instance created.")

    await agent.initialize()

    query = "Perform a macro analysis of the NEO token."
    print(f"\nRunning query: {query}")
    print("-" * 30)

    response = await agent.run(query)

    print(f"\n--- Analysis Complete ---\n{response}")

if __name__ == "__main__":
    asyncio.run(main())
```

---

## 4. Next Steps

- **Customize the Agent**: Modify the `system_prompt` to change the agent's behavior or add new tools to the `ToolManager`.
- **Explore More Tools**: Look into other built-in toolkits or connect to different MCP services.
- **Read the Configuration Guide**: For more advanced setups using `config.json`, see the [Configuration Guide](./configuration.md).
