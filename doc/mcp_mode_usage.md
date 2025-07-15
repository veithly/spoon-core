# Using Web3 Tools via MCP: Build, Connect, and Query with Custom Agents

This guide introduces three integration modes for using Web3 tools with SpoonOS via the MCP protocol, along with code structure and example commands.

## Mode 1: Built-in Agent Mode

In this mode, you encapsulate your custom tools into the MCP tool collection (such as creating a new mcp_thirdweb_collection, or directly changing the mcp_tools_collection.py file), and then call it through an Agent that inherits from SpoonReactAI and MCPClientMixin (such as SpoonThirdWebMCP). This mode is maintained by the platform Agent configuration and can be used directly by users.

### Structural diagram

```markdown
[User Prompt]
↓
[SpoonThirdWebMCP Agent] 🧠
↓ calls
[FastMCP over SSE]
↓
[GetBlocksFromThirdwebInsight / GetWalletTransactionsTool / etc.]
↓
[Thirdweb Insight API]
```

### 📁 Examples Directory

The following directory contains runnable examples:

```yaml
examples/
├── mcp/
│   ├── SpoonThirdWebAgent.py
│   └── mcp_thirdweb_collection.py
```

#### Step1. Install Dependencies

You have two ways to install the spoon-toolkits package:

###### Option A: Install from GitHub source (for development or latest changes):

```bash
git clone https://github.com/XSpoonAi/spoon-toolkit.git
cd spoon-toolkit
pip install -e .
```

###### Option B: From PyPI (recommended)

```bash
pip install spoon-toolkits

```

👉 Tip:

Use Option 1 if you want to modify the toolkit source code or track the latest updates.

#### Step2. Start the MCP Tool Server

Sample: examples/mcp_thirdweb_collection.py

```python
from fastmcp import FastMCP
import asyncio
# from typing import Any, Dict, List, Optional

# Import base tool classes and tool manager
from spoon_ai.tools.base import BaseTool, ToolResult
from spoon_ai.tools.tool_manager import ToolManager

# Import all available tools
from spoon_toolkits import (
    GetContractEventsFromThirdwebInsight,
    GetMultichainTransfersFromThirdwebInsight,
    GetTransactionsTool,
    GetContractTransactionsTool,
    GetContractTransactionsBySignatureTool,
    GetBlocksFromThirdwebInsight,
    GetWalletTransactionsFromThirdwebInsight
)

mcp = FastMCP("SpoonAI MCP Tools")

class MCPToolsCollection:
    """Collection class that wraps existing tools as MCP tools"""

    def __init__(self):
        """Initialize MCP tools collection

        Args:
            name: Name of the MCP server
        """
        self.mcp = mcp
        self._setup_tools()

    def _setup_tools(self):
        """Set up all available tools as MCP tools"""
        # Create all tool instances
        tools = [
            GetContractEventsFromThirdwebInsight(),
            GetMultichainTransfersFromThirdwebInsight(),
            GetTransactionsTool(),
            GetContractTransactionsTool(),
            GetContractTransactionsBySignatureTool(),
            GetBlocksFromThirdwebInsight(),
            GetWalletTransactionsFromThirdwebInsight()
        ]

        # Create tool manager
        self.tool_manager = ToolManager(tools)

        # Create MCP wrapper for each tool
        for tool in tools:
            self.mcp.add_tool(tool.execute, name=tool.name, description=tool.description)

    async def run(self, **kwargs):
        """Start the MCP server

        Args:
            **kwargs: Parameters passed to FastMCP.run()
        """
        await self.mcp.run_async(transport="sse", port=8765, **kwargs)

# Create default instance that can be imported directly
mcp_tools = MCPToolsCollection()

if __name__ == "__main__":
    # Start MCP server when this script is run directly
    asyncio.run(mcp_tools.run())
```

Before calling the agent, make sure the MCP service is running:

```bash
python mcp_thirdweb_collection.py
```

#### Step 3

###### 3.1 Define Agent and connect to MCP

```python
from spoon_ai.agents.spoon_react import SpoonReactAI
from spoon_ai.agents.mcp_client_mixin import MCPClientMixin
from fastmcp.client.transports import SSETransport
from spoon_ai.tools.tool_manager import ToolManager

from pydantic import Field
from spoon_ai.chat import ChatBot
import os
import asyncio


class SpoonThirdWebAgent(SpoonReactAI, MCPClientMixin):
    name: str = "SpoonThirdWebAgent"
    description: str = (
        "An AI assistant specialized in querying EVM blockchain data using the Thirdweb Insight API. "
        "Supports retrieving smart contract events (e.g. Transfer), function call transactions, wallet activity, "
        "recent cross-chain token transfers (especially USDT), block metadata, and contract-specific transaction logs. "
        "Use this agent when the user asks about on-chain behavior, such as token transfers, contract usage, wallet history, or recent block/transaction activity."
    )
    system_prompt: str = """
        You are ThirdwebInsightAgent, a blockchain data analyst assistant powered by Thirdweb Insight API.
        You can fetch EVM contract events, transactions, token transfers, blocks, and wallet activity across multiple chains.

        Use the appropriate tool when the user asks about:
        - contract logs or Transfer events → use `get_contract_events_from_thirdweb_insight`
        - USDT transfers across chains → use `get_multichain_transfers_from_thirdweb_insight`
        - recent cross-chain transactions → use `get_transactions`
        - a specific contract's transaction history → use `get_contract_transactions`
        - contract function call history (e.g., swap, approve) → use `get_contract_transactions_by_signature`
        - recent block info by chain → use `get_blocks_from_thirdweb_insight`
        - wallet activity across chains → use `get_wallet_transactions_from_thirdweb_insight`

        Always extract necessary parameters like:
        - `contract_address` (if user mentions a token, e.g. USDT, WETH, use its address)
        - `chain_id` (Ethereum = 1, Polygon = 137, etc.)
        - `event_signature` (e.g., 'Transfer(address,address,uint256)')
        - `limit` (default to 10 if unspecified)
        - `client_id` can be pulled from environment variable or injected context

        If something is unclear, ask for clarification. Otherwise, call the appropriate tool.
    """

    avaliable_tools: ToolManager = Field(default_factory=lambda: ToolManager([]))
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        MCPClientMixin.__init__(self, mcp_transport=kwargs.get('mcp_transport', SSETransport("http://127.0.0.1:8765/sse")))
```

###### 3.2 User operation mode

Get client_id from https://thirdweb.com/login

```python
async def main():
    # Ensure necessary API keys are set
    # Create an InfoAssistantAgent
    info_agent = SpoonThirdWebAgent(llm=ChatBot(llm_provider="openai",model_name="anthropic/claude-sonnet-4", base_url="https://openrouter.ai/api/v1" ))

    # Query standard ERC20 transfer events (Transfer)
    info_agent.clear()
    result = await info_agent.run("Get the last 10 Transfer events from the USDT contract on Ethereum using client ID xxxx.")
    print(result)

if __name__ == "__main__":
    asyncio.run(main())
```

Run the agent with:

```bash
python SpoonThirdWebAgent.py
```

Expected Result:

```bash
[
{
"block_number": "19202222",
"from": "0x...",
"to": "0x...",
"amount": "1000 USDT"
},
...
]
```

## Mode 2: Community Agent Mode

In this mode, you can reuse agents published by others in the community, without writing your own tool code. These agents are registered via GitHub using the MCP protocol, and called via mcp-proxy.

This is useful when:

You want to quickly try a public Agent from GitHub

You don't want to define Tool, ToolManager, or custom logic

You want to orchestrate many agents from different repos

Register the tool to the MCP service

### Step-by-Step: Community Agent Mode

Use Community Agent Mode to connect with agents hosted on GitHub via the MCP protocol — without writing custom tool or agent code.

##### 1. Install mcp-proxy via UV

```bash
uv tool install mcp-proxy
```

This will install the proxy server that bridges your CLI or client agent to remote GitHub agents.

##### 2. Start the Community Agent via MCP Proxy,Example using @modelcontextprotocol/server-github

```bash
mcp.proxy --sse-port 8123 -- npx -y @modelcontextprotocol/server-github
```

This command will:

Start an SSE server on http://localhost:8123/sse

Load an agent from the @modelcontextprotocol/server-github package

Allow your local agent to communicate with this GitHub-based agent over MCP

##### 3. Connect Your Local Python Agent to the Proxy

```python
from spoon_ai.agents.spoon_react import SpoonReactAI
from spoon_ai.agents.mcp_client_mixin import MCPClientMixin
from fastmcp.client.transports import SSETransport
from spoon_ai.tools.tool_manager import ToolManager
from pydantic import Field
class SpoonReactMCP(SpoonReactAI, MCPClientMixin):
    description: str = ()
    system_prompt: str = """ """
    name: str = "spoon_react_mcp"
    description: str = "A smart ai agent in neo blockchain with mcp"
    avaliable_tools: ToolManager = Field(default_factory=lambda: ToolManager([]))
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        MCPClientMixin.__init__(self, mcp_transport=kwargs.get('mcp_transport', SSETransport("http://127.0.0.1:8123/sse")))
```

## ✅ Next Steps

To further enhance your Web3 agent:

- 🤖 [Build custom agents](./agent.md)
