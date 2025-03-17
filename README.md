# SpoonAI

## CLI User Guide

SpoonAI CLI is a powerful command-line tool that provides various functions such as interacting with AI agents, managing chat history, processing cryptocurrency transactions, and loading documents. This document will detail how to install and use SpoonAI CLI.

### Installation

1. Clone the repository:
```bash
git clone git@github.com:XSpoonAi/spoon-core.git
cd SpoonAI
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Configure environment variables (optional):
For cryptocurrency-related functions, you may need to set the following environment variables:
- `RPC_URL`: Blockchain RPC service address
- `CHAIN_ID`: Blockchain network ID (default is 1, which is the Ethereum mainnet)
- `SCAN_URL`: Blockchain explorer address (default is "https://etherscan.io")

### API Key Configuration

SpoonAI CLI supports various AI services and functions, requiring different API keys to be configured. Below are the methods for configuring the main API keys:

#### Configuration Methods

1. **Via CLI command**:
```
> config <key_name> <key_value>
```

2. **Via environment variables**:
Set the corresponding environment variables in your system

3. **Via configuration file**:
Edit the `~/.config/spoonai/config.json` file

#### Common API Keys

| API Key Name | Description | How to Obtain |
|------------|------|---------|
| `OPENAI_API_KEY` | OpenAI API key for GPT models | [OpenAI Website](https://platform.openai.com/api-keys) |
| `ANTHROPIC_API_KEY` | Anthropic API key for Claude models | [Anthropic Website](https://console.anthropic.com/keys) |
| `DEEPSEEK_API_KEY` | DeepSeek API key | [DeepSeek Website](https://platform.deepseek.com/) |
| `PRIVATE_KEY` | Blockchain wallet private key for cryptocurrency transactions | Export from your wallet |

#### Configuration Examples

1. Configure OpenAI API key:
```
> config OPENAI_API_KEY sk-your-openai-api-key
OPENAI_API_KEY updated
```

2. Configure Anthropic API key:
```
> config ANTHROPIC_API_KEY sk-ant-your-anthropic-api-key
ANTHROPIC_API_KEY updated
```

3. Configure wallet private key:
```
> config PRIVATE_KEY your-private-key-here
PRIVATE_KEY updated
```

#### View Current Configuration

Use the following command to view all current configurations:
```
> config
Current configuration:
OPENAI_API_KEY: sk-***********
ANTHROPIC_API_KEY: sk-ant-***********
MODEL: gpt-4
...
```

#### Key Security Considerations

1. API keys are sensitive; do not share them with others or expose them in public
2. Wallet private keys are especially important; leakage may result in asset loss
3. It is recommended to store keys using environment variables or configuration files rather than entering them directly in the command line
4. Regularly change API keys to improve security

### Starting the CLI

```bash
python main.py
```

### Basic Commands

| Command | Aliases | Description |
|------|------|------|
| `help` | `h`, `?` | Display help information |
| `exit` | `quit`, `q` | Exit the CLI |
| `load-agent <name>` | `load` | Load an agent with the specified name |
| `list-agents` | `agents` | List all available agents |
| `config` | `cfg`, `settings` | Configure settings (such as API keys) |
| `reload-config` | `reload` | Reload the current agent's configuration |
| `action <action>` | `a` | Perform a specific action using the current agent |

### Chat Management Commands

| Command | Aliases | Description |
|------|------|------|
| `new-chat` | `new` | Start a new chat (clear history) |
| `list-chats` | `chats` | List available chat history records |
| `load-chat <ID>` | - | Load a specific chat history record |

### Cryptocurrency-Related Commands

| Command | Aliases | Description |
|------|------|------|
| `transfer <address> <amount> <token>` | `send` | Transfer tokens to a specified address |
| `swap <source_token> <target_token> <amount>` | - | Exchange tokens using an aggregator |
| `token-info <address>` | `token` | Get token information by address |
| `token-by-symbol <symbol>` | `symbol` | Get token information by symbol |

### Document Management Commands

| Command | Aliases | Description |
|------|------|------|
| `load-docs <directory_path>` | `docs` | Load documents from the specified directory to the current agent |

### Atomic Capabilities

#### Basic Interaction

1. Start the CLI and load an agent:
```
> load-agent chat
chat agent loaded
```

2. Start a new chat:
```
> new-chat
New chat session started
```

3. Directly input text to interact with the AI agent:
```
> Hello, please introduce yourself
[AI reply will be displayed here]
```

#### Configure Settings

1. View current configuration:
```
> config
Current configuration:
API_KEY: sk-***********
MODEL: gpt-4
...
```

2. Modify configuration:
```
> config API_KEY sk-your-new-api-key
API_KEY updated
```

#### Cryptocurrency Operations

1. View token information:
```
> token-by-symbol ETH
Token information:
Name: Ethereum
Symbol: ETH
Address: 0x...
Decimals: 18
...
```

2. Transfer operation:
```
> transfer 0x123... 0.1 ETH
Preparing to transfer 0.1 ETH to 0x123...
[Transfer details will be displayed here]
```

#### Document Management

1. Load documents:
```
> load-docs /path/to/documents
Loading documents...
Successfully loaded 5 documents
```

### ReAct Agent

#### ReAct Intelligent Agent Introduction

SpoonAI implements an intelligent agent based on the ReAct (Reasoning + Acting) paradigm, which is an advanced AI agent architecture that combines reasoning and action capabilities. The ReAct Agent can think, plan, and execute in complex tasks, solving problems through an iterative reasoning-action loop.

#### ReAct Workflow

The ReAct Agent workflow includes the following key steps:

1. **Observation**: Collecting environment and task-related information
2. **Reasoning**: Analyzing information and reasoning
3. **Acting**: Executing specific operations
4. **Feedback**: Obtaining action results and updating cognition

This cycle repeats continuously until the task is completed or the preset goal is achieved.

Through the ReAct Agent, SpoonAI can handle complex tasks that require deep thinking and multi-step actions, providing users with more intelligent and autonomous problem-solving capabilities.

## Agent Usage and Customization Guide

SpoonAI offers two ways to use Agents:
1. Using predefined Agents - Simple declaration and execution
2. Custom Agents - Creating your own tools and logic

### Using Predefined Agents

SpoonAI comes with several predefined Agents, such as SpoonChatAI and SpoonReactAI. Using these Agents is very simple and requires just a few lines of code:

```python
from spoon_ai.agents import SpoonChatAI, SpoonReactAI
from spoon_ai.chat import ChatBot

# Create a Chat Agent
chat_agent = SpoonChatAI(llm=ChatBot())

# Run the Agent and get a response
response = await chat_agent.run("Hello, please introduce yourself")
print(response)

# Create a ReAct Agent
react_agent = SpoonReactAI(llm=ChatBot())

# Run the ReAct Agent and get a response
response = await react_agent.run("Analyze the transaction history of this wallet address: 0x123...")
print(response)
```

Predefined Agents are already configured with appropriate system prompts and tools, ready to use. You can also customize some parameters when creating them:

```python
# Create a ReAct Agent with custom parameters
custom_react = SpoonReactAI(
    llm=ChatBot(model="gpt-4"),  # Specify the model to use
    max_steps=15,                # Set the maximum number of steps
    system_prompt="Custom system prompt"  # Override the default system prompt
)
```

### Custom Agents

If predefined Agents don't meet your needs, you can create your own. SpoonAI provides a flexible framework to support custom Agents.

#### 1. Creating Custom Tools

First, you need to create custom tools. Each tool should inherit from the `BaseTool` class:

```python
from spoon_ai.tools.base import BaseTool

class MyCustomTool(BaseTool):
    name: str = "my_custom_tool"
    description: str = "This is a custom tool for performing specific tasks"
    parameters: dict = {
        "type": "object",
        "properties": {
            "param1": {
                "type": "string",
                "description": "Description of the first parameter"
            },
            "param2": {
                "type": "integer",
                "description": "Description of the second parameter"
            }
        },
        "required": ["param1"]
    }

    async def execute(self, param1: str, param2: int = 0) -> str:
        """Implement the tool's specific logic"""
        # Implement your tool logic here
        result = f"Processing parameters: {param1}, {param2}"
        return result
```

Tool definitions include three main parts:
- `name`: The unique name of the tool
- `description`: A detailed description of the tool (AI will decide when to use it based on this)
- `parameters`: JSON Schema definition of the tool parameters
- `execute()`: Method implementing the tool's specific logic

#### 2. Creating Custom Agents

There are two ways to create custom Agents:

**Method 1: Inheriting from an existing Agent class**

```python
from spoon_ai.agents import ToolCallAgent
from spoon_ai.tools import ToolManager
from pydantic import Field

class MyCustomAgent(ToolCallAgent):
    name: str = "my_custom_agent"
    description: str = "This is my custom Agent"
    
    system_prompt: str = """You are an AI assistant specialized in performing specific tasks.
    You can use the provided tools to complete tasks."""
    
    next_step_prompt: str = "What should be the next step?"
    
    max_steps: int = 8
    
    # Define available tools
    avaliable_tools: ToolManager = Field(default_factory=lambda: ToolManager([
        MyCustomTool(),
        # Add other tools...
    ]))
```

**Method 2: Directly using ToolCallAgent and configuring tools**

```python
from spoon_ai.agents import ToolCallAgent
from spoon_ai.tools import ToolManager
from spoon_ai.chat import ChatBot

# Create a tool manager
tool_manager = ToolManager([
    MyCustomTool(),
    AnotherTool(),
    # Add more tools...
])

# Create an Agent
my_agent = ToolCallAgent(
    name="my_agent",
    description="Custom configured Agent",
    llm=ChatBot(model="gpt-4"),
    avaliable_tools=tool_manager,
    system_prompt="Custom system prompt",
    max_steps=12
)
```

#### 3. Running Custom Agents

After creating an Agent, you can run it just like predefined Agents:

```python
# Run the custom Agent
response = await my_agent.run("Perform a specific task")
print(response)
```

### Advanced Usage: Tool Combination and Indexing

SpoonAI supports dynamic tool combination and semantic indexing, allowing Agents to more intelligently select appropriate tools:

```python
from spoon_ai.tools import ToolManager

# Create multiple tools
tools = [
    MyCustomTool(),
    AnotherTool(),
    ThirdTool(),
    # More tools...
]

# Create a tool manager
tool_manager = ToolManager(tools)

# Create a semantic index for tools (requires OpenAI API key)
tool_manager.index_tools()

# Find the most relevant tools based on a query
relevant_tools = tool_manager.query_tools(
    query="I need to analyze this data", 
    top_k=3  # Return the top 3 most relevant tools
)

# Create a new Agent with the found tools
from spoon_ai.agents import ToolCallAgent
from spoon_ai.chat import ChatBot

specialized_agent = ToolCallAgent(
    name="specialized_agent",
    llm=ChatBot(),
    avaliable_tools=ToolManager(relevant_tools)
)

# Run the specialized Agent
response = await specialized_agent.run("Analyze this dataset")
```

### Best Practices

1. **Tool Design**:
   - Each tool should have a clear, single responsibility
   - Provide detailed descriptions to help AI understand when to use the tool
   - Parameters should have clear types and descriptions

2. **System Prompts**:
   - Provide clear guidance and constraints for the Agent
   - Explain the task's goals and expected behavior
   - Explain how to use the available tools

3. **Error Handling**:
   - Tools should gracefully handle errors and return useful error messages
   - Use the `ToolFailure` class to return error results

4. **Step Limitations**:
   - Set reasonable `max_steps` values to avoid infinite loops
   - Complex tasks may require more steps, simple tasks fewer

By following these guidelines, you can create powerful and flexible custom Agents to meet the needs of various complex tasks.
