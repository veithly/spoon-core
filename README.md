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
