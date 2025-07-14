# 🛠️ SpoonOS CLI Usage Guide

SCDF CLI is a powerful command-line tool that provides rich functionality, including interacting with AI agents, managing chat history, processing cryptocurrency transactions, and loading documents.

## 📦 Prerequisite: Start the MCP Server

Before starting the CLI, make sure the MCP (Message Connectivity Protocol) server is running:

```bash
python -m spoon_ai.tools.mcp_tools_collection
```

## 🚀 Start the CLI

Once the MCP server is running, launch the CLI:

```bash
python main.py
```

## Basic Commands

| Command             | Aliases           | Description                                                                                                               |
| ------------------- | ----------------- | ------------------------------------------------------------------------------------------------------------------------- |
| `help`              | `h`, `?`          | Display help information                                                                                                  |
| `exit`              | `quit`, `q`       | Exit the CLI                                                                                                              |
| `load-agent <name>` | `load`            | Load an agent with the specified name                                                                                     |
| `list-agents`       | `agents`          | List all available agents                                                                                                 |
| `config`            | `cfg`, `settings` | Configure settings (such as API keys)                                                                                     |
| `reload-config`     | `reload`          | Reload the current agent's configuration                                                                                  |
| `action <action>`   | `a`               | Perform a specific action using the current agent. For example, `action react` to start a step-by-step reasoning session. |

### Chat Management Commands

| Command          | Aliases | Description                         |
| ---------------- | ------- | ----------------------------------- |
| `new-chat`       | `new`   | Start a new chat (clear history)    |
| `list-chats`     | `chats` | List available chat history records |
| `load-chat <ID>` | -       | Load a specific chat history record |

### Cryptocurrency-Related Commands

| Command                                       | Aliases  | Description                            |
| --------------------------------------------- | -------- | -------------------------------------- |
| `transfer <address> <amount> <token>`         | `send`   | Transfer tokens to a specified address |
| `swap <source_token> <target_token> <amount>` | -        | Exchange tokens using an aggregator    |
| `token-info <address>`                        | `token`  | Get token information by address       |
| `token-by-symbol <symbol>`                    | `symbol` | Get token information by symbol        |

### Document Management Commands

| Command                      | Aliases | Description                                                      |
| ---------------------------- | ------- | ---------------------------------------------------------------- |
| `load-docs <directory_path>` | `docs`  | Load documents from the specified directory to the current agent |

## CLI Usage Examples

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

#### Basic Interaction

1. Start a new chat:

```
> action chat
New chat session started
```

3. Directly input text to interact with the AI agent:

```
> Hello, please introduce yourself
[AI reply will be displayed here]
```

#### Cryptocurrency Operations

1. View token information:

```
> token-by-symbol SPO
Token information:
Name: SpoonOS not a meme
Symbol:SPO
Address: 0x...
Decimals: 18
...
```

2. Transfer operation:

```
> transfer 0x123... 0.1 SPO
Preparing to transfer 0.1 SPO to 0x123...
[Transfer details will be displayed here]
```

## ✅ Next Steps

To extend CLI usage:

- 🤖 [Explore agent capabilities](./agent.md)
- 🌐 [Use Web3 tools via MCP](./mcp_mode_usage.md)
