# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Quick Start Commands

```bash
# Start the MCP server with all available tools
python -m spoon_ai.tools.mcp_tools_collection

# Run the interactive CLI
python main.py

# Install dependencies
pip install -r requirements.txt
# OR use uv for faster installation:
uv sync
```

## Testing Commands

```bash
# Run tests with pytest
pytest

# Run tests for specific environments using tox
tox -e py310

# Run coverage report
tox -e report
```

## Development Commands

```bash
# Install in development mode
pip install -e .

# Update requirements from pyproject.toml
pip-compile --output-file=requirements.txt pyproject.toml
```

## Architecture Overview

SpoonOS Core Developer Framework (SCDF) is an AI agent framework with the following key components:

### Core Agent Architecture
- **ReAct Agents**: Located in `spoon_ai/agents/`, implements reasoning-action loop
- **ToolCall Agents**: Base class for agents with tool execution capabilities  
- **MCP Agents**: Runtime pluggable agents using Message Connectivity Protocol
- **SpoonReactAI**: Main intelligent agent combining reasoning and action

### Tool System
- **Tool Manager**: Centralized tool registration and execution (`spoon_ai/tools/tool_manager.py`)
- **Base Tool**: Abstract base class for all tools (`spoon_ai/tools/base.py`)
- **MCP Tools**: Runtime discoverable tools via MCP protocol
- **Crypto Tools**: Web3-specific tools for blockchain interaction

### Configuration System
- **Hybrid Configuration**: Uses `.env` for initial setup, `config.json` for runtime configuration
- **ConfigManager**: Handles dynamic configuration loading (`spoon_ai/utils/config_manager.py`)
- **Loading Priority**: `config.json` overrides `.env` variables

### Key Directories
- `spoon_ai/agents/`: Agent implementations and base classes
- `spoon_ai/tools/`: Tool definitions and management
- `spoon_ai/llm/`: LLM provider abstractions and factory
- `spoon_ai/trade/`: Web3 trading functionality
- `spoon_ai/monitoring/`: System monitoring and alerts
- `spoon_ai/social_media/`: Social platform integrations
- `cli/`: Command-line interface implementation
- `examples/`: Usage examples for agents and MCP

### LLM Provider Support
Supports multiple LLM providers via unified interface:
- OpenAI (GPT models)
- Anthropic (Claude models)
- DeepSeek
- OpenRouter (multi-LLM gateway)
- Google Gemini

### MCP (Message Connectivity Protocol)
- Runtime tool discovery and execution
- Supports stdio, http, and websocket transports
- Community agent mode via mcp-proxy
- Built-in agent mode with custom MCP servers

## Project Configuration Notes

- Uses `pyproject.toml` for package management and dependencies
- Python 3.9+ required, tested on 3.9-3.13
- FastMCP dependency requires websockets==15.0.1 (fixed version)
- Web3 integration with Web3.py 7.11.0

## Environment Setup

Required environment variables (see `.env.example`):
- `OPENAI_API_KEY`: For OpenAI models
- `ANTHROPIC_API_KEY`: For Claude models  
- `DEEPSEEK_API_KEY`: For DeepSeek models
- `PRIVATE_KEY`: Wallet private key for Web3 operations
- `RPC_URL`: Blockchain RPC endpoint
- `CHAIN_ID`: Target blockchain chain ID