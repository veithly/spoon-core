# 🌐 MCP (Message Connectivity Protocol)

<div align="center">
  <h3>Connect • Orchestrate • Scale</h3>
  <p><strong>The neural network of SpoonOS - enabling intelligent agent communication</strong></p>
  
  [![Python 3.12+](https://img.shields.io/badge/Python-3.12+-blue.svg)](https://www.python.org/downloads/)
  [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
  [![Async Support](https://img.shields.io/badge/Async-Supported-green.svg)](https://docs.python.org/3/library/asyncio.html)
</div>

<hr>

## 🚀 Overview

MCP provides a powerful communication backbone for the SpoonOS ecosystem, enabling:

- **🔄 Agent-to-Agent Communication** - Create networks of specialized agents that collaborate
- **⚡ Streaming Responses** - Real-time streaming output from language models
- **📈 Horizontal Scaling** - Distribute agents across multiple processes or machines
- **📡 Pub/Sub Messaging** - Flexible topic-based publish-subscribe pattern
- **🔌 WebSocket Protocol** - Fast, bidirectional communication channel

<div align="center">
  <img src="https://i.imgur.com/placeholder-for-diagram.png" alt="MCP Architecture" width="600"/>
  <p><em>MCP Architecture: Connecting Intelligent Agents Across the Network</em></p>
</div>

## 🧩 Core Components

The MCP module consists of these essential building blocks:

| Component | Description |
|-----------|-------------|
| **`MCPClient`** | Base client interface for MCP communications |
| **`SpoonMCPClient`** | Full-featured WebSocket client implementation |
| **`MCPAgentAdapter`** | Bridge between SpoonAI agents and MCP network |
| **`MCPConfig`** | Configuration settings for fine-tuning MCP connections |

## ⚡ Quick Start

### Basic Client Example

```python
import asyncio
from spoon_ai.mcp import MCPConfig, SpoonMCPClient

async def main():
    # Create MCP configuration
    config = MCPConfig(server_url="ws://localhost:8765")
    
    # Initialize the client
    client = SpoonMCPClient(config)
    
    # Connect to the MCP server
    await client.connect()
    
    # Define a message callback
    async def message_handler(message):
        print(f"Received message: {message}")
    
    # Subscribe to a topic
    await client.subscribe("test_topic", message_handler)
    
    # Send a message
    await client.send_message(
        recipient="agent_id",
        message={"text": "Hello, Agent!"},
        topic="test_topic"
    )
    
    # Wait for a while to receive messages
    await asyncio.sleep(10)
    
    # Disconnect
    await client.disconnect()

if __name__ == "__main__":
    asyncio.run(main())
```

### Agent Adapter Example

```python
import asyncio
from spoon_ai.mcp import MCPConfig, MCPAgentAdapter
from spoon_ai.agents.custom_agent import CustomAgent

async def main():
    # Create MCP configuration
    config = MCPConfig(server_url="ws://localhost:8765")
    
    # Initialize the adapter
    adapter = MCPAgentAdapter(config=config)
    
    # Connect to the MCP server
    await adapter.connect()
    
    # Create an agent
    agent_id = await adapter.create_custom_agent(
        name="test_agent",
        description="A test agent for demonstration",
        system_prompt="You are a helpful assistant."
    )
    
    # Subscribe the agent to a topic
    await adapter.agent_subscribe(agent_id, "test_topic")
    
    # Start the agent
    await adapter.start_agent(agent_id)
    
    # Send a message to the agent
    await adapter.send_message_to_agent(
        agent_id=agent_id,
        message="Hello! Can you help me with a question?",
        sender_id="user123",
        topic="test_topic"
    )
    
    # Wait for processing
    await asyncio.sleep(10)
    
    # Stop the agent
    await adapter.stop_agent(agent_id)
    
    # Disconnect
    await adapter.disconnect()

if __name__ == "__main__":
    asyncio.run(main())
```

## 🔥 Advanced Features

### 🤖 Multi-Agent Communication

Create powerful agent networks where specialized AIs collaborate to solve complex problems:

<div align="center">
  <pre>
  User Request → [Coordinator] → [Researcher] → [Writer] → Final Response
                      ↓               ↑
                 [Calculator] ←→ [Data Analyst]
  </pre>
</div>

```python
async def setup_agent_network():
    adapter = MCPAgentAdapter(config=MCPConfig(server_url="ws://localhost:8765"))
    await adapter.connect()
    
    # Create specialized agents
    coordinator_id = await adapter.create_custom_agent(
        name="coordinator",
        description="Task coordinator",
        system_prompt="You coordinate tasks among specialized agents."
    )
    
    research_id = await adapter.create_custom_agent(
        name="researcher",
        description="Research specialist",
        system_prompt="You research information on given topics."
    )
    
    writer_id = await adapter.create_custom_agent(
        name="writer",
        description="Content writer",
        system_prompt="You write content based on research."
    )
    
    # Subscribe agents to appropriate topics
    await adapter.agent_subscribe(coordinator_id, "tasks")
    await adapter.agent_subscribe(research_id, "research_requests")
    await adapter.agent_subscribe(writer_id, "writing_requests")
    
    # Start all agents
    await adapter.start_agent(coordinator_id)
    await adapter.start_agent(research_id)
    await adapter.start_agent(writer_id)
    
    # Initial request to coordinator
    await adapter.send_message_to_agent(
        agent_id=coordinator_id,
        message="Create a report on artificial intelligence trends",
        sender_id="user",
        topic="tasks"
    )
```

### ⚡ Streaming Support

Get real-time responses with streaming output support:

```python
async def streaming_example():
    client = SpoonMCPClient(MCPConfig(server_url="ws://localhost:8765"))
    await client.connect()
    
    # Stream processing callback
    async def stream_handler(message):
        content = message.get("content", {})
        if isinstance(content, dict) and content.get("type") == "stream_chunk":
            chunk = content.get("text", "")
            print(chunk, end="", flush=True)
    
    # Subscribe to streaming topic
    await client.subscribe("streaming_topic", stream_handler)
    
    # Send message requesting streaming response
    await client.send_message(
        recipient="StreamingAgent",
        message={
            "text": "Explain quantum computing",
            "metadata": {
                "request_stream": True
            }
        },
        topic="streaming_topic"
    )
```

## 🖥️ Running the MCP Server

The MCP system requires a WebSocket server to handle communication. A simple server implementation is provided in the examples directory.

```bash
# Start the MCP server
python -m spoon_ai.mcp.examples.simple_server

# With custom port
python -m spoon_ai.mcp.examples.simple_server --port 8888
```

The server will start on `localhost:8765` by default.

## ⚙️ Configuration Options

Fine-tune your MCP connection with these configuration options:

```python
config = MCPConfig(
    server_url="ws://localhost:8765",    # WebSocket server URL
    heartbeat_interval=15.0,             # Heartbeat interval in seconds
    reconnect_interval=5.0,              # Reconnection interval after disconnect
    connection_timeout=30.0,             # Connection timeout in seconds
    max_reconnect_attempts=5,            # Maximum reconnection attempts
    auto_reconnect=True,                 # Auto reconnect on disconnect
    auth_token=None                      # Authentication token (if required)
)
```

## 🛡️ Error Handling

The MCP module provides specialized exceptions for robust error handling:

```python
try:
    await client.connect()
except MCPConnectionError as e:
    print(f"Connection error: {e}")
except MCPAuthenticationError as e:
    print(f"Authentication error: {e}")
    
try:
    await client.subscribe("topic", message_handler)
except MCPSubscriptionError as e:
    print(f"Subscription error: {e}")
    
try:
    await client.send_message(recipient="agent", message={"text": "Hello"}, topic="topic")
except MCPMessageError as e:
    print(f"Message error: {e}")
```

## 🔧 Integration with Custom Agents

Create your own agent types and integrate them with the MCP network:

```python
from spoon_ai.agents.custom_agent import CustomAgent
from spoon_ai.mcp import MCPAgentAdapter, MCPConfig

class MyCustomAgent(CustomAgent):
    name = "custom_agent"
    description = "My specialized agent"
    
    async def process_message(self, message):
        # Custom message processing logic
        return {"response": "Processed your message"}

async def integrate_custom_agent():
    adapter = MCPAgentAdapter(config=MCPConfig(server_url="ws://localhost:8765"))
    await adapter.connect()
    
    # Register the custom agent class
    adapter.register_agent_class(MyCustomAgent)
    
    # Create an instance of the custom agent
    agent_id = await adapter.create_agent(
        agent_class=MyCustomAgent,
        name="my_custom_agent",
        description="Custom agent implementation"
    )
    
    # Rest of the setup remains the same
    await adapter.agent_subscribe(agent_id, "custom_topic")
    await adapter.start_agent(agent_id)
```

## 📚 Examples

Explore these examples to understand MCP's full potential:

| Example | Description |
|---------|-------------|
| **Basic Integration** | Simple client-server communication |
| **Multi-Agent** | Setting up agent networks for collaboration |
| **Streaming** | Working with real-time streaming responses |
| **Custom Agents** | Integrating with custom agent implementations |

For detailed implementation examples, refer to the `agent_integration.py` file in the MCP module.

<div align="center">
  <hr>
  <h3>🔮 Build the future of AI agent networks with SpoonOS MCP</h3>
</div> 