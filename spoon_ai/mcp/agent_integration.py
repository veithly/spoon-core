#!/usr/bin/env python
"""
MCP Agent 集成示例

本示例演示如何使用 MCPAgentAdapter 将 SpoonAI 的 Agent 与 MCP 系统集成
"""

import asyncio
import logging
import sys
import time
from typing import Dict, Any, List, Optional

from spoon_ai.mcp import MCPConfig, MCPAgentAdapter, SpoonMCPClient
from spoon_ai.agents.custom_agent import CustomAgent
from spoon_ai.agents.spoon_react import SpoonReactAI

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def basic_example():
    """Basic example of using MCP Agent Adapter"""
    
    adapter = MCPAgentAdapter(config=MCPConfig(server_url="ws://localhost:8765"))
    
    try:
        await adapter.connect()
        logger.info("Connected to MCP server")
        
        agent_id = await adapter.create_custom_agent(
            name="basic_agent",
            description="Basic agent example",
            system_prompt="You are a helpful assistant that can answer questions and perform tasks."
        )
        logger.info(f"Created agent: {agent_id}")
        
        await adapter.agent_subscribe(agent_id, "test")
        logger.info(f"Agent subscribed to topic: test")
        
        await adapter.start_agent(agent_id)
        logger.info(f"Agent started")
        
        await adapter.send_message_to_agent(
            agent_id=agent_id,
            message="Hello! Please introduce yourself.",
            sender_id="test_user",
            topic="test"
        )
        logger.info("Test message sent to agent")
        
        await asyncio.sleep(10)
        
        status = await adapter.get_agent_status(agent_id)
        logger.info(f"Agent status: {status}")
        
        await adapter.stop_agent(agent_id)
        logger.info(f"Agent stopped")
        
    except Exception as e:
        logger.error(f"Error occurred: {str(e)}")
    finally:
        await adapter.disconnect()
        logger.info("Disconnected from MCP server")

async def multi_agent_example():
    """Example with multiple agents communicating through MCP"""
    
    adapter = MCPAgentAdapter(config=MCPConfig(server_url="ws://localhost:8765"))
    
    try:
        await adapter.connect()
        logger.info("Connected to MCP server")
        
        assistant_id = await adapter.create_custom_agent(
            name="assistant",
            description="General assistant",
            system_prompt="You are a helpful assistant. When you receive a query that requires computation, forward it to the calculator agent."
        )
        
        calculator_id = await adapter.create_custom_agent(
            name="calculator",
            description="Calculation specialist",
            system_prompt="You are a calculation specialist. You can solve math problems and provide numerical answers."
        )
        
        calc_agent = adapter.agent_registry[calculator_id]
        calc_agent.add_tool(Calculator())
        
        logger.info(f"Created agents: {assistant_id}, {calculator_id}")
        
        await adapter.agent_subscribe(assistant_id, "general")
        await adapter.agent_subscribe(calculator_id, "calculations")
        
        await adapter.start_agent(assistant_id)
        await adapter.start_agent(calculator_id)
        logger.info("Agents started")
        
        await adapter.send_message_to_agent(
            agent_id=assistant_id,
            message="Can you help me calculate 1234 * 5678?",
            sender_id="user",
            topic="general"
        )
        logger.info("Query sent to assistant agent")
        
        await asyncio.sleep(5)
        
        await adapter.send_message_to_agent(
            agent_id=calculator_id,
            message="What is 1234 * 5678?",
            sender_id=assistant_id,
            topic="calculations"
        )
        logger.info("Calculation request sent to calculator agent")
        
        await asyncio.sleep(10)
        
        await adapter.stop_agent(assistant_id)
        await adapter.stop_agent(calculator_id)
        logger.info("Agents stopped")
        
    except Exception as e:
        logger.error(f"Error occurred: {str(e)}")
    finally:
        await adapter.disconnect()

async def agent_chain_example():
    """Example of a chain of agents working together"""
    
    adapter = MCPAgentAdapter(config=MCPConfig(server_url="ws://localhost:8765"))
    
    try:
        await adapter.connect()
        logger.info("Connected to MCP server")
        
        # Create a chain of 3 agents
        coordinator_id = await adapter.create_custom_agent(
            name="coordinator",
            description="Coordinates the workflow",
            system_prompt="You are a coordinator. Your job is to break down tasks and delegate to the appropriate specialist."
        )
        
        researcher_id = await adapter.create_custom_agent(
            name="researcher",
            description="Performs research",
            system_prompt="You are a researcher. Your job is to find information on topics provided by the coordinator."
        )
        
        summarizer_id = await adapter.create_custom_agent(
            name="summarizer",
            description="Summarizes information",
            system_prompt="You are a summarizer. Your job is to take detailed information and create concise summaries."
        )
        
        # Add tools to the researcher
        researcher = adapter.agent_registry[researcher_id]
        researcher.add_tool(WebSearch())
        
        logger.info(f"Created agent chain: {coordinator_id} -> {researcher_id} -> {summarizer_id}")
        
        # Subscribe agents to topics
        await adapter.agent_subscribe(coordinator_id, "requests")
        await adapter.agent_subscribe(researcher_id, "research")
        await adapter.agent_subscribe(summarizer_id, "summary")
        
        # Start all agents
        await adapter.start_agent(coordinator_id)
        await adapter.start_agent(researcher_id)
        await adapter.start_agent(summarizer_id)
        logger.info("All agents started")
        
        # Send initial request to the coordinator
        await adapter.send_message_to_agent(
            agent_id=coordinator_id,
            message="I need information about the history of artificial intelligence.",
            sender_id="user",
            topic="requests"
        )
        logger.info("Initial request sent to coordinator")
        
        # Allow time for the agent chain to process
        await asyncio.sleep(30)
        
        # Stop all agents
        await adapter.stop_agent(coordinator_id)
        await adapter.stop_agent(researcher_id)
        await adapter.stop_agent(summarizer_id)
        logger.info("All agents stopped")
        
    except Exception as e:
        logger.error(f"Error occurred: {str(e)}")
    finally:
        await adapter.disconnect()

async def streaming_example():
    """Streaming example: demonstrates Agent streaming output capabilities"""
    
    # Create MCP config and adapter
    adapter = MCPAgentAdapter(config=MCPConfig(server_url="ws://localhost:8765"))
    
    try:
        await adapter.connect()
        logger.info("Connected to MCP server")
        
        # Create custom streaming Agent
        agent_id = await adapter.create_custom_agent(
            name="StreamingAssistant",
            description="Assistant with streaming output support",
            system_prompt="You are a chat assistant. Please provide detailed answers with paragraphs and rich content."
        )
        logger.info(f"Created streaming Agent: {agent_id}")
        
        # Subscribe Agent to topic
        await adapter.agent_subscribe(agent_id, "stream")
        logger.info(f"Agent subscribed to topic: stream")
        
        # Start Agent
        await adapter.start_agent(agent_id)
        logger.info(f"Agent started")
        
        # Send test message to Agent, requesting streaming response
        await adapter.send_message_to_agent(
            agent_id=agent_id,
            message="Please provide a detailed overview of the history and future trends of artificial intelligence.",
            sender_id="stream_user",
            topic="stream",
            stream=True  # Request streaming response
        )
        logger.info("Test message sent and streaming response requested")
        
        # Run for a while to allow streaming response to complete
        logger.info("Waiting for Agent to process message and generate streaming response...")
        await asyncio.sleep(60)  # Allow enough time for streaming response to complete
        
        # Stop Agent
        await adapter.stop_agent(agent_id)
        logger.info(f"Agent stopped")
        
    except Exception as e:
        logger.error(f"Error occurred: {str(e)}")
    finally:
        await adapter.disconnect()
        logger.info("Disconnected from MCP server")

async def streaming_client_example():
    """Streaming client example: demonstrates how to receive and process streaming output"""
    
    # Create MCP config
    config = MCPConfig(
        server_url="ws://localhost:8765",
        heartbeat_interval=15.0,
        auto_reconnect=True
    )
    
    # Create MCP client
    client = SpoonMCPClient(config)
    
    # Current streams being received
    current_streams = {}
    
    # Callback to process streaming messages
    async def stream_callback(message):
        content = message.get("content", {})
        
        # Check if this is a streaming chunk
        if isinstance(content, dict) and content.get("type") == "stream_chunk":
            metadata = content.get("metadata", {})
            stream_id = metadata.get("stream_id", "unknown")
            is_final = metadata.get("is_final", False)
            chunk = content.get("text", "")
            
            # Initialize stream or append to existing stream
            if stream_id not in current_streams:
                current_streams[stream_id] = {
                    "content": "",
                    "metadata": metadata,
                    "started_at": time.time()
                }
                print(f"\nStarted receiving stream {stream_id} from {metadata.get('agent_name', 'unknown')}")
                
            # Append content
            current_streams[stream_id]["content"] += chunk
            
            # Print progress
            print(f"\rReceiving stream {stream_id}: {len(current_streams[stream_id]['content'])} characters received...", end="")
            
            # If this is the final chunk, output complete content
            if is_final:
                print(f"\nStream {stream_id} complete, time taken: {time.time() - current_streams[stream_id]['started_at']:.2f} seconds")
                print("\nComplete content:")
                print("-" * 50)
                print(current_streams[stream_id]["content"])
                print("-" * 50)
                
                # Remove completed stream
                del current_streams[stream_id]
        else:
            # Handle regular messages
            sender = message.get("sender", "unknown")
            if isinstance(content, dict) and "text" in content:
                text = content["text"]
            else:
                text = str(content)
            print(f"\nReceived message from {sender}: {text[:100]}...")
    
    try:
        # Connect to MCP server
        await client.connect()
        print("Connected to MCP server")
        
        # Subscribe to streaming topic
        await client.subscribe("stream", stream_callback)
        print("Subscribed to streaming topic")
        
        # Send message and request streaming response
        message = "Please provide a detailed overview of AI applications in healthcare."
        result = await client.send_message(
            recipient="StreamingAssistant",  # Assuming this Agent exists on the server
            message={
                "text": message,
                "metadata": {
                    "request_stream": True  # Request streaming response
                }
            },
            topic="stream"
        )
        print(f"Message sent: {message}")
        print(f"Send result: {result}")
        
        # Keep client running for a while
        print("Waiting for streaming responses...")
        await asyncio.sleep(120)  # Allow enough time to receive all streaming responses
        
    except Exception as e:
        print(f"Error occurred: {str(e)}")
    finally:
        await client.disconnect()
        print("Disconnected from MCP server")

async def spoon_react_streaming_example():
    """Example demonstrating SpoonReactAI with streaming capabilities"""
    
    # Create MCP config and adapter
    adapter = MCPAgentAdapter(config=MCPConfig(server_url="ws://localhost:8765"))
    
    try:
        await adapter.connect()
        logger.info("Connected to MCP server")
        
        # Create SpoonReactAI agent
        agent_id = await adapter.create_agent(
            agent_class=SpoonReactAI,
            name="SpoonReactStreaming",
            description="SpoonReact agent with streaming support"
        )
        logger.info(f"Created SpoonReactAI agent: {agent_id}")
        
        # Subscribe agent to topic
        await adapter.agent_subscribe(agent_id, "spoon_stream")
        logger.info(f"Agent subscribed to topic: spoon_stream")
        
        # Start agent
        await adapter.start_agent(agent_id)
        logger.info(f"Agent started")
        
        # Send test message to agent, requesting streaming response
        await adapter.send_message_to_agent(
            agent_id=agent_id,
            message="Can you analyze the price trend for NEO token?",
            sender_id="spoon_user",
            topic="spoon_stream",
            stream=True  # Request streaming response
        )
        logger.info("Test message sent and streaming response requested")
        
        # Run for a while to allow streaming response to complete
        logger.info("Waiting for Agent to process message and generate streaming response...")
        await asyncio.sleep(90)  # Allow enough time for streaming response to complete
        
        # Stop agent
        await adapter.stop_agent(agent_id)
        logger.info(f"Agent stopped")
        
    except Exception as e:
        logger.error(f"Error occurred: {str(e)}")
    finally:
        await adapter.disconnect()
        logger.info("Disconnected from MCP server")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        example = sys.argv[1]
        if example == "basic":
            asyncio.run(basic_example())
        elif example == "multi":
            asyncio.run(multi_agent_example())
        elif example == "chain":
            asyncio.run(agent_chain_example())
        elif example == "stream":
            asyncio.run(streaming_example())
        elif example == "stream_client":
            asyncio.run(streaming_client_example())
        elif example == "spoon_stream":
            asyncio.run(spoon_react_streaming_example())
        else:
            print(f"Unknown example: {example}")
            print("Available examples: basic, multi, chain, stream, stream_client, spoon_stream")
    else:
        print("Please specify which example to run: python agent_integration.py [basic|multi|chain|stream|stream_client|spoon_stream]")
        print("Example: python agent_integration.py spoon_stream") 