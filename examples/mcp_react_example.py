import asyncio
import logging
from typing import Optional

from fastmcp.client.transports import WSTransport, PythonStdioTransport, FastMCPTransport
from spoon_ai.agents.spoon_react import SpoonReactAI
from spoon_ai.chat import ChatBot

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def run_mcp_agent(websocket_url: str, topics: Optional[list] = None):
    """
    Run a SpoonReactAI agent using MCP
    
    Args:
        websocket_url: WebSocket URL of the MCP server
        topics: List of topics to subscribe to
    """
    # Create WebSocket 
    # transport
    
    transport = WSTransport(websocket_url)
    
    # Create SpoonReactAI instance
    agent = SpoonReactAI(
        mcp_transport=transport,
        mcp_topics=topics or ["spoon_react", "general"],
        llm=ChatBot()  # Use default model
    )
    
    # Initialize agent with connection and topic subscriptions
    class Context:
        async def register_async_task(self, task):
            await task
            
        async def report_error(self, error):
            logger.error(f"Context received error: {str(error)}")
    
    try:
        # Initialize the agent (this will connect to MCP and subscribe to topics)
        logger.info("Initializing agent...")
        await agent.initialize(Context())
        
        logger.info(f"Agent {agent.name} is running. Press Ctrl+C to stop.")
        
        # Keep the agent running until interrupted
        while True:
            await agent.run("你好")
            await asyncio.sleep(1)
    
    except KeyboardInterrupt:
        logger.info("Shutting down agent due to keyboard interrupt...")
    except Exception as e:
        logger.error(f"Error running agent: {str(e)}")

async def send_test_message_option1(websocket_url: str, message: str):
    """
    Send a test message to the MCP server using our mixin pattern
    
    Args:
        websocket_url: WebSocket URL of the MCP server
        message: Message to send
    """
    # Create WebSocket transport
    transport = WSTransport(websocket_url)
    
    # Create a temporary agent for sending messages
    agent = SpoonReactAI(mcp_transport=transport)
    
    try:
        # Verify connection first
        logger.info("Connecting to MCP server...")
        await agent.connect()
        
        logger.info(f"Sending message: {message}")
        
        # Send a message to the spoon_react topic
        result = await agent.send_mcp_message(
            recipient="spoon_react",
            message=message,
            topic="spoon_react",
            metadata={"sender_type": "user", "request_stream": True}
        )
        
        logger.info(f"Message sent successfully: {result}")
    except Exception as e:
        logger.error(f"Error sending message: {str(e)}")

async def send_test_message_option2(websocket_url: str, message: str):
    """
    Send a test message to the MCP server using the native Client directly
    (showing how our implementation matches the official usage)
    
    Args:
        websocket_url: WebSocket URL of the MCP server
        message: Message to send
    """
    from fastmcp.client import Client
    
    # Create transport
    transport = WSTransport(websocket_url)
    
    try:
        logger.info("Connecting to MCP server using native Client...")
        
        # Using the official client pattern with context manager
        async with Client(transport) as client:
            logger.info(f"Connected. Sending message: {message}")
            
            # Prepare message content
            content = {
                "text": message,
                "source": "user",
                "metadata": {
                    "sender_type": "user", 
                    "request_stream": True
                }
            }
            
            # Send message
            await client.send_message(
                recipient="spoon_react",
                message=content,
                topic="spoon_react"
            )
            
            logger.info("Message sent successfully using native Client")
    except Exception as e:
        logger.error(f"Error sending message with native Client: {str(e)}")

if __name__ == "__main__":
    import sys
    
    # WebSocket URL of the MCP server
    ws_url = "mcp_server.py"
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "send1":
            # Send test message using our custom mixin implementation
            message = " ".join(sys.argv[2:]) if len(sys.argv) > 2 else "Hello from MCP client (option 1)!"
            asyncio.run(send_test_message_option1(ws_url, message))
        elif sys.argv[1] == "send2":
            # Send test message using native client directly
            message = " ".join(sys.argv[2:]) if len(sys.argv) > 2 else "Hello from MCP client (option 2)!"
            asyncio.run(send_test_message_option2(ws_url, message))
        else:
            # Unknown command
            print(f"Unknown command: {sys.argv[1]}")
            print("Usage: python mcp_react_example.py [send1|send2|<nothing>]")
    else:
        # Run agent mode
        asyncio.run(run_mcp_agent(ws_url)) 