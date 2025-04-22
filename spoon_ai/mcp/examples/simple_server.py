#!/usr/bin/env python
"""
Simple WebSocket server that implements the MCP protocol for testing purposes.
"""

import asyncio
import json
import logging
import signal
import time
import uuid
from typing import Dict, Set, Any, List, Optional

import websockets
from websockets.server import WebSocketServerProtocol

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger("mcp_simple_server")

# Global state
clients: Dict[str, WebSocketServerProtocol] = {}  # Map of client_id to websocket
client_info: Dict[str, Dict[str, Any]] = {}      # Client metadata
topics: Dict[str, Set[str]] = {}                 # Map of topic to set of client_ids
auth_tokens: Dict[str, str] = {                  # Sample auth tokens (token -> client_id)
    "test_token": "test_client",
    "admin_token": "admin"
}


async def send_message(websocket: WebSocketServerProtocol, message_type: str, data: Any):
    """Send a formatted message to a websocket"""
    message = {
        "type": message_type,
        "data": data
    }
    await websocket.send(json.dumps(message))


async def broadcast_to_topic(topic: str, message: Dict[str, Any], exclude_client: Optional[str] = None):
    """Broadcast a message to all clients subscribed to a topic"""
    if topic not in topics:
        return
        
    message_type = "message"
    
    for client_id in topics[topic]:
        if exclude_client and client_id == exclude_client:
            continue
            
        if client_id in clients:
            try:
                await send_message(clients[client_id], message_type, message)
                logger.debug(f"Broadcasted message to client {client_id} on topic {topic}")
            except Exception as e:
                logger.error(f"Error broadcasting to client {client_id}: {str(e)}")


async def handle_client(websocket: WebSocketServerProtocol, path: str = ""):
    """Handle a client connection"""
    client_id = str(uuid.uuid4())
    remote = websocket.remote_address
    
    # Store client connection
    clients[client_id] = websocket
    client_info[client_id] = {
        "connected_at": time.time(),
        "address": f"{remote[0]}:{remote[1]}",
        "authenticated": False,
        "subscribed_topics": set(),
        "last_activity": time.time()
    }
    
    logger.info(f"New client connected: {client_id} from {remote[0]}:{remote[1]}")
    
    try:
        # Welcome message
        await send_message(websocket, "system", {
            "content": "Welcome to MCP Test Server",
            "server_time": time.time()
        })
        
        # Process messages
        async for message in websocket:
            client_info[client_id]["last_activity"] = time.time()
            
            try:
                data = json.loads(message)
                message_type = data.get("type")
                
                if message_type == "auth":
                    # Handle authentication
                    token = data.get("token")
                    if token in auth_tokens:
                        authenticated_id = auth_tokens[token]
                        
                        # If this token is already associated with another connection, use that ID
                        if authenticated_id != client_id:
                            # Remove the old client entry if it exists
                            if authenticated_id in clients:
                                old_client = clients[authenticated_id]
                                await send_message(old_client, "system", {
                                    "content": "Your session has been taken over by a new connection",
                                    "server_time": time.time()
                                })
                                await old_client.close()
                                
                            # Update the client ID references
                            clients[authenticated_id] = websocket
                            client_info[authenticated_id] = client_info[client_id]
                            del clients[client_id]
                            client_id = authenticated_id
                            
                        client_info[client_id]["authenticated"] = True
                        logger.info(f"Client {client_id} authenticated")
                        
                        await send_message(websocket, "system", {
                            "content": "Authentication successful",
                            "client_id": client_id,
                            "server_time": time.time()
                        })
                    else:
                        logger.warning(f"Failed authentication attempt from client {client_id}")
                        await send_message(websocket, "error", {
                            "content": "Authentication failed: Invalid token",
                            "server_time": time.time()
                        })
                        
                elif message_type == "heartbeat":
                    # Respond to heartbeat
                    await send_message(websocket, "heartbeat", {
                        "server_time": time.time()
                    })
                    
                elif message_type == "subscribe":
                    # Handle topic subscription
                    topic = data.get("topic")
                    if not topic:
                        await send_message(websocket, "error", {
                            "content": "Subscribe failed: Missing topic",
                            "server_time": time.time()
                        })
                        continue
                        
                    # Create topic if it doesn't exist
                    if topic not in topics:
                        topics[topic] = set()
                    
                    # Add client to topic
                    topics[topic].add(client_id)
                    client_info[client_id]["subscribed_topics"].add(topic)
                    
                    logger.info(f"Client {client_id} subscribed to topic: {topic}")
                    await send_message(websocket, "system", {
                        "content": f"Subscribed to topic: {topic}",
                        "topic": topic,
                        "server_time": time.time()
                    })
                    
                elif message_type == "unsubscribe":
                    # Handle topic unsubscription
                    topic = data.get("topic")
                    if not topic:
                        await send_message(websocket, "error", {
                            "content": "Unsubscribe failed: Missing topic",
                            "server_time": time.time()
                        })
                        continue
                        
                    # Remove client from topic
                    if topic in topics and client_id in topics[topic]:
                        topics[topic].remove(client_id)
                        # Clean up empty topics
                        if not topics[topic]:
                            del topics[topic]
                            
                    if topic in client_info[client_id]["subscribed_topics"]:
                        client_info[client_id]["subscribed_topics"].remove(topic)
                    
                    logger.info(f"Client {client_id} unsubscribed from topic: {topic}")
                    await send_message(websocket, "system", {
                        "content": f"Unsubscribed from topic: {topic}",
                        "topic": topic,
                        "server_time": time.time()
                    })
                    
                elif message_type == "message":
                    # Handle client message
                    message_data = data.get("data", {})
                    
                    # Validate message format
                    required_fields = ["id", "sender", "recipient", "content", "timestamp"]
                    if not all(field in message_data for field in required_fields):
                        await send_message(websocket, "error", {
                            "content": "Invalid message format: Missing required fields",
                            "server_time": time.time()
                        })
                        continue
                    
                    # Override sender with actual client ID to prevent spoofing
                    message_data["sender"] = client_id
                    
                    # Get topic and recipient
                    topic = message_data.get("topic")
                    recipient = message_data.get("recipient")
                    
                    # Send acknowledgement to sender
                    await send_message(websocket, "ack", {
                        "message_id": message_data["id"],
                        "server_time": time.time()
                    })
                    
                    # Process message based on recipient and topic
                    if recipient == "all" and topic:
                        # Broadcast to topic
                        await broadcast_to_topic(topic, message_data, exclude_client=client_id)
                        logger.info(f"Broadcast message from {client_id} to topic {topic}")
                    elif recipient != "all" and recipient in clients:
                        # Direct message to specific client
                        try:
                            await send_message(clients[recipient], "message", message_data)
                            logger.info(f"Sent direct message from {client_id} to {recipient}")
                        except Exception as e:
                            logger.error(f"Error sending direct message: {str(e)}")
                    else:
                        logger.warning(f"Could not deliver message from {client_id}: Unknown recipient {recipient}")
                
                else:
                    # Unknown message type
                    logger.warning(f"Unknown message type from client {client_id}: {message_type}")
                    await send_message(websocket, "error", {
                        "content": f"Unknown message type: {message_type}",
                        "server_time": time.time()
                    })
                    
            except json.JSONDecodeError:
                logger.error(f"Invalid JSON from client {client_id}: {message}")
                await send_message(websocket, "error", {
                    "content": "Invalid JSON format",
                    "server_time": time.time()
                })
            except Exception as e:
                logger.error(f"Error processing message from client {client_id}: {str(e)}")
                
    except websockets.exceptions.ConnectionClosed:
        logger.info(f"Client {client_id} connection closed")
    except Exception as e:
        logger.error(f"Error handling client {client_id}: {str(e)}")
    finally:
        # Clean up client resources
        for topic in list(topics.keys()):
            if client_id in topics[topic]:
                topics[topic].remove(client_id)
                if not topics[topic]:
                    del topics[topic]
        
        if client_id in clients:
            del clients[client_id]
        if client_id in client_info:
            del client_info[client_id]
            
        logger.info(f"Client {client_id} disconnected")


async def status_reporter():
    """Periodically report server status"""
    while True:
        try:
            logger.info(f"Server status: {len(clients)} active clients, {len(topics)} active topics")
            if clients:
                client_list = ", ".join(clients.keys())
                logger.info(f"Connected clients: {client_list}")
            if topics:
                for topic, subscribers in topics.items():
                    logger.info(f"Topic '{topic}': {len(subscribers)} subscribers")
                    
            await asyncio.sleep(60)  # Report every minute
        except Exception as e:
            logger.error(f"Error in status reporter: {str(e)}")
            await asyncio.sleep(60)


async def start_server():
    """Start the MCP server"""
    # Start the WebSocket server
    server = await websockets.serve(
        handle_client,
        "0.0.0.0",  # Listen on all interfaces
        8765,       # Default MCP port
        ping_interval=30,
        ping_timeout=10
    )
    
    logger.info("MCP Test Server started on ws://0.0.0.0:8765")
    
    # Start the status reporter
    status_task = asyncio.create_task(status_reporter())
    
    # Setup signal handling for graceful shutdown
    loop = asyncio.get_running_loop()
    
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, lambda: asyncio.create_task(shutdown(server, status_task)))
        
    try:
        # Keep the server running
        await server.wait_closed()
    except asyncio.CancelledError:
        logger.info("Server task cancelled")


async def shutdown(server, status_task):
    """Gracefully shut down the server"""
    logger.info("Shutting down MCP Test Server...")
    
    # Notify all clients
    for client_id, websocket in clients.items():
        try:
            await send_message(websocket, "system", {
                "content": "Server shutting down",
                "server_time": time.time()
            })
        except Exception:
            pass
    
    # Cancel the status reporter task
    if not status_task.done():
        status_task.cancel()
        
    # Close all client connections
    for client_id, websocket in list(clients.items()):
        try:
            await websocket.close()
        except Exception:
            pass
            
    # Close the server
    server.close()
    await server.wait_closed()
    
    logger.info("MCP Test Server shutdown complete")


if __name__ == "__main__":
    try:
        asyncio.run(start_server())
    except KeyboardInterrupt:
        logger.info("Server interrupted by user")
    except Exception as e:
        logger.error(f"Server error: {str(e)}")
    finally:
        logger.info("Server shutdown") 