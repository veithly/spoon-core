import logging
import asyncio
import json
import time
import uuid
from typing import Dict, Any, List, Optional, Union, Callable, Type
import websockets
from pydantic import BaseModel

from .base import MCPClient
from .config import MCPConfig
from .exceptions import MCPConnectionError, MCPMessageError, MCPError

logger = logging.getLogger(__name__)

class MCPMessage(BaseModel):
    """MCP Message model"""
    id: str
    sender: str
    recipient: str
    content: Union[str, Dict[str, Any]]
    timestamp: float
    topic: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

class SpoonMCPClient(MCPClient):
    """
    Implementation of MCP client for SpoonAI.
    Provides message-based communication capabilities using WebSocket.
    Supports integration with agents for automated message processing.
    """
    
    def __init__(self, config: Optional[MCPConfig] = None):
        """
        Initialize the MCP client
        
        Args:
            config: Configuration for the MCP client
        """
        self.config = config or MCPConfig()
        self.connection = None
        self.subscriptions = {}
        self.client_id = str(uuid.uuid4())
        self.connected = False
        self.message_queue = asyncio.Queue()
        self.listener_task = None
        self.heartbeat_task = None
        self.connection_monitor_task = None
        self.last_heartbeat_received = time.time()
        self.connection_error_count = 0
        
        # Agent management
        self.agents = {}  # Maps agent_id to agent instance
        self.agent_topics = {}  # Maps agent_id to list of subscribed topics
        self.agent_tasks = {}  # Maps agent_id to running tasks
        
        # Message history
        self.message_history = []
        self.max_history_size = 1000
        
        # Message tracking for retries
        self.pending_messages = {}  # Maps message_id to message data
        self.retry_counts = {}  # Maps message_id to retry count
        
        # Error tracking
        self.error_history = []
        self.max_error_history = 100
        
        # Connection event callbacks
        self.on_connect_callbacks = []
        self.on_disconnect_callbacks = []
        self.on_error_callbacks = []
        
    async def connect(self) -> bool:
        """
        Connect to the MCP server
        
        Returns:
            bool: True if connection is successful, False otherwise
        """
        try:
            # Reset error count if we're explicitly trying to connect
            self.connection_error_count = 0
            
            self.connection = await websockets.connect(
                self.config.server_url,
                ping_interval=self.config.heartbeat_interval,
                ping_timeout=self.config.connection_timeout,
                close_timeout=self.config.connection_timeout
            )
            self.connected = True
            
            # Start the background listener
            if not self.listener_task or self.listener_task.done():
                self.listener_task = asyncio.create_task(self._listen_for_messages())
                
            # Start the heartbeat task
            if not self.heartbeat_task or self.heartbeat_task.done():
                self.heartbeat_task = asyncio.create_task(self._send_heartbeat())
            
            # Start the connection monitor task
            if not self.connection_monitor_task or self.connection_monitor_task.done():
                self.connection_monitor_task = asyncio.create_task(self._monitor_connection())
                
            # Send authentication if credentials are provided
            if self.config.auth_token:
                auth_message = {
                    "type": "auth",
                    "client_id": self.client_id,
                    "token": self.config.auth_token
                }
                await self.connection.send(json.dumps(auth_message))
                
            logger.info(f"Connected to MCP server at {self.config.server_url}")
            
            # Trigger connect callbacks
            for callback in self.on_connect_callbacks:
                try:
                    await callback(self.client_id)
                except Exception as e:
                    logger.error(f"Error in connect callback: {str(e)}")
                
            return True
            
        except Exception as e:
            self.connected = False
            self.connection_error_count += 1
            
            # Log error with increasing severity based on retry count
            if self.connection_error_count <= 3:
                logger.warning(f"Failed to connect to MCP server (attempt {self.connection_error_count}): {str(e)}")
            else:
                logger.error(f"Failed to connect to MCP server (attempt {self.connection_error_count}): {str(e)}")
                
            # Add to error history
            self._add_to_error_history("connection_error", str(e))
            
            # Trigger error callbacks
            for callback in self.on_error_callbacks:
                try:
                    await callback("connection_error", str(e))
                except Exception as callback_err:
                    logger.error(f"Error in error callback: {str(callback_err)}")
                
            # Raise exception after recording error info
            raise MCPConnectionError(f"Connection error: {str(e)}")
            
    async def disconnect(self) -> bool:
        """
        Disconnect from the MCP server
        
        Returns:
            bool: True if disconnection is successful, False otherwise
        """
        if not self.connected or not self.connection:
            return True
            
        try:
            if self.listener_task and not self.listener_task.done():
                self.listener_task.cancel()
                
            if self.heartbeat_task and not self.heartbeat_task.done():
                self.heartbeat_task.cancel()
                
            if self.connection_monitor_task and not self.connection_monitor_task.done():
                self.connection_monitor_task.cancel()
                
            await self.connection.close()
            self.connected = False
            logger.info("Disconnected from MCP server")
            
            # Trigger disconnect callbacks
            for callback in self.on_disconnect_callbacks:
                try:
                    await callback(self.client_id)
                except Exception as e:
                    logger.error(f"Error in disconnect callback: {str(e)}")
                
            return True
            
        except Exception as e:
            logger.error(f"Error during disconnection: {str(e)}")
            
            # Add to error history
            self._add_to_error_history("disconnect_error", str(e))
            return False
    
    async def send_message(self, recipient: str, message: Union[str, Dict[str, Any]], topic: Optional[str] = None) -> Dict[str, Any]:
        """
        Send a message to a specific recipient
        
        Args:
            recipient: The recipient identifier
            message: The message content as string or dict
            topic: Optional topic for the message
            
        Returns:
            Dict[str, Any]: Response data including message ID and status
        """
        if not self.connected:
            await self.connect()
            
        message_id = str(uuid.uuid4())
        timestamp = time.time()
        
        mcp_message = MCPMessage(
            id=message_id,
            sender=self.client_id,
            recipient=recipient,
            content=message,
            timestamp=timestamp,
            topic=topic
        )
        
        try:
            message_dict = mcp_message.dict()
            raw_message = {
                "type": "message",
                "data": message_dict
            }
            await self.connection.send(json.dumps(raw_message))
            
            # Track the message for potential retry
            self.pending_messages[message_id] = {
                "raw_message": raw_message,
                "timestamp": timestamp,
                "recipient": recipient
            }
            self.retry_counts[message_id] = 0
            
            # Add to history
            self._add_to_history(message_dict)
            
            logger.debug(f"Message sent: {message_id} to {recipient}")
            return {
                "message_id": message_id,
                "timestamp": timestamp,
                "status": "sent"
            }
            
        except Exception as e:
            logger.error(f"Failed to send message: {str(e)}")
            raise MCPMessageError(f"Failed to send message: {str(e)}")
    
    async def receive_messages(self, count: int = 10) -> List[Dict[str, Any]]:
        """
        Retrieve recent messages from the queue
        
        Args:
            count: Maximum number of messages to retrieve
            
        Returns:
            List[Dict[str, Any]]: List of received messages with metadata
        """
        if not self.connected:
            await self.connect()
            
        messages = []
        for _ in range(min(count, self.message_queue.qsize())):
            try:
                message = self.message_queue.get_nowait()
                messages.append(message)
                self.message_queue.task_done()
            except asyncio.QueueEmpty:
                break
                
        return messages
    
    async def subscribe(self, topic: str, callback: Callable) -> bool:
        """
        Subscribe to a specific topic/channel
        
        Args:
            topic: The topic or channel to subscribe to
            callback: The callback function to be called when a message is received
            
        Returns:
            bool: True if subscription is successful, False otherwise
        """
        if not self.connected:
            await self.connect()
            
        try:
            subscription_message = {
                "type": "subscribe",
                "client_id": self.client_id,
                "topic": topic
            }
            await self.connection.send(json.dumps(subscription_message))
            
            # Register the callback
            self.subscriptions[topic] = callback
            logger.info(f"Subscribed to topic: {topic}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to subscribe to topic {topic}: {str(e)}")
            return False
    
    async def unsubscribe(self, topic: str) -> bool:
        """
        Unsubscribe from a specific topic/channel
        
        Args:
            topic: The topic or channel to unsubscribe from
            
        Returns:
            bool: True if unsubscription is successful, False otherwise
        """
        if not self.connected:
            return True
            
        try:
            unsubscription_message = {
                "type": "unsubscribe",
                "client_id": self.client_id,
                "topic": topic
            }
            await self.connection.send(json.dumps(unsubscription_message))
            
            # Remove the callback
            if topic in self.subscriptions:
                del self.subscriptions[topic]
                
            logger.info(f"Unsubscribed from topic: {topic}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to unsubscribe from topic {topic}: {str(e)}")
            return False
    
    async def get_status(self) -> Dict[str, Any]:
        """
        Get the current status of the MCP client
        
        Returns:
            Dict[str, Any]: Status information
        """
        status = {
            "connected": self.connected,
            "client_id": self.client_id,
            "server_url": self.config.server_url,
            "subscriptions": list(self.subscriptions.keys()),
            "message_queue_size": self.message_queue.qsize(),
            "last_heartbeat": self.last_heartbeat_received,
            "heartbeat_age": time.time() - self.last_heartbeat_received,
            "pending_messages": len(self.pending_messages),
            "connection_error_count": self.connection_error_count,
            "recent_errors": self.error_history[-5:] if self.error_history else []
        }
        
        # Add agent information if any
        if self.agents:
            status["agents"] = {
                agent_id: {
                    "type": agent.__class__.__name__,
                    "is_running": agent_id in self.agent_tasks and not self.agent_tasks[agent_id].done(),
                    "subscribed_topics": self.agent_topics.get(agent_id, [])
                } for agent_id, agent in self.agents.items()
            }
            
        return status
        
    async def _listen_for_messages(self):
        """Background task for listening to incoming messages"""
        if not self.connected or not self.connection:
            return
            
        try:
            async for message in self.connection:
                try:
                    data = json.loads(message)
                    message_type = data.get("type")
                    
                    if message_type == "message":
                        message_data = data.get("data", {})
                        
                        # Convert to MCPMessage for validation
                        mcp_message = MCPMessage(**message_data)
                        
                        # Add to message history
                        self._add_to_history(mcp_message.dict())
                        
                        # Process the message
                        await self._process_message(mcp_message)
                        
                    elif message_type == "system":
                        # Handle system messages
                        logger.info(f"System message: {data.get('content')}")
                        
                    elif message_type == "error":
                        # Handle error messages
                        error_content = data.get('content', 'Unknown error')
                        logger.error(f"MCP error: {error_content}")
                        
                        # Add to error history
                        self._add_to_error_history("server_error", error_content)
                        
                        # Trigger error callbacks
                        for callback in self.on_error_callbacks:
                            try:
                                await callback("server_error", error_content)
                            except Exception as e:
                                logger.error(f"Error in error callback: {str(e)}")
                        
                    elif message_type == "heartbeat":
                        # Update last heartbeat time
                        self.last_heartbeat_received = time.time()
                        logger.debug("Heartbeat received from server")
                        
                    elif message_type == "ack":
                        # Handle message acknowledgement
                        msg_id = data.get("message_id")
                        if msg_id and msg_id in self.pending_messages:
                            del self.pending_messages[msg_id]
                            if msg_id in self.retry_counts:
                                del self.retry_counts[msg_id]
                            logger.debug(f"Message {msg_id} acknowledged by server")
                        
                except json.JSONDecodeError:
                    logger.error(f"Received invalid JSON: {message}")
                    self._add_to_error_history("invalid_json", str(message)[:100])
                except Exception as e:
                    logger.error(f"Error processing message: {str(e)}")
                    self._add_to_error_history("message_processing_error", str(e))
                    
        except websockets.exceptions.ConnectionClosed:
            logger.info("MCP connection closed")
            self.connected = False
            
            # Trigger disconnect callbacks
            for callback in self.on_disconnect_callbacks:
                try:
                    await callback(self.client_id)
                except Exception as e:
                    logger.error(f"Error in disconnect callback: {str(e)}")
                    
        except Exception as e:
            logger.error(f"Listener error: {str(e)}")
            self.connected = False
            self._add_to_error_history("listener_error", str(e))
            
        # Try to reconnect if disconnected unexpectedly
        if not self.connected and self.config.auto_reconnect:
            logger.info("Attempting to reconnect...")
            await asyncio.sleep(self.config.reconnect_delay)
            asyncio.create_task(self.connect())
    
    async def _process_message(self, message: MCPMessage):
        """
        Process an incoming message
        
        Args:
            message: The received message
        """
        # Add to the message queue
        await self.message_queue.put(message.dict())
        
        # Check if this message belongs to a subscribed topic
        if message.topic and message.topic in self.subscriptions:
            callback = self.subscriptions[message.topic]
            try:
                asyncio.create_task(callback(message.dict()))
            except Exception as e:
                logger.error(f"Error in subscription callback: {str(e)}")
                
        logger.debug(f"Received message: {message.id} from {message.sender}")
    
    # ===== Agent Integration Methods =====
    
    async def register_agent(self, agent_id: str, agent_instance: Any) -> bool:
        """
        Register an agent with the MCP client
        
        Args:
            agent_id: Unique identifier for the agent
            agent_instance: Instance of the agent to register
            
        Returns:
            bool: True if registration is successful, False otherwise
        """
        if agent_id in self.agents:
            logger.warning(f"Agent {agent_id} is already registered")
            return False
            
        self.agents[agent_id] = agent_instance
        self.agent_topics[agent_id] = []
        
        logger.info(f"Agent {agent_id} registered with MCP client")
        return True
    
    async def unregister_agent(self, agent_id: str) -> bool:
        """
        Unregister an agent from the MCP client
        
        Args:
            agent_id: The ID of the agent to unregister
            
        Returns:
            bool: True if unregistration is successful, False otherwise
        """
        if agent_id not in self.agents:
            logger.warning(f"Agent {agent_id} is not registered")
            return False
            
        # Unsubscribe agent from all topics
        for topic in self.agent_topics.get(agent_id, []):
            await self.agent_unsubscribe(agent_id, topic)
            
        # Cancel any running tasks
        if agent_id in self.agent_tasks:
            task = self.agent_tasks[agent_id]
            if not task.done():
                task.cancel()
            del self.agent_tasks[agent_id]
            
        # Remove agent from registries
        del self.agents[agent_id]
        if agent_id in self.agent_topics:
            del self.agent_topics[agent_id]
            
        logger.info(f"Agent {agent_id} unregistered from MCP client")
        return True
    
    async def agent_send_message(self, agent_id: str, recipient: str, message: Union[str, Dict[str, Any]], topic: Optional[str] = None) -> Dict[str, Any]:
        """
        Send a message from an agent to a recipient
        
        Args:
            agent_id: The ID of the agent sending the message
            recipient: The recipient identifier
            message: The message content as string or dict
            topic: Optional topic for the message
            
        Returns:
            Dict[str, Any]: Response data including message ID and status
        """
        if agent_id not in self.agents:
            raise MCPError(f"Agent {agent_id} is not registered")
            
        # Add agent metadata to the message
        agent_metadata = {
            "agent_id": agent_id,
            "agent_name": getattr(self.agents[agent_id], "name", "unknown"),
            "agent_type": self.agents[agent_id].__class__.__name__
        }
        
        # Convert message content if it's a string
        content = message
        if isinstance(message, str):
            content = {
                "text": message,
                "source": "agent",
                "metadata": agent_metadata
            }
        elif isinstance(message, dict):
            if "metadata" not in message:
                message["metadata"] = {}
            message["metadata"].update(agent_metadata)
            content = message
            
        # Send the message
        return await self.send_message(recipient, content, topic)
    
    async def agent_subscribe(self, agent_id: str, topic: str) -> bool:
        """
        Subscribe an agent to a topic
        
        Args:
            agent_id: The ID of the agent to subscribe
            topic: The topic to subscribe to
            
        Returns:
            bool: True if subscription is successful, False otherwise
        """
        if agent_id not in self.agents:
            raise MCPError(f"Agent {agent_id} is not registered")
            
        # Create a callback that will handle messages for this agent
        async def agent_message_handler(message):
            agent = self.agents[agent_id]
            try:
                # Extract the actual message content
                content = message.get("content", "")
                sender = message.get("sender", "unknown")
                
                # Process the message with the agent
                if hasattr(agent, "process_mcp_message") and callable(agent.process_mcp_message):
                    await agent.process_mcp_message(content, sender, message, agent_id)
                elif hasattr(agent, "add_message") and callable(agent.add_message):
                    # Fall back to basic message handling if available
                    if isinstance(content, dict) and "text" in content:
                        agent.add_message("user", content["text"])
                    elif isinstance(content, str):
                        agent.add_message("user", content)
                    else:
                        logger.warning(f"Unsupported message format for agent {agent_id}: {type(content)}")
                else:
                    logger.warning(f"Agent {agent_id} does not have a message handling method")
            except Exception as e:
                logger.error(f"Error processing message for agent {agent_id}: {str(e)}")
        
        # Subscribe to the topic with the agent's handler
        success = await self.subscribe(topic, agent_message_handler)
        
        if success:
            # Track the agent's subscription
            if agent_id not in self.agent_topics:
                self.agent_topics[agent_id] = []
            self.agent_topics[agent_id].append(topic)
            logger.info(f"Agent {agent_id} subscribed to topic: {topic}")
            
        return success
    
    async def agent_unsubscribe(self, agent_id: str, topic: str) -> bool:
        """
        Unsubscribe an agent from a topic
        
        Args:
            agent_id: The ID of the agent to unsubscribe
            topic: The topic to unsubscribe from
            
        Returns:
            bool: True if unsubscription is successful, False otherwise
        """
        if agent_id not in self.agents:
            raise MCPError(f"Agent {agent_id} is not registered")
            
        success = await self.unsubscribe(topic)
        
        if success and agent_id in self.agent_topics and topic in self.agent_topics[agent_id]:
            self.agent_topics[agent_id].remove(topic)
            logger.info(f"Agent {agent_id} unsubscribed from topic: {topic}")
            
        return success
    
    async def start_agent(self, agent_id: str, input_message: Optional[str] = None) -> bool:
        """
        Start an agent's processing loop
        
        Args:
            agent_id: The ID of the agent to start
            input_message: Optional initial message to start processing
            
        Returns:
            bool: True if agent was started successfully, False otherwise
        """
        if agent_id not in self.agents:
            raise MCPError(f"Agent {agent_id} is not registered")
            
        agent = self.agents[agent_id]
        
        # Check if agent is already running
        if agent_id in self.agent_tasks and not self.agent_tasks[agent_id].done():
            logger.warning(f"Agent {agent_id} is already running")
            return False
            
        # Define the agent task
        async def agent_task():
            try:
                logger.info(f"Starting agent {agent_id}")
                
                # Run the agent with the input message
                if hasattr(agent, "run") and callable(agent.run):
                    result = await agent.run(input_message)
                    
                    # Send the result as a message if needed
                    if result and hasattr(agent, "output_topic") and agent.output_topic:
                        await self.agent_send_message(
                            agent_id, 
                            "system", 
                            {"result": result, "status": "completed"},
                            agent.output_topic
                        )
                        
                logger.info(f"Agent {agent_id} completed its task")
            except Exception as e:
                logger.error(f"Error running agent {agent_id}: {str(e)}")
                
                # Send error notification if needed
                try:
                    if hasattr(agent, "output_topic") and agent.output_topic:
                        await self.agent_send_message(
                            agent_id, 
                            "system", 
                            {"error": str(e), "status": "error"},
                            agent.output_topic
                        )
                except Exception:
                    pass
                    
        # Create and store the task
        task = asyncio.create_task(agent_task())
        self.agent_tasks[agent_id] = task
        
        return True
    
    async def stop_agent(self, agent_id: str) -> bool:
        """
        Stop a running agent
        
        Args:
            agent_id: The ID of the agent to stop
            
        Returns:
            bool: True if agent was stopped successfully, False otherwise
        """
        if agent_id not in self.agents:
            raise MCPError(f"Agent {agent_id} is not registered")
            
        if agent_id not in self.agent_tasks or self.agent_tasks[agent_id].done():
            logger.warning(f"Agent {agent_id} is not running")
            return False
            
        # Cancel the agent task
        task = self.agent_tasks[agent_id]
        task.cancel()
        
        try:
            await task
        except asyncio.CancelledError:
            logger.info(f"Agent {agent_id} task cancelled")
        except Exception as e:
            logger.error(f"Error while stopping agent {agent_id}: {str(e)}")
            
        return True
    
    async def get_agent_status(self, agent_id: str) -> Dict[str, Any]:
        """
        Get the status of a registered agent
        
        Args:
            agent_id: The ID of the agent
            
        Returns:
            Dict[str, Any]: Agent status information
        """
        if agent_id not in self.agents:
            raise MCPError(f"Agent {agent_id} is not registered")
            
        agent = self.agents[agent_id]
        
        # Check if agent task is running
        is_running = (
            agent_id in self.agent_tasks and 
            not self.agent_tasks[agent_id].done()
        )
        
        # Get agent state if available
        agent_state = "unknown"
        if hasattr(agent, "state"):
            agent_state = str(agent.state)
            
        return {
            "agent_id": agent_id,
            "agent_type": agent.__class__.__name__,
            "is_running": is_running,
            "state": agent_state,
            "subscribed_topics": self.agent_topics.get(agent_id, []),
            "details": {
                "name": getattr(agent, "name", "unknown"),
                "description": getattr(agent, "description", None),
                "current_step": getattr(agent, "current_step", 0),
                "max_steps": getattr(agent, "max_steps", 0)
            }
        }

    async def _send_heartbeat(self):
        """Background task for sending periodic heartbeats to the server"""
        while self.connected:
            try:
                if self.connection:
                    try:
                        heartbeat_message = {
                            "type": "heartbeat",
                            "client_id": self.client_id,
                            "timestamp": time.time()
                        }
                        await self.connection.send(json.dumps(heartbeat_message))
                        logger.debug("Heartbeat sent to server")
                        
                        # Check for message retries
                        await self._retry_pending_messages()
                        
                        # Check connection health
                        heartbeat_age = time.time() - self.last_heartbeat_received
                        if heartbeat_age > self.config.heartbeat_interval * 3:
                            logger.warning(f"No heartbeat received for {heartbeat_age:.1f} seconds, reconnecting...")
                            self.connected = False
                            await self.connection.close()
                            await self.connect()
                    except websockets.exceptions.ConnectionClosed:
                        logger.warning("Connection closed while sending heartbeat, reconnecting...")
                        self.connected = False
                        await self.connect()
                        
                await asyncio.sleep(self.config.heartbeat_interval)
            except Exception as e:
                logger.error(f"Error in heartbeat task: {str(e)}")
                await asyncio.sleep(self.config.reconnect_delay)

    def _add_to_history(self, message: Dict[str, Any]):
        """Add a message to the history with size limit enforcement"""
        self.message_history.append(message)
        # Trim history if needed
        if len(self.message_history) > self.max_history_size:
            self.message_history = self.message_history[-self.max_history_size:]
            
    async def _retry_pending_messages(self):
        """Retry sending any pending messages that have not been acknowledged"""
        current_time = time.time()
        retry_timeout = self.config.heartbeat_interval * 2
        
        for message_id, message_data in list(self.pending_messages.items()):
            if current_time - message_data["timestamp"] > retry_timeout:
                retry_count = self.retry_counts.get(message_id, 0)
                
                # Give up after 3 retries
                if retry_count >= 3:
                    logger.warning(f"Message {message_id} failed after 3 retries, giving up")
                    del self.pending_messages[message_id]
                    if message_id in self.retry_counts:
                        del self.retry_counts[message_id]
                    continue
                    
                # Retry sending the message
                try:
                    logger.info(f"Retrying message {message_id}, attempt {retry_count + 1}")
                    await self.connection.send(json.dumps(message_data["raw_message"]))
                    self.pending_messages[message_id]["timestamp"] = current_time
                    self.retry_counts[message_id] = retry_count + 1
                except Exception as e:
                    logger.error(f"Error retrying message {message_id}: {str(e)}")

    async def get_message_history(self, count: int = 50, filter_by: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Retrieve message history with optional filtering
        
        Args:
            count: Maximum number of messages to retrieve
            filter_by: Optional filter criteria (e.g., {"sender": "client_id", "topic": "my_topic"})
            
        Returns:
            List[Dict[str, Any]]: List of historical messages matching criteria
        """
        if not filter_by:
            # Return most recent messages up to count
            return self.message_history[-count:]
            
        # Apply filters
        filtered_history = self.message_history.copy()
        for key, value in filter_by.items():
            filtered_history = [msg for msg in filtered_history if msg.get(key) == value]
            
        return filtered_history[-count:]

    async def send_batch_messages(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Send multiple messages in a batch
        
        Args:
            messages: List of message dictionaries, each containing:
                    - recipient: The recipient identifier
                    - content: The message content
                    - topic: Optional topic for the message
            
        Returns:
            List[Dict[str, Any]]: List of send results for each message
        """
        if not self.connected:
            await self.connect()
            
        results = []
        
        for msg_data in messages:
            recipient = msg_data.get("recipient")
            content = msg_data.get("content")
            topic = msg_data.get("topic")
            
            if not recipient or not content:
                results.append({
                    "status": "error",
                    "error": "Missing recipient or content"
                })
                continue
                
            try:
                result = await self.send_message(recipient, content, topic)
                results.append(result)
            except Exception as e:
                results.append({
                    "status": "error",
                    "error": str(e)
                })
                
        return results

    def _add_to_error_history(self, error_type: str, error_message: str):
        """Add an error to the history with size limit enforcement"""
        error_entry = {
            "type": error_type,
            "message": error_message,
            "timestamp": time.time()
        }
        self.error_history.append(error_entry)
        # Trim history if needed
        if len(self.error_history) > self.max_error_history:
            self.error_history = self.error_history[-self.max_error_history:]
            
    async def _monitor_connection(self):
        """Background task for monitoring connection health"""
        while self.connected:
            try:
                # Check if the connection is still alive using a ping
                if self.connection:
                    try:
                        # Try to ping the server directly
                        pong_waiter = await self.connection.ping()
                        await asyncio.wait_for(pong_waiter, timeout=self.config.connection_timeout)
                        logger.debug("Server responded to ping, connection is still alive")
                    except (asyncio.TimeoutError, websockets.exceptions.ConnectionClosed, Exception) as e:
                        logger.warning(f"Connection appears to be closed or unresponsive: {str(e)}")
                        self.connected = False
                        
                        # Force close the connection
                        try:
                            await self.connection.close()
                        except Exception:
                            pass
                            
                        # Trigger disconnect callbacks
                        for callback in self.on_disconnect_callbacks:
                            try:
                                await callback(self.client_id)
                            except Exception as e:
                                logger.error(f"Error in disconnect callback: {str(e)}")
                                
                        asyncio.create_task(self.connect())
                        break
                    
                # Check if we've received a heartbeat recently
                heartbeat_age = time.time() - self.last_heartbeat_received
                if heartbeat_age > self.config.heartbeat_interval * 3:
                    logger.warning(f"No heartbeat received for {heartbeat_age:.1f} seconds")
                    
                    # Try to ping the server directly as a last resort
                    try:
                        pong_waiter = await self.connection.ping()
                        await asyncio.wait_for(pong_waiter, timeout=self.config.connection_timeout)
                        logger.info("Server responded to ping, connection is still alive")
                        self.last_heartbeat_received = time.time()  # Reset heartbeat timer
                    except Exception:
                        logger.error("Server did not respond to ping, reconnecting")
                        self.connected = False
                        
                        # Force close the connection
                        try:
                            await self.connection.close()
                        except Exception:
                            pass
                            
                        # Trigger disconnect callbacks
                        for callback in self.on_disconnect_callbacks:
                            try:
                                await callback(self.client_id)
                            except Exception as e:
                                logger.error(f"Error in disconnect callback: {str(e)}")
                                
                        asyncio.create_task(self.connect())
                        break
                        
                await asyncio.sleep(self.config.heartbeat_interval)
            except Exception as e:
                logger.error(f"Error in connection monitor: {str(e)}")
                self._add_to_error_history("monitor_error", str(e))
                await asyncio.sleep(self.config.reconnect_delay)

    async def register_connection_callback(self, event_type: str, callback: Callable) -> bool:
        """
        Register a callback for connection events
        
        Args:
            event_type: Type of event ('connect', 'disconnect', 'error')
            callback: Async callback function to be called when the event occurs
            
        Returns:
            bool: True if registration is successful, False otherwise
        """
        if event_type == 'connect':
            self.on_connect_callbacks.append(callback)
        elif event_type == 'disconnect':
            self.on_disconnect_callbacks.append(callback)
        elif event_type == 'error':
            self.on_error_callbacks.append(callback)
        else:
            logger.warning(f"Unknown event type: {event_type}")
            return False
            
        return True
        
    async def unregister_connection_callback(self, event_type: str, callback: Callable) -> bool:
        """
        Unregister a connection event callback
        
        Args:
            event_type: Type of event ('connect', 'disconnect', 'error')
            callback: The callback function to unregister
            
        Returns:
            bool: True if unregistration is successful, False otherwise
        """
        if event_type == 'connect' and callback in self.on_connect_callbacks:
            self.on_connect_callbacks.remove(callback)
        elif event_type == 'disconnect' and callback in self.on_disconnect_callbacks:
            self.on_disconnect_callbacks.remove(callback)
        elif event_type == 'error' and callback in self.on_error_callbacks:
            self.on_error_callbacks.remove(callback)
        else:
            return False
            
        return True
        
    async def get_error_history(self, count: int = 10, error_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get the error history
        
        Args:
            count: Maximum number of errors to retrieve
            error_type: Optional filter by error type
            
        Returns:
            List[Dict[str, Any]]: List of error entries
        """
        if not error_type:
            return self.error_history[-count:]
            
        filtered_errors = [e for e in self.error_history if e.get("type") == error_type]
        return filtered_errors[-count:]
