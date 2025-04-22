import logging
import asyncio
import json
import uuid
from typing import Dict, Any, List, Optional, Union, Type, Callable

from pydantic import BaseModel

from .client import SpoonMCPClient, MCPMessage
from .config import MCPConfig
from .exceptions import MCPError

from spoon_ai.agents.base import BaseAgent
from spoon_ai.agents.custom_agent import CustomAgent
from spoon_ai.schema import AgentState, Message, Role

logger = logging.getLogger(__name__)

class MCPAgentAdapter:
    """
    Adapter class for integrating SpoonAI Agents with MCP client.
    Allows Agents to communicate through the MCP protocol.
    """
    
    def __init__(self, mcp_client: Optional[SpoonMCPClient] = None, config: Optional[MCPConfig] = None):
        """
        Initialize the MCP Agent adapter
        
        Args:
            mcp_client: Optional MCP client instance, if not provided a new one will be created
            config: MCP configuration, used when mcp_client is not provided
        """
        self.mcp_client = mcp_client or SpoonMCPClient(config)
        self.agent_registry: Dict[str, BaseAgent] = {}
        self.agent_topics: Dict[str, List[str]] = {}
        self.agent_tasks: Dict[str, asyncio.Task] = {}
        self.agent_message_queues: Dict[str, asyncio.Queue] = {}
        
    async def connect(self) -> bool:
        """
        Connect to the MCP server
        
        Returns:
            bool: Whether the connection was successful
        """
        if not self.mcp_client.connected:
            return await self.mcp_client.connect()
        return True
        
    async def disconnect(self) -> bool:
        """
        Disconnect from the MCP server
        
        Returns:
            bool: Whether the disconnection was successful
        """
        for agent_id in list(self.agent_registry.keys()):
            await self.stop_agent(agent_id)
            
        return await self.mcp_client.disconnect()
        
    async def register_agent(self, agent: BaseAgent, agent_id: Optional[str] = None) -> str:
        """
        Register an Agent with the MCP client
        
        Args:
            agent: The Agent instance to register
            agent_id: Optional Agent ID, if not provided a UUID will be used
            
        Returns:
            str: The registered Agent ID
        """
        if not agent_id:
            agent_id = f"{agent.name}_{str(uuid.uuid4())[:8]}"
            
        if agent_id in self.agent_registry:
            logger.warning(f"Agent ID '{agent_id}' already exists, will be overwritten")
            if agent_id in self.agent_tasks and not self.agent_tasks[agent_id].done():
                await self.stop_agent(agent_id)
        
        self.agent_registry[agent_id] = agent
        self.agent_topics[agent_id] = []
        self.agent_message_queues[agent_id] = asyncio.Queue()
        
        await self.mcp_client.register_agent(agent_id, agent)
        
        setattr(agent, "_mcp_adapter", self)
        setattr(agent, "_agent_id", agent_id)
        
        logger.info(f"Agent '{agent_id}' successfully registered with MCP adapter")
        
        return agent_id
        
    async def unregister_agent(self, agent_id: str) -> bool:
        """
        Unregister an Agent from the MCP client
        
        Args:
            agent_id: The Agent ID to unregister
            
        Returns:
            bool: Whether the unregistration was successful
        """
        if agent_id not in self.agent_registry:
            logger.warning(f"Agent ID '{agent_id}' does not exist")
            return False
            
        if agent_id in self.agent_tasks and not self.agent_tasks[agent_id].done():
            await self.stop_agent(agent_id)
            
        success = await self.mcp_client.unregister_agent(agent_id)
        
        if success:
            agent = self.agent_registry[agent_id]
            if hasattr(agent, "_mcp_adapter"):
                delattr(agent, "_mcp_adapter")
            if hasattr(agent, "_agent_id"):
                delattr(agent, "_agent_id")
                
            del self.agent_registry[agent_id]
            del self.agent_topics[agent_id]
            del self.agent_message_queues[agent_id]
            logger.info(f"Agent '{agent_id}' successfully unregistered from MCP adapter")
            
        return success
        
    async def agent_subscribe(self, agent_id: str, topic: str) -> bool:
        """
        Subscribe an Agent to a topic
        
        Args:
            agent_id: Agent ID
            topic: The topic to subscribe to
            
        Returns:
            bool: Whether the subscription was successful
        """
        if agent_id not in self.agent_registry:
            logger.error(f"Agent ID '{agent_id}' does not exist")
            raise MCPError(f"Agent ID '{agent_id}' does not exist")
            
        success = await self.mcp_client.agent_subscribe(agent_id, topic)
        
        if success and topic not in self.agent_topics[agent_id]:
            self.agent_topics[agent_id].append(topic)
            logger.info(f"Agent '{agent_id}' successfully subscribed to topic '{topic}'")
            
        return success
        
    async def agent_unsubscribe(self, agent_id: str, topic: str) -> bool:
        """
        Unsubscribe an Agent from a topic
        
        Args:
            agent_id: Agent ID
            topic: The topic to unsubscribe from
            
        Returns:
            bool: Whether the unsubscription was successful
        """
        if agent_id not in self.agent_registry:
            logger.error(f"Agent ID '{agent_id}' does not exist")
            raise MCPError(f"Agent ID '{agent_id}' does not exist")
            
        success = await self.mcp_client.agent_unsubscribe(agent_id, topic)
        
        if success and topic in self.agent_topics[agent_id]:
            self.agent_topics[agent_id].remove(topic)
            logger.info(f"Agent '{agent_id}' successfully unsubscribed from topic '{topic}'")
            
        return success
        
    async def _process_agent_message_queue(self, agent_id: str):
        """
        Process an Agent's message queue
        
        Args:
            agent_id: Agent ID
        """
        agent = self.agent_registry[agent_id]
        queue = self.agent_message_queues[agent_id]
        
        try:
            while True:
                message = await queue.get()
                queue.task_done()
                
                content = message.get("content", "")
                sender = message.get("sender", "unknown")
                topic = message.get("topic", "general")
                
                logger.info(f"Agent '{agent_id}' processing message from '{sender}', topic: '{topic}'")
                
                if isinstance(content, dict) and "text" in content:
                    text_content = content["text"]
                elif isinstance(content, str):
                    text_content = content
                else:
                    text_content = str(content)
                
                agent.add_message("user", text_content)
                
                if hasattr(agent, "stream") and callable(agent.stream):
                    try:
                        stream_metadata = {
                            "agent_id": agent_id,
                            "agent_name": agent.name,
                            "topic": topic,
                            "is_streaming": True,
                            "stream_id": str(uuid.uuid4())
                        }
                        
                        while not agent.output_queue.empty():
                            try:
                                agent.output_queue.get_nowait()
                                agent.output_queue.task_done()
                            except asyncio.QueueEmpty:
                                break
                                
                        agent.task_done = asyncio.Event()
                        
                        asyncio.create_task(self._run_agent_and_set_done(agent))
                        
                        buffer = []
                        buffer_size = 10
                        
                        async for token in agent.stream():
                            buffer.append(token)
                            
                            if len(buffer) >= buffer_size:
                                chunk = "".join(buffer)
                                await self._send_stream_chunk(
                                    agent_id=agent_id,
                                    recipient=sender,
                                    chunk=chunk,
                                    topic=topic,
                                    metadata=stream_metadata,
                                    is_final=False
                                )
                                buffer = []
                        
                        if buffer:
                            final_chunk = "".join(buffer)
                            await self._send_stream_chunk(
                                agent_id=agent_id,
                                recipient=sender,
                                chunk=final_chunk,
                                topic=topic,
                                metadata=stream_metadata,
                                is_final=True
                            )
                            
                        logger.info(f"Agent '{agent_id}' completed streaming response to '{sender}'")
                        
                    except Exception as e:
                        logger.error(f"Agent '{agent_id}' error during stream processing: {str(e)}")
                        error_response = {
                            "text": f"Stream processing error: {str(e)}",
                            "status": "error",
                            "metadata": {
                                "agent_id": agent_id,
                                "error": str(e)
                            }
                        }
                        
                        await self.mcp_client.agent_send_message(
                            agent_id=agent_id, 
                            recipient=sender, 
                            message=error_response,
                            topic=topic
                        )
                else:
                    try:
                        result = await agent.run()
                        
                        response = {
                            "text": result,
                            "source": "agent",
                            "metadata": {
                                "agent_id": agent_id,
                                "agent_name": agent.name,
                                "topic": topic
                            }
                        }
                        
                        await self.mcp_client.agent_send_message(
                            agent_id=agent_id,
                            recipient=sender,
                            message=response,
                            topic=topic
                        )
                        
                        logger.info(f"Agent '{agent_id}' sent response to '{sender}'")
                        
                    except Exception as e:
                        logger.error(f"Agent '{agent_id}' error processing message: {str(e)}")
                        error_response = {
                            "text": f"Error processing message: {str(e)}",
                            "status": "error",
                            "metadata": {
                                "agent_id": agent_id,
                                "error": str(e)
                            }
                        }
                        
                        await self.mcp_client.agent_send_message(
                            agent_id=agent_id, 
                            recipient=sender, 
                            message=error_response,
                            topic=topic
                        )
        
        except asyncio.CancelledError:
            logger.info(f"Agent '{agent_id}' message processing task cancelled")
        except Exception as e:
            logger.error(f"Agent '{agent_id}' message processing loop error: {str(e)}")
            
    async def _run_agent_and_set_done(self, agent: BaseAgent):
        """
        Run the Agent and set the done event when complete
        
        Args:
            agent: Agent instance
        """
        try:
            await agent.run()
        except Exception as e:
            logger.error(f"Agent {agent.name} run error: {str(e)}")
        finally:
            agent.task_done.set()
            
    async def _send_stream_chunk(self, agent_id: str, recipient: str, chunk: str, 
                               topic: str, metadata: Dict[str, Any], is_final: bool):
        """
        Send a stream chunk as part of a streaming response
        
        Args:
            agent_id: Agent ID
            recipient: Recipient ID
            chunk: Text chunk to send
            topic: Message topic
            metadata: Metadata
            is_final: Whether this is the final chunk
        """
        chunk_metadata = metadata.copy()
        chunk_metadata["is_final"] = is_final
        
        content = {
            "text": chunk,
            "source": "agent",
            "type": "stream_chunk",
            "metadata": chunk_metadata
        }
        
        await self.mcp_client.agent_send_message(
            agent_id=agent_id,
            recipient=recipient,
            message=content,
            topic=topic
        )
        
    async def start_agent(self, agent_id: str, topics: Optional[List[str]] = None) -> bool:
        """
        Start an Agent to listen for and process messages
        
        Args:
            agent_id: Agent ID
            topics: Optional list of topics to subscribe to, if empty will use existing subscriptions
            
        Returns:
            bool: Whether the start was successful
        """
        if agent_id not in self.agent_registry:
            logger.error(f"Agent ID '{agent_id}' does not exist")
            raise MCPError(f"Agent ID '{agent_id}' does not exist")
            
        agent = self.agent_registry[agent_id]
        
        if agent_id in self.agent_tasks and not self.agent_tasks[agent_id].done():
            logger.warning(f"Agent '{agent_id}' is already running, will restart")
            await self.stop_agent(agent_id)
            
        if not self.mcp_client.connected:
            await self.connect()
            
        if topics:
            for old_topic in self.agent_topics[agent_id][:]:
                await self.agent_unsubscribe(agent_id, old_topic)
                
            for topic in topics:
                await self.agent_subscribe(agent_id, topic)
        elif not self.agent_topics[agent_id]:
            default_topic = agent.name.lower()
            await self.agent_subscribe(agent_id, default_topic)
            
        task = asyncio.create_task(self._process_agent_message_queue(agent_id))
        self.agent_tasks[agent_id] = task
        
        logger.info(f"Agent '{agent_id}' started, listening to topics: {self.agent_topics[agent_id]}")
        
        return True
        
    async def stop_agent(self, agent_id: str) -> bool:
        """
        Stop a running Agent
        
        Args:
            agent_id: Agent ID
            
        Returns:
            bool: Whether the stop was successful
        """
        if agent_id not in self.agent_registry:
            logger.error(f"Agent ID '{agent_id}' does not exist")
            raise MCPError(f"Agent ID '{agent_id}' does not exist")
            
        if agent_id not in self.agent_tasks or self.agent_tasks[agent_id].done():
            logger.warning(f"Agent '{agent_id}' is not running")
            return False
            
        task = self.agent_tasks[agent_id]
        task.cancel()
        
        try:
            await task
        except asyncio.CancelledError:
            logger.info(f"Agent '{agent_id}' task cancelled")
        except Exception as e:
            logger.error(f"Error stopping Agent '{agent_id}': {str(e)}")
            
        logger.info(f"Agent '{agent_id}' stopped")
        return True
        
    async def get_agent_status(self, agent_id: str) -> Dict[str, Any]:
        """
        Get the status of an Agent
        
        Args:
            agent_id: Agent ID
            
        Returns:
            Dict[str, Any]: Agent status information
        """
        if agent_id not in self.agent_registry:
            logger.error(f"Agent ID '{agent_id}' does not exist")
            raise MCPError(f"Agent ID '{agent_id}' does not exist")
            
        agent = self.agent_registry[agent_id]
        
        is_running = (
            agent_id in self.agent_tasks and 
            not self.agent_tasks[agent_id].done()
        )
        
        queue_size = self.agent_message_queues[agent_id].qsize() if agent_id in self.agent_message_queues else 0
        
        return {
            "agent_id": agent_id,
            "agent_name": agent.name,
            "agent_type": agent.__class__.__name__,
            "description": getattr(agent, "description", None),
            "is_running": is_running,
            "state": str(agent.state),
            "subscribed_topics": self.agent_topics.get(agent_id, []),
            "message_queue_size": queue_size,
            "current_step": agent.current_step,
            "max_steps": agent.max_steps
        }
        
    async def create_custom_agent(self, name: str, description: Optional[str] = None, system_prompt: Optional[str] = None) -> str:
        """
        Create a custom Agent and register it with the MCP adapter
        
        Args:
            name: Agent name
            description: Agent description
            system_prompt: System prompt
            
        Returns:
            str: The registered Agent ID
        """
        agent = CustomAgent(
            name=name,
            description=description or f"{name} Agent",
            system_prompt=system_prompt
        )
        
        agent_id = await self.register_agent(agent)
        return agent_id
        
    async def load_and_register_predefined_agent(self, agent_type: str, agent_id: Optional[str] = None) -> str:
        """
        Load a predefined Agent and register it with the MCP adapter
        
        Args:
            agent_type: Agent type, like "react", "chat", etc.
            agent_id: Optional Agent ID
            
        Returns:
            str: The registered Agent ID
        """
        agent = CustomAgent.load_predefined(agent_type)
        return await self.register_agent(agent, agent_id)
        
    async def process_agent_mcp_message(self, content: Any, sender: str, message: Dict[str, Any], agent_id: str):
        """
        Process a MCP message sent to an Agent
        
        Args:
            content: Message content
            sender: Sender ID
            message: Complete message
            agent_id: Agent ID
        """
        if agent_id not in self.agent_registry:
            logger.error(f"Agent ID '{agent_id}' does not exist")
            return
            
        await self.agent_message_queues[agent_id].put(message)
        logger.debug(f"Message added to Agent '{agent_id}' queue from '{sender}'")
        
    async def send_message_to_agent(self, agent_id: str, message: str, sender_id: str = "user", topic: Optional[str] = None, stream: bool = False) -> bool:
        """
        Send a message to a specific Agent
        
        Args:
            agent_id: Target Agent ID
            message: Message content
            sender_id: Sender ID
            topic: Optional topic
            stream: Whether to request streaming response
            
        Returns:
            bool: Whether the send was successful
        """
        if agent_id not in self.agent_registry:
            logger.error(f"Agent ID '{agent_id}' does not exist")
            return False
            
        if not topic and self.agent_topics[agent_id]:
            topic = self.agent_topics[agent_id][0]
        elif not topic:
            topic = self.agent_registry[agent_id].name.lower()
            
        message_content = {
            "text": message,
            "source": "external",
            "metadata": {
                "sender_id": sender_id,
                "request_stream": stream
            }
        }
        
        message_data = {
            "id": str(uuid.uuid4()),
            "sender": sender_id,
            "recipient": agent_id,
            "content": message_content,
            "timestamp": asyncio.get_event_loop().time(),
            "topic": topic
        }
        
        await self.agent_message_queues[agent_id].put(message_data)
        logger.info(f"Message sent directly to Agent '{agent_id}', stream request: {stream}")
        
        return True
        
    async def broadcast_to_agents(self, message: str, topics: List[str], sender_id: str = "system") -> Dict[str, bool]:
        """
        Broadcast a message to topics, to be received by all subscribed Agents
        
        Args:
            message: Message content
            topics: List of topics
            sender_id: Sender ID
            
        Returns:
            Dict[str, bool]: Result for each Agent
        """
        if not self.mcp_client.connected:
            await self.connect()
            
        results = {}
        
        for topic in topics:
            for agent_id, subscribed_topics in self.agent_topics.items():
                if topic in subscribed_topics:
                    try:
                        result = await self.send_message_to_agent(
                            agent_id=agent_id,
                            message=message,
                            sender_id=sender_id,
                            topic=topic
                        )
                        results[agent_id] = result
                    except Exception as e:
                        logger.error(f"Error broadcasting to Agent '{agent_id}': {str(e)}")
                        results[agent_id] = False
                        
        return results 