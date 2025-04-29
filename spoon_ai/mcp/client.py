"""
MCP client implementation for managing and communicating with agents.
"""

from typing import Any, Dict, List, Optional, Union, Generator, Callable, AsyncContextManager
import asyncio
import logging
from contextlib import asynccontextmanager
from spoon_ai.agents.base import BaseAgent

# Import FastMCP package
from fastmcp import Client as FastMCPClientBase
from fastmcp.client.transports import PythonStdioTransport, SSETransport

logger = logging.getLogger(__name__)

class MCPClient:
    """
    Client for managing and communicating with MCP agents.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the MCP client.
        
        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        self._agents: Dict[str, BaseAgent] = {}
        self._message_queue: asyncio.Queue = asyncio.Queue()
        
    async def register_agent(self, agent: BaseAgent) -> None:
        """
        Register a new agent with the client.
        
        Args:
            agent: The agent to register
        """
        if agent.name in self._agents:
            raise ValueError(f"Agent with name {agent.name} already registered")
        self._agents[agent.name] = agent
        logger.info(f"Registered agent: {agent.name}")
        
    async def unregister_agent(self, agent_name: str) -> None:
        """
        Unregister an agent from the client.
        
        Args:
            agent_name: Name of the agent to unregister
        """
        if agent_name not in self._agents:
            raise ValueError(f"Agent with name {agent_name} not found")
        del self._agents[agent_name]
        logger.info(f"Unregistered agent: {agent_name}")
        
    async def send_message(
        self, 
        target_agent_name: str, 
        content: Any,
        sender: str = "system",
        metadata: Optional[Dict[str, Any]] = None
    ) -> Union[str, Generator[str, None, None]]:
        """
        Send a message to a specific agent and get its response.
        
        Args:
            target_agent_name: Name of the target agent
            content: Message content to send
            sender: ID of the sender
            metadata: Optional metadata for the message
            
        Returns:
            Response from the agent, either as a string or a generator for streaming
        """
        if target_agent_name not in self._agents:
            raise ValueError(f"Agent with name {target_agent_name} not found")
            
        agent = self._agents[target_agent_name]
        message = {
            "content": content,
            "sender": sender,
            "metadata": metadata or {},
            "topic": "general"
        }
        
        logger.info(f"Sending message to agent {target_agent_name}")
        return await agent.process_mcp_message(
            content=content,
            sender=sender,
            message=message,
            agent_id=target_agent_name
        )
        
    async def broadcast_message(
        self, 
        content: Any,
        sender: str = "system",
        metadata: Optional[Dict[str, Any]] = None
    ) -> List[Union[str, Generator[str, None, None]]]:
        """
        Broadcast a message to all registered agents and collect their responses.
        
        Args:
            content: Message content to broadcast
            sender: ID of the sender
            metadata: Optional metadata for the message
            
        Returns:
            List of responses from all agents
        """
        message = {
            "content": content,
            "sender": sender,
            "metadata": metadata or {},
            "topic": "broadcast"
        }
        
        tasks = []
        for agent_name, agent in self._agents.items():
            tasks.append(
                agent.process_mcp_message(
                    content=content,
                    sender=sender,
                    message=message,
                    agent_id=agent_name
                )
            )
            
        logger.info(f"Broadcasting message to {len(tasks)} agents")
        return await asyncio.gather(*tasks)
        
    def get_agent_state(self, agent_name: str) -> str:
        """
        Get the current state of a specific agent.
        
        Args:
            agent_name: Name of the agent
            
        Returns:
            Current state of the agent
        """
        if agent_name not in self._agents:
            raise ValueError(f"Agent with name {agent_name} not found")
            
        return self._agents[agent_name].state.value
        
    def get_registered_agents(self) -> List[str]:
        """
        Get a list of all registered agent names.
        
        Returns:
            List of agent names
        """
        return list(self._agents.keys())
        
    async def save_agent_chat_history(self, agent_name: str) -> None:
        """
        Save the chat history of a specific agent.
        
        Args:
            agent_name: Name of the agent
        """
        if agent_name not in self._agents:
            raise ValueError(f"Agent with name {agent_name} not found")
            
        agent = self._agents[agent_name]
        agent.save_chat_history()
        logger.info(f"Saved chat history for agent {agent_name}")


class FastMCPClient:
    """
    Client for integrating with MCP servers using FastMCP.
    """
    
    def __init__(
        self,
        server_path: Optional[str] = None,
        server_url: Optional[str] = None,
        transport_type: str = "stdio",
    ):
        """
        Initialize the FastMCP client.
        
        Args:
            server_path: Path to the server script (for stdio transport)
            server_url: URL of the server (for SSE transport)
            transport_type: Type of transport ('stdio' or 'sse')
        """
        self.transport_type = transport_type
        self.server_path = server_path
        self.server_url = server_url
        self.client: Optional[FastMCPClientBase] = None
        self.client_context = None
        self._agent_message_handlers: Dict[str, Callable] = {}
        
        if transport_type not in ["stdio", "sse"]:
            raise ValueError(f"Unsupported transport type: {transport_type}. Choose 'stdio' or 'sse'.")
            
        if transport_type == "stdio" and not server_path:
            raise ValueError("server_path is required for stdio transport")
            
        if transport_type == "sse" and not server_url:
            raise ValueError("server_url is required for sse transport")
        
        # Create transport based on type
        if self.transport_type == "stdio":
            self.transport = PythonStdioTransport(self.server_path)
        else:  # sse
            self.transport = SSETransport(self.server_url)
        
    def register_agent_message_handler(self, agent_name: str, handler: Callable) -> None:
        """
        Register a message handler for a specific agent.
        
        Args:
            agent_name: Name of the agent
            handler: Async function that processes messages for this agent
        """
        self._agent_message_handlers[agent_name] = handler
        logger.info(f"Registered message handler for agent: {agent_name}")
        
    async def list_prompts(self) -> List[Dict[str, Any]]:
        """
        List available prompts from the MCP server.
        
        Returns:
            List of available prompts
        """
        if not self.client:
            raise RuntimeError("Client is not connected. Use 'async with FastMCPClient:' context manager.")
        return await self.client.list_prompts()
        
    async def get_prompt(self, name: str, arguments: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Get a prompt from the MCP server.
        
        Args:
            name: Name of the prompt
            arguments: Optional arguments for the prompt
            
        Returns:
            Prompt result
        """
        if not self.client:
            raise RuntimeError("Client is not connected. Use 'async with FastMCPClient:' context manager.")
        return await self.client.get_prompt(name, arguments or {})
        
    async def list_resources(self) -> List[Dict[str, Any]]:
        """
        List available resources from the MCP server.
        
        Returns:
            List of available resources
        """
        if not self.client:
            raise RuntimeError("Client is not connected. Use 'async with FastMCPClient:' context manager.")
        return await self.client.list_resources()
        
    async def read_resource(self, uri: str) -> str:
        """
        Read a resource from the MCP server.
        
        Args:
            uri: URI of the resource
            
        Returns:
            Content of the resource
        """
        if not self.client:
            raise RuntimeError("Client is not connected. Use 'async with FastMCPClient:' context manager.")
        return await self.client.read_resource(uri)
        
    async def list_tools(self) -> List[Dict[str, Any]]:
        """
        List available tools from the MCP server.
        
        Returns:
            List of available tools
        """
        if not self.client:
            raise RuntimeError("Client is not connected. Use 'async with FastMCPClient:' context manager.")
        return await self.client.list_tools()
        
    async def call_tool(self, name: str, arguments: Optional[Dict[str, Any]] = None) -> Any:
        """
        Call a tool on the MCP server.
        
        Args:
            name: Name of the tool
            arguments: Optional arguments for the tool
            
        Returns:
            Result of the tool call
        """
        if not self.client:
            raise RuntimeError("Client is not connected. Use 'async with FastMCPClient:' context manager.")
        return await self.client.call_tool(name, arguments or {})
        
    async def __aenter__(self) -> "FastMCPClient":
        """
        Async context manager entry.
        """
        logger.info(f"Creating FastMCP client via {self.transport_type}")
        # Create the client with the configured transport
        self.client = FastMCPClientBase(transport=self.transport)
        # Enter the client's context manager
        self.client_context = await self.client.__aenter__()
        logger.info(f"Connected to MCP server via {self.transport_type}")
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """
        Async context manager exit.
        """
        if self.client:
            # Exit the client's context manager
            await self.client.__aexit__(exc_type, exc_val, exc_tb)
            self.client = None
            self.client_context = None
            logger.info("Disconnected from MCP server") 