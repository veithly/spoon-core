"""
Adapter module to bridge SpoonAI agents with FastMCP.
"""

from typing import Any, Dict, List, Optional, Union, Callable
import asyncio
import logging
import json
from spoon_ai.agents.base import BaseAgent

logger = logging.getLogger(__name__)

class MCPAgentAdapter:
    """
    Adapter to bridge SpoonAI BaseAgent with the FastMCP protocol.
    """
    
    def __init__(self, agent: BaseAgent):
        """
        Initialize the MCP agent adapter.
        
        Args:
            agent: The SpoonAI agent to adapt
        """
        self.agent = agent
        
    async def handle_mcp_message(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle an MCP message and convert it to a format compatible with SpoonAI agents.
        
        Args:
            message: The MCP message
            
        Returns:
            The MCP message result
        """
        # Extract content from the message
        role = message.get("role", "user")
        content = message.get("content", "")
        metadata = message.get("metadata", {})
        
        # Convert MCP message to SpoonAI format
        agent_message = {
            "content": content,
            "sender": role,
            "metadata": metadata,
            "topic": "general"
        }
        
        # Process message with the agent
        response = await self.agent.process_mcp_message(
            content=content,
            sender=role,
            message=agent_message,
            agent_id=self.agent.name
        )
        
        # Handle both streaming and non-streaming responses
        if asyncio.iscoroutine(response) or hasattr(response, "__aiter__"):
            # For streaming responses, collect all chunks
            chunks = []
            async for chunk in response:
                chunks.append(chunk)
            response_text = "".join(chunks)
        else:
            # For non-streaming responses
            response_text = str(response)
        
        # Convert SpoonAI response to MCP format
        return {
            "role": "assistant",
            "content": response_text,
            "model": metadata.get("model", "default"),
        }
        
    def create_tool_handler(self, tool_name: str) -> Callable:
        """
        Create a handler function for an MCP tool that delegates to agent methods.
        
        Args:
            tool_name: Name of the tool
            
        Returns:
            Handler function for the tool
        """
        async def tool_handler(**arguments) -> Any:
            method_name = f"handle_{tool_name.replace('-', '_')}"
            if hasattr(self.agent, method_name) and callable(getattr(self.agent, method_name)):
                method = getattr(self.agent, method_name)
                return await method(**arguments)
            else:
                error_msg = f"Agent {self.agent.name} does not implement handler for {tool_name}"
                logger.error(error_msg)
                return {"error": error_msg}
                
        return tool_handler
        
    def create_resource_handler(self, resource_pattern: str) -> Callable:
        """
        Create a handler function for an MCP resource that delegates to agent methods.
        
        Args:
            resource_pattern: Pattern of the resource
            
        Returns:
            Handler function for the resource
        """
        async def resource_handler(**uri_parts) -> str:
            # Extract method name from resource pattern
            parts = resource_pattern.split("://")
            if len(parts) != 2:
                return "Invalid resource pattern"
                
            resource_type = parts[0]
            method_name = f"get_resource_{resource_type}"
            
            if hasattr(self.agent, method_name) and callable(getattr(self.agent, method_name)):
                method = getattr(self.agent, method_name)
                content = await method(**uri_parts)
                
                # Convert dict/list to JSON string if needed
                if isinstance(content, (dict, list)):
                    return json.dumps(content)
                    
                return str(content)
            else:
                error_msg = f"Agent {self.agent.name} does not implement handler for resource {resource_pattern}"
                logger.error(error_msg)
                return error_msg
                
        return resource_handler 