import logging
from typing import List, Optional, Dict, Any, Union
import asyncio

from pydantic import Field

from spoon_ai.agents.toolcall import ToolCallAgent
from spoon_ai.chat import ChatBot
from spoon_ai.schema import AgentState
from spoon_ai.tools import BaseTool, ToolManager, Terminate

logger = logging.getLogger(__name__)

class CustomAgent(ToolCallAgent):
    """
    Custom Agent class allowing users to create their own agents and add custom tools
    
    Usage:
    1. Directly instantiate and run predefined agents:
       agent = CustomAgent.load_predefined("react")
       result = await agent.run("Analyze this wallet address")
       
    2. Create custom agent and add tools:
       agent = CustomAgent(name="my_agent", description="My custom agent")
       agent.add_tool(MyCustomTool())
       result = await agent.run("Use my custom tool")
    """
    
    name: str = "custom_agent"
    description: str = "Intelligent agent with customizable tools"
    
    system_prompt: str = """You are a powerful AI assistant that can use various tools to complete tasks.
You will follow this workflow:
1. Analyze the user's request
2. Determine which tools to use
3. Call the appropriate tools
4. Analyze the tool output
5. Provide a useful response

When you need to use tools, please use the provided tool API. Don't pretend to call non-existent tools.
"""
    
    next_step_prompt: str = "Please think about what to do next. You can use available tools or directly answer the user's question."
    
    max_steps: int = 10
    tool_choice: str = "auto"
    
    avaliable_tools: ToolManager = Field(default_factory=lambda: ToolManager([Terminate()]))
    special_tools: List[str] = Field(default=["terminate"])
    llm: ChatBot = Field(default_factory=lambda: ChatBot())
    
    # MCP integration configuration
    output_topic: Optional[str] = None
    mcp_enabled: bool = True
    
    @classmethod
    def load_predefined(cls, agent_type: str) -> "CustomAgent":
        """
        Load a predefined agent
        
        Args:
            agent_type: agent type, such as "react", "chat", etc.
            
        Returns:
            Predefined agent instance
        """
        if agent_type.lower() == "react":
            from spoon_ai.agents import SpoonReactAI
            return SpoonReactAI()
        elif agent_type.lower() == "chat":
            from spoon_ai.agents import SpoonChatAI
            return SpoonChatAI("chat")
        else:
            raise ValueError(f"Unknown agent type: {agent_type}")
    
    def add_tool(self, tool: BaseTool) -> None:
        """
        Add a tool to the agent
        
        Args:
            tool: Tool instance to add
        """
        self.avaliable_tools.add_tool(tool)
        logger.info(f"Added tool: {tool.name}")
    
    def add_tools(self, tools: List[BaseTool]) -> None:
        """
        Add multiple tools to the agent
        
        Args:
            tools: List of tool instances to add
        """
        for tool in tools:
            self.add_tool(tool)
    
    def remove_tool(self, tool_name: str) -> None:
        """
        Remove a tool from the agent
        
        Args:
            tool_name: Name of the tool to remove
        """
        self.avaliable_tools.remove_tool(tool_name)
        logger.info(f"Removed tool: {tool_name}")
    
    def list_tools(self) -> List[str]:
        """
        List all available tools in the agent
        
        Returns:
            List of tool names
        """
        return [tool.name for tool in self.avaliable_tools.tools]
    
    async def run(self, request: Optional[str] = None) -> str:
        """
        Run the agent to process a request
        
        Args:
            request: User request
            
        Returns:
            Processing result
        """
        if self.state != AgentState.IDLE:
            self.clear()
        
        return await super().run(request)
    
    async def process_mcp_message(self, content: Any, sender: str, message: Dict[str, Any], agent_id: str):
        """
        Process messages from the MCP system
        
        Args:
            content: Message content
            sender: Sender ID
            message: Complete message
            agent_id: Agent ID
            
        Returns:
            Processing result
        """
        if isinstance(content, dict) and "text" in content:
            text_content = content["text"]
        elif isinstance(content, str):
            text_content = content
        else:
            text_content = str(content)
            
        metadata = {}
        if isinstance(content, dict) and "metadata" in content:
            metadata = content.get("metadata", {})
            
        topic = message.get("topic", "general")
        if not self.output_topic:
            self.output_topic = topic
            
        self._last_sender = sender
        self._last_topic = topic
        self._last_message_id = message.get("id")
            
        message_type = None
        if isinstance(content, dict) and "type" in content:
            message_type = content.get("type")
            
        request_stream = False
        if isinstance(content, dict) and "metadata" in content:
            request_stream = content.get("metadata", {}).get("request_stream", False)
            
        self.add_message("user", text_content)
        
        if request_stream and hasattr(self, "stream") and callable(self.stream):
            while not self.output_queue.empty():
                try:
                    self.output_queue.get_nowait()
                    self.output_queue.task_done()
                except:
                    pass
            self.task_done = asyncio.Event()
            return "STREAMING"
            
        if message_type == "command":
            command = content.get("command", "")
            args = content.get("args", {})
            
            if command == "clear_memory":
                self.memory.clear()
                return "Memory cleared"
            elif command == "add_tool":
                tool_name = args.get("tool_name")
                if tool_name and hasattr(self, "add_tool_by_name"):
                    try:
                        self.add_tool_by_name(tool_name)
                        return f"Tool {tool_name} added"
                    except Exception as e:
                        return f"Failed to add tool {tool_name}: {str(e)}"
                return "Invalid tool name"
            elif command == "set_system_prompt":
                new_prompt = args.get("prompt")
                if new_prompt:
                    self.system_prompt = new_prompt
                    return "System prompt updated"
                return "Invalid system prompt"
            else:
                return await self.run()
        else:
            return await self.run()
            
    async def reply_to_mcp(self, message: str, topic: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Reply to an MCP message
        
        Args:
            message: Reply content
            topic: Reply topic, defaults to the topic of the last received message
            metadata: Additional metadata
            
        Returns:
            bool: Whether the reply was successful
        """
        if not hasattr(self, "_last_sender"):
            logger.warning(f"Agent {self.name} has no message to reply to")
            return False
            
        recipient = getattr(self, "_last_sender", "unknown")
        reply_topic = topic or getattr(self, "_last_topic", "general")
        
        reply_metadata = {
            "agent_name": self.name,
            "reply_to": getattr(self, "_last_message_id", None)
        }
        
        if metadata:
            reply_metadata.update(metadata)
            
        content = {
            "text": message,
            "source": "agent",
            "metadata": reply_metadata
        }
        
        if hasattr(self, "_mcp_adapter") and hasattr(self, "_agent_id"):
            adapter = getattr(self, "_mcp_adapter")
            agent_id = getattr(self, "_agent_id")
            
            try:
                await adapter.send_message_to_agent(
                    agent_id=agent_id,
                    message=message,
                    sender_id=recipient,
                    topic=reply_topic
                )
                return True
            except Exception as e:
                logger.error(f"Failed to reply via MCPAgentAdapter: {str(e)}")
                return False
                
        logger.warning(f"Agent {self.name} not connected to MCPAgentAdapter, cannot reply to message")
        return False
        
    async def reply_to_mcp_stream(self, token: str, is_final: bool = False, topic: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Stream reply to an MCP message
        
        Args:
            token: Token to send
            is_final: Whether this is the final token
            topic: Reply topic, defaults to the topic of the last received message
            metadata: Additional metadata
            
        Returns:
            bool: Whether the stream reply was successful
        """
        if not hasattr(self, "_last_sender"):
            logger.warning(f"Agent {self.name} has no message to reply to")
            return False
            
        recipient = getattr(self, "_last_sender", "unknown")
        reply_topic = topic or getattr(self, "_last_topic", "general")
        
        reply_metadata = {
            "agent_name": self.name,
            "reply_to": getattr(self, "_last_message_id", None),
            "is_streaming": True,
            "is_final": is_final
        }
        
        if metadata:
            reply_metadata.update(metadata)
            
        content = {
            "text": token,
            "source": "agent",
            "type": "stream_chunk",
            "metadata": reply_metadata
        }
        
        if hasattr(self, "_mcp_adapter") and hasattr(self, "_agent_id"):
            adapter = getattr(self, "_mcp_adapter")
            agent_id = getattr(self, "_agent_id")
            
            try:
                await adapter.mcp_client.agent_send_message(
                    agent_id=agent_id,
                    recipient=recipient,
                    message=content,
                    topic=reply_topic
                )
                return True
            except Exception as e:
                logger.error(f"Failed to stream reply via MCPAgentAdapter: {str(e)}")
                return False
                
        logger.warning(f"Agent {self.name} not connected to MCPAgentAdapter, cannot stream reply to message")
        return False
        
    def clear(self):
        """Clear the Agent's state and memory"""
        super().clear()
        self.current_step = 0
        self.state = AgentState.IDLE
        
        if hasattr(self, "_last_sender"):
            delattr(self, "_last_sender")
        if hasattr(self, "_last_topic"):
            delattr(self, "_last_topic")
        if hasattr(self, "_last_message_id"):
            delattr(self, "_last_message_id")
