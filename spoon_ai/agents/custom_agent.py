import logging
from typing import List, Optional

from pydantic import Field

from spoon_ai.agents.toolcall import ToolCallAgent
from spoon_ai.chat import ChatBot
from spoon_ai.schema import AgentState
from spoon_ai.tools import BaseTool, ToolManager, Terminate

logger = logging.getLogger(__name__)

class CustomAgent(ToolCallAgent):
    """
    自定义Agent类，允许用户创建自己的agent并添加自定义工具
    
    使用方法:
    1. 直接实例化并运行预定义的agent:
       agent = CustomAgent.load_predefined("react")
       result = await agent.run("分析这个钱包地址")
       
    2. 创建自定义agent并添加工具:
       agent = CustomAgent(name="my_agent", description="我的自定义agent")
       agent.add_tool(MyCustomTool())
       result = await agent.run("使用我的自定义工具")
    """
    
    name: str = "custom_agent"
    description: str = "可自定义工具的智能agent"
    
    system_prompt: str = """你是一个强大的AI助手，可以使用各种工具来完成任务。
你将遵循以下工作流程:
1. 分析用户的请求
2. 确定需要使用的工具
3. 调用适当的工具
4. 分析工具的输出
5. 提供有用的回应

当你需要使用工具时，请使用提供的工具API。不要假装调用不存在的工具。
"""
    
    next_step_prompt: str = "请思考下一步应该做什么。你可以使用可用的工具，或者直接回答用户的问题。"
    
    max_steps: int = 10
    tool_choice: str = "auto"
    
    avaliable_tools: ToolManager = Field(default_factory=lambda: ToolManager([Terminate()]))
    special_tools: List[str] = Field(default=["terminate"])
    llm: ChatBot = Field(default_factory=lambda: ChatBot())
    
    @classmethod
    def load_predefined(cls, agent_type: str) -> "CustomAgent":
        """
        加载预定义的agent
        
        Args:
            agent_type: agent类型，如"react"、"chat"等
            
        Returns:
            预定义的agent实例
        """
        if agent_type.lower() == "react":
            from spoon_ai.agents import SpoonReactAI
            return SpoonReactAI()
        elif agent_type.lower() == "chat":
            from spoon_ai.agents import SpoonChatAI
            return SpoonChatAI("chat")
        else:
            raise ValueError(f"未知的agent类型: {agent_type}")
    
    def add_tool(self, tool: BaseTool) -> None:
        """
        添加工具到agent
        
        Args:
            tool: 要添加的工具实例
        """
        self.avaliable_tools.add_tool(tool)
        logger.info(f"已添加工具: {tool.name}")
    
    def add_tools(self, tools: List[BaseTool]) -> None:
        """
        批量添加工具到agent
        
        Args:
            tools: 要添加的工具实例列表
        """
        for tool in tools:
            self.add_tool(tool)
    
    def remove_tool(self, tool_name: str) -> None:
        """
        从agent中移除工具
        
        Args:
            tool_name: 要移除的工具名称
        """
        self.avaliable_tools.remove_tool(tool_name)
        logger.info(f"已移除工具: {tool_name}")
    
    def list_tools(self) -> List[str]:
        """
        列出agent中所有可用的工具
        
        Returns:
            工具名称列表
        """
        return [tool.name for tool in self.avaliable_tools.tools]
    
    async def run(self, request: Optional[str] = None) -> str:
        """
        运行agent处理请求
        
        Args:
            request: 用户请求
            
        Returns:
            处理结果
        """
        if self.state != AgentState.IDLE:
            self.clear()
        
        return await super().run(request)
import logging
from typing import Any, Dict, List, Optional, Type, Union

from pydantic import Field

from spoon_ai.agents.base import BaseAgent
from spoon_ai.agents.toolcall import ToolCallAgent
from spoon_ai.chat import ChatBot
from spoon_ai.schema import AgentState
from spoon_ai.tools import BaseTool, ToolManager, Terminate

logger = logging.getLogger(__name__)

class CustomAgent(ToolCallAgent):
    """
    自定义Agent类，允许用户创建自己的agent并添加自定义工具
    
    使用方法:
    1. 直接实例化并运行预定义的agent:
       agent = CustomAgent.load_predefined("react")
       result = await agent.run("分析这个钱包地址")
       
    2. 创建自定义agent并添加工具:
       agent = CustomAgent(name="my_agent", description="我的自定义agent")
       agent.add_tool(MyCustomTool())
       result = await agent.run("使用我的自定义工具")
    """
    
    name: str = "custom_agent"
    description: str = "可自定义工具的智能agent"
    
    system_prompt: str = """你是一个强大的AI助手，可以使用各种工具来完成任务。
你将遵循以下工作流程:
1. 分析用户的请求
2. 确定需要使用的工具
3. 调用适当的工具
4. 分析工具的输出
5. 提供有用的回应

当你需要使用工具时，请使用提供的工具API。不要假装调用不存在的工具。
"""
    
    next_step_prompt: str = "请思考下一步应该做什么。你可以使用可用的工具，或者直接回答用户的问题。"
    
    max_steps: int = 10
    tool_choice: str = "auto"
    
    avaliable_tools: ToolManager = Field(default_factory=lambda: ToolManager([Terminate()]))
    special_tools: List[str] = Field(default=["terminate"])
    llm: ChatBot = Field(default_factory=lambda: ChatBot())
    
    @classmethod
    def load_predefined(cls, agent_type: str) -> "CustomAgent":
        """
        加载预定义的agent
        
        Args:
            agent_type: agent类型，如"react"、"chat"等
            
        Returns:
            预定义的agent实例
        """
        if agent_type.lower() == "react":
            from spoon_ai.agents import SpoonReactAI
            return SpoonReactAI()
        elif agent_type.lower() == "chat":
            from spoon_ai.agents import SpoonChatAI
            return SpoonChatAI("chat")
        else:
            raise ValueError(f"未知的agent类型: {agent_type}")
    
    def add_tool(self, tool: BaseTool) -> None:
        """
        添加工具到agent
        
        Args:
            tool: 要添加的工具实例
        """
        self.avaliable_tools.add_tool(tool)
        logger.info(f"已添加工具: {tool.name}")
    
    def add_tools(self, tools: List[BaseTool]) -> None:
        """
        批量添加工具到agent
        
        Args:
            tools: 要添加的工具实例列表
        """
        for tool in tools:
            self.add_tool(tool)
    
    def remove_tool(self, tool_name: str) -> None:
        """
        从agent中移除工具
        
        Args:
            tool_name: 要移除的工具名称
        """
        self.avaliable_tools.remove_tool(tool_name)
        logger.info(f"已移除工具: {tool_name}")
    
    def list_tools(self) -> List[str]:
        """
        列出agent中所有可用的工具
        
        Returns:
            工具名称列表
        """
        return [tool.name for tool in self.avaliable_tools.tools]
    
    async def run(self, request: Optional[str] = None) -> str:
        """
        运行agent处理请求
        
        Args:
            request: 用户请求
            
        Returns:
            处理结果
        """
        if self.state != AgentState.IDLE:
            self.clear()
        
        return await super().run(request)
