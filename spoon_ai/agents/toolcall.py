from typing import List

from pydantic import Field

from spoon_ai.agents.react import ReActAgent
from spoon_ai.prompts.toolcall import \
    NEXT_STEP_PROMPT as TOOLCALL_NEXT_STEP_PROMPT
from spoon_ai.prompts.toolcall import SYSTEM_PROMPT as TOOLCALL_SYSTEM_PROMPT
from spoon_ai.schema import TOOL_CHOICE_TYPE, ToolCall, ToolChoice
from spoon_ai.tools.tool_manager import ToolManager


class ToolCallAgent(ReActAgent):
    
    name: str = "toolcall"
    description: str = "Useful when you need to call a tool"
    
    system_prompt: str = TOOLCALL_SYSTEM_PROMPT
    next_step_prompt: str = TOOLCALL_NEXT_STEP_PROMPT
    
    avaliable_tools: ToolManager = Field(default_factory=ToolManager)
    
    tool_choices: TOOL_CHOICE_TYPE = ToolChoice.AUTO # type: ignore
    
    tool_calls: List[ToolCall] = Field(default_factory=list)
    
    async def think(self) -> bool:
        if self.next_step_prompt:
            self.add_message("user", self.next_step_prompt)
        
        response = await self.llm.ask_tool(
            
        )
    