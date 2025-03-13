from typing import List

from pydantic import Field

from spoon_ai.tools import Terminate, ToolManager

from .toolcall import ToolCallAgent


class SpoonReactAI(ToolCallAgent):
    
    name: str = "spoon_react"
    description: str = "A smart ai agent in neo blockchain"
    
    system_prompt: str = ""
    next_action_prompt: str = ""
    
    max_steps: int = 10
    
    avaliable_tools: ToolManager = Field(default_factory=lambda: ToolManager(Terminate()))
    special_tools: List[str] = Field(default=["terminate"])

        