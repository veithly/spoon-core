from typing import List

from pydantic import Field

from spoon_ai.tools import (PredictPrice, Terminate, TokenHolders, ToolManager,
                            TradingHistory, UniswapLiquidity, WalletAnalysis)
from spoon_ai.prompts.spoon_react import SYSTEM_PROMPT, NEXT_STEP_PROMPT

from .toolcall import ToolCallAgent
from spoon_ai.chat import ChatBot

class SpoonReactAI(ToolCallAgent):
    
    name: str = "spoon_react"
    description: str = "A smart ai agent in neo blockchain"
    
    system_prompt: str = SYSTEM_PROMPT
    next_step_prompt: str = NEXT_STEP_PROMPT
    
    max_steps: int = 10
    tool_choice: str = "auto"
    
    avaliable_tools: ToolManager = Field(default_factory=lambda: ToolManager([Terminate(), PredictPrice(), TokenHolders(), TradingHistory(), UniswapLiquidity(), WalletAnalysis()]))
    special_tools: List[str] = Field(default=["terminate"])
    llm: ChatBot = Field(default_factory=lambda: ChatBot())
