import asyncio
import logging
import os
from typing import Any, Dict, List, Optional

from pydantic import Field, PrivateAttr

from spoon_ai.agents.toolcall import ToolCallAgent
from spoon_ai.chat import ChatBot
from spoon_ai.memory.mem0_client import SpoonMem0
from spoon_ai.schema import Message
from spoon_ai.tools.tool_manager import ToolManager
from spoon_ai.tools.base import BaseTool

logger = logging.getLogger(__name__)

class Mem0Agent(ToolCallAgent):
    """ToolCallAgent extension that adds Mem0 recall/store without changing core SpoonOS."""

    mem0_config: Dict[str, Any] = Field(default_factory=dict)

    _mem0_client: Optional[SpoonMem0] = PrivateAttr(default=None)
    _mem0_user_id: Optional[str] = PrivateAttr(default=None)

    def model_post_init(self, __context: Any = None) -> None:
        super().model_post_init(__context)
        self._mem0_user_id = self.mem0_config.get("user_id") or self.mem0_config.get("agent_id")
        self._mem0_client = SpoonMem0(self.mem0_config)
        
        # Ensure available_tools is a ToolManager of proper tool objects
        tools_val = getattr(self, "available_tools", None)
        if isinstance(tools_val, list):
            safe_tools = [t for t in tools_val if isinstance(t, BaseTool)]
            self.available_tools = ToolManager(safe_tools)
        elif not isinstance(tools_val, ToolManager):
            self.available_tools = ToolManager([])

    async def _inject_memories(self, request: Optional[str]) -> None:
        memories: List[str] = self._mem0_client.search_memory(
            request, user_id=self._mem0_user_id
        )
        context = "\n".join(f"- {m}" for m in memories)
        # add_message only supports user/assistant/tool; add directly
        self.memory.add_message(Message(role="system", content=f"Relevant User History:\n{context}"))

    async def _store_interaction(self, request: Optional[str], result: Optional[str]) -> None:

        payload = []
        if request:
            payload.append({"role": "user", "content": request})
        if result:
            payload.append({"role": "assistant", "content": result})
        if not payload:
            return
        self._mem0_client.add_memory(payload, user_id=self._mem0_user_id)


    async def run(self, request: Optional[str] = None) -> str:
        await self._inject_memories(request)
        result = await super().run(request)
        await self._store_interaction(request, result)
        return result


async def main() -> None:
    mem0_cfg = {"user_id": "defi_user_001", "async_mode": False}

    def build_agent() -> Mem0Agent:
        llm = ChatBot(
            llm_provider="openrouter",
            base_url="https://openrouter.ai/api/v1",
            model_name="anthropic/claude-3.5-sonnet",
            enable_long_term_memory=False,  # we manage long-term memory here
        )
        return Mem0Agent(
            llm=llm,
            available_tools=ToolManager([]),  # pass an explicit ToolManager to avoid stray entries
            mem0_config=mem0_cfg,
            system_prompt=(
                "You are a DeFi investment advisor. Always use user_id 'defi_user_001' "
                "for memory operations. Recall past preferences before advising."
            ),
        )

    print("=== Session 1: Teach preferences and store in Mem0 ===")
    agent1 = build_agent()
    teach_msg = "I only invest in Arbitrum USDC pools."
    reply1 = await agent1.run(teach_msg)
    print("Agent reply:", reply1, "\n")

    print("=== Session 2: Recall after restart ===")
    agent2 = build_agent()  # simulate restart; new agent but same mem0_config/user
    recall_msg = "Recommend a strategy for my new capital."
    reply2 = await agent2.run(recall_msg)
    print("Agent reply:", reply2)


if __name__ == "__main__":
    asyncio.run(main())
