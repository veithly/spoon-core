"""
Mem0 toolkit demo using SpoonReactAI.
This demo requires spoon-toolkit to be installed (pip install spoon-toolkit or local install).
"""

import asyncio
from typing import Dict, List, Optional

from pydantic import Field

from spoon_ai.agents.spoon_react import SpoonReactAI
from spoon_ai.chat import ChatBot
from spoon_ai.llm.manager import get_llm_manager
from spoon_ai.memory.utils import extract_memories, extract_first_memory_id
from spoon_ai.tools.tool_manager import ToolManager
from spoon_ai.tools.base import ToolResult
from spoon_toolkits.memory import (
    AddMemoryTool,
    SearchMemoryTool,
    GetAllMemoryTool,
    UpdateMemoryTool,
    DeleteMemoryTool,
)


USER_ID = "defi_user_005"


class DeFiMemoryAgent(SpoonReactAI):
    """Brain from spoon-core + memory tools from spoon-toolkit."""

    mem0_config: Dict[str, Any] = Field(default_factory=dict)
    available_tools: ToolManager = Field(default_factory=lambda: ToolManager([]))

    def model_post_init(self, __context: Any = None) -> None:
        super().model_post_init(__context)
        memory_tools = [
            AddMemoryTool(mem0_config=self.mem0_config),
            SearchMemoryTool(mem0_config=self.mem0_config),
            GetAllMemoryTool(mem0_config=self.mem0_config),
            UpdateMemoryTool(mem0_config=self.mem0_config),
            DeleteMemoryTool(mem0_config=self.mem0_config),
        ]
        self.available_tools = ToolManager(memory_tools)
        if hasattr(self, "_refresh_prompts"):
            self._refresh_prompts()


def build_agent(mem0_cfg: Dict[str, Any]) -> DeFiMemoryAgent:
    return DeFiMemoryAgent(
        llm=ChatBot(
            llm_provider="openrouter",
            base_url="https://openrouter.ai/api/v1",
            model_name="anthropic/claude-3.5-sonnet",
            enable_long_term_memory=False,  # memory comes from toolkit tools instead
        ),
        mem0_config=mem0_cfg,
        system_prompt=(
            "You are a DeFi investment advisor. Use the provided Mem0 tools to recall "
            "and update user preferences before answering."
        ),
    )


def print_memories(result: ToolResult, label: str) -> None:
    memories = extract_memories(result)
    print(f"[Mem0] {label}:")
    if not memories:
        print("  (none)")
    for m in memories:
        print(f"  - {m}")


async def phase_capture(agent: DeFiMemoryAgent) -> None:
    print("\n=== Phase 1: Capture high-risk Solana preferences ===")
    await agent.available_tools.execute(
        name="add_memory",
        tool_input={
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "I am a high-risk degen trader. I exclusively trade meme coins on Solana "
                        "and dislike Ethereum gas fees."
                    ),
                }
            ],
            "user_id": USER_ID,
            "async_mode": False,
        },
    )
    # Verify storage immediately after add to avoid read-after-write surprises
    verified: ToolResult = ToolResult()
    for attempt in range(3):
        verified = await agent.available_tools.execute(
            name="get_all_memory",
            tool_input={"user_id": USER_ID, "limit": 5},
        )
        if extract_memories(verified):
            break
        await asyncio.sleep(0.5)
    print_memories(verified, "Verification after Phase 1 store")

    memories = await agent.available_tools.execute(
        name="search_memory",
        tool_input={"query": "Solana meme coins high risk", "user_id": USER_ID},
    )
    print_memories(memories, "After Phase 1 store")


async def phase_recall(mem0_cfg: Dict[str, Any]) -> None:
    print("\n=== Phase 2: Recall with a fresh agent instance ===")
    agent = build_agent(mem0_cfg)
    memories = await agent.available_tools.execute(
        name="search_memory",
        tool_input={"query": "trading strategy solana meme", "user_id": USER_ID},
    )
    print_memories(memories, "Retrieved for Phase 2")


async def phase_update(agent: DeFiMemoryAgent, memory_id: Optional[str]) -> None:
    print("\n=== Phase 3: Update preferences to safer Arbitrum yield ===")
    await agent.available_tools.execute(
        name="add_memory",
        tool_input={
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "I lost too much money. I want to pivot to safe stablecoin yield farming on Arbitrum now."
                    ),
                }
            ],
            "user_id": USER_ID,
            "async_mode": False,
        },
    )
    update_result = await agent.available_tools.execute(
        name="update_memory",
        tool_input={
            "memory_id": memory_id,
            "text": "User pivoted to safer Arbitrum stablecoin yield farming with low risk.",
            "user_id": USER_ID,
        },
    )
    print(f"[Mem0] Update result: {update_result}")
    memories = await agent.available_tools.execute(
        name="search_memory",
        tool_input={"query": "stablecoin yield chain choice", "user_id": USER_ID},
    )
    print_memories(memories, "Retrieved after update (Phase 3)")


async def phase_cleanup(agent: DeFiMemoryAgent, memory_id: Optional[str]) -> None:
    print("\n=== Phase 4: Clean up a memory entry ===")
    delete_result = await agent.available_tools.execute(
        name="delete_memory",
        tool_input={"memory_id": memory_id, "user_id": USER_ID},
    )
    print(f"[Mem0] Delete result: {delete_result}")
    remaining = await agent.available_tools.execute(
        name="get_all_memory",
        tool_input={"limit": 5, "user_id": USER_ID},
    )
    print_memories(remaining, "Remaining memories after delete")


async def main() -> None:
    mem0_cfg = {
        "user_id": USER_ID,
        "metadata": {"project": "defi-investment-advisor"},
        "async_mode": False,  # synchronous writes so the next search sees new data
    }

    try:
        agent = build_agent(mem0_cfg)
        await phase_capture(agent)
        await phase_recall(mem0_cfg)

        all_memories = await agent.available_tools.execute(
            name="get_all_memory", tool_input={"limit": 5, "user_id": USER_ID}
        )
        print_memories(all_memories, "All memories before update/delete")
        first_id = extract_first_memory_id(all_memories)

        await phase_update(agent, first_id)
        await phase_cleanup(agent, first_id)
    finally:
        await get_llm_manager().cleanup()


if __name__ == "__main__":
    asyncio.run(main())
