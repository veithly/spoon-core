"""spoon_ai short-term memory demos."""

import asyncio
import os
from types import SimpleNamespace
from typing import List, Dict, Any, Optional
import uuid

from spoon_ai.chat import ChatBot
from spoon_ai.schema import Message
from spoon_ai.memory.short_term_manager import (
    ShortTermMemoryManager,
    TrimStrategy,
)
from spoon_ai.memory.remove_message import RemoveMessage, REMOVE_ALL_MESSAGES
from spoon_ai.graph import StateGraph, END
from spoon_ai.graph.reducers import add_messages
from spoon_ai.graph.checkpointer_sqlite import SQLiteCheckpointer
from spoon_ai.graph.checkpointer_postgres import PostgresCheckpointer


def _ensure_message_ids(messages: List[Message]) -> None:
    for msg in messages:
        if not getattr(msg, "id", None):
            msg.id = str(uuid.uuid4())


def _basic_summary_text(messages: List[Message]) -> str:
    user_queries = [msg.content for msg in messages if msg.role == "user" and msg.content]
    assistant_responses = [
        msg.content for msg in messages if msg.role == "assistant" and msg.content
    ]
    parts: List[str] = []
    if user_queries:
        topics = "; ".join(user_queries[:3])
        parts.append(f"The user asked about: {topics}.")
    if assistant_responses:
        excerpt = assistant_responses[0][:120]
        parts.append(f"The assistant responded with examples such as: {excerpt}...")
    return " ".join(parts) or "Summary unavailable."


class FallbackSummarizer:
    async def chat(
        self,
        messages: List[Message],
        provider: Optional[str] = None,
        model: Optional[str] = None,
    ) -> SimpleNamespace:
        return SimpleNamespace(content=_basic_summary_text(messages))

class ShortTermMemoryDemoAgent:
    """Generates a demo conversation using ChatBot, with a scripted fallback."""

    def __init__(self, system_prompt: str, prompts: List[str]) -> None:
        self.system_prompt = system_prompt
        self.prompts = prompts
        self.chatbot: Optional[ChatBot] = None
        self._history: Optional[List[Message]] = None
        self.source: str = "uninitialized"

    async def get_history(self) -> List[Message]:
        if self._history is None:
            await self._build_history()
        history = self._history or []
        return [msg.model_copy(deep=True) for msg in history]

    def get_llm_manager(self):
        return self.chatbot.llm_manager if self.chatbot else None

    def get_model_name(self) -> Optional[str]:
        return self.chatbot.model_name if self.chatbot else None

    def get_llm_provider(self) -> Optional[str]:
        return self.chatbot.llm_provider if self.chatbot else None

    async def _build_history(self) -> None:
        self.source = "initializing"
        self.chatbot = ChatBot(enable_short_term_memory=False)

        history: List[Message] = []
        if self.system_prompt:
            history.append(
                Message(role="system", content=self.system_prompt)
            )

        
        for prompt in self.prompts:
            user_msg = Message(role="user", content=prompt)
            history.append(user_msg)
            response_text = await self.chatbot.ask(list(history))
            assistant_msg = Message(
                role="assistant",
                content=response_text,
            )
            history.append(assistant_msg)

        _ensure_message_ids(history)
        self._history = history
        self.source = "live"


DEMO_AGENT = ShortTermMemoryDemoAgent(
    system_prompt="You are a patient teacher who explains technical ideas in plain language.",
    prompts=[
        "Hello, what is machine learning?",
        "Can you describe supervised learning in one sentence?",
        "How do neural networks train?",
        "Mention one risk when training models.",
    ],
)

def _print_messages(title: str, messages: List[Message]) -> None:
    print(title)
    for idx, msg in enumerate(messages):
        preview = msg.content
        print(
            f"  {idx}: id={getattr(msg, 'id', None)} role={msg.role} -> {preview}"
        )


def _print_checkpoint_list(title: str, items: List[Dict[str, Any]]) -> None:
    print(title)
    for entry in items:
        print(
            f"  -> id={entry.get('checkpoint_id')} created={entry.get('created_at')} count={entry.get('message_count', '?')}"
        )


async def example_trim_messages() -> None:
    print("Example 1: Trim Messages")
    manager = ShortTermMemoryManager()
    messages = await DEMO_AGENT.get_history()
    _print_messages("Original messages", messages)

    total_tokens = await manager.token_counter.count_tokens(messages)
    print(f"Total tokens: {total_tokens}")

    max_tokens = 200
    print(f"Max tokens allowed: {max_tokens}")

    trimmed = await manager.trim_messages(
        messages=messages,
        max_tokens=max_tokens,
        strategy=TrimStrategy.FROM_END,
        keep_system=True,
    )
    _print_messages("Messages after trim", trimmed)


async def example_remove_messages() -> None:
    print("Example 2: RemoveMessage Directives")
    history = await DEMO_AGENT.get_history()
    assistant_ids = [msg.id for msg in history if msg.role == "assistant"]

    chatbot = ChatBot(enable_short_term_memory=True)
    removals = [
        chatbot.remove_message(assistant_ids[0]),
        chatbot.remove_message(assistant_ids[-1]),
        chatbot.remove_all_messages(),
    ]

    print("Removal directives emitted:")
    for rm in removals:
        print(f"  -> type={rm.type} id={rm.target_id}")

    updated_history = add_messages(history, removals)
    remaining_ids = [getattr(msg, "id", None) for msg in updated_history]
    if remaining_ids:
        print("History after applying removals (ids preserved):")
        print(f"  -> {remaining_ids}")
    else:
        print("History after applying removals: [] (all messages cleared)")


async def example_summarise_messages() -> None:
    print("Example 3: Summarise Messages")
    conversation = await DEMO_AGENT.get_history()
    
    manager: ShortTermMemoryManager = ShortTermMemoryManager()
    llm_manager = DEMO_AGENT.get_llm_manager() if DEMO_AGENT.source == "live" else None
    summary_model = DEMO_AGENT.get_model_name()
    summary_provider = DEMO_AGENT.get_llm_provider()

    fallback_used = False
    if llm_manager and summary_provider and not summary_model:
        provider_cfg = llm_manager.config_manager.load_provider_config(
            summary_provider
        )
        summary_model = provider_cfg.model or summary_model

    if llm_manager is None:

        chatbot = ChatBot(
            llm_provider=summary_provider,
            model_name=summary_model,
            enable_short_term_memory=True,
        )
        manager = chatbot.short_term_memory_manager or manager
        llm_manager = chatbot.llm_manager
        summary_model = (
            chatbot.short_term_memory_config.summary_model
            or chatbot.model_name
            or summary_model
        )
        summary_provider = chatbot.llm_provider or summary_provider
    elif not summary_model:
        print(
            "Summary model not configured; falling back to offline summarizer. "
        )
        llm_manager = FallbackSummarizer()
        summary_model = "fallback-summary"
        fallback_used = True

    max_tokens_threshold = 200
    print(f"Max tokens before summary: {max_tokens_threshold}")

    summary_state = ""
    llm_messages, removals, summary_text = await manager.summarize_messages(
        messages=conversation,
        max_tokens_before_summary=max_tokens_threshold,
        messages_to_keep=2,
        summary_model=summary_model,
        llm_manager=llm_manager,
        existing_summary=summary_state,
    )

    summary_state = summary_text
    print(f"  {summary_text}...")
    if fallback_used:
        print("  (Generated by fallback summarizer)")
    
    print("Removal directives issued:")
    for rm in removals:
        print(f"  -> remove {rm.target_id}")
    reduced_history = add_messages(conversation, removals)
    reduced_ids = [msg.id for msg in reduced_history]
    print("History after applying removals (ids):")
    print(f"  -> {reduced_ids}")


async def example_checkpoint_management() -> None:
    print("Example 4: Checkpoint Management")

    chatbot = ChatBot(enable_short_term_memory=True)
 
    thread_id = "demo-thread"

    messages_v1 = [
        Message(id="m1", role="user", content="Hello"),
        Message(id="m2", role="assistant", content="Hi there!"),
    ]
    cp1 = chatbot.save_checkpoint(
        thread_id, messages_v1, metadata={"stage": "initial"}
    )
    print(f"Saved checkpoint: {cp1}")

    messages_v2 = messages_v1 + [
        Message(id="m3", role="user", content="How are you?"),
        Message(id="m4", role="assistant", content="Doing great!"),
    ]
    cp2 = chatbot.save_checkpoint(
        thread_id, messages_v2, metadata={"stage": "follow_up"}
    )
    print(f"Saved checkpoint: {cp2}")

    print("Checkpoint history:")
    for entry in chatbot.list_checkpoints(thread_id):
        print(f"  -> id={entry['checkpoint_id']} created={entry['created_at']} count={entry['message_count']}")

    restored = chatbot.restore_checkpoint(thread_id, cp1)
    _print_messages("Messages restored from first checkpoint", restored or [])

    chatbot.clear_checkpoints(thread_id)
    print("All checkpoints cleared.")


async def main() -> None:
    await example_trim_messages()
    await example_remove_messages()
    await example_summarise_messages()
    await example_checkpoint_management()

if __name__ == "__main__":
    asyncio.run(main())
