import asyncio
import inspect
from abc import ABC
from typing import Any, Callable, Dict, List, Optional
from uuid import UUID

from spoon_ai.schema import Message, LLMResponse


class BaseCallbackHandler(ABC):
    """Minimal callback interface; override only what you need."""

    async def on_llm_start(self,run_id: UUID,messages: List[Message],**kwargs: Any,) -> None:
        return None

    async def on_llm_new_token(self,token: str,*,chunk: Optional[Any] = None,run_id: UUID = None,**kwargs: Any,) -> None:
        return None

    async def on_llm_end(self,response: Any,*,run_id: UUID,**kwargs: Any,) -> None:
        return None

    async def on_llm_error(self,error: Exception,*,run_id: UUID,**kwargs: Any,) -> None:
        return None

    async def on_tool_start(self,tool_name: str,tool_input: Dict[str, Any],*,run_id: UUID,**kwargs: Any,) -> None:
        return None

    async def on_tool_end(self,tool_name: str,tool_output: Any,*,run_id: UUID,**kwargs: Any,) -> None:
        return None

    async def on_tool_error(self,error: Exception,*,run_id: UUID,**kwargs: Any,) -> None:
        return None
    
    async def on_retriever_start(self, run_id: UUID, query: Any, **kwargs: Any) -> None:
        return None

    async def on_retriever_end(self, run_id: UUID, documents: Any, **kwargs: Any) -> None:
        return None

    async def on_prompt_start(self, run_id: UUID, inputs: Any, **kwargs: Any) -> None:
        return None

    async def on_prompt_end(self, run_id: UUID, output: Any, **kwargs: Any) -> None:
        return None
    
    @staticmethod
    def _ensure_async(method: Callable) -> Callable:
        if inspect.iscoroutinefunction(method):
            return method
        
        async def async_wrapper(*args, **kwargs):
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, lambda: method(*args, **kwargs))

        return async_wrapper

class AsyncCallbackHandler(BaseCallbackHandler):
    """Marker class for handlers guaranteed to be async."""
    pass

class LLMManagerMixin:
    """Mixin for LLM-related callbacks."""
    pass

class ChainManagerMixin:
    """Mixin for Chain-related callbacks."""
    pass

class ToolManagerMixin:
    """Mixin for Tool-related callbacks."""
    pass

class RetrieverManagerMixin:
    """Mixin for Retriever-related callbacks."""
    pass

class PromptManagerMixin:
    """Mixin for prompt-related callbacks."""
    pass

CallbackHandlerLike = BaseCallbackHandler
