import logging
from typing import Any, Dict, List, Optional

from pydantic import Field

from spoon_ai.memory.mem0_client import SpoonMem0
from spoon_ai.tools.base import BaseTool, ToolFailure

logger = logging.getLogger(__name__)


class _Mem0ToolMixin(BaseTool):
    """Shared helpers for Mem0-backed tools."""

    mem0_client: Optional[SpoonMem0] = Field(default=None, exclude=True)
    mem0_config: Optional[Dict[str, Any]] = Field(default=None, exclude=True)

    model_config = {"arbitrary_types_allowed": True, "extra": "allow"}

    def _client(self) -> SpoonMem0:
        if self.mem0_client:
            return self.mem0_client
        client = SpoonMem0(self.mem0_config or {})
        object.__setattr__(self, "mem0_client", client)
        return client

    def _resolve_user_id(self, provided: Optional[str]) -> Optional[str]:
        if provided:
            return provided
        if self.mem0_config and self.mem0_config.get("user_id"):
            return self.mem0_config["user_id"]
        if self.mem0_client and getattr(self.mem0_client, "user_id", None):
            return self.mem0_client.user_id
        return None

    def _format_results(self, results: List[str]) -> str:
        if not results:
            return "No memories found."
        return "\n".join(f"- {item}" for item in results)


class AddMemoryTool(_Mem0ToolMixin):
    name: str = "add_memory"
    description: str = (
        "Stores important information, user preferences, or key facts into long-term memory."
    )
    parameters: dict = {
        "type": "object",
        "properties": {
            "data": {"type": "string", "description": "The information to remember."},
            "user_id": {
                "type": "string",
                "description": "Optional user identifier to scope the memory.",
            },
            "metadata": {
                "type": "object",
                "description": "Optional metadata to store with the memory.",
            },
        },
        "required": ["data"],
    }

    async def execute(self, data: str, user_id: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None) -> str:
        client = self._client()
        try:
            resolved_user = self._resolve_user_id(user_id)
            if metadata:
                client.add_text(data, user_id=resolved_user, metadata=metadata)
            else:
                client.add_text(data, user_id=resolved_user)
            return "Memory stored successfully."
        except Exception as exc:
            logger.warning("AddMemoryTool failed: %s", exc)
            raise ToolFailure(f"Failed to store memory: {exc}") from exc


class SearchMemoryTool(_Mem0ToolMixin):
    name: str = "search_memory"
    description: str = "Searches for relevant past memories or information based on a query."
    parameters: dict = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Search query for long-term memory."},
            "user_id": {
                "type": "string",
                "description": "Optional user identifier to scope the search.",
            },
            "limit": {
                "type": "integer",
                "description": "Maximum number of memories to return.",
                "default": 5,
            },
        },
        "required": ["query"],
    }

    async def execute(self, query: str, user_id: Optional[str] = None, limit: int = 5) -> str:
        client = self._client()
        try:
            resolved_user = self._resolve_user_id(user_id)
            results = client.search_memory(query, user_id=resolved_user, limit=limit)
            return self._format_results(results)
        except Exception as exc:
            logger.warning("SearchMemoryTool failed: %s", exc)
            raise ToolFailure(f"Failed to search memory: {exc}") from exc


class GetAllMemoryTool(_Mem0ToolMixin):
    name: str = "get_all_memory"
    description: str = (
        "Retrieves all stored memories for a specific user to understand their full context."
    )
    parameters: dict = {
        "type": "object",
        "properties": {
            "user_id": {
                "type": "string",
                "description": "Optional user identifier whose memories should be fetched. Defaults to configured user.",
            },
            "limit": {
                "type": "integer",
                "description": "Maximum number of memories to return.",
                "default": 100,
            },
        },
    }

    async def execute(self, user_id: Optional[str] = None, limit: int = 100) -> str:
        client = self._client()
        try:
            resolved_user = self._resolve_user_id(user_id)
            results = client.get_all_memory(user_id=resolved_user, limit=limit)
            return self._format_results(results)
        except Exception as exc:
            logger.warning("GetAllMemoryTool failed: %s", exc)
            raise ToolFailure(f"Failed to fetch memories: {exc}") from exc
