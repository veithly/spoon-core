"""
GraphAgent implementation for the graph package.
"""
import asyncio
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from .engine import StateGraph


@dataclass
class AgentStateCheckpoint:
    messages: List[Any]
    current_step: int
    agent_state: str
    preserved_state: Optional[Dict[str, Any]]
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)


class MockMemory:
    def __init__(self):
        self.messages = []
    def clear(self):
        self.messages = []
    def add_message(self, msg):
        self.messages.append(msg)
    def get_messages(self):
        return self.messages


class GraphAgent:
    def __init__(self, name: str, graph: StateGraph, preserve_state: bool = False, **kwargs):
        self.name = name
        self.graph = graph.compile()
        self.preserve_state = preserve_state
        self.memory = kwargs.get('memory')
        self.current_step = 0
        self.state = "IDLE"
        self._state_lock = asyncio.Lock()
        self._max_metadata_size = kwargs.get('max_metadata_size', 1024)
        self._last_state = None
        self.execution_metadata: Dict[str, Any] = {}
        if self.memory is None:
            self.memory = MockMemory()

    async def run(self, request: Optional[str] = None) -> str:
        async with self._state_lock:
            checkpoint = self._create_checkpoint()
            try:
                initial_state = {"input": request} if request else {}
                if self.preserve_state and self._last_state:
                    if self._validate_preserved_state(self._last_state):
                        initial_state.update(self._last_state)
                    else:
                        self._last_state = None
                result = await self.graph.invoke(initial_state)
                if self.preserve_state:
                    self._last_state = self._sanitize_preserved_state(result)
                self.execution_metadata = {
                    "execution_successful": True,
                    "execution_time": time.time(),
                    "last_request": request[:100] if request else None,
                }
                return str(result.get("output", result))
            except Exception as error:
                await self._handle_execution_error(error, checkpoint)
                raise

    def _create_checkpoint(self) -> AgentStateCheckpoint:
        messages = []
        if self.memory and hasattr(self.memory, 'get_messages'):
            try:
                messages = [m for m in self.memory.get_messages() if self._validate_message(m)]
            except Exception:
                messages = []
        return AgentStateCheckpoint(messages=messages, current_step=self.current_step, agent_state=self.state, preserved_state=self._last_state.copy() if self._last_state else None)

    async def _handle_execution_error(self, error: Exception, checkpoint: AgentStateCheckpoint):
        try:
            self._restore_from_checkpoint(checkpoint)
            self.execution_metadata = {
                "error": str(error)[:500],
                "error_type": type(error).__name__,
                "execution_successful": False,
                "execution_time": time.time(),
                "recovery_attempted": True,
            }
        except Exception:
            self._emergency_reset()
        self._safe_clear_preserved_state()

    def _restore_from_checkpoint(self, checkpoint: AgentStateCheckpoint):
        if not self.memory:
            return
        try:
            self.memory.clear()
            valid = []
            for msg in checkpoint.messages:
                if self._validate_message(msg):
                    valid.append(msg)
            for msg in valid:
                self.memory.add_message(msg)
            self.current_step = checkpoint.current_step
            self.state = checkpoint.agent_state
            if checkpoint.preserved_state and self._validate_preserved_state(checkpoint.preserved_state):
                self._last_state = checkpoint.preserved_state.copy()
            else:
                self._last_state = None
        except Exception:
            self._emergency_reset()

    def _validate_message(self, msg) -> bool:
        try:
            if not hasattr(msg, 'role') or not hasattr(msg, 'content'):
                return False
            if getattr(msg, 'role') not in ['user', 'assistant', 'tool', 'system']:
                return False
            if not isinstance(getattr(msg, 'content', None), (str, type(None))):
                return False
            return True
        except Exception:
            return False

    def _validate_preserved_state(self, state: Any) -> bool:
        try:
            if not isinstance(state, dict):
                return False
            if len(str(state)) > 10000:
                return False
            return True
        except Exception:
            return False

    def _sanitize_preserved_state(self, state: Dict[str, Any]) -> Dict[str, Any]:
        try:
            if not isinstance(state, dict):
                return {}
            sanitized: Dict[str, Any] = {}
            for k, v in state.items():
                if str(k).startswith('__'):
                    continue
                if isinstance(v, (str, int, float, bool, type(None))):
                    sanitized[k] = v
                elif isinstance(v, (list, dict)) and len(str(v)) <= 1000:
                    sanitized[k] = v
            return sanitized
        except Exception:
            return {}

    def _safe_clear_preserved_state(self):
        try:
            if self._last_state and not self._validate_preserved_state(self._last_state):
                self._last_state = None
        except Exception:
            self._last_state = None

    def _emergency_reset(self):
        try:
            if self.memory and hasattr(self.memory, 'clear'):
                self.memory.clear()
            self.current_step = 0
            self.state = "IDLE"
            self._last_state = None
            self.execution_metadata = {"emergency_reset": True, "timestamp": time.time()}
        except Exception:
            pass

    # Convenience APIs used by examples
    def clear_state(self):
        try:
            if self.memory and hasattr(self.memory, 'clear'):
                self.memory.clear()
        except Exception:
            pass
        self._last_state = None
        self.current_step = 0
        self.execution_metadata = {}

    def get_execution_metadata(self) -> Dict[str, Any]:
        try:
            return dict(self.execution_metadata) if isinstance(self.execution_metadata, dict) else {}
        except Exception:
            return {}

    def get_execution_history(self) -> List[Dict[str, Any]]:
        try:
            if hasattr(self.graph, 'execution_history'):
                return list(self.graph.execution_history)
        except Exception:
            pass
        return []

