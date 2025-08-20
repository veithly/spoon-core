"""
In-memory checkpointer for the graph package.
"""
from typing import Any, Dict, List, Optional
from datetime import datetime
from .types import StateSnapshot
from .exceptions import CheckpointError


class InMemoryCheckpointer:
    def __init__(self, max_checkpoints_per_thread: int = 100, *, max_threads: int | None = None, ttl_seconds: int | None = None):
        self.checkpoints: Dict[str, List[StateSnapshot]] = {}
        self.max_checkpoints_per_thread = max_checkpoints_per_thread
        self.max_threads = max_threads
        self.ttl_seconds = ttl_seconds
        self.last_access: Dict[str, datetime] = {}

    def _gc(self) -> None:
        # TTL-based cleanup
        if self.ttl_seconds is not None:
            cutoff = datetime.now().timestamp() - self.ttl_seconds
            remove_keys = []
            for tid, snaps in self.checkpoints.items():
                # remove snapshots older than TTL
                kept = [s for s in snaps if s.created_at.timestamp() >= cutoff]
                if kept:
                    self.checkpoints[tid] = kept[-self.max_checkpoints_per_thread:]
                else:
                    remove_keys.append(tid)
            for tid in remove_keys:
                self.checkpoints.pop(tid, None)
                self.last_access.pop(tid, None)
        # Global thread limit
        if self.max_threads is not None and len(self.checkpoints) > self.max_threads:
            # evict least recently used threads by last_access
            sorted_threads = sorted(self.last_access.items(), key=lambda kv: kv[1])
            to_evict = len(self.checkpoints) - self.max_threads
            for tid, _ in sorted_threads[:to_evict]:
                self.checkpoints.pop(tid, None)
                self.last_access.pop(tid, None)

    def save_checkpoint(self, thread_id: str, snapshot: StateSnapshot) -> None:
        try:
            if not thread_id:
                raise CheckpointError("Thread ID cannot be empty", operation="save")
            # update access time and run GC
            self.last_access[thread_id] = datetime.now()
            if thread_id not in self.checkpoints:
                self.checkpoints[thread_id] = []
            self.checkpoints[thread_id].append(snapshot)
            if len(self.checkpoints[thread_id]) > self.max_checkpoints_per_thread:
                self.checkpoints[thread_id] = self.checkpoints[thread_id][-self.max_checkpoints_per_thread:]
            self._gc()
        except Exception as e:
            raise CheckpointError(f"Failed to save checkpoint: {str(e)}", thread_id=thread_id, operation="save") from e

    def get_checkpoint(self, thread_id: str, checkpoint_id: str = None) -> Optional[StateSnapshot]:
        try:
            if not thread_id:
                raise CheckpointError("Thread ID cannot be empty", operation="get")
            if thread_id not in self.checkpoints:
                return None
            self.last_access[thread_id] = datetime.now()
            checkpoints = self.checkpoints[thread_id]
            if not checkpoints:
                return None
            if checkpoint_id:
                for checkpoint in checkpoints:
                    if str(checkpoint.created_at.timestamp()) == checkpoint_id:
                        return checkpoint
                return None
            return checkpoints[-1]
        except Exception as e:
            raise CheckpointError(
                f"Failed to get checkpoint: {str(e)}",
                thread_id=thread_id,
                checkpoint_id=checkpoint_id,
                operation="get",
            ) from e

    def list_checkpoints(self, thread_id: str) -> List[StateSnapshot]:
        try:
            if not thread_id:
                raise CheckpointError("Thread ID cannot be empty", operation="list")
            self.last_access[thread_id] = datetime.now()
            return self.checkpoints.get(thread_id, [])
        except Exception as e:
            raise CheckpointError(f"Failed to list checkpoints: {str(e)}", thread_id=thread_id, operation="list") from e

    def clear_thread(self, thread_id: str) -> None:
        if thread_id in self.checkpoints:
            del self.checkpoints[thread_id]
        if thread_id in self.last_access:
            del self.last_access[thread_id]

