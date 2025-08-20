"""
Graph engine: StateGraph, CompiledGraph, and interrupt API.
"""
import asyncio
import logging
import uuid
import inspect
import time
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

from .exceptions import (
    GraphExecutionError,
    NodeExecutionError,
    GraphConfigurationError,
    EdgeRoutingError,
    InterruptError,
)
from .types import (
    NodeContext,
    NodeResult,
    RouterResult,
    ParallelBranchConfig,
    Command,
    StateSnapshot,
)
from .reducers import (
    merge_dicts,
)
from .decorators import node_decorator
from .checkpointer import InMemoryCheckpointer

logger = logging.getLogger(__name__)


def interrupt(data: Dict[str, Any]) -> Any:
    """Interrupt execution and wait for human input."""
    raise InterruptError(data)


class StateGraph:
    """Minimal-yet-complete StateGraph used by examples和tests。"""

    def __init__(self, state_schema: type, checkpointer: InMemoryCheckpointer = None):
        self.state_schema = state_schema
        self.checkpointer = checkpointer or InMemoryCheckpointer()
        self.nodes: Dict[str, Callable] = {}
        # edges: start_node -> either callable(state)->str or tuple(condition, path_map)
        self.edges: Dict[str, Any] = {}
        self.entry_point: Optional[str] = None
        self._compiled = False
        # parallel execution support
        self.parallel_groups: Dict[str, List[str]] = {}
        self.node_to_group: Dict[str, str] = {}
        self.parallel_entry_nodes: set[str] = set()
        self.parallel_group_configs: Dict[str, Dict[str, Any]] = {}
        # monitoring & cleanup
        self.monitoring_enabled: bool = False
        self.monitoring_metrics: List[str] = []
        self.state_cleanup = None  # Optional[Callable[[Dict[str, Any]], None]]

    def enable_monitoring(self, metrics: Optional[List[str]] = None) -> "StateGraph":
        self.monitoring_enabled = True
        if metrics:
            self.monitoring_metrics = metrics
        return self

    def configure_parallel_group(self, group_name: str, *, join_strategy: str = "all_complete", timeout: Optional[float] = None, error_strategy: str = "fail_fast", join_condition: Optional[Callable] = None) -> "StateGraph":
        self.parallel_group_configs[group_name] = {
            "join_strategy": join_strategy,
            "timeout": timeout,
            "error_strategy": error_strategy,
            "join_condition": join_condition,
        }
        return self

    def set_state_cleanup(self, cleaner: Callable[[Dict[str, Any]], None]) -> "StateGraph":
        self.state_cleanup = cleaner
        return self

    def set_default_state_cleanup(self) -> "StateGraph":
        """Set a default lightweight state cleanup that removes large temporary data"""
        def default_cleaner(state: Dict[str, Any]) -> None:
            # Remove temporary data that might accumulate
            keys_to_clean = []
            for key, value in state.items():
                if key.startswith("__temp_") or key.startswith("_cache_"):
                    keys_to_clean.append(key)
                elif isinstance(value, (list, dict)) and key.endswith("_history"):
                    # Limit history lists to last 100 entries
                    if isinstance(value, list) and len(value) > 100:
                        state[key] = value[-100:]
                elif isinstance(value, str) and len(value) > 10000:
                    # Truncate very large strings
                    state[key] = value[:10000] + "...[truncated]"

            for key in keys_to_clean:
                state.pop(key, None)

        self.state_cleanup = default_cleaner
        return self

    def add_node(self, name: str, action: Callable, parallel_group: Optional[str] = None) -> "StateGraph":
        if not name or not isinstance(name, str):
            raise GraphConfigurationError("Node name must be a non-empty string", component="node", details={"name": name})
        if name in ["START", "END"]:
            raise GraphConfigurationError(f"Node name '{name}' is reserved", component="node", details={"name": name})
        if name in self.nodes:
            raise GraphConfigurationError(f"Node '{name}' already exists", component="node", details={"name": name})
        if not callable(action):
            raise GraphConfigurationError("Node action must be callable", component="node", details={"name": name})
        # Keep original action (tests compare identity). No decorator wrapping here.
        self.nodes[name] = action
        # Register parallel group if provided
        if parallel_group:
            self.parallel_groups.setdefault(parallel_group, []).append(name)
            self.node_to_group[name] = parallel_group
        return self

    def add_edge(self, start_node: str, end_node: str) -> "StateGraph":
        if start_node != "START" and start_node not in self.nodes:
            raise GraphConfigurationError(f"Start node '{start_node}' does not exist", component="edge")
        if end_node != "END" and end_node not in self.nodes:
            raise GraphConfigurationError(f"End node '{end_node}' does not exist", component="edge")
        self.edges[start_node] = lambda state: end_node
        # Mark entry node for parallel group if destination is in a group
        if end_node in self.node_to_group:
            self.parallel_entry_nodes.add(end_node)
        return self

    def add_conditional_edges(self, start_node: str, condition: Callable[[Dict[str, Any]], str], path_map: Dict[str, str]) -> "StateGraph":
        if start_node not in self.nodes:
            raise GraphConfigurationError(f"Start node '{start_node}' does not exist", component="conditional_edge")
        if not callable(condition):
            raise GraphConfigurationError("Condition function must be callable", component="conditional_edge")
        if not path_map:
            raise GraphConfigurationError("Path map cannot be empty", component="conditional_edge")
        # Validate destinations
        invalid = [dest for dest in path_map.values() if dest != "END" and dest not in self.nodes]
        if invalid:
            raise GraphConfigurationError(f"Destination nodes do not exist: {invalid}", component="conditional_edge")
        self.edges[start_node] = (condition, path_map)
        return self

    def set_entry_point(self, node_name: str) -> "StateGraph":
        if node_name not in self.nodes:
            raise GraphConfigurationError(f"Entry point node '{node_name}' does not exist", component="entry_point")
        self.entry_point = node_name
        return self

    def compile(self) -> "CompiledGraph":
        errors: List[str] = []
        if not self.entry_point:
            errors.append("Graph must have an entry point")
        if not self.nodes:
            errors.append("Graph must have at least one node")
        if errors:
            raise GraphConfigurationError(
                f"Graph compilation failed: {'; '.join(errors)}",
                component="compilation",
                details={"errors": errors},
            )
        self._compiled = True
        return CompiledGraph(self)

    def resume_from_checkpoint(self, thread_id: str, checkpoint_id: Optional[str] = None) -> "CompiledGraph":
        """Create a CompiledGraph that can resume from a specific checkpoint"""
        compiled = CompiledGraph(self)
        compiled._resume_thread_id = thread_id
        compiled._resume_checkpoint_id = checkpoint_id
        return compiled


class CompiledGraph:
    def __init__(self, graph: StateGraph):
        self.graph = graph
        self.execution_history: List[Dict[str, Any]] = []
        self.max_execution_history = 1000  # Ring buffer limit
        self._resume_thread_id: Optional[str] = None
        self._resume_checkpoint_id: Optional[str] = None

    async def invoke(self, initial_state: Optional[Dict[str, Any]] = None, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        config = config or {}
        thread_id = config.get("configurable", {}).get("thread_id", str(uuid.uuid4()))

        # Handle resume from checkpoint
        if self._resume_thread_id:
            thread_id = self._resume_thread_id
            checkpoint = self.graph.checkpointer.get_checkpoint(thread_id, self._resume_checkpoint_id)
            if checkpoint:
                state = checkpoint.values.copy()
                current_node = checkpoint.next[0] if checkpoint.next else self.graph.entry_point
                iteration = checkpoint.metadata.get("iteration", 0)
            else:
                state = self._initialize_state(initial_state)
                current_node = self.graph.entry_point
                iteration = 0
        else:
            state = self._initialize_state(initial_state)
            current_node = self.graph.entry_point
            iteration = 0

        max_iterations = 100
        self._current_thread_id = thread_id
        try:
            while current_node and iteration < max_iterations:
                self._current_iteration = iteration
                iteration += 1
                # checkpoint (best-effort)
                try:
                    snapshot = StateSnapshot(values=state.copy(), next=(current_node,), config=config, metadata={"iteration": iteration, "node": current_node}, created_at=datetime.now())
                    self.graph.checkpointer.save_checkpoint(thread_id, snapshot)
                except Exception:
                    pass
                # execute current node or parallel group
                try:
                    # If current node is a parallel group entry, execute the entire group concurrently
                    if current_node in self.graph.node_to_group and current_node in self.graph.parallel_entry_nodes:
                        group_name = self.graph.node_to_group[current_node]
                        await self._execute_parallel_group(group_name, state)
                    else:
                        result = await self._execute_node(current_node, state)
                        if isinstance(result, Command):
                            if result.update:
                                self._update_state_with_reducers(state, result.update)
                                self._maybe_cleanup_state(state)
                            if result.goto:
                                current_node = result.goto
                                if current_node == "END":
                                    break
                                continue
                        elif isinstance(result, dict):
                            self._update_state_with_reducers(state, result)
                            self._maybe_cleanup_state(state)
                except InterruptError as e:
                    # record interrupt + checkpoint
                    try:
                        interrupt_snapshot = StateSnapshot(values=state.copy(), next=(current_node,), config=config, metadata={"iteration": iteration, "node": current_node, "interrupt_id": e.interrupt_id, "interrupt_data": e.interrupt_data, "status": "interrupted"}, created_at=datetime.now())
                        self.graph.checkpointer.save_checkpoint(thread_id, interrupt_snapshot)
                    except Exception:
                        pass
                    return {**state, "__interrupt__": [{"interrupt_id": e.interrupt_id, "value": e.interrupt_data, "node": current_node, "iteration": iteration}]}

                # next
                next_node = await self._get_next_node(current_node, state)
                if next_node == current_node:
                    break
                current_node = next_node
                if current_node == "END" or current_node is None:
                    break
            if iteration >= max_iterations:
                raise GraphExecutionError(f"Graph execution exceeded maximum iterations ({max_iterations})", node=current_node, iteration=iteration)
            return state
        except GraphExecutionError:
            raise
        except Exception as e:
            raise GraphExecutionError(f"Graph execution failed: {e}", node=current_node, iteration=iteration) from e

    def _maybe_cleanup_state(self, state: Dict[str, Any]) -> None:
        try:
            if self.graph.state_cleanup:
                self.graph.state_cleanup(state)
        except Exception:
            pass

    def _record_execution_metrics(self, node_name: str, start_time: datetime, end_time: datetime, success: bool, error: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None) -> None:
        if not self.graph.monitoring_enabled:
            return
        try:
            execution_time = (end_time - start_time).total_seconds()
            record = {
                "node_name": node_name,
                "start_time": start_time.isoformat(),
                "end_time": end_time.isoformat(),
                "execution_time": execution_time,
                "success": success,
                "error": error,
                "metadata": metadata or {}
            }
            self.execution_history.append(record)
            # Ring buffer: keep only last max_execution_history entries
            if len(self.execution_history) > self.max_execution_history:
                self.execution_history = self.execution_history[-self.max_execution_history:]
        except Exception:
            pass  # Don't let monitoring break execution

    def get_execution_metrics(self) -> Dict[str, Any]:
        """Get aggregated execution metrics"""
        if not self.execution_history:
            return {"total_executions": 0, "avg_execution_time": 0, "success_rate": 0, "node_stats": {}}

        total = len(self.execution_history)
        successful = sum(1 for h in self.execution_history if h["success"])
        total_time = sum(h["execution_time"] for h in self.execution_history)

        node_stats = {}
        for h in self.execution_history:
            node = h["node_name"]
            if node not in node_stats:
                node_stats[node] = {"count": 0, "total_time": 0, "errors": 0}
            node_stats[node]["count"] += 1
            node_stats[node]["total_time"] += h["execution_time"]
            if not h["success"]:
                node_stats[node]["errors"] += 1

        for node, stats in node_stats.items():
            stats["avg_time"] = stats["total_time"] / stats["count"]
            stats["error_rate"] = stats["errors"] / stats["count"]

        return {
            "total_executions": total,
            "avg_execution_time": total_time / total,
            "success_rate": successful / total,
            "node_stats": node_stats
        }

    async def _execute_parallel_group(self, group_name: str, state: Dict[str, Any]) -> None:
        nodes = self.graph.parallel_groups.get(group_name, [])
        if not nodes:
            return
        cfg = self.graph.parallel_group_configs.get(group_name, {})
        join_strategy = cfg.get("join_strategy", "all_complete")
        timeout = cfg.get("timeout")
        error_strategy = cfg.get("error_strategy", "fail_fast")
        join_condition = cfg.get("join_condition")

        # create tasks
        loop = asyncio.get_event_loop()
        tasks = {}
        for n in nodes:
            tasks[n] = loop.create_task(self._execute_node(n, state))

        completed_nodes: List[str] = []
        updates_to_merge: List[Dict[str, Any]] = []
        errors: List[Dict[str, Any]] = []

        async def handle_done(done_set):
            for t in done_set:
                # find node name by task
                node_name = next((name for name, task in tasks.items() if task is t), None)
                try:
                    result = t.result()
                    if isinstance(result, Command):
                        if result.update:
                            updates_to_merge.append(result.update)
                    elif isinstance(result, dict):
                        updates_to_merge.append(result)
                    elif isinstance(result, RouterResult):
                        # RouterResult in parallel branch is unusual; ignore routing but record metadata
                        updates_to_merge.append({"__router__": {"node": node_name, "next": result.next_node}})
                    completed_nodes.append(node_name or "")
                except Exception as e:
                    err_info = {"node": node_name, "error": str(e)}
                    errors.append(err_info)
                    if error_strategy == "fail_fast":
                        # cancel all other tasks
                        for task in tasks.values():
                            if not task.done():
                                task.cancel()
                        raise e

        try:
            if join_strategy == "any_first":
                done, pending = await asyncio.wait(tasks.values(), timeout=timeout, return_when=asyncio.FIRST_COMPLETED)
                await handle_done(done)
                # cancel pending
                for p in pending:
                    p.cancel()
            else:
                # all_complete or quorum
                if isinstance(join_strategy, str) and join_strategy.startswith("quorum_"):
                    quorum_val = join_strategy.split("_", 1)[1]
                    quorum_n = None
                    quorum_p = None
                    # attempt parse as int or float
                    try:
                        quorum_n = int(quorum_val)
                    except Exception:
                        try:
                            quorum_p = float(quorum_val)
                        except Exception:
                            pass
                    needed = 0
                    total = len(tasks)
                    if quorum_n is not None:
                        needed = min(total, max(1, quorum_n))
                    elif quorum_p is not None and 0 < quorum_p <= 1:
                        needed = max(1, int(total * quorum_p + 0.9999))
                    else:
                        needed = total
                    remaining = set(tasks.values())
                    start = datetime.now()
                    while remaining and len(completed_nodes) < needed:
                        to_wait_timeout = None
                        if timeout is not None:
                            elapsed = (datetime.now() - start).total_seconds()
                            left = max(0.0, timeout - elapsed)
                            to_wait_timeout = left
                            if left == 0:
                                break
                        done, pending = await asyncio.wait(remaining, timeout=to_wait_timeout, return_when=asyncio.FIRST_COMPLETED)
                        remaining = pending
                        if done:
                            await handle_done(done)
                    # timeout reached or quorum met: cancel remaining
                    for p in remaining:
                        p.cancel()
                else:
                    # all_complete
                    done, pending = await asyncio.wait(tasks.values(), timeout=timeout, return_when=asyncio.ALL_COMPLETED)
                    await handle_done(done)
                    for p in pending:
                        p.cancel()
        except asyncio.CancelledError:
            # Surface cancellation upwards
            raise
        except Exception:
            if error_strategy == "collect_errors":
                # merge successful updates and attach errors into state
                for upd in updates_to_merge:
                    self._update_state_with_reducers(state, upd)
                self._update_state_with_reducers(state, {"__errors__": errors})
                return
            raise

        # join_condition hook: allow custom early merge decision
        if callable(join_condition):
            try:
                allow_merge = await join_condition(state, completed_nodes) if asyncio.iscoroutinefunction(join_condition) else join_condition(state, completed_nodes)
                if allow_merge is False:
                    # skip merging if condition blocks it
                    for task in tasks.values():
                        if not task.done():
                            task.cancel()
                    return
            except Exception:
                # ignore join_condition errors and proceed to merge
                pass

        # finally merge accumulated updates
        for upd in updates_to_merge:
            self._update_state_with_reducers(state, upd)
        if errors and error_strategy == "ignore_errors":
            # attach errors but don't raise
            self._update_state_with_reducers(state, {"__errors__": errors})
        # optional cleanup per group
        self._maybe_cleanup_state(state)


    async def stream(self, initial_state: Optional[Dict[str, Any]] = None, config: Optional[Dict[str, Any]] = None, stream_mode: str = "values"):
        config = config or {}
        state = self._initialize_state(initial_state)
        current_node = self.graph.entry_point
        iteration = 0
        max_iterations = 100
        while current_node and iteration < max_iterations:
            iteration += 1
            try:
                # If current node is a parallel group entry, stream merged results after group finishes
                if current_node in self.graph.node_to_group and current_node in self.graph.parallel_entry_nodes:
                    await self._execute_parallel_group(self.graph.node_to_group[current_node], state)
                    if stream_mode == "values":
                        yield state.copy()
                else:
                    result = await self._execute_node(current_node, state)
                    if isinstance(result, Command):
                        if result.update:
                            self._update_state_with_reducers(state, result.update)
                            if stream_mode == "values":
                                yield state.copy()
                        if result.goto:
                            current_node = result.goto
                            continue
                    elif isinstance(result, dict):
                        self._update_state_with_reducers(state, result)
                        if stream_mode == "values":
                            yield state.copy()
            except InterruptError as e:
                yield {"type": "interrupt", "node": current_node, "interrupt_id": e.interrupt_id, "interrupt_data": e.interrupt_data, "state": state.copy()}
                return
            next_node = await self._get_next_node(current_node, state)
            if next_node == "END" or next_node is None:
                if stream_mode == "values":
                    yield state.copy()
                break
            current_node = next_node

    def _initialize_state(self, initial_state: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        state: Dict[str, Any] = {}
        # fill defaults for Annotated list types to [] for reducer usage
        if hasattr(self.graph.state_schema, "__annotations__"):
            for field_name, field_type in self.graph.state_schema.__annotations__.items():
                # heuristic for list-like fields
                if "List" in str(field_type) or "list" in str(field_type):
                    state[field_name] = []
                else:
                    state[field_name] = None
        if initial_state:
            state.update(initial_state)
        return state

    async def _execute_node(self, node_name: str, state: Dict[str, Any]) -> Any:
        node_action = self.graph.nodes[node_name]
        start_time = datetime.now()
        try:
            # Create NodeContext for nodes that expect it
            context = NodeContext(
                node_name=node_name,
                iteration=getattr(self, '_current_iteration', 0),
                thread_id=getattr(self, '_current_thread_id', 'default'),
                start_time=start_time
            )

            # Check if node function expects context parameter
            sig = inspect.signature(node_action)
            expects_context = len(sig.parameters) > 1

            if asyncio.iscoroutinefunction(node_action):
                if expects_context:
                    result = await node_action(state, context)
                else:
                    result = await node_action(state)
            else:
                if expects_context:
                    result = node_action(state, context)
                else:
                    result = node_action(state)

            end_time = datetime.now()
            context.execution_time = (end_time - start_time).total_seconds()
            self._record_execution_metrics(node_name, start_time, end_time, True)

            # Handle RouterResult - allow nodes to directly control routing
            if isinstance(result, RouterResult):
                # Convert RouterResult to Command with goto
                return Command(
                    update={"__router_metadata__": {
                        "node": node_name,
                        "next_node": result.next_node,
                        "confidence": result.confidence,
                        "reasoning": result.reasoning,
                        "metadata": result.metadata,
                        "alternative_paths": result.alternative_paths
                    }},
                    goto=result.next_node
                )

            return result
        except InterruptError:
            end_time = datetime.now()
            self._record_execution_metrics(node_name, start_time, end_time, False, "InterruptError")
            raise
        except Exception as e:
            end_time = datetime.now()
            self._record_execution_metrics(node_name, start_time, end_time, False, str(e))
            raise NodeExecutionError(f"Node '{node_name}' failed: {e}", node_name=node_name, original_error=e, state=state) from e

    async def _get_next_node(self, current_node: str, state: Dict[str, Any]) -> Optional[str]:
        if current_node not in self.graph.edges:
            return None
        edge = self.graph.edges[current_node]
        if isinstance(edge, tuple):
            condition_func, path_map = edge
            try:
                cond = await condition_func(state) if asyncio.iscoroutinefunction(condition_func) else condition_func(state)
                if cond in path_map:
                    return path_map[cond]
                else:
                    raise EdgeRoutingError(
                        f"Condition result '{cond}' not found in path map for node '{current_node}'. Available paths: {list(path_map.keys())}",
                        source_node=current_node,
                        condition_result=cond,
                        available_paths=list(path_map.keys()),
                    )
            except EdgeRoutingError:
                raise
            except Exception as e:
                raise NodeExecutionError(f"Condition function failed: {e}", node_name=current_node, original_error=e, state=state) from e
        elif callable(edge):
            try:
                return await edge(state) if asyncio.iscoroutinefunction(edge) else edge(state)
            except Exception as e:
                raise NodeExecutionError(f"Edge function failed: {e}", node_name=current_node, original_error=e, state=state) from e
        return None

    def _update_state_with_reducers(self, state: Dict[str, Any], updates: Dict[str, Any]) -> None:
        for key, value in updates.items():
            if key not in state:
                state[key] = value
                continue
            # merge dicts deeply, else replace
            if isinstance(state[key], dict) and isinstance(value, dict):
                state[key] = merge_dicts(state[key], value)
            elif isinstance(state[key], list) and isinstance(value, list):
                state[key] = state[key] + value
            else:
                state[key] = value

