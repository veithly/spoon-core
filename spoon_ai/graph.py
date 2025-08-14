"""
Graph-based execution system for SpoonOS agents.

This module provides a LangGraph-inspired framework with advanced features:
- State management with TypedDict and reducers
- LLM Manager integration
- Error handling and recovery
- Human-in-the-loop patterns
- Multi-agent coordination
- Comprehensive testing support
- Checkpointing and persistence
"""

import asyncio
import logging
import uuid
import json
import time
import functools
from typing import Dict, Any, Callable, Union, Optional, Tuple, List, Annotated, TypedDict, Literal, Set
from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime
import operator
from concurrent.futures import ThreadPoolExecutor, as_completed

from spoon_ai.schema import Message
from spoon_ai.llm.manager import get_llm_manager

logger = logging.getLogger(__name__)


class GraphExecutionError(Exception):
    """Raised when graph execution encounters an error."""
    
    def __init__(self, message: str, node: str = None, iteration: int = None, context: Dict[str, Any] = None):
        self.node = node
        self.iteration = iteration
        self.context = context or {}
        super().__init__(message)


class NodeExecutionError(Exception):
    """Raised when a node fails to execute."""
    
    def __init__(self, message: str, node_name: str, original_error: Exception = None, state: Dict[str, Any] = None):
        self.node_name = node_name
        self.original_error = original_error
        self.state = state
        super().__init__(message)


class StateValidationError(Exception):
    """Raised when state validation fails."""
    
    def __init__(self, message: str, field: str = None, expected_type: type = None, actual_value: Any = None):
        self.field = field
        self.expected_type = expected_type
        self.actual_value = actual_value
        super().__init__(message)


class CheckpointError(Exception):
    """Raised when checkpoint operations fail."""
    
    def __init__(self, message: str, thread_id: str = None, checkpoint_id: str = None, operation: str = None):
        self.thread_id = thread_id
        self.checkpoint_id = checkpoint_id
        self.operation = operation
        super().__init__(message)


class GraphConfigurationError(Exception):
    """Raised when graph configuration is invalid."""
    
    def __init__(self, message: str, component: str = None, details: Dict[str, Any] = None):
        self.component = component
        self.details = details or {}
        super().__init__(message)


class EdgeRoutingError(Exception):
    """Raised when edge routing fails."""
    
    def __init__(self, message: str, source_node: str, condition_result: Any = None, available_paths: List[str] = None):
        self.source_node = source_node
        self.condition_result = condition_result
        self.available_paths = available_paths or []
        super().__init__(message)


class InterruptError(Exception):
    """Raised when graph execution is interrupted for human input."""
    
    def __init__(self, interrupt_data: Dict[str, Any], interrupt_id: str = None, node: str = None):
        self.interrupt_data = interrupt_data
        self.interrupt_id = interrupt_id or str(uuid.uuid4())
        self.node = node
        super().__init__(f"Graph interrupted at node '{node}': {interrupt_data}")


# Enhanced Data Structures for improved functionality
@dataclass
class NodeContext:
    """Enhanced node execution context with runtime information"""
    node_name: str
    iteration: int
    thread_id: str
    execution_time: float = 0.0
    start_time: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    parent_context: Optional['NodeContext'] = None
    execution_path: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        if self.start_time is None:
            self.start_time = datetime.now()


@dataclass
class NodeResult:
    """Standardized node execution result"""
    updates: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    logs: List[str] = field(default_factory=list)
    goto: Optional[str] = None
    error: Optional[str] = None
    confidence: float = 1.0
    reasoning: Optional[str] = None
    
    def to_command(self) -> 'Command':
        """Convert to Command object for backward compatibility"""
        return Command(
            update=self.updates,
            goto=self.goto,
            resume=None
        )


@dataclass
class RouterResult:
    """Enhanced routing result with confidence and reasoning"""
    next_node: str
    confidence: float
    reasoning: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    alternative_paths: List[Tuple[str, float]] = field(default_factory=list)


@dataclass
class ParallelBranchConfig:
    """Configuration for parallel branch execution"""
    branch_name: str
    nodes: List[str]
    join_strategy: Literal["all_complete", "first_complete", "timeout"] = "all_complete"
    timeout: Optional[float] = None
    join_condition: Optional[Callable[[List[NodeResult]], bool]] = None
    error_strategy: Literal["fail_fast", "continue", "retry"] = "fail_fast"


@dataclass
class Command:
    """Command object for controlling graph flow and state updates."""
    update: Optional[Dict[str, Any]] = None
    goto: Optional[str] = None
    resume: Optional[Any] = None


@dataclass
class StateSnapshot:
    """Snapshot of graph state at a specific point in time."""
    values: Dict[str, Any]
    next: Tuple[str, ...]
    config: Dict[str, Any]
    metadata: Dict[str, Any]
    created_at: datetime
    parent_config: Optional[Dict[str, Any]] = None
    tasks: Tuple[Any, ...] = field(default_factory=tuple)


class InMemoryCheckpointer:
    """
    Simple in-memory checkpointer for development and testing.
    
    This checkpointer stores state snapshots in memory and provides
    basic checkpoint management functionality. For production use,
    consider using persistent checkpointers like Redis or PostgreSQL.
    """
    
    def __init__(self, max_checkpoints_per_thread: int = 100):
        """
        Initialize the in-memory checkpointer.
        
        Args:
            max_checkpoints_per_thread: Maximum number of checkpoints to keep per thread
        """
        self.checkpoints: Dict[str, List[StateSnapshot]] = {}
        self.max_checkpoints_per_thread = max_checkpoints_per_thread
    
    def save_checkpoint(self, thread_id: str, snapshot: StateSnapshot) -> None:
        """
        Save a checkpoint for a thread.
        
        Args:
            thread_id: Unique identifier for the thread
            snapshot: State snapshot to save
            
        Raises:
            CheckpointError: If checkpoint saving fails
        """
        try:
            if not thread_id:
                raise CheckpointError("Thread ID cannot be empty", operation="save")
            
            if thread_id not in self.checkpoints:
                self.checkpoints[thread_id] = []
            
            self.checkpoints[thread_id].append(snapshot)
            
            # Limit the number of checkpoints per thread
            if len(self.checkpoints[thread_id]) > self.max_checkpoints_per_thread:
                self.checkpoints[thread_id] = self.checkpoints[thread_id][-self.max_checkpoints_per_thread:]
                
            logger.debug(f"Saved checkpoint for thread {thread_id} (total: {len(self.checkpoints[thread_id])})")
            
        except Exception as e:
            raise CheckpointError(
                f"Failed to save checkpoint: {str(e)}", 
                thread_id=thread_id, 
                operation="save"
            ) from e
    
    def get_checkpoint(self, thread_id: str, checkpoint_id: str = None) -> Optional[StateSnapshot]:
        """
        Get a specific checkpoint or the latest one.
        
        Args:
            thread_id: Unique identifier for the thread
            checkpoint_id: Optional specific checkpoint ID
            
        Returns:
            StateSnapshot or None if not found
            
        Raises:
            CheckpointError: If checkpoint retrieval fails
        """
        try:
            if not thread_id:
                raise CheckpointError("Thread ID cannot be empty", operation="get")
                
            if thread_id not in self.checkpoints:
                return None
            
            checkpoints = self.checkpoints[thread_id]
            if not checkpoints:
                return None
                
            if checkpoint_id:
                # Find specific checkpoint by ID (using created_at as ID for simplicity)
                for checkpoint in checkpoints:
                    if str(checkpoint.created_at.timestamp()) == checkpoint_id:
                        return checkpoint
                return None
            
            return checkpoints[-1]  # Return latest
            
        except Exception as e:
            raise CheckpointError(
                f"Failed to get checkpoint: {str(e)}", 
                thread_id=thread_id, 
                checkpoint_id=checkpoint_id,
                operation="get"
            ) from e
    
    def list_checkpoints(self, thread_id: str) -> List[StateSnapshot]:
        """
        List all checkpoints for a thread.
        
        Args:
            thread_id: Unique identifier for the thread
            
        Returns:
            List of state snapshots
            
        Raises:
            CheckpointError: If checkpoint listing fails
        """
        try:
            if not thread_id:
                raise CheckpointError("Thread ID cannot be empty", operation="list")
                
            return self.checkpoints.get(thread_id, [])
            
        except Exception as e:
            raise CheckpointError(
                f"Failed to list checkpoints: {str(e)}", 
                thread_id=thread_id,
                operation="list"
            ) from e
    
    def clear_thread(self, thread_id: str) -> None:
        """
        Clear all checkpoints for a thread.
        
        Args:
            thread_id: Unique identifier for the thread
        """
        if thread_id in self.checkpoints:
            del self.checkpoints[thread_id]
            logger.debug(f"Cleared all checkpoints for thread {thread_id}")
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get checkpointer statistics.
        
        Returns:
            Dictionary with statistics
        """
        total_checkpoints = sum(len(checkpoints) for checkpoints in self.checkpoints.values())
        return {
            "total_threads": len(self.checkpoints),
            "total_checkpoints": total_checkpoints,
            "max_checkpoints_per_thread": self.max_checkpoints_per_thread,
            "threads": {
                thread_id: len(checkpoints) 
                for thread_id, checkpoints in self.checkpoints.items()
            }
        }


# Enhanced Reducers and Validators
def add_messages(existing: List[Any], new: List[Any]) -> List[Any]:
    """Reducer function for adding messages to a list."""
    if existing is None:
        existing = []
    if new is None:
        return existing
    return existing + new


def merge_dicts(existing: Dict, new: Dict) -> Dict:
    """Merge dictionaries with deep merge support"""
    if existing is None:
        return new or {}
    if new is None:
        return existing
    
    result = existing.copy()
    for key, value in new.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_dicts(result[key], value)
        else:
            result[key] = value
    return result


def append_history(existing: List, new: Dict) -> List:
    """Append history records with timestamp"""
    if existing is None:
        existing = []
    if new is None:
        return existing
    
    history_entry = {"timestamp": datetime.now().isoformat(), **new}
    return existing + [history_entry]


def union_sets(existing: Set, new: Set) -> Set:
    """Set union operation"""
    if existing is None:
        existing = set()
    if new is None:
        return existing
    return existing | new


def validate_range(min_val: float, max_val: float):
    """Range validator factory"""
    def validator(value: float) -> float:
        if not isinstance(value, (int, float)):
            raise ValueError(f"Value must be numeric, got {type(value)}")
        if not min_val <= value <= max_val:
            raise ValueError(f"Value {value} not in range [{min_val}, {max_val}]")
        return float(value)
    return validator


def validate_enum(allowed_values: List[Any]):
    """Enum validator factory"""
    def validator(value: Any) -> Any:
        if value not in allowed_values:
            raise ValueError(f"Value {value} not in allowed values: {allowed_values}")
        return value
    return validator


# Node Decorators for standardization
def node_decorator(func: Callable) -> Callable:
    """Enhanced node function standardization decorator"""
    @functools.wraps(func)
    async def async_wrapper(state: Dict[str, Any], context: NodeContext = None) -> NodeResult:
        return await _execute_node_with_context(func, state, context, is_async=True)
    
    @functools.wraps(func)
    def sync_wrapper(state: Dict[str, Any], context: NodeContext = None) -> NodeResult:
        return _execute_node_with_context(func, state, context, is_async=False)
    
    if asyncio.iscoroutinefunction(func):
        return async_wrapper
    else:
        return sync_wrapper


def router_decorator(func: Callable) -> Callable:
    """Enhanced router function decorator"""
    @functools.wraps(func)
    async def async_wrapper(state: Dict[str, Any], context: NodeContext = None) -> RouterResult:
        if asyncio.iscoroutinefunction(func):
            return await func(state, context)
        else:
            return func(state, context)
    
    @functools.wraps(func)
    def sync_wrapper(state: Dict[str, Any], context: NodeContext = None) -> RouterResult:
        return func(state, context)
    
    if asyncio.iscoroutinefunction(func):
        return async_wrapper
    else:
        return sync_wrapper


async def _execute_node_with_context(func: Callable, state: Dict[str, Any], 
                                    context: NodeContext, is_async: bool) -> NodeResult:
    """Execute node with enhanced context and error handling"""
    start_time = time.time()
    
    try:
        if context:
            context.start_time = datetime.now()
        
        if is_async:
            result = await func(state, context)
        else:
            result = func(state, context)
        
        execution_time = time.time() - start_time
        
        # Handle different return types
        if isinstance(result, NodeResult):
            result.metadata.setdefault("execution_time", execution_time)
            return result
        elif isinstance(result, dict):
            return NodeResult(
                updates=result,
                metadata={"execution_time": execution_time},
                logs=[f"Node executed successfully in {execution_time:.3f}s"]
            )
        else:
            return NodeResult(
                updates={"result": result},
                metadata={"execution_time": execution_time},
                logs=[f"Node executed successfully in {execution_time:.3f}s"]
            )
            
    except Exception as e:
        execution_time = time.time() - start_time
        logger.error(f"Node execution failed: {str(e)}")
        return NodeResult(
            error=str(e),
            metadata={"execution_time": execution_time, "error_type": type(e).__name__},
            logs=[f"Node execution failed after {execution_time:.3f}s: {str(e)}"]
        )


def interrupt(data: Dict[str, Any]) -> Any:
    """Interrupt execution and wait for human input."""
    raise InterruptError(data)


class StateGraph:
    """
    Enhanced StateGraph with LangGraph-inspired features and SpoonOS integration.
    
    Features:
    - TypedDict state management with reducers
    - LLM Manager integration
    - Error handling and recovery
    - Human-in-the-loop patterns
    - Checkpointing and persistence
    - Multi-agent coordination support
    """

    def __init__(self, state_schema: type, checkpointer: InMemoryCheckpointer = None):
        """
        Initialize the enhanced state graph.

        Args:
            state_schema: TypedDict class defining the state structure
            checkpointer: Optional checkpointer for state persistence
        """
        self.state_schema = state_schema
        self.checkpointer = checkpointer or InMemoryCheckpointer()
        self.nodes: Dict[str, Callable] = {}
        self.edges: Dict[str, Union[Callable, Tuple[Callable, Dict[str, str]]]] = {}
        self.parallel_branches: Dict[str, ParallelBranchConfig] = {}
        self.entry_point: Optional[str] = None
        self._compiled = False
        self.interrupts: Dict[str, InterruptError] = {}
        self.llm_manager = get_llm_manager()
        self.monitoring_enabled = False
        self.execution_metrics: Dict[str, Any] = {}
        self.monitored_metrics: List[str] = []

    def add_node(self, name: str, action: Callable, parallel_group: Optional[str] = None) -> "StateGraph":
        """
        Add a node to the graph with optional parallel group assignment.
        
        Args:
            name: Unique identifier for the node
            action: Function or coroutine that processes the state
                   Should accept state dict and return dict of updates or Command
            parallel_group: Optional parallel group name for concurrent execution
        
        Returns:
            Self for method chaining
            
        Raises:
            GraphConfigurationError: If node name already exists or is invalid
        """
        if not name or not isinstance(name, str):
            raise GraphConfigurationError(
                "Node name must be a non-empty string", 
                component="node",
                details={"name": name}
            )
            
        if name in ["START", "END"]:
            raise GraphConfigurationError(
                f"Node name '{name}' is reserved", 
                component="node",
                details={"name": name, "reserved_names": ["START", "END"]}
            )
            
        if name in self.nodes:
            raise GraphConfigurationError(
                f"Node '{name}' already exists in the graph", 
                component="node",
                details={"name": name, "existing_nodes": list(self.nodes.keys())}
            )
            
        if not callable(action):
            raise GraphConfigurationError(
                f"Node action must be callable", 
                component="node",
                details={"name": name, "action_type": type(action)}
            )

        # Wrap function with node decorator if not already wrapped
        if not hasattr(action, '__wrapped__'):
            action = node_decorator(action)

        self.nodes[name] = action
        
        # Handle parallel group assignment
        if parallel_group:
            if parallel_group not in self.parallel_branches:
                self.parallel_branches[parallel_group] = ParallelBranchConfig(
                    branch_name=parallel_group,
                    nodes=[]
                )
            self.parallel_branches[parallel_group].nodes.append(name)
        
        logger.debug(f"Added node '{name}' to graph" + 
                    (f" in parallel group '{parallel_group}'" if parallel_group else ""))
        return self
    
    def add_llm_node(self, 
                     name: str,
                     system_prompt: str,
                     provider: Optional[str] = None,
                     model_params: Optional[Dict[str, Any]] = None) -> "StateGraph":
        """
        Add an LLM-powered node to the graph.
        
        Args:
            name: Unique identifier for the node
            system_prompt: System prompt for the LLM
            provider: Specific LLM provider to use
            model_params: Parameters for the LLM call
            
        Returns:
            Self for method chaining
        """
        async def llm_action(state: Dict[str, Any]) -> Dict[str, Any]:
            messages = [
                Message(role="system", content=system_prompt),
                Message(role="user", content=str(state.get("input", state)))
            ]
            
            params = model_params or {}
            response = await self.llm_manager.chat(messages, provider=provider, **params)
            
            return {
                "llm_response": response.content,
                "llm_metadata": {
                    "provider": response.provider,
                    "model": response.model,
                    "usage": response.usage
                }
            }
        
        return self.add_node(name, llm_action)

    def add_edge(self, start_node: str, end_node: str) -> "StateGraph":
        """
        Add a direct, unconditional edge between two nodes.
        
        Args:
            start_node: Name of the source node (or "START")
            end_node: Name of the destination node (or "END")
        
        Returns:
            Self for method chaining
            
        Raises:
            GraphConfigurationError: If nodes don't exist or edge is invalid
        """
        if start_node != "START" and start_node not in self.nodes:
            raise GraphConfigurationError(
                f"Start node '{start_node}' does not exist", 
                component="edge",
                details={
                    "start_node": start_node, 
                    "end_node": end_node,
                    "available_nodes": list(self.nodes.keys())
                }
            )
            
        if end_node != "END" and end_node not in self.nodes:
            raise GraphConfigurationError(
                f"End node '{end_node}' does not exist", 
                component="edge",
                details={
                    "start_node": start_node, 
                    "end_node": end_node,
                    "available_nodes": list(self.nodes.keys())
                }
            )

        self.edges[start_node] = lambda state: end_node
        logger.debug(f"Added edge from '{start_node}' to '{end_node}'")
        return self
    
    def add_parallel_nodes(self, node_names: List[str], 
                          join_strategy: Literal["all_complete", "first_complete", "timeout"] = "all_complete",
                          timeout: Optional[float] = None) -> "StateGraph":
        """Add multiple nodes for parallel execution"""
        branch_name = f"parallel_branch_{len(self.parallel_branches)}"
        
        config = ParallelBranchConfig(
            branch_name=branch_name,
            nodes=node_names,
            join_strategy=join_strategy,
            timeout=timeout
        )
        
        self.parallel_branches[branch_name] = config
        logger.debug(f"Added parallel branch '{branch_name}' with {len(node_names)} nodes")
        return self
    
    def add_parallel_branch(self, branch_name: str, nodes: List[str],
                           join_condition: Optional[Callable[[List[NodeResult]], bool]] = None,
                           timeout: Optional[float] = None,
                           join_strategy: Literal["all_complete", "first_complete", "timeout"] = "all_complete") -> "StateGraph":
        """Add a named parallel branch with custom join condition"""
        config = ParallelBranchConfig(
            branch_name=branch_name,
            nodes=nodes,
            join_strategy=join_strategy,
            timeout=timeout,
            join_condition=join_condition
        )
        
        self.parallel_branches[branch_name] = config
        logger.debug(f"Added parallel branch '{branch_name}' with custom join condition")
        return self

    def add_conditional_edges(
        self,
        start_node: str,
        condition: Callable[[Dict[str, Any]], str],
        path_map: Dict[str, str]
    ) -> "StateGraph":
        """
        Add conditional edges that route to different nodes based on state.
        
        Args:
            start_node: Name of the source node
            condition: Function that takes state and returns a key from path_map
            path_map: Mapping from condition results to destination node names
        
        Returns:
            Self for method chaining
            
        Raises:
            GraphConfigurationError: If configuration is invalid
        """
        if start_node not in self.nodes:
            raise GraphConfigurationError(
                f"Start node '{start_node}' does not exist", 
                component="conditional_edge",
                details={
                    "start_node": start_node,
                    "available_nodes": list(self.nodes.keys())
                }
            )
            
        if not callable(condition):
            raise GraphConfigurationError(
                "Condition function must be callable", 
                component="conditional_edge",
                details={"start_node": start_node, "condition_type": type(condition)}
            )
            
        if not path_map:
            raise GraphConfigurationError(
                "Path map cannot be empty", 
                component="conditional_edge",
                details={"start_node": start_node}
            )

        # Validate all destination nodes exist
        invalid_nodes = []
        for path_key, dest_node in path_map.items():
            if dest_node != "END" and dest_node not in self.nodes:
                invalid_nodes.append(dest_node)
                
        if invalid_nodes:
            raise GraphConfigurationError(
                f"Destination nodes do not exist: {invalid_nodes}", 
                component="conditional_edge",
                details={
                    "start_node": start_node,
                    "invalid_nodes": invalid_nodes,
                    "available_nodes": list(self.nodes.keys()),
                    "path_map": path_map
                }
            )

        self.edges[start_node] = (condition, path_map)
        logger.debug(f"Added conditional edges from '{start_node}' with {len(path_map)} paths")
        return self
    
    def add_enhanced_conditional_edges(self, start_node: str, 
                                     router_func: Callable[[Dict[str, Any], NodeContext], RouterResult],
                                     fallback_node: str = "END") -> "StateGraph":
        """Add conditional edges with enhanced routing"""
        if start_node not in self.nodes:
            raise GraphConfigurationError(
                f"Start node '{start_node}' does not exist",
                component="enhanced_conditional_edge",
                details={"start_node": start_node, "available_nodes": list(self.nodes.keys())}
            )
        
        # Wrap router function if needed
        if not hasattr(router_func, '__wrapped__'):
            router_func = router_decorator(router_func)
        
        async def enhanced_router(state: Dict[str, Any], context: NodeContext = None) -> str:
            try:
                result = await router_func(state, context)
                if isinstance(result, RouterResult):
                    logger.debug(f"Router decision: {result.next_node} (confidence: {result.confidence:.2f}) - {result.reasoning}")
                    return result.next_node
                else:
                    return result
            except Exception as e:
                logger.error(f"Router function failed: {e}, using fallback: {fallback_node}")
                return fallback_node
        
        self.edges[start_node] = enhanced_router
        return self

    def set_entry_point(self, node_name: str) -> "StateGraph":
        """
        Set the starting node for graph execution.
        
        Args:
            node_name: Name of the node to start execution from
        
        Returns:
            Self for method chaining
            
        Raises:
            GraphConfigurationError: If entry point node doesn't exist
        """
        if node_name not in self.nodes:
            raise GraphConfigurationError(
                f"Entry point node '{node_name}' does not exist", 
                component="entry_point",
                details={
                    "node_name": node_name,
                    "available_nodes": list(self.nodes.keys())
                }
            )

        self.entry_point = node_name
        logger.debug(f"Set entry point to '{node_name}'")
        return self
    
    def enable_monitoring(self, metrics: List[str] = None) -> "StateGraph":
        """Enable real-time monitoring"""
        self.monitoring_enabled = True
        default_metrics = ["execution_time", "node_success_rate", "state_size", "memory_usage"]
        self.monitored_metrics = metrics or default_metrics
        logger.info(f"Enabled monitoring for metrics: {self.monitored_metrics}")
        return self

    def compile(self) -> "CompiledGraph":
        """
        Compile the graph into an executable form.
        
        Returns:
            CompiledGraph instance ready for execution
        
        Raises:
            GraphConfigurationError: If graph configuration is invalid
        """
        # Validate graph configuration
        validation_errors = []
        
        if not self.entry_point:
            validation_errors.append("Graph must have an entry point set before compilation")

        if not self.nodes:
            validation_errors.append("Graph must have at least one node")
            
        # Check for unreachable nodes
        reachable_nodes = set()
        if self.entry_point:
            self._find_reachable_nodes(self.entry_point, reachable_nodes)
            
        unreachable_nodes = set(self.nodes.keys()) - reachable_nodes
        if unreachable_nodes:
            logger.warning(f"Unreachable nodes detected: {unreachable_nodes}")
            
        # Check for nodes without outgoing edges (potential dead ends)
        dead_end_nodes = []
        for node_name in self.nodes.keys():
            if node_name not in self.edges:
                dead_end_nodes.append(node_name)
                
        if dead_end_nodes:
            logger.warning(f"Nodes without outgoing edges (potential dead ends): {dead_end_nodes}")
        
        if validation_errors:
            raise GraphConfigurationError(
                f"Graph compilation failed: {'; '.join(validation_errors)}", 
                component="compilation",
                details={
                    "errors": validation_errors,
                    "nodes": list(self.nodes.keys()),
                    "entry_point": self.entry_point,
                    "unreachable_nodes": list(unreachable_nodes),
                    "dead_end_nodes": dead_end_nodes
                }
            )

        self._compiled = True
        logger.info(f"Compiled enhanced graph with {len(self.nodes)} nodes, "
                   f"{len(self.parallel_branches)} parallel branches, and entry point '{self.entry_point}'")
        return CompiledGraph(self)
    
    def _find_reachable_nodes(self, current_node: str, reachable: set, visited: set = None) -> None:
        """Find all nodes reachable from the current node."""
        if visited is None:
            visited = set()
            
        if current_node in visited or current_node == "END":
            return
            
        visited.add(current_node)
        reachable.add(current_node)
        
        if current_node in self.edges:
            edge = self.edges[current_node]
            
            if isinstance(edge, tuple):
                # Conditional edges
                _, path_map = edge
                for dest_node in path_map.values():
                    if dest_node != "END":
                        self._find_reachable_nodes(dest_node, reachable, visited)
            elif callable(edge):
                # This is more complex for dynamic edges, skip for now
                pass


class CompiledGraph:
    """
    Executable version of a StateGraph with advanced features.
    """

    def __init__(self, graph: StateGraph):
        """Initialize with a compiled StateGraph."""
        self.graph = graph
        self.execution_history: List[Dict[str, Any]] = []
        self.parallel_executor = ThreadPoolExecutor(max_workers=10)

    async def invoke(self, initial_state: Optional[Dict[str, Any]] = None, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Execute the graph from the entry point."""
        config = config or {}
        thread_id = config.get("configurable", {}).get("thread_id", str(uuid.uuid4()))
        
        # Initialize state
        state = self._initialize_state(initial_state)
        
        # Check for resume command
        if isinstance(initial_state, Command) and initial_state.resume is not None:
            return await self._resume_execution(initial_state, config, thread_id)

        current_node = self.graph.entry_point
        execution_path = []
        max_iterations = 100
        iteration = 0

        logger.info(f"Starting graph execution from '{current_node}' (thread: {thread_id})")

        try:
            while current_node and iteration < max_iterations:
                iteration += 1
                execution_path.append(current_node)

                # Create node context
                context = NodeContext(
                    node_name=current_node,
                    iteration=iteration,
                    thread_id=thread_id,
                    execution_path=execution_path.copy()
                )

                logger.debug(f"Executing node '{current_node}' (iteration {iteration})")

                # Save checkpoint before execution
                try:
                    snapshot = StateSnapshot(
                        values=state.copy(),
                        next=(current_node,),
                        config=config,
                        metadata={"iteration": iteration, "node": current_node},
                        created_at=datetime.now()
                    )
                    self.graph.checkpointer.save_checkpoint(thread_id, snapshot)
                except CheckpointError as e:
                    logger.warning(f"Failed to save checkpoint before node execution: {e}")
                    # Continue execution even if checkpoint fails
                except Exception as e:
                    logger.warning(f"Unexpected error saving checkpoint: {e}")
                    # Continue execution even if checkpoint fails

                # Check if this is a parallel branch
                parallel_branch = self._find_parallel_branch(current_node)
                if parallel_branch:
                    result = await self._execute_parallel_branch(parallel_branch, state, context)
                    # Skip to next node after parallel execution
                    current_node = await self._get_next_node(current_node, state, context)
                    continue

                # Execute the current node
                try:
                    result = await self._execute_node_enhanced(current_node, state, context)
                    
                    # Handle NodeResult objects
                    if isinstance(result, NodeResult):
                        if result.error:
                            logger.error(f"Node '{current_node}' failed: {result.error}")
                            # Could implement retry logic here
                            break
                        
                        # Update state
                        if result.updates:
                            self._update_state_with_reducers(state, result.updates)
                        
                        # Handle goto
                        if result.goto:
                            current_node = result.goto
                            if current_node == "END":
                                logger.info("Graph execution completed via NodeResult")
                                break
                            continue
                    
                    # Handle Command objects (backward compatibility)
                    elif isinstance(result, Command):
                        if result.update:
                            self._update_state_with_reducers(state, result.update)
                        if result.goto:
                            current_node = result.goto
                            # Special handling for END
                            if current_node == "END":
                                logger.info("Graph execution completed via Command")
                                break
                            continue
                    elif isinstance(result, dict):
                        self._update_state_with_reducers(state, result)

                except InterruptError as e:
                    # Handle human-in-the-loop interrupts
                    e.node = current_node  # Set the node where interrupt occurred
                    logger.info(f"Graph interrupted at node '{current_node}': {e.interrupt_data}")
                    self.graph.interrupts[e.interrupt_id] = e
                    
                    try:
                        # Save interrupt state
                        interrupt_snapshot = StateSnapshot(
                            values=state.copy(),
                            next=(current_node,),
                            config=config,
                            metadata={
                                "iteration": iteration,
                                "node": current_node,
                                "interrupt_id": e.interrupt_id,
                                "interrupt_data": e.interrupt_data,
                                "status": "interrupted"
                            },
                            created_at=datetime.now()
                        )
                        self.graph.checkpointer.save_checkpoint(thread_id, interrupt_snapshot)
                    except CheckpointError as checkpoint_error:
                        logger.error(f"Failed to save interrupt checkpoint: {checkpoint_error}")
                        # Continue with interrupt handling even if checkpoint fails
                    
                    # Return state with interrupt information
                    return {
                        **state,
                        "__interrupt__": [{
                            "interrupt_id": e.interrupt_id,
                            "value": e.interrupt_data,
                            "node": current_node,
                            "iteration": iteration
                        }]
                    }

                # Record execution step
                self.execution_history.append({
                    "node": current_node,
                    "iteration": iteration,
                    "state_before": state.copy(),
                    "result": result,
                    "timestamp": datetime.now().isoformat()
                })

                # Determine next node
                next_node = await self._get_next_node(current_node, state, context)

                if next_node == current_node:
                    logger.warning(f"Node '{current_node}' routes to itself, stopping execution")
                    break

                current_node = next_node

                # Special handling for END node
                if current_node == "END" or current_node is None:
                    logger.info("Graph execution completed")
                    break

            if iteration >= max_iterations:
                raise GraphExecutionError(
                    f"Graph execution exceeded maximum iterations ({max_iterations})",
                    node=current_node,
                    iteration=iteration,
                    context={
                        "execution_path": execution_path,
                        "thread_id": thread_id,
                        "max_iterations": max_iterations
                    }
                )

            # Save final checkpoint
            try:
                final_snapshot = StateSnapshot(
                    values=state.copy(),
                    next=(),
                    config=config,
                    metadata={
                        "iteration": iteration, 
                        "status": "completed",
                        "execution_path": execution_path
                    },
                    created_at=datetime.now()
                )
                self.graph.checkpointer.save_checkpoint(thread_id, final_snapshot)
            except CheckpointError as e:
                logger.warning(f"Failed to save final checkpoint: {e}")
                # Don't fail the execution for checkpoint errors

            logger.info(f"Graph execution completed in {iteration} steps: {' -> '.join(execution_path)}")
            return state

        except GraphExecutionError:
            # Re-raise graph execution errors as-is
            raise
        except Exception as e:
            logger.error(f"Graph execution failed: {str(e)}")
            raise GraphExecutionError(
                f"Graph execution failed: {str(e)}",
                node=current_node,
                iteration=iteration,
                context={
                    "execution_path": execution_path,
                    "thread_id": thread_id,
                    "original_error": str(e)
                }
            ) from e

    async def stream(self, initial_state: Optional[Dict[str, Any]] = None, config: Optional[Dict[str, Any]] = None, stream_mode: str = "values"):
        """Stream graph execution with different modes."""
        config = config or {}
        thread_id = config.get("configurable", {}).get("thread_id", str(uuid.uuid4()))
        
        state = self._initialize_state(initial_state)
        current_node = self.graph.entry_point
        iteration = 0
        max_iterations = 100

        logger.info(f"Starting graph streaming from '{current_node}' (mode: {stream_mode})")

        try:
            while current_node and iteration < max_iterations:
                iteration += 1
                
                if stream_mode == "debug":
                    yield {
                        "type": "debug",
                        "node": current_node,
                        "iteration": iteration,
                        "state": state.copy()
                    }

                # Execute node
                try:
                    result = await self._execute_node(current_node, state)
                    
                    if isinstance(result, Command):
                        if result.update:
                            old_state = state.copy()
                            self._update_state_with_reducers(state, result.update)
                            
                            if stream_mode == "updates":
                                yield {current_node: result.update}
                            elif stream_mode == "values":
                                yield state.copy()
                        
                        if result.goto:
                            current_node = result.goto
                            continue
                    elif isinstance(result, dict):
                        old_state = state.copy()
                        self._update_state_with_reducers(state, result)
                        
                        if stream_mode == "updates":
                            yield {current_node: result}
                        elif stream_mode == "values":
                            yield state.copy()

                except InterruptError as e:
                    logger.info(f"Graph interrupted during streaming at node '{current_node}'")
                    yield {
                        "type": "interrupt",
                        "node": current_node,
                        "interrupt_id": e.interrupt_id,
                        "interrupt_data": e.interrupt_data,
                        "state": state.copy()
                    }
                    return

                # Determine next node
                next_node = await self._get_next_node(current_node, state)
                
                if next_node == "END" or next_node is None:
                    if stream_mode == "values":
                        yield state.copy()
                    break
                    
                current_node = next_node

        except Exception as e:
            logger.error(f"Graph streaming failed: {str(e)}")
            yield {
                "type": "error",
                "error": str(e),
                "state": state.copy()
            }

    async def _resume_execution(self, command: Command, config: Dict[str, Any], thread_id: str) -> Dict[str, Any]:
        """Resume execution after an interrupt."""
        # Get the latest checkpoint
        latest_checkpoint = self.graph.checkpointer.get_checkpoint(thread_id)
        if not latest_checkpoint:
            raise GraphExecutionError("No checkpoint found for resumption")

        # Check if there's an interrupt to resume
        interrupt_id = latest_checkpoint.metadata.get("interrupt_id")
        if not interrupt_id or interrupt_id not in self.graph.interrupts:
            raise GraphExecutionError("No interrupt found for resumption")

        # Get the interrupt and resume data
        interrupt_error = self.graph.interrupts[interrupt_id]
        resume_data = command.resume

        # Clear the interrupt
        del self.graph.interrupts[interrupt_id]

        # Resume from the interrupted state
        state = latest_checkpoint.values.copy()
        current_node = latest_checkpoint.metadata.get("node")
        
        logger.info(f"Resuming execution at node '{current_node}' with data: {resume_data}")

        # Continue execution with the resume data
        # For simplicity, we'll inject the resume data into the state
        state["__resume_data__"] = resume_data
        
        # Continue normal execution
        return await self.invoke(state, config)

    def _initialize_state(self, initial_state: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Initialize state with schema defaults."""
        state = {}
        
        # Initialize with schema defaults if available
        if hasattr(self.graph.state_schema, '__annotations__'):
            for field_name, field_type in self.graph.state_schema.__annotations__.items():
                # Handle Annotated types (for reducers)
                if hasattr(field_type, '__origin__') and field_type.__origin__ is Annotated:
                    # Initialize with empty list for list types
                    if 'list' in str(field_type) or 'List' in str(field_type):
                        state[field_name] = []
                    else:
                        state[field_name] = None
                else:
                    state[field_name] = None

        if initial_state and not isinstance(initial_state, Command):
            state.update(initial_state)
        
        return state

    async def _execute_node(self, node_name: str, state: Dict[str, Any]) -> Any:
        """Execute a node with error handling."""
        node_action = self.graph.nodes[node_name]
        
        try:
            if asyncio.iscoroutinefunction(node_action):
                result = await node_action(state)
            else:
                result = node_action(state)
            return result
        except InterruptError:
            # Re-raise interrupt errors
            raise
        except Exception as e:
            logger.error(f"Node '{node_name}' execution failed: {str(e)}")
            raise NodeExecutionError(
                f"Node '{node_name}' failed: {str(e)}", 
                node_name=node_name,
                original_error=e,
                state=state
            ) from e

    async def _get_next_node(self, current_node: str, state: Dict[str, Any], context: NodeContext = None) -> Optional[str]:
        """
        Determine the next node to execute based on edges and state.
        
        Args:
            current_node: Current node name
            state: Current state
            
        Returns:
            Next node name or None if execution should stop
            
        Raises:
            EdgeRoutingError: If edge routing fails
            NodeExecutionError: If edge function execution fails
        """
        if current_node not in self.graph.edges:
            return None

        edge = self.graph.edges[current_node]

        # Handle conditional edges
        if isinstance(edge, tuple):
            condition_func, path_map = edge
            try:
                if asyncio.iscoroutinefunction(condition_func):
                    condition_result = await condition_func(state)
                else:
                    condition_result = condition_func(state)

                if condition_result in path_map:
                    next_node = path_map[condition_result]
                    logger.debug(f"Conditional edge from '{current_node}': {condition_result} -> '{next_node}'")
                    return next_node
                else:
                    available_paths = list(path_map.keys())
                    raise EdgeRoutingError(
                        f"Condition result '{condition_result}' not found in path map for node '{current_node}'. Available paths: {available_paths}",
                        source_node=current_node,
                        condition_result=condition_result,
                        available_paths=available_paths
                    )

            except EdgeRoutingError:
                # Re-raise edge routing errors as-is
                raise
            except Exception as e:
                logger.error(f"Condition function failed for node '{current_node}': {str(e)}")
                raise NodeExecutionError(
                    f"Condition function failed: {str(e)}", 
                    node_name=current_node,
                    original_error=e,
                    state=state
                ) from e

        # Handle simple edges
        elif callable(edge):
            try:
                # Check if edge function expects context
                import inspect
                sig = inspect.signature(edge)
                params = list(sig.parameters.keys())
                
                if len(params) >= 2 and 'context' in params:
                    # Edge function expects context
                    if asyncio.iscoroutinefunction(edge):
                        next_node = await edge(state, context)
                    else:
                        next_node = edge(state, context)
                else:
                    # Legacy edge function
                    if asyncio.iscoroutinefunction(edge):
                        next_node = await edge(state)
                    else:
                        next_node = edge(state)

                logger.debug(f"Edge from '{current_node}' -> '{next_node}'")
                return next_node

            except Exception as e:
                logger.error(f"Edge function failed for node '{current_node}': {str(e)}")
                raise NodeExecutionError(
                    f"Edge function failed: {str(e)}", 
                    node_name=current_node,
                    original_error=e,
                    state=state
                ) from e

        return None

    def get_execution_history(self) -> List[Dict[str, Any]]:
        """Get the execution history for debugging and analysis."""
        return self.execution_history.copy()

    def get_state(self, config: Dict[str, Any]) -> Optional[StateSnapshot]:
        """Get the current state snapshot for a thread."""
        thread_id = config.get("configurable", {}).get("thread_id")
        if not thread_id:
            return None
        return self.graph.checkpointer.get_checkpoint(thread_id)

    def _update_state_with_reducers(self, state: Dict[str, Any], updates: Dict[str, Any]) -> None:
        """
        Update state using reducer functions where applicable.
        
        Args:
            state: Current state dictionary
            updates: Updates to apply to the state
            
        Raises:
            StateValidationError: If state validation fails
        """
        try:
            for key, value in updates.items():
                if key in state and hasattr(self.graph.state_schema, '__annotations__'):
                    field_type = self.graph.state_schema.__annotations__.get(key)
                    
                    # Check if this field has a reducer (Annotated type)
                    if hasattr(field_type, '__origin__') and field_type.__origin__ is Annotated:
                        # Get the reducer function from the annotation
                        if len(field_type.__args__) > 1:
                            reducer = field_type.__args__[1]
                            if callable(reducer):
                                try:
                                    # Apply the reducer
                                    state[key] = reducer(state[key], value)
                                    continue
                                except Exception as e:
                                    raise StateValidationError(
                                        f"Reducer function failed for field '{key}': {str(e)}",
                                        field=key,
                                        actual_value=value
                                    ) from e
                
                # Default behavior - direct assignment
                state[key] = value
                
        except StateValidationError:
            # Re-raise state validation errors as-is
            raise
        except Exception as e:
            raise StateValidationError(
                f"State update failed: {str(e)}",
                actual_value=updates
            ) from e
    
    async def _execute_node_enhanced(self, node_name: str, state: Dict[str, Any], context: NodeContext) -> Any:
        """Execute a node with enhanced context and error handling."""
        node_action = self.graph.nodes[node_name]
        
        try:
            # Check if the node function expects context
            import inspect
            sig = inspect.signature(node_action)
            params = list(sig.parameters.keys())
            
            if len(params) >= 2 and 'context' in params:
                # Node expects context
                if asyncio.iscoroutinefunction(node_action):
                    result = await node_action(state, context)
                else:
                    result = node_action(state, context)
            else:
                # Legacy node function
                if asyncio.iscoroutinefunction(node_action):
                    result = await node_action(state)
                else:
                    result = node_action(state)
            
            return result
        except InterruptError:
            # Re-raise interrupt errors
            raise
        except Exception as e:
            logger.error(f"Node '{node_name}' execution failed: {str(e)}")
            raise NodeExecutionError(
                f"Node '{node_name}' failed: {str(e)}", 
                node_name=node_name,
                original_error=e,
                state=state
            ) from e
    
    async def _execute_parallel_branch(self, branch_config: ParallelBranchConfig, 
                                     state: Dict[str, Any], context: NodeContext) -> List[NodeResult]:
        """Execute a parallel branch of nodes"""
        logger.info(f"Executing parallel branch '{branch_config.branch_name}' with {len(branch_config.nodes)} nodes")
        
        # Create tasks for parallel execution
        tasks = []
        for node_name in branch_config.nodes:
            if node_name in self.graph.nodes:
                node_context = NodeContext(
                    node_name=node_name,
                    iteration=context.iteration,
                    thread_id=context.thread_id,
                    parent_context=context,
                    execution_path=context.execution_path + [node_name]
                )
                
                task = asyncio.create_task(self._execute_node_enhanced(node_name, state.copy(), node_context))
                tasks.append((node_name, task))
        
        # Execute based on join strategy
        results = []
        if branch_config.join_strategy == "all_complete":
            # Wait for all tasks to complete
            for node_name, task in tasks:
                try:
                    if branch_config.timeout:
                        result = await asyncio.wait_for(task, timeout=branch_config.timeout)
                    else:
                        result = await task
                    results.append((node_name, result))
                except asyncio.TimeoutError:
                    logger.warning(f"Node '{node_name}' timed out in parallel branch")
                    results.append((node_name, NodeResult(error="Timeout")))
                except Exception as e:
                    logger.error(f"Node '{node_name}' failed in parallel branch: {e}")
                    results.append((node_name, NodeResult(error=str(e))))
        
        elif branch_config.join_strategy == "first_complete":
            # Wait for first completion
            done, pending = await asyncio.wait(
                [task for _, task in tasks], 
                return_when=asyncio.FIRST_COMPLETED,
                timeout=branch_config.timeout
            )
            
            # Cancel pending tasks
            for task in pending:
                task.cancel()
            
            # Get first result
            for node_name, task in tasks:
                if task in done:
                    try:
                        result = await task
                        results.append((node_name, result))
                        break
                    except Exception as e:
                        logger.error(f"First completed node '{node_name}' failed: {e}")
                        results.append((node_name, NodeResult(error=str(e))))
        
        # Apply join condition if specified
        node_results = [result for _, result in results]
        if branch_config.join_condition and not branch_config.join_condition(node_results):
            logger.warning(f"Parallel branch '{branch_config.branch_name}' join condition failed")
        
        # Merge results into state
        for node_name, result in results:
            if isinstance(result, NodeResult) and result.updates and not result.error:
                self._update_state_with_reducers(state, result.updates)
            elif isinstance(result, dict):
                self._update_state_with_reducers(state, result)
        
        logger.info(f"Parallel branch '{branch_config.branch_name}' completed with {len(results)} results")
        return node_results
    
    def _find_parallel_branch(self, node_name: str) -> Optional[ParallelBranchConfig]:
        """Find if a node belongs to a parallel branch"""
        for branch_config in self.graph.parallel_branches.values():
            if node_name in branch_config.nodes:
                return branch_config
        return None
    
    def visualize_execution_path(self) -> str:
        """Generate DOT format diagram for execution path"""
        if not self.execution_history:
            return "digraph empty { }"
        
        dot_content = "digraph execution_path {\n"
        dot_content += "  rankdir=TB;\n"
        dot_content += "  node [shape=box, style=rounded];\n"
        
        for i, step in enumerate(self.execution_history):
            node = step["node"]
            exec_time = step.get("execution_time", 0)
            dot_content += f'  "{node}_{i}" [label="{node}\\n{exec_time:.3f}s"];\n'
            
            if i > 0:
                prev_node = self.execution_history[i-1]["node"]
                dot_content += f'  "{prev_node}_{i-1}" -> "{node}_{i}";\n'
        
        dot_content += "}"
        return dot_content
    
    def get_execution_stats(self) -> Dict[str, Any]:
        """Get execution statistics"""
        if not self.execution_history:
            return {}
        
        total_time = sum(step.get("execution_time", 0) for step in self.execution_history)
        node_counts = {}
        
        for step in self.execution_history:
            node = step["node"]
            node_counts[node] = node_counts.get(node, 0) + 1
        
        return {
            "total_execution_time": total_time,
            "total_steps": len(self.execution_history),
            "average_step_time": total_time / len(self.execution_history) if self.execution_history else 0,
            "node_execution_counts": node_counts,
            "unique_nodes_executed": len(node_counts)
        }


# Example Enhanced State Schema for Cryptocurrency Analysis
class CryptoAnalysisState(TypedDict):
    """Enhanced state schema for cryptocurrency analysis"""
    # Basic data
    query: str
    top_pairs: Annotated[List[Dict[str, Any]], operator.add]
    
    # Market data with custom reducers
    market_data: Annotated[Dict[str, Any], merge_dicts]
    kline_data: Annotated[Dict[str, Any], merge_dicts]
    
    # Analysis results
    trending_coins: Annotated[List[str], operator.add]
    news_data: Annotated[Dict[str, Any], merge_dicts]
    
    # Execution tracking
    execution_history: Annotated[List[Dict[str, Any]], append_history]
    analysis_flags: Annotated[Set[str], union_sets]
    
    # Final results
    investment_advice: str
    confidence_score: Annotated[float, validate_range(0.0, 1.0)]
    risk_level: Annotated[str, validate_enum(["LOW", "MEDIUM", "HIGH"])]