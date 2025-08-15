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


# Enhanced Data Structures
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


@dataclass 
class AgentStateCheckpoint:
    """State checkpoint for GraphAgent recovery"""
    messages: List[Any]
    current_step: int
    agent_state: str
    preserved_state: Optional[Dict[str, Any]]
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)


class InMemoryCheckpointer:
    """
    Simple in-memory checkpointer for development and testing.
    """
    
    def __init__(self, max_checkpoints_per_thread: int = 100):
        self.checkpoints: Dict[str, List[StateSnapshot]] = {}
        self.max_checkpoints_per_thread = max_checkpoints_per_thread
    
    def save_checkpoint(self, thread_id: str, snapshot: StateSnapshot) -> None:
        """Save a checkpoint for a thread."""
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
        """Get a specific checkpoint or the latest one."""
        try:
            if not thread_id:
                raise CheckpointError("Thread ID cannot be empty", operation="get")
                
            if thread_id not in self.checkpoints:
                return None
            
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
                operation="get"
            ) from e
    
    def list_checkpoints(self, thread_id: str) -> List[StateSnapshot]:
        """List all checkpoints for a thread."""
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
        """Clear all checkpoints for a thread."""
        if thread_id in self.checkpoints:
            del self.checkpoints[thread_id]
            logger.debug(f"Cleared all checkpoints for thread {thread_id}")


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


# Node Decorators
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
    """Enhanced StateGraph with LangGraph-inspired features and SpoonOS integration."""

    def __init__(self, state_schema: type, checkpointer: InMemoryCheckpointer = None):
        self.state_schema = state_schema
        self.checkpointer = checkpointer or InMemoryCheckpointer()
        self.nodes: Dict[str, Callable] = {}
        self.edges: Dict[str, Union[Callable, Tuple[Callable, Dict[str, str]]]] = {}
        self.parallel_branches: Dict[str, ParallelBranchConfig] = {}
        self.entry_point: Optional[str] = None
        self._compiled = False
        self.interrupts: Dict[str, InterruptError] = {}
        self.llm_manager = get_llm_manager()

    def add_node(self, name: str, action: Callable, parallel_group: Optional[str] = None) -> "StateGraph":
        """Add a node to the graph with optional parallel group assignment."""
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

        if not hasattr(action, '__wrapped__'):
            action = node_decorator(action)

        self.nodes[name] = action
        
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
        """Add an LLM-powered node to the graph."""
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
        """Add a direct, unconditional edge between two nodes."""
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

    def add_conditional_edges(
        self,
        start_node: str,
        condition: Callable[[Dict[str, Any]], str],
        path_map: Dict[str, str]
    ) -> "StateGraph":
        """Add conditional edges that route to different nodes based on state."""
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

    def set_entry_point(self, node_name: str) -> "StateGraph":
        """Set the starting node for graph execution."""
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

    def compile(self) -> "CompiledGraph":
        """Compile the graph into an executable form."""
        validation_errors = []
        
        if not self.entry_point:
            validation_errors.append("Graph must have an entry point set before compilation")

        if not self.nodes:
            validation_errors.append("Graph must have at least one node")
            
        reachable_nodes = set()
        if self.entry_point:
            self._find_reachable_nodes(self.entry_point, reachable_nodes)
            
        unreachable_nodes = set(self.nodes.keys()) - reachable_nodes
        if unreachable_nodes:
            logger.warning(f"Unreachable nodes detected: {unreachable_nodes}")
            
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
        logger.info(f"Compiled graph with {len(self.nodes)} nodes and entry point '{self.entry_point}'")
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
                _, path_map = edge
                for dest_node in path_map.values():
                    if dest_node != "END":
                        self._find_reachable_nodes(dest_node, reachable, visited)


class GraphAgent:
    """
    Enhanced GraphAgent with atomic state management and error recovery.
    
    Fixes critical state corruption issues:
    - Atomic memory restoration with validation
    - Bounded error metadata to prevent memory leaks  
    - Proper synchronization for concurrent access
    - Safe preserved state handling with corruption detection
    """
    
    def __init__(self, name: str, graph: StateGraph, preserve_state: bool = False, **kwargs):
        self.name = name
        self.graph = graph.compile()
        self.preserve_state = preserve_state
        self.memory = kwargs.get('memory')  # Assume memory object exists
        self.current_step = 0
        self.state = "IDLE"
        
        # Enhanced state management
        self._state_lock = asyncio.Lock()
        self._max_metadata_size = kwargs.get('max_metadata_size', 1024)
        self._last_state = None
        self.execution_metadata = {}
        
        # Initialize with safe defaults
        if self.memory is None:
            self.memory = MockMemory()  # Fallback for testing

    async def run(self, request: Optional[str] = None) -> str:
        """Run the agent with atomic state management and error recovery."""
        async with self._state_lock:  # Synchronize all state operations
            # Create checkpoint before execution
            checkpoint = self._create_checkpoint()
            
            try:
                # Execute graph
                initial_state = {"input": request} if request else {}
                if self.preserve_state and self._last_state:
                    # Validate preserved state before use
                    if self._validate_preserved_state(self._last_state):
                        initial_state.update(self._last_state)
                    else:
                        logger.warning(f"GraphAgent '{self.name}' cleared corrupted preserved state")
                        self._last_state = None
                
                result = await self.graph.invoke(initial_state)
                
                # Update preserved state safely
                if self.preserve_state:
                    self._last_state = self._sanitize_preserved_state(result)
                
                # Update execution metadata with bounds
                self.execution_metadata = {
                    "execution_successful": True,
                    "execution_time": time.time(),
                    "last_request": request[:100] if request else None  # Truncate long requests
                }
                
                return str(result.get("output", result))
                
            except Exception as error:
                # Atomic error recovery
                await self._handle_execution_error(error, checkpoint)
                raise
    
    def _create_checkpoint(self) -> AgentStateCheckpoint:
        """Create a state checkpoint for recovery."""
        messages = []
        if self.memory and hasattr(self.memory, 'get_messages'):
            try:
                messages = [msg for msg in self.memory.get_messages() if self._validate_message(msg)]
            except Exception as e:
                logger.warning(f"Failed to create message checkpoint: {e}")
                messages = []
        
        return AgentStateCheckpoint(
            messages=messages,
            current_step=self.current_step,
            agent_state=self.state,
            preserved_state=self._last_state.copy() if self._last_state else None
        )
    
    async def _handle_execution_error(self, error: Exception, checkpoint: AgentStateCheckpoint):
        """Enhanced error handling with atomic rollback."""
        try:
            # Atomic restoration with validation
            self._restore_from_checkpoint(checkpoint)
            
            # Bounded error metadata to prevent memory leaks
            error_str = str(error)[:500]  # Truncate long errors
            self.execution_metadata = {
                "error": error_str,
                "error_type": type(error).__name__,
                "execution_successful": False,
                "execution_time": time.time(),
                "recovery_attempted": True
            }
            
            logger.error(f"GraphAgent '{self.name}' execution failed: {error_str}")
            
        except Exception as restore_error:
            logger.critical(f"State corruption detected in {self.name}: {restore_error}")
            # Emergency reset to safe state
            self._emergency_reset()
            
        # Clear any corrupted preserved state
        self._safe_clear_preserved_state()
    
    def _restore_from_checkpoint(self, checkpoint: AgentStateCheckpoint):
        """Atomically restore agent state from checkpoint."""
        if not self.memory:
            return
            
        try:
            # Clear and validate messages
            self.memory.clear()
            
            valid_messages = []
            for msg in checkpoint.messages:
                if self._validate_message(msg):
                    valid_messages.append(msg)
                else:
                    logger.warning(f"Skipping corrupted message during restoration: {type(msg)}")
            
            # Batch restore all valid messages
            for msg in valid_messages:
                self.memory.add_message(msg)
            
            # Restore other state atomically
            self.current_step = checkpoint.current_step
            self.state = checkpoint.agent_state
            
            # Safely restore preserved state
            if checkpoint.preserved_state and self._validate_preserved_state(checkpoint.preserved_state):
                self._last_state = checkpoint.preserved_state.copy()
            else:
                self._last_state = None
                
            logger.info(f"GraphAgent '{self.name}' state restored from checkpoint")
            
        except Exception as e:
            logger.error(f"Checkpoint restoration failed: {e}")
            self._emergency_reset()
    
    def _validate_message(self, msg) -> bool:
        """Validate message integrity before restoration."""
        try:
            if not hasattr(msg, 'role') or not hasattr(msg, 'content'):
                return False
            if not hasattr(msg, 'role') or msg.role not in ['user', 'assistant', 'tool', 'system']:
                return False
            if not isinstance(msg.content, (str, type(None))):
                return False
            return True
        except Exception:
            return False
    
    def _validate_preserved_state(self, state: Any) -> bool:
        """Validate preserved state integrity."""
        try:
            if not isinstance(state, dict):
                return False
            # Check for reasonable size limits
            if len(str(state)) > 10000:  # Reasonable size limit
                return False
            # Check for basic structure
            return True
        except Exception:
            return False
    
    def _sanitize_preserved_state(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Sanitize state data for preservation."""
        try:
            if not isinstance(state, dict):
                return {}
                
            # Remove system keys and large objects
            sanitized = {}
            for key, value in state.items():
                if key.startswith('__'):  # Skip system keys
                    continue
                if isinstance(value, (str, int, float, bool, type(None))):
                    sanitized[key] = value
                elif isinstance(value, (list, dict)):
                    # Limit size of complex objects
                    if len(str(value)) <= 1000:
                        sanitized[key] = value
            
            return sanitized
        except Exception:
            return {}
    
    def _safe_clear_preserved_state(self):
        """Safely clear preserved state if corrupted."""
        try:
            if self._last_state and not self._validate_preserved_state(self._last_state):
                logger.warning(f"GraphAgent '{self.name}' cleared corrupted preserved state")
                self._last_state = None
        except Exception as e:
            logger.error(f"Error clearing preserved state: {e}")
            self._last_state = None
    
    def _emergency_reset(self):
        """Emergency reset to safe state when corruption is detected."""
        try:
            if self.memory and hasattr(self.memory, 'clear'):
                self.memory.clear()
            self.current_step = 0
            self.state = "IDLE"
            self._last_state = None
            self.execution_metadata = {"emergency_reset": True, "timestamp": time.time()}
            logger.warning(f"GraphAgent '{self.name}' performed emergency state reset")
        except Exception as e:
            logger.critical(f"Emergency reset failed for {self.name}: {e}")


class MockMemory:
    """Mock memory for testing and fallback."""
    def __init__(self):
        self.messages = []
    
    def clear(self):
        self.messages = []
    
    def add_message(self, msg):
        self.messages.append(msg)
    
    def get_messages(self):
        return self.messages


class CompiledGraph:
    """Executable version of a StateGraph with advanced features."""

    def __init__(self, graph: StateGraph):
        self.graph = graph
        self.execution_history: List[Dict[str, Any]] = []
        self.parallel_executor = ThreadPoolExecutor(max_workers=10)

    async def invoke(self, initial_state: Optional[Dict[str, Any]] = None, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Execute the graph from the entry point."""
        config = config or {}
        thread_id = config.get("configurable", {}).get("thread_id", str(uuid.uuid4()))
        
        state = self._initialize_state(initial_state)
        
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

                context = NodeContext(
                    node_name=current_node,
                    iteration=iteration,
                    thread_id=thread_id,
                    execution_path=execution_path.copy()
                )

                logger.debug(f"Executing node '{current_node}' (iteration {iteration})")

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
                    logger.warning(f"Failed to save checkpoint: {e}")

                try:
                    result = await self._execute_node_enhanced(current_node, state, context)
                    
                    if isinstance(result, NodeResult):
                        if result.error:
                            logger.error(f"Node '{current_node}' failed: {result.error}")
                            break
                        
                        if result.updates:
                            self._update_state_with_reducers(state, result.updates)
                        
                        if result.goto:
                            current_node = result.goto
                            if current_node == "END":
                                logger.info("Graph execution completed via NodeResult")
                                break
                            continue
                    
                    elif isinstance(result, Command):
                        if result.update:
                            self._update_state_with_reducers(state, result.update)
                        if result.goto:
                            current_node = result.goto
                            if current_node == "END":
                                logger.info("Graph execution completed via Command")
                                break
                            continue
                    elif isinstance(result, dict):
                        self._update_state_with_reducers(state, result)

                except InterruptError as e:
                    e.node = current_node
                    logger.info(f"Graph interrupted at node '{current_node}': {e.interrupt_data}")
                    self.graph.interrupts[e.interrupt_id] = e
                    
                    try:
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
                    
                    return {
                        **state,
                        "__interrupt__": [{
                            "interrupt_id": e.interrupt_id,
                            "value": e.interrupt_data,
                            "node": current_node,
                            "iteration": iteration
                        }]
                    }

                self.execution_history.append({
                    "node": current_node,
                    "iteration": iteration,
                    "state_before": state.copy(),
                    "result": result,
                    "timestamp": datetime.now().isoformat()
                })

                next_node = await self._get_next_node(current_node, state, context)

                if next_node == current_node:
                    logger.warning(f"Node '{current_node}' routes to itself, stopping execution")
                    break

                current_node = next_node

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

            logger.info(f"Graph execution completed in {iteration} steps: {' -> '.join(execution_path)}")
            return state

        except GraphExecutionError:
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
        latest_checkpoint = self.graph.checkpointer.get_checkpoint(thread_id)
        if not latest_checkpoint:
            raise GraphExecutionError("No checkpoint found for resumption")

        interrupt_id = latest_checkpoint.metadata.get("interrupt_id")
        if not interrupt_id or interrupt_id not in self.graph.interrupts:
            raise GraphExecutionError("No interrupt found for resumption")

        interrupt_error = self.graph.interrupts[interrupt_id]
        resume_data = command.resume

        del self.graph.interrupts[interrupt_id]

        state = latest_checkpoint.values.copy()
        current_node = latest_checkpoint.metadata.get("node")
        
        logger.info(f"Resuming execution at node '{current_node}' with data: {resume_data}")

        state["__resume_data__"] = resume_data
        
        return await self.invoke(state, config)

    def _initialize_state(self, initial_state: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Initialize state with schema defaults."""
        state = {}
        
        if hasattr(self.graph.state_schema, '__annotations__'):
            for field_name, field_type in self.graph.state_schema.__annotations__.items():
                if hasattr(field_type, '__origin__') and field_type.__origin__ is Annotated:
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
        """Determine the next node to execute based on edges and state."""
        if current_node not in self.graph.edges:
            return None

        edge = self.graph.edges[current_node]

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
                raise
            except Exception as e:
                logger.error(f"Condition function failed for node '{current_node}': {str(e)}")
                raise NodeExecutionError(
                    f"Condition function failed: {str(e)}", 
                    node_name=current_node,
                    original_error=e,
                    state=state
                ) from e

        elif callable(edge):
            try:
                import inspect
                sig = inspect.signature(edge)
                params = list(sig.parameters.keys())
                
                if len(params) >= 2 and 'context' in params:
                    if asyncio.iscoroutinefunction(edge):
                        next_node = await edge(state, context)
                    else:
                        next_node = edge(state, context)
                else:
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

    def _update_state_with_reducers(self, state: Dict[str, Any], updates: Dict[str, Any]) -> None:
        """Update state using reducer functions where applicable."""
        try:
            for key, value in updates.items():
                if key in state and hasattr(self.graph.state_schema, '__annotations__'):
                    field_type = self.graph.state_schema.__annotations__.get(key)
                    
                    if hasattr(field_type, '__origin__') and field_type.__origin__ is Annotated:
                        if len(field_type.__args__) > 1:
                            reducer = field_type.__args__[1]
                            if callable(reducer):
                                try:
                                    state[key] = reducer(state[key], value)
                                    continue
                                except Exception as e:
                                    raise StateValidationError(
                                        f"Reducer function failed for field '{key}': {str(e)}",
                                        field=key,
                                        actual_value=value
                                    ) from e
                
                state[key] = value
                
        except StateValidationError:
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
            import inspect
            sig = inspect.signature(node_action)
            params = list(sig.parameters.keys())
            
            if len(params) >= 2 and 'context' in params:
                if asyncio.iscoroutinefunction(node_action):
                    result = await node_action(state, context)
                else:
                    result = node_action(state, context)
            else:
                if asyncio.iscoroutinefunction(node_action):
                    result = await node_action(state)
                else:
                    result = node_action(state)
            
            return result
        except InterruptError:
            raise
        except Exception as e:
            logger.error(f"Node '{node_name}' execution failed: {str(e)}")
            raise NodeExecutionError(
                f"Node '{node_name}' failed: {str(e)}", 
                node_name=node_name,
                original_error=e,
                state=state
            ) from e

    def get_execution_history(self) -> List[Dict[str, Any]]:
        """Get the execution history for debugging and analysis."""
        return self.execution_history.copy()

    def get_state(self, config: Dict[str, Any]) -> Optional[StateSnapshot]:
        """Get the current state snapshot for a thread."""
        thread_id = config.get("configurable", {}).get("thread_id")
        if not thread_id:
            return None
        return self.graph.checkpointer.get_checkpoint(thread_id)


# Example Enhanced State Schema for Cryptocurrency Analysis
class CryptoAnalysisState(TypedDict):
    """Enhanced state schema for cryptocurrency analysis"""
    query: str
    top_pairs: Annotated[List[Dict[str, Any]], operator.add]
    market_data: Annotated[Dict[str, Any], merge_dicts]
    kline_data: Annotated[Dict[str, Any], merge_dicts]
    trending_coins: Annotated[List[str], operator.add]
    news_data: Annotated[Dict[str, Any], merge_dicts]
    execution_history: Annotated[List[Dict[str, Any]], append_history]
    analysis_flags: Annotated[Set[str], union_sets]
    investment_advice: str
    confidence_score: Annotated[float, validate_range(0.0, 1.0)]
    risk_level: Annotated[str, validate_enum(["LOW", "MEDIUM", "HIGH"])]
