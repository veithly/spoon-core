"""
Graph-based agent implementation for SpoonOS.

This module provides the GraphAgent class that executes StateGraph workflows,
integrating the graph execution system with the existing agent architecture.
"""

import logging
import asyncio
import time
from typing import Optional, Dict, Any
from pydantic import Field, validator

from spoon_ai.agents.base import BaseAgent
from spoon_ai.graph import StateGraph, CompiledGraph, GraphExecutionError
from spoon_ai.schema import AgentState

logger = logging.getLogger(__name__)


class GraphAgent(BaseAgent):
    """
    An agent that executes StateGraph workflows.

    This agent provides a bridge between the existing SpoonOS agent architecture
    and the new graph-based execution system. It allows complex, stateful workflows
    to be defined as graphs and executed with proper state management.

    Key Features:
    - Executes StateGraph workflows
    - Maintains compatibility with existing agent interfaces
    - Provides detailed execution logging and error handling
    - Supports both sync and async node functions
    - Atomic state management with corruption prevention
    - Bounded metadata storage with sensitive data protection
    """

    graph: StateGraph = Field(..., description="The StateGraph to execute")
    compiled_graph: Optional[CompiledGraph] = Field(None, description="Compiled graph instance")
    initial_state: Dict[str, Any] = Field(default_factory=dict, description="Initial state for graph execution")
    preserve_state: bool = Field(default=False, description="Whether to preserve state between runs")
    execution_metadata: Dict[str, Any] = Field(default_factory=dict, description="Metadata from last execution")
    max_metadata_size: int = Field(default=1024, description="Maximum size for execution metadata")
    max_error_length: int = Field(default=500, description="Maximum length for error messages in metadata")

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, **kwargs):
        """
        Initialize the GraphAgent.

        Args:
            graph: StateGraph instance to execute
            **kwargs: Additional arguments passed to BaseAgent

        Raises:
            ValueError: If no graph is provided
        """
        if "graph" not in kwargs:
            raise ValueError("A StateGraph instance must be provided via 'graph' parameter")

        super().__init__(**kwargs)
        
        # Initialize state synchronization lock
        self._state_lock = asyncio.Lock()
        self._recovery_attempted = False

        # Compile the graph during initialization
        try:
            self.compiled_graph = self.graph.compile()
            logger.info(f"GraphAgent '{self.name}' initialized with compiled graph")
        except Exception as e:
            logger.error(f"Failed to compile graph for agent '{self.name}': {str(e)}")
            raise ValueError(f"Graph compilation failed: {str(e)}") from e

    @validator('graph')
    def validate_graph(cls, v):
        """Validate that the provided graph is a StateGraph instance."""
        if not isinstance(v, StateGraph):
            raise ValueError("graph must be a StateGraph instance")
        return v

    @validator('max_metadata_size')
    def validate_max_metadata_size(cls, v):
        """Ensure metadata size limit is reasonable."""
        if v < 256 or v > 10240:
            raise ValueError("max_metadata_size must be between 256 and 10240 bytes")
        return v

    @validator('max_error_length')
    def validate_max_error_length(cls, v):
        """Ensure error length limit is reasonable."""
        if v < 100 or v > 2048:
            raise ValueError("max_error_length must be between 100 and 2048 characters")
        return v

    async def run(self, request: Optional[str] = None) -> str:
        """Execute the graph workflow with robust error handling and state management."""
        # Synchronize all state operations to prevent race conditions
        async with self._state_lock:
            if self.state != AgentState.IDLE:
                raise RuntimeError(f"Agent {self.name} is not in the IDLE state (currently: {self.state})")

            logger.info(f"GraphAgent '{self.name}' starting execution")
            
            # Create state checkpoint for atomic recovery
            checkpoint = self._create_checkpoint()
            
            try:
                # Prepare initial state
                execution_state = self._prepare_execution_state(request)
                
                async with self.state_context(AgentState.RUNNING):
                    # Execute the compiled graph with timeout
                    try:
                        final_state = await asyncio.wait_for(
                            self.compiled_graph.invoke(execution_state), 
                            timeout=300  # 5 minute timeout
                        )
                    except asyncio.TimeoutError:
                        logger.error(f"GraphAgent '{self.name}' execution timed out")
                        raise RuntimeError("Graph execution timed out after 5 minutes")
                    
                    # Validate final state
                    if not self._validate_final_state(final_state):
                        logger.error(f"GraphAgent '{self.name}' produced invalid final state")
                        raise RuntimeError("Graph execution produced invalid final state")
                    
                    # Store execution metadata with bounds checking
                    self._store_execution_metadata(final_state, success=True)

                    # Preserve state if enabled and valid
                    if self.preserve_state:
                        sanitized_state = self._sanitize_preserved_state(final_state)
                        if self._validate_preserved_state(sanitized_state):
                            self._last_state = sanitized_state
                        else:
                            logger.warning(f"GraphAgent '{self.name}' skipped preserving corrupted state")
                            self._safe_clear_preserved_state()

                    # Extract output with validation
                    output = self._extract_output(final_state)
                    
                    # Validate output before adding to memory
                    if not output or not isinstance(output, str):
                        logger.warning(f"GraphAgent '{self.name}' produced invalid output: {type(output)}")
                        output = "Graph execution completed but produced invalid output"

                    # Add assistant response to memory
                    self.add_message("assistant", output)

                    logger.info(f"GraphAgent '{self.name}' execution completed successfully")
                    return output

            except GraphExecutionError as e:
                logger.error(f"Graph execution failed for agent '{self.name}': {str(e)}")
                await self._handle_execution_error(e, checkpoint)
                raise RuntimeError(f"Graph execution failed: {str(e)}") from e

            except Exception as e:
                logger.error(f"Unexpected error during graph execution for agent '{self.name}': {str(e)}")
                await self._handle_execution_error(e, checkpoint)
                raise RuntimeError(f"Graph agent execution failed: {str(e)}") from e

    def _create_checkpoint(self) -> Dict[str, Any]:
        """Create a state checkpoint for atomic recovery."""
        try:
            # Create deep copy of messages to prevent reference corruption
            original_messages = []
            for msg in self.memory.get_messages():
                if self._validate_message(msg):
                    original_messages.append(msg)
                else:
                    logger.warning(f"Skipping invalid message during checkpoint: {type(msg)}")
            
            checkpoint = {
                'messages': original_messages,
                'current_step': getattr(self, 'current_step', 0),
                'state': self.state,
                'preserve_state_backup': getattr(self, '_last_state', None),
                'timestamp': time.time()
            }
            
            return checkpoint
            
        except Exception as e:
            logger.error(f"Failed to create checkpoint for agent '{self.name}': {e}")
            # Return minimal safe checkpoint
            return {
                'messages': [],
                'current_step': 0,
                'state': AgentState.IDLE,
                'preserve_state_backup': None,
                'timestamp': time.time()
            }

    def _validate_message(self, msg) -> bool:
        """Validate message integrity before restoration."""
        try:
            if not hasattr(msg, 'role') or not hasattr(msg, 'content'):
                return False
            if msg.role not in ['user', 'assistant', 'tool', 'system']:
                return False
            if not isinstance(msg.content, (str, type(None))):
                return False
            # Additional validation for tool messages
            if msg.role == 'tool':
                if not hasattr(msg, 'tool_call_id'):
                    return False
            return True
        except Exception:
            return False

    def _validate_preserved_state(self, state: Any) -> bool:
        """Validate preserved state integrity."""
        try:
            if not isinstance(state, dict):
                return False
            if not state:  # Empty dict is valid
                return True
            # Check for basic corruption indicators
            for key, value in state.items():
                if not isinstance(key, str):
                    return False
                # Reject obviously corrupted values
                if isinstance(value, type) or callable(value):
                    return False
            return True
        except Exception:
            return False

    async def _handle_execution_error(self, error: Exception, checkpoint: Dict[str, Any]):
        """Enhanced error handling with atomic rollback."""
        try:
            logger.info(f"GraphAgent '{self.name}' beginning error recovery")
            
            # Atomic restoration with validation
            await self._restore_from_checkpoint(checkpoint)
            
            # Store bounded error metadata
            self._store_execution_metadata(None, success=False, error=error)
            
            # Mark recovery attempt
            self._recovery_attempted = True
            
            logger.info(f"GraphAgent '{self.name}' error recovery completed")
            
        except Exception as restore_error:
            logger.critical(f"State corruption detected in {self.name}: {restore_error}")
            # Emergency reset to safe state
            await self._emergency_reset()

    async def _restore_from_checkpoint(self, checkpoint: Dict[str, Any]):
        """Atomically restore agent state from checkpoint."""
        try:
            # Clear current memory
            self.memory.clear()
            
            # Validate and restore messages
            valid_messages = []
            for msg in checkpoint.get('messages', []):
                if self._validate_message(msg):
                    valid_messages.append(msg)
                else:
                    logger.warning(f"Skipping corrupted message during restoration: {type(msg)}")
            
            # Batch restore all valid messages
            for msg in valid_messages:
                self.memory.add_message(msg)
            
            # Restore other state components
            if hasattr(self, 'current_step'):
                self.current_step = checkpoint.get('current_step', 0)
            
            # Restore preserved state backup if valid
            preserved_backup = checkpoint.get('preserve_state_backup')
            if preserved_backup and self._validate_preserved_state(preserved_backup):
                self._last_state = preserved_backup
            else:
                self._safe_clear_preserved_state()
                
            logger.debug(f"GraphAgent '{self.name}' state restored from checkpoint")
            
        except Exception as e:
            logger.error(f"Failed to restore from checkpoint: {e}")
            raise RuntimeError(f"Checkpoint restoration failed: {e}") from e

    async def _emergency_reset(self):
        """Emergency reset to safe state when corruption is detected."""
        try:
            logger.warning(f"GraphAgent '{self.name}' performing emergency reset")
            
            # Clear all potentially corrupted state
            self.memory.clear()
            self._safe_clear_preserved_state()
            
            # Reset to safe defaults
            if hasattr(self, 'current_step'):
                self.current_step = 0
            
            # Store emergency reset metadata
            self.execution_metadata = {
                "emergency_reset": True,
                "reset_time": time.time(),
                "execution_successful": False,
                "error": "Emergency reset due to state corruption"
            }
            
            logger.warning(f"GraphAgent '{self.name}' emergency reset completed")
            
        except Exception as e:
            logger.critical(f"Emergency reset failed for {self.name}: {e}")
            # At this point, the agent may be in an undefined state

    def _safe_clear_preserved_state(self):
        """Safely clear preserved state."""
        try:
            if hasattr(self, '_last_state'):
                delattr(self, '_last_state')
        except Exception as e:
            logger.warning(f"Failed to clear preserved state: {e}")

    def _store_execution_metadata(self, final_state: Optional[Dict[str, Any]], 
                                success: bool, error: Optional[Exception] = None):
        """Store execution metadata with size bounds and sensitive data protection."""
        try:
            metadata = {
                "execution_successful": success,
                "execution_time": time.time(),
                "recovery_attempted": getattr(self, '_recovery_attempted', False)
            }
            
            if success and final_state:
                # Store successful execution metadata
                if self.compiled_graph:
                    history = self.compiled_graph.get_execution_history()
                    # Truncate history if too large
                    if len(str(history)) > self.max_metadata_size // 2:
                        metadata["execution_history"] = "Truncated due to size limit"
                        metadata["execution_steps_count"] = len(history)
                    else:
                        metadata["execution_history"] = history
                
                metadata["final_state_keys"] = list(final_state.keys())[:10]  # Limit keys
                
            elif error:
                # Store error metadata with bounds
                error_str = str(error)[:self.max_error_length]
                metadata.update({
                    "error": error_str,
                    "error_type": type(error).__name__
                })
            
            # Ensure total metadata size is bounded
            metadata_str = str(metadata)
            if len(metadata_str) > self.max_metadata_size:
                # Truncate less critical fields
                if "execution_history" in metadata:
                    metadata["execution_history"] = "Truncated due to size limit"
                if "error" in metadata:
                    metadata["error"] = metadata["error"][:self.max_error_length // 2]
            
            self.execution_metadata = metadata
            
        except Exception as e:
            logger.warning(f"Failed to store execution metadata: {e}")
            # Store minimal safe metadata
            self.execution_metadata = {
                "execution_successful": success,
                "execution_time": time.time(),
                "metadata_error": "Failed to store full metadata"
            }

    def _prepare_execution_state(self, request: Optional[str]) -> Dict[str, Any]:
        """Prepare initial execution state with validation."""
        execution_state = {}

        # Start with preserved state if enabled and valid
        if self.preserve_state and hasattr(self, '_last_state'):
            if self._validate_preserved_state(self._last_state):
                execution_state.update(self._last_state)
            else:
                logger.warning(f"GraphAgent '{self.name}' discarding invalid preserved state")
                self._safe_clear_preserved_state()

        # Add configured initial state with validation
        if isinstance(self.initial_state, dict):
            # Filter out potentially dangerous values
            safe_initial_state = {}
            for k, v in self.initial_state.items():
                if isinstance(k, str) and not callable(v) and not isinstance(v, type):
                    safe_initial_state[k] = v
            execution_state.update(safe_initial_state)

        # Add request to state if provided
        if request is not None:
            execution_state["input"] = request
            execution_state["request"] = request
            self.add_message("user", request)

        # Add messages with validation
        messages = []
        for msg in self.memory.get_messages():
            if self._validate_message(msg):
                msg_dict = {
                    "role": msg.role,
                    "content": msg.content,
                }
                # Add optional fields if present
                if hasattr(msg, 'tool_calls') and msg.tool_calls:
                    msg_dict["tool_calls"] = msg.tool_calls
                if hasattr(msg, 'tool_call_id') and msg.tool_call_id:
                    msg_dict["tool_call_id"] = msg.tool_call_id
                messages.append(msg_dict)
        
        execution_state["messages"] = messages
        execution_state["agent_name"] = self.name
        execution_state["agent_description"] = self.description
        execution_state["system_prompt"] = self.system_prompt

        return execution_state

    def _validate_final_state(self, final_state: Any) -> bool:
        """Validate that final state is properly formed."""
        try:
            if not isinstance(final_state, dict):
                return False
            
            # Should have at least some content
            if not final_state:
                return False
            
            # Check for obvious corruption indicators
            for key, value in final_state.items():
                if not isinstance(key, str):
                    return False
                # Reject function objects or types
                if callable(value) or isinstance(value, type):
                    return False
                    
            return True
        except Exception:
            return False

    def _sanitize_preserved_state(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Sanitize state for preservation, removing sensitive data."""
        try:
            sanitized = {}
            
            # List of sensitive patterns
            sensitive_patterns = [
                'api_key', 'private_key', 'password', 'token', 'secret', 
                'auth', 'credential', 'session_id', 'cookie'
            ]
            
            for key, value in state.items():
                # Skip non-string keys
                if not isinstance(key, str):
                    continue
                    
                # Skip sensitive keys
                key_lower = key.lower()
                if any(pattern in key_lower for pattern in sensitive_patterns):
                    continue
                
                # Skip callable values or types
                if callable(value) or isinstance(value, type):
                    continue
                
                # Limit size of individual values
                if isinstance(value, str) and len(value) > 1000:
                    sanitized[key] = value[:1000] + "... (truncated)"
                elif isinstance(value, (dict, list)) and len(str(value)) > 1000:
                    sanitized[key] = str(value)[:1000] + "... (truncated)"
                else:
                    sanitized[key] = value
            
            return sanitized
            
        except Exception as e:
            logger.warning(f"Failed to sanitize preserved state: {e}")
            return {}

    async def step(self) -> str:
        """
        Step method for compatibility with BaseAgent.

        Since GraphAgent uses graph execution instead of step-based execution,
        this method is not used in normal operation but is required by the
        BaseAgent interface.

        Returns:
            Status message indicating graph-based execution
        """
        return "GraphAgent uses graph-based execution, not step-based execution"

    def _extract_output(self, final_state: Dict[str, Any]) -> str:
        """Extract the output message from the final graph state with robust validation."""
        if not isinstance(final_state, dict):
            logger.error(f"GraphAgent '{self.name}' received invalid final_state type: {type(final_state)}")
            return "Error: Invalid graph execution result"
        
        # Try different common output keys with validation
        output_keys = ["output", "result", "response", "answer", "final_output", "content"]
        
        for key in output_keys:
            if key in final_state:
                output = final_state[key]
                if output is not None:  # Allow empty strings but not None
                    if isinstance(output, str):
                        return output if output.strip() else f"Graph completed (empty {key})"
                    elif isinstance(output, (dict, list)):
                        try:
                            import json
                            json_str = json.dumps(output, indent=2)
                            # Limit JSON output size
                            if len(json_str) > 2000:
                                return json_str[:2000] + "\n... (truncated)"
                            return json_str
                        except (TypeError, ValueError):
                            str_output = str(output)
                            return str_output[:2000] + "... (truncated)" if len(str_output) > 2000 else str_output
                    else:
                        str_output = str(output)
                        return str_output[:2000] + "... (truncated)" if len(str_output) > 2000 else str_output
        
        # Message extraction with validation
        if "messages" in final_state and isinstance(final_state["messages"], list):
            assistant_messages = []
            for msg in reversed(final_state["messages"]):
                if (isinstance(msg, dict) and 
                    msg.get("role") == "assistant" and 
                    msg.get("content") and 
                    isinstance(msg.get("content"), str)):
                    content = msg["content"].strip()
                    if content:
                        assistant_messages.append(content)
            
            if assistant_messages:
                # Return the most recent non-empty assistant message
                output = assistant_messages[0]
                return output[:2000] + "... (truncated)" if len(output) > 2000 else output
        
        # Check for error states
        if "error" in final_state:
            error_msg = str(final_state["error"])[:500]  # Limit error message size
            return f"Graph execution encountered an error: {error_msg}"
        
        # Fallback with state information
        state_keys = list(final_state.keys()) if final_state else []
        if state_keys:
            return f"Graph execution completed. Available state: {', '.join(state_keys[:5])}"
        else:
            logger.warning(f"GraphAgent '{self.name}' completed with empty final state")
            return "Graph execution completed with no output"

    def get_execution_history(self) -> list:
        """
        Get the execution history from the last graph run.

        Returns:
            List of execution steps with metadata
        """
        if self.compiled_graph:
            return self.compiled_graph.get_execution_history()
        return []

    def get_execution_metadata(self) -> Dict[str, Any]:
        """
        Get metadata from the last execution.

        Returns:
            Dictionary containing execution metadata
        """
        return self.execution_metadata.copy()

    async def clear_state(self):
        """Clear preserved state and execution history with synchronization."""
        async with self._state_lock:
            self._safe_clear_preserved_state()
            self.execution_metadata = {}
            if self.compiled_graph:
                self.compiled_graph.execution_history = []
            self._recovery_attempted = False
            logger.debug(f"Cleared state for GraphAgent '{self.name}'")

    async def update_initial_state(self, updates: Dict[str, Any]):
        """
        Update the initial state for future executions.

        Args:
            updates: Dictionary of state updates to merge
        """
        async with self._state_lock:
            # Validate updates before applying
            safe_updates = {}
            for k, v in updates.items():
                if isinstance(k, str) and not callable(v) and not isinstance(v, type):
                    safe_updates[k] = v
                else:
                    logger.warning(f"Skipping unsafe initial state update: {k}={type(v)}")
            
            self.initial_state.update(safe_updates)
            logger.debug(f"Updated initial state for GraphAgent '{self.name}' with keys: {list(safe_updates.keys())}")

    async def set_preserve_state(self, preserve: bool):
        """
        Enable or disable state preservation between runs.

        Args:
            preserve: Whether to preserve state between runs
        """
        async with self._state_lock:
            self.preserve_state = preserve
            if not preserve:
                self._safe_clear_preserved_state()
            logger.debug(f"Set preserve_state={preserve} for GraphAgent '{self.name}'")
