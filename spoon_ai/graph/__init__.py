"""
spoon_ai.graph package

Public facade for the graph engine. Import from here.
"""
# Re-export public API
from .exceptions import (
    GraphExecutionError,
    NodeExecutionError,
    StateValidationError,
    CheckpointError,
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
    add_messages,
    merge_dicts,
    append_history,
    union_sets,
    validate_range,
    validate_enum,
)
from .decorators import (
    node_decorator,
    router_decorator,
)
from .checkpointer import InMemoryCheckpointer

# Engine and agent implementations (now within this package)
from .engine import (
    StateGraph,
    CompiledGraph,
    BaseNode,
    RunnableNode,
    ToolNode,
    ConditionNode,
    START,
    END,
    interrupt
)
from .agent import GraphAgent, AgentStateCheckpoint, MockMemory, Memory
from .config import GraphConfig, ParallelGroupConfig, ParallelRetryPolicy, RouterConfig
from .builder import (
    Intent,
    IntentAnalyzer,
    AdaptiveStateBuilder,
    ParameterInferenceEngine,
    NodeSpec,
    EdgeSpec,
    ParallelGroupSpec,
    GraphTemplate,
    DeclarativeGraphBuilder,
    NodePlugin,
    NodePluginSystem,
    HighLevelGraphAPI,
)
from .mcp_integration import (
    MCPToolSpec,
    MCPConfigManager,
    MCPToolDiscoveryEngine,
    MCPIntegrationManager,
)

