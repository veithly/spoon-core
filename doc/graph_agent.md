# 📊 SpoonOS Enhanced Graph System - Complete Guide

The SpoonOS Enhanced Graph System is a comprehensive, LangGraph-inspired framework for building intelligent AI agent workflows. It provides advanced features including state management, multi-agent coordination, human-in-the-loop patterns, streaming execution, and seamless LLM integration.

## 📋 Table of Contents

1. [🚀 Quick Start](#-quick-start)
2. [✨ Key Features](#-key-features)
3. [🔧 Core Concepts](#-core-concepts)
4. [📖 Basic Usage](#-basic-usage)
5. [🤖 Multi-Agent Workflows](#-multi-agent-workflows)
6. [👥 Human-in-the-Loop](#-human-in-the-loop)
7. [🧠 LLM Integration](#-llm-integration)
8. [🔄 Streaming & Monitoring](#-streaming--monitoring)
9. [💾 State Persistence](#-state-persistence)
10. [🛡️ Error Handling](#-error-handling)
11. [⚡ Quick Reference](#-quick-reference)
12. [🎯 Best Practices](#-best-practices)
13. [📚 Complete Examples](#-complete-examples)
14. [🎯 Use Cases](#-use-cases)
15. [🔄 Migration from LangGraph](#-migration-from-langgraph)

## 🚀 Quick Start

### Basic Example

```python
from spoon_ai.graph import StateGraph, Command
from typing import TypedDict, Dict, Any

class WorkflowState(TypedDict):
    counter: int
    completed: bool

def increment(state: WorkflowState) -> Dict[str, Any]:
    return {"counter": state["counter"] + 1}

def complete(state: WorkflowState) -> Dict[str, Any]:
    return {"completed": True}

# Build graph
graph = StateGraph(WorkflowState)
graph.add_node("increment", increment)
graph.add_node("complete", complete)
graph.add_edge("increment", "complete")
graph.set_entry_point("increment")

# Execute
compiled = graph.compile()
result = await compiled.invoke({"counter": 0, "completed": False})
print(result)  # {"counter": 1, "completed": True}
```

## ✨ Key Features

### 🏗️ StateGraph Architecture
- **TypedDict state management** with automatic validation
- **Node-based workflows** with conditional routing
- **Command objects** for fine-grained execution control

### 🤖 Multi-Agent Coordination
- **Supervisor patterns** for agent orchestration
- **Agent routing** and task delegation
- **State sharing** between agents

### 👥 Human-in-the-Loop
- **Interrupt/resume** mechanisms for human approval
- **Interactive decision points** in workflows
- **Approval workflows** with custom logic

### 🔄 Streaming & Monitoring
- **Real-time streaming** (values, updates, debug modes)
- **Execution monitoring** and performance tracking
- **State history** and checkpoint management

### 🧠 LLM Integration
- **Seamless integration** with SpoonOS LLM Manager
- **Provider-agnostic** LLM calls (OpenAI, Anthropic, Gemini)
- **Automatic fallback** chains and error handling

### 💾 State Persistence
- **Checkpointing** for workflow resumption
- **Thread-based** session management
- **State history** tracking and recovery

## 🔧 Core Concepts

### StateGraph

The main class for building graph-based workflows:

```python
from spoon_ai.graph import StateGraph
from typing import TypedDict, Annotated

class MyState(TypedDict):
    messages: Annotated[List[str], add_messages]
    counter: int
    completed: bool

graph = StateGraph(MyState)
```

### Nodes

Functions that process state and return updates:

```python
def my_node(state: MyState) -> Dict[str, Any]:
    return {
        "counter": state["counter"] + 1,
        "messages": ["Node executed"]
    }

graph.add_node("my_node", my_node)
```

### Edges

Define flow between nodes:

```python
# Simple edge
graph.add_edge("node1", "node2")

# Conditional edge
def routing_function(state: MyState) -> str:
    return "next_node" if state["counter"] < 5 else "END"

graph.add_conditional_edges("node1", routing_function, {
    "next_node": "node2",
    "END": "END"
})
```

### Command Objects

Fine-grained control over execution:

```python
from spoon_ai.graph import Command

def advanced_node(state: MyState) -> Command:
    return Command(
        update={"counter": state["counter"] + 1},
        goto="next_node"
    )
```

## 🤖 Multi-Agent Workflows

### Supervisor Pattern

```python
class MultiAgentState(TypedDict):
    messages: Annotated[List[Dict], add_messages]
    current_agent: str
    task_type: str
    result: Dict[str, Any]

def supervisor(state: MultiAgentState) -> Command:
    task_type = state["task_type"]
    
    if "research" in task_type:
        return Command(
            update={"current_agent": "researcher"},
            goto="researcher"
        )
    elif "analysis" in task_type:
        return Command(
            update={"current_agent": "analyst"},
            goto="analyst"
        )
    else:
        return Command(goto="END")

# Build multi-agent graph
graph.add_node("supervisor", supervisor)
graph.add_node("researcher", research_agent)
graph.add_node("analyst", analysis_agent)
graph.set_entry_point("supervisor")
```

## 👥 Human-in-the-Loop

### Interrupt Mechanism

```python
from spoon_ai.graph import interrupt

def approval_node(state: MyState) -> Dict[str, Any]:
    if "__resume_data__" in state:
        # Resuming after human input
        approval = state["__resume_data__"]
        return {"approved": approval.get("approved", False)}
    else:
        # Request human input
        interrupt({
            "message": "Please approve this action",
            "data": state["current_task"]
        })
        return {}

# Execute with interrupts
compiled = graph.compile()
config = {"configurable": {"thread_id": "approval_flow"}}

# First execution - will interrupt
result = await compiled.invoke(initial_state, config)

if "__interrupt__" in result:
    # Handle interrupt and resume
    final_result = await compiled.invoke(
        Command(resume={"approved": True}),
        config
    )
```

## 🧠 LLM Integration

### Using LLM Manager

```python
async def llm_node(state: ChatState) -> Dict[str, Any]:
    from spoon_ai.llm.manager import LLMManager
    
    llm_manager = LLMManager()
    response = await llm_manager.chat(state["messages"])
    
    return {
        "messages": [{"role": "assistant", "content": response["content"]}]
    }

# Or use the convenience method
graph.add_llm_node("chat", llm_node)
```

## 🔄 Streaming & Monitoring

### Stream Modes

```python
# Stream state values
async for state in compiled.stream(initial_state, stream_mode="values"):
    print(f"Current state: {state}")

# Stream node updates
async for update in compiled.stream(initial_state, stream_mode="updates"):
    print(f"Node update: {update}")

# Stream debug information
async for debug in compiled.stream(initial_state, stream_mode="debug"):
    print(f"Debug info: {debug}")
```

## 💾 State Persistence

### Execution Tracking

```python
# Track state changes during execution
compiled = graph.compile()
states = []

async for state in compiled.stream(initial_state, stream_mode="values"):
    states.append(state.copy())
    print(f"State: {state}")

print(f"Captured {len(states)} state snapshots")

# Access execution history
execution_history = compiled.execution_history
print(f"Execution steps: {len(execution_history)}")
```

## 🛡️ Error Handling

### Error Types and Recovery

```python
from spoon_ai.graph import (
    GraphExecutionError,
    NodeExecutionError,
    InterruptError
)

try:
    result = await compiled.invoke(initial_state)
except GraphExecutionError as e:
    print(f"Graph execution failed: {e}")
except NodeExecutionError as e:
    print(f"Node {e.node_name} failed: {e}")

# Recovery patterns
def robust_node(state: MyState) -> Dict[str, Any]:
    try:
        result = risky_operation(state)
        return {"result": result, "error": None}
    except Exception as e:
        return {"result": None, "error": str(e)}
```

## 📚 Complete Examples

### Run the Comprehensive Demo

```bash
# Run the complete demo showcasing all features
python examples/llm_integrated_graph_demo.py

# Run the test suite
python -m pytest tests/test_graph.py -v
```

The demo includes:
- ✅ Basic graph execution and state management
- ✅ Streaming execution (values, updates, debug modes)
- ✅ Error handling and recovery
- ✅ Checkpointing and persistence
- ✅ LLM integration with SpoonOS LLM Manager
- ✅ Multi-agent coordination patterns
- ✅ Human-in-the-loop workflows

## 📖 Basic Usage

### Simple Graph Example

```python
import asyncio
from spoon_ai.graph import StateGraph
from typing import TypedDict

class SimpleState(TypedDict):
    value: int
    completed: bool

def increment(state: SimpleState):
    return {"value": state["value"] + 1}

def complete(state: SimpleState):
    return {"completed": True}

def should_continue(state: SimpleState) -> str:
    return "complete" if state["value"] >= 3 else "increment"

# Build graph
graph = StateGraph(SimpleState)
graph.add_node("increment", increment)
graph.add_node("complete", complete)
graph.add_conditional_edges("increment", should_continue, {
    "increment": "increment",
    "complete": "complete"
})
graph.set_entry_point("increment")

# Execute
async def main():
    compiled = graph.compile()
    result = await compiled.invoke({"value": 0, "completed": False})
    print(result)  # {"value": 3, "completed": True}

asyncio.run(main())
```

### State Reducers

Use `Annotated` types to define how state updates are merged:

```python
from typing import Annotated
from spoon_ai.graph import add_messages

class ChatState(TypedDict):
    messages: Annotated[List[Dict], add_messages]  # Messages are appended
    counter: int  # Counter is replaced
```

## 🎯 Use Cases

The Enhanced Graph System is perfect for:

- **AI Agent Workflows** - Multi-step reasoning and action chains
- **Multi-Agent Systems** - Agent coordination and communication
- **Human-AI Collaboration** - Approval workflows and interactive decision making
- **Data Processing Pipelines** - ETL workflows with error handling
- **Customer Service Automation** - Multi-tier support with human handoff

## ⚡ Quick Reference

### Core API

```python
from spoon_ai.graph import StateGraph, Command, add_messages
from typing import TypedDict, Annotated, Dict, Any

# 1. Define State
class MyState(TypedDict):
    messages: Annotated[List[Dict], add_messages]
    counter: int
    completed: bool

# 2. Create Graph
graph = StateGraph(MyState)

# 3. Add Nodes
def my_node(state: MyState) -> Dict[str, Any]:
    return {"counter": state["counter"] + 1}

graph.add_node("my_node", my_node)

# 4. Add Edges
graph.add_edge("my_node", "END")
graph.set_entry_point("my_node")

# 5. Execute
compiled = graph.compile()
result = await compiled.invoke({"counter": 0, "messages": [], "completed": False})
```

### StateGraph Methods

```python
graph = StateGraph(StateSchema)
graph.add_node("name", function)
graph.add_edge("from", "to")
graph.add_conditional_edges("from", condition_func, mapping)
graph.set_entry_point("start_node")
compiled = graph.compile()
```

### Command Objects

```python
def node_with_command(state) -> Command:
    return Command(
        update={"key": "value"},
        goto="next_node"
    )
```

### Conditional Routing

```python
def should_continue(state) -> str:
    return "continue" if state["counter"] < 5 else "END"

graph.add_conditional_edges("node", should_continue, {
    "continue": "node",
    "END": "END"
})
```

### Error Recovery

```python
def safe_node(state):
    try:
        return {"result": risky_operation()}
    except Exception as e:
        return {"error": str(e)}

def error_router(state) -> str:
    return "recovery" if state.get("error") else "success"
```

### Agent Handoff

```python
def agent_router(state) -> str:
    current = state["current_agent"]
    if current == "researcher" and state["research_done"]:
        return "analyst"
    elif current == "analyst" and state["analysis_done"]:
        return "supervisor"
    return "END"
```

## 🎯 Best Practices

### 1. State Design

```python
# Good: Clear, typed state
class WellDesignedState(TypedDict):
    # Use descriptive names
    user_messages: Annotated[List[Dict], add_messages]
    processing_step: str
    analysis_results: Dict[str, Any]
    is_complete: bool

# Avoid: Unclear, untyped state
class PoorState(TypedDict):
    data: Any  # Too generic
    flag: bool  # Unclear purpose
```

### 2. Node Design

```python
# Good: Single responsibility, clear inputs/outputs
def analyze_sentiment(state: ChatState) -> Dict[str, Any]:
    """Analyze sentiment of the last user message."""
    last_message = state["user_messages"][-1]["content"]
    sentiment = perform_sentiment_analysis(last_message)
    
    return {
        "analysis_results": {
            **state.get("analysis_results", {}),
            "sentiment": sentiment
        }
    }

# Avoid: Multiple responsibilities, unclear purpose
def do_everything(state: Any) -> Any:
    # Does too many things
    pass
```

### 3. Error Handling

```python
# Good: Specific error handling
def safe_api_call(state: MyState) -> Dict[str, Any]:
    try:
        response = call_external_api(state["query"])
        return {"api_response": response, "error": None}
    except APITimeoutError:
        return {"api_response": None, "error": "timeout"}
    except APIRateLimitError:
        return {"api_response": None, "error": "rate_limit"}
    except Exception as e:
        return {"api_response": None, "error": f"unexpected: {str(e)}"}
```

### 4. Testing

```python
import pytest

@pytest.mark.asyncio
async def test_graph_execution():
    graph = StateGraph(MyState)
    # ... build graph
    
    compiled = graph.compile()
    result = await compiled.invoke({"counter": 0})
    
    assert result["counter"] == 3
    assert result["completed"] is True

@pytest.mark.asyncio
async def test_error_handling():
    # Test error scenarios
    with pytest.raises(NodeExecutionError):
        await compiled.invoke({"invalid": "state"})
```

## 🔄 Migration from LangGraph

The system is designed to be compatible with LangGraph patterns while providing enhanced features:

```python
# LangGraph style (works)
from spoon_ai.graph import StateGraph

# SpoonOS enhancements (recommended)
from spoon_ai.graph import StateGraph, Command, interrupt
```

### Key Differences

- **Enhanced error handling** with specific exception types
- **Built-in LLM integration** with SpoonOS LLM Manager
- **Additional streaming modes** and monitoring capabilities
- **Improved checkpointing** and persistence options
- **Extended Command objects** for fine-grained control