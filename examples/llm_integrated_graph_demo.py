"""
Comprehensive Graph System Demo - Complete showcase of SpoonOS Enhanced Graph System.

This comprehensive demo showcases ALL features of the enhanced graph system:
- Basic graph execution with state management
- LLM integration with SpoonOS LLM Manager
- Multi-agent coordination patterns
- Human-in-the-loop workflows with interrupts
- Streaming execution (values, updates, debug modes)
- Error handling and recovery
- Checkpointing and persistence
- Conditional routing and dynamic edges
- Command objects and state reducers

This is the ONLY demo file needed - it contains all functionality examples.
"""

import asyncio
import logging
from typing import List, Dict, Any, Annotated, TypedDict
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import our enhanced graph components
from spoon_ai.graph import (
    StateGraph,
    CompiledGraph,
    InMemoryCheckpointer,
    Command,
    StateSnapshot,
    interrupt,
    add_messages,
    GraphExecutionError,
    NodeExecutionError,
    InterruptError,
    GraphConfigurationError,
    StateValidationError,
    CheckpointError
)

# Try to import real LLM manager, fall back to mock
try:
    from spoon_ai.llm.manager import LLMManager
    REAL_LLM_AVAILABLE = True
except ImportError:
    REAL_LLM_AVAILABLE = False

# Mock LLM Manager for demonstration (replace with actual import)
class MockLLMManager:
    """Mock LLM Manager for demonstration purposes."""
    
    async def chat(self, messages: List[Dict[str, str]], **kwargs) -> Dict[str, Any]:
        """Mock chat method."""
        last_message = messages[-1]["content"] if messages else ""
        
        # Simple mock responses based on content
        if "research" in last_message.lower():
            response = "I'll help you research that topic. Let me gather some information..."
        elif "analyze" in last_message.lower():
            response = "Based on my analysis, here are the key findings..."
        elif "summarize" in last_message.lower():
            response = "Here's a summary of the main points..."
        else:
            response = "I understand. How can I help you further?"
        
        return {
            "content": response,
            "provider": "mock",
            "model": "mock-model",
            "usage": {"total_tokens": 50}
        }


# Global mock LLM manager instance
llm_manager = MockLLMManager()


# State schemas for different demo types
class BasicState(TypedDict):
    counter: int
    messages: Annotated[List[str], add_messages]
    completed: bool

class LLMWorkflowState(TypedDict):
    messages: Annotated[List[Dict[str, str]], add_messages]
    task_type: str
    current_step: str
    results: Dict[str, Any]
    completed: bool

class MultiAgentState(TypedDict):
    messages: Annotated[List[Dict[str, str]], add_messages]
    current_agent: str
    task_type: str
    result: Dict[str, Any]


async def demo_basic_graph_execution():
    """Demo 1: Basic graph execution with state management."""
    print("\n" + "="*70)
    print("DEMO 1: Basic Graph Execution & State Management")
    print("="*70)
    
    graph = StateGraph(BasicState)
    
    def increment_node(state: BasicState) -> Dict[str, Any]:
        """Simple node that increments counter."""
        return {
            "counter": state["counter"] + 1,
            "messages": [f"Counter incremented to {state['counter'] + 1}"]
        }
    
    def completion_node(state: BasicState) -> Dict[str, Any]:
        """Node that marks completion."""
        return {
            "completed": True,
            "messages": ["Task completed successfully"]
        }
    
    def should_continue(state: BasicState) -> str:
        """Conditional routing based on counter."""
        if state["counter"] >= 3:
            return "complete"
        return "increment"
    
    # Build graph
    graph.add_node("increment", increment_node)
    graph.add_node("complete", completion_node)
    graph.add_conditional_edges("increment", should_continue, {
        "increment": "increment",
        "complete": "complete"
    })
    graph.set_entry_point("increment")
    
    compiled = graph.compile()
    
    # Execute
    initial_state = {"counter": 0, "messages": [], "completed": False}
    result = await compiled.invoke(initial_state)
    
    print(f"✅ Final state: counter={result['counter']}, completed={result['completed']}")
    print(f"📝 Messages: {result['messages']}")


async def demo_streaming_execution():
    """Demo 2: Streaming execution with different modes."""
    print("\n" + "="*70)
    print("DEMO 2: Streaming Execution (Values, Updates, Debug)")
    print("="*70)
    
    graph = StateGraph(BasicState)
    
    def step1(state: BasicState) -> Dict[str, Any]:
        return {"counter": 1, "messages": ["Step 1 completed"]}
    
    def step2(state: BasicState) -> Dict[str, Any]:
        return {"counter": 2, "messages": ["Step 2 completed"]}
    
    def step3(state: BasicState) -> Dict[str, Any]:
        return {"counter": 3, "completed": True, "messages": ["All steps completed"]}
    
    graph.add_node("step1", step1)
    graph.add_node("step2", step2)
    graph.add_node("step3", step3)
    graph.add_edge("step1", "step2")
    graph.add_edge("step2", "step3")
    graph.set_entry_point("step1")
    
    compiled = graph.compile()
    initial_state = {"counter": 0, "messages": [], "completed": False}
    
    # Stream values
    print("🔄 Streaming values:")
    async for state in compiled.stream(initial_state, stream_mode="values"):
        print(f"  State: counter={state['counter']}, completed={state['completed']}")
    
    # Stream updates
    print("\n🔄 Streaming updates:")
    async for update in compiled.stream(initial_state, stream_mode="updates"):
        print(f"  Update: {update}")
    
    # Stream debug
    print("\n🔄 Streaming debug:")
    async for debug_info in compiled.stream(initial_state, stream_mode="debug"):
        print(f"  Debug: {debug_info}")


async def demo_error_handling():
    """Demo 3: Error handling and recovery."""
    print("\n" + "="*70)
    print("DEMO 3: Error Handling & Recovery")
    print("="*70)
    
    graph = StateGraph(BasicState)
    
    def error_prone_node(state: BasicState) -> Dict[str, Any]:
        """Node that might fail."""
        if state["counter"] == 2:
            raise NodeExecutionError("Simulated node failure", "error_prone_node")
        return {"counter": state["counter"] + 1, "messages": ["Node executed successfully"]}
    
    def recovery_node(state: BasicState) -> Dict[str, Any]:
        """Recovery node."""
        return {
            "counter": state["counter"] + 1,
            "messages": ["Recovered from error"],
            "completed": True
        }
    
    graph.add_node("error_prone", error_prone_node)
    graph.add_node("recovery", recovery_node)
    graph.add_edge("error_prone", "recovery")
    graph.set_entry_point("error_prone")
    
    compiled = graph.compile()
    
    # Test normal execution
    print("✅ Testing normal execution:")
    result = await compiled.invoke({"counter": 0, "messages": [], "completed": False})
    print(f"  Result: {result['messages']}")
    
    # Test error handling
    print("\n❌ Testing error handling:")
    try:
        await compiled.invoke({"counter": 2, "messages": [], "completed": False})
    except (NodeExecutionError, GraphExecutionError) as e:
        print(f"  ✅ Caught expected error: {e}")
        if hasattr(e, 'node_name'):
            print(f"  Error node: {e.node_name}")


async def demo_checkpointing():
    """Demo 4: Checkpointing and persistence."""
    print("\n" + "="*70)
    print("DEMO 4: Checkpointing & Persistence")
    print("="*70)
    
    # Create checkpointer
    checkpointer = InMemoryCheckpointer()
    
    graph = StateGraph(BasicState)
    
    def slow_step1(state: BasicState) -> Dict[str, Any]:
        return {"counter": 1, "messages": ["Slow step 1 completed"]}
    
    def slow_step2(state: BasicState) -> Dict[str, Any]:
        return {"counter": 2, "messages": ["Slow step 2 completed"]}
    
    def slow_step3(state: BasicState) -> Dict[str, Any]:
        return {"counter": 3, "completed": True, "messages": ["All slow steps completed"]}
    
    graph.add_node("slow1", slow_step1)
    graph.add_node("slow2", slow_step2)
    graph.add_node("slow3", slow_step3)
    graph.add_edge("slow1", "slow2")
    graph.add_edge("slow2", "slow3")
    graph.set_entry_point("slow1")
    
    compiled = graph.compile(checkpointer=checkpointer)
    config = {"configurable": {"thread_id": "checkpoint_demo"}}
    
    # Execute with checkpointing
    initial_state = {"counter": 0, "messages": [], "completed": False}
    result = await compiled.invoke(initial_state, config)
    
    print(f"✅ Execution completed with checkpointing")
    print(f"📝 Final messages: {result['messages']}")
    
    # Get checkpoint history
    history = [snapshot async for snapshot in compiled.get_state_history(config)]
    print(f"📊 Checkpoint history: {len(history)} snapshots saved")


async def demo_llm_chat_workflow():
    """Demo: LLM-powered chat workflow with graph orchestration."""
    print("\n" + "="*60)
    print("DEMO: LLM-Powered Chat Workflow")
    print("="*60)
    
    graph = StateGraph(LLMWorkflowState)
    
    async def llm_chat_node(state: LLMWorkflowState) -> Dict[str, Any]:
        """Node that uses LLM manager for chat."""
        # Get the conversation history
        messages = state["messages"]
        
        # Call LLM manager
        response = await llm_manager.chat(messages)
        
        # Add LLM response to messages
        assistant_message = {
            "role": "assistant",
            "content": response["content"]
        }
        
        return {
            "messages": [assistant_message],
            "results": {
                "llm_response": response,
                "step": "chat_completed"
            }
        }
    
    def completion_check(state: LLMWorkflowState) -> str:
        """Check if the conversation should continue."""
        last_message = state["messages"][-1]["content"] if state["messages"] else ""
        
        # Simple completion logic
        if any(word in last_message.lower() for word in ["goodbye", "bye", "thanks", "complete"]):
            return "END"
        else:
            return "continue"
    
    # Build the graph
    graph.add_node("llm_chat", llm_chat_node)
    graph.add_conditional_edges("llm_chat", completion_check, {
        "continue": "llm_chat",
        "END": "END"
    })
    graph.set_entry_point("llm_chat")
    
    # Execute with different user inputs
    compiled = graph.compile()
    
    test_conversations = [
        "Hello, can you help me research AI trends?",
        "Please analyze the current market conditions",
        "Can you summarize what we've discussed?",
        "Thank you, that's all for now"
    ]
    
    state = {
        "messages": [],
        "task_type": "chat",
        "current_step": "start",
        "results": {},
        "completed": False
    }
    
    for user_input in test_conversations:
        print(f"\n👤 User: {user_input}")
        
        # Add user message to state
        user_message = {"role": "user", "content": user_input}
        state["messages"].append(user_message)
        
        # Execute graph
        result = await compiled.invoke(state)
        
        # Get the latest assistant response
        assistant_messages = [msg for msg in result["messages"] if msg["role"] == "assistant"]
        if assistant_messages:
            latest_response = assistant_messages[-1]["content"]
            print(f"🤖 Assistant: {latest_response}")
        
        # Update state for next iteration
        state = result
        
        # Check if conversation ended
        if result.get("completed") or "goodbye" in user_input.lower():
            break
    
    print(f"\n✅ Conversation completed with {len(state['messages'])} messages exchanged")


async def demo_multi_agent_llm_workflow():
    """Demo: Multi-agent workflow with LLM integration."""
    print("\n" + "="*60)
    print("DEMO: Multi-Agent LLM Workflow")
    print("="*60)
    
    graph = StateGraph(LLMWorkflowState)
    
    async def supervisor_node(state: LLMWorkflowState) -> Command:
        """Supervisor that routes tasks to appropriate agents."""
        task_type = state["task_type"]
        current_step = state["current_step"]
        
        if current_step == "start":
            if "research" in task_type:
                return Command(
                    update={"current_step": "research"},
                    goto="research_agent"
                )
            elif "analysis" in task_type:
                return Command(
                    update={"current_step": "analysis"},
                    goto="analysis_agent"
                )
        elif current_step in ["research", "analysis"]:
            # Task completed, go to summary
            return Command(
                update={"current_step": "summary"},
                goto="summary_agent"
            )
        else:
            # All done
            return Command(
                update={"completed": True},
                goto="END"
            )
    
    async def research_agent_node(state: LLMWorkflowState) -> Command:
        """Research agent powered by LLM."""
        messages = state["messages"] + [
            {"role": "system", "content": "You are a research specialist. Provide detailed research insights."}
        ]
        
        response = await llm_manager.chat(messages)
        
        return Command(
            update={
                "messages": [{"role": "assistant", "content": response["content"]}],
                "results": {**state["results"], "research": response["content"]},
                "current_step": "research_complete"
            },
            goto="supervisor"
        )
    
    async def analysis_agent_node(state: LLMWorkflowState) -> Command:
        """Analysis agent powered by LLM."""
        messages = state["messages"] + [
            {"role": "system", "content": "You are an analysis expert. Provide thorough analysis."}
        ]
        
        response = await llm_manager.chat(messages)
        
        return Command(
            update={
                "messages": [{"role": "assistant", "content": response["content"]}],
                "results": {**state["results"], "analysis": response["content"]},
                "current_step": "analysis_complete"
            },
            goto="supervisor"
        )
    
    async def summary_agent_node(state: LLMWorkflowState) -> Command:
        """Summary agent that consolidates results."""
        # Create summary prompt based on previous results
        summary_prompt = "Please provide a comprehensive summary of our work: "
        if "research" in state["results"]:
            summary_prompt += f"Research findings: {state['results']['research'][:100]}... "
        if "analysis" in state["results"]:
            summary_prompt += f"Analysis results: {state['results']['analysis'][:100]}... "
        
        messages = state["messages"] + [
            {"role": "user", "content": summary_prompt}
        ]
        
        response = await llm_manager.chat(messages)
        
        return Command(
            update={
                "messages": [{"role": "assistant", "content": response["content"]}],
                "results": {**state["results"], "summary": response["content"]},
                "current_step": "summary_complete"
            },
            goto="supervisor"
        )
    
    # Build the graph
    graph.add_node("supervisor", supervisor_node)
    graph.add_node("research_agent", research_agent_node)
    graph.add_node("analysis_agent", analysis_agent_node)
    graph.add_node("summary_agent", summary_agent_node)
    graph.set_entry_point("supervisor")
    
    compiled = graph.compile()
    
    # Test research workflow
    print("🔬 Testing Research Workflow:")
    research_state = {
        "messages": [{"role": "user", "content": "I need research on AI market trends"}],
        "task_type": "research task",
        "current_step": "start",
        "results": {},
        "completed": False
    }
    
    result = await compiled.invoke(research_state)
    print(f"✅ Research workflow completed")
    print(f"📊 Results keys: {list(result['results'].keys())}")
    print(f"💬 Total messages: {len(result['messages'])}")
    
    # Test analysis workflow
    print("\n📈 Testing Analysis Workflow:")
    analysis_state = {
        "messages": [{"role": "user", "content": "Please analyze the market data"}],
        "task_type": "analysis task",
        "current_step": "start",
        "results": {},
        "completed": False
    }
    
    result = await compiled.invoke(analysis_state)
    print(f"✅ Analysis workflow completed")
    print(f"📊 Results keys: {list(result['results'].keys())}")
    print(f"💬 Total messages: {len(result['messages'])}")


async def demo_human_llm_collaboration():
    """Demo: Human-in-the-loop workflow with LLM assistance."""
    print("\n" + "="*60)
    print("DEMO: Human-LLM Collaboration Workflow")
    print("="*60)
    
    graph = StateGraph(LLMWorkflowState)
    
    async def llm_draft_node(state: LLMWorkflowState) -> Dict[str, Any]:
        """LLM creates initial draft."""
        messages = state["messages"] + [
            {"role": "system", "content": "Create a draft response based on the user's request."}
        ]
        
        response = await llm_manager.chat(messages)
        
        return {
            "messages": [{"role": "assistant", "content": f"DRAFT: {response['content']}"}],
            "results": {"draft": response["content"]},
            "current_step": "draft_ready"
        }
    
    async def human_review_node(state: LLMWorkflowState) -> Dict[str, Any]:
        """Human reviews and potentially edits the draft."""
        if "__resume_data__" in state:
            # We're resuming after human input
            human_feedback = state["__resume_data__"]
            
            if human_feedback.get("approved"):
                return {
                    "messages": [{"role": "system", "content": "Draft approved by human reviewer"}],
                    "results": {**state["results"], "final": state["results"]["draft"]},
                    "current_step": "approved",
                    "completed": True
                }
            else:
                # Human provided edits
                edited_content = human_feedback.get("edits", "No specific edits provided")
                return {
                    "messages": [{"role": "system", "content": f"Human edits: {edited_content}"}],
                    "results": {**state["results"], "human_edits": edited_content},
                    "current_step": "needs_revision"
                }
        else:
            # First time - request human review
            draft = state["results"].get("draft", "No draft available")
            human_input = interrupt({
                "message": "Please review the draft and provide feedback",
                "draft": draft,
                "options": ["approve", "edit"]
            })
            return {"results": {"review_requested": True}}
    
    async def llm_revise_node(state: LLMWorkflowState) -> Dict[str, Any]:
        """LLM revises based on human feedback."""
        draft = state["results"].get("draft", "")
        human_edits = state["results"].get("human_edits", "")
        
        revision_prompt = f"Please revise this draft based on human feedback:\nDraft: {draft}\nFeedback: {human_edits}"
        
        messages = state["messages"] + [
            {"role": "user", "content": revision_prompt}
        ]
        
        response = await llm_manager.chat(messages)
        
        return {
            "messages": [{"role": "assistant", "content": f"REVISED: {response['content']}"}],
            "results": {**state["results"], "final": response["content"]},
            "current_step": "revised",
            "completed": True
        }
    
    def routing_logic(state: LLMWorkflowState) -> str:
        """Route based on current step."""
        step = state["current_step"]
        
        if step == "draft_ready":
            return "human_review"
        elif step == "needs_revision":
            return "llm_revise"
        else:
            return "END"
    
    # Build the graph
    graph.add_node("llm_draft", llm_draft_node)
    graph.add_node("human_review", human_review_node)
    graph.add_node("llm_revise", llm_revise_node)
    
    graph.add_edge("llm_draft", "human_review")
    graph.add_conditional_edges("human_review", routing_logic, {
        "human_review": "human_review",
        "llm_revise": "llm_revise",
        "END": "END"
    })
    graph.add_edge("llm_revise", "END")
    graph.set_entry_point("llm_draft")
    
    compiled = graph.compile()
    config = {"configurable": {"thread_id": "collaboration_demo"}}
    
    initial_state = {
        "messages": [{"role": "user", "content": "Write a brief introduction to AI ethics"}],
        "task_type": "content_creation",
        "current_step": "start",
        "results": {},
        "completed": False
    }
    
    print("📝 Starting collaborative content creation...")
    
    # First execution - should create draft and interrupt for human review
    result = await compiled.invoke(initial_state, config)
    
    if "__interrupt__" in result:
        print("🔄 Workflow interrupted for human review!")
        interrupt_info = result["__interrupt__"][0]
        print(f"📋 Draft to review: {interrupt_info['value']['draft'][:100]}...")
        
        # Simulate human approval
        print("✅ Simulating human approval...")
        final_result = await compiled.invoke(
            Command(resume={"approved": True}),
            config
        )
        
        print("✅ Collaboration workflow completed!")
        print(f"📄 Final content available: {len(final_result['results'].get('final', ''))} characters")
    else:
        print("❌ Expected interrupt did not occur")


async def main():
    """Run comprehensive graph system demos."""
    print("🚀 Comprehensive Graph System Demo Suite")
    print("Showcasing ALL features of SpoonOS Enhanced Graph System")
    print("=" * 70)
    
    try:
        # Basic functionality demos
        await demo_basic_graph_execution()
        await demo_streaming_execution()
        await demo_error_handling()
        await demo_checkpointing()
        
        # Advanced LLM integration demos
        await demo_llm_chat_workflow()
        await demo_multi_agent_llm_workflow()
        await demo_human_llm_collaboration()
        
        print("\n" + "=" * 70)
        print("✅ ALL DEMOS COMPLETED SUCCESSFULLY!")
        print("🎯 Graph System Features Demonstrated:")
        print("   ✅ Basic execution & state management")
        print("   ✅ Streaming (values, updates, debug)")
        print("   ✅ Error handling & recovery")
        print("   ✅ Checkpointing & persistence")
        print("   ✅ LLM integration")
        print("   ✅ Multi-agent coordination")
        print("   ✅ Human-in-the-loop workflows")
        print("   ✅ Conditional routing")
        print("   ✅ Command objects")
        print("   ✅ State reducers")
        print("=" * 70)
        
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())