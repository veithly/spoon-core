import asyncio
import logging
import uuid
import json
import datetime
from pathlib import Path
from abc import ABC
from contextlib import asynccontextmanager
from typing import Literal, Optional, List

from spoon_ai.schema import Message, Role
from pydantic import BaseModel, Field

from spoon_ai.chat import ChatBot, Memory
from spoon_ai.schema import AgentState, ToolCall

DEBUG = False
def debug_log(message):
    if DEBUG:
        logger.info(f"DEBUG: {message}\n")

logger = logging.getLogger(__name__)

class BaseAgent(BaseModel, ABC):
    """
    Base class for all agents.
    """
    name: str = Field(..., description="The name of the agent")
    description: Optional[str] = Field(None, description="The description of the agent")
    system_prompt: Optional[str] = Field(None, description="The system prompt for the agent")
    next_step_prompt: Optional[str] = Field(None, description="Prompt for determining next action")
    
    llm: ChatBot = Field(..., description="The LLM to use for the agent")
    memory: Memory = Field(default_factory=Memory, description="The memory to use for the agent")
    state: AgentState = Field(default=AgentState.IDLE, description="The state of the agent")
    
    max_steps: int = Field(default=10, description="The maximum number of steps the agent can take")
    current_step: int = Field(default=0, description="The current step of the agent")

    output_queue: asyncio.Queue = Field(default_factory=asyncio.Queue, description="The queue to store the output of the agent")
    
    class Config:
        arbitrary_types_allowed = True
        extra = "allow"
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.state = AgentState.IDLE
    
    def add_message(self, role: Literal["user", "assistant", "tool"], content: str, tool_call_id: Optional[str] = None, tool_calls: Optional[List[ToolCall]] = None, tool_name: Optional[str] = None):
        if role not in ["user", "assistant", "tool"]:
            raise ValueError(f"Invalid role: {role}")
        
        if role == "user":
            self.memory.add_message(Message(role=Role.USER, content=content))
        elif role == "assistant":
            if tool_calls:
                self.memory.add_message(Message(role=Role.ASSISTANT, content=content, tool_calls=[{"id": toolcall.id, "type": "function", "function": toolcall.function} for toolcall in tool_calls]))
            else:
                self.memory.add_message(Message(role=Role.ASSISTANT, content=content))
        elif role == "tool":
            self.memory.add_message(Message(role=Role.TOOL, content=content, tool_call_id=tool_call_id))
    
    @asynccontextmanager
    async def state_context(self, new_state: AgentState):
        if not isinstance(new_state, AgentState):
            raise ValueError(f"Invalid state: {new_state}")
        
        old_state = self.state
        self.state = new_state
        try:
            yield
        except Exception as e:
            self.state = AgentState.ERROR
            raise e
        finally:
            self.state = old_state
    
    async def run(self, request: Optional[str] = None) -> str:
        if self.state != AgentState.IDLE:
            raise RuntimeError(f"Agent {self.name} is not in the IDLE state")
        
        self.state = AgentState.RUNNING
        
        if request is not None:
            self.memory.add_message(Message(role=Role.USER, content=request))
        results: List[str] = []
        async with self.state_context(AgentState.RUNNING):
            while (
                self.current_step < self.max_steps and
                self.state == AgentState.RUNNING
            ):
                self.current_step += 1
                logger.info(f"Agent {self.name} is running step {self.current_step}/{self.max_steps}")
                
                step_result = await self.step()
                if self.is_stuck():
                    self.handle_struck_state()
                
                results.append(f"Step {self.current_step}: {step_result}")
                logger.info(f"Step {self.current_step}: {step_result}")
            
            if self.current_step >= self.max_steps:
                results.append(f"Step {self.current_step}: Stuck in loop. Resetting state.")
                
        return "\n".join(results) if results else "No results"
    
    async def step(self) -> str:
        raise NotImplementedError("Subclasses must implement this method")
    
    def is_stuck(self) -> bool:
        if len(self.memory.get_messages()) < 2:
            return False
        
        last_message = self.memory.get_messages()[-1]
        if not last_message.content:
            return False
        
        duplicate_count = sum(
            1
            for msg in reversed(self.memory.get_messages()[:-1])
            if msg.role == Role.ASSISTANT and msg.content == last_message.content
        )
        return duplicate_count >= 2
    
    def handle_struck_state(self):
        logger.warning(f"Agent {self.name} is stuck. Resetting state.")
        struck_prompt = "Observed duplicate response. Consider new strategies and avoid repeating ineffective paths already attempted."
        self.next_step_prompt = f"{struck_prompt}\n\n{self.next_step_prompt}"
        logger.warning(f"Added struck prompt: {struck_prompt}")
    
    def save_chat_history(self):
        history_dir = Path('chat_logs')
        history_dir.mkdir(exist_ok=True)
        
        history_file = history_dir / f'{self.name}_history.json'
        
        now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        if isinstance(self.chat_history, list):
            save_data = {
                'metadata': {
                    'agent_name': self.name,
                    'created_at': now,
                    'updated_at': now
                },
                'messages': self.chat_history
            }
        elif isinstance(self.chat_history, dict) and 'metadata' in self.chat_history:
            save_data = self.chat_history
            save_data['metadata']['updated_at'] = now
        else:
            save_data = {
                'metadata': {
                    'agent_name': self.name,
                    'created_at': now,
                    'updated_at': now
                },
                'messages': []
            }
        
        try:
            with open(history_file, 'w', encoding='utf-8') as f:
                json.dump(save_data, f, ensure_ascii=False, indent=2)
            debug_log(f"Saved chat history with {len(save_data.get('messages', []))} messages")
        except Exception as e:
            debug_log(f"Error saving chat history: {e}")