from typing import List, Optional, Union

from langchain_core.messages import BaseMessage
from openai import AsyncOpenAI
from pydantic import BaseModel, Field
from tenacity import retry, stop_after_attempt, wait_random_exponential


class Memory(BaseModel):
    messages: List[BaseMessage] = Field(default_factory=list)
    max_messages: int = 100
    
    def add_message(self, message: BaseMessage) -> None:
        self.messages.append(message)
        if len(self.messages) > self.max_messages:
            self.messages.pop(0)
    
    def get_messages(self) -> List[BaseMessage]:
        return self.messages
    
    def clear(self) -> None:
        self.messages.clear()

def to_dict(message: BaseMessage) -> dict:
    return {
        "role": message.role,
        "content": message.content
    }

class ChatBot:
    def __init__(self, model_name: str = "gpt-4.5-preview", llm_config: dict = None, llm_provider: str = "openai", api_key: str = None):
        if llm_provider == "openai":
            self.llm = AsyncOpenAI(model=model_name, api_key=api_key, **llm_config)
        else:
            raise ValueError(f"Invalid LLM provider: {llm_provider}")
    
    async def ask(self, messages: List[Union[dict, BaseMessage]], system_msg: Optional[str] = None) -> str:
        formatted_messages = [] if system_msg is None else [{"role": "system", "content": system_msg}]
        for message in messages:
            if isinstance(message, dict):
                formatted_messages.append(message)
            elif isinstance(message, BaseMessage):
                formatted_messages.append(to_dict(message))
            else:
                raise ValueError(f"Invalid message type: {type(message)}")
        response = await self.llm.chat.completions.create(messages=formatted_messages, model=self.model_name, max_tokens=4096, temperature=0.3,stream=False)
        return response.choices[0].message.content
    
    @retry(stop=stop_after_attempt(3), wait=wait_random_exponential(min=1, max=60))
    async def ask_tool(self,messages: List[Union[dict, BaseMessage]], system_msg: Optional[str] = None, tools: Optional[List[dict]] = None, tool_choice: Optional[str] = None, **kwargs):
        if tool_choice not in ["auto", "none", "required"]:
            raise ValueError(f"Invalid tool choice: {tool_choice}")
        if tools is None:
            tools = []
        if tool_choice == "auto":
            tool_choice = "required" if tools else "none"
        
        formatted_messages = [] if system_msg is None else [{"role": "system", "content": system_msg}]
        for message in messages:
            if isinstance(message, dict):
                formatted_messages.append(message)
            elif isinstance(message, BaseMessage):
                formatted_messages.append(to_dict(message))
            else:
                raise ValueError(f"Invalid message type: {type(message)}")
        
        response = await self.llm.chat.completions.create(messages=formatted_messages, model=self.model_name, max_tokens=4096, temperature=0.3,stream=False, tools=tools, tool_choice=tool_choice, **kwargs)
        return response.choices[0].message