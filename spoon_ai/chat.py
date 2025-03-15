import os
from logging import getLogger
from typing import List, Optional, Union

from spoon_ai.schema import Message, Role

from openai import AsyncOpenAI
from pydantic import BaseModel, Field
from tenacity import retry, stop_after_attempt, wait_random_exponential

logger = getLogger(__name__)

class Memory(BaseModel):
    messages: List[Message] = Field(default_factory=list)
    max_messages: int = 100
    
    def add_message(self, message:  Message) -> None:
        self.messages.append(message)
        if len(self.messages) > self.max_messages:
            self.messages.pop(0)
    
    def get_messages(self) -> List[Message]:
        return self.messages
    
    def clear(self) -> None:
        self.messages.clear()

def to_dict(message: Message) -> dict:
    # if message.role == Role.USER:
    #     return {
    #         "role": "user",
    #         "content": message.content
    #     }
    # elif message.role == Role.ASSISTANT:
    #     return {
    #         "role": "assistant",
    #         "content": message.content
    #     }
    # elif message.role == Role.SYSTEM:
    #     return {
    #         "role": "system",
    #         "content": message.content
    #     }
    # elif message.role == Role.TOOL:
    #     return {
    #         "role": "tool",
    #         "content": message.content,
    #         "tool_call_id": message.tool_call_id
    #     }
    # else:
    #     raise ValueError(f"Invalid message type: {type(message)}")
    messages = {"role": message.role}
    if message.content:
        messages["content"] = message.content
    if message.tool_calls:
        messages["tool_calls"] = [tool_call.model_dump() for tool_call in message.tool_calls]
    if message.name:
        messages["name"] = message.name
    if message.tool_call_id:
        messages["tool_call_id"] = message.tool_call_id
    return messages

class ChatBot:
    # def __init__(self, model_name: str = "gpt-4.5-preview", llm_config: dict = None, llm_provider: str = "openai", api_key: str = None):
    def __init__(self, model_name: str = "claude-3-7-sonnet-20250219", llm_config: dict = None, llm_provider: str = "anthropic", api_key: str = None):
        if llm_provider == "openai":
            self.llm = AsyncOpenAI()
            self.model_name = model_name
            self.api_key = api_key
            self.llm_config = llm_config
            self.llm.api_key = api_key if api_key else os.getenv("OPENAI_API_KEY")
        elif llm_provider == "anthropic":
            self.llm = AsyncOpenAI(base_url="https://api.anthropic.com/v1")
            self.model_name = "claude-3-7-sonnet-20250219"
            self.api_key = api_key
            self.llm_config = llm_config
            self.llm.api_key = api_key if api_key else os.getenv("ANTHROPIC_API_KEY")
        else:
            raise ValueError(f"Invalid LLM provider: {llm_provider}")
    
    async def ask(self, messages: List[Union[dict, Message]], system_msg: Optional[str] = None) -> str:
        formatted_messages = [] if system_msg is None else [{"role": "system", "content": system_msg}]
        for message in messages:
            if isinstance(message, dict):
                formatted_messages.append(message)
            elif isinstance(message, Message):
                formatted_messages.append(to_dict(message))
            else:
                raise ValueError(f"Invalid message type: {type(message)}")
        response = await self.llm.chat.completions.create(messages=formatted_messages, model=self.model_name, max_tokens=4096, temperature=0.3,stream=False)
        return response.choices[0].message.content
    
    # @retry(stop=stop_after_attempt(3), wait=wait_random_exponential(min=1, max=60))
    async def ask_tool(self,messages: List[Union[dict, Message]], system_msg: Optional[str] = None, tools: Optional[List[dict]] = None, tool_choice: Optional[str] = None, **kwargs):
        if tool_choice not in ["auto", "none", "required"]:
            raise ValueError(f"Invalid tool choice: {tool_choice}")
        if tools is None:
            tools = []
        
        formatted_messages = [] if system_msg is None else [{"role": "system", "content": system_msg}]
        for message in messages:
            if isinstance(message, dict):
                formatted_messages.append(message)
            elif isinstance(message, Message):
                formatted_messages.append(to_dict(message))
            else:
                raise ValueError(f"Invalid message type: {type(message)}")
        try:
            response = await self.llm.chat.completions.create(messages=formatted_messages, model=self.model_name, max_tokens=4096, temperature=0.3,stream=False, tools=tools, tool_choice=tool_choice, **kwargs)
            return response.choices[0].message
        except Exception as e:
            logger.error(f"Error during tool call: {e}")
            raise e
