import os
from logging import getLogger
from typing import List, Optional, Union

from spoon_ai.schema import Message, LLMResponse, ToolCall

from openai import AsyncOpenAI
from anthropic import AsyncAnthropic
from httpx import AsyncClient
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
        self.llm_provider = llm_provider
        self.model_name = model_name
        self.api_key = api_key
        self.llm_config = llm_config
        
        if llm_provider == "openai":
            self.llm = AsyncOpenAI()
            self.llm.api_key = api_key if api_key else os.getenv("OPENAI_API_KEY")
        elif llm_provider == "anthropic":
            http_client = AsyncClient(follow_redirects=True)
            self.llm = AsyncAnthropic(api_key=api_key if api_key else os.getenv("ANTHROPIC_API_KEY"), http_client=http_client)
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
        
        if self.llm_provider == "openai":
            response = await self.llm.chat.completions.create(messages=formatted_messages, model=self.model_name, max_tokens=4096, temperature=0.3, stream=False)
            return response.choices[0].message.content
        elif self.llm_provider == "anthropic":
            response = await self.llm.messages.create(
                model=self.model_name,
                max_tokens=4096,
                temperature=0.3,
                system=system_msg,
                messages=[m for m in formatted_messages if m.get("role") != "system"]
            )
            return response.content[0].text
    
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
            if self.llm_provider == "openai":
                response = await self.llm.chat.completions.create(
                    messages=formatted_messages, 
                    model=self.model_name, 
                    max_tokens=4096, 
                    temperature=0.3,
                    stream=False, 
                    tools=tools, 
                    tool_choice=tool_choice, 
                    **kwargs
                )
                return response.choices[0].message
            elif self.llm_provider == "anthropic":
                def to_anthropic_tools(tools: List[dict]) -> List[dict]:
                    return [{"name": tool["function"]["name"], "description": tool["function"]["description"], "input_schema": tool["function"]["parameters"]} for tool in tools]
                content = ""
                buffer = ""
                buffer_type = ""
                current_tool = None
                tool_calls = []
                async with self.llm.messages.stream(
                    model=self.model_name,
                    max_tokens=4096,
                    temperature=0.3,
                    system=system_msg,
                    messages=[m for m in formatted_messages if m.get("role") != "system"],
                    tools=to_anthropic_tools(tools),
                    **kwargs
                ) as stream:
                    async for chunk in stream:
                        if chunk.type in ["message_start", "text", "input_json", "message_delta", "message_stop"]:
                            continue
                        if chunk.type == "content_block_start":
                            buffer_type = chunk.content_block.type
                            if buffer_type == "tool_use":
                                current_tool = {
                                    "id": chunk.content_block.id,
                                    "function": {
                                        "name": chunk.content_block.name,
                                        "arguments": {}
                                    }
                                }
                                continue
                        if chunk.type == "content_block_delta" and chunk.delta.type == "text_delta":
                            buffer += chunk.delta.text
                            continue
                        if chunk.type == "content_block_delta" and chunk.delta.type == "input_json_delta":
                            buffer += chunk.delta.partial_json

                        if chunk.type == "content_block_stop":
                            content += buffer
                            if buffer_type == "tool_use":
                                current_tool["function"]["arguments"] = buffer
                                current_tool = ToolCall(**current_tool)
                                tool_calls.append(current_tool)
                            buffer = ""
                            buffer_type = ""
                            current_tool = None
                return LLMResponse(content=content, tool_calls=tool_calls)
        except Exception as e:
            logger.error(f"Error during tool call: {e}")
            raise e
