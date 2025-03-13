from typing import List, Optional, Union

from langchain_core.messages import BaseMessage
from langchain_deepseek import ChatDeepSeek
from langchain_openai import ChatOpenAI
from tenacity import retry, stop_after_attempt, wait_random_exponential
from pydantic import BaseModel, Field

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


class ChatBot:
    def __init__(self, model_name: str = "gpt-4.5-preview", llm_config: dict = None, llm_provider: str = "openai", api_key: str = None):
        if llm_provider == "openai":
            self.llm = ChatOpenAI(model=model_name, **llm_config, api_key=api_key, streaming=True)
        elif llm_provider == "deepseek":
            self.llm = ChatDeepSeek(model=model_name, **llm_config, api_key=api_key, streaming=True)
        else:
            raise ValueError(f"Invalid LLM provider: {llm_provider}")
    
    async def ask(self, messages: List[Union[dict, BaseMessage]]) -> str:
        response = await self.llm.ainvoke(messages)
        return response.content
    
    @retry(stop=stop_after_attempt(3), wait=wait_random_exponential(min=1, max=60))
    async def ask_tool(self, timeout: int=60, tools: Optional[List[dict]] = None, tool_choice: Optional[str] = None, **kwargs):
        ...
