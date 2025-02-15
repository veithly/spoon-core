from typing import Any

from langchain_anthropic import ChatAnthropic
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_deepseek import ChatDeepSeek
from langchain_openai import ChatOpenAI
from models.base.model import Model
from pydantic import BaseModel

from utils import (ANTHROPIC_API_KEY, DEEPSEEK_API_KEY, OPENAI_API_KEY,
                   get_llm_type)


class Workflow(BaseModel):
    session_id: str = None
    model: Model = None
    chains: list[BaseChatModel] = []
    callback: Any = None

    def __init__(self, model: Model, session_id: str, callback: Any):
        super().__init__()
        self.model = model
        self.session_id = session_id
        self.callback = callback
        self.__prepare_chain()

    def __prepare_chain(self):
        assert hasattr(self, "model"), "Model not found"
        assert hasattr(self, "session_id"), "Session ID not found"
        self.chains = []
        for chain in self.model.chains:
            llm_type = get_llm_type(chain.llm.name)
            if llm_type == "openai":
                self.chains.append(
                    ChatOpenAI(
                        api_key=OPENAI_API_KEY,
                        model=chain.llm.name,
                        max_tokens=chain.llm.max_tokens,
                        temperature=chain.llm.temperature,
                        top_p=chain.llm.top_p,
                        frequency_penalty=chain.llm.frequency_penalty,
                        presence_penalty=chain.llm.presence_penalty,
                        streaming=True,
                        callbacks=[self.callback],
                    )
                )
            elif llm_type == "deepseek":
                self.chains.append(
                    ChatDeepSeek(
                        api_key = DEEPSEEK_API_KEY,
                        streaming=True,
                        callbacks=[self.callback],
                        temperature=chain.llm.temperature,
                        max_tokens=chain.llm.max_tokens,
                        model=chain.llm.name,
                        top_p=chain.llm.top_p,
                        frequency_penalty=chain.llm.frequency_penalty,
                        presence_penalty=chain.llm.presence_penalty,
                    )
                )
            elif llm_type == "anthropic":
                self.chains.append(
                    ChatAnthropic(
                        api_key = ANTHROPIC_API_KEY,
                        streaming=True,
                        callbacks=[self.callback],
                        temperature=chain.llm.temperature,
                        max_tokens=chain.llm.max_tokens,
                        model=chain.llm.name,
                        top_p=chain.llm.top_p,
                        frequency_penalty=chain.llm.frequency_penalty,
                        presence_penalty=chain.llm.presence_penalty,
                    )
                )

    async def agenerate(self, messages: list[BaseMessage]):
        async for message in self.chains[0].astream(input=messages):
            yield message
