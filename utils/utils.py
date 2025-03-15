from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from spoon_ai.schema import Message, Role
from typing import List

def get_llm_type(llm_name: str) -> str:
    if llm_name.startswith("gpt"):
        return "openai"
    elif llm_name.startswith("deepseek"):
        return "deepseek"
    elif llm_name.startswith("claude"):
        return "anthropic"
    
def to_langchain_messages(messages: List[Message]) -> List[BaseMessage]:
            langchain_messages = []
            for msg in messages:
                if msg.role == Role.USER:
                    langchain_messages.append(HumanMessage(content=msg.content))
                elif msg.role == Role.ASSISTANT:
                    langchain_messages.append(AIMessage(content=msg.content))
                elif msg.role == Role.SYSTEM:
                    langchain_messages.append(SystemMessage(content=msg.content))
            return langchain_messages