from logging import getLogger
from typing import List, Optional, Union
import asyncio

from spoon_ai.schema import Message, LLMResponse
from spoon_ai.llm.manager import get_llm_manager
from pydantic import BaseModel, Field

logger = getLogger(__name__)


class Memory(BaseModel):
    messages: List[Message] = Field(default_factory=list)
    max_messages: int = 100

    def add_message(self, message: Message) -> None:
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
    def __init__(self, model_name: str = None, llm_provider: str = None, api_key: str = None, base_url: str = None, **kwargs):
        """Initialize ChatBot with LLM Manager architecture only."""
        self.model_name = model_name
        self.llm_provider = llm_provider
        self.api_key = api_key
        self.base_url = base_url
        
        # Initialize LLM manager
        self.llm_manager = get_llm_manager()
        
        # If api_key and/or base_url are provided, update the provider configuration
        if self.api_key or self.base_url:
            self._update_provider_config(self.llm_provider, self.api_key, self.base_url, self.model_name)
        
        logger.info("ChatBot initialized with LLM Manager architecture")

    def _update_provider_config(self, provider: str, api_key: str = None, base_url: str = None, model_name: str = None):
        """Update provider configuration in the LLM manager."""
        if not provider:
            logger.warning("No provider specified for configuration update")
            return
            
        try:
            # Get the current configuration manager
            config_manager = self.llm_manager.config_manager
            
            # Create a temporary configuration update
            config_updates = {}
            if api_key:
                config_updates['api_key'] = api_key
            if base_url:
                config_updates['base_url'] = base_url
            if model_name:
                config_updates['model'] = model_name
                
            # Update the provider configuration in memory
            if hasattr(config_manager, '_provider_configs'):
                if provider in config_manager._provider_configs:
                    # Update existing config
                    existing_config = config_manager._provider_configs[provider]
                    for key, value in config_updates.items():
                        setattr(existing_config, key, value)
                    logger.info(f"Updated existing provider config for {provider}")
                else:
                    # Create new config
                    from spoon_ai.llm.config import ProviderConfig
                    new_config = ProviderConfig(
                        name=provider,
                        api_key=api_key or '',
                        base_url=base_url,
                        model=model_name or '',
                        max_tokens=4096,
                        temperature=0.3,
                        timeout=30,
                        retry_attempts=3,
                        custom_headers={},
                        extra_params={}
                    )
                    config_manager._provider_configs[provider] = new_config
                    logger.info(f"Created new provider config for {provider}")
            
        except Exception as e:
            logger.error(f"Failed to update provider configuration: {e}")

    async def ask(self, messages: List[Union[dict, Message]], system_msg: Optional[str] = None, output_queue: Optional[asyncio.Queue] = None) -> str:
        """Ask method using the LLM manager architecture."""
        # Convert messages to the expected format
        formatted_messages = []
        if system_msg:
            formatted_messages.append(Message(role="system", content=system_msg))

        for message in messages:
            if isinstance(message, dict):
                formatted_messages.append(Message(**message))
            elif isinstance(message, Message):
                formatted_messages.append(message)
            else:
                raise ValueError(f"Invalid message type: {type(message)}")

        # Use LLM manager for the request
        response = await self.llm_manager.chat(
            messages=formatted_messages,
            provider=self.llm_provider
        )

        return response.content

    async def ask_tool(self, messages: List[Union[dict, Message]], system_msg: Optional[str] = None, tools: Optional[List[dict]] = None, tool_choice: Optional[str] = None, output_queue: Optional[asyncio.Queue] = None, **kwargs) -> LLMResponse:
        """Ask tool method using the LLM manager architecture."""
        # Convert messages to the expected format
        formatted_messages = []
        if system_msg:
            formatted_messages.append(Message(role="system", content=system_msg))

        for message in messages:
            if isinstance(message, dict):
                formatted_messages.append(Message(**message))
            elif isinstance(message, Message):
                formatted_messages.append(message)
            else:
                raise ValueError(f"Invalid message type: {type(message)}")

        # Use LLM manager for the tool request
        response = await self.llm_manager.chat_with_tools(
            messages=formatted_messages,
            tools=tools or [],
            provider=self.llm_provider,
            **kwargs
        )

        return response
