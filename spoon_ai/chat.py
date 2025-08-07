from logging import getLogger
from typing import List, Optional, Union
import asyncio

from spoon_ai.schema import Message, LLMResponse
from spoon_ai.llm.manager import get_llm_manager
from spoon_ai.llm.errors import ConfigurationError
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
    def __init__(self, use_llm_manager: bool = True, model_name: str = None, llm_provider: str = None, api_key: str = None, base_url: str = None, **kwargs):
        """Initialize ChatBot with hierarchical configuration priority system.

        Configuration Priority System:
        1. Full manual override (highest priority) - all params provided
        2. Partial override with config fallback - llm_provider provided, api_key from config
        3. Full config-based loading - only use_llm_manager=True, loads from config.json

        Args:
            use_llm_manager: Enable LLM manager architecture (default: True)
            model_name: Model name override
            llm_provider: Provider name override
            api_key: API key override
            base_url: Base URL override
            **kwargs: Additional parameters
        """
        self.use_llm_manager = use_llm_manager
        self.model_name = model_name
        self.llm_provider = llm_provider
        self.api_key = api_key
        self.base_url = base_url
        self.llm_manager = None

        # Store original parameters for priority mode detection
        self._original_llm_provider = llm_provider
        self._original_api_key = api_key

        if not self.use_llm_manager:
            logger.warning("use_llm_manager=False is deprecated. LLM Manager architecture is now required.")
            # Force use of LLM manager for compatibility
            self.use_llm_manager = True

        # Initialize based on configuration priority
        self._initialize_with_priority()

        logger.info(f"ChatBot initialized with LLM Manager architecture (priority mode: {self._get_priority_mode()})")

    def _get_priority_mode(self) -> str:
        """Determine which priority mode is being used."""
        # Use original parameters to determine mode (before config loading)
        if self._original_api_key and self._original_llm_provider:
            return "full_manual_override"
        elif self._original_llm_provider and not self._original_api_key:
            return "partial_override_with_config_fallback"
        elif self.use_llm_manager and not self._original_llm_provider:
            return "full_config_based_loading"
        else:
            return "default_config_loading"

    def _initialize_with_priority(self) -> None:
        """Initialize ChatBot based on configuration priority system."""
        from spoon_ai.llm.config import ConfigurationManager

        # Always initialize LLM manager
        self.llm_manager = get_llm_manager()

        # Priority 1: Full manual override (highest priority)
        if self.api_key and self.llm_provider:
            logger.info("Using full manual override mode")
            self._apply_manual_override()
            return

        # Priority 2: Partial override with config fallback
        if self.llm_provider and not self.api_key:
            logger.info("Using partial override with config fallback mode")
            self._apply_partial_override_with_config()
            return

        # Priority 3: Full config-based loading
        if self.use_llm_manager and not self.llm_provider:
            logger.info("Using full config-based loading mode")
            self._apply_full_config_loading()
            return

        # Default: Use config with any provided overrides
        logger.info("Using default config loading with overrides")
        self._apply_default_config_loading()

    def _apply_manual_override(self) -> None:
        """Apply full manual override configuration."""
        # All parameters provided manually - highest priority
        self._update_provider_config(
            provider=self.llm_provider,
            api_key=self.api_key,
            base_url=self.base_url,
            model_name=self.model_name
        )
        logger.info(f"Applied manual override for provider: {self.llm_provider}")

    def _apply_partial_override_with_config(self) -> None:
        """Apply partial override with config fallback."""
        from spoon_ai.llm.config import ConfigurationManager

        config_manager = ConfigurationManager()

        # Get config values for the specified provider
        try:
            provider_config = config_manager._get_provider_config_dict(self.llm_provider)

            # Use config values for missing parameters
            config_api_key = provider_config.get('api_key')
            config_base_url = provider_config.get('base_url')
            config_model = provider_config.get('model')

            # Validate provider consistency: ensure API key matches requested provider
            final_api_key = self.api_key or config_api_key
            if final_api_key and not self._validate_provider_api_key_match(self.llm_provider, final_api_key):
                # Get available providers for helpful error message
                available_providers = config_manager.list_configured_providers()
                logger.error(f"Provider/API key mismatch detected: requested '{self.llm_provider}' but API key appears to be for different provider")
                raise ConfigurationError(
                    f"API key mismatch for provider '{self.llm_provider}'. "
                    f"Please ensure the API key in your configuration matches the requested provider. "
                    f"Available providers: {available_providers}",
                    config_key=self.llm_provider,
                    context={
                        "requested_provider": self.llm_provider,
                        "available_providers": available_providers,
                        "api_key_prefix": final_api_key[:10] + "..." if final_api_key else None
                    }
                )

            # Apply configuration with manual overrides taking priority
            self._update_provider_config(
                provider=self.llm_provider,
                api_key=final_api_key,
                base_url=self.base_url or config_base_url,
                model_name=self.model_name or config_model
            )

            logger.info(f"Applied partial override with config fallback for provider: {self.llm_provider}")

        except Exception as e:
            logger.error(f"Failed to load config for provider {self.llm_provider}: {e}")
            # Fallback to manual values only
            self._update_provider_config(
                provider=self.llm_provider,
                api_key=self.api_key,
                base_url=self.base_url,
                model_name=self.model_name
            )

    def _apply_full_config_loading(self) -> None:
        """Apply full config-based loading using default provider and fallback chain."""
        from spoon_ai.llm.config import ConfigurationManager

        config_manager = ConfigurationManager()

        try:
            # Use default provider from config
            default_provider = config_manager.get_default_provider()
            fallback_chain = config_manager.get_fallback_chain()

            if default_provider:
                self.llm_provider = default_provider
                logger.info(f"Using default provider from config: {default_provider}")

            if fallback_chain:
                self.llm_manager.set_fallback_chain(fallback_chain)
                logger.info(f"Set fallback chain from config: {fallback_chain}")

        except Exception as e:
            logger.error(f"Failed to load full config: {e}")
            # Let LLM manager handle provider selection
            pass

    def _apply_default_config_loading(self) -> None:
        """Apply default config loading with any provided overrides."""
        # Apply any manual overrides if provided
        if self.api_key or self.base_url or self.model_name:
            self._update_provider_config(
                provider=self.llm_provider,
                api_key=self.api_key,
                base_url=self.base_url,
                model_name=self.model_name
            )

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

    def _validate_provider_api_key_match(self, provider_name: str, api_key: str) -> bool:
        """Validate that an API key belongs to the specified provider family.

        Args:
            provider_name: Name of the provider (e.g., 'openai', 'anthropic')
            api_key: API key to validate

        Returns:
            bool: True if API key matches provider, False otherwise
        """
        if not api_key or not provider_name:
            return True  # Skip validation for empty values

        # Provider family mapping based on API key prefixes
        provider_families = {
            'openai': ['sk-'],
            'anthropic': ['sk-ant-'],
            'openrouter': ['sk-or-'],
            'deepseek': ['sk-'],
            'gemini': ['AIza']
        }

        expected_prefixes = provider_families.get(provider_name.lower(), [])
        if not expected_prefixes:
            # Unknown provider - skip validation
            logger.debug(f"Unknown provider '{provider_name}', skipping API key validation")
            return True

        # Check if API key starts with any expected prefix
        is_match = any(api_key.startswith(prefix) for prefix in expected_prefixes)

        if not is_match:
            logger.debug(f"API key validation failed: provider '{provider_name}' expects prefixes {expected_prefixes}, got key starting with '{api_key[:10]}...'")
        else:
            logger.debug(f"API key validation passed for provider '{provider_name}'")

        return is_match

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
