"""
Gemini Provider implementation for the unified LLM interface.
"""

import asyncio
import json
import os
import time
from typing import List, Dict, Any, Optional, AsyncGenerator
from logging import getLogger

from google import genai
from google.genai import types

from spoon_ai.schema import Message, ToolCall, Function
from ..interface import LLMProviderInterface, LLMResponse, ProviderMetadata, ProviderCapability
from ..errors import ProviderError, AuthenticationError, RateLimitError, ModelNotFoundError, NetworkError
from ..registry import register_provider

logger = getLogger(__name__)


@register_provider("gemini", [
    ProviderCapability.CHAT,
    ProviderCapability.COMPLETION,
    ProviderCapability.STREAMING,
    ProviderCapability.IMAGE_GENERATION,
    ProviderCapability.VISION
])
class GeminiProvider(LLMProviderInterface):
    """Gemini provider implementation."""
    
    def __init__(self):
        self.client: Optional[genai.Client] = None
        self.config: Dict[str, Any] = {}
        self.model: str = ""
        self.max_tokens: int = 4096
        self.temperature: float = 0.3
        
    async def initialize(self, config: Dict[str, Any]) -> None:
        """Initialize the Gemini provider with configuration."""
        try:
            self.config = config
            self.model = config.get('model', 'gemini-2.5-pro')
            self.max_tokens = config.get('max_tokens', 4096)
            self.temperature = config.get('temperature', 0.3)
            
            api_key = config.get('api_key')
            if not api_key:
                raise AuthenticationError("gemini", context={"config": config})
            
            self.client = genai.Client(api_key=api_key)
            
            logger.info(f"Gemini provider initialized with model: {self.model}")
            
        except Exception as e:
            if isinstance(e, (AuthenticationError, ProviderError)):
                raise
            raise ProviderError("gemini", f"Failed to initialize: {str(e)}", original_error=e)
    
    def _convert_messages(self, messages: List[Message]) -> tuple[Optional[str], str]:
        """Convert Message objects to Gemini format."""
        system_content = ""
        user_message = ""
        
        # Extract system messages
        for message in messages:
            if message.role == "system":
                if system_content:
                    system_content += " " + message.content
                else:
                    system_content = message.content
        
        # Get the last user message
        for message in reversed(messages):
            if message.role == "user":
                user_message = message.content
                break
        
        # If no user message found, use a default
        if not user_message:
            user_message = "Hello"
        
        return system_content, user_message
    
    async def chat(self, messages: List[Message], **kwargs) -> LLMResponse:
        """Send chat request to Gemini."""
        if not self.client:
            raise ProviderError("gemini", "Provider not initialized")
        
        try:
            start_time = asyncio.get_event_loop().time()
            
            system_content, user_message = self._convert_messages(messages)
            
            # Extract parameters
            model = kwargs.get('model', self.model)
            max_tokens = kwargs.get('max_tokens', self.max_tokens)
            temperature = kwargs.get('temperature', self.temperature)
            response_modalities = kwargs.get('response_modalities')
            
            # Build request content
            if isinstance(user_message, str):
                contents = [types.Part.from_text(text=user_message)]
            else:
                contents = [user_message]
            
            # Generate configuration
            generate_config = types.GenerateContentConfig(
                max_output_tokens=max_tokens,
                temperature=temperature
            )
            
            # Add system instruction if available
            if system_content:
                generate_config.system_instruction = system_content
            
            # Add response modalities if specified
            if response_modalities:
                generate_config.response_modalities = response_modalities
            
            # Check for structured output requirements
            if "IMPORTANT INSTRUCTION" in system_content and "JSON format" in system_content:
                schema = {
                    "type": "object",
                    "properties": {
                        "response": {
                            "type": "string",
                            "description": "Response content to the user"
                        },
                        "should_hand_off": {
                            "type": "boolean",
                            "description": "Whether the conversation should be handed off to a design expert"
                        }
                    },
                    "required": ["response", "should_hand_off"],
                    "propertyOrdering": ["should_hand_off", "response"]
                }
                generate_config.response_schema = schema
                generate_config.response_mime_type = 'application/json'
            
            # Send request
            response = self.client.models.generate_content(
                model=model,
                contents=contents,
                config=generate_config
            )
            
            duration = asyncio.get_event_loop().time() - start_time
            return self._convert_response(response, duration)
            
        except Exception as e:
            await self._handle_error(e)
    
    async def chat_stream(self, messages: List[Message], **kwargs) -> AsyncGenerator[str, None]:
        """Send streaming chat request to Gemini."""
        if not self.client:
            raise ProviderError("gemini", "Provider not initialized")
        
        try:
            system_content, user_message = self._convert_messages(messages)
            
            # Extract parameters
            model = kwargs.get('model', self.model)
            max_tokens = kwargs.get('max_tokens', self.max_tokens)
            temperature = kwargs.get('temperature', self.temperature)
            
            # Build request content
            if isinstance(user_message, str):
                contents = [types.Part.from_text(text=user_message)]
            else:
                contents = [user_message]
            
            # Generate configuration
            generate_config = types.GenerateContentConfig(
                max_output_tokens=max_tokens,
                temperature=temperature
            )
            
            if system_content:
                generate_config.system_instruction = system_content
            
            # Send streaming request
            stream = self.client.models.generate_content_stream(
                model=model,
                contents=contents,
                config=generate_config
            )
            
            for part_response in stream:
                if part_response.candidates and part_response.candidates[0].content.parts:
                    chunk = part_response.candidates[0].content.parts[0].text
                    if chunk:
                        yield chunk
                        
        except Exception as e:
            await self._handle_error(e)
    
    async def completion(self, prompt: str, **kwargs) -> LLMResponse:
        """Send completion request to Gemini."""
        # Convert to chat format
        messages = [Message(role="user", content=prompt)]
        return await self.chat(messages, **kwargs)
    
    async def chat_with_tools(self, messages: List[Message], tools: List[Dict], **kwargs) -> LLMResponse:
        """Send chat request with tools to Gemini."""
        # Gemini doesn't support tool calls in the same way as OpenAI/Anthropic
        # We'll add tool descriptions to the system message and use regular chat
        logger.warning("Gemini doesn't support native tool calls, adding tool descriptions to system message")
        
        system_content, user_message = self._convert_messages(messages)
        
        # Add tool descriptions to system content
        if tools:
            tools_desc = "Available tools:\n"
            for tool in tools:
                if 'function' in tool:
                    func = tool['function']
                    tools_desc += f"- {func.get('name', 'unknown')}: {func.get('description', 'No description')}\n"
            
            if system_content:
                system_content += "\n\n" + tools_desc
            else:
                system_content = tools_desc
        
        # Create modified messages with enhanced system content
        modified_messages = []
        if system_content:
            modified_messages.append(Message(role="system", content=system_content))
        
        # Add non-system messages
        for message in messages:
            if message.role != "system":
                modified_messages.append(message)
        
        return await self.chat(modified_messages, **kwargs)
    
    def _convert_response(self, response, duration: float) -> LLMResponse:
        """Convert Gemini response to standardized LLMResponse."""
        content = ""
        response_text = ""
        image_paths = []
        tool_calls = []
        
        # Check if there are candidate results
        if hasattr(response, "candidates") and response.candidates:
            candidate = response.candidates[0]
            if hasattr(candidate, "content") and candidate.content:
                # Iterate through all parts
                for part in candidate.content.parts:
                    # Check if there is text content
                    if hasattr(part, "text") and part.text:
                        if content:
                            content += "\n" + part.text
                        else:
                            content = part.text
                        response_text = content
                    # Check if there is image content
                    elif hasattr(part, "inline_data"):
                        # Save image data
                        image_data = {
                            "mime_type": getattr(part.inline_data, "mime_type", "image/jpeg"),
                            "data": getattr(part.inline_data, "data", b"")
                        }
                        
                        # Save image to local storage
                        output_dir = "runtime-image-output"
                        os.makedirs(output_dir, exist_ok=True)
                        
                        timestamp = int(time.time())
                        ext = image_data['mime_type'].split('/')[-1]
                        filename = f"gemini_image_{timestamp}.{ext}"
                        filepath = os.path.join(output_dir, filename)
                        
                        try:
                            with open(filepath, "wb") as f:
                                f.write(image_data['data'])
                            
                            image_paths.append({
                                "filepath": filepath,
                                "mime_type": image_data['mime_type']
                            })
                            
                            # Add to tool calls for compatibility
                            tool_calls.append({
                                "type": "image",
                                "image": {
                                    "mime_type": image_data["mime_type"],
                                    "data": image_data["data"]
                                }
                            })
                            
                            # Update content
                            if content:
                                content += f"\n[Generated image saved to: {filepath}]"
                            else:
                                content = f"[Generated image saved to: {filepath}]"
                            response_text = content
                            
                        except Exception as e:
                            logger.error(f"Failed to save image: {str(e)}")
        
        # If no content was obtained from candidates, try using the text attribute
        if not content and hasattr(response, "text"):
            response_text = response.text
            content = response_text
        
        # Default content for image responses
        if not content and image_paths:
            content = "【Image response】"
        
        return LLMResponse(
            content=content or "",
            provider="gemini",
            model=self.model,
            finish_reason="stop",  # Gemini doesn't provide detailed finish reasons
            native_finish_reason="stop",
            tool_calls=[],  # Gemini doesn't support native tool calls
            duration=duration,
            metadata={
                "image_paths": image_paths,
                "has_images": len(image_paths) > 0
            }
        )
    
    def get_metadata(self) -> ProviderMetadata:
        """Get Gemini provider metadata."""
        return ProviderMetadata(
            name="gemini",
            version="1.0.0",
            capabilities=[
                ProviderCapability.CHAT,
                ProviderCapability.COMPLETION,
                ProviderCapability.STREAMING,
                ProviderCapability.IMAGE_GENERATION,
                ProviderCapability.VISION
            ],
            max_tokens=32768,  # Gemini Pro context limit
            supports_system_messages=True,
            rate_limits={
                "requests_per_minute": 60,
                "tokens_per_minute": 32000
            }
        )
    
    async def health_check(self) -> bool:
        """Check if Gemini provider is healthy."""
        if not self.client:
            return False
        
        try:
            # Simple test request
            contents = [types.Part.from_text(text="test")]
            config = types.GenerateContentConfig(max_output_tokens=1)
            
            response = self.client.models.generate_content(
                model=self.model,
                contents=contents,
                config=config
            )
            return True
        except Exception as e:
            logger.warning(f"Gemini health check failed: {e}")
            return False
    
    async def cleanup(self) -> None:
        """Cleanup Gemini provider resources."""
        # Gemini client doesn't require explicit cleanup
        self.client = None
        logger.info("Gemini provider cleaned up")
    
    async def _handle_error(self, error: Exception) -> None:
        """Handle and convert Gemini errors to standardized errors."""
        error_str = str(error).lower()
        
        if "authentication" in error_str or "api key" in error_str:
            raise AuthenticationError("gemini", context={"original_error": str(error)})
        elif "quota" in error_str or "rate limit" in error_str:
            raise RateLimitError("gemini", context={"original_error": str(error)})
        elif "model" in error_str and "not found" in error_str:
            raise ModelNotFoundError("gemini", self.model, context={"original_error": str(error)})
        elif "timeout" in error_str or "connection" in error_str:
            raise NetworkError("gemini", "Network error", original_error=error)
        else:
            raise ProviderError("gemini", f"Request failed: {str(error)}", original_error=error)