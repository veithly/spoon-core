import json
from enum import Enum
from typing import Any, List, Literal, Optional, Union

from pydantic import BaseModel, Field


class Function(BaseModel):
    name: str
    arguments: str
    
    def get_arguments_dict(self) -> dict:
        """Parse arguments string to dictionary.
        
        Returns:
            dict: Parsed arguments as dictionary
        """
        if isinstance(self.arguments, str):
            arguments = self.arguments.strip()
            if not arguments:
                return {}
            try:
                return json.loads(arguments)
            except json.JSONDecodeError:
                return {}
        elif isinstance(self.arguments, dict):
            return self.arguments
        else:
            return {}
    
    @classmethod
    def create(cls, name: str, arguments: Union[str, dict]) -> "Function":
        """Create Function with arguments as string or dict.
        
        Args:
            name: Function name
            arguments: Function arguments as string or dict
            
        Returns:
            Function: Function instance with arguments as JSON string
        """
        if isinstance(arguments, dict):
            arguments_str = json.dumps(arguments)
        else:
            arguments_str = str(arguments)
        
        return cls(name=name, arguments=arguments_str)

class ToolCall(BaseModel):
    id: str
    type: str = "function"
    function: Function

class AgentState(str, Enum):
    """
    The state of the agent.
    """
    IDLE = "IDLE"
    RUNNING = "RUNNING"
    FINISHED = "FINISHED"
    ERROR = "ERROR"

class ToolChoice(str, Enum):
    """Tool choice options"""
    NONE = "none"
    AUTO = "auto"
    REQUIRED = "required"


class Role(str, Enum):
    """Message role options"""
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"

ROLE_VALUES = tuple(role.value for role in Role)
ROLE_TYPE = Literal[ROLE_VALUES]  # type: ignore


class Message(BaseModel):
    """Represents a chat message in the conversation"""

    role: ROLE_TYPE = Field(...) # type: ignore
    content: Optional[str] = Field(default=None)
    tool_calls: Optional[List[ToolCall]] = Field(default=None)
    name: Optional[str] = Field(default=None)
    tool_call_id: Optional[str] = Field(default=None)


TOOL_CHOICE_VALUES = tuple(choice.value for choice in ToolChoice)
TOOL_CHOICE_TYPE = Literal[TOOL_CHOICE_VALUES] # type: ignore

class LLMConfig(BaseModel):
    """Configuration for LLM providers"""
    model: str = ""
    api_key: str = ""
    base_url: Optional[str] = None
    api_type: Optional[str] = None
    api_version: Optional[str] = None
    max_tokens: int = 4096
    temperature: float = 0.3

class LLMResponse(BaseModel):
    """Unified LLM response model"""
    content: str
    text: str = ""  # Original text response
    image_paths: List[dict] = Field(default_factory=list)
    tool_calls: List[ToolCall] = Field(default_factory=list)
    finish_reason: Optional[str] = Field(default=None)
    native_finish_reason: Optional[str] = Field(default=None)
