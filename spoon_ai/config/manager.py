"""Configuration manager for unified tool configuration."""

import json
import logging
import os
from typing import Any, Dict, List, Optional, Tuple

from .models import (
    SpoonConfig, AgentConfig, ToolConfig, MCPServerConfig
)
from .mcp_manager import MCPServerManager
from .tool_factory import ToolFactory
from .errors import ConfigurationError, ValidationError
from ..tools.base import BaseTool

logger = logging.getLogger(__name__)


class ConfigManager:
    """Manages SpoonAI configuration with unified tool configuration support."""
    
    def __init__(self, config_path: str = "config.json"):
        self.config_path = config_path
        self.config: Optional[SpoonConfig] = None
        self.mcp_manager = MCPServerManager()
        self.tool_factory = ToolFactory(self.mcp_manager)

    
    def load_config(self) -> SpoonConfig:
        """Load and validate configuration from file."""
        try:
            if not os.path.exists(self.config_path):
                logger.warning(f"Config file not found: {self.config_path}, using defaults")
                self.config = SpoonConfig()
                return self.config
            
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config_data = json.load(f)
            
            # Validate and create configuration
            self.config = SpoonConfig(**config_data)
            
            logger.info(f"Loaded configuration from {self.config_path}")
            return self.config
            
        except json.JSONDecodeError as e:
            raise ConfigurationError(f"Invalid JSON in config file: {str(e)}")
        except ValidationError as e:
            raise ConfigurationError(f"Configuration validation failed: {str(e)}")
        except Exception as e:
            raise ConfigurationError(f"Failed to load configuration: {str(e)}")
    
    def save_config(self, config: SpoonConfig = None) -> None:
        """Save configuration to file."""
        if config is None:
            config = self.config
        
        if config is None:
            raise ConfigurationError("No configuration to save")
        
        try:
            config_data = config.model_dump(exclude_none=True, by_alias=True)
            
            with open(self.config_path, 'w', encoding='utf-8') as f:
                json.dump(config_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Saved configuration to {self.config_path}")
            
        except Exception as e:
            raise ConfigurationError(f"Failed to save configuration: {str(e)}")
    
    async def load_agent_tools(self, agent_name: str) -> List[BaseTool]:
        """Load and create tool instances for an agent."""
        if not self.config:
            raise ConfigurationError("Configuration not loaded")
        
        if agent_name not in self.config.agents:
            raise ConfigurationError(f"Agent not found: {agent_name}")
        
        agent_config = self.config.agents[agent_name]
        tools = []
        
        try:
            for tool_config in agent_config.tools:
                try:
                    tool = await self.tool_factory.create_tool(tool_config)
                    if tool:  # Skip disabled tools
                        tools.append(tool)
                        logger.info(f"Created tool: {tool_config.name}")
                except Exception as e:
                    logger.error(f"Failed to create tool {tool_config.name}: {e}")
                    # Continue with other tools instead of failing completely
                    continue
            
            logger.info(f"Loaded {len(tools)} tools for agent {agent_name}")
            return tools
            
        except Exception as e:
            raise ConfigurationError(f"Failed to load tools for agent {agent_name}: {str(e)}")
    
    def get_agent_config(self, agent_name: str) -> AgentConfig:
        """Get configuration for a specific agent."""
        if not self.config:
            raise ConfigurationError("Configuration not loaded")
        
        # Check direct name match
        if agent_name in self.config.agents:
            return self.config.agents[agent_name]
        
        # Check aliases
        for name, agent_config in self.config.agents.items():
            if agent_name in agent_config.aliases:
                return agent_config
        
        # Check default agent
        if agent_name == "default" and self.config.default_agent:
            return self.get_agent_config(self.config.default_agent)
        
        raise ConfigurationError(f"Agent not found: {agent_name}")
    
    def list_agents(self) -> Dict[str, Dict[str, Any]]:
        """List all available agents with their metadata."""
        if not self.config:
            raise ConfigurationError("Configuration not loaded")
        
        agents = {}
        for name, agent_config in self.config.agents.items():
            agents[name] = {
                "class": agent_config.class_name,
                "description": agent_config.description,
                "aliases": agent_config.aliases,
                "tool_count": len(agent_config.tools),
                "tools": [tool.name for tool in agent_config.tools if tool.enabled]
            }
        
        return agents
    
    def validate_configuration(self) -> List[str]:
        """Validate the current configuration and return any issues."""
        if not self.config:
            return ["Configuration not loaded"]
        
        issues = []
        
        # Validate agents
        for agent_name, agent_config in self.config.agents.items():
            try:
                # Validate tool configurations
                for tool_config in agent_config.tools:
                    if tool_config.type == "mcp" and not tool_config.mcp_server:
                        issues.append(f"Agent {agent_name}: MCP tool {tool_config.name} missing server config")
                    
                    # Check for required environment variables
                    if tool_config.mcp_server:
                        for env_var, value in tool_config.mcp_server.env.items():
                            if not value or value.startswith("your-") or value == "":
                                issues.append(f"Agent {agent_name}: Tool {tool_config.name} missing environment variable {env_var}")
                
            except Exception as e:
                issues.append(f"Agent {agent_name}: Validation error - {str(e)}")
        
        # Validate default agent exists
        if self.config.default_agent and self.config.default_agent not in self.config.agents:
            issues.append(f"Default agent '{self.config.default_agent}' not found")
        
        return issues
    
    async def cleanup(self) -> None:
        """Clean up resources (stop MCP servers, etc.)."""
        try:
            await self.mcp_manager.stop_all_servers()
            logger.info("Cleaned up all MCP servers")
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get a configuration value by key for backward compatibility."""
        if not self.config:
            # Load config if not already loaded
            self.load_config()
        
        if not self.config:
            return default
        
        # Handle specific keys that the CLI expects
        if key == "default_agent":
            return getattr(self.config, 'default_agent', default)
        
        # Handle nested access with dot notation (e.g., "api_keys.openai")
        if '.' in key:
            parts = key.split('.')
            value = self.config
            for part in parts:
                if hasattr(value, part):
                    value = getattr(value, part)
                elif isinstance(value, dict) and part in value:
                    value = value[part]
                else:
                    return default
            return value
        
        # Try to get from config attributes
        if hasattr(self.config, key):
            return getattr(self.config, key)
        
        # Try to get from config dict representation
        config_dict = self.config.model_dump()
        if key in config_dict:
            return config_dict[key]
        
        return default
    
