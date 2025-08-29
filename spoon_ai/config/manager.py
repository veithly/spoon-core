"""Configuration manager for unified tool configuration."""

import asyncio
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
                tool_created = False
                max_retries = 2 if tool_config.type == "mcp" else 1
                
                for attempt in range(max_retries):
                    try:
                        tool = await self.tool_factory.create_tool(tool_config)
                        if tool:  # Skip disabled tools
                            tools.append(tool)
                            logger.info(f"Created tool: {tool_config.name}")
                            tool_created = True
                            break
                    except Exception as e:
                        if attempt < max_retries - 1:
                            logger.warning(f"Failed to create tool {tool_config.name} (attempt {attempt + 1}/{max_retries}): {e}")
                            logger.info(f"Retrying tool {tool_config.name}...")
                            await asyncio.sleep(1)  # Wait 1 second before retry
                        else:
                            logger.error(f"Failed to create tool {tool_config.name} after {max_retries} attempts: {e}")
                            # Continue with other tools instead of failing completely
                            continue
                
                if not tool_created:
                    logger.warning(f"Skipping tool {tool_config.name} due to creation failure")

            logger.info(f"Loaded {len(tools)} tools for agent {agent_name}")
            if len(tools) == 0:
                logger.warning(f"No tools were successfully loaded for agent {agent_name}")
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
        # Built-in fallback for common agents (works without config file)
        # Canonicalize aliases
        alias_map = {
            "spoon_react": "react",
        }
        canonical = alias_map.get(agent_name, agent_name)
        builtin_agents = {
            "react": AgentConfig(class_name="SpoonReactAI", aliases=["spoon_react"], description="A smart AI agent for blockchain operations", config={}, tools=[]),
            "spoon_react_mcp": AgentConfig(class_name="SpoonReactMCP", aliases=[], description="SpoonReact agent with MCP protocol support", config={}, tools=[]),
        }
        if canonical in builtin_agents:
            # Ensure config object exists
            if self.config is None:
                self.config = SpoonConfig()
            # Inject into in-memory agents map for downstream tool loading (empty tools by default)
            if canonical not in self.config.agents:
                self.config.agents[canonical] = builtin_agents[canonical]
            return self.config.agents[canonical]

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

    # --- Backward-compat helpers used by CLI ---
    def list_config(self) -> Dict[str, Any]:
        """Return the current configuration as a plain dictionary.

        Secrets are NOT masked here; masking is handled by the CLI caller.
        """
        if not self.config:
            self.load_config()
        if not self.config:
            # If still not available, return empty structure
            return {"api_keys": {}, "providers": {}, "llm_settings": {}, "agents": {}}
        return self.config.model_dump(exclude_none=True, by_alias=True)

    def set_api_key(self, provider: str, api_key: str) -> None:
        """Set API key for a provider and persist to disk."""
        if not self.config:
            self.load_config()
        if not self.config:
            self.config = SpoonConfig()
        # Ensure api_keys map exists
        if self.config.api_keys is None:
            self.config.api_keys = {}
        self.config.api_keys[provider] = api_key
        self.save_config(self.config)

    def set(self, key: str, value: Any) -> None:
        """Set a configuration value by key and persist to disk.

        Supports dot-notation for nested keys, e.g.:
          - providers.openai.model
          - llm_settings.default_provider
          - api_keys.openai (delegates to set_api_key)
        """
        if not self.config:
            self.load_config()
        if not self.config:
            self.config = SpoonConfig()

        # Route api_keys.* to set_api_key
        if key.startswith("api_keys."):
            provider = key.split(".", 1)[1]
            self.set_api_key(provider, value)
            return

        parts = key.split('.') if '.' in key else [key]

        # Handle top-level direct attributes of SpoonConfig
        top_level_keys = {"providers", "llm_settings", "default_agent", "agents", "RPC_URL", "SCAN_URL", "CHAIN_ID"}
        if parts[0] in top_level_keys and len(parts) == 1:
            setattr(self.config, parts[0], value)
            self.save_config(self.config)
            return

        # Nested updates for providers and llm_settings
        if parts[0] in {"providers", "llm_settings"}:
            target: Dict[str, Any]
            if parts[0] == "providers":
                if self.config.providers is None:
                    self.config.providers = {}
                target = self.config.providers
            else:
                if self.config.llm_settings is None:
                    self.config.llm_settings = {}
                target = self.config.llm_settings

            # Ensure path exists
            current: Dict[str, Any] = target
            for sub_key in parts[1:-1]:
                if sub_key not in current or not isinstance(current[sub_key], dict):
                    current[sub_key] = {}
                current = current[sub_key]
            current[parts[-1]] = value
            self.save_config(self.config)
            return

        # Fallback: try to set attribute directly if present
        if hasattr(self.config, key):
            setattr(self.config, key, value)
            self.save_config(self.config)
            return

        # As a last resort, update the underlying dict and save
        cfg = self.config.model_dump()
        cfg[key] = value
        try:
            # Re-validate by constructing a new SpoonConfig
            self.config = SpoonConfig(**cfg)
        except Exception as e:
            raise ConfigurationError(f"Invalid configuration key or value: {key}={value} ({e})")
        self.save_config(self.config)

    @staticmethod
    def _is_placeholder_value(value: Optional[str]) -> bool:
        """Heuristically determine if a value looks like a placeholder."""
        if not value or not isinstance(value, str):
            return False
        lower = value.lower()
        return (
            lower.startswith("your-")
            or "<your" in lower
            or "replace" in lower
            or "changeme" in lower
            or value in {"", "sk-xxxx", "sk-test-xxxx"}
        )

    @staticmethod
    def _detect_legacy_config(config_data: Dict[str, Any]) -> bool:
        """Detect whether a given config dict uses legacy structure.

        Very lightweight detection used by the CLI 'check-config' helper.
        """
        if not isinstance(config_data, dict):
            return False
        legacy_keys = {"tool_sets", "mcp_servers"}
        if any(k in config_data for k in legacy_keys):
            return True
        # Legacy if agents are missing or not objects
        agents = config_data.get("agents")
        if agents is None:
            return True
        if isinstance(agents, dict):
            # If any agent entries are not dict-like, consider legacy
            for _name, val in agents.items():
                if not isinstance(val, dict):
                    return True
        return False

