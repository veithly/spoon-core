from typing import Union, Dict, Any, Optional, List
import asyncio
import uuid
import os
import time
from contextlib import asynccontextmanager
import logging

from fastmcp.client.transports import (FastMCPTransport, PythonStdioTransport,
                                       SSETransport, WSTransport, NpxStdioTransport,
                                       FastMCPStdioTransport, UvxStdioTransport, StdioTransport)
from fastmcp.client import Client as MCPClient
from pydantic import Field

from .base import BaseTool
from ..agents.mcp_client_mixin import MCPClientMixin

logger = logging.getLogger(__name__)

class MCPTool(BaseTool, MCPClientMixin):
    # Declare public fields
    tool_name_mapping: Dict[str, str] = Field(default_factory=dict, description="Mapping of logical tool names to actual MCP tool names")
    mcp_config: Dict[str, Any] = Field(default_factory=dict, description="MCP transport and tool configuration")

    class Config:
        arbitrary_types_allowed = True

    def __init__(self,
                 name: str = "mcp_tool",
                 description: str = "MCP tool for calling external MCP servers",
                 parameters: dict = None,
                 tool_name_mapping: Dict[str, str] = None,
                 mcp_config: Dict[str, Any] = None):

        if mcp_config is None:
            raise ValueError("`mcp_config` is required to initialize an MCPTool.")

        transport_obj = self._create_transport_from_config(mcp_config)

        config_mapping = mcp_config.get('tool_name_mapping', {})
        final_mapping = {**config_mapping, **(tool_name_mapping or {})}

        BaseTool.__init__(
            self,
            name=name,
            description=description,
            parameters=parameters or {
                "type": "object",
                "properties": {
                    "tool_name": {
                        "type": "string",
                        "description": "Name of the MCP tool to call"
                    },
                    "arguments": {
                        "type": "object",
                        "description": "Arguments to pass to the MCP tool"
                    }
                },
                "required": ["tool_name"]
            },
            tool_name_mapping=final_mapping,
            mcp_config=mcp_config
        )

        MCPClientMixin.__init__(self, transport_obj)

        # Initialize state for lazy loading
        self._parameters_loaded = False
        self._parameters_loading = False
        self._last_health_check = 0
        self._health_check_interval = mcp_config.get('health_check_interval', 300)  # 5 minutes default
        self._connection_timeout = mcp_config.get('connection_timeout', 30)
        self._max_retries = mcp_config.get('max_retries', 3)

        # Do not perform async operations in constructor
        logger.info(f"Initialized MCP tool '{self.name}' with deferred parameter loading")

    def _create_transport_from_config(self, config: dict):
        """Create transport object from configuration dictionary."""
        command = config.get("command", "")
        args = config.get("args", [])
        env = config.get("env", {})

        # Merge environment variables
        merged_env = os.environ.copy()
        merged_env.update(env)

        # Set environment variables in the current process for MCP tools to use
        for key, value in env.items():
            os.environ[key] = value

        # Determine transport type based on command
        if command == "npx":
            # NpxStdioTransport expects package as first argument
            if not args:
                raise ValueError("No package specified in args for npx transport")
            package = args[0]
            additional_args = args[1:] if len(args) > 1 else []
            return NpxStdioTransport(package=package, args=additional_args, env_vars=env)
        elif command == "uvx":
            # UvxStdioTransport expects package as first argument
            if not args:
                raise ValueError("No package specified in args for uvx transport")
            package = args[0]
            additional_args = args[1:] if len(args) > 1 else []
            return UvxStdioTransport(package=package, args=additional_args, env_vars=env)
        elif command in ["python", "python3"]:
            # PythonStdioTransport accepts args and env
            return PythonStdioTransport(args=args, env=merged_env)
        else:
            # Default to StdioTransport for other commands
            full_command = [command] + args if args else [command]
            return StdioTransport(command=full_command[0], args=full_command[1:], env=merged_env)

    async def _fetch_and_set_parameters(self):
        """Fetch tool schema from the MCP server and set the parameters."""
        if self._parameters_loading:
            # Avoid concurrent parameter loading
            return

        self._parameters_loading = True
        try:
            # Check if health check is needed
            if not await self._check_mcp_health():
                logger.warning(f"MCP server health check failed for tool '{self.name}'")
                return

            retry_count = 0
            while retry_count < self._max_retries:
                try:
                    async with self.get_session() as session:
                        # Apply timeout using asyncio.wait_for for individual operations
                        tools = await asyncio.wait_for(session.list_tools(), timeout=self._connection_timeout)
                        if tools:
                            # Find the tool that matches our name or use the first one
                            target_tool = None
                            actual_tool_name = self._get_actual_tool_name()

                            for tool in tools:
                                tool_name = getattr(tool, 'name', '')
                                if tool_name == actual_tool_name or tool_name == self.name:
                                    target_tool = tool
                                    break

                            if not target_tool and tools:
                                target_tool = tools[0]  # Fallback to first tool

                            if target_tool:
                                # Handle both dict and object representations
                                input_schema = None
                                tool_description = None

                                if hasattr(target_tool, 'inputSchema'):
                                    input_schema = target_tool.inputSchema
                                    tool_description = getattr(target_tool, 'description', None)
                                elif hasattr(target_tool, 'dict'):
                                    tool_dict = target_tool.dict()
                                    input_schema = tool_dict.get('inputSchema')
                                    tool_description = tool_dict.get('description')
                                else:
                                    input_schema = getattr(target_tool, 'parameters', None)
                                    tool_description = getattr(target_tool, 'description', None)

                                # Apply the schema directly if available
                                if input_schema:
                                    self.parameters = input_schema
                                    logger.debug(f"Applied dynamic schema from MCP server for tool '{self.name}': {input_schema}")
                                else:
                                    logger.warning(f"No input schema found for MCP tool '{self.name}'")

                                # Update description if available
                                if tool_description:
                                    self.description = tool_description
                                    logger.debug(f"Updated description for tool '{self.name}': {tool_description}")

                                self._parameters_loaded = True
                                logger.debug(f"Successfully configured parameters for tool '{self.name}' from MCP server.")
                                return
                            else:
                                logger.warning(f"No matching tool found for '{self.name}' on MCP server")
                                return
                        else:
                            logger.warning(f"No tools available from MCP server for '{self.name}'")
                            return

                except asyncio.TimeoutError:
                    retry_count += 1
                    if retry_count < self._max_retries:
                        wait_time = 2 ** retry_count  # Exponential backoff
                        logger.warning(f"MCP connection timeout for '{self.name}', retrying in {wait_time}s (attempt {retry_count}/{self._max_retries})")
                        await asyncio.sleep(wait_time)
                    else:
                        logger.error(f"Max retries exceeded for MCP parameter fetch for '{self.name}'")
                        raise
                except Exception as e:
                    retry_count += 1
                    if retry_count < self._max_retries:
                        wait_time = 2 ** retry_count
                        logger.warning(f"MCP parameter fetch failed for '{self.name}': {e}, retrying in {wait_time}s (attempt {retry_count}/{self._max_retries})")
                        await asyncio.sleep(wait_time)
                    else:
                        logger.error(f"Failed to fetch parameters for tool '{self.name}' after {self._max_retries} retries: {e}")
                        raise

        except Exception as e:
            logger.error(f"Critical error fetching parameters for tool '{self.name}': {e}")
            raise
        finally:
            self._parameters_loading = False

    async def ensure_parameters_loaded(self):
        """Ensure parameters are loaded from MCP server if not already done."""
        if self._parameters_loaded:
            return

        if not self.parameters or self.parameters == {
            "type": "object",
            "properties": {
                "tool_name": {"type": "string", "description": "Name of the MCP tool to call"},
                "arguments": {"type": "object", "description": "Arguments to pass to the MCP tool"}
            },
            "required": ["tool_name"]
        }:
            await self._fetch_and_set_parameters()

    async def execute(self, tool_name: str = None, arguments: dict = None, **kwargs) -> Any:
        """
        Execute an MCP tool call with robust error handling and health checks.

        Args:
            tool_name: Name of the MCP tool to call (optional, uses configured tool if not provided)
            arguments: Arguments to pass to the tool
            **kwargs: Additional arguments that will be merged with arguments

        Returns:
            Result from the MCP tool call

        Raises:
            Exception: If execution fails after all retries
        """
        actual_tool_name = None
        try:
            # Ensure parameters are loaded from MCP server
            await self.ensure_parameters_loaded()

            # Check MCP health before execution
            if not await self._check_mcp_health():
                raise ConnectionError(f"MCP server for '{self.name}' is not healthy")

            # If no tool_name provided, this is a direct tool call with parameters
            if tool_name is None:
                # This is a direct call - use kwargs as arguments and get the actual tool name
                final_args = kwargs
                actual_tool_name = self._get_actual_tool_name()
                logger.debug(f"Direct tool call to '{actual_tool_name}' with args: {final_args}")
            else:
                # Traditional mode: tool_name and arguments provided
                final_args = arguments or {}
                final_args.update(kwargs)
                # Map tool name using the mapping
                actual_tool_name = self.tool_name_mapping.get(tool_name, tool_name)
                logger.debug(f"Mapped tool call '{tool_name}' -> '{actual_tool_name}' with args: {final_args}")

            # Call the MCP tool with retry logic
            retry_count = 0
            last_exception = None

            while retry_count < self._max_retries:
                try:
                    result = await self.call_mcp_tool(actual_tool_name, **final_args)
                    return result
                except asyncio.TimeoutError as e:
                    last_exception = e
                    retry_count += 1
                    if retry_count < self._max_retries:
                        wait_time = 2 ** retry_count
                        logger.warning(f"MCP tool '{actual_tool_name}' timeout, retrying in {wait_time}s (attempt {retry_count}/{self._max_retries})")
                        await asyncio.sleep(wait_time)
                    else:
                        logger.error(f"MCP tool '{actual_tool_name}' failed after {self._max_retries} timeout retries")
                        raise
                except ConnectionError as e:
                    last_exception = e
                    retry_count += 1
                    if retry_count < self._max_retries:
                        wait_time = 2 ** retry_count
                        logger.warning(f"MCP connection error for '{actual_tool_name}', retrying in {wait_time}s (attempt {retry_count}/{self._max_retries})")
                        await asyncio.sleep(wait_time)
                        # Reset health check timestamp to force recheck
                        self._last_health_check = 0
                    else:
                        logger.error(f"MCP tool '{actual_tool_name}' failed after {self._max_retries} connection retries")
                        raise
                except Exception as e:
                    # For other exceptions, don't retry
                    logger.error(f"MCP tool '{actual_tool_name}' execution failed: {e}")
                    raise

        except asyncio.CancelledError:
            logger.warning(f"MCP tool execution '{actual_tool_name or tool_name}' was cancelled")
            raise
        except Exception as e:
            error_msg = f"Failed to execute MCP tool '{actual_tool_name or tool_name}': {e}"
            logger.error(error_msg)
            # Consistent error handling - always raise exceptions instead of returning error strings
            raise RuntimeError(error_msg) from e

    def _get_actual_tool_name(self) -> str:
        """Get the actual MCP tool name based on the tool's name using configurable mapping."""
        return self.tool_name_mapping.get(self.name, self.name)

    def add_tool_name_mapping(self, logical_name: str, actual_name: str):
        """Add or update a tool name mapping."""
        self.tool_name_mapping[logical_name] = actual_name
        logger.info(f"Added tool name mapping: '{logical_name}' -> '{actual_name}'")

    def get_tool_name_mappings(self) -> Dict[str, str]:
        """Get all current tool name mappings."""
        return self.tool_name_mapping.copy()

    async def _check_mcp_health(self) -> bool:
        """Check if the MCP server is healthy and responsive."""
        current_time = time.time()
        if current_time - self._last_health_check < self._health_check_interval:
            return True  # Skip health check if done recently

        try:
            async with self.get_session() as session:
                # Try to ping or list tools as a health check
                tools = await asyncio.wait_for(session.list_tools(), timeout=10)
                self._last_health_check = current_time
                logger.debug(f"MCP health check passed for '{self.name}' - {len(tools) if tools else 0} tools available")
                return True
        except asyncio.TimeoutError:
            logger.warning(f"MCP health check timeout for '{self.name}'")
            return False
        except Exception as e:
            logger.warning(f"MCP health check failed for '{self.name}': {e}")
            return False

    # Override call_mcp_tool to add specific error handling for tool execution
    async def call_mcp_tool(self, tool_name: str, **kwargs):
        """Override the mixin method to add tool-specific error handling."""
        try:
            async with self.get_session() as session:
                logger.debug(f"Calling MCP tool '{tool_name}' with args: {kwargs}")
                res = await asyncio.wait_for(session.call_tool(tool_name, arguments=kwargs), timeout=self._connection_timeout)
                if not res:
                    return ""
                for item in res:
                    if hasattr(item, 'text') and item.text is not None:
                        text = item.text
                        if "<coroutine object" in text and "at 0x" in text:
                            raise RuntimeError(f"MCP tool '{tool_name}' returned a coroutine object instead of executing it.")
                        return text
                    elif hasattr(item, 'json') and item.json is not None:
                        import json
                        return json.dumps(item.json, ensure_ascii=False, indent=2)
                if res:
                    result_str = str(res[0])
                    if "<coroutine object" in result_str and "at 0x" in result_str:
                        raise RuntimeError(f"MCP tool '{tool_name}' returned a coroutine object instead of executing it.")
                    return result_str
                return ""
        except asyncio.TimeoutError:
            logger.error(f"MCP tool '{tool_name}' call timed out after {self._connection_timeout}s")
            raise
        except asyncio.CancelledError:
            logger.warning(f"MCP tool '{tool_name}' call was cancelled")
            raise
        except Exception as e:
            logger.error(f"MCP tool '{tool_name}' call failed: {e}")
            raise RuntimeError(f"MCP tool '{tool_name}' execution failed: {str(e)}") from e

    # All other MCP-related methods are inherited from MCPClientMixin