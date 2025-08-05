"""MCP Server lifecycle management."""

import asyncio
import hashlib
import logging
import os
import subprocess
from typing import Dict, Optional, Set, Any
from dataclasses import dataclass

from fastmcp import Client as MCPClient

from .models import MCPServerConfig
from .errors import MCPServerError

logger = logging.getLogger(__name__)


@dataclass
class MCPServerInstance:
    """Represents a running MCP server instance."""
    
    server_id: str
    config: MCPServerConfig
    process: Optional[subprocess.Popen] = None
    client: Optional[MCPClient] = None  # FastMCP client instance
    transport: Optional[object] = None  # Transport instance
    reference_count: int = 0
    status: str = "stopped"  # stopped, starting, running, error
    error_message: Optional[str] = None
    available_tools: Optional[list] = None  # Cache of available tools


class MCPServerManager:
    """Manages MCP server lifecycle and reuse."""
    
    def __init__(self):
        self.active_servers: Dict[str, MCPServerInstance] = {}
        self.server_configs: Dict[str, MCPServerConfig] = {}
        self._lock = asyncio.Lock()
    
    def _generate_server_id(self, config: MCPServerConfig) -> str:
        """Generate a unique server ID based on configuration."""
        config_str = f"{config.command}:{':'.join(config.args)}:{config.cwd or ''}"
        return hashlib.md5(config_str.encode()).hexdigest()[:8]
    
    async def get_or_create_server(self, config: MCPServerConfig) -> MCPServerInstance:
        """Get existing server or create a new one."""
        server_id = self._generate_server_id(config)
        
        async with self._lock:
            # Check if server already exists and is compatible
            if server_id in self.active_servers:
                server = self.active_servers[server_id]
                if self._is_config_compatible(server.config, config):
                    server.reference_count += 1
                    logger.info(f"Reusing MCP server {server_id} (refs: {server.reference_count})")
                    return server
                else:
                    # Configuration changed, need to restart
                    await self._stop_server_internal(server_id)
            
            # Create new server
            server = MCPServerInstance(
                server_id=server_id,
                config=config,
                reference_count=1
            )
            
            self.active_servers[server_id] = server
            self.server_configs[server_id] = config
            
            try:
                await self._start_server_internal(server)
                logger.info(f"Started MCP server {server_id}")
                return server
            except Exception as e:
                # Clean up on failure
                self.active_servers.pop(server_id, None)
                self.server_configs.pop(server_id, None)
                raise MCPServerError(server_id, f"Failed to start server: {str(e)}")
    
    async def release_server(self, server_id: str) -> None:
        """Release a reference to a server, stopping it if no more references."""
        async with self._lock:
            if server_id not in self.active_servers:
                logger.warning(f"Attempted to release unknown server {server_id}")
                return
            
            server = self.active_servers[server_id]
            server.reference_count -= 1
            
            logger.info(f"Released MCP server {server_id} (refs: {server.reference_count})")
            
            if server.reference_count <= 0:
                await self._stop_server_internal(server_id)
                logger.info(f"Stopped MCP server {server_id} (no more references)")
    
    async def stop_all_servers(self) -> None:
        """Stop all running servers."""
        async with self._lock:
            server_ids = list(self.active_servers.keys())
            for server_id in server_ids:
                await self._stop_server_internal(server_id)
    
    async def restart_server(self, server_id: str) -> None:
        """Restart a specific server."""
        async with self._lock:
            if server_id not in self.active_servers:
                raise MCPServerError(server_id, "Server not found")
            
            server = self.active_servers[server_id]
            await self._stop_server_internal(server_id, keep_instance=True)
            await self._start_server_internal(server)
    
    def get_server_for_tool(self, tool_name: str) -> Optional[MCPServerInstance]:
        """Get the MCP server instance for a specific tool."""
        # This would need to be implemented based on tool-to-server mapping
        # For now, return the first running server that has the tool in autoApprove
        for server in self.active_servers.values():
            if server.status == "running" and (
                not server.config.autoApprove or tool_name in server.config.autoApprove
            ):
                return server
        return None
    
    def get_server_status(self, server_id: str) -> Dict[str, any]:
        """Get status information for a server."""
        if server_id not in self.active_servers:
            return {"status": "not_found"}
        
        server = self.active_servers[server_id]
        return {
            "status": server.status,
            "reference_count": server.reference_count,
            "error_message": server.error_message,
            "config": server.config.model_dump()
        }
    
    def list_servers(self) -> Dict[str, Dict[str, any]]:
        """List all servers and their status."""
        return {
            server_id: self.get_server_status(server_id)
            for server_id in self.active_servers
        }
    
    def _is_config_compatible(self, existing: MCPServerConfig, new: MCPServerConfig) -> bool:
        """Check if two configurations are compatible (can reuse server)."""
        # Core command and args must match
        if existing.command != new.command or existing.args != new.args:
            return False
        
        # Working directory must match
        if existing.cwd != new.cwd:
            return False
        
        # Environment variables must be compatible
        # New config can add variables, but can't change existing ones
        for key, value in existing.env.items():
            if key in new.env and new.env[key] != value:
                return False
        
        return True
    
    def _create_transport(self, config: MCPServerConfig):
        """Create appropriate transport for the MCP server."""
        from fastmcp.client.transports import StdioTransport
        import shutil
        import platform
        
        # Handle Windows-specific command path issues
        command = config.command
        if platform.system() == "Windows" and command == "npx":
            # On Windows, use npx.cmd instead of npx to avoid path issues
            command = "npx.cmd"
        
        # Get full path to command
        command_path = shutil.which(command)
        if not command_path:
            # For npx/npx.cmd, try to find npm instead and use npx from there
            if command in ["npx", "npx.cmd"]:
                npm_path = shutil.which("npm")
                if npm_path:
                    # Try to use npx from the npm directory
                    import os
                    npm_dir = os.path.dirname(npm_path)
                    npx_path = os.path.join(npm_dir, "npx.cmd" if platform.system() == "Windows" else "npx")
                    if os.path.exists(npx_path):
                        command_path = npx_path
            
            if not command_path:
                raise MCPServerError("unknown", f"Command not found: {config.command}")
        
        # For FastMCP, we use the generic StdioTransport
        # and pass the command and args directly
        return StdioTransport(
            command=command_path,
            args=config.args,
            env=config.env,
            cwd=config.cwd
        )
    
    async def _start_server_internal(self, server: MCPServerInstance) -> None:
        """Internal method to start a server."""
        config = server.config
        
        if config.disabled:
            server.status = "disabled"
            return
        
        try:
            server.status = "starting"
            logger.info(f"Starting MCP server {server.server_id} with command: {config.command} {' '.join(config.args)}")
            
            # Test if the command works first
            import subprocess
            import shutil
            import os
            import platform
            
            # Handle Windows-specific command path issues
            command = config.command
            if platform.system() == "Windows" and command == "npx":
                # On Windows, use npx.cmd instead of npx to avoid path issues
                command = "npx.cmd"
            
            # Get full path to command
            command_path = shutil.which(command)
            if not command_path:
                # For npx/npx.cmd, try to find npm instead and use npx from there
                if command in ["npx", "npx.cmd"]:
                    npm_path = shutil.which("npm")
                    if npm_path:
                        # Try to use npx from the npm directory
                        npm_dir = os.path.dirname(npm_path)
                        npx_path = os.path.join(npm_dir, "npx.cmd" if platform.system() == "Windows" else "npx")
                        if os.path.exists(npx_path):
                            command_path = npx_path
            
            if not command_path:
                raise Exception(f"Command not found: {config.command}")
            
            test_process = subprocess.Popen(
                [command_path] + config.args,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env={**os.environ, **config.env}
            )
            
            # Send initialize request
            import json
            init_request = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {},
                    "clientInfo": {"name": "test-client", "version": "1.0.0"}
                }
            }
            
            stdout, stderr = test_process.communicate(input=json.dumps(init_request), timeout=10)
            
            if test_process.returncode != 0:
                raise Exception(f"Process failed with return code {test_process.returncode}: {stderr}")
            
            logger.info(f"MCP server test successful: {stdout[:100]}...")
            
            # Create transport and client
            server.transport = self._create_transport(config)
            server.client = MCPClient(server.transport)
            
            # Test connection by trying to ping the server
            async with server.client:
                await server.client.ping()
                
                # Cache available tools
                try:
                    tools = await server.client.list_tools()
                    server.available_tools = [
                        {
                            "name": tool.name,
                            "description": tool.description,
                            "schema": tool.inputSchema
                        }
                        for tool in tools
                    ]
                    logger.info(f"MCP server {server.server_id} has {len(tools)} tools available")
                except Exception as e:
                    logger.warning(f"Could not list tools for server {server.server_id}: {e}")
                    server.available_tools = []
            
            server.status = "running"
            server.error_message = None
            logger.info(f"MCP server {server.server_id} started successfully")
            
        except Exception as e:
            server.status = "error"
            server.error_message = str(e)
            logger.error(f"Failed to start MCP server {server.server_id}: {e}")
            logger.debug(f"Server config: command={config.command}, args={config.args}, env_keys={list(config.env.keys())}")
            
            # Clean up on failure
            if server.client:
                server.client = None
            if server.transport:
                server.transport = None
            
            raise
    
    async def _stop_server_internal(self, server_id: str, keep_instance: bool = False) -> None:
        """Internal method to stop a server."""
        if server_id not in self.active_servers:
            return
        
        server = self.active_servers[server_id]
        
        try:
            # Clean up MCP client and transport
            if server.client:
                try:
                    # FastMCP clients clean up automatically with context managers
                    # No explicit cleanup needed
                    server.client = None
                except Exception as e:
                    logger.warning(f"Error cleaning up MCP client for {server_id}: {e}")
            
            if server.transport:
                server.transport = None
            
            # Clear cached tools
            server.available_tools = None
            
            server.status = "stopped"
            server.error_message = None
            
        except Exception as e:
            logger.error(f"Error stopping server {server_id}: {e}")
            server.status = "error"
            server.error_message = str(e)
        
        finally:
            if not keep_instance:
                self.active_servers.pop(server_id, None)
                self.server_configs.pop(server_id, None)
    
    async def get_server_tools(self, server_id: str) -> list:
        """Get available tools for a specific server."""
        if server_id not in self.active_servers:
            return []
        
        server = self.active_servers[server_id]
        if server.status != "running":
            return []
        
        # Return cached tools if available
        if server.available_tools is not None:
            return server.available_tools
        
        # Try to refresh tools list
        try:
            if server.client:
                async with server.client:
                    tools = await server.client.list_tools()
                    server.available_tools = [
                        {
                            "name": tool.name,
                            "description": tool.description,
                            "schema": tool.inputSchema
                        }
                        for tool in tools
                    ]
                    return server.available_tools
        except Exception as e:
            logger.error(f"Failed to get tools for server {server_id}: {e}")
        
        return []
    
    async def call_tool(self, server_id: str, tool_name: str, **kwargs) -> Any:
        """Call a tool on a specific MCP server."""
        if server_id not in self.active_servers:
            raise MCPServerError(server_id, "Server not found")
        
        server = self.active_servers[server_id]
        if server.status != "running":
            raise MCPServerError(server_id, f"Server is not running (status: {server.status})")
        
        if not server.client:
            raise MCPServerError(server_id, "No client available")
        
        try:
            async with server.client:
                logger.debug(f"Calling tool {tool_name} on server {server_id} with args: {kwargs}")
                result = await server.client.call_tool(tool_name, kwargs)
                
                # Process result similar to MCPToolWrapper
                if hasattr(result, 'data') and result.data is not None:
                    return result.data
                elif hasattr(result, 'content') and result.content:
                    text_results = []
                    for content in result.content:
                        if hasattr(content, 'text') and content.text:
                            text_results.append(content.text)
                    
                    if text_results:
                        return '\n'.join(text_results)
                    else:
                        return ""
                elif isinstance(result, list):
                    # Handle direct list of content blocks
                    text_results = []
                    for content in result:
                        if hasattr(content, 'text') and content.text:
                            text_results.append(content.text)
                    
                    if text_results:
                        return '\n'.join(text_results)
                    else:
                        return ""
                else:
                    return str(result)
                    
        except Exception as e:
            logger.error(f"Tool call failed on server {server_id}: {e}")
            raise MCPServerError(server_id, f"Tool call failed: {str(e)}")