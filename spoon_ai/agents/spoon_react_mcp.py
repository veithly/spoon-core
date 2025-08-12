import asyncio
from spoon_ai.agents.spoon_react import SpoonReactAI
from spoon_ai.tools.tool_manager import ToolManager
from pydantic import Field
import logging

logger = logging.getLogger(__name__)

class SpoonReactMCP(SpoonReactAI):
    name: str = "spoon_react_mcp"
    description: str = "A smart ai agent in neo blockchain with mcp"
    avaliable_tools: ToolManager = Field(default_factory=lambda: ToolManager([]))

    def __init__(self, **kwargs):
        # Initialize SpoonReactAI
        super().__init__(**kwargs)
        logger.info(f"Initialized SpoonReactMCP agent: {self.name}")

    async def list_mcp_tools(self):
        """Return MCP tools from avaliable_tools manager"""
        # Import here to avoid circular imports
        from mcp.types import Tool as MCPTool

        # Return MCP tools that are available in the tool manager
        # Create proper MCPTool objects that match the expected interface
        mcp_tools = []
        
        # Pre-load parameters for all MCP tools concurrently
        async def load_tool_params(tool):
            if hasattr(tool, 'ensure_parameters_loaded'):
                try:
                    await asyncio.wait_for(tool.ensure_parameters_loaded(), timeout=10.0)
                except asyncio.TimeoutError:
                    logger.warning(f"Timeout loading parameters for tool: {tool.name}")
            return tool

        mcp_tool_instances = [tool for tool in self.avaliable_tools.tool_map.values() if hasattr(tool, 'mcp_config')]
        loaded_tools = await asyncio.gather(*[load_tool_params(tool) for tool in mcp_tool_instances])

        for tool in loaded_tools:
            # Create proper MCPTool instance for the tool system
            mcp_tool = MCPTool(
                name=tool.name,
                description=tool.description,
                inputSchema=tool.parameters if tool.parameters else {
                    "type": "object",
                    "properties": {},
                    "required": []
                }
            )
            mcp_tools.append(mcp_tool)

        if mcp_tools:
            logger.info(f"Found {len(mcp_tools)} MCP tools: {[t.name for t in mcp_tools]}")
        else:
            logger.info("No MCP tools found in available tools")

        return mcp_tools
