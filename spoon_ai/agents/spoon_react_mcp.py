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
        """Return empty list since MCP tools are handled through ToolFactory/ToolManager"""
        # MCP tools are loaded and managed by the ToolFactory and added to avaliable_tools
        # The _get_cached_mcp_tools() in toolcall.py will call this method
        # but since we manage MCP tools through the regular tool system, return empty list
        return []
