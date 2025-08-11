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

        for tool_name, tool in self.avaliable_tools.tool_map.items():
            # Check if this is an MCP tool by looking for mcp_transport attribute
            if hasattr(tool, 'mcp_transport') and tool.mcp_transport is not None:
                # Pre-load the schema from the MCP server
                if hasattr(tool, 'ensure_parameters_loaded'):
                    await tool.ensure_parameters_loaded()

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
