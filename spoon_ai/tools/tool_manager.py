from typing import Any, Dict, Iterator, List

from spoon_ai.tools.base import BaseTool, ToolFailure, ToolResult


class ToolManager:
    def __init__(self, *tools: BaseTool):
        self.tools = tools
        self.tool_map = {tool.name: tool for tool in tools}
        
    def __getitem__(self, name: str) -> BaseTool:
        return self.tool_map[name]
    
    def __iter__(self) -> Iterator[BaseTool]:
        return iter(self.tools)
    
    def __len__(self) -> int:
        return len(self.tools)
    
    def to_params(self) -> List[Dict[str, Any]]:
        return [tool.to_param() for tool in self.tools]
    
    async def execute(self, * ,name: str, tool_input: Dict[str, Any] =None) -> ToolResult:
        tool = self.tool_map[name]
        if not tool:
            return ToolFailure(error=f"Tool {name} not found")
    
        try:
            result = await tool(**tool_input)
            return result
        except Exception as e:
            return ToolFailure(error=str(e))
        
    def get_tool(self, name: str) -> BaseTool:
        tool = self.tool_map.get(name)
        if not tool:
            raise ValueError(f"Tool {name} not found")
        return tool
    
    def add_tool(self, tool: BaseTool) -> None:
        self.tools.append(tool)
        self.tool_map[tool.name] = tool
        
    def add_tools(self, *tools: BaseTool) -> None:
        for tool in tools:
            self.add_tool(tool)
            
    def remove_tool(self, name: str) -> None:
        self.tools = [tool for tool in self.tools if tool.name != name]
        del self.tool_map[name]       
