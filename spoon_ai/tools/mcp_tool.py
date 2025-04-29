from .base import BaseTool

class MCPTool(BaseTool):
    
    def execute(self, **kwargs):
        raise NotImplementedError("Please execute in your own tool call agent")
        
        