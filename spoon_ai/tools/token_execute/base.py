from spoon_ai.tools.base import BaseTool
from abc import abstractmethod
class TokenExecuteBaseTool(BaseTool):
    name: str = "token_execute"
    
    @abstractmethod
    async def execute(self, **kwargs) -> str:
        raise NotImplementedError("Subclasses must implement this method")