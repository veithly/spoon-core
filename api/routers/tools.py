from typing import Dict, Any, List, Optional
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from datetime import datetime
import uuid

from spoon_ai.tools import (
    ToolManager, BaseTool, Terminate, 
    PredictPrice, TokenHolders, TradingHistory, UniswapLiquidity, 
    WalletAnalysis, GetTokenPriceTool, Get24hStatsTool, GetKlineDataTool,
    PriceThresholdAlertTool, LpRangeCheckTool, SuddenPriceIncreaseTool,
    LendingRateMonitorTool
)

router = APIRouter()

# Initialize tool manager and add all tools
tools = [
    Terminate(),
    PredictPrice(),
    TokenHolders(),
    TradingHistory(),
    UniswapLiquidity(),
    WalletAnalysis(),
    GetTokenPriceTool(),
    Get24hStatsTool(),
    GetKlineDataTool(),
    PriceThresholdAlertTool(),
    LpRangeCheckTool(),
    SuddenPriceIncreaseTool(),
    LendingRateMonitorTool()
]

tool_manager = ToolManager(tools)

# Request and response models
class ToolExecuteRequest(BaseModel):
    name: str
    tool_input: Optional[Dict[str, Any]] = None

class ToolResponse(BaseModel):
    name: str
    output: Any
    error: Optional[str] = None
    
class ToolSchemaResponse(BaseModel):
    name: str
    description: str
    parameters: str
    strict: bool = True
    published: bool = True
    tool_id: str
    created_at: str
    updated_at: str

class ToolListResponse(BaseModel):
    code: int
    message: str
    data: Dict[str, Any]

# Get all available tools with pagination
@router.get("/tools", response_model=ToolListResponse)
async def get_tools(skip: int = Query(0, ge=0), limit: int = Query(10, ge=1, le=100)):
    # Get all tools
    all_tools = list(tool_manager)
    
    # Calculate total count
    total_count = len(all_tools)
    
    # Apply pagination
    paginated_tools = all_tools[skip:skip + limit]
    
    # Current time for created_at and updated_at
    current_time = datetime.now().isoformat()
    
    # Convert to response format
    tool_results = []
    for tool in paginated_tools:
        # Generate unique tool ID
        tool_id = str(uuid.uuid4())
        
        # Convert parameters to string
        params_str = str(tool.parameters).replace("'", "\"")
        
        tool_results.append(
            ToolSchemaResponse(
                name=tool.name,
                description=tool.description,
                parameters=params_str,
                strict=True,
                published=True,
                tool_id=tool_id,
                created_at=current_time,
                updated_at=current_time
            )
        )
    
    return ToolListResponse(
        code=200,
        message="Success",
        data={
            "tools": tool_results,
            "total": total_count
        }
    )

# Execute a specific tool
@router.post("/execute", response_model=ToolResponse)
async def execute_tool(request: ToolExecuteRequest):
    try:
        result = await tool_manager.execute(name=request.name, tool_input=request.tool_input or {})
        return ToolResponse(
            name=request.name,
            output=result.output,
            error=result.error
        )
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Tool '{request.name}' not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Query relevant tools
@router.get("/query", response_model=List[str])
async def query_tools(query: str, top_k: int = 5):
    try:
        relevant_tools = tool_manager.query_tools(query=query, top_k=top_k)
        return relevant_tools
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
