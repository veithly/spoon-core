# spoon_ai/monitoring/api/routes.py
from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field

from ..core.tasks import MonitoringTaskManager, TaskStatus

router = APIRouter(
    prefix="/monitoring",
    tags=["monitoring"],
    responses={404: {"description": "Not found"}},
)

task_manager = MonitoringTaskManager()

class MonitoringTaskCreate(BaseModel):
    """创建监控任务的请求模型"""
    market: str = Field("cex", description="市场类型: cex, dex, etc.")
    provider: str = Field(..., description="数据提供者: bn (Binance), cb (Coinbase), etc.")
    symbol: str = Field(..., description="交易对符号，例如 BTCUSDT")
    metric: str = Field(..., description="监控指标: price, volume, price_change, price_change_percent")
    threshold: float = Field(..., description="警报阈值")
    comparator: str = Field(..., description="比较运算符: >, <, =, >=, <=")
    name: Optional[str] = Field(None, description="警报名称")
    check_interval_minutes: int = Field(5, description="检查间隔（分钟）")
    expires_in_hours: int = Field(24, description="任务过期时间（小时）")
    notification_channels: List[str] = Field(["telegram"], description="通知渠道")
    notification_params: Dict[str, Any] = Field({}, description="通知渠道的额外参数")

class TaskExtendRequest(BaseModel):
    """延长任务有效期请求模型"""
    hours: int = Field(..., description="延长的小时数")

class MonitoringTaskResponse(BaseModel):
    """监控任务响应模型"""
    task_id: str
    created_at: str
    expires_at: str
    status: str
    config: Dict[str, Any]

class MonitoringChannelsResponse(BaseModel):
    """可用通知渠道响应模型"""
    available_channels: List[str]

@router.post("/tasks", response_model=MonitoringTaskResponse)
async def create_monitoring_task(task: MonitoringTaskCreate):
    """创建新的监控任务"""
    try:
        result = task_manager.create_task(task.dict())
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create task: {str(e)}")

@router.get("/tasks", response_model=Dict[str, Any])
async def list_monitoring_tasks():
    """获取所有监控任务"""
    return task_manager.get_tasks()

@router.get("/tasks/{task_id}", response_model=Dict[str, Any])
async def get_monitoring_task(task_id: str):
    """获取特定监控任务"""
    task = task_manager.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")
    return task

@router.delete("/tasks/{task_id}")
async def delete_monitoring_task(task_id: str):
    """删除监控任务"""
    success = task_manager.delete_task(task_id)
    if not success:
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")
    return {"status": "success", "message": f"Task {task_id} deleted"}

@router.post("/tasks/{task_id}/pause")
async def pause_monitoring_task(task_id: str):
    """暂停监控任务"""
    success = task_manager.pause_task(task_id)
    if not success:
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")
    return {"status": "success", "message": f"Task {task_id} paused"}

@router.post("/tasks/{task_id}/resume")
async def resume_monitoring_task(task_id: str):
    """恢复监控任务"""
    success = task_manager.resume_task(task_id)
    if not success:
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found or expired")
    return {"status": "success", "message": f"Task {task_id} resumed"}

@router.post("/tasks/{task_id}/extend", response_model=Dict[str, Any])
async def extend_monitoring_task(task_id: str, request: TaskExtendRequest):
    """延长监控任务有效期"""
    try:
        result = task_manager.extend_task(task_id, request.hours)
        return result
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to extend task: {str(e)}")

@router.get("/channels", response_model=MonitoringChannelsResponse)
async def get_notification_channels():
    """获取可用的通知渠道"""
    from ..notifiers.notification import NotificationManager
    manager = NotificationManager()
    return {"available_channels": manager.get_available_channels()}

@router.post("/tasks/{task_id}/test")
async def test_notification(task_id: str):
    """测试特定任务的通知"""
    task = task_manager.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")
    
    success = task_manager.test_notification(task_id)
    if not success:
        raise HTTPException(status_code=500, detail="Failed to send test notification")
    
    return {"status": "success", "message": "Test notification sent"}