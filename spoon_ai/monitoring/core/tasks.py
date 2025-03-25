# spoon_ai/monitoring/core/tasks.py
import logging
import uuid
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta

from .scheduler import MonitoringScheduler
from .alerts import AlertManager, Metric, Comparator

logger = logging.getLogger(__name__)

class TaskStatus:
    """任务状态枚举"""
    ACTIVE = "active"
    EXPIRED = "expired"
    PAUSED = "paused"

class MonitoringTaskManager:
    """监控任务管理器，处理任务的创建、删除和执行"""
    
    def __init__(self):
        self.scheduler = MonitoringScheduler()
        self.alert_manager = AlertManager()
        self.tasks = {}  # 存储任务状态和元数据
        self.scheduler.start()
        
    def create_task(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """创建新的监控任务"""
        # 生成任务ID
        task_id = config.get("task_id", f"task_{uuid.uuid4().hex[:8]}")
        
        # 验证配置
        self._validate_config(config)
        
        # 设置过期时间（默认24小时）
        expires_in_hours = config.get("expires_in_hours", 24)
        expiry_time = datetime.now() + timedelta(hours=expires_in_hours)
        
        # 存储任务元数据
        self.tasks[task_id] = {
            "status": TaskStatus.ACTIVE,
            "created_at": datetime.now(),
            "expires_at": expiry_time,
            "config": config,
            "last_checked": None,
            "alert_count": 0
        }
        
        # 添加到调度器
        interval_minutes = config.get("check_interval_minutes", 5)
        self.scheduler.add_job(
            task_id, 
            self._task_wrapper,
            interval_minutes,
            task_id=task_id,
            alert_config=config
        )
        
        # 添加过期检查任务
        expiry_task_id = f"{task_id}_expiry"
        self.scheduler.add_job(
            expiry_task_id,
            self._check_task_expiry,
            10,  # 每10分钟检查一次过期状态
            task_id=task_id
        )
        
        # 返回任务信息
        return {
            "task_id": task_id,
            "created_at": datetime.now().isoformat(),
            "expires_at": expiry_time.isoformat(),
            "config": config,
            "status": TaskStatus.ACTIVE
        }
    
    def _task_wrapper(self, task_id: str, alert_config: Dict[str, Any]) -> None:
        """任务执行包装器，用于更新任务状态并处理过期任务"""
        task_info = self.tasks.get(task_id)
        if not task_info:
            logger.warning(f"任务不存在: {task_id}")
            return
            
        # 检查任务是否过期或暂停
        if task_info["status"] != TaskStatus.ACTIVE:
            logger.info(f"任务 {task_id} 状态为 {task_info['status']}，跳过执行")
            return
            
        # 执行任务
        try:
            is_triggered = self.alert_manager.check_alert(alert_config)
            task_info["last_checked"] = datetime.now()
            
            if is_triggered:
                task_info["alert_count"] += 1
        except Exception as e:
            logger.error(f"执行任务 {task_id} 出错: {str(e)}")
    
    def _check_task_expiry(self, task_id: str) -> None:
        """检查任务是否过期"""
        task_info = self.tasks.get(task_id)
        if not task_info:
            return
            
        if task_info["status"] == TaskStatus.ACTIVE and datetime.now() > task_info["expires_at"]:
            # 任务过期
            task_info["status"] = TaskStatus.EXPIRED
            logger.info(f"任务 {task_id} 已过期")
            
            # 发送过期通知
            self._send_expiry_notification(task_id, task_info)
    
    def _send_expiry_notification(self, task_id: str, task_info: Dict[str, Any]) -> None:
        """发送任务过期通知"""
        config = task_info["config"]
        channels = config.get("notification_channels", ["telegram"])
        notification_params = config.get("notification_params", {})
        
        message = (
            f"🕒 **监控任务已过期** 🕒\n\n"
            f"任务ID: {task_id}\n"
            f"名称: {config.get('name', '未命名任务')}\n"
            f"创建时间: {task_info['created_at'].strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"过期时间: {task_info['expires_at'].strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"触发次数: {task_info['alert_count']}\n\n"
            f"此监控任务已自动停止。如需继续监控，请重新创建或继续任务。"
        )
        
        # 发送通知
        for channel in channels:
            self.alert_manager.notification.send(channel, message, **notification_params)
    
    def extend_task(self, task_id: str, hours: int = 24) -> Dict[str, Any]:
        """延长任务过期时间"""
        if task_id not in self.tasks:
            raise ValueError(f"任务不存在: {task_id}")
            
        task_info = self.tasks[task_id]
        
        # 计算新的过期时间
        new_expiry = datetime.now() + timedelta(hours=hours)
        task_info["expires_at"] = new_expiry
        
        # 如果任务已过期，重新激活
        if task_info["status"] == TaskStatus.EXPIRED:
            task_info["status"] = TaskStatus.ACTIVE
            
            # 重新添加到调度器
            config = task_info["config"]
            interval_minutes = config.get("check_interval_minutes", 5)
            self.scheduler.add_job(
                task_id, 
                self._task_wrapper,
                interval_minutes,
                task_id=task_id,
                alert_config=config
            )
        
        return {
            "task_id": task_id,
            "status": task_info["status"],
            "expires_at": new_expiry.isoformat()
        }
    
    def pause_task(self, task_id: str) -> bool:
        """暂停任务"""
        if task_id not in self.tasks:
            return False
            
        self.tasks[task_id]["status"] = TaskStatus.PAUSED
        return True
    
    def resume_task(self, task_id: str) -> bool:
        """恢复任务"""
        if task_id not in self.tasks:
            return False
            
        task_info = self.tasks[task_id]
        
        # 检查是否已过期
        if datetime.now() > task_info["expires_at"]:
            return False
            
        task_info["status"] = TaskStatus.ACTIVE
        return True
    
    def delete_task(self, task_id: str) -> bool:
        """删除监控任务"""
        if task_id in self.tasks:
            # 删除任务元数据
            del self.tasks[task_id]
            
            # 移除调度任务
            self.scheduler.remove_job(task_id)
            
            # 移除过期检查任务
            expiry_task_id = f"{task_id}_expiry"
            self.scheduler.remove_job(expiry_task_id)
            
            return True
        return False
    
    def get_tasks(self) -> Dict[str, Any]:
        """获取所有任务，包含状态信息"""
        result = {}
        for task_id, task_info in self.tasks.items():
            result[task_id] = {
                "status": task_info["status"],
                "created_at": task_info["created_at"].isoformat(),
                "expires_at": task_info["expires_at"].isoformat(),
                "config": task_info["config"],
                "last_checked": task_info["last_checked"].isoformat() if task_info["last_checked"] else None,
                "alert_count": task_info["alert_count"]
            }
        return result
    
    def get_task(self, task_id: str) -> Optional[Dict[str, Any]]:
        """获取特定任务信息"""
        if task_id not in self.tasks:
            return None
            
        task_info = self.tasks[task_id]
        return {
            "status": task_info["status"],
            "created_at": task_info["created_at"].isoformat(),
            "expires_at": task_info["expires_at"].isoformat(),
            "config": task_info["config"],
            "last_checked": task_info["last_checked"].isoformat() if task_info["last_checked"] else None,
            "alert_count": task_info["alert_count"]
        }
    
    def test_notification(self, task_id: str) -> bool:
        """测试任务通知"""
        if task_id not in self.tasks:
            return False
            
        alert_config = self.tasks[task_id]["config"]
        return self.alert_manager.test_notification(alert_config)
    
    def _validate_config(self, config: Dict[str, Any]) -> None:
        """验证任务配置"""
        required_fields = ["provider", "symbol", "metric", "threshold", "comparator"]
        for field in required_fields:
            if field not in config:
                raise ValueError(f"Missing required field: {field}")
        
        # 验证市场类型
        market = config.get("market", "cex").lower()
        if market not in ["cex", "dex"]:  # 添加更多支持的市场类型
            raise ValueError(f"Invalid market type: {market}. Supported types: cex, dex")
        
        # 验证提供者
        provider = config["provider"].lower()
        try:
            from ..clients.base import DataClient
            # 这会检查提供者是否有效
            DataClient.get_client(market, provider)
        except ValueError as e:
            raise ValueError(f"Invalid provider: {str(e)}")
        
        # 验证指标类型
        if "metric" in config and not any(config["metric"] == m.value for m in Metric):
            valid_metrics = [m.value for m in Metric]
            raise ValueError(f"Invalid metric: {config['metric']}. Valid options are: {valid_metrics}")
        
        # 验证比较运算符
        if "comparator" in config and not any(config["comparator"] == c.value for c in Comparator):
            valid_comparators = [c.value for c in Comparator]
            raise ValueError(f"Invalid comparator: {config['comparator']}. Valid options are: {valid_comparators}")
        
        # 验证过期时间
        if "expires_in_hours" in config:
            try:
                expires_in_hours = int(config["expires_in_hours"])
                if expires_in_hours <= 0:
                    raise ValueError("Expiration time must be positive")
            except (TypeError, ValueError):
                raise ValueError("Invalid expiration time: must be a positive integer")