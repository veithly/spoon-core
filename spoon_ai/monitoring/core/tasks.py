import logging
import uuid
from typing import Dict, Any, List, Optional
from datetime import datetime

from .scheduler import MonitoringScheduler
from .alerts import AlertManager, Metric, Comparator

logger = logging.getLogger(__name__)

class MonitoringTaskManager:
    """监控任务管理器，处理任务的创建、删除和执行"""
    
    def __init__(self):
        self.scheduler = MonitoringScheduler()
        self.alert_manager = AlertManager()
        self.scheduler.start()
        
    def create_task(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """创建新的监控任务"""
        # 生成任务ID
        task_id = config.get("task_id", f"task_{uuid.uuid4().hex[:8]}")
        
        # 验证配置
        self._validate_config(config)
        
        # 添加到调度器
        interval_minutes = config.get("check_interval_minutes", 5)
        self.scheduler.add_job(
            task_id, 
            self.alert_manager.monitor_task,
            interval_minutes,
            alert_config=config
        )
        
        # 返回任务信息
        return {
            "task_id": task_id,
            "created_at": datetime.now().isoformat(),
            "config": config
        }
    
    def delete_task(self, task_id: str) -> bool:
        """删除监控任务"""
        return self.scheduler.remove_job(task_id)
    
    def get_tasks(self) -> Dict[str, Any]:
        """获取所有任务"""
        return self.scheduler.get_jobs()
    
    def get_task(self, task_id: str) -> Optional[Dict[str, Any]]:
        """获取特定任务"""
        return self.scheduler.get_job(task_id)
    
    def test_notification(self, task_id: str) -> bool:
        """测试任务通知"""
        job = self.scheduler.get_job(task_id)
        if not job:
            return False
            
        alert_config = job.get("kwargs", {}).get("alert_config", {})
        if not alert_config:
            return False
            
        return self.alert_manager.test_notification(alert_config)
    
    def _validate_config(self, config: Dict[str, Any]) -> None:
        """验证任务配置"""
        required_fields = ["symbol", "metric", "threshold", "comparator"]
        for field in required_fields:
            if field not in config:
                raise ValueError(f"Missing required field: {field}")
        
        # 验证指标类型
        if "metric" in config and not any(config["metric"] == m.value for m in Metric):
            valid_metrics = [m.value for m in Metric]
            raise ValueError(f"Invalid metric: {config['metric']}. Valid options are: {valid_metrics}")
        
        # 验证比较运算符
        if "comparator" in config and not any(config["comparator"] == c.value for c in Comparator):
            valid_comparators = [c.value for c in Comparator]
            raise ValueError(f"Invalid comparator: {config['comparator']}. Valid options are: {valid_comparators}")# spoon_ai/monitoring/core/tasks.py 中的 _validate_config 方法

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