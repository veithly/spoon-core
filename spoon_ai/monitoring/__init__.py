"""
加密货币监控模块
提供加密货币价格和指标的监控、警报和通知功能
"""

from .core.tasks import MonitoringTaskManager

# 导出主要类
__all__ = ['MonitoringTaskManager']