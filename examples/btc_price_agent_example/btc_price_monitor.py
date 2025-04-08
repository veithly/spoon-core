#!/usr/bin/env python
# btc_price_agent/btc_price_monitor.py

import os
import sys
import asyncio
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime

# 添加父目录到路径，确保能够导入 spoon_ai 包
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# 导入 SpoonAI 组件
from spoon_ai.chat import ChatBot, Memory
from spoon_ai.schema import Message
from spoon_ai.monitoring.core.tasks import MonitoringTaskManager
from spoon_ai.monitoring.core.alerts import Metric, Comparator

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)

logger = logging.getLogger("btc-price-agent")

class BTCPriceAgent:
    """比特币价格监控 Agent"""
    
    def __init__(
        self,
        llm_provider: str = "anthropic",
        model_name: str = "claude-3-7-sonnet-20250219",
        notification_channels: List[str] = ["telegram"],
        check_interval_minutes: int = 5,
    ):
        # 初始化监控任务管理器
        self.task_manager = MonitoringTaskManager()
        
        # 初始化 ChatBot
        self.chatbot = ChatBot(
            llm_provider=llm_provider,
            model_name=model_name,
            api_key=os.getenv("ANTHROPIC_API_KEY") if llm_provider == "anthropic" else os.getenv("OPENAI_API_KEY")
        )
        
        # 初始化记忆
        self.memory = Memory()
        
        # 保存配置
        self.notification_channels = notification_channels
        self.check_interval_minutes = check_interval_minutes
        
        # 系统消息
        self.system_message = """你是一个专业的加密货币市场分析师，负责监控比特币价格波动并提供分析。
        当价格触发阈值时，你需要提供简洁明了的价格提醒和市场分析。
        分析应该考虑价格趋势、重要的支撑/阻力位以及近期市场情绪。
        """
        
        logger.info("BTC 价格监控 Agent 已初始化")
    
    def setup_price_monitor(
        self, 
        symbol: str = "BTCUSDT",
        price_threshold: float = None,
        price_change_threshold: float = 5.0,
        market: str = "cex",
        provider: str = "binance",
        expires_in_hours: int = 24,
        notification_params: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """设置价格监控任务"""
        
        # 为了同时设置价格阈值和价格变化监控
        tasks_created = []
        
        # 如果设置了具体价格阈值，创建价格阈值监控
        if price_threshold is not None:
            price_monitor_config = {
                "market": market,
                "provider": provider,
                "symbol": symbol,
                "metric": Metric.PRICE.value,
                "threshold": price_threshold,
                "comparator": Comparator.GREATER_THAN.value,  # 可以根据需要修改比较运算符
                "name": f"比特币价格监控 - {price_threshold} 美元",
                "check_interval_minutes": self.check_interval_minutes,
                "expires_in_hours": expires_in_hours,
                "notification_channels": self.notification_channels,
                "notification_params": notification_params or {}
            }
            
            # 创建价格阈值监控任务
            price_task = self.task_manager.create_task(price_monitor_config)
            tasks_created.append(price_task)
            logger.info(f"创建价格阈值监控: {symbol} > {price_threshold}")
        
        # 创建价格变化百分比监控
        price_change_config = {
            "market": market,
            "provider": provider,
            "symbol": symbol,
            "metric": Metric.PRICE_CHANGE_PERCENT.value,
            "threshold": price_change_threshold,
            "comparator": Comparator.GREATER_THAN.value if price_change_threshold > 0 else Comparator.LESS_THAN.value,
            "name": f"比特币价格变化监控 - {price_change_threshold}%",
            "check_interval_minutes": self.check_interval_minutes,
            "expires_in_hours": expires_in_hours,
            "notification_channels": self.notification_channels,
            "notification_params": notification_params or {}
        }
        
        # 创建价格变化监控任务
        price_change_task = self.task_manager.create_task(price_change_config)
        tasks_created.append(price_change_task)
        logger.info(f"创建价格变化监控: {symbol} 变化 {price_change_threshold}%")
        
        return {
            "tasks_created": tasks_created,
            "task_count": len(tasks_created)
        }
    
    async def process_notification(self, alert_data: Dict[str, Any]) -> str:
        """处理并增强通知内容"""
        # 从警报数据中提取相关信息
        symbol = alert_data.get("symbol", "BTCUSDT")
        current_value = alert_data.get("current_value", 0)
        threshold = alert_data.get("threshold", 0)
        metric = alert_data.get("metric", "price")
        
        # 创建给 LLM 的提示消息
        user_message = Message(
            role="user", 
            content=f"""比特币价格监控触发了警报:
            - 交易对: {symbol}
            - 当前值: {current_value}
            - 阈值: {threshold}
            - 指标: {metric}
            - 时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
            
            请提供简短的价格提醒和市场情况分析，包括:
            1. 价格动向简述
            2. 可能的近期支撑/阻力位
            3. 对短期走势的简要看法
            """
        )
        
        # 添加消息到记忆
        self.memory.add_message(user_message)
        
        # 获取 LLM 回复
        response = await self.chatbot.ask(self.memory.get_messages(), system_msg=self.system_message)
        
        # 添加助手回复到记忆
        assistant_message = Message(role="assistant", content=response)
        self.memory.add_message(assistant_message)
        
        return response
    
    def get_active_tasks(self) -> Dict[str, Any]:
        """获取所有活跃的监控任务"""
        return self.task_manager.get_tasks()
    
    def stop_all_tasks(self) -> bool:
        """停止所有监控任务"""
        tasks = self.task_manager.get_tasks()
        success = True
        
        for task_id in tasks:
            if not self.task_manager.delete_task(task_id):
                success = False
                logger.error(f"无法删除任务: {task_id}")
        
        return success

async def main():
    """主函数，设置并运行比特币价格监控"""
    try:
        # 创建比特币价格 Agent
        btc_agent = BTCPriceAgent(
            notification_channels=["telegram"],
            check_interval_minutes=2
        )
        
        # 设置监控参数
        # 可以根据需要调整这些参数
        notification_params = {
            "telegram": {
                "chat_id": os.getenv("TELEGRAM_CHAT_ID", "")  # 替换为您的 Telegram chat_id
            }
        }
        
        # 设置价格阈值和价格变化监控
        result = btc_agent.setup_price_monitor(
            symbol="BTCUSDT",
            price_threshold=70000,  # 当 BTC 价格超过 70000 美元时触发
            price_change_threshold=3.0,  # 当 BTC 价格在 24 小时内变化超过 3% 时触发
            notification_params=notification_params,
            expires_in_hours=48  # 监控持续 48 小时
        )
        
        logger.info(f"已创建 {result['task_count']} 个监控任务")
        
        # 运行一段时间后，可以调用下面的代码来停止所有任务
        # 这里我们等待 3 分钟只是为了演示，实际使用中可以根据需要调整
        # await asyncio.sleep(180)  # 等待 3 分钟
        # btc_agent.stop_all_tasks()
        # logger.info("已停止所有监控任务")
        
        # 保持主程序运行
        while True:
            await asyncio.sleep(60)
            logger.info("监控服务正在运行...")
        
    except KeyboardInterrupt:
        logger.info("程序被用户中断")
    except Exception as e:
        logger.error(f"运行出错: {str(e)}")

if __name__ == "__main__":
    asyncio.run(main()) 