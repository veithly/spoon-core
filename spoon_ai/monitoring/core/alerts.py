# spoon_ai/monitoring/core/alerts.py
import logging
from typing import Dict, Any, List, Optional
from enum import Enum
from datetime import datetime

from ..clients.base import DataClient
from ..notifiers.notification import NotificationManager

logger = logging.getLogger(__name__)

class Comparator(str, Enum):
    """比较运算符枚举"""
    GREATER_THAN = ">"
    LESS_THAN = "<"
    EQUAL = "="
    GREATER_EQUAL = ">="
    LESS_EQUAL = "<="

class Metric(str, Enum):
    """监控指标枚举"""
    PRICE = "price"
    VOLUME = "volume"
    PRICE_CHANGE = "price_change"
    PRICE_CHANGE_PERCENT = "price_change_percent"

class AlertManager:
    """警报管理器，处理指标监控和通知发送"""
    
    def __init__(self):
        self.notification = NotificationManager()
        self.clients_cache = {}  # 缓存创建的客户端
        
    def _get_client(self, market: str, provider: str) -> DataClient:
        """获取数据客户端，带缓存"""
        cache_key = f"{market}:{provider}"
        if cache_key not in self.clients_cache:
            self.clients_cache[cache_key] = DataClient.get_client(market, provider)
        return self.clients_cache[cache_key]
        
    def check_condition(self, value: float, threshold: float, comparator: Comparator) -> bool:
        """检查条件是否满足"""
        if comparator == Comparator.GREATER_THAN:
            return value > threshold
        elif comparator == Comparator.LESS_THAN:
            return value < threshold
        elif comparator == Comparator.EQUAL:
            return value == threshold
        elif comparator == Comparator.GREATER_EQUAL:
            return value >= threshold
        elif comparator == Comparator.LESS_EQUAL:
            return value <= threshold
        return False
    
    def get_metric_value(self, market: str, provider: str, symbol: str, metric: Metric) -> float:
        """获取指标的当前值"""
        client = self._get_client(market, provider)
        
        if metric == Metric.PRICE:
            data = client.get_ticker_price(symbol)
            return float(data["price"])
        
        # 需要24小时统计数据的指标
        elif metric in [Metric.VOLUME, Metric.PRICE_CHANGE, Metric.PRICE_CHANGE_PERCENT]:
            data = client.get_ticker_24h(symbol)
            
            if metric == Metric.VOLUME:
                return float(data["volume"])
            elif metric == Metric.PRICE_CHANGE:
                return float(data["priceChange"])
            elif metric == Metric.PRICE_CHANGE_PERCENT:
                return float(data["priceChangePercent"])
        
        raise ValueError(f"Unsupported metric: {metric}")
    
    def check_alert(self, alert_config: Dict[str, Any], test_mode: bool = False) -> bool:
        """检查警报条件是否触发"""
        try:
            market = alert_config.get("market", "cex")
            provider = alert_config["provider"]
            symbol = alert_config["symbol"]
            metric = Metric(alert_config["metric"])
            threshold = float(alert_config["threshold"])
            comparator = Comparator(alert_config["comparator"])
            
            current_value = self.get_metric_value(market, provider, symbol, metric)
            
            is_triggered = self.check_condition(current_value, threshold, comparator) or test_mode
            
            if is_triggered:
                logger.info(f"Alert triggered: {provider}/{symbol} {metric} {current_value} {comparator} {threshold}")
                
                # 准备通知内容
                message = self._format_alert_message(
                    market, provider, symbol, metric, current_value, comparator, threshold, 
                    alert_config.get("name", "Crypto Alert"),
                    test_mode
                )
                
                # 发送通知
                channels = alert_config.get("notification_channels", ["telegram"])
                for channel in channels:
                    notification_params = alert_config.get("notification_params", {})
                    self.notification.send(channel, message, **notification_params)
                
            return is_triggered
            
        except Exception as e:
            logger.error(f"Error checking alert: {str(e)}")
            return False
    
    def _format_alert_message(self, market: str, provider: str, symbol: str, 
                             metric: Metric, value: float, comparator: Comparator, 
                             threshold: float, alert_name: str,
                             test_mode: bool = False) -> str:
        """格式化警报消息"""
        status_emoji = "🧪" if test_mode else "🚨"
        test_prefix = "[TEST] " if test_mode else ""
        
        return (
            f"{status_emoji} **{test_prefix}{alert_name}** {status_emoji}\n\n"
            f"Market: {market.upper()}\n"
            f"Provider: {provider.upper()}\n"
            f"Symbol: {symbol}\n"
            f"Condition: {metric.value} {comparator} {threshold}\n"
            f"Current Value: {value}\n"
            f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )
        
    def monitor_task(self, alert_config: Dict[str, Any]) -> None:
        """用于调度器执行的监控任务"""
        self.check_alert(alert_config)
        
    def test_notification(self, alert_config: Dict[str, Any]) -> bool:
        """测试通知功能，忽略条件直接发送"""
        return self.check_alert(alert_config, test_mode=True)