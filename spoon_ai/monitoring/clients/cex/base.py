# spoon_ai/monitoring/clients/cex/base.py
import logging
from abc import abstractmethod
from typing import Dict, Any, List, Optional

from ..base import DataClient

logger = logging.getLogger(__name__)

class CEXClient(DataClient):
    """中心化交易所客户端基类"""
    
    @abstractmethod
    def get_ticker_price(self, symbol: str) -> Dict[str, Any]:
        """获取交易对价格"""
        pass
    
    @abstractmethod
    def get_ticker_24h(self, symbol: str) -> Dict[str, Any]:
        """获取24小时统计数据"""
        pass
    
    @abstractmethod
    def get_klines(self, symbol: str, interval: str, limit: int = 500) -> List[Any]:
        """获取K线数据"""
        pass
    
    @abstractmethod
    def get_server_time(self) -> int:
        """获取服务器时间"""
        pass