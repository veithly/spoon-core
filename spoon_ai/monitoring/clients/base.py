# spoon_ai/monitoring/clients/base.py
import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)

class DataClient(ABC):
    """数据客户端抽象基类"""
    
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
    
    @classmethod
    def get_client(cls, market: str, provider: str) -> 'DataClient':
        """工厂方法：根据市场和提供者创建适当的客户端"""
        # CEX客户端
        if market.lower() == "cex":
            from .cex import get_cex_client
            return get_cex_client(provider)
            
        # DEX客户端
        elif market.lower() == "dex":
            from .dex import get_dex_client
            return get_dex_client(provider)
            
        # 其他市场类型
        else:
            raise ValueError(f"Unsupported market type: {market}")