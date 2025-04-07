# spoon_ai/monitoring/clients/base.py
import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)

class DataClient(ABC):
    """Abstract base class for data clients"""
    
    @abstractmethod
    def get_ticker_price(self, symbol: str) -> Dict[str, Any]:
        """Get trading pair price"""
        pass
    
    @abstractmethod
    def get_ticker_24h(self, symbol: str) -> Dict[str, Any]:
        """Get 24-hour statistics"""
        pass
    
    @abstractmethod
    def get_klines(self, symbol: str, interval: str, limit: int = 500) -> List[Any]:
        """Get K-line data"""
        pass
    
    @classmethod
    def get_client(cls, market: str, provider: str) -> 'DataClient':
        """Factory method: create appropriate client based on market and provider"""
        # CEX client
        if market.lower() == "cex":
            from .cex import get_cex_client
            return get_cex_client(provider)
            
        # DEX client
        elif market.lower() == "dex":
            from .dex import get_dex_client
            return get_dex_client(provider)
            
        # Other market types
        else:
            raise ValueError(f"Unsupported market type: {market}")