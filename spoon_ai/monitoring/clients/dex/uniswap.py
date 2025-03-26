# spoon_ai/monitoring/clients/dex/uniswap.py
import logging
from typing import Dict, Any, List, Optional
import time
import asyncio

from .base import DEXClient
from spoon_ai.tools.dex.price_data import UniswapPriceProvider

logger = logging.getLogger(__name__)

class UniswapClient(DEXClient):
    """Uniswap API client"""
    
    def __init__(self, rpc_url: Optional[str] = None):
        self.rpc_url = rpc_url or "https://eth-mainnet.g.alchemy.com/v2/demo"
        self.provider = UniswapPriceProvider(rpc_url=self.rpc_url)
    
    def get_ticker_price(self, symbol: str) -> Dict[str, Any]:
        """Get trading pair price
        
        In Uniswap, symbol should be in the format "TOKEN0-TOKEN1",
        e.g., "ETH-USDC" represents the ETH/USDC trading pair
        """
        logger.info(f"Getting Uniswap price for: {symbol}")
        # Run the async method in a synchronous context
        return asyncio.run(self.provider.get_ticker_price(symbol))
            
    def get_ticker_24h(self, symbol: str) -> Dict[str, Any]:
        """Get 24-hour price change statistics"""
        logger.info(f"Getting Uniswap 24h data for: {symbol}")
        # Run the async method in a synchronous context
        return asyncio.run(self.provider.get_ticker_24h(symbol))
    
    def get_klines(self, symbol: str, interval: str, limit: int = 500) -> List[List]:
        """Get K-line data
        
        Note: Uniswap contracts don't provide K-line data directly, this would typically
        need to be obtained from a Graph API or by collecting and processing historical event data.
        
        This method would need to integrate with a service like The Graph to get real K-line data.
        """
        logger.info(f"Getting Uniswap K-line data: {symbol}, interval: {interval}, limit: {limit}")
        # Run the async method in a synchronous context
        return asyncio.run(self.provider.get_klines(symbol, interval, limit))