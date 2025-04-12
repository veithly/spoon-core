import logging
from typing import Dict, Any, List, Optional
import time
import asyncio
import json
import requests
from solana.rpc.api import Client as SolanaClient

from .base import DEXClient

logger = logging.getLogger(__name__)

class RaydiumClient(DEXClient):
    """Raydium (Solana) DEX client"""
    
    def __init__(self, rpc_url: Optional[str] = None):
        self.rpc_url = rpc_url or "https://api.mainnet-beta.solana.com"
        self.solana_client = SolanaClient(self.rpc_url)
        self.raydium_pools_url = "https://api.raydium.io/v2/main/pairs"
        self._pools_cache = {}
        self._pools_cache_timestamp = 0
        self._cache_ttl = 300  # 5 minutes cache
    
    def _refresh_pools_cache(self) -> None:
        """Refresh the Raydium pools cache"""
        current_time = time.time()
        if current_time - self._pools_cache_timestamp > self._cache_ttl:
            try:
                response = requests.get(self.raydium_pools_url)
                response.raise_for_status()
                self._pools_cache = {
                    f"{pool['name']}": pool
                    for pool in response.json()['data']
                }
                self._pools_cache_timestamp = current_time
                logger.info(f"Refreshed Raydium pools cache, found {len(self._pools_cache)} pools")
            except Exception as e:
                logger.error(f"Failed to refresh Raydium pools: {str(e)}")
                if not self._pools_cache:  # Only raise if we don't have any cached data
                    raise
    
    def _get_pool_by_symbol(self, symbol: str) -> Dict[str, Any]:
        """Get pool data by symbol (e.g., 'SOL-USDC')"""
        self._refresh_pools_cache()
        
        if symbol in self._pools_cache:
            return self._pools_cache[symbol]
        
        # Try alternative formats if exact match not found
        symbol_parts = symbol.split('-')
        if len(symbol_parts) == 2:
            reversed_symbol = f"{symbol_parts[1]}-{symbol_parts[0]}"
            if reversed_symbol in self._pools_cache:
                return self._pools_cache[reversed_symbol]
        
        available_symbols = list(self._pools_cache.keys())[:10]  # List first 10 as examples
        raise ValueError(f"Symbol {symbol} not found in Raydium pools. Available symbols include: {available_symbols}...")
    
    def get_ticker_price(self, symbol: str) -> Dict[str, Any]:
        """Get trading pair price
        
        Symbol should be in the format "TOKEN0-TOKEN1", e.g., "SOL-USDC"
        """
        logger.info(f"Getting Raydium price for: {symbol}")
        try:
            pool_data = self._get_pool_by_symbol(symbol)
            return {
                "symbol": symbol,
                "price": float(pool_data.get("price", 0)),
                "last_price": float(pool_data.get("price", 0)),
                "base_volume": float(pool_data.get("volume24h", 0)),
                "quote_volume": float(pool_data.get("volume24h", 0)) * float(pool_data.get("price", 0)),
                "time": int(time.time() * 1000),
                "amm_id": pool_data.get("ammId", ""),
                "lp_mint": pool_data.get("lpMint", ""),
                "market_id": pool_data.get("marketId", ""),
                "liquidity": float(pool_data.get("liquidity", 0)),
            }
        except Exception as e:
            logger.error(f"Error getting Raydium price for {symbol}: {str(e)}")
            raise
    
    def get_ticker_24h(self, symbol: str) -> Dict[str, Any]:
        """Get 24-hour price change statistics"""
        logger.info(f"Getting Raydium 24h data for: {symbol}")
        try:
            pool_data = self._get_pool_by_symbol(symbol)
            return {
                "symbol": symbol,
                "price_change": float(pool_data.get("priceChange24h", 0)),
                "price_change_percent": float(pool_data.get("priceChange24hPercent", 0)) * 100,  # Convert to percentage
                "volume": float(pool_data.get("volume24h", 0)),
                "volume_change_percent": float(pool_data.get("volumeChange24hPercent", 0)) * 100,  # Convert to percentage
                "liquidity": float(pool_data.get("liquidity", 0)),
                "last_price": float(pool_data.get("price", 0)),
                "time": int(time.time() * 1000),
            }
        except Exception as e:
            logger.error(f"Error getting Raydium 24h data for {symbol}: {str(e)}")
            raise
    
    def get_klines(self, symbol: str, interval: str, limit: int = 500) -> List[List]:
        """Get K-line data
        
        Note: Raydium doesn't provide direct K-line data through their API.
        This would need to be built using historical data from another source or
        by tracking price changes over time.
        """
        logger.info(f"Getting Raydium K-line data: {symbol}, interval: {interval}, limit: {limit}")
        # For production, would need to integrate with a service that provides historical 
        # Raydium price data such as a custom indexer or other data provider
        
        # For now, returning a simulated response with current price data
        pool_data = self._get_pool_by_symbol(symbol)
        current_price = float(pool_data.get("price", 0))
        current_time = int(time.time() * 1000)
        
        # Return a basic placeholder with the current price repeated
        # In a real implementation, this would fetch actual historical data
        interval_seconds = self._parse_interval_to_seconds(interval)
        
        # Mock structure: [timestamp, open, high, low, close, volume]
        klines = []
        for i in range(limit):
            timestamp = current_time - (limit - i - 1) * interval_seconds * 1000
            # Simple price simulation
            variation = 0.01 * (((i % 10) - 5) / 5.0)
            simulated_price = current_price * (1 + variation)
            kline = [
                timestamp,                   # Open time
                simulated_price,             # Open
                simulated_price * 1.01,      # High
                simulated_price * 0.99,      # Low
                simulated_price,             # Close
                pool_data.get("volume24h", 0) / 24.0,  # Volume (divided by 24 for hourly estimate)
            ]
            klines.append(kline)
        
        return klines
    
    def _parse_interval_to_seconds(self, interval: str) -> int:
        """Parse interval string to seconds"""
        unit = interval[-1]
        value = int(interval[:-1])
        
        if unit == "m":
            return value * 60
        elif unit == "h":
            return value * 60 * 60
        elif unit == "d":
            return value * 24 * 60 * 60
        elif unit == "w":
            return value * 7 * 24 * 60 * 60
        else:
            return 60 * 60  # Default to 1 hour if parsing fails