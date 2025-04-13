import logging
from typing import Dict, Any, List, Optional
import time

from spoon_ai.tools.crypto.price_data import RaydiumPriceProvider
from .base import DEXClient

logger = logging.getLogger(__name__)

class RaydiumClient(DEXClient):
    """Raydium (Solana) DEX client with V3 API support"""
    
    def __init__(self, rpc_url: Optional[str] = None):
        """Initialize Raydium client with optional RPC URL"""
        self.provider = RaydiumPriceProvider(rpc_url)
    
    def get_tvl_and_volume(self) -> Dict[str, float]:
        """Get TVL and 24h volume information"""
        return self.provider.get_tvl_and_volume()
    
    def get_mint_prices(self, mint_ids: List[str]) -> Dict[str, str]:
        """Get prices for multiple token mints"""
        return self.provider.get_mint_prices(mint_ids)
    
    def get_mint_info(self, mint_ids: List[str]) -> List[Dict[str, Any]]:
        """Get detailed information for token mints"""
        return self.provider.get_mint_info(mint_ids)
    
    def get_pools_list(self, 
                       pool_type: str = "all", 
                       sort_field: str = "liquidity", 
                       sort_type: str = "desc", 
                       page_size: int = 100, 
                       page: int = 1) -> List[Dict[str, Any]]:
        """
        Get list of pools with sorting and pagination
        
        Args:
            pool_type: Type of pools ("all", "concentrated", "standard", "allFarm", etc.)
            sort_field: Field to sort by (default, liquidity, volume24h, etc.)
            sort_type: Sort direction (desc, asc)
            page_size: Number of results per page (max 1000)
            page: Page number
        """
        return self.provider.get_pools_list(
            pool_type=pool_type,
            sort_field=sort_field,
            sort_type=sort_type,
            page_size=page_size,
            page=page
        )
    
    def get_pool_info_by_ids(self, pool_ids: List[str]) -> List[Dict[str, Any]]:
        """Get pool information by pool IDs"""
        return self.provider.get_pool_info_by_ids(pool_ids)
    
    def get_pool_info_by_lp_mints(self, lp_mints: List[str]) -> List[Dict[str, Any]]:
        """Get pool information by LP token mints"""
        return self.provider.get_pool_info_by_lp_mints(lp_mints)
    
    def get_pool_liquidity_history(self, pool_id: str) -> Dict[str, Any]:
        """Get pool liquidity history (max 30 days)"""
        return self.provider.get_pool_liquidity_history(pool_id)
    
    def get_ticker_price(self, symbol: str) -> Dict[str, Any]:
        """Get trading pair price (direct synchronous implementation)"""
        try:
            # For BTC, ETH, SOL and other major tokens, get price directly by mint
            if symbol.upper() in self.provider.TOKEN_MINTS:
                mint_address = self.provider.TOKEN_MINTS[symbol.upper()]
                mint_prices = self.provider.get_mint_prices([mint_address])
                
                if mint_address in mint_prices:
                    price = float(mint_prices[mint_address])
                    return {
                        "symbol": symbol,
                        "price": price,
                        "last_price": price,
                        "time": int(time.time() * 1000)
                    }
            
            # For trading pairs, try to get pool data
            if symbol in self.provider.COMMON_POOLS:
                pool_id = self.provider.COMMON_POOLS[symbol]
                pools_data = self.provider.get_pool_info_by_ids([pool_id])
                
                if pools_data and len(pools_data) > 0 and pools_data[0]:
                    pool_data = pools_data[0]
                    price = 0
                    
                    # Try to extract price
                    for price_field in ['price', 'currentPrice', 'lastPrice']:
                        if price_field in pool_data and pool_data[price_field]:
                            try:
                                price = float(pool_data[price_field])
                                break
                            except (ValueError, TypeError):
                                continue
                    
                    return {
                        "symbol": symbol,
                        "price": price,
                        "last_price": price,
                        "time": int(time.time() * 1000)
                    }
            
            # Default response if not found
            logger.warning(f"Unable to get price for {symbol}")
            return {
                "symbol": symbol,
                "price": 0,
                "last_price": 0,
                "time": int(time.time() * 1000)
            }
        except Exception as e:
            logger.error(f"Error getting ticker price for {symbol}: {str(e)}")
            return {
                "symbol": symbol,
                "price": 0,
                "last_price": 0,
                "time": int(time.time() * 1000)
            }
    
    def get_ticker_24h(self, symbol: str) -> Dict[str, Any]:
        """Get 24-hour price change statistics (direct implementation)"""
        try:
            # Check if symbol is a common pool
            pool_id = None
            if symbol in self.provider.COMMON_POOLS:
                pool_id = self.provider.COMMON_POOLS[symbol]
            elif len(symbol) > 30:  # Likely a pool ID
                pool_id = symbol
            
            if pool_id:
                pools_data = self.provider.get_pool_info_by_ids([pool_id])
                if pools_data and len(pools_data) > 0 and pools_data[0]:
                    pool_data = pools_data[0]
                    
                    # Extract data
                    price = float(pool_data.get("price", 0))
                    volume = 0
                    liquidity = 0
                    
                    # Try to get volume
                    if 'day' in pool_data and isinstance(pool_data['day'], dict) and 'volume' in pool_data['day']:
                        try:
                            volume = float(pool_data['day']['volume'])
                        except (ValueError, TypeError):
                            pass
                    
                    # Try to get liquidity
                    for liq_field in ['liquidity', 'tvl']:
                        if liq_field in pool_data and pool_data[liq_field] is not None:
                            try:
                                liquidity = float(pool_data[liq_field])
                                break
                            except (ValueError, TypeError):
                                continue
                    
                    return {
                        "symbol": symbol,
                        "price_change": 0,  # No reliable way to get this directly
                        "price_change_percent": 0,
                        "volume": volume,
                        "volume_change_percent": 0,
                        "liquidity": liquidity,
                        "last_price": price,
                        "time": int(time.time() * 1000)
                    }
            
            # Default response if not found
            return {
                "symbol": symbol,
                "price_change": 0,
                "price_change_percent": 0,
                "volume": 0,
                "volume_change_percent": 0,
                "liquidity": 0,
                "last_price": 0,
                "time": int(time.time() * 1000)
            }
        except Exception as e:
            logger.error(f"Error getting 24h data for {symbol}: {str(e)}")
            return {
                "symbol": symbol,
                "price_change": 0,
                "price_change_percent": 0,
                "volume": 0,
                "volume_change_percent": 0,
                "liquidity": 0,
                "last_price": 0,
                "time": int(time.time() * 1000)
            }
    
    def get_klines(self, symbol: str, interval: str, limit: int = 500) -> List[List]:
        """Get K-line data (simplified implementation)"""
        try:
            # Get current price to generate mock data
            current_data = self.get_ticker_price(symbol)
            current_price = current_data["price"]
            
            # Generate mock kline data
            return self._generate_mock_klines(current_price, interval, limit)
        except Exception as e:
            logger.error(f"Error getting klines for {symbol}: {str(e)}")
            return []
    
    def _generate_mock_klines(self, current_price: float, interval: str, limit: int) -> List[List]:
        """Generate mock kline data similar to provider's implementation"""
        current_time = int(time.time() * 1000)
        interval_seconds = self._parse_interval_to_seconds(interval)
        
        klines = []
        for i in range(limit):
            timestamp = current_time - (limit - i - 1) * interval_seconds * 1000
            variation = 0.01 * (((i % 10) - 5) / 5.0)
            simulated_price = current_price * (1 + variation) if current_price > 0 else 1000
            
            kline = [
                timestamp,               # Open time
                simulated_price,         # Open price
                simulated_price * 1.01,  # High price
                simulated_price * 0.99,  # Low price
                simulated_price,         # Close price
                0,                       # Volume
            ]
            klines.append(kline)
        
        return klines
    
    def _parse_interval_to_seconds(self, interval: str) -> int:
        """Parse interval string to seconds"""
        unit = interval[-1]
        try:
            value = int(interval[:-1])
        except ValueError:
            return 60 * 60  # Default to 1 hour
        
        if unit == "m":
            return value * 60
        elif unit == "h":
            return value * 60 * 60
        elif unit == "d":
            return value * 24 * 60 * 60
        elif unit == "w":
            return value * 7 * 24 * 60 * 60
        else:
            return 60 * 60  # Default to 1 hour