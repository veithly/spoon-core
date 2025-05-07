import json
import logging
import time
from typing import Dict, Any, List, Optional, Union
import os

from web3 import Web3, HTTPProvider
from web3.middleware import ExtraDataToPOAMiddleware
from pydantic import Field, validator

from spoon_ai.tools.base import BaseTool, ToolResult
from spoon_ai.tools.crypto.base import DexBaseTool
from solana.rpc.api import Client as SolanaClient
import nest_asyncio
import requests

logger = logging.getLogger(__name__)

# Uniswap V3 Factory ABI (only includes functions we need)
UNISWAP_FACTORY_ABI = json.loads('''
[
  {
    "inputs": [
      {"internalType": "address", "name": "tokenA", "type": "address"},
      {"internalType": "address", "name": "tokenB", "type": "address"},
      {"internalType": "uint24", "name": "fee", "type": "uint24"}
    ],
    "name": "getPool",
    "outputs": [{"internalType": "address", "name": "", "type": "address"}],
    "stateMutability": "view",
    "type": "function"
  }
]
''')

# Uniswap V3 Pool ABI (only includes functions we need)
UNISWAP_POOL_ABI = json.loads('''
[
  {
    "inputs": [],
    "name": "slot0",
    "outputs": [
      {"internalType": "uint160", "name": "sqrtPriceX96", "type": "uint160"},
      {"internalType": "int24", "name": "tick", "type": "int24"},
      {"internalType": "uint16", "name": "observationIndex", "type": "uint16"},
      {"internalType": "uint16", "name": "observationCardinality", "type": "uint16"},
      {"internalType": "uint16", "name": "observationCardinalityNext", "type": "uint16"},
      {"internalType": "uint8", "name": "feeProtocol", "type": "uint8"},
      {"internalType": "bool", "name": "unlocked", "type": "bool"}
    ],
    "stateMutability": "view",
    "type": "function"
  },
  {
    "inputs": [{"internalType": "uint32[]", "name": "secondsAgos", "type": "uint32[]"}],
    "name": "observe",
    "outputs": [
      {"internalType": "int56[]", "name": "tickCumulatives", "type": "int56[]"},
      {"internalType": "uint160[]", "name": "secondsPerLiquidityCumulativeX128s", "type": "uint160[]"}
    ],
    "stateMutability": "view",
    "type": "function"
  }
]
''')

# ERC20 interface for getting token information
ERC20_ABI = json.loads('''
[
  {
    "constant": true,
    "inputs": [],
    "name": "decimals",
    "outputs": [{"name": "", "type": "uint8"}],
    "payable": false,
    "stateMutability": "view",
    "type": "function"
  },
  {
    "constant": true,
    "inputs": [],
    "name": "symbol",
    "outputs": [{"name": "", "type": "string"}],
    "payable": false,
    "stateMutability": "view",
    "type": "function"
  }
]
''')

# Common token address mappings
TOKEN_ADDRESSES = {
    "ETH": "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2",  # WETH
    "USDC": "0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48",
    "USDT": "0xdAC17F958D2ee523a2206206994597C13D831ec7",
    "DAI": "0x6B175474E89094C44Da98b954EedeAC495271d0F",
    "WBTC": "0x2260FAC5E5542a773Aa44fBCfeDf7C193bc2C599"
}

# Uniswap V3 factory address
UNISWAP_V3_FACTORY = "0x1F98431c8aD98523631AE4a59f267346ea31F984"

class PriceDataProvider:
    """Base class for price data providers"""
    
    async def get_ticker_price(self, symbol: str) -> Dict[str, Any]:
        """Get trading pair price"""
        raise NotImplementedError("Subclasses must implement this method")
    
    async def get_ticker_24h(self, symbol: str) -> Dict[str, Any]:
        """Get 24-hour price change statistics"""
        raise NotImplementedError("Subclasses must implement this method")
    
    async def get_klines(self, symbol: str, interval: str, limit: int = 500) -> List[List]:
        """Get K-line data"""
        raise NotImplementedError("Subclasses must implement this method")

class UniswapPriceProvider(PriceDataProvider):
    """Uniswap price data provider"""
    
    def __init__(self, rpc_url: Optional[str] = None):
        if not rpc_url:
            rpc_url = os.getenv("RPC_URL")
        if not rpc_url:
            raise ValueError("rpc_url is required")
        self.rpc_url = rpc_url
        self.w3 = Web3(HTTPProvider(self.rpc_url))
        self.w3.middleware_onion.inject(ExtraDataToPOAMiddleware, layer=0)

        self.factory = self.w3.eth.contract(
            address=self.w3.to_checksum_address("0x1F98431c8aD98523631AE4a59f267346ea31F984"), 
            abi=UNISWAP_FACTORY_ABI
        )
        
        # Add client-side rate limiting
        self.last_request_time = 0
        self.min_request_interval = 1.0  # seconds
        
    def _rate_limit(self):
        """Implement simple rate limiting"""
        current_time = time.time()
        elapsed = current_time - self.last_request_time
        if elapsed < self.min_request_interval:
            sleep_time = self.min_request_interval - elapsed
            logger.info(f"Rate limiting: sleeping for {sleep_time:.2f} seconds")
            time.sleep(sleep_time)
        self.last_request_time = time.time()
    
    def _get_token_addresses(self, symbol: str) -> tuple:
        """Get token addresses from trading pair symbol"""
        # Same implementation as original
        self._rate_limit()
        tokens = symbol.split("-")
        if len(tokens) != 2:
            raise ValueError(f"Invalid trading pair format: {symbol}. Expected format: TOKEN0-TOKEN1")
            
        token0_symbol, token1_symbol = tokens
        
        # Common token address mappings
        TOKEN_ADDRESSES = {
            "ETH": "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2",  # WETH
            "USDC": "0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48",
            "USDT": "0xdAC17F958D2ee523a2206206994597C13D831ec7",
            "DAI": "0x6B175474E89094C44Da98b954EedeAC495271d0F",
            "WBTC": "0x2260FAC5E5542a773Aa44fBCfeDf7C193bc2C599"
        }
        
        token0_address = TOKEN_ADDRESSES.get(token0_symbol.upper())
        token1_address = TOKEN_ADDRESSES.get(token1_symbol.upper())
        
        if not token0_address or not token1_address:
            raise ValueError(f"Token address not found: {token0_symbol} or {token1_symbol}")
            
        return (self.w3.to_checksum_address(token0_address), 
                self.w3.to_checksum_address(token1_address))
    
    def _get_pool_address(self, token0: str, token1: str, fee: int = 3000) -> str:
        """Get Uniswap V3 pool address"""
        self._rate_limit()
        # Ensure token0 and token1 are sorted by address
        if int(token0, 16) > int(token1, 16):
            token0, token1 = token1, token0
            
        pool_address = self.factory.functions.getPool(token0, token1, fee).call()
        if pool_address == "0x0000000000000000000000000000000000000000":
            raise ValueError(f"Pool not found: {token0}-{token1} with fee {fee}")
        return pool_address
    
    def _get_token_decimals(self, token_address: str) -> int:
        """Get token decimals"""
        self._rate_limit()
        token_contract = self.w3.eth.contract(address=token_address, abi=ERC20_ABI)
        return token_contract.functions.decimals().call()
    
    def _calculate_price_from_sqrt_price_x96(self, sqrt_price_x96: int, decimals0: int, decimals1: int) -> float:
        """Calculate price from sqrtPriceX96"""
        # Convert sqrtPriceX96 to price
        price = (sqrt_price_x96 / 2**96) ** 2
        # Adjust for decimals
        adjusted_price = price * (10 ** (decimals0 - decimals1))
        return adjusted_price
    
    async def get_ticker_price(self, symbol: str, fee: int = 3000) -> Dict[str, Any]:
        """Get trading pair price with improved error handling and rate limiting"""
        try:
            logger.info(f"Getting Uniswap price for: {symbol}")
            token0_address, token1_address = self._get_token_addresses(symbol)
            
            # Get pool address with specified fee
            try:
                pool_address = self._get_pool_address(token0_address, token1_address, fee)
            except ValueError as e:
                if "Pool not found" in str(e):
                    return {
                        "price": "0",
                        "pair": symbol,
                        "timestamp": int(time.time()),
                        "error": f"Pool not found for fee {fee}"
                    }
                raise
                
            pool_contract = self.w3.eth.contract(address=pool_address, abi=UNISWAP_POOL_ABI)
            
            # Rate limit before making the next calls
            self._rate_limit()
            
            # Get current price
            slot0 = pool_contract.functions.slot0().call()
            sqrt_price_x96 = slot0[0]
            
            # Get token decimals
            decimals0 = self._get_token_decimals(token0_address)
            decimals1 = self._get_token_decimals(token1_address)
            
            # Calculate price
            price = self._calculate_price_from_sqrt_price_x96(sqrt_price_x96, decimals0, decimals1)
            
            return {
                "price": str(price),
                "pair": symbol,
                "timestamp": int(time.time()),
                "error": None
            }
        except ValueError as e:
            # Handle case where pool does not exist
            logger.error(f"Pool not found for {symbol} with fee {fee}: {e}")
            return {
                "price": "0",
                "pair": symbol,
                "timestamp": int(time.time()),
                "error": f"Pool not found for fee {fee}: {str(e)}"
            }
        except Exception as e:
            logger.error(f"Failed to get Uniswap price: {e}")
            return {
                "price": "0",
                "pair": symbol,
                "timestamp": int(time.time()),
                "error": str(e)
            }
            
    async def get_ticker_24h(self, symbol: str) -> Dict[str, Any]:
        """Get 24-hour price change statistics with improved error handling"""
        try:
            logger.info(f"Getting Uniswap 24h data for: {symbol}")
            token0_address, token1_address = self._get_token_addresses(symbol)
            
            # Get pool address
            try:
                pool_address = self._get_pool_address(token0_address, token1_address)
                pool_contract = self.w3.eth.contract(address=pool_address, abi=UNISWAP_POOL_ABI)
            except ValueError as e:
                logger.error(f"Failed to get pool for {symbol}: {e}")
                return {
                    "price": "0",
                    "volume": "0",
                    "priceChange": "0",
                    "priceChangePercent": "0",
                    "pair": symbol,
                    "timestamp": int(time.time()),
                    "error": f"Failed to get pool: {str(e)}"
                }
            
            # Rate limit before making the next calls
            self._rate_limit()
            
            # Get current price
            slot0 = pool_contract.functions.slot0().call()
            sqrt_price_x96 = slot0[0]
            
            # Rate limit again
            self._rate_limit()
            
            decimals0 = self._get_token_decimals(token0_address)
            decimals1 = self._get_token_decimals(token1_address)
            current_P = (sqrt_price_x96 / 2**96) ** 2
            current_Q = current_P * 10 ** (decimals0 - decimals1)
            
            try:
                # Get observations for time-weighted average
                secondsAgos = [86400, 0]  # 24 hours ago and now
                self._rate_limit()
                result = pool_contract.functions.observe(secondsAgos).call()
                tickCumulatives = result[0]  # [tickCumulative_24h_ago, tickCumulative_now]
                
                tickCumulative_24h_ago = tickCumulatives[0]
                tickCumulative_now = tickCumulatives[1]
                delta_tickCumulative = tickCumulative_now - tickCumulative_24h_ago
                time_interval = 86400
                
                if delta_tickCumulative == 0:
                    # Not enough data; use current price as average
                    avg_tick = slot0[1]  # Current tick
                else:
                    avg_tick = delta_tickCumulative / time_interval
                
                # Compute average price
                avg_P = (1.0001 ** avg_tick) ** 2
                avg_Q = avg_P * 10 ** (decimals0 - decimals1)
                
                # Compute price change
                price_change = current_Q - avg_Q
                price_change_percent = (price_change / avg_Q) * 100 if avg_Q != 0 else 0
            except Exception as e:
                logger.warning(f"Failed to get historical data for {symbol}, using current price only: {e}")
                # If we can't get historical data, just use the current price
                price_change = 0
                price_change_percent = 0
            
            # Volume is not available directly; set to 0
            volume = 0
            
            return {
                "price": str(current_Q),
                "volume": str(volume),
                "priceChange": str(price_change),
                "priceChangePercent": str(price_change_percent),
                "pair": symbol,
                "timestamp": int(time.time()),
                "error": None
            }
        except Exception as e:
            logger.error(f"Failed to get Uniswap 24h data: {e}")
            return {
                "price": "0",
                "volume": "0",
                "priceChange": "0",
                "priceChangePercent": "0",
                "pair": symbol,
                "timestamp": int(time.time()),
                "error": str(e)
            }
    
    async def get_klines(self, symbol: str, interval: str, limit: int = 500) -> List[List]:
        """Get K-line data (placeholder implementation)"""
        logger.info(f"Getting Uniswap K-line data: {symbol}, interval: {interval}, limit: {limit}")
        logger.warning("Uniswap contracts don't provide K-line data directly, need to use Graph API or event logs")
        
        # Return empty list, actual implementation should integrate with Graph API
        return []
        
class RaydiumPriceProvider(PriceDataProvider):
    """Raydium (Solana) price data provider with V3 API support"""
    
    API_BASE = "https://api-v3.raydium.io"
    CACHE_TIMEOUT = 300  # 5 minutes cache
    
    # Token address mapping - using accurate addresses
    TOKEN_MINTS = {
        "SOL": "So11111111111111111111111111111111111111112",
        "USDC": "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v",
        "BTC": "3NZ9JMVBmGAqocybic2c7LQCJScmgsAZ6vQqTDzcqmJh",  # Wormhole wrapped BTC
        "ETH": "7vfCXTUXx5WJV5JADk17DUJ4ksgau7utNKj4b963voxs",  # Wormhole wrapped ETH
        "RAY": "4k3Dyjzvzp8eMZWUXbBCjEvwSkkk59S5iCNLY3QrkX6R",  # RAY token
        "USDT": "Es9vMFrzaCERmJfrF4H2FYD4KCoNkY11McCe8BenwNYB"  # USDT
    }
    
    # Common pool ID mapping - for backward compatibility
    COMMON_POOLS = {
        "SOL-USDC": "58oQChx4yWmvKdwLLZzBi4ChoCc2fqCUWBkwMihLYQo2",
        "BTC-USDC": "6kbC5epG18DF2DwPEW34tBy5pGFS7pEGALR3v5MGxgc5",
        "WBTC-USDC": "5qoTq3qC4U7vFxo3iCzbXcaD1UJK36EjNNjuv6MHzYAD",
        "ETH-USDC": "5ZgP9EmPFPsYGaqzuHt6h1o4Q1PfJSwxUBaYNxXGaKeZ",
        "RAY-USDC": "6UmmUiYoBjSrhakAobJw8BvkmJtDVxaeBtbt7rxWo1mg",
        "RAY-SOL": "AVs9TA4nWDzfPJE9gGVNJMVhcQy3V9PGazuz33BfG2RA",
        "USDT-USDC": "Epm4KfTj4DMrvqn6Bwg2Tr2N8vhQuNbuK8bESFp4k33K"
    }
    
    def __init__(self, rpc_url: Optional[str] = None, session=None):
        """Initialize Raydium price provider"""
        self.rpc_url = rpc_url or "https://api.mainnet-beta.solana.com"
        self.solana_client = SolanaClient(self.rpc_url)
        self.session = session
        self._pools_cache = {}
        self._last_pools_refresh = 0
        self._mint_prices_cache = {}
        self._last_mint_prices_refresh = 0
    
    def _make_request(self, endpoint: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Send request to Raydium V3 API"""
        url = f"{self.API_BASE}{endpoint}"
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            
            # V3 API contains success/failure information
            if isinstance(data, dict) and data.get('success') is False:
                error_message = data.get('msg', 'Unknown error')
                logger.error(f"Raydium API error: {error_message}")
                raise ValueError(f"Raydium API error: {error_message}")
            
            # Handle wrapped and unwrapped responses
            if isinstance(data, dict) and 'data' in data:
                return data.get('data', {})
            return data
            
        except Exception as e:
            logger.error(f"API request failed for {endpoint}: {str(e)}")
            raise
    
    def get_tvl_and_volume(self) -> Dict[str, float]:
        """Get TVL and 24h trading volume information"""
        logger.info("Getting Raydium TVL and 24h volume information")
        data = self._make_request("/main/info")
        return {
            "tvl": float(data.get("tvl", 0)),
            "volume24h": float(data.get("volume24", 0))
        }
    
    def get_mint_prices(self, mint_ids: List[str]) -> Dict[str, str]:
        """Get prices for multiple token addresses"""
        logger.info(f"Getting prices for {len(mint_ids)} mint IDs")
        current_time = time.time()
        
        # Try to get prices from cache
        missing_mints = []
        result_prices = {}
        
        for mint_id in mint_ids:
            if mint_id in self._mint_prices_cache and current_time - self._last_mint_prices_refresh < self.CACHE_TIMEOUT:
                result_prices[mint_id] = self._mint_prices_cache[mint_id]
            else:
                missing_mints.append(mint_id)
        
        # If there are missing prices, get them
        if missing_mints:
            params = {"mints": ",".join(missing_mints)}
            try:
                new_prices = self._make_request("/mint/price", params)
                
                # Update cache
                self._mint_prices_cache.update(new_prices)
                self._last_mint_prices_refresh = current_time
                
                # Update results
                result_prices.update(new_prices)
            except Exception as e:
                logger.error(f"Error fetching mint prices: {str(e)}")
                # Return existing cached prices instead of failing
        
        return result_prices
    
    def get_mint_info(self, mint_ids: List[str]) -> List[Dict[str, Any]]:
        """Get detailed information for token addresses"""
        logger.info(f"Getting info for {len(mint_ids)} mint IDs")
        params = {"mints": ",".join(mint_ids)}
        return self._make_request("/mint/ids", params)
    
    def get_pools_list(self, 
                       pool_type: str = "all", 
                       sort_field: str = "liquidity", 
                       sort_type: str = "desc", 
                       page_size: int = 100, 
                       page: int = 1) -> Dict[str, Any]:
        """
        Get pool list, supporting sorting and pagination
        
        Args:
            pool_type: Pool type ("all", "concentrated", "standard", "allFarm", etc.)
            sort_field: Sort field (default, liquidity, volume24h, etc.)
            sort_type: Sort direction (desc, asc)
            page_size: Results per page (max 1000)
            page: Page number
        """
        logger.info(f"Getting pools list: type={pool_type}, sort={sort_field}, page={page}")
        params = {
            "poolType": pool_type,
            "poolSortField": sort_field,
            "sortType": sort_type,
            "pageSize": page_size,
            "page": page
        }
        return self._make_request("/pools/info/list", params)
    
    def get_pool_info_by_ids(self, pool_ids: List[str]) -> List[Dict[str, Any]]:
        """Get pool information by pool IDs"""
        logger.info(f"Getting pool info for {len(pool_ids)} pool IDs")
        params = {"ids": ",".join(pool_ids)}
        return self._make_request("/pools/info/ids", params)
    
    def get_pool_info_by_lp_mints(self, lp_mints: List[str]) -> List[Dict[str, Any]]:
        """Get pool information by LP token addresses"""
        logger.info(f"Getting pool info for {len(lp_mints)} LP mints")
        params = {"lps": ",".join(lp_mints)}
        return self._make_request("/pools/info/lps", params)
    
    def get_pool_liquidity_history(self, pool_id: str) -> Dict[str, Any]:
        """Get pool liquidity history (up to 30 days)"""
        logger.info(f"Getting liquidity history for pool {pool_id}")
        params = {"id": pool_id}
        return self._make_request("/pools/line/liquidity", params)
    
    def _create_default_response(self, identifier: str) -> Dict[str, Any]:
        """Create default response object"""
        return {
            "symbol": f"token-{identifier[:8]}" if len(identifier) > 8 else identifier,
            "price": 0,
            "last_price": 0,
            "base_volume": 0,
            "quote_volume": 0,
            "time": int(time.time() * 1000),
            "amm_id": "",
            "lp_mint": "",
            "market_id": "",
            "liquidity": 0,
        }
    
    async def get_ticker_price_by_mint(self, symbol: str) -> Dict[str, Any]:
        """Get price via token address - recommended for special tokens like BTC"""
        logger.info(f"Getting {symbol} price via Mint address")
        
        try:
            # Get token address
            if symbol.upper() in self.TOKEN_MINTS:
                mint_address = self.TOKEN_MINTS[symbol.upper()]
            else:
                # Assume input might be an address directly
                mint_address = symbol
            
            # Get USDC address (for USD value calculation)
            usdc_mint = self.TOKEN_MINTS["USDC"]
            
            # Get prices
            mint_prices = self.get_mint_prices([mint_address, usdc_mint])
            
            # Check if prices were retrieved
            if mint_address not in mint_prices:
                logger.error(f"Unable to get price information for {symbol}")
                return self._create_default_response(symbol)
            
            # Calculate price (in USD)
            token_price_usd = float(mint_prices[mint_address])
            
            # Build response
            return {
                "symbol": symbol,
                "price": token_price_usd,  # Price is already in USD
                "last_price": token_price_usd,
                "base_volume": 0,  # Cannot get volume information via Mint API
                "quote_volume": 0,
                "time": int(time.time() * 1000),
                "amm_id": "",
                "lp_mint": "",
                "market_id": "",
                "liquidity": 0,  # Cannot get liquidity information via Mint API
            }
            
        except Exception as e:
            logger.error(f"Failed to get {symbol} price: {str(e)}")
            return self._create_default_response(symbol)
    
    async def get_ticker_price(self, symbol: str) -> Dict[str, Any]:
        """Get price information - automatically selects the most suitable method"""
        try:
            # Special handling for high-value tokens like BTC, ETH, directly use token address
            if symbol.upper() in ["BTC", "ETH", "SOL"]:
                logger.info(f"Using token address method to get {symbol} price")
                return await self.get_ticker_price_by_mint(symbol.upper())
            
            # Use pool method for other tokens
            logger.info(f"Using pool method to get {symbol} price")
            
            # Check if symbol is for a common pool
            if symbol in self.COMMON_POOLS:
                pool_id = self.COMMON_POOLS[symbol]
                return await self.get_ticker_price_by_id(pool_id)
            
            # Check if it might be a single token symbol
            if '-' not in symbol and not symbol.startswith("pool-"):
                # Try to get price using token address
                return await self.get_ticker_price_by_mint(symbol)
            
            # Check if pool ID was used directly
            if len(symbol) > 30:
                return await self.get_ticker_price_by_id(symbol)
            
            # If symbol cannot be parsed or pool ID found, return default response
            logger.warning(f"Cannot parse symbol or find pool ID: {symbol}")
            return self._create_default_response(symbol)
            
        except Exception as e:
            logger.error(f"Failed to get price information: {str(e)}")
            return self._create_default_response(symbol)
    
    async def get_ticker_price_by_id(self, pool_id: str) -> Dict[str, Any]:
        """Get trading pair price by pool ID"""
        logger.info(f"Getting Raydium price by ID: {pool_id}")
        try:
            # Use /pools/info/ids API endpoint to get pool information directly
            pools_data = self.get_pool_info_by_ids([pool_id])
            
            # Check response structure
            if not isinstance(pools_data, list) or len(pools_data) == 0 or pools_data[0] is None:
                logger.error(f"Unexpected Raydium pool response structure: {type(pools_data)}")
                return self._create_default_response(pool_id)
            
            pool_data = pools_data[0]
            
            # Make sure we have a dictionary
            if not isinstance(pool_data, dict):
                logger.error(f"Pool data is not a dictionary: {type(pool_data)}")
                return self._create_default_response(pool_id)
            
            # Build standardized symbol (BASE-QUOTE)
            symbol = ""
            if 'mintA' in pool_data and 'mintB' in pool_data and \
               isinstance(pool_data['mintA'], dict) and isinstance(pool_data['mintB'], dict) and \
               'symbol' in pool_data['mintA'] and 'symbol' in pool_data['mintB']:
                symbol = f"{pool_data['mintA']['symbol']}-{pool_data['mintB']['symbol']}"
            else:
                symbol = f"pool-{pool_id[:8]}"
            
            # Handle different field names and structures
            price = 0
            base_volume = 0
            liquidity = 0
            
            # Try different price field names
            for price_field in ['price', 'currentPrice', 'lastPrice']:
                if price_field in pool_data and pool_data[price_field] is not None:
                    try:
                        price = float(pool_data[price_field])
                        break
                    except (ValueError, TypeError):
                        continue
            
            # Try to get volume
            if 'day' in pool_data and isinstance(pool_data['day'], dict) and 'volume' in pool_data['day']:
                try:
                    base_volume = float(pool_data['day']['volume'])
                except (ValueError, TypeError):
                    pass
            else:
                for vol_field in ['volume24h', 'volume24H', 'volume']:
                    if vol_field in pool_data and pool_data[vol_field] is not None:
                        try:
                            base_volume = float(pool_data[vol_field])
                            break
                        except (ValueError, TypeError):
                            continue
            
            # Try to get liquidity
            for liq_field in ['liquidity', 'tvl']:
                if liq_field in pool_data and pool_data[liq_field] is not None:
                    try:
                        liquidity = float(pool_data[liq_field])
                        break
                    except (ValueError, TypeError):
                        continue
            
            # Safely get other fields
            amm_id = pool_data.get("id", pool_data.get("ammId", ""))
            lp_mint = ""
            if 'lpMint' in pool_data:
                if isinstance(pool_data['lpMint'], dict) and 'address' in pool_data['lpMint']:
                    lp_mint = pool_data['lpMint']['address']
                else:
                    lp_mint = str(pool_data['lpMint'])
            
            market_id = pool_data.get("marketId", "")
            
            return {
                "symbol": symbol,
                "price": price,
                "last_price": price,
                "base_volume": base_volume,
                "quote_volume": base_volume * price if price else 0,
                "time": int(time.time() * 1000),
                "amm_id": amm_id,
                "lp_mint": lp_mint,
                "market_id": market_id,
                "liquidity": liquidity,
            }
            
        except Exception as e:
            logger.error(f"Failed to get Raydium pool data for ID {pool_id}: {str(e)}")
            return self._create_default_response(pool_id)
    
    async def get_ticker_24h(self, symbol_or_id: str) -> Dict[str, Any]:
        """Get 24-hour price change statistics"""
        logger.info(f"Getting Raydium 24h data: {symbol_or_id}")
        try:
            # Check if symbol is for a common pool
            pool_id = None
            if symbol_or_id in self.COMMON_POOLS:
                pool_id = self.COMMON_POOLS[symbol_or_id]
            # Check if it might be a single token symbol
            elif '-' not in symbol_or_id and not symbol_or_id.startswith("pool-"):
                usdc_pair = f"{symbol_or_id}-USDC"
                if usdc_pair in self.COMMON_POOLS:
                    pool_id = self.COMMON_POOLS[usdc_pair]
            # Check if pool ID was used directly
            elif len(symbol_or_id) > 30:
                pool_id = symbol_or_id
            
            if not pool_id:
                logger.warning(f"Cannot find pool ID: {symbol_or_id}")
                return {
                    "symbol": symbol_or_id,
                    "price_change": 0,
                    "price_change_percent": 0,
                    "volume": 0,
                    "volume_change_percent": 0,
                    "liquidity": 0,
                    "last_price": 0,
                    "time": int(time.time() * 1000),
                }
            
            # Get pool data
            pools_data = self.get_pool_info_by_ids([pool_id])
            
            if not isinstance(pools_data, list) or len(pools_data) == 0:
                logger.error(f"Unexpected Raydium pool response structure: {type(pools_data)}")
                return {
                    "symbol": symbol_or_id,
                    "price_change": 0,
                    "price_change_percent": 0,
                    "volume": 0,
                    "volume_change_percent": 0,
                    "liquidity": 0,
                    "last_price": 0,
                    "time": int(time.time() * 1000),
                }
            
            pool_data = pools_data[0]
            
            # Extract required information from pool data
            symbol = symbol_or_id
            if 'mintA' in pool_data and 'mintB' in pool_data and \
               isinstance(pool_data['mintA'], dict) and isinstance(pool_data['mintB'], dict) and \
               'symbol' in pool_data['mintA'] and 'symbol' in pool_data['mintB']:
                symbol = f"{pool_data['mintA']['symbol']}-{pool_data['mintB']['symbol']}"
            
            # Get price and price change data
            price = float(pool_data.get("price", 0))
            price_change = 0
            price_change_percent = 0
            
            # Try to get price change data
            if 'day' in pool_data and isinstance(pool_data['day'], dict):
                day_data = pool_data['day']
                if 'priceMin' in day_data and 'priceMax' in day_data:
                    try:
                        price_min = float(day_data['priceMin'])
                        price_max = float(day_data['priceMax'])
                        # Use midpoint of range as approximate value
                        prev_price = (price_min + price_max) / 2
                        price_change = price - prev_price
                        if prev_price > 0:
                            price_change_percent = (price_change / prev_price) * 100
                    except (ValueError, TypeError):
                        pass
            
            # Get volume data
            volume = 0
            volume_change_percent = 0
            
            if 'day' in pool_data and isinstance(pool_data['day'], dict) and 'volume' in pool_data['day']:
                try:
                    volume = float(pool_data['day']['volume'])
                except (ValueError, TypeError):
                    pass
            elif 'volume24h' in pool_data:
                try:
                    volume = float(pool_data['volume24h'])
                except (ValueError, TypeError):
                    pass
            
            # Get liquidity data
            liquidity = 0
            for liq_field in ['liquidity', 'tvl']:
                if liq_field in pool_data and pool_data[liq_field] is not None:
                    try:
                        liquidity = float(pool_data[liq_field])
                        break
                    except (ValueError, TypeError):
                        continue
            
            return {
                "symbol": symbol,
                "price_change": price_change,
                "price_change_percent": price_change_percent,
                "volume": volume,
                "volume_change_percent": volume_change_percent,
                "liquidity": liquidity,
                "last_price": price,
                "time": int(time.time() * 1000),
            }
            
        except Exception as e:
            logger.error(f"Failed to get Raydium 24h data: {str(e)}")
            return {
                "symbol": symbol_or_id,
                "price_change": 0,
                "price_change_percent": 0,
                "volume": 0,
                "volume_change_percent": 0,
                "liquidity": 0,
                "last_price": 0,
                "time": int(time.time() * 1000),
            }
    
    async def get_klines(self, symbol: str, interval: str, limit: int = 500) -> List[List]:
        """Get K-line data"""
        logger.info(f"Getting Raydium K-line data: {symbol}, interval: {interval}, limit: {limit}")
        
        # Get current price
        current_data = await self.get_ticker_price(symbol)
        current_price = current_data["price"]
        
        # Generate simulated K-line data
        return self._generate_mock_klines(current_price, interval, limit, current_data.get("base_volume", 0))
    
    def _generate_mock_klines(self, current_price: float, interval: str, limit: int, volume24h: float = 0) -> List[List]:
        """Generate simulated K-line data"""
        current_time = int(time.time() * 1000)
        interval_seconds = self._parse_interval_to_seconds(interval)
        
        # Simulated structure: [timestamp, open, high, low, close, volume]
        klines = []
        for i in range(limit):
            timestamp = current_time - (limit - i - 1) * interval_seconds * 1000
            # Simple price simulation
            variation = 0.01 * (((i % 10) - 5) / 5.0)
            simulated_price = current_price * (1 + variation) if current_price > 0 else 1000
            
            # Estimate interval volume
            interval_volume = 0
            if volume24h > 0:
                if interval.endswith('m'):
                    interval_value = int(interval[:-1])
                    interval_volume = volume24h / (24 * 60 / interval_value)
                elif interval.endswith('h'):
                    interval_value = int(interval[:-1])
                    interval_volume = volume24h / (24 / interval_value)
                elif interval.endswith('d'):
                    interval_value = int(interval[:-1])
                    interval_volume = volume24h * interval_value
                else:
                    interval_volume = volume24h / 24  # Default to hourly
            
            kline = [
                timestamp,                   # Opening time
                simulated_price,             # Opening price
                simulated_price * 1.01,      # Highest price
                simulated_price * 0.99,      # Lowest price
                simulated_price,             # Closing price
                interval_volume,             # Trading volume
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

class GetTokenPriceTool(DexBaseTool):
    """Tool to get current token price from DEX"""
    name: str = "get_token_price"
    description: str = "Get the current price of a token pair from a decentralized exchange"
    parameters: dict = {
        "type": "object",
        "properties": {
            "symbol": {
                "type": "string",
                "description": "Trading pair symbol (e.g., 'ETH-USDC')"
            },
            "exchange": {
                "type": "string",
                "description": "Exchange name (e.g., 'uniswap', default is 'uniswap')",
                "enum": ["uniswap"]
            }
        },
        "required": ["symbol"]
    }
    
    # Provider instances for different exchanges
    _providers: Dict[str, PriceDataProvider] = {}
    
    def _get_provider(self, exchange: str) -> PriceDataProvider:
        """Get or create price data provider for the specified exchange"""
        exchange = exchange.lower()
        if exchange not in self._providers:
            if exchange == "uniswap":
                self._providers[exchange] = UniswapPriceProvider()
            else:
                raise ValueError(f"Unsupported exchange: {exchange}")
        return self._providers[exchange]
    
    async def execute(self, symbol: str, exchange: str = "uniswap") -> ToolResult:
        """Execute the tool"""
        try:
            provider = self._get_provider(exchange)
            result = await provider.get_ticker_price(symbol)
            return ToolResult(output=result)
        except Exception as e:
            logger.error(f"Failed to get token price: {e}")
            return ToolResult(error=f"Failed to get token price: {str(e)}")

class Get24hStatsTool(DexBaseTool):
    """Tool to get 24-hour price statistics from DEX"""
    name: str = "get_24h_stats"
    description: str = "Get 24-hour price change statistics for a token pair from a decentralized exchange"
    parameters: dict = {
        "type": "object",
        "properties": {
            "symbol": {
                "type": "string",
                "description": "Trading pair symbol (e.g., 'ETH-USDC')"
            },
            "exchange": {
                "type": "string",
                "description": "Exchange name (e.g., 'uniswap', default is 'uniswap')",
                "enum": ["uniswap"]
            }
        },
        "required": ["symbol"]
    }
    
    # Reuse provider instances from GetTokenPriceTool
    _providers = GetTokenPriceTool._providers
    
    def _get_provider(self, exchange: str) -> PriceDataProvider:
        """Get or create price data provider for the specified exchange"""
        exchange = exchange.lower()
        if exchange not in self._providers:
            if exchange == "uniswap":
                self._providers[exchange] = UniswapPriceProvider()
            else:
                raise ValueError(f"Unsupported exchange: {exchange}")
        return self._providers[exchange]
    
    async def execute(self, symbol: str, exchange: str = "uniswap") -> ToolResult:
        """Execute the tool"""
        try:
            provider = self._get_provider(exchange)
            result = await provider.get_ticker_24h(symbol)
            return ToolResult(output=result)
        except Exception as e:
            logger.error(f"Failed to get 24h stats: {e}")
            return ToolResult(error=f"Failed to get 24h stats: {str(e)}")

class GetKlineDataTool(DexBaseTool):
    """Tool to get k-line data from DEX"""
    name: str = "get_kline_data"
    description: str = "Get k-line (candlestick) data for a token pair from a decentralized exchange"
    parameters: dict = {
        "type": "object",
        "properties": {
            "symbol": {
                "type": "string",
                "description": "Trading pair symbol (e.g., 'ETH-USDC')"
            },
            "interval": {
                "type": "string",
                "description": "Time interval for k-line data (e.g., '1h', '1d')",
                "enum": ["1m", "5m", "15m", "30m", "1h", "4h", "1d", "1w"]
            },
            "limit": {
                "type": "integer",
                "description": "Number of k-line data points to return (default: 500)",
                "default": 500
            },
            "exchange": {
                "type": "string",
                "description": "Exchange name (e.g., 'uniswap', default is 'uniswap')",
                "enum": ["uniswap"]
            }
        },
        "required": ["symbol", "interval"]
    }
    
    # Reuse provider instances from GetTokenPriceTool
    _providers = GetTokenPriceTool._providers
    
    def _get_provider(self, exchange: str) -> PriceDataProvider:
        """Get or create price data provider for the specified exchange"""
        exchange = exchange.lower()
        if exchange not in self._providers:
            if exchange == "uniswap":
                self._providers[exchange] = UniswapPriceProvider()
            else:
                raise ValueError(f"Unsupported exchange: {exchange}")
        return self._providers[exchange]
    
    async def execute(self, symbol: str, interval: str, limit: int = 500, exchange: str = "uniswap") -> ToolResult:
        """Execute the tool"""
        try:
            provider = self._get_provider(exchange)
            result = await provider.get_klines(symbol, interval, limit)
            return ToolResult(output=result)
        except Exception as e:
            logger.error(f"Failed to get k-line data: {e}")
            return ToolResult(error=f"Failed to get k-line data: {str(e)}") 