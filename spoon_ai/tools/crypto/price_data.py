import json
import logging
import time
from typing import Dict, Any, List, Optional, Union

from web3 import Web3
from web3.middleware import geth_poa_middleware
from pydantic import Field, validator

from spoon_ai.tools.base import BaseTool, ToolResult
from spoon_ai.tools.crypto.base import DexBaseTool

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
        self.rpc_url = rpc_url or "https://eth-mainnet.g.alchemy.com/v2/demo"
        self.w3 = Web3(Web3.HTTPProvider(self.rpc_url))
        self.w3.middleware_onion.inject(geth_poa_middleware, layer=0)
        self.factory = self.w3.eth.contract(
            address=self.w3.to_checksum_address(UNISWAP_V3_FACTORY), 
            abi=UNISWAP_FACTORY_ABI
        )
    
    def _get_token_addresses(self, symbol: str) -> tuple:
        """Get token addresses from trading pair symbol"""
        tokens = symbol.split("-")
        if len(tokens) != 2:
            raise ValueError(f"Invalid trading pair format: {symbol}. Expected format: TOKEN0-TOKEN1")
            
        token0_symbol, token1_symbol = tokens
        token0_address = TOKEN_ADDRESSES.get(token0_symbol.upper())
        token1_address = TOKEN_ADDRESSES.get(token1_symbol.upper())
        
        if not token0_address or not token1_address:
            raise ValueError(f"Token address not found: {token0_symbol} or {token1_symbol}")
            
        return (self.w3.to_checksum_address(token0_address), 
                self.w3.to_checksum_address(token1_address))
    
    def _get_pool_address(self, token0: str, token1: str, fee: int = 3000) -> str:
        """Get Uniswap V3 pool address"""
        # Ensure token0 and token1 are sorted by address
        if int(token0, 16) > int(token1, 16):
            token0, token1 = token1, token0
            
        pool_address = self.factory.functions.getPool(token0, token1, fee).call()
        if pool_address == "0x0000000000000000000000000000000000000000":
            raise ValueError(f"Pool not found: {token0}-{token1} with fee {fee}")
        return pool_address
    
    def _get_token_decimals(self, token_address: str) -> int:
        """Get token decimals"""
        token_contract = self.w3.eth.contract(address=token_address, abi=ERC20_ABI)
        return token_contract.functions.decimals().call()
    
    def _calculate_price_from_sqrt_price_x96(self, sqrt_price_x96: int, decimals0: int, decimals1: int) -> float:
        """Calculate price from sqrtPriceX96"""
        # Convert sqrtPriceX96 to price
        price = (sqrt_price_x96 / 2**96) ** 2
        # Adjust for decimals
        adjusted_price = price * (10 ** (decimals0 - decimals1))
        return adjusted_price
    
    async def get_ticker_price(self, symbol: str) -> Dict[str, Any]:
        """Get trading pair price
        
        In Uniswap, symbol should be in the format "TOKEN0-TOKEN1",
        e.g., "ETH-USDC" represents the ETH/USDC trading pair
        """
        try:
            logger.info(f"Getting Uniswap price for: {symbol}")
            token0_address, token1_address = self._get_token_addresses(symbol)
            
            # Get pool address
            pool_address = self._get_pool_address(token0_address, token1_address)
            pool_contract = self.w3.eth.contract(address=pool_address, abi=UNISWAP_POOL_ABI)
            
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
                "timestamp": int(time.time())
            }
        except Exception as e:
            logger.error(f"Failed to get Uniswap price: {e}")
            # Return default data on error to prevent system crash
            return {
                "price": "0",
                "pair": symbol,
                "timestamp": int(time.time()),
                "error": str(e)
            }
            
    async def get_ticker_24h(self, symbol: str) -> Dict[str, Any]:
        """Get 24-hour price change statistics"""
        try:
            logger.info(f"Getting Uniswap 24h data for: {symbol}")
            token0_address, token1_address = self._get_token_addresses(symbol)
            
            # Get pool address
            pool_address = self._get_pool_address(token0_address, token1_address)
            pool_contract = self.w3.eth.contract(address=pool_address, abi=UNISWAP_POOL_ABI)
            
            # Get current and 24 hours ago prices
            secondsAgos = [0, 86400]  # now and 24 hours ago
            observations = pool_contract.functions.observe(secondsAgos).call()
            
            # Get current price
            slot0 = pool_contract.functions.slot0().call()
            sqrt_price_x96 = slot0[0]
            
            # Get token decimals
            decimals0 = self._get_token_decimals(token0_address)
            decimals1 = self._get_token_decimals(token1_address)
            
            # Calculate current price
            current_price = self._calculate_price_from_sqrt_price_x96(sqrt_price_x96, decimals0, decimals1)
            
            # Calculate price change (simplified calculation, should use tick cumulative values in practice)
            tick_now = observations[0][0]
            tick_24h_ago = observations[0][1]
            tick_diff = tick_now - tick_24h_ago
            
            # Estimate price change from tick difference
            price_24h_ago = current_price * (1.0001 ** -tick_diff)
            price_change = current_price - price_24h_ago
            price_change_percent = (price_change / price_24h_ago) * 100 if price_24h_ago != 0 else 0
            
            # Note: Volume data should be obtained through other means, such as Graph API or event logs
            # This is an estimated value
            estimated_volume = current_price * 1000000  # Simplified estimate
            
            return {
                "price": str(current_price),
                "volume": str(estimated_volume),
                "priceChange": str(price_change),
                "priceChangePercent": str(price_change_percent),
                "pair": symbol,
                "timestamp": int(time.time())
            }
        except Exception as e:
            logger.error(f"Failed to get Uniswap 24h data: {e}")
            # Return default data on error
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
        """Get K-line data
        
        Note: Uniswap contracts don't provide K-line data directly, this would typically
        need to be obtained from a Graph API or by collecting and processing historical event data.
        
        This method would need to integrate with a service like The Graph to get real K-line data.
        """
        logger.info(f"Getting Uniswap K-line data: {symbol}, interval: {interval}, limit: {limit}")
        logger.warning("Uniswap contracts don't provide K-line data directly, need to use Graph API or event logs")
        
        # Return empty list, actual implementation should integrate with Graph API
        return []

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