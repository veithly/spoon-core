from .base import DexBaseTool
from datetime import datetime, timedelta
import asyncio
import aiohttp
import json
from typing import ClassVar, Dict, Any, List, Optional, Union, TypedDict, Literal
from dataclasses import dataclass
import requests
from pydantic import Field
import os

@dataclass
class SwapStep:
    """Swap step details"""
    type: str
    from_token: str
    to_token: str
    rate: float
    fee_percent: float
    dex: Optional[str] = None
    gas_cost_eth: Optional[float] = None
    delay_days: Optional[float] = None
    protocol: Optional[str] = None

@dataclass
class SwapPath:
    """Swap path result"""
    from_token: str
    to_token: str
    input_amount: float
    output_amount: float
    exchange_rate: float
    route_type: str
    steps: List[SwapStep]
    net_output_amount: Optional[float] = None
    gas_cost_eth: Optional[float] = None
    gas_cost_usd: Optional[float] = None
    profit: Optional[float] = None
    profit_percent: Optional[float] = None
    annual_profit_percent: Optional[float] = None
    is_profitable: bool = False

class TokenInfo(TypedDict):
    """Token information structure"""
    address: str
    is_native: bool
    decimals: int
    protocol: Optional[str]
    protocol_address: Optional[str]
    staking_rate_function: Optional[str]
    unstaking_function: Optional[str]
    chain: Optional[str]
    unstaking_time: Optional[int]
    unstaking_fee: Optional[float]
    related_to: Optional[str]
    coingecko_id: Optional[str]

class LstArbitrageTool(DexBaseTool):
    name: str = "lst_arbitrage_tool"
    description: str = "Analyze arbitrage opportunities between different LSTs (Liquid Staking Tokens), including between ETH and LSTs, and between different LSTs"
    # Declare cache and state variables as Pydantic fields
    price_cache: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    staking_rate_cache: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    market_stats_cache: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    liquidity_cache: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    last_gas_price_check: Optional[float] = Field(default=None)
    current_gas_price: Optional[float] = Field(default=None)
    parameters: dict = {
        "type": "object",
        "properties": {
            "from_token": {
                "type": "string",
                "description": "Source token (ETH, stETH, rETH, cbETH, etc.), if not specified, all possible combinations will be analyzed"
            },
            "to_token": {
                "type": "string",
                "description": "Target token (ETH, stETH, rETH, cbETH, etc.), if not specified, all possible combinations will be analyzed"
            },
            "amount": {
                "type": "number",
                "description": "Token amount to analyze, default is 1.0"
            },
            "find_all_routes": {
                "type": "boolean",
                "description": "Whether to find all possible arbitrage paths, default is false"
            },
            "risk_preference": {
                "type": "number",
                "description": "Risk preference coefficient, determines the profit threshold when selecting arbitrage paths, default is 5.0 (percentage)"
            }
        }
    }
    
    # Use variable GraphQL queries, unified style
    graph_template: str = ""  # Not using global template
    
    # Query LST price GraphQL - New EVM API
    lst_price_query: str = """
query ($baseAddress: String!, $quoteAddress: String!) {
  EVM(network: eth) {
    DEXTrades(
      limit: {count: 10}
      orderBy: {descending: Block_Time}
      where: {Trade: {Buy: {Currency: {SmartContract: {is: $baseAddress}}}, Sell: {Currency: {SmartContract: {is: $quoteAddress}}}}}
    ) {
      Block {
        Number
        Time
      }
      Trade {
        Buy {
          Amount
          Currency {
            Symbol
          }
        }
        Sell {
          Amount
          Currency {
            Symbol
          }
        }
        Transaction {
          Hash
        }
        Dex {
          ProtocolName
        }
        USD: Buy {
          Amount(in: USD)
        }
      }
    }
  }
}
"""

    # Query LST staking rate GraphQL - New EVM API
    lst_staking_rate_query: str = """
query ($protocolAddress: String!) {
  EVM(network: eth) {
    SmartContractCalls(
      limit: {count: 1}
      orderBy: {descending: Block_Time}
      where: {SmartContractCall: {To: {is: $protocolAddress}}}
    ) {
      Arguments {
        Name
        Value
      }
      Block {
        Number
        Time
      }
      Transaction {
        Hash
      }
    }
  }
}
"""
    gas_price_query: str = """
    query {
    EVM(network: eth) {
        Blocks(
        limit: {count: 1}
        orderBy: {descending: Block_Time}
        ) {
        Block {
            Number
            Time
            BaseFee
        }
        }
    }
    }
    """

    # Query market statistics GraphQL - New EVM API
    market_stats_query: str = """
query ($tokenAddress: String!) {
  EVM(network: eth) {
    DEXTrades(
      limit: {count: 30}
      orderBy: {descending: Block_Time}
      where: {Trade: {Buy: {Currency: {SmartContract: {is: $tokenAddress}}}, Sell: {Currency: {SmartContract: {is: "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2"}}}}}
    ) {
      Block {
        Time
      }
      Trade {
        Buy {
          Amount
        }
        Sell {
          Amount
        }
        USD: Buy {
          Amount(in: USD)
        }
      }
    }
  }
}
"""

    # Query DEX liquidity information GraphQL - New EVM API
    liquidity_query: str = """
query ($baseAddress: String!, $quoteAddress: String!) {
  EVM(network: eth) {
    DEXTrades(
      limit: {count: 1}
      where: {Trade: {Buy: {Currency: {SmartContract: {is: $baseAddress}}}, Sell: {Currency: {SmartContract: {is: $quoteAddress}}}}}
    ) {
      Dex {
        ProtocolName
      }
      Count: count
      Trade {
        USD: Buy {
          Amount(in: USD)
        }
      }
      UniqueAddresses: count(uniq: Trade_Sender)
    }
  }
}
"""

    # Supported token list
    supported_tokens: ClassVar[Dict[str, TokenInfo]] = {
        "ETH": {
            "address": "0xEeeeeEeeeEeEeeEeEeEeeEEEeeeeEeeeeeeeEEeE",  # Special address for ETH
            "is_native": True,
            "decimals": 18,
            "coingecko_id": "ethereum"
        },
        "WETH": {
            "address": "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2",
            "is_native": False,
            "decimals": 18,
            "coingecko_id": "weth"
        },
        "stETH": {
            "address": "0xae7ab96520DE3A18E5e111B5EaAb095312D7fE84",
            "protocol": "Lido",
            "protocol_address": "0xae7ab96520DE3A18E5e111B5EaAb095312D7fE84",
            "staking_rate_function": "getPooledEthByShares",
            "unstaking_function": "withdraw",
            "chain": "ethereum",
            "unstaking_time": 7 * 24 * 60 * 60,  # 7 days
            "unstaking_fee": 0.001,  # 0.1%
            "is_native": False,
            "decimals": 18,
            "coingecko_id": "staked-ether"
        },
        "rETH": {
            "address": "0xae78736Cd615f374D3085123A210448E74Fc6393",
            "protocol": "Rocket Pool",
            "protocol_address": "0xae78736Cd615f374D3085123A210448E74Fc6393",
            "staking_rate_function": "getExchangeRate",
            "unstaking_function": "burn",
            "chain": "ethereum",
            "unstaking_time": 24 * 60 * 60,  # 1 day
            "unstaking_fee": 0.0005,  # 0.05%
            "is_native": False,
            "decimals": 18,
            "coingecko_id": "rocket-pool-eth"
        },
        "cbETH": {
            "address": "0xBe9895146f7AF43049ca1c1AE358B0541Ea49704",
            "protocol": "Coinbase",
            "protocol_address": "0xBe9895146f7AF43049ca1c1AE358B0541Ea49704",
            "staking_rate_function": "exchangeRate",
            "unstaking_function": "redeem",
            "chain": "ethereum",
            "unstaking_time": 3 * 24 * 60 * 60,  # 3 days
            "unstaking_fee": 0.0025,  # 0.25%
            "is_native": False,
            "decimals": 18,
            "coingecko_id": "coinbase-wrapped-staked-eth"
        },
        "sfrxETH": {
            "address": "0xac3E018457B222d93114458476f3E3416Abbe38F",
            "protocol": "Frax",
            "protocol_address": "0xac3E018457B222d93114458476f3E3416Abbe38F",
            "staking_rate_function": "pricePerShare",
            "unstaking_function": "redeem",
            "chain": "ethereum",
            "unstaking_time": 14 * 24 * 60 * 60,  # 14 days
            "unstaking_fee": 0.002,  # 0.2%
            "is_native": False,
            "decimals": 18,
            "coingecko_id": "staked-frax-ether"
        },
        "wstETH": {
            "address": "0x7f39C581F595B53c5cb19bD0b3f8dA6c935E2Ca0",
            "protocol": "Lido (wrapped)",
            "protocol_address": "0x7f39C581F595B53c5cb19bD0b3f8dA6c935E2Ca0",
            "staking_rate_function": "stEthPerToken",
            "related_to": "stETH",
            "chain": "ethereum",
            "unstaking_time": 7 * 24 * 60 * 60,  # Same as stETH
            "unstaking_fee": 0.001,  # Same as stETH
            "is_native": False,
            "decimals": 18,
            "coingecko_id": "wrapped-steth"
        }
    }
    
    # Trading path configuration
    trade_paths: ClassVar[Dict[str, Dict[str, Dict[str, Any]]]] = {
        # ETH to LST paths
        "ETH_TO_LST": {
            "stETH": {
                "protocol": "Lido",
                "contract": "0xae7ab96520DE3A18E5e111B5EaAb095312D7fE84",
                "function": "submit",
                "fee": 0.001  # 0.1%
            },
            "rETH": {
                "protocol": "Rocket Pool",
                "contract": "0xae78736Cd615f374D3085123A210448E74Fc6393",
                "function": "deposit",
                "fee": 0.0005  # 0.05%
            },
            "cbETH": {
                "protocol": "Coinbase",
                "contract": "0xBe9895146f7AF43049ca1c1AE358B0541Ea49704",
                "function": "mint",
                "fee": 0.0025  # 0.25%
            }
        },
        # LST to ETH paths
        "LST_TO_ETH": {
            "stETH": {
                "protocol": "Lido",
                "contract": "0xae7ab96520DE3A18E5e111B5EaAb095312D7fE84",
                "function": "withdraw",
                "fee": 0.001,  # 0.1%
                "delay": 7 * 24 * 60 * 60  # 7 days
            },
            "rETH": {
                "protocol": "Rocket Pool",
                "contract": "0xae78736Cd615f374D3085123A210448E74Fc6393",
                "function": "burn",
                "fee": 0.0005,  # 0.05%
                "delay": 24 * 60 * 60  # 1 day
            },
            "cbETH": {
                "protocol": "Coinbase",
                "contract": "0xBe9895146f7AF43049ca1c1AE358B0541Ea49704",
                "function": "redeem",
                "fee": 0.0025,  # 0.25%
                "delay": 3 * 24 * 60 * 60  # 3 days
            }
        }
    }
    
    # Configuration parameters
    config: Dict[str, Any] = Field(default={
        "risk_preference": 5.0,  # Default risk preference coefficient, 5%
        "time_value_factor": 0.1,  # Time value coefficient, 0.1% per day
        "cache_ttl": 300,  # Cache time-to-live in seconds
        "default_slippage": 0.001,  # Default base slippage
        "max_slippage": 0.02,  # Maximum slippage limit
        "daily_yield": 0.0001,  # Default daily yield for opportunity cost calculation
        "daily_volatility": 0.02,  # Default daily volatility for opportunity cost calculation
        "min_profit_threshold": 0.0001,  # Minimum valid profit threshold
    })
    
    # Initialize configuration
    def __init__(self, **data):
        # First call super().__init__ to initialize Pydantic model
        super().__init__(**data)
        
        # fetch eth price
        def get_eth_price():
            # get eth price from coingecko
            try:
                response = requests.get("https://api.coingecko.com/api/v3/simple/price?ids=ethereum&vs_currencies=usd")
                return response.json()["ethereum"]["usd"]
            except Exception as e:
                return 3000

        # Extend config with additional values if needed
        self.config.setdefault("eth_price_usd", get_eth_price())
        
        # Override default configuration
        if "config" in data and isinstance(data["config"], dict):
            for key, value in data["config"].items():
                self.config[key] = value  # Update all keys, not just existing ones

    async def execute_query(self, query: str, variables: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Execute GraphQL query and handle response (async version)
        
        Args:
            query: GraphQL query string
            variables: Query variables
            
        Returns:
            dict: Query result
        """
        # Build GraphQL request
        if variables:
            payload = {"query": query, "variables": variables}
        else:
            payload = {"query": query}
        
        # Use synchronous method to get authentication headers
        try:
            headers = self.oAuth()
            
            # Use synchronous request (simplify problem location)
            response = requests.post(self.bitquery_endpoint, json=payload, headers=headers)
            if response.status_code != 200:
                raise Exception(f"GraphQL API error: {response.status_code}, {response.text}")
            
            return response.json()
        
        except Exception as e:
            raise Exception(f"GraphQL query execution error: {str(e)}")

    async def get_current_gas_price(self) -> float:
        """Get current gas price, return float value in Gwei unit"""
        # Check cache
        if hasattr(self, 'last_gas_price_check') and hasattr(self, 'current_gas_price'):
            if self.last_gas_price_check and self.current_gas_price:
                # If checked within last minute, use cached value
                if datetime.now().timestamp() - self.last_gas_price_check < 60:
                    return self.current_gas_price
        
        # Default gas price (in Gwei)
        default_gas_price = 20.0
        
        # Define simplified query
        gas_price_query = """
        query {
        EVM(network: eth) {
            Transactions(
            limit: {count: 10}
            orderBy: {descending: Block_Time}
            ) {
            Transaction {
                Gas
                GasPrice
            }
            }
        }
        }
        """
        
        # Query latest transaction gas prices
        try:
            print("Executing GraphQL query to get gas price...")
            
            response = await self.execute_query(gas_price_query)
            
            # Check response integrity
            if 'data' not in response or not response['data']:
                raise Exception("Missing valid data in API response")
            
            # Parse data
            evm_data = response['data'].get('EVM')
            if not evm_data:
                raise Exception("Missing 'EVM' field in API response")
                
            transactions = evm_data.get('Transactions')
            if not transactions or len(transactions) == 0:
                raise Exception("No transaction data in API response")
            
            # Collect gas price information from all transactions
            gas_prices_gwei = []
            
            print("Parsing transaction gas prices:")
            for tx_data in transactions:
                tx = tx_data.get('Transaction')
                if not tx:
                    continue
                    
                # Get GasPrice
                if 'GasPrice' in tx and tx['GasPrice']:
                    try:
                        # Parse string directly to float
                        gas_price_eth = float(tx['GasPrice'])
                        
                        # Convert to Gwei (1 ETH = 10^9 Gwei)
                        gas_price_gwei = gas_price_eth * 1e9
                        
                        # Print debug info
                        print(f"  Transaction GasPrice: {tx['GasPrice']} ETH = {gas_price_gwei:.2f} Gwei")
                        
                        gas_prices_gwei.append(gas_price_gwei)
                    except (ValueError, TypeError) as e:
                        print(f"  Error parsing GasPrice: {str(e)}")
            
            # If no valid gas prices obtained
            if not gas_prices_gwei:
                raise Exception("Could not get valid gas prices from transactions")
            
            print(f"Collected {len(gas_prices_gwei)} valid gas price values")
                
            # Calculate average gas price (using median)
            gas_prices_gwei.sort()
            if len(gas_prices_gwei) % 2 == 0:
                mid1 = gas_prices_gwei[len(gas_prices_gwei)//2]
                mid2 = gas_prices_gwei[len(gas_prices_gwei)//2 - 1]
                median_gas_price_gwei = (mid1 + mid2) / 2
            else:
                median_gas_price_gwei = gas_prices_gwei[len(gas_prices_gwei)//2]
            
            # Also calculate mean for reference
            avg_gas_price_gwei = sum(gas_prices_gwei) / len(gas_prices_gwei)
            
            print(f"Gas price statistics: Mean={avg_gas_price_gwei:.2f} Gwei, Median={median_gas_price_gwei:.2f} Gwei")
            
            # Use median as base fee
            base_fee_gwei = median_gas_price_gwei
            
            # Validate reasonability (0.5 Gwei ~ 500 Gwei range)
            if base_fee_gwei < 0.5 or base_fee_gwei > 500.0:
                print(f"Calculated gas price {base_fee_gwei:.2f} Gwei is not in reasonable range, using default value")
                gas_price_gwei = default_gas_price
            else:
                gas_price_gwei = base_fee_gwei
                print(f"Using API returned gas price: {gas_price_gwei:.2f} Gwei")
                
            # Add 15% safety margin to ensure quick confirmation
            gas_price_gwei = gas_price_gwei * 1.15
            
            # Ensure not below minimum reasonable value
            gas_price_gwei = max(gas_price_gwei, 1.0)
            
            # Print final price used
            print(f"Final gas price: {gas_price_gwei:.2f} Gwei (including 15% safety margin)")
            
            # Update cache
            self.current_gas_price = gas_price_gwei
            self.last_gas_price_check = datetime.now().timestamp()
            
            return gas_price_gwei
            
        except Exception as e:
            print(f"Error getting gas price: {str(e)}")
            
            # If API call fails, use default value
            print("Using default gas price due to API error")
            
            # If not initialized, set cache properties
            if not hasattr(self, 'current_gas_price'):
                self.current_gas_price = default_gas_price
            if not hasattr(self, 'last_gas_price_check'):
                self.last_gas_price_check = datetime.now().timestamp()
            
            return default_gas_price
    
    async def get_market_statistics(self, token: str) -> Dict[str, Any]:
        """
        Get token market statistics (volatility, yield, etc.)
        
        Args:
            token: Token symbol
            
        Returns:
            dict: Market statistics data
        """
        # Check cache
        if token in self.market_stats_cache:
            cache_entry = self.market_stats_cache[token]
            # Check if cache is expired
            if datetime.now().timestamp() - cache_entry["timestamp"] < self.config["cache_ttl"]:
                return cache_entry["data"]
        
        token_info = self.supported_tokens.get(token)
        if not token_info:
            return {
                "daily_volatility": self.config["daily_volatility"],
                "daily_yield": self.config["daily_yield"],
                "source": "Default parameters"
            }
        
        token_address = token_info['address']
        # Handle ETH special case
        if token == "ETH":
            token_address = self.supported_tokens["WETH"]["address"]
        
        try:
            # Build query
            variables = {"tokenAddress": token_address}
            response = await self.execute_query(self.market_stats_query, variables)
            
            if 'data' in response and 'EVM' in response['data'] and 'DEXTrades' in response['data']['EVM']:
                trades = response['data']['EVM']['DEXTrades']
                if trades:
                    # Extract price and volatility data from trades
                    prices = []
                    for trade in trades:
                        if 'Trade' in trade and 'Buy' in trade['Trade'] and 'Sell' in trade['Trade']:
                            buy_amount = float(trade['Trade']['Buy']['Amount'])
                            sell_amount = float(trade['Trade']['Sell']['Amount'])
                            if sell_amount > 0:
                                price = buy_amount / sell_amount
                                prices.append(price)
                    
                    # Calculate volatility
                    if prices and len(prices) > 5:
                        avg_price = sum(prices) / len(prices)
                        squared_diffs = [(p - avg_price) ** 2 for p in prices]
                        variance = sum(squared_diffs) / len(squared_diffs)
                        volatility = (variance ** 0.5) / avg_price
                    else:
                        volatility = self.config["daily_volatility"]
                    
                    # Calculate yield (simplified)
                    if prices and len(prices) > 5:
                        first_prices = prices[-5:]
                        last_prices = prices[:5]
                        first_avg = sum(first_prices) / len(first_prices)
                        last_avg = sum(last_prices) / len(last_prices)
                        
                        if first_avg > 0:
                            daily_change = (last_avg - first_avg) / first_avg / len(trades)
                            daily_yield = max(0, daily_change)  # Only consider positive yield
                        else:
                            daily_yield = self.config["daily_yield"]
                    else:
                        daily_yield = self.config["daily_yield"]
                    
                    result = {
                        "token": token,
                        "daily_volatility": volatility,
                        "daily_yield": daily_yield,
                        "source": "Bitquery market data"
                    }
                    
                    # Update cache
                    self.market_stats_cache[token] = {
                        "data": result,
                        "timestamp": datetime.now().timestamp()
                    }
                    
                    return result
        
        except Exception as e:
            print(f"Failed to get market statistics: {str(e)}")
        
        # Return default values
        result = {
            "token": token,
            "daily_volatility": self.config["daily_volatility"],
            "daily_yield": self.config["daily_yield"],
            "source": "Default parameters"
        }
        
        # Update cache
        self.market_stats_cache[token] = {
            "data": result,
            "timestamp": datetime.now().timestamp()
        }
        
        return result
    
    async def get_token_liquidity(self, from_token: str, to_token: str) -> Dict[str, Any]:
        """
        Get token pair liquidity information
        
        Args:
            from_token: Source token symbol
            to_token: Target token symbol
            
        Returns:
            dict: Liquidity information
        """
        # Check cache
        cache_key = f"{from_token}_{to_token}"
        if hasattr(self, 'liquidity_cache') and cache_key in self.liquidity_cache:
            cache_entry = self.liquidity_cache[cache_key]
            # Check if cache is expired
            # Get cache TTL, use default if not exists
            cache_ttl = getattr(self, 'config', {}).get("cache_ttl", 300)  # Default 5 minutes
            if datetime.now().timestamp() - cache_entry["timestamp"] < cache_ttl:
                return cache_entry["data"]
        
        from_info = self.supported_tokens.get(from_token)
        to_info = self.supported_tokens.get(to_token)
        
        if not from_info or not to_info:
            default_result = {
                "from_token": from_token,
                "to_token": to_token,
                "liquidity_score": 500,  # Default medium liquidity
                "exchange": "Unknown DEX",
                "trade_count": 0,
                "trade_amount_usd": 0,
                "unique_addresses": 0,
                "source": "Default parameters (unsupported token)"
            }
            
            # Update cache
            self.liquidity_cache[cache_key] = {
                "data": default_result,
                "timestamp": datetime.now().timestamp()
            }
            
            return default_result
        
        from_address = from_info['address']
        to_address = to_info['address']
        
        # Handle ETH special case
        if from_token == "ETH":
            from_address = self.supported_tokens["WETH"]["address"]
        if to_token == "ETH":
            to_address = self.supported_tokens["WETH"]["address"]
        
        # Modify query again - remove USD field
        liquidity_query = """
        query ($baseAddress: String!, $quoteAddress: String!) {
        EVM(network: eth) {
            DEXTrades(
            limit: {count: 10}
            orderBy: {descending: Block_Time}
            where: {Trade: {Buy: {Currency: {SmartContract: {is: $baseAddress}}}, Sell: {Currency: {SmartContract: {is: $quoteAddress}}}}}
            ) {
            Block {
                Time
                Number
            }
            Trade {
                Buy {
                Amount
                Currency {
                    Symbol
                }
                }
                Sell {
                Amount
                Currency {
                    Symbol
                }
                }
                Dex {
                ProtocolName
                }
            }
            }
        }
        }
        """
        
        print(f"Querying liquidity data for {from_token}/{to_token}...")
        
        try:
            # Build query
            variables = {
                "baseAddress": from_address,
                "quoteAddress": to_address
            }
            
            response = await self.execute_query(liquidity_query, variables)
            
            # Debug print
            #print(f"Liquidity query API response: {json.dumps(response, indent=2)}")
            
            if response and 'data' in response and response['data'] and 'EVM' in response['data']:
                # Check if DEXTrades exists and is not empty
                dex_trades = response['data']['EVM'].get('DEXTrades', [])
                
                if dex_trades and len(dex_trades) > 0:
                    # Calculate trade statistics
                    trade_count = len(dex_trades)
                    unique_protocols = set()
                    total_volume = 0.0
                    
                    for trade in dex_trades:
                        # Extract exchange name
                        if ('Trade' in trade and 'Dex' in trade['Trade'] and 
                            'ProtocolName' in trade['Trade']['Dex']):
                            unique_protocols.add(trade['Trade']['Dex']['ProtocolName'])
                        
                        # Accumulate trading volume (in base token)
                        if ('Trade' in trade and 'Buy' in trade['Trade'] and 
                            'Amount' in trade['Trade']['Buy']):
                            try:
                                amount = float(trade['Trade']['Buy']['Amount'])
                                total_volume += amount
                            except (ValueError, TypeError):
                                pass
                    
                    # Get main exchange
                    exchange_name = "Unknown DEX"
                    if unique_protocols:
                        exchange_name = list(unique_protocols)[0]  # Use first protocol
                    
                    # Calculate liquidity score using trade count and volume
                    # Since we don't have USD value, multiply volume by 100 as weight
                    liquidity_score = min(1000, trade_count*50 + total_volume*100 + len(unique_protocols)*50)
                    
                    result = {
                        "from_token": from_token,
                        "to_token": to_token,
                        "liquidity_score": liquidity_score,
                        "exchange": exchange_name,
                        "trade_count": trade_count,
                        "trade_volume": total_volume,  # Trading volume in base token
                        "unique_protocols": len(unique_protocols),
                        "source": "Bitquery liquidity data"
                    }
                    
                    # Update cache
                    self.liquidity_cache[cache_key] = {
                        "data": result,
                        "timestamp": datetime.now().timestamp()
                    }
                    
                    return result
                else:
                    print(f"No DEX trade data found for {from_token}/{to_token}")
            else:
                error_msg = "API response structure not as expected"
                if 'errors' in response:
                    error_detail = str(response.get('errors', ''))
                    error_msg += f": {error_detail}"
                print(f"{error_msg}")
        
        except Exception as e:
            print(f"Failed to get liquidity data for {from_token}/{to_token}: {str(e)}")
        
        # Return default values
        default_liquidity = {
            "ETH_stETH": 900,
            "ETH_rETH": 700,
            "ETH_cbETH": 600,
            "stETH_rETH": 400,
            "stETH_cbETH": 300,
            "rETH_cbETH": 200
        }
        
        # Try to get from default mapping
        liquidity_score = default_liquidity.get(f"{from_token}_{to_token}", 
                        default_liquidity.get(f"{to_token}_{from_token}", 500))
        
        result = {
            "from_token": from_token,
            "to_token": to_token,
            "liquidity_score": liquidity_score,
            "exchange": "Unknown DEX",
            "trade_count": 0,
            "trade_volume": 0.0,
            "unique_protocols": 0,
            "source": "Default parameters (API query failed)"
        }
        
        # Update cache
        self.liquidity_cache[cache_key] = {
            "data": result,
            "timestamp": datetime.now().timestamp()
        }
        
        return result
    
    async def calculate_slippage(self, from_token: str, to_token: str, amount: float) -> float:
        """
        Calculate expected slippage based on token pair and trade volume
        
        Args:
            from_token: Source token
            to_token: Target token
            amount: Trade volume
            
        Returns:
            float: Expected slippage (decimal percentage)
        """
        # Get token pair liquidity data
        liquidity_data = await self.get_token_liquidity(from_token, to_token)
        liquidity_score = liquidity_data.get("liquidity_score", 500)
        
        # Base slippage
        base_slippage = self.config["default_slippage"]
        
        # Liquidity factor
        liquidity_factor = max(100, liquidity_score)
        
        # Volume impact
        volume_impact = min(self.config["max_slippage"], amount / liquidity_factor)
        
        # Adjust slippage for specific exchanges
        exchange = liquidity_data.get("exchange", "").lower()
        
        exchange_factors = {
            "uniswap": 1.0,
            "sushiswap": 1.2,
            "curve": 0.5,  # Curve usually has lower slippage
            "balancer": 0.8
        }
        
        exchange_factor = 1.0
        for key, factor in exchange_factors.items():
            if key in exchange:
                exchange_factor = factor
                break
        
        # Calculate final slippage
        slippage = (base_slippage + volume_impact) * exchange_factor
        
        # Ensure slippage is within reasonable range
        return min(self.config["max_slippage"], max(base_slippage, slippage))
    
    async def calculate_opportunity_cost(self, delay_days: float, token: str) -> float:
        """
        Calculate opportunity cost of holding tokens
        
        Args:
            delay_days: Number of days delayed
            token: Token symbol
            
        Returns:
            float: Opportunity cost (decimal percentage)
        """
        # Get market statistics
        market_data = await self.get_market_statistics(token)
        
        # Get daily volatility and yield from market data
        daily_volatility = market_data.get("daily_volatility", self.config["daily_volatility"])
        daily_yield = market_data.get("daily_yield", self.config["daily_yield"])
        
        # Volatility risk (longer holding time, higher risk)
        volatility_cost = delay_days * daily_volatility * 0.5
        
        # Yield loss (loss from not participating in other investment opportunities)
        yield_cost = delay_days * daily_yield
        
        # Total opportunity cost
        return volatility_cost + yield_cost
    
    async def calculate_gas_cost(self, operation_type: str) -> Dict[str, float]:
        """
        Calculate gas cost
        
        Args:
            operation_type: Operation type (swap, stake, unstake, etc.)
            
        Returns:
            dict: Gas cost information
        """
        # Approximate gas consumption for different operation types
        gas_limits = {
            "swap": 150000,      # Regular DEX swap
            "stake": 200000,     # Staking
            "unstake": 250000,   # Unstaking
            "approve": 50000,    # Token approval
        }
        
        gas_limit = gas_limits.get(operation_type, 200000)  # Default 200k gas
        
        # Get current gas price
        gas_price_gwei = await self.get_current_gas_price()
        
        # Calculate ETH cost
        gas_cost_eth = (gas_limit * gas_price_gwei * 10**-9)
        
        # Get ETH price (USD) - Use ETH/USDT pair (adapt to new EVM API)
        eth_price_query = """
        query {
          EVM(network: eth) {
            DEXTrades(
              limit: {count: 1}
              orderBy: {descending: Block_Time}
              where: {Trade: {Buy: {Currency: {SmartContract: {is: "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2"}}}, Sell: {Currency: {SmartContract: {is: "0xdAC17F958D2ee523a2206206994597C13D831ec7"}}}}}
            ) {
              Trade {
                Buy {
                  Amount
                }
                Sell {
                  Amount
                }
              }
            }
          }
        }
        """
        
        try:
            response = await self.execute_query(eth_price_query)
            
            # Handle new API structure
            if ('data' in response and 'EVM' in response['data'] and 
                'DEXTrades' in response['data']['EVM'] and response['data']['EVM']['DEXTrades']):
                
                trade = response['data']['EVM']['DEXTrades'][0]['Trade']
                eth_amount = float(trade['Buy']['Amount'])
                usdt_amount = float(trade['Sell']['Amount'])
                
                eth_price_usd = usdt_amount / eth_amount
            else:
                # Default ETH price (if API call fails)
                eth_price_usd = 2500.0
            
            # Calculate USD cost
            gas_cost_usd = gas_cost_eth * eth_price_usd
            
            return {
                "gas_limit": gas_limit,
                "gas_price_gwei": gas_price_gwei,
                "gas_cost_eth": gas_cost_eth,
                "gas_cost_usd": gas_cost_usd
            }
        except Exception as e:
            # Use default ETH price
            eth_price_usd = 2500.0
            gas_cost_usd = gas_cost_eth * eth_price_usd
            
            return {
                "gas_limit": gas_limit,
                "gas_price_gwei": gas_price_gwei,
                "gas_cost_eth": gas_cost_eth,
                "gas_cost_usd": gas_cost_usd,
                "error": str(e)
            }
    async def get_token_exchange_data(self, from_token: str, to_token: str) -> Dict[str, Any]:
        """Get exchange data between two tokens (via DEX trades)"""
        # Check cache
        cache_key = f"{from_token}_{to_token}"
        if hasattr(self, 'price_cache') and cache_key in self.price_cache:
            cache_entry = self.price_cache[cache_key]
            # Check if cache is expired
            # Get cache TTL, use default if not exists
            cache_ttl = getattr(self, 'config', {}).get("cache_ttl", 300)  # Default 5 minutes
            if datetime.now().timestamp() - cache_entry["timestamp"] < cache_ttl:
                return cache_entry["data"]
        
        from_info = self.supported_tokens.get(from_token)
        to_info = self.supported_tokens.get(to_token)
        
        if not from_info or not to_info:
            raise Exception(f"Unsupported token pair: {from_token}/{to_token}")
        
        # Special handling for ETH and WETH trading
        is_eth_weth_pair = (from_token == "ETH" and to_token == "WETH") or (from_token == "WETH" and to_token == "ETH")
        if is_eth_weth_pair:
            result = {
                "from_token": from_token,
                "to_token": to_token,
                "price": 1.0,  # ETH and WETH always 1:1
                "inverse_price": 1.0,
                "timestamp": datetime.now().isoformat(),
                "block_height": 0,
                "transaction_hash": "",
                "trade_amount_usd": 0.0,
                "exchange": "ETH/WETH Direct",
                "source": "Direct ETH/WETH conversion"
            }
            
            # Update cache
            self.price_cache[cache_key] = {
                "data": result,
                "timestamp": datetime.now().timestamp()
            }
            
            return result
        
        from_address = from_info['address']
        to_address = to_info['address']
        
        # Handle ETH special case
        if from_token == "ETH":
            from_address = self.supported_tokens["WETH"]["address"]
        if to_token == "ETH":
            to_address = self.supported_tokens["WETH"]["address"]
        
        # Debug info
        print(f"Querying token exchange data: {from_token}({from_address}) → {to_token}({to_address})")
        
        # Simplified query - use only basic fields
        query = """
        query ($baseAddress: String!, $quoteAddress: String!) {
        EVM(network: eth) {
            DEXTrades(
            limit: {count: 5}
            orderBy: {descending: Block_Time}
            where: {Trade: {Buy: {Currency: {SmartContract: {is: $baseAddress}}}, Sell: {Currency: {SmartContract: {is: $quoteAddress}}}}}
            ) {
            Block {
                Number
                Time
            }
            Trade {
                Buy {
                Amount
                Currency {
                    Symbol
                }
                }
                Sell {
                Amount
                Currency {
                    Symbol
                }
                }
                Dex {
                ProtocolName
                }
            }
            }
        }
        }
        """
        
        # Build query variables
        variables = {
            "baseAddress": from_address,
            "quoteAddress": to_address
        }
        
        try:
            # Execute query
            response = await self.execute_query(query, variables)
            
            # Parse response
            if 'data' in response and 'EVM' in response['data'] and 'DEXTrades' in response['data']['EVM']:
                trades = response['data']['EVM']['DEXTrades']
                
                # Check if there is trade data
                if not trades or len(trades) == 0:
                    print(f"No trade data found between {from_token} and {to_token}, using default values")
                    
                    # Use preset default values based on staking rates and market data for each LST
                    default_rates = {
                        # Default exchange rates from ETH to LSTs
                        "ETH_stETH": 1.01,    # 1 ETH = 1.01 stETH
                        "ETH_rETH": 0.96,     # 1 ETH = 0.96 rETH
                        "ETH_cbETH": 0.98,    # 1 ETH = 0.98 cbETH
                        "ETH_wstETH": 1.01,   # 1 ETH = 1.01 wstETH
                        "ETH_sfrxETH": 1.02,  # 1 ETH = 1.02 sfrxETH
                        
                        # Default exchange rates between LSTs
                        "stETH_rETH": 0.95,   # 1 stETH = 0.95 rETH
                        "stETH_cbETH": 0.97,  # 1 stETH = 0.97 cbETH
                        "stETH_wstETH": 1.0,  # 1 stETH = 1.0 wstETH
                        "stETH_sfrxETH": 1.01,# 1 stETH = 1.01 sfrxETH
                        
                        "rETH_stETH": 1.05,   # 1 rETH = 1.05 stETH
                        "rETH_cbETH": 1.02,   # 1 rETH = 1.02 cbETH
                        "rETH_wstETH": 1.05,  # 1 rETH = 1.05 wstETH
                        "rETH_sfrxETH": 1.06, # 1 rETH = 1.06 sfrxETH
                        
                        "cbETH_stETH": 1.03,  # 1 cbETH = 1.03 stETH
                        "cbETH_rETH": 0.98,   # 1 cbETH = 0.98 rETH
                        "cbETH_wstETH": 1.03, # 1 cbETH = 1.03 wstETH
                        "cbETH_sfrxETH": 1.04,# 1 cbETH = 1.04 sfrxETH
                        
                        "wstETH_stETH": 1.0,  # 1 wstETH = 1.0 stETH
                        "wstETH_rETH": 0.95,  # 1 wstETH = 0.95 rETH
                        "wstETH_cbETH": 0.97, # 1 wstETH = 0.97 cbETH
                        "wstETH_sfrxETH": 1.01,# 1 wstETH = 1.01 sfrxETH,
                    }
                    
                    key = f"{from_token}_{to_token}"
                    reverse_key = f"{to_token}_{from_token}"
                    
                    if key in default_rates:
                        price = default_rates[key]
                    elif reverse_key in default_rates:
                        price = 1.0 / default_rates[reverse_key]
                    else:
                        # If preset default value can't be found, use 1:1
                        price = 1.0
                    
                    result = {
                        "from_token": from_token,
                        "to_token": to_token,
                        "price": price,
                        "inverse_price": 1.0 / price,
                        "timestamp": datetime.now().isoformat(),
                        "block_height": 0,
                        "transaction_hash": "",
                        "trade_amount_usd": 0.0,
                        "exchange": "Default",
                        "source": "Default values (no API data)"
                    }
                    
                    # Update cache
                    self.price_cache[cache_key] = {
                        "data": result,
                        "timestamp": datetime.now().timestamp()
                    }
                    
                    return result
                
                # Has trade data, parse first one
                latest_trade = trades[0]
                
                if 'Trade' in latest_trade:
                    trade = latest_trade['Trade']
                    
                    # Ensure necessary fields exist
                    if ('Buy' not in trade or 'Amount' not in trade['Buy'] or
                        'Sell' not in trade or 'Amount' not in trade['Sell']):
                        raise Exception(f"Invalid trade data format: {trade}")
                    
                    # Extract amounts
                    buy_amount = float(trade['Buy']['Amount'])
                    sell_amount = float(trade['Sell']['Amount'])
                    
                    # Calculate price ratio
                    price = buy_amount / sell_amount
                    inverse_price = sell_amount / buy_amount
                    
                    # Extract other information
                    block_number = latest_trade['Block']['Number']
                    block_time = latest_trade['Block']['Time']
                    exchange_name = trade['Dex']['ProtocolName']
                    
                    # Build result object - no transaction hash field
                    result = {
                        "from_token": from_token,
                        "to_token": to_token,
                        "price": price,
                        "inverse_price": inverse_price,
                        "timestamp": block_time,
                        "block_height": block_number,
                        "transaction_hash": "",  # Not available from API
                        "trade_amount_usd": 0.0,  # USD amount data may not be available
                        "exchange": exchange_name,
                        "source": "Bitquery DEX data"
                    }
                    
                    # Update cache
                    self.price_cache[cache_key] = {
                        "data": result,
                        "timestamp": datetime.now().timestamp()
                    }
                    
                    return result
                else:
                    raise Exception(f"Invalid trade data structure: {latest_trade}")
            else:
                error_msg = "API response structure not as expected"
                if 'errors' in response:
                    error_detail = str(response.get('errors', ''))
                    error_msg += f": {error_detail}"
                print(f"{error_msg}, using default values")
                
                # Use default values
                default_price = 1.0
                if from_token == "ETH" and to_token == "stETH":
                    default_price = 1.01
                elif from_token == "stETH" and to_token == "ETH":
                    default_price = 0.99
                
                result = {
                    "from_token": from_token,
                    "to_token": to_token,
                    "price": default_price,
                    "inverse_price": 1.0 / default_price,
                    "timestamp": datetime.now().isoformat(),
                    "block_height": 0,
                    "transaction_hash": "",
                    "trade_amount_usd": 0.0,
                    "exchange": "Default",
                    "source": "Default values (API structure error)"
                }
                
                # Update cache
                self.price_cache[cache_key] = {
                    "data": result,
                    "timestamp": datetime.now().timestamp()
                }
                
                return result
        
        except Exception as e:
            print(f"Failed to get trade data: {str(e)}, using default values")
            
            # Use default values
            default_price = 1.0
            if from_token == "ETH" and to_token == "stETH":
                default_price = 1.01
            elif from_token == "stETH" and to_token == "ETH":
                default_price = 0.99
            
            result = {
                "from_token": from_token,
                "to_token": to_token,
                "price": default_price,
                "inverse_price": 1.0 / default_price,
                "timestamp": datetime.now().isoformat(),
                "block_height": 0,
                "transaction_hash": "",
                "trade_amount_usd": 0.0,
                "exchange": "Default",
                "source": f"Default values (Error: {str(e)})"
            }
            
            # Update cache
            self.price_cache[cache_key] = {
                "data": result,
                "timestamp": datetime.now().timestamp()
            }
            
            return result
    
    async def get_staking_rate(self, token: str) -> Dict[str, Any]:
        """
        Get LST token staking rate (ETH:LST)
        
        Args:
            token: LST token symbol
            
        Returns:
            dict: Dictionary containing staking rate information
        """
        # Check cache
        if token in self.staking_rate_cache:
            cache_entry = self.staking_rate_cache[token]
            # Check if cache is expired
            if datetime.now().timestamp() - cache_entry["timestamp"] < self.config["cache_ttl"]:
                return cache_entry["data"]
        
        token_info = self.supported_tokens.get(token)
        
        # If it's ETH or not an LST token, return 1:1 ratio
        if not token_info or token == "ETH" or token == "WETH" or 'protocol_address' not in token_info:
            result = {
                "token": token,
                "staking_rate": 1.0,
                "source": "Default value"
            }
            self.staking_rate_cache[token] = {
                "data": result,
                "timestamp": datetime.now().timestamp()
            }
            return result
        
        protocol_address = token_info['protocol_address']
        
        # Build query variables
        variables = {
            "protocolAddress": protocol_address
        }
        
        # Execute query
        response = await self.execute_query(self.lst_staking_rate_query, variables)
        
        # Parse response - adapt to new EVM API structure
        if 'data' in response and 'EVM' in response['data'] and 'SmartContractCalls' in response['data']['EVM']:
            contract_calls = response['data']['EVM']['SmartContractCalls']
            if not contract_calls:
                raise Exception(f"No staking rate data found for {token}")
                
            latest_call = contract_calls[0]
            
            # Try to extract staking rate
            staking_rate = None
            staking_rate_function = token_info.get('staking_rate_function')
            
            if 'Arguments' in latest_call:
                for arg in latest_call['Arguments']:
                    if arg['Name'] == staking_rate_function or 'rate' in arg['Name'].lower() or 'price' in arg['Name'].lower():
                        staking_rate = float(arg['Value'])
                        break
            
            if staking_rate is None:
                # If unable to extract staking rate from parameters, use default values
                if token == "stETH":
                    staking_rate = 1.03  # Assume 1 stETH ≈ 1.03 ETH
                elif token == "rETH":
                    staking_rate = 1.04  # Assume 1 rETH ≈ 1.04 ETH
                elif token == "cbETH":
                    staking_rate = 1.02  # Assume 1 cbETH ≈ 1.02 ETH
                else:
                    staking_rate = 1.0  # Default 1:1 ratio
            
            # Build result
            result = {
                "token": token,
                "staking_rate": staking_rate,
                "timestamp": latest_call['Block']['Time'],
                "block_height": latest_call['Block']['Number'],
                "transaction_hash": latest_call['Transaction']['Hash'],
                "source": "Bitquery contract data"
            }
            
            # Update cache
            self.staking_rate_cache[token] = {
                "data": result,
                "timestamp": datetime.now().timestamp()
            }
            
            return result
        else:
            # If API call fails, use default values
            default_rates = {
                "stETH": 1.03,
                "rETH": 1.04,
                "cbETH": 1.02,
                "wstETH": 1.05,
                "sfrxETH": 1.035
            }
            
            staking_rate = default_rates.get(token, 1.0)
            
            result = {
                "token": token,
                "staking_rate": staking_rate,
                "source": "Default values (API failed)"
            }
            
            # Update cache
            self.staking_rate_cache[token] = {
                "data": result,
                "timestamp": datetime.now().timestamp()
            }
            
            return result
    async def calculate_direct_swap(self, from_token: str, to_token: str, amount: float) -> SwapPath:
        """
        Calculate the result of directly swapping two tokens on DEX
        
        Args:
            from_token: Source token
            to_token: Target token
            amount: Source token amount
            
        Returns:
            SwapPath: Object containing swap result information
        """
        # Check if it's ETH/WETH swap
        is_eth_weth_swap = (from_token == "ETH" and to_token == "WETH") or (from_token == "WETH" and to_token == "ETH")
        
        # Get exchange rate
        exchange_data = await self.get_token_exchange_data(from_token, to_token)
        
        # Determine DEX and fees from trade data
        dex_fee = 0.003  # Default 0.3% (Uniswap V2/V3 standard)
        dex_name = exchange_data.get("exchange", "Unknown DEX")
        
        # Adjust fees based on DEX
        dex_fees = {
            "uniswap": 0.003,  # 0.3%
            "sushiswap": 0.003,  # 0.3%
            "curve": 0.0004,  # 0.04%
            "balancer": 0.002,  # 0.2%
            "dodo": 0.001  # 0.1%
        }
        
        # For ETH/WETH swaps, no DEX fee
        if is_eth_weth_swap:
            dex_fee = 0.0
            dex_name = "ETH/WETH Direct"
        else:
            for key, fee in dex_fees.items():
                if key.lower() in dex_name.lower():
                    dex_fee = fee
                    break
        
        # Get transaction gas cost
        # ETH/WETH swaps have lower gas cost
        if is_eth_weth_swap:
            # Use lower gas cost for wrap/unwrap
            gas_cost = await self.calculate_gas_cost("wrap_eth" if from_token == "ETH" else "unwrap_eth")
        else:
            gas_cost = await self.calculate_gas_cost("swap")
        
        # Use simple internal slippage calculation, avoid calling method that might have parameter issues
        # No slippage for ETH/WETH swaps
        if is_eth_weth_swap:
            slippage = 0.0
        else:
            # Use predefined slippage table
            token_pair = f"{from_token}_{to_token}"
            default_slippages = {
                "ETH_stETH": 0.001,   # 0.1%
                "ETH_rETH": 0.002,    # 0.2%
                "ETH_cbETH": 0.003,   # 0.3%
                "stETH_rETH": 0.005,  # 0.5%
                "stETH_cbETH": 0.005, # 0.5%
                "rETH_cbETH": 0.008   # 0.8%
            }
            
            # Try to get slippage directly, if not exists check reverse pair, finally use default
            slippage = default_slippages.get(token_pair, 
                    default_slippages.get(f"{to_token}_{from_token}", 0.005))
            
            # Adjust slippage based on volume
            # Higher volume, higher slippage
            volume_multiplier = 1.0
            if amount > 100:
                volume_multiplier = 1.5
            elif amount > 10:
                volume_multiplier = 1.2
            
            slippage = slippage * volume_multiplier
        
        # Calculate actual received token amount
        received_amount = amount * exchange_data['price'] * (1 - dex_fee) * (1 - slippage)
        
        # Calculate net profit (considering gas cost)
        eth_price_in_to_token = 1.0
        if to_token != "ETH":
            try:
                eth_to_token_data = await self.get_token_exchange_data("ETH", to_token)
                eth_price_in_to_token = eth_to_token_data.get("price", 1.0)
            except Exception:
                # If unable to get ETH to target token rate, use default value
                eth_price_in_to_token = 1.0
        
        gas_cost_in_to_token = gas_cost["gas_cost_eth"] * eth_price_in_to_token
        net_received_amount = received_amount - gas_cost_in_to_token
        
        swap_step = SwapStep(
            type="dex_swap" if not is_eth_weth_swap else "eth_wrap_unwrap",
            from_token=from_token,
            to_token=to_token,
            rate=exchange_data['price'],
            fee_percent=dex_fee * 100,
            dex=dex_name,
            gas_cost_eth=gas_cost["gas_cost_eth"]
        )
        
        return SwapPath(
            from_token=from_token,
            to_token=to_token,
            input_amount=amount,
            output_amount=received_amount,
            net_output_amount=net_received_amount,
            exchange_rate=exchange_data['price'],
            route_type="direct_swap",
            steps=[swap_step],
            gas_cost_eth=gas_cost["gas_cost_eth"],
            gas_cost_usd=gas_cost["gas_cost_usd"]
        )
    
    async def calculate_staking_unstaking(self, from_token: str, to_token: str, amount: float) -> SwapPath:
        """
        Calculate exchange result through staking and unstaking path
        
        Args:
            from_token: Source token
            to_token: Target token
            amount: Source token amount
            
        Returns:
            SwapPath: Object containing exchange result information
        """
        from_info = self.supported_tokens.get(from_token)
        to_info = self.supported_tokens.get(to_token)
        
        if not from_info or not to_info:
            raise Exception(f"Unsupported token pair: {from_token}/{to_token}")
        
        # ETH -> LST (staking)
        if from_token == "ETH" and to_token in self.trade_paths["ETH_TO_LST"]:
            path_info = self.trade_paths["ETH_TO_LST"][to_token]
            staking_fee = path_info.get("fee", 0.001)
            
            # Get staking rate
            staking_data = await self.get_staking_rate(to_token)
            staking_rate = 1.0 / staking_data["staking_rate"]  # ETH->LST ratio
            
            # Get gas cost
            gas_cost = await self.calculate_gas_cost("stake")
            
            received_amount = amount * staking_rate * (1 - staking_fee)
            
            # Calculate net profit (considering gas cost)
            eth_price_in_to_token = 1.0
            try:
                eth_to_token_data = await self.get_token_exchange_data("ETH", to_token)
                eth_price_in_to_token = eth_to_token_data.get("price", 1.0)
            except Exception:
                # If unable to get ETH to target token rate, use default value
                eth_price_in_to_token = 1.0
            
            gas_cost_in_to_token = gas_cost["gas_cost_eth"] * eth_price_in_to_token
            net_received_amount = received_amount - gas_cost_in_to_token
            
            swap_step = SwapStep(
                type="stake",
                from_token=from_token,
                to_token=to_token,
                rate=staking_rate,
                fee_percent=staking_fee * 100,
                protocol=path_info.get("protocol"),
                gas_cost_eth=gas_cost["gas_cost_eth"]
            )
            
            return SwapPath(
                from_token=from_token,
                to_token=to_token,
                input_amount=amount,
                output_amount=received_amount,
                net_output_amount=net_received_amount,
                exchange_rate=staking_rate,
                route_type="staking",
                steps=[swap_step],
                gas_cost_eth=gas_cost["gas_cost_eth"],
                gas_cost_usd=gas_cost["gas_cost_usd"]
            )
            
        # LST -> ETH (unstaking)
        elif from_token in self.trade_paths["LST_TO_ETH"] and to_token == "ETH":
            path_info = self.trade_paths["LST_TO_ETH"][from_token]
            unstaking_fee = path_info.get("fee", 0.001)
            unstaking_delay = path_info.get("delay", 7 * 24 * 60 * 60)  # Default 7 days
            
            # Get staking rate
            staking_data = await self.get_staking_rate(from_token)
            staking_rate = staking_data["staking_rate"]  # LST->ETH ratio
            
            # Get gas cost
            gas_cost = await self.calculate_gas_cost("unstake")
            
            received_amount = amount * staking_rate * (1 - unstaking_fee)
            
            # Calculate delay days
            delay_days = unstaking_delay / (24 * 60 * 60)
            
            # Calculate opportunity cost
            opportunity_cost_percent = await self.calculate_opportunity_cost(delay_days, from_token)
            
            # Calculate net profit (considering gas cost and time value)
            net_received_amount = received_amount * (1 - opportunity_cost_percent) - gas_cost["gas_cost_eth"]
            
            swap_step = SwapStep(
                type="unstake",
                from_token=from_token,
                to_token=to_token,
                rate=staking_rate,
                fee_percent=unstaking_fee * 100,
                delay_days=delay_days,
                protocol=path_info.get("protocol"),
                gas_cost_eth=gas_cost["gas_cost_eth"]
            )
            
            return SwapPath(
                from_token=from_token,
                to_token=to_token,
                input_amount=amount,
                output_amount=received_amount,
                net_output_amount=net_received_amount,
                exchange_rate=staking_rate,
                route_type="unstaking",
                steps=[swap_step],
                gas_cost_eth=gas_cost["gas_cost_eth"],
                gas_cost_usd=gas_cost["gas_cost_usd"]
            )
        
        # LST -> LST (needs to go through ETH)
        elif from_token in self.trade_paths["LST_TO_ETH"] and to_token in self.trade_paths["ETH_TO_LST"]:
            # Step 1: LST -> ETH
            unstake_path = self.trade_paths["LST_TO_ETH"][from_token]
            unstaking_fee = unstake_path.get("fee", 0.001)
            unstaking_delay = unstake_path.get("delay", 7 * 24 * 60 * 60)
            
            from_staking_data = await self.get_staking_rate(from_token)
            from_staking_rate = from_staking_data["staking_rate"]
            
            # Get gas cost for first step
            gas_cost_1 = await self.calculate_gas_cost("unstake")
            
            intermediate_amount = amount * from_staking_rate * (1 - unstaking_fee)
            
            # Step 2: ETH -> LST
            stake_path = self.trade_paths["ETH_TO_LST"][to_token]
            staking_fee = stake_path.get("fee", 0.001)
            
            to_staking_data = await self.get_staking_rate(to_token)
            to_staking_rate = 1.0 / to_staking_data["staking_rate"]
            
            # Get gas cost for second step
            gas_cost_2 = await self.calculate_gas_cost("stake")
            
            # Calculate delay days
            delay_days = unstaking_delay / (24 * 60 * 60)
            
            # Calculate opportunity cost
            opportunity_cost_percent = await self.calculate_opportunity_cost(delay_days, from_token)
            
            # Consider first step gas cost
            intermediate_amount = intermediate_amount - gas_cost_1["gas_cost_eth"]
            
            # Consider delay opportunity cost
            intermediate_amount_after_delay = intermediate_amount * (1 - opportunity_cost_percent)
            
            # Consider second step
            final_amount = intermediate_amount_after_delay * to_staking_rate * (1 - staking_fee)
            
            # Consider second step gas cost
            net_final_amount = final_amount - gas_cost_2["gas_cost_eth"]
            
            # Calculate total gas cost (ETH and USD)
            total_gas_cost_eth = gas_cost_1["gas_cost_eth"] + gas_cost_2["gas_cost_eth"]
            total_gas_cost_usd = gas_cost_1["gas_cost_usd"] + gas_cost_2["gas_cost_usd"]
            
            unstake_step = SwapStep(
                type="unstake",
                from_token=from_token,
                to_token="ETH",
                rate=from_staking_rate,
                fee_percent=unstaking_fee * 100,
                delay_days=delay_days,
                protocol=unstake_path.get("protocol"),
                gas_cost_eth=gas_cost_1["gas_cost_eth"]
            )
            
            stake_step = SwapStep(
                type="stake",
                from_token="ETH",
                to_token=to_token,
                rate=to_staking_rate,
                fee_percent=staking_fee * 100,
                protocol=stake_path.get("protocol"),
                gas_cost_eth=gas_cost_2["gas_cost_eth"]
            )
            
            return SwapPath(
                from_token=from_token,
                to_token=to_token,
                input_amount=amount,
                output_amount=final_amount,
                net_output_amount=net_final_amount,
                exchange_rate=(final_amount / amount),
                route_type="unstake_then_stake",
                steps=[unstake_step, stake_step],
                gas_cost_eth=total_gas_cost_eth,
                gas_cost_usd=total_gas_cost_usd
            )
        
        else:
            raise Exception(f"No available staking/unstaking path: {from_token} -> {to_token}")
    
    async def analyze_arbitrage_path(self, from_token: str, to_token: str, amount: float, risk_preference: float = None) -> Dict[str, Any]:
        """Analyze arbitrage paths between two tokens"""
        # Use provided risk preference or default value
        if risk_preference is None:
            risk_preference = self.config["risk_preference"]
        
        # Calculate direct swap path
        try:
            direct_swap = await self.calculate_direct_swap(from_token, to_token, amount)
            direct_swap_dict = direct_swap.__dict__
            has_direct_swap = True
        except Exception as e:
            direct_swap_dict = {
                "from_token": from_token,
                "to_token": to_token,
                "input_amount": amount,
                "error": f"Direct swap failed: {str(e)}"
            }
            has_direct_swap = False
        
        # Calculate staking/unstaking path
        try:
            staking_path = await self.calculate_staking_unstaking(from_token, to_token, amount)
            staking_path_dict = staking_path.__dict__
            has_staking_path = True
        except Exception as e:
            staking_path_dict = {
                "from_token": from_token,
                "to_token": to_token,
                "input_amount": amount,
                "error": f"Staking/unstaking path failed: {str(e)}"
            }
            has_staking_path = False
        
        # If both paths fail, return error
        if not has_direct_swap and not has_staking_path:
            return {
                "from_token": from_token,
                "to_token": to_token,
                "amount": amount,
                "error": "No available exchange paths"
            }
        
        # Calculate annualized yield
        results = []
        
        if has_direct_swap:
            # Use net profit (considering gas cost)
            direct_swap_profit = direct_swap.net_output_amount - amount
            direct_swap_profit_percent = (direct_swap_profit / amount) * 100
            
            # Direct swap is usually immediate, so annual yield equals trade yield
            direct_swap_dict["profit"] = direct_swap_profit
            direct_swap_dict["profit_percent"] = direct_swap_profit_percent
            direct_swap_dict["annual_profit_percent"] = direct_swap_profit_percent  # Immediate trade, no annualization needed
            direct_swap_dict["is_profitable"] = direct_swap_profit > self.config["min_profit_threshold"]
            
            # Convert SwapStep objects to dictionaries
            if "steps" in direct_swap_dict and isinstance(direct_swap_dict["steps"], list):
                direct_swap_dict["steps"] = [step.__dict__ for step in direct_swap_dict["steps"]]
            
            results.append(direct_swap_dict)
        
        if has_staking_path:
            # Use net profit (considering gas cost and time value)
            staking_profit = staking_path.net_output_amount - amount
            staking_profit_percent = (staking_profit / amount) * 100
            
            # For unstaking paths that require waiting, calculate annualized yield
            unstaking_time_days = 0
            for step in staking_path.steps:
                if step.type == "unstake" and step.delay_days:
                    unstaking_time_days = step.delay_days
                    break
            
            if unstaking_time_days > 0:
                annual_profit_percent = staking_profit_percent * (365 / unstaking_time_days)
            else:
                annual_profit_percent = staking_profit_percent  # If immediate trade, no annualization needed
            
            staking_path_dict["profit"] = staking_profit
            staking_path_dict["profit_percent"] = staking_profit_percent
            staking_path_dict["annual_profit_percent"] = annual_profit_percent
            staking_path_dict["unstaking_time_days"] = unstaking_time_days
            staking_path_dict["is_profitable"] = staking_profit > self.config["min_profit_threshold"]
            
            # Convert SwapStep objects to dictionaries
            if "steps" in staking_path_dict and isinstance(staking_path_dict["steps"], list):
                staking_path_dict["steps"] = [step.__dict__ for step in staking_path_dict["steps"]]
            
            results.append(staking_path_dict)
        
        # Find the best path
        profitable_results = [r for r in results if r.get('is_profitable', False)]
        
        if not profitable_results:
            # If no profitable paths, choose the one with minimal loss
            if results:
                best_path = max(results, key=lambda x: x.get('profit_percent', -float('inf')))
            else:
                return {
                    "from_token": from_token,
                    "to_token": to_token,
                    "amount": amount,
                    "error": "No available exchange paths"
                }
        else:
            # Find the best path
            if len(profitable_results) == 1:
                best_path = profitable_results[0]
            else:
                # Compare direct swap and staking/unstaking paths
                direct_paths = [r for r in profitable_results if r['route_type'] == 'direct_swap']
                staking_paths = [r for r in profitable_results if r['route_type'] in ['staking', 'unstaking', 'unstake_then_stake']]
                
                if direct_paths and staking_paths:
                    direct_path = direct_paths[0]
                    staking_path = max(staking_paths, key=lambda x: x['annual_profit_percent'])
                    
                    # Calculate time value adjustment
                    time_value_adjustment = staking_path.get('unstaking_time_days', 0) * self.config["time_value_factor"]
                    
                    # Consider risk preference and time value
                    if staking_path['annual_profit_percent'] > direct_path['annual_profit_percent'] + risk_preference + time_value_adjustment:
                        best_path = staking_path
                    else:
                        best_path = direct_path
                elif direct_paths:
                    best_path = direct_paths[0]
                else:
                    best_path = max(staking_paths, key=lambda x: x['annual_profit_percent'])
        
        # Add current timestamp and market conditions
        try:
            gas_price = await self.get_current_gas_price()
        except:
            gas_price = None
            
        current_time = datetime.now().isoformat()
        
        return {
            "from_token": from_token,
            "to_token": to_token,
            "amount": amount,
            "analysis_time": current_time,
            "current_gas_price_gwei": gas_price,
            "paths": results,
            "best_path": best_path
        }
    
    async def find_circular_arbitrage(self, start_token: str, amount: float) -> Dict[str, Any]:
        """Find circular arbitrage opportunities"""
        # This method's code remains unchanged as it calls other updated methods
        # Get all supported tokens
        tokens = list(self.supported_tokens.keys())
        
        # If start token not in list, return error
        if start_token not in tokens:
            return {
                "start_token": start_token,
                "error": "Unsupported token"
            }
        
        # Find all possible two-hop paths
        two_hop_paths = []
        for intermediate_token in tokens:
            if intermediate_token != start_token:
                try:
                    # Analyze start_token -> intermediate_token -> start_token path
                    first_hop = await self.calculate_direct_swap(start_token, intermediate_token, amount)
                    
                    # Use net output considering gas
                    intermediate_amount = first_hop.net_output_amount
                    second_hop = await self.calculate_direct_swap(intermediate_token, start_token, intermediate_amount)
                    
                    final_amount = second_hop.net_output_amount
                    profit = final_amount - amount
                    profit_percent = (profit / amount) * 100
                    
                    if profit > self.config["min_profit_threshold"]:
                        two_hop_paths.append({
                            "path": [start_token, intermediate_token, start_token],
                            "amounts": [amount, intermediate_amount, final_amount],
                            "profit": profit,
                            "profit_percent": profit_percent,
                            "gas_cost_eth": first_hop.gas_cost_eth + second_hop.gas_cost_eth,
                            "gas_cost_usd": first_hop.gas_cost_usd + second_hop.gas_cost_usd,
                            "steps": [
                                first_hop.steps[0].__dict__,
                                second_hop.steps[0].__dict__
                            ]
                        })
                except Exception as e:
                    # Log error but continue processing
                    print(f"Two-hop path analysis failed {start_token}->{intermediate_token}: {str(e)}")
        
        # Find potentially valuable intermediate tokens (based on two-hop results)
        potential_tokens = set()
        for path in two_hop_paths:
            intermediate_token = path["path"][1]
            potential_tokens.add(intermediate_token)
        
        # If no valuable intermediate tokens, use some major LST tokens
        if not potential_tokens:
            potential_tokens = set(["stETH", "rETH", "cbETH", "WETH"])
        
        # Find all possible three-hop paths
        three_hop_paths = []
        for intermediate1 in potential_tokens:
            if intermediate1 != start_token:
                for intermediate2 in potential_tokens:
                    if intermediate2 != start_token and intermediate2 != intermediate1:
                        try:
                            # Analyze start_token -> intermediate1 -> intermediate2 -> start_token path
                            first_hop = await self.calculate_direct_swap(start_token, intermediate1, amount)
                            
                            intermediate1_amount = first_hop.net_output_amount
                            second_hop = await self.calculate_direct_swap(intermediate1, intermediate2, intermediate1_amount)
                            
                            intermediate2_amount = second_hop.net_output_amount
                            third_hop = await self.calculate_direct_swap(intermediate2, start_token, intermediate2_amount)
                            
                            final_amount = third_hop.net_output_amount
                            profit = final_amount - amount
                            profit_percent = (profit / amount) * 100
                            
                            if profit > self.config["min_profit_threshold"]:
                                total_gas_cost_eth = (
                                    first_hop.gas_cost_eth + 
                                    second_hop.gas_cost_eth + 
                                    third_hop.gas_cost_eth
                                )
                                total_gas_cost_usd = (
                                    first_hop.gas_cost_usd + 
                                    second_hop.gas_cost_usd + 
                                    third_hop.gas_cost_usd
                                )
                                
                                three_hop_paths.append({
                                    "path": [start_token, intermediate1, intermediate2, start_token],
                                    "amounts": [amount, intermediate1_amount, intermediate2_amount, final_amount],
                                    "profit": profit,
                                    "profit_percent": profit_percent,
                                    "gas_cost_eth": total_gas_cost_eth,
                                    "gas_cost_usd": total_gas_cost_usd,
                                    "steps": [
                                        first_hop.steps[0].__dict__,
                                        second_hop.steps[0].__dict__,
                                        third_hop.steps[0].__dict__
                                    ]
                                })
                        except Exception as e:
                            # Log error but continue processing
                            print(f"Three-hop path analysis failed {start_token}->{intermediate1}->{intermediate2}: {str(e)}")
        
        # Merge all paths and sort by profit
        all_paths = sorted(two_hop_paths + three_hop_paths, key=lambda x: x['profit_percent'], reverse=True)
        
        # Add current timestamp and market conditions
        current_time = datetime.now().isoformat()
        try:
            gas_price = await self.get_current_gas_price()
        except:
            gas_price = None
        
        return {
            "start_token": start_token,
            "amount": amount,
            "analysis_time": current_time,
            "current_gas_price_gwei": gas_price,
            "circular_arbitrage_opportunities": all_paths,
            "best_opportunity": all_paths[0] if all_paths else None
        }
    
    async def execute(self, **params):
        """Execute LST arbitrage analysis"""
        # This method's code remains unchanged as it calls other updated methods
        from_token = params.get('from_token')
        to_token = params.get('to_token')
        amount = params.get('amount', 1.0)
        find_all_routes = params.get('find_all_routes', False)
        risk_preference = params.get('risk_preference', self.config["risk_preference"])
        
        # Ensure amount is a number
        try:
            amount = float(amount)
        except (ValueError, TypeError):
            amount = 1.0
        
        results = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "amount": amount,
            "risk_preference": risk_preference
        }
        
        try:
            # Get current gas price
            gas_price = await self.get_current_gas_price()
            results["current_gas_price_gwei"] = gas_price
        except Exception as e:
            # If getting gas price fails, should not affect entire analysis
            print(f"Failed to get gas price: {str(e)}")
        
        # If from_token and to_token specified, analyze specific path
        if from_token and to_token:
            try:
                arbitrage_analysis = await self.analyze_arbitrage_path(from_token, to_token, amount, risk_preference)
                results["arbitrage_analysis"] = arbitrage_analysis
            except Exception as e:
                results["arbitrage_analysis"] = {
                    "from_token": from_token,
                    "to_token": to_token,
                    "amount": amount,
                    "error": f"Analysis failed: {str(e)}"
                }
            
            # If requested to find circular arbitrage, analyze additionally
            if find_all_routes:
                try:
                    circular_analysis = await self.find_circular_arbitrage(from_token, amount)
                    results["circular_arbitrage"] = circular_analysis
                except Exception as e:
                    results["circular_arbitrage"] = {
                        "start_token": from_token,
                        "error": f"Circular arbitrage analysis failed: {str(e)}"
                    }
            
            return results
        
        # If only from_token specified, analyze paths from this token to all other tokens
        elif from_token:
            all_analyses = []
            tokens = [t for t in self.supported_tokens.keys() if t != from_token]
            
            for token in tokens:
                try:
                    arbitrage_analysis = await self.analyze_arbitrage_path(from_token, token, amount, risk_preference)
                    all_analyses.append(arbitrage_analysis)
                except Exception as e:
                    all_analyses.append({
                        "from_token": from_token,
                        "to_token": token,
                        "amount": amount,
                        "error": f"Analysis failed: {str(e)}"
                    })
            
            # Find best path
            profitable_analyses = [a for a in all_analyses if 'best_path' in a and a['best_path'].get('is_profitable', False)]
            
            if profitable_analyses:
                best_analysis = max(profitable_analyses, key=lambda x: x['best_path']['annual_profit_percent'])
            else:
                best_analysis = None
            
            # If requested to find circular arbitrage, analyze additionally
            if find_all_routes:
                try:
                    circular_analysis = await self.find_circular_arbitrage(from_token, amount)
                    results["circular_arbitrage"] = circular_analysis
                except Exception as e:
                    results["circular_arbitrage"] = {
                        "start_token": from_token,
                        "error": f"Circular arbitrage analysis failed: {str(e)}"
                    }
            
            results["from_token"] = from_token
            results["all_analyses"] = all_analyses
            results["best_analysis"] = best_analysis
            
            return results
        
        # If none specified, analyze best arbitrage opportunities among all possible combinations
        else:
            # Optimization: only analyze main token pairs to reduce API calls
            main_tokens = ["ETH", "stETH", "rETH", "cbETH"]
            all_pairs = []
            
            # Generate key token pairs
            for from_token in main_tokens:
                for to_token in main_tokens:
                    if from_token != to_token:
                        all_pairs.append((from_token, to_token))
            
            all_analyses = []
            
            for from_token, to_token in all_pairs:
                try:
                    arbitrage_analysis = await self.analyze_arbitrage_path(from_token, to_token, amount, risk_preference)
                    all_analyses.append(arbitrage_analysis)
                except Exception as e:
                    all_analyses.append({
                        "from_token": from_token,
                        "to_token": to_token,
                        "amount": amount,
                        "error": f"Analysis failed: {str(e)}"
                    })
            
            # Find best arbitrage opportunity
            profitable_analyses = [a for a in all_analyses if 'best_path' in a and a['best_path'].get('is_profitable', False)]
            
            if profitable_analyses:
                best_analysis = max(profitable_analyses, key=lambda x: x['best_path']['annual_profit_percent'])
            else:
                best_analysis = None
            
            # Find best circular arbitrage opportunity
            circular_results = []
            
            for token in main_tokens:
                try:
                    circular_analysis = await self.find_circular_arbitrage(token, amount)
                    if circular_analysis.get('best_opportunity'):
                        circular_results.append(circular_analysis)
                except Exception as e:
                    print(f"{token} circular arbitrage analysis failed: {str(e)}")
            
            best_circular = max(circular_results, key=lambda x: x['best_opportunity']['profit_percent']) if circular_results else None
            
            results["all_analyses"] = all_analyses
            results["best_direct_analysis"] = best_analysis
            results["best_circular_arbitrage"] = best_circular
            
            # Compare direct arbitrage and circular arbitrage
            if best_analysis and best_circular:
                direct_profit_percent = best_analysis['best_path']['annual_profit_percent']
                circular_profit_percent = best_circular['best_opportunity']['profit_percent']
                
                # Consider risk preference
                if circular_profit_percent > direct_profit_percent + risk_preference:
                    results["overall_best_strategy"] = {
                        "type": "circular",
                        "details": best_circular
                    }
                else:
                    results["overall_best_strategy"] = {
                        "type": "direct",
                        "details": best_analysis
                    }
            elif best_analysis:
                results["overall_best_strategy"] = {
                    "type": "direct",
                    "details": best_analysis
                }
            elif best_circular:
                results["overall_best_strategy"] = {
                    "type": "circular",
                    "details": best_circular
                }
            
            return results

    def get_cache_ttl(self):
        """Safely get cache TTL value"""
        default_ttl = 300  # Default 5 minutes
        if not hasattr(self, 'config') or not isinstance(self.config, dict):
            return default_ttl
        return self.config.get("cache_ttl", default_ttl)

    def update_config(self, new_config: Dict[str, Any]) -> None:
        """
        Update tool configuration parameters
        
        Args:
            new_config: Dictionary with new configuration parameters
        """
        if not hasattr(self, 'config'):
            self.config = {}
        
        # Update configuration
        for key, value in new_config.items():
            self.config[key] = value
            
        # Clear caches
        self.price_cache.clear()
        self.staking_rate_cache.clear()
        self.market_stats_cache.clear()
        self.liquidity_cache.clear()
        self.last_gas_price_check = None
        self.current_gas_price = None
        
        print(f"Configuration updated: {new_config}")