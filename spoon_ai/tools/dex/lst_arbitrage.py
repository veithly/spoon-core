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
    """交换步骤详情"""
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
    """交换路径结果"""
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
    """代币信息结构"""
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
    description: str = "分析不同LST（流动性质押代币）之间的套利机会，包括ETH与LST之间及LST与LST之间"
    # 声明缓存和状态变量为Pydantic字段
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
                "description": "起始代币(ETH, stETH, rETH, cbETH等)，如不指定则分析所有可能组合"
            },
            "to_token": {
                "type": "string",
                "description": "目标代币(ETH, stETH, rETH, cbETH等)，如不指定则分析所有可能组合"
            },
            "amount": {
                "type": "number",
                "description": "要分析的代币数量，默认为1.0"
            },
            "find_all_routes": {
                "type": "boolean",
                "description": "是否查找所有可能的套利路径，默认为false"
            },
            "risk_preference": {
                "type": "number",
                "description": "风险偏好系数，决定选择套利路径时的收益阈值，默认为5.0(百分比)"
            }
        }
    }
    
    # 使用变量化的GraphQL查询，统一风格
    graph_template: str = ""  # 不使用全局模板
    
    # 查询LST价格的GraphQL查询 - 新版 EVM API
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

    # 查询LST质押率的GraphQL查询 - 新版 EVM API
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

    # 查询市场统计数据的GraphQL查询 - 新版 EVM API
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

    # 查询DEX流动性信息的GraphQL查询 - 新版 EVM API
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

    # 支持的代币列表
    supported_tokens: ClassVar[Dict[str, TokenInfo]] = {
        "ETH": {
            "address": "0xEeeeeEeeeEeEeeEeEeEeeEEEeeeeEeeeeeeeEEeE",  # 使用特殊地址表示ETH
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
            "unstaking_time": 7 * 24 * 60 * 60,  # 7天
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
            "unstaking_time": 24 * 60 * 60,  # 1天
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
            "unstaking_time": 3 * 24 * 60 * 60,  # 3天
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
            "unstaking_time": 14 * 24 * 60 * 60,  # 14天
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
            "unstaking_time": 7 * 24 * 60 * 60,  # 同stETH
            "unstaking_fee": 0.001,  # 同stETH
            "is_native": False,
            "decimals": 18,
            "coingecko_id": "wrapped-steth"
        }
    }
    
    # 交易路径配置
    trade_paths: ClassVar[Dict[str, Dict[str, Dict[str, Any]]]] = {
        # ETH到LST的路径
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
        # LST到ETH的路径
        "LST_TO_ETH": {
            "stETH": {
                "protocol": "Lido",
                "contract": "0xae7ab96520DE3A18E5e111B5EaAb095312D7fE84",
                "function": "withdraw",
                "fee": 0.001,  # 0.1%
                "delay": 7 * 24 * 60 * 60  # 7天
            },
            "rETH": {
                "protocol": "Rocket Pool",
                "contract": "0xae78736Cd615f374D3085123A210448E74Fc6393",
                "function": "burn",
                "fee": 0.0005,  # 0.05%
                "delay": 24 * 60 * 60  # 1天
            },
            "cbETH": {
                "protocol": "Coinbase",
                "contract": "0xBe9895146f7AF43049ca1c1AE358B0541Ea49704",
                "function": "redeem",
                "fee": 0.0025,  # 0.25%
                "delay": 3 * 24 * 60 * 60  # 3天
            }
        }
    }
    
    # 配置参数
    config: Dict[str, Any] = Field(default={
        "risk_preference": 5.0,  # 默认风险偏好系数，5%
        "time_value_factor": 0.1,  # 时间价值系数，每天0.1%
        "cache_ttl": 300,  # 缓存有效期，秒
        "default_slippage": 0.001,  # 默认基础滑点
        "max_slippage": 0.02,  # 最大滑点限制
        "daily_yield": 0.0001,  # 默认日收益率，用于机会成本计算
        "daily_volatility": 0.02,  # 默认日波动率，用于机会成本计算
        "min_profit_threshold": 0.0001,  # 最小有效利润阈值
    })
    
    # 初始化配置
    def __init__(self, **data):
        # 设置默认配置
        self.config : {
            "cache_ttl": 300,  # 缓存生存时间，默认5分钟(300秒)
            "min_profit_threshold": 0.0,  # 最小利润阈值
            "eth_price_usd": 3000,  # ETH/USD汇率
            # 其他默认配置...
        }
        
        super().__init__(**data)
        
        # 覆盖默认配置
        if "config" in data and isinstance(data["config"], dict):
            for key, value in data["config"].items():
                self.config[key] = value  # 更新所有键，不只是已存在的

    async def execute_query(self, query: str, variables: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        执行GraphQL查询并处理响应（异步版本）
        
        Args:
            query: GraphQL查询字符串
            variables: 查询变量
            
        Returns:
            dict: 查询结果
        """
        # 构建GraphQL请求
        if variables:
            payload = {"query": query, "variables": variables}
        else:
            payload = {"query": query}
        
        # 使用同步方法获取认证头
        try:
            headers = self.oAuth()
            
            # 直接使用同步请求（简化问题定位）
            response = requests.post(self.bitquery_endpoint, json=payload, headers=headers)
            if response.status_code != 200:
                raise Exception(f"GraphQL API错误: {response.status_code}, {response.text}")
            
            return response.json()
        
        except Exception as e:
            raise Exception(f"GraphQL查询执行异常: {str(e)}")
    async def get_current_gas_price(self) -> float:
        """获取当前gas价格，返回Gwei单位的浮点数"""
        # 检查缓存
        if hasattr(self, 'last_gas_price_check') and hasattr(self, 'current_gas_price'):
            if self.last_gas_price_check and self.current_gas_price:
                # 如果在过去1分钟内检查过，直接使用缓存的值
                if datetime.now().timestamp() - self.last_gas_price_check < 60:
                    return self.current_gas_price
        
        # 默认gas价格（Gwei单位）
        default_gas_price = 20.0
        
        # 定义简化的查询
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
        
        # 查询最新交易的gas价格
        try:
            print("正在执行 GraphQL 查询获取 gas 价格...")
            
            response = await self.execute_query(gas_price_query)
            
            # 检查响应完整性
            if 'data' not in response or not response['data']:
                raise Exception("API 响应中缺少有效数据")
            
            # 解析数据
            evm_data = response['data'].get('EVM')
            if not evm_data:
                raise Exception("API 响应中缺少 'EVM' 字段")
                
            transactions = evm_data.get('Transactions')
            if not transactions or len(transactions) == 0:
                raise Exception("API 响应中没有交易数据")
            
            # 收集所有交易的gas价格信息
            gas_prices_gwei = []
            
            print("解析交易Gas价格:")
            for tx_data in transactions:
                tx = tx_data.get('Transaction')
                if not tx:
                    continue
                    
                # 获取GasPrice
                if 'GasPrice' in tx and tx['GasPrice']:
                    try:
                        # 直接将字符串解析为浮点数
                        gas_price_eth = float(tx['GasPrice'])
                        
                        # 转换为Gwei (1 ETH = 10^9 Gwei)
                        gas_price_gwei = gas_price_eth * 1e9
                        
                        # 输出调试信息
                        print(f"  交易GasPrice: {tx['GasPrice']} ETH = {gas_price_gwei:.2f} Gwei")
                        
                        gas_prices_gwei.append(gas_price_gwei)
                    except (ValueError, TypeError) as e:
                        print(f"  解析GasPrice出错: {str(e)}")
            
            # 如果没有获取到任何有效的gas价格
            if not gas_prices_gwei:
                raise Exception("未能从交易中获取有效的gas价格")
            
            print(f"收集到 {len(gas_prices_gwei)} 个有效Gas价格值")
                
            # 计算平均gas价格(使用中位数)
            gas_prices_gwei.sort()
            if len(gas_prices_gwei) % 2 == 0:
                mid1 = gas_prices_gwei[len(gas_prices_gwei)//2]
                mid2 = gas_prices_gwei[len(gas_prices_gwei)//2 - 1]
                median_gas_price_gwei = (mid1 + mid2) / 2
            else:
                median_gas_price_gwei = gas_prices_gwei[len(gas_prices_gwei)//2]
            
            # 也计算平均值作为参考
            avg_gas_price_gwei = sum(gas_prices_gwei) / len(gas_prices_gwei)
            
            print(f"Gas价格统计: 平均值={avg_gas_price_gwei:.2f} Gwei, 中位数={median_gas_price_gwei:.2f} Gwei")
            
            # 使用中位数作为基础费用
            base_fee_gwei = median_gas_price_gwei
            
            # 验证合理性 (0.5 Gwei ~ 500 Gwei范围)
            if base_fee_gwei < 0.5 or base_fee_gwei > 500.0:
                print(f"计算的Gas价格 {base_fee_gwei:.2f} Gwei 不在合理范围内，使用默认值")
                gas_price_gwei = default_gas_price
            else:
                gas_price_gwei = base_fee_gwei
                print(f"使用API返回的Gas价格: {gas_price_gwei:.2f} Gwei")
                
            # 增加15%的安全边际，确保交易能快速被确认
            gas_price_gwei = gas_price_gwei * 1.15
            
            # 确保不低于最小合理值
            gas_price_gwei = max(gas_price_gwei, 1.0)
            
            # 打印最终使用的价格
            print(f"最终Gas价格: {gas_price_gwei:.2f} Gwei (含15%安全边际)")
            
            # 更新缓存
            self.current_gas_price = gas_price_gwei
            self.last_gas_price_check = datetime.now().timestamp()
            
            return gas_price_gwei
            
        except Exception as e:
            print(f"获取gas价格错误: {str(e)}")
            
            # 如果API调用失败，使用默认值
            print("由于API错误，使用默认gas价格")
            
            # 如果未初始化，设置缓存属性
            if not hasattr(self, 'current_gas_price'):
                self.current_gas_price = default_gas_price
            if not hasattr(self, 'last_gas_price_check'):
                self.last_gas_price_check = datetime.now().timestamp()
            
            return default_gas_price
    
    async def get_market_statistics(self, token: str) -> Dict[str, Any]:
        """
        获取代币的市场统计数据(波动性、收益率等)
        
        Args:
            token: 代币符号
            
        Returns:
            dict: 市场统计数据
        """
        # 检查缓存
        if token in self.market_stats_cache:
            cache_entry = self.market_stats_cache[token]
            # 检查缓存是否过期
            if datetime.now().timestamp() - cache_entry["timestamp"] < self.config["cache_ttl"]:
                return cache_entry["data"]
        
        token_info = self.supported_tokens.get(token)
        if not token_info:
            return {
                "daily_volatility": self.config["daily_volatility"],
                "daily_yield": self.config["daily_yield"],
                "source": "默认参数"
            }
        
        token_address = token_info['address']
        # 处理ETH特殊情况
        if token == "ETH":
            token_address = self.supported_tokens["WETH"]["address"]
        
        try:
            # 构建查询
            variables = {"tokenAddress": token_address}
            response = await self.execute_query(self.market_stats_query, variables)
            
            if 'data' in response and 'EVM' in response['data'] and 'DEXTrades' in response['data']['EVM']:
                trades = response['data']['EVM']['DEXTrades']
                if trades:
                    # 从交易中提取价格和波动性数据
                    prices = []
                    for trade in trades:
                        if 'Trade' in trade and 'Buy' in trade['Trade'] and 'Sell' in trade['Trade']:
                            buy_amount = float(trade['Trade']['Buy']['Amount'])
                            sell_amount = float(trade['Trade']['Sell']['Amount'])
                            if sell_amount > 0:
                                price = buy_amount / sell_amount
                                prices.append(price)
                    
                    # 计算波动性
                    if prices and len(prices) > 5:
                        avg_price = sum(prices) / len(prices)
                        squared_diffs = [(p - avg_price) ** 2 for p in prices]
                        variance = sum(squared_diffs) / len(squared_diffs)
                        volatility = (variance ** 0.5) / avg_price
                    else:
                        volatility = self.config["daily_volatility"]
                    
                    # 计算收益率（简化）
                    if prices and len(prices) > 5:
                        first_prices = prices[-5:]
                        last_prices = prices[:5]
                        first_avg = sum(first_prices) / len(first_prices)
                        last_avg = sum(last_prices) / len(last_prices)
                        
                        if first_avg > 0:
                            daily_change = (last_avg - first_avg) / first_avg / len(trades)
                            daily_yield = max(0, daily_change)  # 只考虑正收益
                        else:
                            daily_yield = self.config["daily_yield"]
                    else:
                        daily_yield = self.config["daily_yield"]
                    
                    result = {
                        "token": token,
                        "daily_volatility": volatility,
                        "daily_yield": daily_yield,
                        "source": "Bitquery市场数据"
                    }
                    
                    # 更新缓存
                    self.market_stats_cache[token] = {
                        "data": result,
                        "timestamp": datetime.now().timestamp()
                    }
                    
                    return result
        
        except Exception as e:
            print(f"获取市场统计数据失败: {str(e)}")
        
        # 返回默认值
        result = {
            "token": token,
            "daily_volatility": self.config["daily_volatility"],
            "daily_yield": self.config["daily_yield"],
            "source": "默认参数"
        }
        
        # 更新缓存
        self.market_stats_cache[token] = {
            "data": result,
            "timestamp": datetime.now().timestamp()
        }
        
        return result
    
    async def get_token_liquidity(self, from_token: str, to_token: str) -> Dict[str, Any]:
        """
        获取代币对的流动性信息
        
        Args:
            from_token: 源代币符号
            to_token: 目标代币符号
            
        Returns:
            dict: 流动性信息
        """
        # 检查缓存
        cache_key = f"{from_token}_{to_token}"
        if hasattr(self, 'liquidity_cache') and cache_key in self.liquidity_cache:
            cache_entry = self.liquidity_cache[cache_key]
            # 检查缓存是否过期
            # 获取缓存TTL，如果不存在则使用默认值
            cache_ttl = getattr(self, 'config', {}).get("cache_ttl", 300)  # 默认5分钟
            if datetime.now().timestamp() - cache_entry["timestamp"] < cache_ttl:
                return cache_entry["data"]
        
        from_info = self.supported_tokens.get(from_token)
        to_info = self.supported_tokens.get(to_token)
        
        if not from_info or not to_info:
            default_result = {
                "from_token": from_token,
                "to_token": to_token,
                "liquidity_score": 500,  # 默认中等流动性
                "exchange": "Unknown DEX",
                "trade_count": 0,
                "trade_amount_usd": 0,
                "unique_addresses": 0,
                "source": "默认参数(不支持的代币)"
            }
            
            # 更新缓存
            self.liquidity_cache[cache_key] = {
                "data": default_result,
                "timestamp": datetime.now().timestamp()
            }
            
            return default_result
        
        from_address = from_info['address']
        to_address = to_info['address']
        
        # 处理ETH特殊情况
        if from_token == "ETH":
            from_address = self.supported_tokens["WETH"]["address"]
        if to_token == "ETH":
            to_address = self.supported_tokens["WETH"]["address"]
        
        # 再次修正查询 - 移除USD字段
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
        
        print(f"查询{from_token}/{to_token}的流动性数据...")
        
        try:
            # 构建查询
            variables = {
                "baseAddress": from_address,
                "quoteAddress": to_address
            }
            
            response = await self.execute_query(liquidity_query, variables)
            
            # 调试打印
            #print(f"流动性查询API响应: {json.dumps(response, indent=2)}")
            
            if response and 'data' in response and response['data'] and 'EVM' in response['data']:
                # 检查DEXTrades是否存在且不为空
                dex_trades = response['data']['EVM'].get('DEXTrades', [])
                
                if dex_trades and len(dex_trades) > 0:
                    # 计算交易统计信息
                    trade_count = len(dex_trades)
                    unique_protocols = set()
                    total_volume = 0.0
                    
                    for trade in dex_trades:
                        # 提取交易所名称
                        if ('Trade' in trade and 'Dex' in trade['Trade'] and 
                            'ProtocolName' in trade['Trade']['Dex']):
                            unique_protocols.add(trade['Trade']['Dex']['ProtocolName'])
                        
                        # 累计交易量（以base token计）
                        if ('Trade' in trade and 'Buy' in trade['Trade'] and 
                            'Amount' in trade['Trade']['Buy']):
                            try:
                                amount = float(trade['Trade']['Buy']['Amount'])
                                total_volume += amount
                            except (ValueError, TypeError):
                                pass
                    
                    # 获取主要交易所
                    exchange_name = "Unknown DEX"
                    if unique_protocols:
                        exchange_name = list(unique_protocols)[0]  # 使用第一个协议
                    
                    # 使用交易数量和交易量计算流动性评分
                    # 由于没有USD值，我们将交易量乘以100作为权重
                    liquidity_score = min(1000, trade_count*50 + total_volume*100 + len(unique_protocols)*50)
                    
                    result = {
                        "from_token": from_token,
                        "to_token": to_token,
                        "liquidity_score": liquidity_score,
                        "exchange": exchange_name,
                        "trade_count": trade_count,
                        "trade_volume": total_volume,  # 以base token计的交易量
                        "unique_protocols": len(unique_protocols),
                        "source": "Bitquery流动性数据"
                    }
                    
                    # 更新缓存
                    self.liquidity_cache[cache_key] = {
                        "data": result,
                        "timestamp": datetime.now().timestamp()
                    }
                    
                    return result
                else:
                    print(f"未找到{from_token}/{to_token}的DEX交易数据")
            else:
                error_msg = "API响应结构不符合预期"
                if 'errors' in response:
                    error_detail = str(response.get('errors', ''))
                    error_msg += f": {error_detail}"
                print(f"{error_msg}")
        
        except Exception as e:
            print(f"获取{from_token}/{to_token}流动性数据失败: {str(e)}")
        
        # 返回默认值
        default_liquidity = {
            "ETH_stETH": 900,
            "ETH_rETH": 700,
            "ETH_cbETH": 600,
            "stETH_rETH": 400,
            "stETH_cbETH": 300,
            "rETH_cbETH": 200
        }
        
        # 尝试从默认映射获取
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
            "source": "默认参数(API查询失败)"
        }
        
        # 更新缓存
        self.liquidity_cache[cache_key] = {
            "data": result,
            "timestamp": datetime.now().timestamp()
        }
        
        return result
    
    async def calculate_slippage(self, from_token: str, to_token: str, amount: float) -> float:
        """
        根据代币对和交易量计算预期滑点
        
        Args:
            from_token: 源代币
            to_token: 目标代币
            amount: 交易量
            
        Returns:
            float: 预期滑点(百分比小数)
        """
        # 获取代币对流动性数据
        liquidity_data = await self.get_token_liquidity(from_token, to_token)
        liquidity_score = liquidity_data.get("liquidity_score", 500)
        
        # 基础滑点
        base_slippage = self.config["default_slippage"]
        
        # 流动性因子
        liquidity_factor = max(100, liquidity_score)
        
        # 交易量影响
        volume_impact = min(self.config["max_slippage"], amount / liquidity_factor)
        
        # 对于特定的交易所调整滑点
        exchange = liquidity_data.get("exchange", "").lower()
        
        exchange_factors = {
            "uniswap": 1.0,
            "sushiswap": 1.2,
            "curve": 0.5,  # Curve通常滑点较小
            "balancer": 0.8
        }
        
        exchange_factor = 1.0
        for key, factor in exchange_factors.items():
            if key in exchange:
                exchange_factor = factor
                break
        
        # 计算最终滑点
        slippage = (base_slippage + volume_impact) * exchange_factor
        
        # 确保滑点在合理范围内
        return min(self.config["max_slippage"], max(base_slippage, slippage))
    
    async def calculate_opportunity_cost(self, delay_days: float, token: str) -> float:
        """
        计算持有代币的机会成本
        
        Args:
            delay_days: 延迟天数
            token: 代币符号
            
        Returns:
            float: 机会成本(百分比小数)
        """
        # 获取市场统计数据
        market_data = await self.get_market_statistics(token)
        
        # 从市场数据中获取日波动性和收益率
        daily_volatility = market_data.get("daily_volatility", self.config["daily_volatility"])
        daily_yield = market_data.get("daily_yield", self.config["daily_yield"])
        
        # 波动性风险（持有时间越长，风险越大）
        volatility_cost = delay_days * daily_volatility * 0.5
        
        # 收益损失（无法参与其他投资机会的损失）
        yield_cost = delay_days * daily_yield
        
        # 总机会成本
        return volatility_cost + yield_cost
    
    async def calculate_gas_cost(self, operation_type: str) -> Dict[str, float]:
        """
        计算gas成本
        
        Args:
            operation_type: 操作类型（swap, stake, unstake等）
            
        Returns:
            dict: gas成本信息
        """
        # 不同操作类型的大致gas消耗
        gas_limits = {
            "swap": 150000,      # 普通DEX交换
            "stake": 200000,     # 质押
            "unstake": 250000,   # 解质押
            "approve": 50000,    # 代币授权
        }
        
        gas_limit = gas_limits.get(operation_type, 200000)  # 默认20万gas
        
        # 获取当前gas价格
        gas_price_gwei = await self.get_current_gas_price()
        
        # 计算ETH成本
        gas_cost_eth = (gas_limit * gas_price_gwei * 10**-9)
        
        # 获取ETH价格（美元）- 使用ETH/USDT对 (适配新EVM API)
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
            
            # 处理新的API结构
            if ('data' in response and 'EVM' in response['data'] and 
                'DEXTrades' in response['data']['EVM'] and response['data']['EVM']['DEXTrades']):
                
                trade = response['data']['EVM']['DEXTrades'][0]['Trade']
                eth_amount = float(trade['Buy']['Amount'])
                usdt_amount = float(trade['Sell']['Amount'])
                
                eth_price_usd = usdt_amount / eth_amount
            else:
                # 默认ETH价格（如果API调用失败）
                eth_price_usd = 2500.0
            
            # 计算美元成本
            gas_cost_usd = gas_cost_eth * eth_price_usd
            
            return {
                "gas_limit": gas_limit,
                "gas_price_gwei": gas_price_gwei,
                "gas_cost_eth": gas_cost_eth,
                "gas_cost_usd": gas_cost_usd
            }
        except Exception as e:
            # 使用默认ETH价格
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
        """获取两个代币之间的兑换数据(通过DEX交易)"""
        # 检查缓存
        cache_key = f"{from_token}_{to_token}"
        if hasattr(self, 'price_cache') and cache_key in self.price_cache:
            cache_entry = self.price_cache[cache_key]
            # 检查缓存是否过期
            # 获取缓存TTL，如果不存在则使用默认值
            cache_ttl = getattr(self, 'config', {}).get("cache_ttl", 300)  # 默认5分钟
            if datetime.now().timestamp() - cache_entry["timestamp"] < cache_ttl:
                return cache_entry["data"]
        
        from_info = self.supported_tokens.get(from_token)
        to_info = self.supported_tokens.get(to_token)
        
        if not from_info or not to_info:
            raise Exception(f"不支持的代币对: {from_token}/{to_token}")
        
        from_address = from_info['address']
        to_address = to_info['address']
        
        # 处理ETH特殊情况
        if from_token == "ETH":
            from_address = self.supported_tokens["WETH"]["address"]
        if to_token == "ETH":
            to_address = self.supported_tokens["WETH"]["address"]
        
        # 调试信息
        print(f"查询代币交换数据: {from_token}({from_address}) → {to_token}({to_address})")
        
        # 简化的查询 - 只使用基本字段
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
        
        # 构建查询变量
        variables = {
            "baseAddress": from_address,
            "quoteAddress": to_address
        }
        
        try:
            # 执行查询
            response = await self.execute_query(query, variables)
            
            # 打印完整响应以便调试
            #print(f"API响应: {json.dumps(response, indent=2)}")
            
            # 解析响应
            if 'data' in response and 'EVM' in response['data'] and 'DEXTrades' in response['data']['EVM']:
                trades = response['data']['EVM']['DEXTrades']
                
                # 检查是否有交易数据
                if not trades or len(trades) == 0:
                    print(f"未找到{from_token}和{to_token}之间的交易数据，使用默认值")
                    
                    # 使用预设默认值
                    default_rates = {
                        "ETH_stETH": 1.01,    # 1 ETH = 1.01 stETH
                        "ETH_rETH": 0.96,     # 1 ETH = 0.96 rETH
                        "ETH_cbETH": 0.98,    # 1 ETH = 0.98 cbETH
                        "stETH_rETH": 0.95,   # 1 stETH = 0.95 rETH
                        "stETH_cbETH": 0.97,  # 1 stETH = 0.97 cbETH
                        "rETH_cbETH": 1.02    # 1 rETH = 1.02 cbETH
                    }
                    
                    key = f"{from_token}_{to_token}"
                    reverse_key = f"{to_token}_{from_token}"
                    
                    if key in default_rates:
                        price = default_rates[key]
                    elif reverse_key in default_rates:
                        price = 1.0 / default_rates[reverse_key]
                    else:
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
                        "source": "默认值(无API数据)"
                    }
                    
                    # 更新缓存
                    self.price_cache[cache_key] = {
                        "data": result,
                        "timestamp": datetime.now().timestamp()
                    }
                    
                    return result
                
                # 有交易数据，解析第一条
                latest_trade = trades[0]
                
                if 'Trade' in latest_trade:
                    trade = latest_trade['Trade']
                    
                    # 确保必要的字段存在
                    if ('Buy' not in trade or 'Amount' not in trade['Buy'] or
                        'Sell' not in trade or 'Amount' not in trade['Sell']):
                        raise Exception(f"交易数据格式不正确: {trade}")
                    
                    # 提取金额
                    buy_amount = float(trade['Buy']['Amount'])
                    sell_amount = float(trade['Sell']['Amount'])
                    
                    # 计算价格比率
                    price = buy_amount / sell_amount
                    inverse_price = sell_amount / buy_amount
                    
                    # 提取其他信息
                    block_number = latest_trade['Block']['Number']
                    block_time = latest_trade['Block']['Time']
                    exchange_name = trade['Dex']['ProtocolName']
                    
                    # 构建结果对象 - 没有交易哈希字段
                    result = {
                        "from_token": from_token,
                        "to_token": to_token,
                        "price": price,
                        "inverse_price": inverse_price,
                        "timestamp": block_time,
                        "block_height": block_number,
                        "transaction_hash": "",  # 无法从API获取
                        "trade_amount_usd": 0.0,  # USD金额数据可能不可用
                        "exchange": exchange_name,
                        "source": "Bitquery DEX数据"
                    }
                    
                    # 更新缓存
                    self.price_cache[cache_key] = {
                        "data": result,
                        "timestamp": datetime.now().timestamp()
                    }
                    
                    return result
                else:
                    raise Exception(f"交易数据结构不正确: {latest_trade}")
            else:
                error_msg = "API返回结构不符合预期"
                if 'errors' in response:
                    error_detail = str(response.get('errors', ''))
                    error_msg += f": {error_detail}"
                print(f"{error_msg}，使用默认值")
                
                # 使用默认值
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
                    "source": "默认值(API结构错误)"
                }
                
                # 更新缓存
                self.price_cache[cache_key] = {
                    "data": result,
                    "timestamp": datetime.now().timestamp()
                }
                
                return result
        
        except Exception as e:
            print(f"获取交易数据失败: {str(e)}，使用默认值")
            
            # 使用默认值
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
                "source": f"默认值(错误: {str(e)})"
            }
            
            # 更新缓存
            self.price_cache[cache_key] = {
                "data": result,
                "timestamp": datetime.now().timestamp()
            }
            
            return result
    
    async def get_staking_rate(self, token: str) -> Dict[str, Any]:
        """
        获取LST代币的质押率(ETH:LST)
        
        Args:
            token: LST代币符号
            
        Returns:
            dict: 包含质押率信息的字典
        """
        # 检查缓存
        if token in self.staking_rate_cache:
            cache_entry = self.staking_rate_cache[token]
            # 检查缓存是否过期
            if datetime.now().timestamp() - cache_entry["timestamp"] < self.config["cache_ttl"]:
                return cache_entry["data"]
        
        token_info = self.supported_tokens.get(token)
        
        # 如果是ETH或不是LST代币，返回1:1的比率
        if not token_info or token == "ETH" or token == "WETH" or 'protocol_address' not in token_info:
            result = {
                "token": token,
                "staking_rate": 1.0,
                "source": "默认值"
            }
            self.staking_rate_cache[token] = {
                "data": result,
                "timestamp": datetime.now().timestamp()
            }
            return result
        
        protocol_address = token_info['protocol_address']
        
        # 构建查询变量
        variables = {
            "protocolAddress": protocol_address
        }
        
        # 执行查询
        response = await self.execute_query(self.lst_staking_rate_query, variables)
        
        # 解析响应 - 适应新的EVM API结构
        if 'data' in response and 'EVM' in response['data'] and 'SmartContractCalls' in response['data']['EVM']:
            contract_calls = response['data']['EVM']['SmartContractCalls']
            if not contract_calls:
                raise Exception(f"没有找到{token}的质押率数据")
                
            latest_call = contract_calls[0]
            
            # 尝试提取质押率
            staking_rate = None
            staking_rate_function = token_info.get('staking_rate_function')
            
            if 'Arguments' in latest_call:
                for arg in latest_call['Arguments']:
                    if arg['Name'] == staking_rate_function or 'rate' in arg['Name'].lower() or 'price' in arg['Name'].lower():
                        staking_rate = float(arg['Value'])
                        break
            
            if staking_rate is None:
                # 如果无法从参数中提取质押率，使用默认值
                if token == "stETH":
                    staking_rate = 1.03  # 假设 1 stETH ≈ 1.03 ETH
                elif token == "rETH":
                    staking_rate = 1.04  # 假设 1 rETH ≈ 1.04 ETH
                elif token == "cbETH":
                    staking_rate = 1.02  # 假设 1 cbETH ≈ 1.02 ETH
                else:
                    staking_rate = 1.0  # 默认1:1比率
            
            # 构建结果
            result = {
                "token": token,
                "staking_rate": staking_rate,
                "timestamp": latest_call['Block']['Time'],
                "block_height": latest_call['Block']['Number'],
                "transaction_hash": latest_call['Transaction']['Hash'],
                "source": "Bitquery合约数据"
            }
            
            # 更新缓存
            self.staking_rate_cache[token] = {
                "data": result,
                "timestamp": datetime.now().timestamp()
            }
            
            return result
        else:
            # 如果API调用失败，使用默认值
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
                "source": "默认值(API失败)"
            }
            
            # 更新缓存
            self.staking_rate_cache[token] = {
                "data": result,
                "timestamp": datetime.now().timestamp()
            }
            
            return result
    async def calculate_direct_swap(self, from_token: str, to_token: str, amount: float) -> SwapPath:
        """
        计算直接在DEX上交换两个代币的结果
        
        Args:
            from_token: 源代币
            to_token: 目标代币
            amount: 源代币数量
            
        Returns:
            SwapPath: 包含交换结果信息的对象
        """
        # 检查是否是ETH/WETH的互换
        is_eth_weth_swap = (from_token == "ETH" and to_token == "WETH") or (from_token == "WETH" and to_token == "ETH")
        
        # 获取交换价格
        exchange_data = await self.get_token_exchange_data(from_token, to_token)
        
        # 从交易数据确定DEX和费用
        dex_fee = 0.003  # 默认0.3% (Uniswap V2/V3标准)
        dex_name = exchange_data.get("exchange", "Unknown DEX")
        
        # 根据DEX调整费用
        dex_fees = {
            "uniswap": 0.003,  # 0.3%
            "sushiswap": 0.003,  # 0.3%
            "curve": 0.0004,  # 0.04%
            "balancer": 0.002,  # 0.2%
            "dodo": 0.001  # 0.1%
        }
        
        # 对于ETH/WETH互换，不收取DEX费用
        if is_eth_weth_swap:
            dex_fee = 0.0
            dex_name = "ETH/WETH Direct"
        else:
            for key, fee in dex_fees.items():
                if key.lower() in dex_name.lower():
                    dex_fee = fee
                    break
        
        # 获取交易gas成本
        # ETH/WETH互换的gas成本较低
        if is_eth_weth_swap:
            # 使用wrap/unwrap的较低gas成本
            gas_cost = await self.calculate_gas_cost("wrap_eth" if from_token == "ETH" else "unwrap_eth")
        else:
            gas_cost = await self.calculate_gas_cost("swap")
        
        # 使用简单的内部滑点计算，避免调用可能存在参数问题的方法
        # ETH/WETH互换不存在滑点
        if is_eth_weth_swap:
            slippage = 0.0
        else:
            # 使用预定义的滑点表
            token_pair = f"{from_token}_{to_token}"
            default_slippages = {
                "ETH_stETH": 0.001,   # 0.1%
                "ETH_rETH": 0.002,    # 0.2%
                "ETH_cbETH": 0.003,   # 0.3%
                "stETH_rETH": 0.005,  # 0.5%
                "stETH_cbETH": 0.005, # 0.5%
                "rETH_cbETH": 0.008   # 0.8%
            }
            
            # 尝试直接获取滑点，如果不存在则检查反向交易对，最后使用默认值
            slippage = default_slippages.get(token_pair, 
                    default_slippages.get(f"{to_token}_{from_token}", 0.005))
            
            # 根据交易量调整滑点
            # 交易量越大，滑点越大
            volume_multiplier = 1.0
            if amount > 100:
                volume_multiplier = 1.5
            elif amount > 10:
                volume_multiplier = 1.2
            
            slippage = slippage * volume_multiplier
        
        # 计算实际收到的代币数量
        received_amount = amount * exchange_data['price'] * (1 - dex_fee) * (1 - slippage)
        
        # 计算净收益（考虑gas成本）
        eth_price_in_to_token = 1.0
        if to_token != "ETH":
            try:
                eth_to_token_data = await self.get_token_exchange_data("ETH", to_token)
                eth_price_in_to_token = eth_to_token_data.get("price", 1.0)
            except Exception:
                # 如果无法获取ETH到目标代币的汇率，使用默认值
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
        计算通过质押和解质押路径的交换结果
        
        Args:
            from_token: 源代币
            to_token: 目标代币
            amount: 源代币数量
            
        Returns:
            SwapPath: 包含交换结果信息的对象
        """
        from_info = self.supported_tokens.get(from_token)
        to_info = self.supported_tokens.get(to_token)
        
        if not from_info or not to_info:
            raise Exception(f"不支持的代币对: {from_token}/{to_token}")
        
        # ETH -> LST (质押)
        if from_token == "ETH" and to_token in self.trade_paths["ETH_TO_LST"]:
            path_info = self.trade_paths["ETH_TO_LST"][to_token]
            staking_fee = path_info.get("fee", 0.001)
            
            # 获取质押率
            staking_data = await self.get_staking_rate(to_token)
            staking_rate = 1.0 / staking_data["staking_rate"]  # ETH->LST的比率
            
            # 获取gas成本
            gas_cost = await self.calculate_gas_cost("stake")
            
            received_amount = amount * staking_rate * (1 - staking_fee)
            
            # 计算净收益（考虑gas成本）
            eth_price_in_to_token = 1.0
            try:
                eth_to_token_data = await self.get_token_exchange_data("ETH", to_token)
                eth_price_in_to_token = eth_to_token_data.get("price", 1.0)
            except Exception:
                # 如果无法获取ETH到目标代币的汇率，使用默认值
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
            
        # LST -> ETH (解质押)
        elif from_token in self.trade_paths["LST_TO_ETH"] and to_token == "ETH":
            path_info = self.trade_paths["LST_TO_ETH"][from_token]
            unstaking_fee = path_info.get("fee", 0.001)
            unstaking_delay = path_info.get("delay", 7 * 24 * 60 * 60)  # 默认7天
            
            # 获取质押率
            staking_data = await self.get_staking_rate(from_token)
            staking_rate = staking_data["staking_rate"]  # LST->ETH的比率
            
            # 获取gas成本
            gas_cost = await self.calculate_gas_cost("unstake")
            
            received_amount = amount * staking_rate * (1 - unstaking_fee)
            
            # 计算延迟天数
            delay_days = unstaking_delay / (24 * 60 * 60)
            
            # 计算机会成本
            opportunity_cost_percent = await self.calculate_opportunity_cost(delay_days, from_token)
            
            # 计算净收益（考虑gas成本和时间价值）
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
        
        # LST -> LST (需要通过ETH中转)
        elif from_token in self.trade_paths["LST_TO_ETH"] and to_token in self.trade_paths["ETH_TO_LST"]:
            # 第一步：LST -> ETH
            unstake_path = self.trade_paths["LST_TO_ETH"][from_token]
            unstaking_fee = unstake_path.get("fee", 0.001)
            unstaking_delay = unstake_path.get("delay", 7 * 24 * 60 * 60)
            
            from_staking_data = await self.get_staking_rate(from_token)
            from_staking_rate = from_staking_data["staking_rate"]
            
            # 获取第一步gas成本
            gas_cost_1 = await self.calculate_gas_cost("unstake")
            
            intermediate_amount = amount * from_staking_rate * (1 - unstaking_fee)
            
            # 第二步：ETH -> LST
            stake_path = self.trade_paths["ETH_TO_LST"][to_token]
            staking_fee = stake_path.get("fee", 0.001)
            
            to_staking_data = await self.get_staking_rate(to_token)
            to_staking_rate = 1.0 / to_staking_data["staking_rate"]
            
            # 获取第二步gas成本
            gas_cost_2 = await self.calculate_gas_cost("stake")
            
            # 计算延迟天数
            delay_days = unstaking_delay / (24 * 60 * 60)
            
            # 计算机会成本
            opportunity_cost_percent = await self.calculate_opportunity_cost(delay_days, from_token)
            
            # 考虑第一步的gas成本
            intermediate_amount = intermediate_amount - gas_cost_1["gas_cost_eth"]
            
            # 考虑延迟的机会成本
            intermediate_amount_after_delay = intermediate_amount * (1 - opportunity_cost_percent)
            
            # 考虑第二步
            final_amount = intermediate_amount_after_delay * to_staking_rate * (1 - staking_fee)
            
            # 考虑第二步的gas成本
            net_final_amount = final_amount - gas_cost_2["gas_cost_eth"]
            
            # 计算总gas成本（ETH和USD）
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
            raise Exception(f"没有可用的质押/解质押路径: {from_token} -> {to_token}")
    
    # 后续方法保持不变...
    async def analyze_arbitrage_path(self, from_token: str, to_token: str, amount: float, risk_preference: float = None) -> Dict[str, Any]:
        """分析两个代币之间的套利路径"""
        # 使用传入的风险偏好或默认值
        if risk_preference is None:
            risk_preference = self.config["risk_preference"]
        
        # 计算直接交换路径
        try:
            direct_swap = await self.calculate_direct_swap(from_token, to_token, amount)
            direct_swap_dict = direct_swap.__dict__
            has_direct_swap = True
        except Exception as e:
            direct_swap_dict = {
                "from_token": from_token,
                "to_token": to_token,
                "input_amount": amount,
                "error": f"直接交换失败: {str(e)}"
            }
            has_direct_swap = False
        
        # 计算质押/解质押路径
        try:
            staking_path = await self.calculate_staking_unstaking(from_token, to_token, amount)
            staking_path_dict = staking_path.__dict__
            has_staking_path = True
        except Exception as e:
            staking_path_dict = {
                "from_token": from_token,
                "to_token": to_token,
                "input_amount": amount,
                "error": f"质押/解质押路径失败: {str(e)}"
            }
            has_staking_path = False
        
        # 如果两种路径都失败，则返回错误
        if not has_direct_swap and not has_staking_path:
            return {
                "from_token": from_token,
                "to_token": to_token,
                "amount": amount,
                "error": "没有可用的交换路径"
            }
        
        # 计算年化收益率
        results = []
        
        if has_direct_swap:
            # 使用净收益（考虑gas成本）
            direct_swap_profit = direct_swap.net_output_amount - amount
            direct_swap_profit_percent = (direct_swap_profit / amount) * 100
            
            # 直接交换通常是即时的，所以年化收益率就是交易收益率
            direct_swap_dict["profit"] = direct_swap_profit
            direct_swap_dict["profit_percent"] = direct_swap_profit_percent
            direct_swap_dict["annual_profit_percent"] = direct_swap_profit_percent  # 即时交易，不需要年化
            direct_swap_dict["is_profitable"] = direct_swap_profit > self.config["min_profit_threshold"]
            
            # 转换SwapStep对象为字典
            if "steps" in direct_swap_dict and isinstance(direct_swap_dict["steps"], list):
                direct_swap_dict["steps"] = [step.__dict__ for step in direct_swap_dict["steps"]]
            
            results.append(direct_swap_dict)
        
        if has_staking_path:
            # 使用净收益（考虑gas成本和时间价值）
            staking_profit = staking_path.net_output_amount - amount
            staking_profit_percent = (staking_profit / amount) * 100
            
            # 对于需要等待的解质押路径，计算年化收益率
            unstaking_time_days = 0
            for step in staking_path.steps:
                if step.type == "unstake" and step.delay_days:
                    unstaking_time_days = step.delay_days
                    break
            
            if unstaking_time_days > 0:
                annual_profit_percent = staking_profit_percent * (365 / unstaking_time_days)
            else:
                annual_profit_percent = staking_profit_percent  # 如果是即时交易，不需要年化
            
            staking_path_dict["profit"] = staking_profit
            staking_path_dict["profit_percent"] = staking_profit_percent
            staking_path_dict["annual_profit_percent"] = annual_profit_percent
            staking_path_dict["unstaking_time_days"] = unstaking_time_days
            staking_path_dict["is_profitable"] = staking_profit > self.config["min_profit_threshold"]
            
            # 转换SwapStep对象为字典
            if "steps" in staking_path_dict and isinstance(staking_path_dict["steps"], list):
                staking_path_dict["steps"] = [step.__dict__ for step in staking_path_dict["steps"]]
            
            results.append(staking_path_dict)
        
        # 找出最佳路径
        profitable_results = [r for r in results if r.get('is_profitable', False)]
        
        if not profitable_results:
            # 如果没有盈利路径，选择损失最小的
            if results:
                best_path = max(results, key=lambda x: x.get('profit_percent', -float('inf')))
            else:
                return {
                    "from_token": from_token,
                    "to_token": to_token,
                    "amount": amount,
                    "error": "没有可用的交换路径"
                }
        else:
            # 找出最佳路径
            if len(profitable_results) == 1:
                best_path = profitable_results[0]
            else:
                # 比较直接交换和解质押路径
                direct_paths = [r for r in profitable_results if r['route_type'] == 'direct_swap']
                staking_paths = [r for r in profitable_results if r['route_type'] in ['staking', 'unstaking', 'unstake_then_stake']]
                
                if direct_paths and staking_paths:
                    direct_path = direct_paths[0]
                    staking_path = max(staking_paths, key=lambda x: x['annual_profit_percent'])
                    
                    # 计算时间价值调整
                    time_value_adjustment = staking_path.get('unstaking_time_days', 0) * self.config["time_value_factor"]
                    
                    # 考虑风险偏好和时间价值
                    if staking_path['annual_profit_percent'] > direct_path['annual_profit_percent'] + risk_preference + time_value_adjustment:
                        best_path = staking_path
                    else:
                        best_path = direct_path
                elif direct_paths:
                    best_path = direct_paths[0]
                else:
                    best_path = max(staking_paths, key=lambda x: x['annual_profit_percent'])
        
        # 添加当前时间戳和市场条件
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
        """寻找循环套利机会"""
        # 此方法代码保持不变，因为它调用其他已更新的方法
        # 获取所有支持的代币
        tokens = list(self.supported_tokens.keys())
        
        # 如果起始代币不在列表中，返回错误
        if start_token not in tokens:
            return {
                "start_token": start_token,
                "error": "不支持的代币"
            }
        
        # 找出所有可能的二级路径
        two_hop_paths = []
        for intermediate_token in tokens:
            if intermediate_token != start_token:
                try:
                    # 分析 start_token -> intermediate_token -> start_token 路径
                    first_hop = await self.calculate_direct_swap(start_token, intermediate_token, amount)
                    
                    # 使用净产出考虑gas
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
                    # 记录错误但继续处理
                    print(f"二级路径分析失败 {start_token}->{intermediate_token}: {str(e)}")
        
        # 找出可能有价值的中间代币（根据二级路径结果）
        potential_tokens = set()
        for path in two_hop_paths:
            intermediate_token = path["path"][1]
            potential_tokens.add(intermediate_token)
        
        # 如果没有有价值的中间代币，使用几个主要的LST代币
        if not potential_tokens:
            potential_tokens = set(["stETH", "rETH", "cbETH", "WETH"])
        
        # 找出所有可能的三级路径
        three_hop_paths = []
        for intermediate1 in potential_tokens:
            if intermediate1 != start_token:
                for intermediate2 in potential_tokens:
                    if intermediate2 != start_token and intermediate2 != intermediate1:
                        try:
                            # 分析 start_token -> intermediate1 -> intermediate2 -> start_token 路径
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
                            # 记录错误但继续处理
                            print(f"三级路径分析失败 {start_token}->{intermediate1}->{intermediate2}: {str(e)}")
        
        # 合并所有路径并按利润排序
        all_paths = sorted(two_hop_paths + three_hop_paths, key=lambda x: x['profit_percent'], reverse=True)
        
        # 添加当前时间戳和市场条件
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
        """执行LST套利分析"""
        # 此方法代码保持不变，因为它调用其他已更新的方法
        from_token = params.get('from_token')
        to_token = params.get('to_token')
        amount = params.get('amount', 1.0)
        find_all_routes = params.get('find_all_routes', False)
        risk_preference = params.get('risk_preference', self.config["risk_preference"])
        
        # 确保amount是数字
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
            # 获取当前gas价格
            gas_price = await self.get_current_gas_price()
            results["current_gas_price_gwei"] = gas_price
        except Exception as e:
            # 如果获取gas价格失败，不应该影响整个分析
            print(f"获取gas价格失败: {str(e)}")
        
        # 如果指定了 from_token 和 to_token，分析特定路径
        if from_token and to_token:
            try:
                arbitrage_analysis = await self.analyze_arbitrage_path(from_token, to_token, amount, risk_preference)
                results["arbitrage_analysis"] = arbitrage_analysis
            except Exception as e:
                results["arbitrage_analysis"] = {
                    "from_token": from_token,
                    "to_token": to_token,
                    "amount": amount,
                    "error": f"分析失败: {str(e)}"
                }
            
            # 如果请求查找循环套利，则额外分析
            if find_all_routes:
                try:
                    circular_analysis = await self.find_circular_arbitrage(from_token, amount)
                    results["circular_arbitrage"] = circular_analysis
                except Exception as e:
                    results["circular_arbitrage"] = {
                        "start_token": from_token,
                        "error": f"循环套利分析失败: {str(e)}"
                    }
            
            return results
        
        # 如果只指定了 from_token，分析从该代币到其他所有代币的路径
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
                        "error": f"分析失败: {str(e)}"
                    })
            
            # 找出最佳路径
            profitable_analyses = [a for a in all_analyses if 'best_path' in a and a['best_path'].get('is_profitable', False)]
            
            if profitable_analyses:
                best_analysis = max(profitable_analyses, key=lambda x: x['best_path']['annual_profit_percent'])
            else:
                best_analysis = None
            
            # 如果请求查找循环套利，则额外分析
            if find_all_routes:
                try:
                    circular_analysis = await self.find_circular_arbitrage(from_token, amount)
                    results["circular_arbitrage"] = circular_analysis
                except Exception as e:
                    results["circular_arbitrage"] = {
                        "start_token": from_token,
                        "error": f"循环套利分析失败: {str(e)}"
                    }
            
            results["from_token"] = from_token
            results["all_analyses"] = all_analyses
            results["best_analysis"] = best_analysis
            
            return results
        
        # 如果都没指定，分析所有可能的组合中最佳的套利机会
        else:
            # 优化: 只分析主要代币对减少API调用
            main_tokens = ["ETH", "stETH", "rETH", "cbETH"]
            all_pairs = []
            
            # 生成关键代币对
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
                        "error": f"分析失败: {str(e)}"
                    })
            
            # 找出最佳套利机会
            profitable_analyses = [a for a in all_analyses if 'best_path' in a and a['best_path'].get('is_profitable', False)]
            
            if profitable_analyses:
                best_analysis = max(profitable_analyses, key=lambda x: x['best_path']['annual_profit_percent'])
            else:
                best_analysis = None
            
            # 查找最佳循环套利机会
            circular_results = []
            
            for token in main_tokens:
                try:
                    circular_analysis = await self.find_circular_arbitrage(token, amount)
                    if circular_analysis.get('best_opportunity'):
                        circular_results.append(circular_analysis)
                except Exception as e:
                    print(f"{token}循环套利分析失败: {str(e)}")
            
            best_circular = max(circular_results, key=lambda x: x['best_opportunity']['profit_percent']) if circular_results else None
            
            results["all_analyses"] = all_analyses
            results["best_direct_analysis"] = best_analysis
            results["best_circular_arbitrage"] = best_circular
            
            # 综合比较直接套利和循环套利
            if best_analysis and best_circular:
                direct_profit_percent = best_analysis['best_path']['annual_profit_percent']
                circular_profit_percent = best_circular['best_opportunity']['profit_percent']
                
                # 考虑风险偏好
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
        """安全获取缓存TTL值"""
        default_ttl = 300  # 默认5分钟
        if not hasattr(self, 'config') or not isinstance(self.config, dict):
            return default_ttl
        return self.config.get("cache_ttl", default_ttl)