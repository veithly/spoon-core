import json
import logging
import asyncio
import time
from typing import Dict, Any, List, Optional, Union, Tuple, ClassVar
from dataclasses import dataclass
from decimal import Decimal

import aiohttp
import requests
from pydantic import Field, validator

from spoon_ai.tools.base import BaseTool, ToolResult
from spoon_ai.tools.crypto.base import DefiBaseTool

logger = logging.getLogger(__name__)

@dataclass
class LendingProtocol:
    """Lending protocol information"""
    name: str
    chain: str
    api_url: Optional[str] = None
    subgraph_url: Optional[str] = None
    supported_assets: Optional[List[str]] = None
    logo: Optional[str] = None

@dataclass
class LendingAsset:
    """Lending asset information"""
    symbol: str
    name: str
    supply_apy: float
    borrow_apy: float
    total_supply: float
    total_borrow: float
    utilization: float
    liquidity: float
    protocol: str
    chain: str
    timestamp: int

class LendingRateProvider:
    """Base class for lending rate data providers"""
    
    async def get_lending_rates(self, 
                                chains: Optional[List[str]] = None, 
                                protocols: Optional[List[str]] = None, 
                                assets: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """Get lending rates for specified chains, protocols and assets"""
        raise NotImplementedError("Subclasses must implement this method")

class ChainsightLendingProvider(LendingRateProvider):
    """Provider that aggregates data from various sources including DeFiLlama, subgraphs, and direct protocol APIs"""
    
    # Protocol configurations
    PROTOCOLS: ClassVar[Dict[str, LendingProtocol]] = {
        "aave-v3": LendingProtocol(
            name="Aave V3",
            chain="ethereum",
            subgraph_url="https://api.thegraph.com/subgraphs/name/aave/protocol-v3"
        ),
        "aave-v3-arbitrum": LendingProtocol(
            name="Aave V3",
            chain="arbitrum",
            subgraph_url="https://api.thegraph.com/subgraphs/name/aave/protocol-v3-arbitrum"
        ),
        "aave-v3-optimism": LendingProtocol(
            name="Aave V3",
            chain="optimism",
            subgraph_url="https://api.thegraph.com/subgraphs/name/aave/protocol-v3-optimism"
        ),
        "compound-v3": LendingProtocol(
            name="Compound V3",
            chain="ethereum",
            api_url="https://api.compound.finance/api/v2/markets"
        ),
        "compound-v3-arbitrum": LendingProtocol(
            name="Compound V3",
            chain="arbitrum",
            api_url="https://api.compound.finance/api/v2/markets?network=arbitrum"
        ),
        "spark": LendingProtocol(
            name="Spark",
            chain="ethereum",
            subgraph_url="https://api.thegraph.com/subgraphs/name/spark-lend/spark-lend"
        ),
        "morpho-aave-v3": LendingProtocol(
            name="Morpho Aave V3",
            chain="ethereum",
            api_url="https://api.morpho.xyz/aave-v3/markets"
        ),
        "morpho-compound": LendingProtocol(
            name="Morpho Compound",
            chain="ethereum",
            api_url="https://api.morpho.xyz/compound/markets"
        ),
        "venus": LendingProtocol(
            name="Venus",
            chain="bnb",
            api_url="https://api.venus.io/api/markets"
        ),
        "solend": LendingProtocol(
            name="Solend",
            chain="solana",
            api_url="https://api.solend.fi/v1/markets"
        ),
    }
    
    # DeFiLlama API endpoint
    DEFILLAMA_YIELDS_API = "https://yields.llama.fi/pools"
    
    # Important stablecoin and asset addresses
    ASSETS = {
        "USDC": {
            "ethereum": "0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48",
            "arbitrum": "0xaf88d065e77c8cc2239327c5edb3a432268e5831",
            "optimism": "0x7f5c764cbc14f9669b88837ca1490cca17c31607",
            "bnb": "0x8ac76a51cc950d9822d68b83fe1ad97b32cd580d",
            "base": "0x833589fcd6edb6e08f4c7c32d4f71b54bda02913",
        },
        "USDT": {
            "ethereum": "0xdac17f958d2ee523a2206206994597c13d831ec7",
            "arbitrum": "0xfd086bc7cd5c481dcc9c85ebe478a1c0b69fcbb9",
            "optimism": "0x94b008aa00579c1307b0ef2c499ad98a8ce58e58",
            "bnb": "0x55d398326f99059ff775485246999027b3197955",
            "base": "0x50c5725949a6f0c72e6c4a641f24049a917db0cb",
        },
        "ETH": {
            "ethereum": "0xeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee",
            "arbitrum": "0xeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee",
            "optimism": "0xeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee",
            "base": "0xeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee",
        },
        "BTC": {
            "ethereum": "0x2260fac5e5542a773aa44fbcfedf7c193bc2c599", # WBTC
            "arbitrum": "0x2f2a2543b76a4166549f7aab2e75bef0aefc5b0f", # WBTC
            "optimism": "0x68f180fcce6836688e9084f035309e29bf0a2095", # WBTC
            "bnb": "0x7130d2a12b9bcbfae4f2634d864a1ee1ce3ead9c", # BTCB
        }
    }
    
    def __init__(self):
        self.session = None
        self.rates_cache = {}
        self.last_updated = {}
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def fetch_defillama_rates(self) -> List[Dict[str, Any]]:
        """Fetch lending rates from DeFiLlama Yields API"""
        if not self.session:
            self.session = aiohttp.ClientSession()
            
        try:
            async with self.session.get(self.DEFILLAMA_YIELDS_API) as response:
                response.raise_for_status()
                data = await response.json()
                return data.get('data', [])
        except Exception as e:
            logger.error(f"Failed to fetch DeFiLlama rates: {e}")
            return []
    
    async def fetch_aave_rates(self, chain: str) -> List[Dict[str, Any]]:
        """Fetch lending rates from Aave subgraph for specified chain"""
        protocol_key = f"aave-v3-{chain}" if chain != "ethereum" else "aave-v3"
        if protocol_key not in self.PROTOCOLS:
            logger.error(f"Unsupported chain for Aave: {chain}")
            return []
            
        protocol = self.PROTOCOLS[protocol_key]
        if not protocol.subgraph_url:
            logger.error(f"No subgraph URL for {protocol_key}")
            return []
            
        query = """
        {
          reserves(first: 100) {
            symbol
            name
            underlyingAsset
            liquidityRate
            variableBorrowRate
            totalATokenSupply
            totalCurrentVariableDebt
            availableLiquidity
          }
        }
        """
            
        try:
            if not self.session:
                self.session = aiohttp.ClientSession()
                
            async with self.session.post(
                protocol.subgraph_url,
                json={"query": query}
            ) as response:
                response.raise_for_status()
                data = await response.json()
                
                result = []
                timestamp = int(time.time())
                
                for reserve in data.get('data', {}).get('reserves', []):
                    # Convert rates from ray format (27 decimals) to percentage
                    supply_apy = float(reserve.get('liquidityRate', '0')) / 1e25
                    borrow_apy = float(reserve.get('variableBorrowRate', '0')) / 1e25
                    
                    total_supply = float(reserve.get('totalATokenSupply', '0'))
                    total_borrow = float(reserve.get('totalCurrentVariableDebt', '0'))
                    liquidity = float(reserve.get('availableLiquidity', '0'))
                    
                    # Calculate utilization rate
                    utilization = 0
                    if total_supply > 0:
                        utilization = (total_borrow / total_supply) * 100
                        
                    result.append({
                        'symbol': reserve.get('symbol'),
                        'name': reserve.get('name'),
                        'supply_apy': supply_apy,
                        'borrow_apy': borrow_apy,
                        'total_supply': total_supply,
                        'total_borrow': total_borrow,
                        'utilization': utilization,
                        'liquidity': liquidity,
                        'protocol': protocol.name,
                        'chain': chain,
                        'timestamp': timestamp
                    })
                
                return result
        except Exception as e:
            logger.error(f"Failed to fetch Aave rates for {chain}: {e}")
            return []
    
    async def get_lending_rates(self, 
                               chains: Optional[List[str]] = None, 
                               protocols: Optional[List[str]] = None, 
                               assets: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """Get lending rates for specified chains, protocols and assets"""
        # Define default values if not provided
        if chains is None:
            chains = ["ethereum", "arbitrum", "optimism", "bnb", "solana", "base"]
        
        if protocols is None:
            protocols = list(self.PROTOCOLS.keys())
        
        if assets is None:
            assets = ["USDC", "USDT", "ETH", "BTC"]
        
        # Check cache age and update if needed (15 minute cache time)
        cache_key = f"{'-'.join(sorted(chains))}-{'-'.join(sorted(protocols))}-{'-'.join(sorted(assets))}"
        current_time = time.time()
        
        if (cache_key in self.rates_cache and cache_key in self.last_updated and
            current_time - self.last_updated[cache_key] < 15 * 60):
            return self.rates_cache[cache_key]
        
        # Initialize result list and async tasks
        all_rates = []
        tasks = []
        
        # Add DeFiLlama task
        tasks.append(self.fetch_defillama_rates())
        
        # Add Aave tasks for each chain
        for chain in chains:
            if chain in ["ethereum", "arbitrum", "optimism"] and any(p.startswith("aave") for p in protocols):
                tasks.append(self.fetch_aave_rates(chain))
        
        # Execute all tasks concurrently
        results = await asyncio.gather(*tasks)
        
        # Process DeFiLlama results
        defillama_rates = results[0]
        
        for pool in defillama_rates:
            # Filter by chain, protocol and asset
            if (pool.get("chain") in chains and 
                any(p.lower() in pool.get("project", "").lower() for p in protocols) and
                any(a.lower() in pool.get("symbol", "").lower() for a in assets)):
                
                all_rates.append({
                    'symbol': pool.get('symbol'),
                    'name': pool.get('pool'),
                    'supply_apy': pool.get('apy'),
                    'borrow_apy': pool.get('apyMean30d', 0),  # Use average if borrow not available
                    'total_supply': pool.get('tvlUsd', 0),
                    'total_borrow': 0,  # Not provided by DeFiLlama
                    'utilization': pool.get('utilization', 0) * 100 if 'utilization' in pool else 0,
                    'liquidity': pool.get('tvlUsd', 0),
                    'protocol': pool.get('project'),
                    'chain': pool.get('chain'),
                    'timestamp': int(time.time())
                })
        
        # Process direct API results (starting from index 1)
        for i in range(1, len(results)):
            all_rates.extend(results[i])
        
        # Filter results by assets if needed
        filtered_rates = all_rates
        if assets:
            filtered_rates = [
                rate for rate in all_rates 
                if any(asset.lower() in rate.get('symbol', '').lower() for asset in assets)
            ]
        
        # Update cache
        self.rates_cache[cache_key] = filtered_rates
        self.last_updated[cache_key] = current_time
        
        return filtered_rates

class LendingRateMonitorTool(DefiBaseTool):
    """Tool for monitoring lending rates across different chains and protocols"""
    
    name: str = "lending_rate_monitor"
    description: str = "Monitor lending and borrowing rates for stablecoins, ETH, and BTC across different chains and protocols"
    parameters: dict = {
        "type": "object",
        "properties": {
            "assets": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List of assets to query (e.g. USDC, USDT, ETH, BTC)",
                "default": ["USDC", "USDT", "ETH", "BTC"]
            },
            "chains": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List of chains to query (e.g. ethereum, arbitrum, optimism, bnb, solana, base)",
                "default": ["ethereum", "arbitrum", "optimism", "bnb", "solana", "base"]
            },
            "protocols": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List of protocols to query (e.g. aave, compound, venus, solend, morpho)",
                "default": ["aave", "compound", "morpho", "venus", "solend", "spark"]
            },
            "sort_by": {
                "type": "string",
                "description": "Field to sort results by (supply_apy, borrow_apy, utilization)",
                "enum": ["supply_apy", "borrow_apy", "utilization"],
                "default": "supply_apy"
            },
            "order": {
                "type": "string",
                "description": "Sort order (asc or desc)",
                "enum": ["asc", "desc"],
                "default": "desc"
            }
        }
    }
    
    # Provider instance cache
    _provider = None
    
    @classmethod
    def get_provider(cls) -> LendingRateProvider:
        """Get or create lending rate provider"""
        if cls._provider is None:
            cls._provider = ChainsightLendingProvider()
        return cls._provider
    
    async def execute(self, 
                     assets: List[str] = None, 
                     chains: List[str] = None, 
                     protocols: List[str] = None,
                     sort_by: str = "supply_apy",
                     order: str = "desc") -> ToolResult:
        """Execute the tool"""
        try:
            provider = self.get_provider()
            rates = await provider.get_lending_rates(
                chains=chains,
                protocols=protocols,
                assets=assets
            )
            
            # Sort results
            if sort_by in ["supply_apy", "borrow_apy", "utilization"]:
                reverse = order == "desc"
                rates = sorted(rates, key=lambda x: x.get(sort_by, 0), reverse=reverse)
            
            # Group by asset for better readability
            result = {
                "timestamp": int(time.time()),
                "count": len(rates),
                "rates_by_asset": {}
            }
            
            for rate in rates:
                symbol = rate.get("symbol")
                if symbol not in result["rates_by_asset"]:
                    result["rates_by_asset"][symbol] = []
                result["rates_by_asset"][symbol].append(rate)
            
            # Find arbitrage opportunities
            result["arbitrage_opportunities"] = self.find_arbitrage_opportunities(rates)
            
            return ToolResult(output=result)
        except Exception as e:
            logger.error(f"Failed to execute lending rate monitor: {e}")
            return ToolResult(error=f"Failed to execute lending rate monitor: {str(e)}")
    
    def find_arbitrage_opportunities(self, rates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Find arbitrage opportunities between different protocols and chains"""
        opportunities = []
        
        # Group rates by asset symbol
        rates_by_symbol = {}
        for rate in rates:
            symbol = rate.get("symbol")
            if symbol not in rates_by_symbol:
                rates_by_symbol[symbol] = []
            rates_by_symbol[symbol].append(rate)
        
        # Find assets with supply_apy differences > 1% or borrow_apy differences > 1%
        for symbol, asset_rates in rates_by_symbol.items():
            if len(asset_rates) <= 1:
                continue
                
            # Sort by supply APY descending
            supply_sorted = sorted(asset_rates, key=lambda x: x.get("supply_apy", 0), reverse=True)
            
            # Sort by borrow APY ascending
            borrow_sorted = sorted(asset_rates, key=lambda x: x.get("borrow_apy", 0))
            
            # Check for supply side arbitrage opportunities (difference > 1%)
            for i in range(len(supply_sorted) - 1):
                high = supply_sorted[i]
                low = supply_sorted[-1]  # Compare against lowest
                diff = high.get("supply_apy", 0) - low.get("supply_apy", 0)
                
                if diff >= 1.0:  # At least 1% difference
                    opportunities.append({
                        "type": "supply",
                        "asset": symbol,
                        "difference_percent": round(diff, 2),
                        "high_rate": {
                            "protocol": high.get("protocol"),
                            "chain": high.get("chain"),
                            "apy": high.get("supply_apy")
                        },
                        "low_rate": {
                            "protocol": low.get("protocol"),
                            "chain": low.get("chain"),
                            "apy": low.get("supply_apy")
                        }
                    })
                    break  # Only include the largest difference
            
            # Check for borrow side arbitrage opportunities (difference > 1%)
            for i in range(len(borrow_sorted) - 1):
                low = borrow_sorted[i]
                high = borrow_sorted[-1]  # Compare against highest
                diff = high.get("borrow_apy", 0) - low.get("borrow_apy", 0)
                
                if diff >= 1.0:  # At least 1% difference
                    opportunities.append({
                        "type": "borrow",
                        "asset": symbol,
                        "difference_percent": round(diff, 2),
                        "high_rate": {
                            "protocol": high.get("protocol"),
                            "chain": high.get("chain"),
                            "apy": high.get("borrow_apy")
                        },
                        "low_rate": {
                            "protocol": low.get("protocol"),
                            "chain": low.get("chain"),
                            "apy": low.get("borrow_apy")
                        }
                    })
                    break  # Only include the largest difference
        
        # Sort opportunities by difference percentage
        return sorted(opportunities, key=lambda x: x.get("difference_percent", 0), reverse=True) 