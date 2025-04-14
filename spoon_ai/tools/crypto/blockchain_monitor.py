# spoon_ai/tools/crypto/blockchain_monitor.py
from spoon_ai.tools.base import BaseTool
from typing import Dict, Any, Optional, Literal

class CryptoMarketMonitor(BaseTool):
    name: str = "crypto_market_monitor"
    description: str = "Monitor price, volume, or other metrics for tokens on both centralized (CEX) and decentralized exchanges (DEX)"
    parameters: dict = {
        "type": "object",
        "properties": {
            "market_type": {
                "type": "string",
                "description": "Market type: cex (centralized exchange) or dex (decentralized exchange)"
            },
            "platform": {
                "type": "string",
                "description": "Trading platform: binance (CEX), ethereum/uniswap (DEX), solana/raydium (DEX)"
            },
            "symbol": {
                "type": "string",
                "description": "Trading pair symbol (e.g., BTCUSDT for Binance, ETH-USDC for Uniswap, SOL-USDC for Raydium)"
            },
            "metric": {
                "type": "string",
                "description": "Metric to monitor: price, volume, price_change, price_change_percent, liquidity (DEX only)"
            },
            "threshold": {
                "type": "number",
                "description": "Alert threshold value"
            },
            "comparator": {
                "type": "string",
                "description": "Comparison operator: >, <, =, >=, <="
            },
            "check_interval_minutes": {
                "type": "integer",
                "description": "Check interval in minutes (default: 5)"
            },
            "name": {
                "type": "string",
                "description": "Custom name for this monitor"
            }
        },
        "required": ["market_type", "platform", "symbol", "metric", "threshold", "comparator"]
    }
    
    async def execute(
        self, 
        market_type: Literal["cex", "dex"],
        platform: str,
        symbol: str,
        metric: str,
        threshold: float,
        comparator: str,
        check_interval_minutes: int = 5,
        name: Optional[str] = None
    ) -> Dict[str, Any]:
        """Create a cryptocurrency market monitoring task for CEX or DEX"""
        import aiohttp  # 异步HTTP客户端
        
        # Map platform to provider code
        provider = self._get_provider_for_platform(market_type, platform)
        print(provider)
        # Create default name if not provided
        if not name:
            platform_name = self._get_platform_display_name(platform)
            name = f"{platform_name} {symbol} {metric} Monitor"
        
        task_config = {
            "market": market_type,
            "provider": provider,
            "symbol": symbol,
            "metric": metric,
            "threshold": threshold,
            "comparator": comparator,
            "name": name,
            "check_interval_minutes": check_interval_minutes,
            "expires_in_hours": 24,  # Default to 24 hours
            "notification_channels": ["Email"]
        }
        
        # Send to the monitoring service
        api_url = "http://localhost:8888/monitoring/tasks"
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(api_url, json=task_config) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        raise ValueError(f"API request failed: {response.status} - {error_text}")
                    
                    result = await response.json()
            
            return {
                "status": "success",
                "message": f"Created {market_type.upper()} monitoring for {symbol} on {platform}",
                "task_id": result.get("task_id", "unknown"),
                "expires_at": result.get("expires_at", "unknown")
            }
        except aiohttp.ClientError as e:
            return {
                "status": "error",
                "message": f"Connection to monitoring service failed: {str(e)}",
                "error_type": "connection_error"
            }
        except ValueError as e:
            return {
                "status": "error",
                "message": str(e),
                "error_type": "api_error"
            }
        except Exception as e:
            return {
                "status": "error",
                "message": f"Unexpected error: {str(e)}",
                "error_type": "unknown_error"
            }
    
    def _get_provider_for_platform(self, market_type: str, platform: str) -> str:
        """Convert platform name to provider code"""
        market_type = market_type.lower()
        platform = platform.lower()
        
        if market_type == "cex":
            if platform in ["binance", "bn"]:
                return "bn"
            # Add more CEX mappings as needed
            else:
                raise ValueError(f"Unsupported CEX platform: {platform}. Supported: binance")
        
        elif market_type == "dex":
            if platform in ["ethereum", "eth", "evm", "uniswap", "uni"]:
                return "uni"  # Uniswap
            elif platform in ["solana", "sol", "raydium", "ray"]:
                return "ray"  # Raydium
            # Add more DEX mappings as needed
            else:
                raise ValueError(f"Unsupported DEX platform: {platform}. Supported: ethereum/uniswap, solana/raydium")
        
        else:
            raise ValueError(f"Unsupported market type: {market_type}. Supported: cex, dex")
    
    def _get_platform_display_name(self, platform: str) -> str:
        """Get a user-friendly display name for the platform"""
        platform = platform.lower()
        
        if platform in ["binance", "bn"]:
            return "Binance"
        elif platform in ["ethereum", "eth", "evm", "uniswap", "uni"]:
            return "Uniswap"
        elif platform in ["solana", "sol", "raydium", "ray"]:
            return "Raydium"
        else:
            return platform.capitalize()