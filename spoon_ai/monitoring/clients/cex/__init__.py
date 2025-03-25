# spoon_ai/monitoring/clients/cex/__init__.py
from typing import Dict
from .base import CEXClient
from .binance import BinanceClient

# 注册支持的CEX提供者
CEX_PROVIDERS = {
    "bn": BinanceClient,
    "binance": BinanceClient,
    # 添加更多提供者...
}

def get_cex_client(provider: str) -> CEXClient:
    """
    根据提供者名称获取相应的CEX客户端
    
    Args:
        provider: 提供者代码 (例如 'bn' 表示 Binance)
        
    Returns:
        CEXClient: 相应的交易所客户端实例
    
    Raises:
        ValueError: 如果提供者不支持
    """
    provider_lower = provider.lower()
    if provider_lower in CEX_PROVIDERS:
        return CEX_PROVIDERS[provider_lower]()
    else:
        supported = ", ".join(CEX_PROVIDERS.keys())
        raise ValueError(f"Unsupported CEX provider: {provider}. Supported providers: {supported}")