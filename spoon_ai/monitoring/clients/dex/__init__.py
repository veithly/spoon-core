# spoon_ai/monitoring/clients/dex/__init__.py
from typing import Dict
from .base import DEXClient
from .uniswap import UniswapClient

# 注册支持的DEX提供者
DEX_PROVIDERS = {
    "uni": UniswapClient,
    "uniswap": UniswapClient,
    # 添加更多提供者...
}

def get_dex_client(provider: str) -> DEXClient:
    """
    根据提供者名称获取相应的DEX客户端
    
    Args:
        provider: 提供者代码 (例如 'uni' 表示 Uniswap)
        
    Returns:
        DEXClient: 相应的交易所客户端实例
    
    Raises:
        ValueError: 如果提供者不支持
    """
    provider_lower = provider.lower()
    if provider_lower in DEX_PROVIDERS:
        return DEX_PROVIDERS[provider_lower]()
    else:
        supported = ", ".join(DEX_PROVIDERS.keys())
        raise ValueError(f"Unsupported DEX provider: {provider}. Supported providers: {supported}")