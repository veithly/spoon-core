from .tool_manager import ToolManager
from .base import BaseTool
from .terminate import Terminate

# Import from spoon-toolkits package
try:
    from spoon_toolkits.crypto.predict_price import PredictPrice
    from spoon_toolkits.crypto.token_holders import TokenHolders
    from spoon_toolkits.crypto.trading_history import TradingHistory
    from spoon_toolkits.crypto.uniswap_liquidity import UniswapLiquidity
    from spoon_toolkits.crypto.wallet_analysis import WalletAnalysis
    from spoon_toolkits.crypto.price_data import GetTokenPriceTool, Get24hStatsTool, GetKlineDataTool
    from spoon_toolkits.crypto.price_alerts import PriceThresholdAlertTool, LpRangeCheckTool, SuddenPriceIncreaseTool
    from spoon_toolkits.crypto.lending_rates import LendingRateMonitorTool
    # from spoon_toolkits.crypto.lst_arbitrage import LstArbitrageTool
    from spoon_toolkits.token_execute.token_transfer import TokenTransfer
except ImportError:
    # Fallback for backward compatibility
    from .crypto.predict_price import PredictPrice
    from .crypto.token_holders import TokenHolders
    from .crypto.trading_history import TradingHistory
    from .crypto.uniswap_liquidity import UniswapLiquidity
    from .crypto.wallet_analysis import WalletAnalysis
    from .crypto.price_data import GetTokenPriceTool, Get24hStatsTool, GetKlineDataTool
    from .crypto.price_alerts import PriceThresholdAlertTool, LpRangeCheckTool, SuddenPriceIncreaseTool
    from .crypto.lending_rates import LendingRateMonitorTool
    # from .crypto.lst_arbitrage import LstArbitrageTool
    from .token_execute.token_transfer import TokenTransfer

__all__ = [
    "ToolManager", 
    "BaseTool", 
    "Terminate", 
    "PredictPrice", 
    "TokenHolders", 
    "TradingHistory", 
    "UniswapLiquidity", 
    "WalletAnalysis",
    "GetTokenPriceTool",
    "Get24hStatsTool",
    "GetKlineDataTool",
    "PriceThresholdAlertTool",
    "LpRangeCheckTool",
    "SuddenPriceIncreaseTool",
    "LendingRateMonitorTool",
    # Add LstArbitrageTool and TokenTransfer to __all__
    # "LstArbitrageTool",
    "TokenTransfer"
]