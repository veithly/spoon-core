from spoon_ai.tools.tool_manager import ToolManager
from spoon_ai.tools.dex.wallet_analysis import WalletAnalysis
from spoon_ai.tools.dex.token_holders import TokenHolders
from spoon_ai.tools.dex.trading_history import TradingHistory
from spoon_ai.tools.dex.uniswap_liquidity import UniswapLiquidity
from spoon_ai.tools.dex.predict_price import PredictPrice


if __name__ == "__main__":
    tool_manager = ToolManager([WalletAnalysis(), TokenHolders(), TradingHistory(), UniswapLiquidity(), PredictPrice()])
    tool_manager.index_tools()
    
    query = "xxxxx"
    
    print(tool_manager.query_tools(query, top_k=20, rerank_k=5))
    
    
    