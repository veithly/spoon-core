from spoon_ai.tools.dex.wallet_analysis import WalletAnalysis
from spoon_ai.tools.dex.token_holders import TokenHolders
from spoon_ai.tools.dex.trading_history import TradingHistory

if __name__ == "__main__":
    import asyncio
    asyncio.run(TradingHistory().execute("0xCf512E9097B417b8Dd1b2e664aa7ceA92AEf7221"))