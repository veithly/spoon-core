from .base import BitqueryTool

class TradingHistory(BitqueryTool):
    name: str = "trading_history"
    description: str = "Returns the trading history of a wallet"
    parameters: dict = {
        "type": "object",
        "properties": {
            "wallet_address": {
                "type": "string",
                "description": "The address of the wallet"
            }
        },
        "required": ["wallet_address"]
    }
    graph_template: str = """
query TradingHistoryQuery {{
  EVM(dataset: combined, network: eth) {{
    buyside: DEXTrades(
      limit: {{count: 30}}
      orderBy: {{descending: Block_Time}}
      where: {{Trade: {{Buy: {{Buyer: {{is: "{wallet_address}"}}}}}}}}
    ) {{
      Block {{
        Time
      }}
      Transaction {{
        Hash
      }}
      Trade {{
        Buy {{
          Amount
          Buyer
          Currency {{
            Name
            Symbol
            SmartContract
          }}
          Seller
          Price
        }}
        Sell {{
          Amount
          Buyer
          Currency {{
            Name
            SmartContract
            Symbol
          }}
          Seller
          Price
        }}
      }}
    }}
    sellside: DEXTrades(
      limit: {{count: 30}}
      orderBy: {{descending: Block_Time}}
      where: {{Trade: {{Buy: {{Seller: {{is: "{wallet_address}"}}}}}}}}
    ) {{
      Block {{
        Time
      }}
      Transaction {{
        Hash
      }}
      Trade {{
        Buy {{
          Amount
          Buyer
          Currency {{
            Name
            Symbol
            SmartContract
          }}
          Seller
          Price
        }}
        Sell {{
          Amount
          Buyer
          Currency {{
            Name
            SmartContract
            Symbol
          }}
          Seller
          Price
        }}
      }}
    }}
  }}
}}
"""
    async def execute(self, wallet_address: str) -> str:
        response = await super().execute(wallet_address=wallet_address)
        print(response)
        return response['data']['EVM']['buyside']