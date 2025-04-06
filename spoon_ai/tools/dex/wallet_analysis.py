from .base import BitqueryTool


class WalletAnalysis(BitqueryTool):
    name: str = "wallet_analysis"
    description: str = "Returns the balance updates of a wallet, which amount is greater than 100"
    parameters: dict = {
        "type": "object",
        "properties": {
            "wallet_address": {
                "type": "string",
                "description": "The address to analyze"
            }
        },
        "required": ["wallet_address"]
    }
    graph_template: str = """
query WalletAnalysisQuery {{
  EVM(dataset: combined, network: eth) {{
    BalanceUpdates(
      orderBy: {{descendingByField: "balance"}}
      where: {{BalanceUpdate: {{Address: {{is: "{wallet_address}"}}, Amount: {{ge: "100"}}}}}}
    ) {{
      Currency {{
        Name
      }}
      balance: sum(of: BalanceUpdate_Amount)
      BalanceUpdate {{
        Address
        AmountInUSD
      }}
    }}
  }}
}}
"""
    async def execute(self, wallet_address: str) -> str:
        response = await super().execute(wallet_address=wallet_address)
        print(response)
        return response['data']['EVM']['BalanceUpdates']