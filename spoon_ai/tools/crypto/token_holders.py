from .base import BitqueryTool
from datetime import datetime, timedelta

class TokenHolders(BitqueryTool):
    name: str = "token_holders"
    description: str = "Get the top token holders of a token"
    parameters: dict = {
        "type": "object",
        "properties": {
            "token_address": {
                "type": "string",
                "description": "The address of the token"
            },
            "limit": {
                "type": "integer",
                "description": "The number of token holders to return"
            }
        },
        "required": ["token_address", "limit"]
    }
    graph_template: str = """
query TokenHoldersQuery {{
  EVM(dataset: archive, network: eth) {{
    TokenHolders(
      date: "{since}"
      tokenSmartContract: "{token_address}"
      where: {{Balance: {{Amount: {{ge: "10000000"}}}}}}
      limit: {{count: {limit}}}
      orderBy: {{descending: Balance_Amount}}
    ) {{
      Balance {{
        Amount
      }}
      Holder {{
        Address
      }}
    }}
  }}
}}
    """
    
    async def execute(self, token_address: str, limit: int) -> str:
        since = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
        response = await super().execute(token_address=token_address, limit=limit, since=since)
        return response['data']['EVM']['TokenHolders']

