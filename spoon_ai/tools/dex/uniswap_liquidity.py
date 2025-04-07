from .base import BitqueryTool

class UniswapLiquidity(BitqueryTool):
    name: str = "uniswap_liquidity"
    description: str = "Returns the liquidity of a token"
    parameters: dict = {
        "type": "object",
        "properties": {
            "token_address": {"type": "string", "description": "The address of the token"}
        },
        "required": ["token_address"]
    }
    
    
    
    
    
    graph_template: str = """
query UniswapLiquidityQuery {{
  EVM(dataset: archive) {{
    Events(
      orderBy: {{descending: Block_Time}}
      limit: {{count: 10}}
      where: {{Log: {{Signature: {{Name: {{in: ["Burn", "Mint"]}}}}}}, LogHeader: {{Address: {{is: "{token_address}"}}}}}}
    ) {{
      Transaction {{
        Hash
      }}
      Log {{
        Signature {{
          Name
        }}
      }}
      Arguments {{
        Name
        Value {{
          ... on EVM_ABI_Integer_Value_Arg {{
            integer
          }}
          ... on EVM_ABI_String_Value_Arg {{
            string
          }}
          ... on EVM_ABI_Address_Value_Arg {{
            address
          }}
          ... on EVM_ABI_BigInt_Value_Arg {{
            bigInteger
          }}
          ... on EVM_ABI_Bytes_Value_Arg {{
            hex
          }}
          ... on EVM_ABI_Boolean_Value_Arg {{
            bool
          }}
        }}
      }}
    }}
  }}
}}
"""
    async def execute(self, token_address: str) -> str:
        response = await super().execute(token_address=token_address)
        return response['data']['EVM']['Events']
