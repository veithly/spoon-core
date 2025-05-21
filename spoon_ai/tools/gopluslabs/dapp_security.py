# from typing import Any
from fastmcp import FastMCP
from spoon_ai.tools.gopluslabs.cache import time_cache
from spoon_ai.tools.gopluslabs.http_client import go_plus_labs_client_v1

mcp = FastMCP("dAppSecurity")

@mcp.tool()
@time_cache()
async def get_nft_security(url: str) -> dict:
    """
    Get risk of dApp by URL. e.g. url = https://www.0x.org
    {
        "audit_info": [
          {
            "audit_firm": "string",
            "audit_link": "string",
            "audit_time": "string"
          }
        ],
        "contracts_security": [
          {
            "chain_id": "string",
            "contracts": [
              {
                "contract_address": "string",
                "creator_address": "string",
                "deployment_time": 0,
                "is_open_source": 0,
                "malicious_behavior": [
                  {}
                ],
                "malicious_contract": 0,
                "malicious_creator": 0,
                "malicious_creator_behavior": [
                  {}
                ]
              }
            ]
          }
        ],
        "is_audit": 0,
        "project_name": "string",
        "trust_list": 0,
        "url": "string"
    }
    """
    r = await go_plus_labs_client_v1.get(f'/dapp_security/?url={url}')
    r = r.json()
    r = r["result"]
    return r