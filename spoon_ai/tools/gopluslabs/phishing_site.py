# from typing import Any
from fastmcp import FastMCP
from spoon_ai.tools.gopluslabs.cache import time_cache
from spoon_ai.tools.gopluslabs.supported_chains import chain_name_to_id
from spoon_ai.tools.gopluslabs.http_client import go_plus_labs_client_v1
from spoon_ai.tools.gopluslabs.utils import normalize_ethereum_contract_address

mcp = FastMCP("PhishingSite")

@mcp.tool()
@time_cache()
async def get_nft_security(url: str) -> dict:
    """
    Check if the URL is a phishing site. e.g. url = go-ethdenver.com
    In the response, a string "1" could mean True, and "0" could mean False.
    {
        "phishing_site": 0,
        "website_contract_security": [
          {
            "address_risk": [
              "string"
            ],
            "contract": "string",
            "is_contract": 0,
            "is_open_source": 0,
            "nft_risk": {
              "nft_open_source": 0,
              "nft_proxy": 0,
              "oversupply_minting": 0,
              "privileged_burn": {
                "owner_address": "string",
                "owner_type": "string",
                "value": 0
              },
              "privileged_minting": {
                "owner_address": "string",
                "owner_type": "string",
                "value": 0
              },
              "restricted_approval": 0,
              "self_destruct": {
                "owner_address": "string",
                "owner_type": "string",
                "value": 0
              },
              "transfer_without_approval": {
                "owner_address": "string",
                "owner_type": "string",
                "value": 0
              }
            },
            "standard": "string"
          }
        ]
    }
    """
    r = await go_plus_labs_client_v1.get(f'/phishing_site/?url={url}')
    r = r.json()
    r = r["result"]
    return r