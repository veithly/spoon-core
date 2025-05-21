from fastmcp import FastMCP
from spoon_ai.tools.gopluslabs.cache import time_cache
from spoon_ai.tools.gopluslabs.supported_chains import chain_name_to_id
from spoon_ai.tools.gopluslabs.http_client import go_plus_labs_client_v1, go_plus_labs_client_v2
from spoon_ai.tools.gopluslabs.utils import normalize_ethereum_contract_address

mcp = FastMCP("RugPullDetection")

@mcp.tool()
@time_cache()
async def rug_pull_detection(contract_address: str, chain_name: str) -> dict:
    """
    Check if a contract has rug-pull risks.
    In the response, a string "1" could mean True, and "0" could mean False.
    {
        "approval_abuse": 0,
        "blacklist": 0,
        "contract_name": "string",
        "is_open_source": 0,
        "is_proxy": 0,
        "owner": {
          "owner_address": "string",
          "owner_name": "string",
          "owner_type": "string"
        },
        "privilege_withdraw": 0,
        "selfdestruct": 0,
        "withdraw_missing": 0
    }
    """
    contract_address = normalize_ethereum_contract_address(contract_address)
    chain_id = await chain_name_to_id(chain_name) if chain_name else ''
    r = await go_plus_labs_client_v1.get(f'/rugpull_detecting/{chain_id}?contract_addresses={contract_address}')
    r = r.json()
    r = r["result"]
    return r