# from typing import Any
from fastmcp import FastMCP
from spoon_ai.tools.gopluslabs.cache import time_cache
from spoon_ai.tools.gopluslabs.supported_chains import chain_name_to_id
from spoon_ai.tools.gopluslabs.http_client import go_plus_labs_client_v1
from spoon_ai.tools.gopluslabs.utils import normalize_ethereum_contract_address

mcp = FastMCP("NftSecurity")

@mcp.tool()
@time_cache()
async def get_nft_security(chain_name: str, contract_address: str, token_id: str = '') -> dict:
    """
    Get NFT security information.
    {
        "average_price_24h": 0,
        "create_block_number": 0,
        "creator_address": "string",
        "discord_url": "string",
        "github_url": "string",
        "highest_price": 0,
        "lowest_price_24h": 0,
        "malicious_nft_contract": 0,
        "medium_url": "string",
        "metadata_frozen": 0,
        "nft_description": "string",
        "nft_erc": "string",
        "nft_items": 0,
        "nft_name": "string",
        "nft_open_source": 0,
        "nft_owner_number": 0,
        "nft_proxy": 0,
        "nft_symbol": "string",
        "nft_verified": 0,
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
        "red_check_mark": 0,
        "restricted_approval": 0,
        "sales_24h": 0,
        "same_nfts": [
          {
            "create_block_number": 0,
            "nft_address": "string",
            "nft_name": "string",
            "nft_owner_number": 0,
            "nft_symbol": "string"
          }
        ],
        "self_destruct": {
          "owner_address": "string",
          "owner_type": "string",
          "value": 0
        },
        "telegram_url": "string",
        "token_id": "string",
        "token_owner": "string",
        "total_volume": 0,
        "traded_volume_24h": 0,
        "transfer_without_approval": {
          "owner_address": "string",
          "owner_type": "string",
          "value": 0
        },
        "trust_list": 0,
        "twitter_url": "string",
        "website_url": "string"
    }
    """
    contract_address = normalize_ethereum_contract_address(contract_address)
    chain_id = await chain_name_to_id(chain_name)
    if token_id:
        r = await go_plus_labs_client_v1.get(f'/nft_security/{chain_id}?contract_addresses={contract_address}&token_id={token_id}')
    else:
        r = await go_plus_labs_client_v1.get(f'/nft_security/{chain_id}?contract_addresses={contract_address}')
    r = r.json()
    r = r["result"]
    return r