from fastmcp import FastMCP
from spoon_ai.tools.gopluslabs.cache import time_cache
from spoon_ai.tools.gopluslabs.supported_chains import chain_name_to_id
from spoon_ai.tools.gopluslabs.http_client import go_plus_labs_client_v1, go_plus_labs_client_v2
from spoon_ai.tools.gopluslabs.utils import normalize_ethereum_contract_address

mcp = FastMCP("ApprovalSecurity")

@mcp.tool()
@time_cache()
async def check_approval_security(contract_address: str, chain_name: str) -> dict:
    """
    Check if an approval of transfer on a blockchain is malicious.
    If any response value is True, the address can be malicious for that response key.
    In the response, a string "1" could mean True, and "0" could mean False.
      {
        "contract_name": "string",
        "creator_address": "string",
        "deployed_time": 0,
        "doubt_list": 0,
        "is_contract": 0,
        "is_open_source": 0,
        "is_proxy": 0,
        "malicious_behavior": [
          "string"
        ],
        "tag": "string",
        "trust_list": 0
      }
    """
    contract_address = normalize_ethereum_contract_address(contract_address)
    chain_id = await chain_name_to_id(chain_name) if chain_name else ''
    r = await go_plus_labs_client_v1.get(f'/approval_security/{chain_id}?contract_addresses={contract_address}')
    r = r.json()
    r = r["result"]
    return r

@mcp.tool()
@time_cache()
async def erc20_approval_security(contract_address: str, chain_name: str) -> list:
    """
    Reports the outstanding token approvals issued to ERC-20 contracts by the given EOA address and associated risk items, including the date that the approval was issued, the allowance of the approval, and the transaction ID issuing the allowance.
    If any response value is True, the address can be malicious for that response key.
    In the response, a string "1" could mean True, and "0" could mean False.
    [
        {
          "approved_list": [
            {
              "address_info": {
                "contract_name": "string",
                "creator_address": "string",
                "deployed_time": 0,
                "doubt_list": 0,
                "is_contract": 0,
                "is_open_source": 0,
                "malicious_behavior": [
                  "string"
                ],
                "tag": "string",
                "trust_list": 0
              },
              "approved_amount": "string",
              "approved_contract": "string",
              "approved_time": 0,
              "hash": "string",
              "initial_approval_hash": "string",
              "initial_approval_time": 0
            }
          ],
          "balance": "string",
          "chain_id": "string",
          "decimals": 0,
          "is_open_source": 0,
          "malicious_address": 0,
          "malicious_behavior": [
            "string"
          ],
          "token_address": "string",
          "token_name": "string",
          "token_symbol": "string"
        }
    ]
    """
    contract_address = normalize_ethereum_contract_address(contract_address)
    chain_id = await chain_name_to_id(chain_name) if chain_name else ''
    r = await go_plus_labs_client_v2.get(f'/token_approval_security/{chain_id}?addresses={contract_address}')
    r = r.json()
    r = r["result"]
    return r

@mcp.tool()
@time_cache()
async def erc721_approval_security(contract_address: str, chain_name: str) -> list:
    """
    Reports the outstanding token approvals issued to ERC-721 contracts by the given EOA address and associated risk items, including the date that the approval was issued, the allowance of the approval, and the transaction ID issuing the allowance.
    If any response value is True, the address can be malicious for that response key.
    In the response, a string "1" could mean True, and "0" could mean False.
    [
        {
          "approved_list": [
            {
              "address_info": {
                "contract_name": "string",
                "creator_address": "string",
                "deployed_time": 0,
                "doubt_list": 0,
                "is_contract": 0,
                "is_open_source": 0,
                "malicious_behavior": [
                  "string"
                ],
                "tag": "string",
                "trust_list": 0
              },
              "approved_contract": "string",
              "approved_for_all": 0,
              "approved_time": 0,
              "approved_token_id": "string",
              "hash": "string",
              "initial_approval_hash": "string",
              "initial_approval_time": 0
            }
          ],
          "chain_id": "string",
          "is_open_source": 0,
          "is_verified": 0,
          "malicious_address": 0,
          "malicious_behavior": [
            "string"
          ],
          "nft_address": "string",
          "nft_name": "string",
          "nft_symbol": "string"
        }
    ]
    """
    contract_address = normalize_ethereum_contract_address(contract_address)
    chain_id = await chain_name_to_id(chain_name) if chain_name else ''
    r = await go_plus_labs_client_v2.get(f'/nft721_approval_security/{chain_id}?addresses={contract_address}')
    r = r.json()
    r = r["result"]
    return r

@mcp.tool()
@time_cache()
async def erc1155_approval_security(contract_address: str, chain_name: str) -> list:
    """
    Reports the outstanding token approvals issued to ERC-1155 contracts by the given EOA address and associated risk items, including the date that the approval was issued, the allowance of the approval, and the transaction ID issuing the allowance.
    If any response value is True, the address can be malicious for that response key.
    In the response, a string "1" could mean True, and "0" could mean False.
    [
        {
          "approved_list": [
            {
              "address_info": {
                "contract_name": "string",
                "creator_address": "string",
                "deployed_time": 0,
                "doubt_list": 0,
                "is_contract": 0,
                "is_open_source": 0,
                "malicious_behavior": [
                  "string"
                ],
                "tag": "string",
                "trust_list": 0
              },
              "approved_contract": "string",
              "approved_time": 0,
              "hash": "string",
              "initial_approval_hash": "string",
              "initial_approval_time": 0
            }
          ],
          "chain_id": "string",
          "is_open_source": 0,
          "is_verified": 0,
          "malicious_address": 0,
          "malicious_behavior": [
            "string"
          ],
          "nft_address": "string",
          "nft_name": "string",
          "nft_symbol": "string"
        }
    ]
    """
    contract_address = normalize_ethereum_contract_address(contract_address)
    chain_id = await chain_name_to_id(chain_name) if chain_name else ''
    r = await go_plus_labs_client_v2.get(f'/nft1155_approval_security/{chain_id}?addresses={contract_address}')
    r = r.json()
    r = r["result"]
    return r
