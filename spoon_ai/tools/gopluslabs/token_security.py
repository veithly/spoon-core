# from typing import Any
from fastmcp import FastMCP
from spoon_ai.tools.gopluslabs.cache import time_cache
import string
from spoon_ai.tools.gopluslabs.supported_chains import chain_name_to_id
from spoon_ai.tools.gopluslabs.http_client import go_plus_labs_client

mcp = FastMCP("TokenSecurity")

@mcp.tool()
@time_cache()
async def get_token_risk_and_security_data(chain_name: str, contract_address: str) -> dict:
    """
    Get the risk and security data of a token contract address on a certain blockchain.
    In the response, a string "1" could mean True, and "0" could mean False.
      "additionalProp": {
      "anti_whale_modifiable": "string",
      "buy_tax": "string",
      "can_take_back_ownership": "string",
      "cannot_buy": "string",
      "cannot_sell_all": "string",
      "creator_address": "string",
      "creator_balance": "string",
      "creator_percent": "string",
      "dex": [
        {
          "liquidity": "string",
          "name": "string",
          "pair": "string"
        }
      ],
      "external_call": "string",
      "fake_token": {
        "true_token_address": "string",
        "value": 0
      },
      "hidden_owner": "string",
      "holder_count": "string",
      "holders": [
        {
          "address": "string",
          "balance": "string",
          "is_contract": 0,
          "is_locked": 0,
          "locked_detail": [
            {
              "amount": "string",
              "end_time": "string",
              "opt_time": "string"
            }
          ],
          "percent": "string",
          "tag": "string"
        }
      ],
      "honeypot_with_same_creator": "string",
      "is_airdrop_scam": "string",
      "is_anti_whale": "string",
      "is_blacklisted": "string",
      "is_honeypot": "string",
      "is_in_dex": "string",
      "is_mintable": "string",
      "is_open_source": "string",
      "is_proxy": "string",
      "is_true_token": "string",
      "is_whitelisted": "string",
      "lp_holder_count": "string",
      "lp_holders": [
        {
          "NFT_list": [
            {
              "NFT_id": "string",
              "NFT_percentage": "string",
              "amount": "string",
              "in_effect": "string",
              "value": "string"
            }
          ],
          "address": "string",
          "balance": "string",
          "is_contract": 0,
          "is_locked": 0,
          "locked_detail": [
            {
              "amount": "string",
              "end_time": "string",
              "opt_time": "string"
            }
          ],
          "percent": "string",
          "tag": "string"
        }
      ],
      "lp_total_supply": "string",
      "note": "string",
      "other_potential_risks": "string",
      "owner_address": "string",
      "owner_balance": "string",
      "owner_change_balance": "string",
      "owner_percent": "string",
      "personal_slippage_modifiable": "string",
      "selfdestruct": "string",
      "sell_tax": "string",
      "slippage_modifiable": "string",
      "token_name": "string",
      "token_symbol": "string",
      "total_supply": "string",
      "trading_cooldown": "string",
      "transfer_pausable": "string",
      "trust_list": "string"
    }
    """
    if not contract_address.startswith('0x'):
        contract_address = '0x' + contract_address
    if len(contract_address) != 42:
        raise ValueError(f'Invalid contract address {contract_address}. Length is not 42.')
    for c in contract_address[2:]:
        if not c in string.hexdigits:
            raise ValueError(f'Invalid contract address {contract_address}. Non hexadecimal char {c}.')
    chain_id = await chain_name_to_id(chain_name)
    r = await go_plus_labs_client.get(f'/token_security/{chain_id}?contract_addresses={contract_address}')
    r = r.json()
    r = r["result"]
    # for d in r:
    #     d: dict[str, Any]
    #     for k, v in d.items():
    #         if k.startswith("is_"):
    #             if v == "0":
    #                 d[k] = False
    #             if v == "1":
    #                 d[v] = True
    return r

