# from typing import Any
import re
from fastmcp import FastMCP
from spoon_ai.tools.gopluslabs.cache import time_cache
import string
from spoon_ai.tools.gopluslabs.supported_chains import chain_name_to_id
from spoon_ai.tools.gopluslabs.http_client import go_plus_labs_client_v1
from spoon_ai.tools.gopluslabs.utils import normalize_ethereum_contract_address

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
    contract_address = normalize_ethereum_contract_address(contract_address)
    chain_id = await chain_name_to_id(chain_name)
    r = await go_plus_labs_client_v1.get(f'/token_security/{chain_id}?contract_addresses={contract_address}')
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

@mcp.tool()
@time_cache()
async def get_token_security_for_solana(contract_address: str):
    """
    Get token security info for Solana blockchain. The contract address should be a Solana address like HZ1JovNiVvGrGNiiYvEozEVgZ58xaU3RKwX8eACQBCt3
    {
        "additionalProp": {
          "balance_mutable_authority": {
            "authority": [
              {
                "address": "string",
                "malicious_address": 0
              }
            ],
            "status": "string"
          },
          "closable": {
            "authority": [
              {
                "address": "string",
                "malicious_address": 0
              }
            ],
            "status": "string"
          },
          "creator": [
            {
              "address": "string",
              "malicious_address": 0
            }
          ],
          "default_account_state": "string",
          "default_account_state_upgradable": {
            "authority": [
              {
                "address": "string",
                "malicious_address": 0
              }
            ],
            "status": "string"
          },
          "dex": [
            {
              "day": {
                "price_max": "string",
                "price_min": "string",
                "volume": "string"
              },
              "dex_name": "string",
              "fee_rate": "string",
              "id": "string",
              "lp_amount": "string",
              "month": {
                "price_max": "string",
                "price_min": "string",
                "volume": "string"
              },
              "open_time": "string",
              "price": "string",
              "tvl": "string",
              "type": "string",
              "week": {
                "price_max": "string",
                "price_min": "string",
                "volume": "string"
              }
            }
          ],
          "freezable": {
            "authority": [
              {
                "address": "string",
                "malicious_address": 0
              }
            ],
            "status": "string"
          },
          "holders": [
            {
              "balance": "string",
              "percent": "string",
              "tag": "string",
              "token_account": "string"
            }
          ],
          "lp_holders": [
            {
              "balance": "string",
              "percent": "string",
              "tag": "string",
              "token_account": "string"
            }
          ],
          "metadata": {
            "description": "string",
            "name": "string",
            "symbol": "string",
            "uri": "string"
          },
          "metadata_mutable": {
            "metadata_upgrade_authority": [
              {
                "address": "string",
                "malicious_address": 0
              }
            ],
            "status": "string"
          },
          "mintable": {
            "authority": [
              {
                "address": "string",
                "malicious_address": 0
              }
            ],
            "status": "string"
          },
          "none_transferable": "string",
          "transfer_fee": {
            "current_fee_rate": {
              "fee_rate": "string",
              "maximum_fee": "string"
            },
            "scheduled_fee_rate": [
              {
                "epoch": "string",
                "fee_rate": "string",
                "maximum_fee": "string"
              }
            ]
          },
          "transfer_fee_upgradable": {
            "authority": [
              {
                "address": "string",
                "malicious_address": 0
              }
            ],
            "status": "string"
          },
          "transfer_hook": [
            {
              "address": "string",
              "malicious_address": 0
            }
          ],
          "transfer_hook_upgradable": {
            "authority": [
              {
                "address": "string",
                "malicious_address": 0
              }
            ],
            "status": "string"
          },
          "trusted_token": "string"
        }
    }
    """
    if not re.fullmatch(r"^[1-9A-HJ-NP-Za-km-z]{44}$", contract_address):
        raise ValueError(f"Invalid Solana address {contract_address}")
    r = await go_plus_labs_client_v1.get(f'/solana/token_security?contract_addresses={contract_address}')
    r = r.json()
    r = r["result"]
    return r