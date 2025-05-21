from typing import Any
from fastmcp import FastMCP
from spoon_ai.tools.gopluslabs.cache import time_cache
import string
from spoon_ai.tools.gopluslabs.supported_chains import chain_name_to_id
from spoon_ai.tools.gopluslabs.http_client import go_plus_labs_client_v1
from spoon_ai.tools.gopluslabs.utils import normalize_ethereum_contract_address

mcp = FastMCP("TokenSecurity")

@mcp.tool()
@time_cache()
async def check_malicious_address(contract_address: str, chain_name: str = '') -> dict:
    """
    Check if an address on a blockchain is malicious.
    You may also specify empty chain name to query the address on all chains.
    If any response value is True, the address can be malicious for that response key.
    In the response, a string "1" could mean True, and "0" could mean False.
    {
    "blacklist_doubt": "string",
    "blackmail_activities": "string",
    "contract_address": "string",
    "cybercrime": "string",
    "darkweb_transactions": "string",
    "data_source": "string",
    "fake_kyc": "string",
    "fake_standard_interface": "string",
    "fake_token": "string",
    "financial_crime": "string",
    "gas_abuse": "string",
    "honeypot_related_address": "string",
    "malicious_mining_activities": "string",
    "mixer": "string",
    "money_laundering": "string",
    "number_of_malicious_contracts_created": "string",
    "phishing_activities": "string",
    "reinit": "string",
    "sanctioned": "string",
    "stealing_attack": "string"
    }
    """
    contract_address = normalize_ethereum_contract_address(contract_address)
    chain_id = await chain_name_to_id(chain_name) if chain_name else ''
    if chain_id:
        r = await go_plus_labs_client_v1.get(f'/address_security/{contract_address}?chain_id={chain_id}')
    else:
        r = await go_plus_labs_client_v1.get(f'/address_security/{contract_address}')
    r = r.json()
    r = r["result"]
    return r