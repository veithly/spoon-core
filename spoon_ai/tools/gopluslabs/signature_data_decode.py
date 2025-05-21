# from typing import Any
from fastmcp import FastMCP
from spoon_ai.tools.gopluslabs.cache import time_cache
from spoon_ai.tools.gopluslabs.supported_chains import chain_name_to_id
from spoon_ai.tools.gopluslabs.http_client import go_plus_labs_client_v1
from spoon_ai.tools.gopluslabs.utils import normalize_ethereum_contract_address

mcp = FastMCP("AbiDecode")

@mcp.tool()
@time_cache()
async def get_abi_decode_info(chain_name: str, data: str, contract_address: str = '') -> dict:
    """
    Get the result of abi.decode in Ethereum. The input data argument looks like 0xa9059cbb00000000000000000000000055d398326f99059ff775485246999027b319795500000000000000000000000000000000000000000000000acc749097d9d00000
    """
    raise NotImplementedError("I do not know how the input args of this API work. See https://docs.gopluslabs.io/reference/getabidatainfousingpost")

