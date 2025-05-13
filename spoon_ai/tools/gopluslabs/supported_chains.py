import asyncio
from fastmcp import FastMCP
from spoon_ai.tools.gopluslabs.cache import time_cache
from spoon_ai.tools.gopluslabs.http_client import go_plus_labs_client_v1

mcp = FastMCP("SupportedChains")

@time_cache()
async def get_supported_chains_inner():
    r = await go_plus_labs_client_v1.get('/supported_chains')
    r = r.json()
    """
    r_example = {
        "code": 0,
        "message": "string",
        "result": [
            {
                "id": "1",
                "name": "Ethereum"
            },
            ...
        ]
    }
    """
    return r

@mcp.resource(uri='resource://SupportedChains')
async def supported_chains() -> list[str]:
    """Get the supported blockchains of this MCP server"""
    r = await get_supported_chains_inner()
    names = [c["name"] for c in r["result"]]  # ["Ethereum", "BSC", ...]
    return names

def test_supported_chains():
    names = asyncio.run(supported_chains())
    assert 'Ethereum' in names

async def chain_name_to_id(name: str) -> str:
    r = await get_supported_chains_inner()
    for c in r["result"]:
        if c["name"].lower() == name.lower():
            return c["id"]
    raise NotImplementedError(f"Chain {name} not supported.")

"""
id	name
1	Ethereum
56	BSC
42161	Arbitrum
137	Polygon
324	zkSync Era
59144	Linea Mainnet
8453	Base
534352	Scroll
10	Optimism
43114	Avalanche
250	Fantom
25	Cronos
66	OKC
128	HECO
100	Gnosis
10001	ETHW
tron	Tron
321	KCC
201022	FON
5000	Mantle
204	opBNB
42766	ZKFair
81457	Blast
169	Manta Pacific
80094	Berachain
2741	Abstract
177	Hashkey Chain
146	Sonic
1514	Story
"""