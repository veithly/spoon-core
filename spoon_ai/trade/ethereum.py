class EthereumConfig:
    def __init__(self, rpc_url: str, chain_id: int, uniswap_v3_router_address: str, uniswap_v3_factory_address: str):
        self.rpc_url = rpc_url
        self.chain_id = chain_id
        self.uniswap_v3_router_address = uniswap_v3_router_address
        self.uniswap_v3_factory_address = uniswap_v3_factory_address
        
CHAINS = {
    "sepolia": EthereumConfig(
        rpc_url="https://rpc.sepolia.org",
        chain_id=11155111,
        uniswap_router_address="0x3bFA4769FB09eefC5a80d6E87c3B9C650f7Ae48E",
        uniswap_factory_address="0x0227628f3F023bb0B980b67D528571c95c6DaC1c"
    )
}