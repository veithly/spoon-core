import os

from pydantic import Field

from spoon_ai.trade.aggregator import Aggregator

from .base import TokenExecuteBaseTool


class TokenTransfer(TokenExecuteBaseTool):
    name: str = "token_transfer"
    description: str = "Transfer a token"
    parameters: dict = {
        "type": "object",
        "properties": {
            "token_address": {
                "type": "string",
                "description": "The address of the token"
            },
            "amount": {
                "type": "string",
                "description": "The amount of tokens to transfer"
            },
            "to": {
                "type": "string",
                "description": "The address of the recipient"
            }
        },
        "required": ["token_address", "amount", "to"]
    }
    aggregator: Aggregator = Field(default_factory=lambda: Aggregator(network="ethereum", rpc_url=os.getenv("RPC_URL"), scan_url=os.getenv("SCAN_URL"), chain_id=os.getenv("CHAIN_ID")))
    
    async def execute(self, token_address: str, amount: str, to: str) -> str:
        amount = float(amount)
        tx_hash = self.aggregator.transfer(to, amount, token_address)
        return tx_hash