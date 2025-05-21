import traceback
from typing import List, Optional

from spoon_ai.tools.base import BaseTool
import os
import requests


class GetContractEventsFromThirdwebInsight(BaseTool):
    name: str = "get_contract_events_from_thirdweb_insight"
    description: str = "Fetch contract events with specific signature using Thirdweb Insight API"
    parameters: str = {
        "type": "object",
        "properties": {
            "client_id": {"type": "string"},
            "chain_id": {"type": "integer"},
            "contract_address": {"type": "string"},
            "event_signature": {"type": "string"},
            "limit": {"type": "integer", "default": 10}
        },
        "required": ["client_id", "chain_id", "contract_address", "event_signature"]
    }

    async def execute(
            self,
            client_id: str,
            chain_id: int,
            contract_address: str,
            event_signature: str,
            limit: int = 10
    ) -> str:
        try:
            base_url = f"https://{chain_id}.insight.thirdweb.com/v1"
            url = f"{base_url}/events/{contract_address}/{event_signature}"
            headers = {"x-client-id": client_id}
            params = {"limit": limit}
            res = requests.get(url, headers=headers, params=params)
            res.raise_for_status()
            data = res.json()
            return f"✅ Success. Found {len(data)} events.\n{data}"
        except Exception as e:
            return f"❌ Failed to fetch events: {e}"


class GetMultichainTransfersFromThirdwebInsight(BaseTool):
    name: str = "get_multichain_transfers_from_thirdweb_insight"
    description: str = "Query recent USDT transfers across multiple chains using Thirdweb Insight"
    parameters: str = {
        "type": "object",
        "properties": {
            "chains": {
                "type": "array",
                "items": {"type": "integer"},
                "description": "List of EVM chain IDs (e.g. [1, 137])"
            },
            "limit": {
                "type": "integer",
                "description": "Number of transfer events to retrieve (default: 10)"
            }
        },
        "required": ["chains"]
    }

    async def execute(self, chains: List[int], limit: int = 10) -> dict:
        try:
            client_id = os.getenv("THIRDWEB_CLIENT_ID")
            if not client_id:
                raise ValueError("Missing THIRDWEB_CLIENT_ID in environment variables!")

            chain_params = "&".join([f"chain={chain}" for chain in chains])
            url = f"https://insight.thirdweb.com/v1/events?{chain_params}&limit={limit}"
            headers = {"x-client-id": client_id}

            response = requests.get(url, headers=headers)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {"error": str(e)}


class GetTransactionsTool(BaseTool):
    name: str = "get_transactions"
    description: str = "Retrieve recent transactions across multiple chains using Thirdweb Insight API"
    parameters: str = {
        "type": "object",
        "properties": {
            "chains": {
                "type": "array",
                "items": {"type": "integer"},
                "description": "List of EVM chain IDs (e.g., [1, 137])"
            },
            "limit": {
                "type": "integer",
                "description": "Number of transactions to retrieve (default: 10)"
            }
        },
        "required": ["chains"]
    }

    async def execute(self, chains: List[int], limit: int = 10) -> dict:
        try:
            client_id = os.getenv("THIRDWEB_CLIENT_ID")
            if not client_id:
                raise ValueError("Missing THIRDWEB_CLIENT_ID in environment variables!")

            chain_params = "&".join([f"chain={chain}" for chain in chains])
            url = f"https://insight.thirdweb.com/v1/transactions?{chain_params}&limit={limit}"
            headers = {"x-client-id": client_id}

            response = requests.get(url, headers=headers)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {"error": str(e)}


class GetContractTransactionsTool(BaseTool):
    name: str = "get_contract_transactions"
    description: str = "Retrieve transactions for a specific contract using Thirdweb Insight API"
    parameters: str = {
        "type": "object",
        "properties": {
            "contract_address": {
                "type": "string",
                "description": "The contract address to query transactions for"
            },
            "chain": {
                "type": "integer",
                "description": "EVM chain ID (e.g., 1 for Ethereum)"
            },
            "limit": {
                "type": "integer",
                "description": "Number of transactions to retrieve (default: 10)"
            }
        },
        "required": ["contract_address", "chain"]
    }

    async def execute(self, contract_address: str, chain: int, limit: int = 10) -> dict:
        try:
            client_id = os.getenv("THIRDWEB_CLIENT_ID")
            if not client_id:
                raise ValueError("Missing THIRDWEB_CLIENT_ID in environment variables!")

            url = f"https://insight.thirdweb.com/v1/transactions/{contract_address}?chain={chain}&limit={limit}"
            headers = {"x-client-id": client_id}

            response = requests.get(url, headers=headers)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {"error": traceback.format_exc()}


class GetContractTransactionsBySignatureTool(BaseTool):
    name: str = "get_contract_transactions_by_signature"
    description: str = "Retrieve transactions for a specific contract and function signature using Thirdweb Insight API"
    parameters: str = {
        "type": "object",
        "properties": {
            "contract_address": {
                "type": "string",
                "description": "The contract address to query transactions for"
            },
            "signature": {
                "type": "string",
                "description": "The function signature to filter transactions by (e.g., 'transfer(address,uint256)')"
            },
            "chain": {
                "type": "integer",
                "description": "EVM chain ID (e.g., 1 for Ethereum)"
            },
            "limit": {
                "type": "integer",
                "description": "Number of transactions to retrieve (default: 10)"
            }
        },
        "required": ["contract_address", "signature", "chain"]
    }

    async def execute(self, contract_address: str, signature: str, chain: int, limit: int = 10) -> dict:
        try:
            client_id = os.getenv("THIRDWEB_CLIENT_ID")
            if not client_id:
                raise ValueError("Missing THIRDWEB_CLIENT_ID in environment variables!")

            url = f"https://insight.thirdweb.com/v1/transactions/{contract_address}/{signature}?chain={chain}&limit={limit}"
            headers = {"x-client-id": client_id}

            response = requests.get(url, headers=headers)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {"error": str(e)}


class GetBlocksFromThirdwebInsight(BaseTool):
    name: str = "get_blocks_from_thirdweb_insight"
    description: str = "Fetch block data from Thirdweb Insight API with optional sorting"
    parameters: str = {
        "type": "object",
        "properties": {
            "client_id": {"type": "string"},
            "chains": {
                "type": "array",
                "items": {"type": "integer"},
                "description": "List of chain IDs to query"
            },
            "limit": {"type": "integer", "default": 10},
            "sort_by": {
                "type": "string",
                "enum": ["block_number", "block_timestamp"],
                "default": "block_number",
                "description": "Field to sort results by"
            },
            "sort_order": {
                "type": "string",
                "enum": ["asc", "desc"],
                "default": "desc",
                "description": "Sort order"
            }
        },
        "required": ["client_id", "chains"]
    }

    async def execute(
        self,
        client_id: str,
        chains: List[int],
        limit: int = 10,
        sort_by: str = "block_number",
        sort_order: str = "desc"
    ) -> str:
        try:
            base_url = "https://insight.thirdweb.com/v1/blocks"
            headers = {"x-client-id": client_id}
            params = [("chain", str(chain)) for chain in chains]
            params.append(("limit", str(limit)))
            params.append(("sort_by", sort_by))
            params.append(("sort_order", sort_order))

            response = requests.get(base_url, headers=headers, params=params)
            response.raise_for_status()
            data = response.json()
            return f"✅ Success. Retrieved {len(data.get('data', []))} blocks.\n{data}"
        except Exception as e:
            return f"❌ Failed to fetch blocks: {e}"


class GetWalletTransactionsFromThirdwebInsight(BaseTool):
    name: str = "get_wallet_transactions_from_thirdweb_insight"
    description: str = "Fetch transactions for a specific wallet address using Thirdweb Insight API"
    parameters: str = {
        "type": "object",
        "properties": {
            "client_id": {"type": "string"},
            "wallet_address": {"type": "string"},
            "chains": {
                "type": "array",
                "items": {"type": "integer"},
                "description": "List of chain IDs to query"
            },
            "limit": {"type": "integer", "default": 10},
            "sort_by": {
                "type": "string",
                "enum": ["block_number", "block_timestamp"],
                "description": "Field to sort results by"
            },
            "sort_order": {
                "type": "string",
                "enum": ["asc", "desc"],
                "description": "Sort order"
            }
        },
        "required": ["client_id", "wallet_address", "chains"]
    }

    async def execute(
        self,
        client_id: str,
        wallet_address: str,
        chains: List[int],
        limit: int = 10,
        sort_by: Optional[str] = "block_number",
        sort_order: Optional[str] = "desc"
    ) -> str:
        try:
            base_url = f"https://insight.thirdweb.com/v1/wallets/{wallet_address}/transactions"
            headers = {"x-client-id": client_id}
            params = [("chain", str(chain)) for chain in chains]
            params.append(("limit", str(limit)))
            params.append(("sort_by", sort_by))
            params.append(("sort_order", sort_order))

            response = requests.get(base_url, headers=headers, params=params)
            response.raise_for_status()
            data = response.json()
            return f"✅ Success. Retrieved {len(data.get('data', []))} transactions.\n{data}"
        except Exception as e:
            return f"❌ Failed to fetch transactions: {e}"


async def test_get_contract_events():
    client_id = os.getenv("THIRDWEB_CLIENT_ID")
    chain_id = 1
    contract_address = "0xdAC17F958D2ee523a2206206994597C13D831ec7"
    event_signature = "Transfer(address,address,uint256)"

    tool = GetContractEventsFromThirdwebInsight()
    result = await tool.execute(
        client_id=client_id,
        chain_id=chain_id,
        contract_address=contract_address,
        event_signature=event_signature,
        limit=5
    )
    print("🧪 Get Contract Events Result:\n", result)


async def test_get_multichain_transfers():
    tool = GetMultichainTransfersFromThirdwebInsight()
    result = await tool.execute(chains=[1, 137], limit=5)
    print("🧪 Multichain Transfers Result:\n", result)


async def test_get_transactions():
    tool = GetTransactionsTool()
    result = await tool.execute(chains=[1, 137], limit=5)
    print("🧪 Get Transactions Result:\n", result)


async def test_get_contract_transactions():
    tool = GetContractTransactionsTool()
    result = await tool.execute(contract_address="0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2", chain=1, limit=5)
    print("🧪 Get Contract Transactions Result:\n", result)


async def test_get_contract_transactions_by_signature():
    tool = GetContractTransactionsBySignatureTool()
    result = await tool.execute(contract_address="0xA8D1eE203cbf39Dd0345398A6F75E7586e0Dd115", signature="transfer(address,uint256)",
                                chain=1, limit=5)
    print("🧪 Get Contract Transactions by Signature Result:\n", result)


if __name__ == '__main__':
    import asyncio

    async def run_all_tests():
        # await test_get_contract_events()
        # await test_get_multichain_transfers()
        # await test_get_transactions()
        await test_get_contract_transactions()
        # await test_get_contract_transactions_by_signature()


    asyncio.run(run_all_tests())
