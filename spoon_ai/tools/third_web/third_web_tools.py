from typing import List

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


class CallContractViaThirdwebEngine(BaseTool):
    name: str = "call_contract_via_thirdweb_engine"
    description: str = "Call a contract method via Thirdweb Engine API"
    parameters: str = {
        "type": "object",
        "properties": {
            "secret_key": {"type": "string"},
            "vault_token": {"type": "string"},
            "server_wallet": {"type": "string"},
            "chain_id": {"type": "integer"},
            "contract_address": {"type": "string"},
            "method_signature": {"type": "string"},
            "method_params": {
                "type": "array",
                "items": {"type": "string"}
            }
        },
        "required": [
            "secret_key", "vault_token", "server_wallet",
            "chain_id", "contract_address", "method_signature", "method_params"
        ]
    }

    async def execute(
        self,
        secret_key: str,
        vault_token: str,
        server_wallet: str,
        chain_id: int,
        contract_address: str,
        method_signature: str,
        method_params: list
    ) -> str:
        try:
            url = "https://engine.thirdweb.com/v1/write/contract"
            headers = {
                "x-secret-key": secret_key,
                "x-vault-access-token": vault_token,
                "Content-Type": "application/json"
            }
            payload = {
                "executionOptions": {
                    "from": server_wallet,
                    "chainId": str(chain_id)
                },
                "params": [
                    {
                        "contractAddress": contract_address,
                        "method": method_signature,
                        "params": method_params
                    }
                ]
            }
            res = requests.post(url, headers=headers, json=payload)
            res.raise_for_status()
            return f"✅ Contract call successful:\n{res.json()}"
        except Exception as e:
            return f"❌ Contract call failed: {e}"


async def test_get_contract_events():
    client_id = os.getenv("THIRDWEB_CLIENT_ID")
    chain_id = 1  # 例如 Ethereum Mainnet
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


async def test_call_contract_method():
    secret_key = os.getenv("THIRDWEB_SECRET_KEY")
    vault_token = os.getenv("THIRDWEB_VAULT_TOKEN")
    server_wallet = os.getenv("THIRDWEB_SERVER_WALLET")
    chain_id = 80001
    contract_address = "0xabcdefabcdefabcdefabcdefabcdefabcdef"
    method_signature = "mintTo(address,uint256)"
    method_params = ["0xrecipientAddressHere", "1"]

    tool = CallContractViaThirdwebEngine()
    result = await tool.execute(
        secret_key=secret_key,
        vault_token=vault_token,
        server_wallet=server_wallet,
        chain_id=chain_id,
        contract_address=contract_address,
        method_signature=method_signature,
        method_params=method_params
    )
    print("🧪 Call Contract Method Result:\n", result)


if __name__ == '__main__':
    import asyncio

    async def run_all_tests():
        await test_get_contract_events()
        await test_get_multichain_transfers()
        # await test_call_contract_method()

    asyncio.run(run_all_tests())
