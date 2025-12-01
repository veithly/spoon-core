"""
ERC-8004 Smart Contract Client
Handles on-chain interactions with agent registries
"""

import json
import os
from typing import List, Dict, Optional, Tuple, Union
from web3 import Web3
from web3.contract import Contract
from eth_account import Account
from eth_account.messages import encode_typed_data
from eth_utils import keccak, to_checksum_address
from eth_abi import encode as abi_encode


class ERC8004Client:
    """Client for interacting with ERC-8004 agent registries"""

    def __init__(
        self,
        rpc_url: str,
        agent_registry_address: str,
        identity_registry_address: str,
        reputation_registry_address: str,
        validation_registry_address: str,
        private_key: Optional[str] = None
    ):
        self.w3 = Web3(Web3.HTTPProvider(rpc_url))
        if not self.w3.is_connected():
            raise ConnectionError(f"Failed to connect to RPC: {rpc_url}")

        self.private_key = private_key
        if private_key:
            self.account = Account.from_key(private_key)
        else:
            self.account = None

        # Load contract ABIs
        self.agent_registry = self._load_contract(agent_registry_address, "SpoonAgentRegistry")
        self.identity_registry = self._load_contract(identity_registry_address, "ERC8004IdentityRegistry")
        self.reputation_registry = self._load_contract(reputation_registry_address, "ERC8004ReputationRegistry")
        self.validation_registry = self._load_contract(validation_registry_address, "ERC8004ValidationRegistry")

    def _load_contract(self, address: str, contract_name: str) -> Contract:
        """Load contract from ABI file"""
        abi_path = os.path.join(
            os.path.dirname(__file__),
            f"../../../contracts/erc8004/artifacts/contracts/erc8004/{contract_name}.sol/{contract_name}.json"
        )

        # If artifacts don't exist, use minimal ABI
        if not os.path.exists(abi_path):
            abi = self._get_minimal_abi(contract_name)
        else:
            with open(abi_path, 'r') as f:
                contract_json = json.load(f)
                abi = contract_json['abi']

        return self.w3.eth.contract(
            address=to_checksum_address(address),
            abi=abi
        )

    def _get_minimal_abi(self, contract_name: str) -> List[Dict]:
        """Minimal ABI for contract interaction (chaoschain-compatible)"""
        if contract_name == "SpoonAgentRegistry":
            return [
                {
                    "inputs": [{"type": "bytes32"}, {"type": "string"}, {"type": "string"}, {"type": "bytes"}],
                    "name": "registerAgent",
                    "outputs": [{"type": "bool"}],
                    "stateMutability": "nonpayable",
                    "type": "function"
                },
                {
                    "inputs": [{"type": "bytes32"}],
                    "name": "resolveAgent",
                    "outputs": [{"components": [
                        {"type": "address[]"}, {"type": "string"}, {"type": "string"},
                        {"type": "string[]"}, {"type": "uint256"}, {"type": "bool"}
                    ], "type": "tuple"}],
                    "stateMutability": "view",
                    "type": "function"
                },
                {
                    "inputs": [{"type": "bytes32"}, {"type": "string[]"}],
                    "name": "updateCapabilities",
                    "outputs": [{"type": "bool"}],
                    "stateMutability": "nonpayable",
                    "type": "function"
                }
            ]
        if contract_name == "ERC8004IdentityRegistry":
            return [
                {
                    "inputs": [{"type": "string"}, {"type": "tuple[]", "components": [{"type": "string"}, {"type": "bytes"}]}],
                    "name": "register",
                    "outputs": [{"type": "uint256"}],
                    "stateMutability": "nonpayable",
                    "type": "function"
                },
                {
                    "inputs": [{"type": "string"}],
                    "name": "register",
                    "outputs": [{"type": "uint256"}],
                    "stateMutability": "nonpayable",
                    "type": "function"
                },
                {
                    "inputs": [],
                    "name": "register",
                    "outputs": [{"type": "uint256"}],
                    "stateMutability": "nonpayable",
                    "type": "function"
                },
                {
                    "inputs": [{"type": "uint256"}, {"type": "string"}, {"type": "bytes"}],
                    "name": "setMetadata",
                    "outputs": [],
                    "stateMutability": "nonpayable",
                    "type": "function"
                }
            ]
        if contract_name == "ERC8004ReputationRegistry":
            return [
                {
                    "inputs": [
                        {"type": "uint256"},
                        {"type": "uint8"},
                        {"type": "bytes32"},
                        {"type": "bytes32"},
                        {"type": "string"},
                        {"type": "bytes32"},
                        {"type": "bytes"}
                    ],
                    "name": "giveFeedback",
                    "outputs": [],
                    "stateMutability": "nonpayable",
                    "type": "function"
                },
                {
                    "inputs": [
                        {"type": "uint256"},
                        {"type": "address"},
                        {"type": "uint64"}
                    ],
                    "name": "revokeFeedback",
                    "outputs": [],
                    "stateMutability": "nonpayable",
                    "type": "function"
                },
                {
                    "inputs": [
                        {"type": "uint256"},
                        {"type": "address[]"},
                        {"type": "bytes32"},
                        {"type": "bytes32"}
                    ],
                    "name": "getSummary",
                    "outputs": [{"type": "uint64"}, {"type": "uint8"}],
                    "stateMutability": "view",
                    "type": "function"
                }
            ]
        if contract_name == "ERC8004ValidationRegistry":
            return [
                {
                    "inputs": [
                        {"type": "address"},
                        {"type": "uint256"},
                        {"type": "string"},
                        {"type": "bytes32"}
                    ],
                    "name": "validationRequest",
                    "outputs": [],
                    "stateMutability": "nonpayable",
                    "type": "function"
                },
                {
                    "inputs": [
                        {"type": "bytes32"},
                        {"type": "uint8"},
                        {"type": "string"},
                        {"type": "bytes32"},
                        {"type": "bytes32"}
                    ],
                    "name": "validationResponse",
                    "outputs": [],
                    "stateMutability": "nonpayable",
                    "type": "function"
                },
                {
                    "inputs": [{"type": "bytes32"}],
                    "name": "getValidationStatus",
                    "outputs": [
                        {"type": "address"},
                        {"type": "uint256"},
                        {"type": "uint8"},
                        {"type": "bytes32"},
                        {"type": "uint256"}
                    ],
                    "stateMutability": "view",
                    "type": "function"
                }
            ]
        return []

    def calculate_did_hash(self, did: str) -> bytes:
        """Calculate keccak256 hash of DID string"""
        return keccak(text=did)

    def create_eip712_signature(self, did_hash: bytes, agent_card_uri: str, did_doc_uri: str) -> str:
        """Create EIP-712 signature for agent registration"""
        if not self.account:
            raise ValueError("Private key required for signing")

        # EIP-712 domain
        domain = {
            "name": "SpoonAgentRegistry",
            "version": "1",
            "chainId": self.w3.eth.chain_id,
            "verifyingContract": self.agent_registry.address
        }

        # Message types
        types = {
            "AgentRegistration": [
                {"name": "didHash", "type": "bytes32"},
                {"name": "agentCardURI", "type": "string"},
                {"name": "didDocURI", "type": "string"}
            ]
        }

        # Message data
        message = {
            "didHash": did_hash.hex() if isinstance(did_hash, bytes) else did_hash,
            "agentCardURI": agent_card_uri,
            "didDocURI": did_doc_uri
        }

        signable_message = encode_typed_data(
            domain_data=domain,
            message_types=types,
            message_data=message
        )

        signed = self.account.sign_message(signable_message)
        return signed.signature.hex()

    def register_agent(
        self,
        did: str,
        agent_card_uri: str,
        did_doc_uri: str
    ) -> str:
        """Register agent on-chain"""
        if not self.account:
            raise ValueError("Private key required for registration")

        did_hash = self.calculate_did_hash(did)
        signature = self.create_eip712_signature(did_hash, agent_card_uri, did_doc_uri)

        tx = self.agent_registry.functions.registerAgent(
            did_hash,
            agent_card_uri,
            did_doc_uri,
            bytes.fromhex(signature.replace('0x', ''))
        ).build_transaction({
            'from': self.account.address,
            'nonce': self.w3.eth.get_transaction_count(self.account.address),
            'gas': 500000,
            'gasPrice': self.w3.eth.gas_price
        })

        signed_tx = self.account.sign_transaction(tx)
        tx_hash = self.w3.eth.send_raw_transaction(signed_tx.raw_transaction)
        receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash)

        if receipt['status'] == 1:
            return tx_hash.hex()
        else:
            raise Exception(f"Transaction failed: {receipt}")

    def resolve_agent(self, did: str) -> Dict:
        """Resolve agent metadata from on-chain registry"""
        did_hash = self.calculate_did_hash(did)
        result = self.agent_registry.functions.resolveAgent(did_hash).call()

        return {
            "controllers": result[0],
            "agentCardURI": result[1],
            "didDocURI": result[2],
            "capabilities": result[3],
            "registeredAt": result[4],
            "exists": result[5]
        }

    def update_capabilities(self, did: str, capabilities: List[str]) -> str:
        """Update agent capabilities on-chain"""
        if not self.account:
            raise ValueError("Private key required")

        did_hash = self.calculate_did_hash(did)

        tx = self.agent_registry.functions.updateCapabilities(
            did_hash,
            capabilities
        ).build_transaction({
            'from': self.account.address,
            'nonce': self.w3.eth.get_transaction_count(self.account.address),
            'gas': 200000,
            'gasPrice': self.w3.eth.gas_price
        })

        signed_tx = self.account.sign_transaction(tx)
        tx_hash = self.w3.eth.send_raw_transaction(signed_tx.raw_transaction)
        receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash)

        if receipt['status'] == 1:
            return tx_hash.hex()
        else:
            raise Exception(f"Transaction failed: {receipt}")

    def _agent_id_bytes(self, did: str) -> bytes:
        return self.calculate_did_hash(did)

    def _agent_id_int(self, did: str) -> int:
        return int.from_bytes(self._agent_id_bytes(did), byteorder="big")

    # ---------- Reputation (ERC8004 v1.0) ----------
    def build_feedback_auth(
        self,
        agent_id: Union[int, bytes],
        client_address: str,
        index_limit: int,
        expiry: int,
        signer_address: Optional[str] = None,
        identity_registry: Optional[str] = None,
        chain_id: Optional[int] = None
    ) -> bytes:
        """
        Build and sign feedbackAuth payload required by chaoschain ERC8004 reputation registry.
        Returns abi.encode(struct) ++ signature (65 bytes).
        """
        if not self.account:
            raise ValueError("Private key required to sign feedbackAuth")

        agent_id_int = agent_id if isinstance(agent_id, int) else int.from_bytes(agent_id, "big")
        signer_addr = signer_address or self.account.address
        chain = chain_id or self.w3.eth.chain_id
        id_reg = identity_registry or self.identity_registry.address

        struct_bytes = abi_encode(
            ["uint256", "address", "uint64", "uint256", "uint256", "address", "address"],
            [agent_id_int, to_checksum_address(client_address), index_limit, expiry, chain, to_checksum_address(id_reg),
             to_checksum_address(signer_addr)]
        )
        struct_hash = Web3.keccak(struct_bytes)
        message_hash = Web3.keccak(b"\x19Ethereum Signed Message:\n32" + struct_hash)
        signed = Account.sign_hash(message_hash, private_key=self.private_key)
        return struct_bytes + signed.signature

    def give_feedback(
        self,
        did: str,
        score: int,
        tag1: bytes = b"",
        tag2: bytes = b"",
        fileuri: str = "",
        filehash: bytes = b"\x00" * 32,
        index_limit: int = 10,
        expiry: Optional[int] = None,
        client_address: Optional[str] = None
    ) -> str:
        """Submit feedback using ERC8004 giveFeedback with feedbackAuth."""
        if not self.account:
            raise ValueError("Private key required")
        agent_id_int = self._agent_id_int(did)
        client = client_address or self.account.address
        deadline = expiry or (int(self.w3.eth.get_block("latest")["timestamp"]) + 3600)

        auth = self.build_feedback_auth(
            agent_id_int,
            client,
            index_limit,
            deadline,
            signer_address=self.account.address
        )

        tx = self.reputation_registry.functions.giveFeedback(
            agent_id_int,
            score,
            tag1,
            tag2,
            fileuri,
            filehash,
            auth
        ).build_transaction({
            "from": self.account.address,
            "nonce": self.w3.eth.get_transaction_count(self.account.address),
            "gas": 300000,
            "gasPrice": self.w3.eth.gas_price
        })

        signed_tx = self.account.sign_transaction(tx)
        tx_hash = self.w3.eth.send_raw_transaction(signed_tx.raw_transaction)
        receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash)
        if receipt["status"] != 1:
            raise Exception(f"giveFeedback failed: {receipt}")
        return tx_hash.hex()

    def revoke_feedback(self, did: str, feedback_index: int) -> str:
        if not self.account:
            raise ValueError("Private key required")
        agent_id_int = self._agent_id_int(did)
        tx = self.reputation_registry.functions.revokeFeedback(
            agent_id_int, feedback_index
        ).build_transaction({
            "from": self.account.address,
            "nonce": self.w3.eth.get_transaction_count(self.account.address),
            "gas": 120000,
            "gasPrice": self.w3.eth.gas_price
        })
        signed = self.account.sign_transaction(tx)
        tx_hash = self.w3.eth.send_raw_transaction(signed.raw_transaction)
        receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash)
        if receipt["status"] != 1:
            raise Exception(f"revokeFeedback failed: {receipt}")
        return tx_hash.hex()

    def get_reputation_summary(
        self,
        did: str,
        client_addresses: Optional[List[str]] = None,
        tag1: bytes = b"\x00" * 32,
        tag2: bytes = b"\x00" * 32
    ) -> Tuple[int, int]:
        """Return (count, averageScore 0-100)."""
        agent_id_int = self._agent_id_int(did)
        addresses = client_addresses or []
        count, avg = self.reputation_registry.functions.getSummary(
            agent_id_int,
            addresses,
            tag1,
            tag2
        ).call()
        return count, avg

    def get_reputation(self, did: str) -> Tuple[int, int]:
        """Backward compatible: (averageScore, count)."""
        count, avg = self.get_reputation_summary(did)
        return avg, count

    # ---------- Validation (ERC8004 v1.0) ----------
    def validation_request(
        self,
        did: str,
        validator: str,
        request_uri: str,
        request_hash: Optional[bytes] = None
    ) -> Tuple[str, bytes]:
        """Create validation request; returns tx hash and requestHash used."""
        if not self.account:
            raise ValueError("Private key required")
        agent_id_int = self._agent_id_int(did)
        key = request_hash or Web3.keccak(text=f"{validator}:{agent_id_int}:{request_uri}:{self.w3.eth.block_number}")
        tx = self.validation_registry.functions.validationRequest(
            to_checksum_address(validator),
            agent_id_int,
            request_uri,
            key
        ).build_transaction({
            "from": self.account.address,
            "nonce": self.w3.eth.get_transaction_count(self.account.address),
            "gas": 220000,
            "gasPrice": self.w3.eth.gas_price
        })
        signed = self.account.sign_transaction(tx)
        tx_hash = self.w3.eth.send_raw_transaction(signed.raw_transaction)
        receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash)
        if receipt["status"] != 1:
            raise Exception(f"validationRequest failed: {receipt}")
        return tx_hash.hex(), key

    def validation_response(
        self,
        request_hash: bytes,
        response: int,
        response_uri: str = "",
        response_hash: bytes = b"\x00" * 32,
        tag: bytes = b"\x00" * 32
    ) -> str:
        if not self.account:
            raise ValueError("Private key required")
        tx = self.validation_registry.functions.validationResponse(
            request_hash,
            response,
            response_uri,
            response_hash,
            tag
        ).build_transaction({
            "from": self.account.address,
            "nonce": self.w3.eth.get_transaction_count(self.account.address),
            "gas": 180000,
            "gasPrice": self.w3.eth.gas_price
        })
        signed = self.account.sign_transaction(tx)
        tx_hash = self.w3.eth.send_raw_transaction(signed.raw_transaction)
        receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash)
        if receipt["status"] != 1:
            raise Exception(f"validationResponse failed: {receipt}")
        return tx_hash.hex()

    def get_validation_status(self, request_hash: bytes) -> Dict:
        """Get per-request validation status."""
        validator_addr, agent_id, resp, tag, last_update = \
            self.validation_registry.functions.getValidationStatus(request_hash).call()
        return {
            "validator": validator_addr,
            "agentId": agent_id,
            "response": resp,
            "tag": tag,
            "lastUpdate": last_update
        }

    def get_validation_summary(
        self,
        did: str,
        validators: Optional[List[str]] = None,
        tag: bytes = b"\x00" * 32
    ) -> Dict:
        agent_id_int = self._agent_id_int(did)
        validator_filter = validators or []
        count, avg = self.validation_registry.functions.getSummary(
            agent_id_int,
            validator_filter,
            tag
        ).call()
        return {"count": count, "averageResponse": avg, "isValidated": count >= 3 and avg > 50}

    # Backwards-compatible aliases
    def submit_reputation(self, did: str, score: int, evidence: str = "") -> str:
        return self.give_feedback(did, score, fileuri=evidence)

    def submit_validation(self, did: str, is_valid: bool, reason: str = "") -> str:
        """Map boolean into 0/100 scale response."""
        resp = 100 if is_valid else 0
        _, req_hash = self.validation_request(did, self.account.address, reason or "validation")
        return self.validation_response(req_hash, resp)
















