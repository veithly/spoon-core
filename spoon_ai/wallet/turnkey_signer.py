"""
TurnkeySigner: adapter to expose Turnkey as an eth_account/web3-compatible signer.
"""

import time
import types
from typing import Any, Mapping, Optional, Union

import rlp
from eth_account._utils.legacy_transactions import (
    Transaction,
    TypedTransaction,
    serializable_unsigned_transaction_from_dict,
)
from eth_account.messages import SignableMessage
from eth_account.typed_transactions.access_list_transaction import (
    transaction_rpc_to_rlp_structure,
)
from eth_utils import keccak, to_hex
from hexbytes import HexBytes

from ..turnkey.client import Turnkey


class TurnkeySigner:
    def __init__(
        self,
        client: Turnkey,
        organization_id: str,
        sign_with: str,
        poll_interval: float = 1.0,
        poll_timeout: Optional[float] = 90.0,
    ):
        self.client = client
        self.organization_id = organization_id
        self.sign_with = sign_with
        self._address = sign_with
        self.poll_interval = poll_interval
        self.poll_timeout = poll_timeout

    @property
    def address(self) -> str:
        return self._address

    def sign_transaction(self, transaction: Union[Mapping[str, Any], str]) -> types.SimpleNamespace:
        """
        Sign a transaction via Turnkey and return a web3-compatible namespace.
        """
        unsigned_hex = self._coerce_unsigned_transaction_hex(transaction)
        response = self.client.sign_evm_transaction(sign_with=self.sign_with, unsigned_tx=unsigned_hex)

        inline_result = self._extract_result(response, "signTransactionResult")
        signed_tx = inline_result.get("signedTransaction") if inline_result else None
        activity_id = self._extract_activity_id(response)

        if not signed_tx:
            activity = self._poll_activity(activity_id, "transaction signing")
            result = self._extract_result(activity, "signTransactionResult")
            signed_tx = result.get("signedTransaction")

        if not signed_tx:
            raise ValueError("Turnkey signing response missing signedTransaction.")

        v, r, s, tx_hash = self._decode_signed_transaction(signed_tx)
        return types.SimpleNamespace(rawTransaction=signed_tx, hash=tx_hash, r=r, s=s, v=v)

    def sign_message(self, message: Union[str, bytes, SignableMessage]) -> types.SimpleNamespace:
        """
        Sign an arbitrary message via Turnkey and return web3-style fields.
        """
        payload, computed_hash = self._prepare_message_payload(message)
        response = self.client.sign_message(sign_with=self.sign_with, message=payload)

        inline_result = self._extract_result(response, "signRawPayloadResult")
        activity_id = self._extract_activity_id(response)

        result = inline_result
        if not result:
            activity = self._poll_activity(activity_id, "message signing")
            result = self._extract_result(activity, "signRawPayloadResult")

        return self._parse_signature_result(result, computed_hash)

    def _prepare_message_payload(
        self, message: Union[str, bytes, SignableMessage]
    ) -> tuple[str, str]:
        if isinstance(message, SignableMessage):
            body = HexBytes(message.body)
            return body.hex(), to_hex(keccak(body))
        if isinstance(message, bytes):
            return message.decode("utf-8"), to_hex(keccak(message))
        if isinstance(message, str):
            return message, to_hex(keccak(text=message))
        raise TypeError("TurnkeySigner.sign_message expects str, bytes, or SignableMessage.")

    def _coerce_unsigned_transaction_hex(self, tx: Union[Mapping[str, Any], str]) -> str:
        if isinstance(tx, str):
            tx_hex = tx.strip()
            return tx_hex if tx_hex.startswith("0x") else f"0x{tx_hex}"

        if not isinstance(tx, Mapping):
            raise TypeError("TurnkeySigner.sign_transaction expects a tx dict or raw hex string.")

        sanitized = {k: v for k, v in tx.items() if k not in {"rawTransaction", "hash", "r", "s", "v"}}
        unsigned_tx = serializable_unsigned_transaction_from_dict(sanitized)
        if isinstance(unsigned_tx, TypedTransaction):
            unsigned_bytes = self._encode_typed_transaction(unsigned_tx)
        else:
            unsigned_bytes = rlp.encode(unsigned_tx)
        return "0x" + unsigned_bytes.hex()

    def _encode_typed_transaction(self, typed_tx: TypedTransaction) -> bytes:
        tx_dict = {k: v for k, v in typed_tx.transaction.dictionary.items() if k not in ("v", "r", "s")}
        rlp_structured = transaction_rpc_to_rlp_structure(tx_dict)
        serializer = typed_tx.transaction.__class__._unsigned_transaction_serializer
        payload = rlp.encode(serializer.from_dict(rlp_structured))
        return bytes([typed_tx.transaction.__class__.transaction_type]) + payload

    def _decode_signed_transaction(self, signed_tx: str) -> tuple[int, int, int, str]:
        raw_bytes = HexBytes(signed_tx)
        try:
            typed_tx = TypedTransaction.from_bytes(raw_bytes)
            v, r, s = typed_tx.transaction.vrs()
            tx_hash_bytes = typed_tx.hash()
        except Exception:
            tx = Transaction.from_bytes(raw_bytes)
            v, r, s = tx.v, tx.r, tx.s
            tx_hash_bytes = tx.hash()
        return int(v), int(r), int(s), to_hex(tx_hash_bytes)

    def _extract_activity_id(self, response: Mapping[str, Any]) -> str:
        activity = response.get("activity") if isinstance(response, Mapping) else None
        if not activity:
            raise ValueError("Turnkey response missing activity payload.")
        activity_id = activity.get("id") or activity.get("activityId")
        if not activity_id:
            raise ValueError("Turnkey response missing activity id.")
        return activity_id

    def _poll_activity(self, activity_id: str, description: str) -> Mapping[str, Any]:
        start = time.time()
        while True:
            activity_resp = self.client.get_activity(activity_id)
            activity = activity_resp.get("activity", {})
            status = activity.get("status")
            if status == "ACTIVITY_STATUS_COMPLETED":
                return activity
            if status in {"ACTIVITY_STATUS_FAILED", "ACTIVITY_STATUS_REJECTED"}:
                raise RuntimeError(f"Turnkey {description} failed with status {status}")
            if self.poll_timeout and (time.time() - start) > self.poll_timeout:
                raise TimeoutError(f"Timed out waiting for Turnkey to complete {description}.")
            time.sleep(self.poll_interval)

    def _extract_result(self, payload: Mapping[str, Any], result_key: str) -> Mapping[str, Any]:
        if not isinstance(payload, Mapping):
            return {}
        result_root = payload.get("result") or payload.get("activity", {}).get("result") or {}
        if not isinstance(result_root, Mapping):
            return {}
        specific = result_root.get(result_key)
        if isinstance(specific, Mapping):
            return specific
        if not specific and result_root:
            return result_root
        return {}

    def _parse_signature_result(self, result: Mapping[str, Any], fallback_hash: Optional[str]) -> types.SimpleNamespace:
        if not isinstance(result, Mapping):
            raise ValueError("Turnkey signature response malformed.")

        signature_hex = result.get("signature") or result.get("signatureHex")
        r = result.get("r")
        s = result.get("s")
        v = result.get("v")

        if signature_hex and (r is None or s is None):
            sig_bytes = HexBytes(signature_hex)
            if len(sig_bytes) == 65:
                r = int.from_bytes(sig_bytes[:32], "big")
                s = int.from_bytes(sig_bytes[32:64], "big")
                v = int(sig_bytes[64])

        signature_hex = to_hex(HexBytes(signature_hex)) if signature_hex is not None else None
        message_hash = result.get("payloadHash") or fallback_hash

        return types.SimpleNamespace(
            messageHash=message_hash,
            r=r,
            s=s,
            v=v,
            signature=signature_hex,
        )
