import base64
import binascii
import hashlib
import os
from dataclasses import dataclass
from typing import Iterable, Optional

from neo3.core import cryptography
from neo3.wallet.account import Account


PREFIX = bytes.fromhex("010001f0")
SUFFIX = bytes.fromhex("0000")


class SignatureError(Exception):
    """Raised when signature payload construction fails."""


def _encode_varint(value: int) -> bytes:
    if value < 0:
        raise ValueError("VarInt cannot encode negative values")
    if value < 0xFD:
        return value.to_bytes(1, "little")
    if value <= 0xFFFF:
        return b"\xFD" + value.to_bytes(2, "little")
    if value <= 0xFFFFFFFF:
        return b"\xFE" + value.to_bytes(4, "little")
    return b"\xFF" + value.to_bytes(8, "little")


def _build_serialized_message(parameters: Iterable[bytes]) -> bytes:
    concatenated = b"".join(parameters)
    length_prefix = _encode_varint(len(concatenated))
    return PREFIX + length_prefix + concatenated + SUFFIX


@dataclass
class SignatureComponents:
    signature: str
    salt: str
    public_key: str

    def signature_header(self) -> str:
        return f"{self.signature}{self.salt}"


def sign_with_salt(private_key_wif: str, *payload_parts: bytes, salt: bytes | None = None) -> SignatureComponents:
    account = Account.from_wif(private_key_wif)
    salt_bytes = salt if salt is not None else os.urandom(16)
    serialized_message = _build_serialized_message((salt_bytes, *payload_parts))
    signature = account.sign(serialized_message)
    return SignatureComponents(
        signature=signature.hex(),
        salt=salt_bytes.hex(),
        public_key=account.public_key.to_array().hex(),
    )


def generate_simple_signature_params(private_key_wif: Optional[str] = None, payload_parts: Iterable[bytes] | None = None, *, components: SignatureComponents | None = None, salt: bytes | None = None) -> dict:
    if components is None:
        if private_key_wif is None:
            raise ValueError("Either components or private_key_wif must be provided")
        parts = tuple(payload_parts or ())
        components = sign_with_salt(private_key_wif, *parts, salt=salt)
    return {
        "signatureParam": components.signature_header(),
        "signatureKeyParam": components.public_key,
        "signatureScheme": "ECDSA_SHA256",
    }


def sign_bearer_token(bearer_token: str, private_key_wif: str, *, wallet_connect: bool = False) -> tuple[str, str]:
    """Produce REST gateway compatible bearer signature."""
    account = Account.from_wif(private_key_wif)
    public_key = account.public_key.to_array().hex()

    if wallet_connect:
        decoded = base64.standard_b64decode(bearer_token)
        normalized_token = base64.standard_b64encode(decoded)
        salt = os.urandom(16)
        salt_hex_bytes = binascii.hexlify(salt)
        payload = _build_serialized_message((salt_hex_bytes, normalized_token))
        signature = cryptography.sign(payload, account.private_key, hash_func=hashlib.sha256)
        signature_hex = binascii.hexlify(signature).decode("utf-8")
        return f"{signature_hex}{salt_hex_bytes.decode('utf-8')}", public_key

    payload = base64.standard_b64decode(bearer_token)
    signature = cryptography.sign(payload, account.private_key, hash_func=hashlib.sha512)
    return f"04{signature.hex()}", public_key
