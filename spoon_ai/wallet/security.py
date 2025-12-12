"""
AES-GCM helpers for encrypting and decrypting private keys.
"""

import base64
import os
from typing import Tuple

from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

ENCRYPTED_PREFIX = "ENC:v1:"
_SALT_BYTES = 16
_IV_BYTES = 12
_KDF_ITERATIONS = 200_000


def _derive_key(password: str, salt: bytes) -> bytes:
    """Derive a symmetric key from the password using PBKDF2-HMAC-SHA256."""
    if not password:
        raise ValueError("Password is required to derive encryption key.")

    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=salt,
        iterations=_KDF_ITERATIONS,
    )
    return kdf.derive(password.encode("utf-8"))


def _split_payload(payload: bytes) -> Tuple[bytes, bytes, bytes]:
    if len(payload) < _SALT_BYTES + _IV_BYTES + 16:
        raise ValueError("Encrypted payload is malformed or truncated.")
    salt = payload[:_SALT_BYTES]
    iv = payload[_SALT_BYTES : _SALT_BYTES + _IV_BYTES]
    ciphertext = payload[_SALT_BYTES + _IV_BYTES :]
    return salt, iv, ciphertext


def encrypt_private_key(raw_private_key: str, password: str) -> str:
    """
    Encrypt a private key using AES-GCM with PBKDF2-derived key.

    Returns:
        str: Versioned payload `ENC:v1:<base64(salt||iv||ciphertext)>`
    """
    if not raw_private_key:
        raise ValueError("raw_private_key is required for encryption.")
    if not password:
        raise ValueError("password is required for encryption.")

    salt = os.urandom(_SALT_BYTES)
    iv = os.urandom(_IV_BYTES)
    key = _derive_key(password, salt)
    aesgcm = AESGCM(key)
    ciphertext = aesgcm.encrypt(iv, raw_private_key.encode("utf-8"), None)
    payload = base64.urlsafe_b64encode(salt + iv + ciphertext).decode("utf-8")
    return f"{ENCRYPTED_PREFIX}{payload}"


def decrypt_private_key(enc_value: str, password: str) -> str:
    """
    Decrypt a value produced by `encrypt_private_key`.

    Args:
        enc_value: Value with ENC:v1 prefix.
        password: Password used to derive the key.

    Returns:
        str: Decrypted private key.
    """
    if not enc_value or not enc_value.startswith(ENCRYPTED_PREFIX):
        raise ValueError("Encrypted value must start with ENC:v1:")
    if not password:
        raise ValueError("Password is required to decrypt the private key.")

    try:
        payload = base64.urlsafe_b64decode(enc_value[len(ENCRYPTED_PREFIX) :])
        salt, iv, ciphertext = _split_payload(payload)
        key = _derive_key(password, salt)
        aesgcm = AESGCM(key)
        plaintext = aesgcm.decrypt(iv, ciphertext, None)
        return plaintext.decode("utf-8")
    except Exception as exc:
        raise ValueError("Failed to decrypt private key; verify the password.") from exc
