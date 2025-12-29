"""
Wallet module for secure key management.

This module provides:
- SecretVault: Thread-safe in-memory storage for sensitive secrets
- encrypt_private_key: Encrypt private keys with AES-256-GCM + Argon2id
- decrypt_private_key: Decrypt private keys (returns str - use with caution)
- decrypt_and_store: Decrypt and store directly in vault (recommended)
- load_wallet: Factory function to load wallets from various sources
"""

from .factory import load_wallet
from .security import (
    ENCRYPTED_PREFIX,
    ENCRYPTED_PREFIX_V2,
    DecryptionError,
    decrypt_and_store,
    decrypt_private_key,
    encrypt_private_key,
)
from .turnkey_signer import TurnkeySigner
from .vault import SecretVault, get_vault

__all__ = [
    # Vault
    "SecretVault",
    "get_vault",
    # Encryption/Decryption
    "encrypt_private_key",
    "decrypt_private_key",
    "decrypt_and_store",
    "DecryptionError",
    # Constants
    "ENCRYPTED_PREFIX",
    "ENCRYPTED_PREFIX_V2",
    # Wallet loading
    "load_wallet",
    "TurnkeySigner",
]
