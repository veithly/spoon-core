"""
Wallet factory implementing a Chain of Responsibility:
Turnkey (MPC) -> Encrypted private key -> Plaintext key.
"""

import getpass
import os
import re

from dotenv import load_dotenv
from eth_account import Account

from ..turnkey.client import Turnkey
from .security import decrypt_private_key
from .turnkey_signer import TurnkeySigner

_TURNKEY_REQUIRED_ENV = ("TURNKEY_API_PUBLIC_KEY", "TURNKEY_API_PRIVATE_KEY", "TURNKEY_ORG_ID")
_MASTER_PASSWORD_ENV = "SPOON_MASTER_PWD"
_PRIVATE_KEY_ENV = "PRIVATE_KEY"


def _normalize_private_key(value: str) -> str:
    key = value.strip()
    return key if key.startswith("0x") else f"0x{key}"


def _looks_like_hex_key(value: str) -> bool:
    key = value.strip()
    return bool(re.fullmatch(r"0x[a-fA-F0-9]{64}", key) or re.fullmatch(r"[a-fA-F0-9]{64}", key))


def _auto_select_turnkey_address(client: Turnkey) -> str:
    wallets_resp = client.list_wallets()
    wallets = wallets_resp.get("wallets") or []
    if not wallets:
        raise ValueError("Turnkey organization has no wallets; configure one or set TURNKEY_DEFAULT_ADDRESS.")

    first_wallet_id = wallets[0].get("walletId")
    if not first_wallet_id:
        raise ValueError("Turnkey wallet response missing walletId.")

    accounts_resp = client.list_wallet_accounts(wallet_id=first_wallet_id)
    accounts = accounts_resp.get("accounts") or []
    if not accounts:
        raise ValueError(f"Wallet {first_wallet_id} has no accounts; add an account or set TURNKEY_DEFAULT_ADDRESS.")

    address = accounts[0].get("address")
    if not address:
        raise ValueError(f"Wallet {first_wallet_id} account payload missing address.")
    print(f"   Auto-selected Turnkey address: {address}")
    return address


def _has_turnkey_credentials() -> bool:
    return all(os.getenv(k) for k in _TURNKEY_REQUIRED_ENV)


def load_wallet():
    """
    Evaluate signing options in priority order and return a wallet/signer.

    Priority:
        1) Turnkey (MPC)
        2) Encrypted PRIVATE_KEY (ENC:...)
        3) Plaintext PRIVATE_KEY (0x...)
    """
    load_dotenv()

    if _has_turnkey_credentials():
        print("🛡️ [Wallet] Turnkey credentials detected; using MPC signer.")
        try:
            client = Turnkey()
            sign_with = os.getenv("TURNKEY_DEFAULT_ADDRESS") or _auto_select_turnkey_address(client)
            return TurnkeySigner(client=client, organization_id=client.org_id, sign_with=sign_with)
        except Exception as exc:
            raise ValueError(f"Failed to initialize Turnkey signer: {exc}") from exc

    private_key = os.getenv(_PRIVATE_KEY_ENV)
    if private_key:
        if private_key.startswith("ENC:"):
            print("🔐 [Wallet] Encrypted PRIVATE_KEY detected. Decryption required.")
            password = os.environ.get(_MASTER_PASSWORD_ENV) or getpass.getpass("Enter password to decrypt PRIVATE_KEY: ")
            decrypted_key = decrypt_private_key(private_key, password)
            account = Account.from_key(_normalize_private_key(decrypted_key))
            print(f"✅ [Wallet] Local account unlocked: {account.address}")
            return account

        if _looks_like_hex_key(private_key):
            print("⚠️ [Wallet] WARNING: Using plaintext PRIVATE_KEY from environment.")
            account = Account.from_key(_normalize_private_key(private_key))
            print(f"✅ [Wallet] Local account loaded: {account.address}")
            return account

        raise ValueError("PRIVATE_KEY is set but not in a recognized format (ENC:v1 or 0x...).")

    raise ValueError(
        "No valid signing method configured; set Turnkey credentials or PRIVATE_KEY in your .env file."
    )
