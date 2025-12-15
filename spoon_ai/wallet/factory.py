"""
Wallet factory implementing a Chain of Responsibility:
Local key (encrypted/plaintext) -> Turnkey (MPC).
"""

import base64
import getpass
import os
import re
from typing import Optional

from dotenv import load_dotenv
from eth_account import Account

from ..turnkey.client import Turnkey
from .security import ENCRYPTED_PREFIX, decrypt_private_key
from .turnkey_signer import TurnkeySigner

_TURNKEY_REQUIRED_ENV = ("TURNKEY_API_PUBLIC_KEY", "TURNKEY_API_PRIVATE_KEY", "TURNKEY_ORG_ID")
_MASTER_PASSWORD_ENV = "SPOON_MASTER_PWD"
_PRIVATE_KEY_ENV_PRIORITY_OVERRIDE = "SPOON_WALLET_ENV_PRIORITY"
_DEFAULT_PRIVATE_KEY_ENVS = ("PRIVATE_KEY", "EVM_PRIVATE_KEY", "SOLANA_PRIVATE_KEY")


def _normalize_private_key(value: str) -> str:
    key = value.strip()
    return key if key.startswith("0x") else f"0x{key}"


def _looks_like_hex_key(value: str) -> bool:
    key = value.strip()
    return bool(re.fullmatch(r"0x[a-fA-F0-9]{64}", key) or re.fullmatch(r"[a-fA-F0-9]{64}", key))


def _decode_base58(value: str) -> Optional[bytes]:
    """Decode a base58 string if the base58 package is available."""
    try:
        import base58  # type: ignore
    except Exception:
        return None

    try:
        return base58.b58decode(value)
    except Exception:
        return None


def _decode_solana_private_key(value: str) -> Optional[bytes]:
    """
    Attempt to decode a Solana private key in base58/base64 format.

    We expect either 64 bytes (secret key + pubkey, typical solana-keygen output)
    or 32 bytes (seed only). Returns None when decoding fails.
    """
    if not isinstance(value, str):
        return None

    candidate = value.strip()
    if not candidate:
        return None

    # Try base58 first (Phantom/solana-keygen style)
    decoded = _decode_base58(candidate)
    if decoded is not None and len(decoded) in (64, 32):
        return decoded

    # Then try base64
    try:
        decoded = base64.b64decode(candidate, validate=True)
        if len(decoded) in (64, 32):
            return decoded
    except Exception:
        pass

    return None


class SolanaLocalWallet:
    """Lightweight holder for a Solana private key."""

    def __init__(self, private_key: str, private_key_bytes: bytes):
        self.private_key = private_key.strip()
        self.private_key_bytes = private_key_bytes
        # Provide an address attr to avoid attribute errors in generic code; not computed here.
        self.address = None

    def __repr__(self) -> str:  # pragma: no cover - trivial
        return "SolanaLocalWallet()"


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


def _private_key_env_priority() -> list[str]:
    """
    Determine which env vars to inspect for a local key.

    Priority can be overridden with comma-separated SPOON_WALLET_ENV_PRIORITY,
    otherwise we fall back to a conservative default list.
    """
    override = os.getenv(_PRIVATE_KEY_ENV_PRIORITY_OVERRIDE)
    if override:
        names = [name.strip() for name in override.split(",") if name.strip()]
        if names:
            # Preserve order while removing duplicates
            return list(dict.fromkeys(names))
    return list(_DEFAULT_PRIVATE_KEY_ENVS)


def load_wallet():
    """
    Evaluate signing options in priority order and return a wallet/signer.

    Priority (prefer local keys for dev/test):
        1) Encrypted key from env priority list (ENC:...)
        2) Plaintext key from env priority list (0x...)
        3) Turnkey (MPC) when no usable local key is provided
    """
    load_dotenv()

    env_priority = _private_key_env_priority()
    for env_name in env_priority:
        private_key = os.getenv(env_name)
        if not private_key:
            continue

        # Special handling for Solana keys (base58/base64)
        if env_name.upper().startswith("SOLANA"):
            decoded = _decode_solana_private_key(private_key)
            if decoded:
                print(f"✅ [Wallet] Solana private key loaded from {env_name}.")
                return SolanaLocalWallet(private_key, decoded)

        if isinstance(private_key, str) and private_key.startswith(ENCRYPTED_PREFIX):
            print(f"🔐 [Wallet] Encrypted {env_name} detected. Decryption required.")
            password = os.environ.get(_MASTER_PASSWORD_ENV) or getpass.getpass(f"Enter password to decrypt {env_name}: ")
            decrypted_key = decrypt_private_key(private_key, password)
            account = Account.from_key(_normalize_private_key(decrypted_key))
            print(f"✅ [Wallet] Local account unlocked from {env_name}: {account.address}")
            return account

        if _looks_like_hex_key(private_key):
            print(f"⚠️ [Wallet] WARNING: Using plaintext {env_name} from environment.")
            account = Account.from_key(_normalize_private_key(private_key))
            print(f"✅ [Wallet] Local account loaded from {env_name}: {account.address}")
            return account

        print(
            f"⚠️ [Wallet] {env_name} is set but not in a recognized hex private key format; "
            "skipping and falling back to next option."
        )

    if _has_turnkey_credentials():
        print("🛡️ [Wallet] Turnkey credentials detected; using MPC signer.")
        try:
            client = Turnkey()
            sign_with = os.getenv("TURNKEY_DEFAULT_ADDRESS") or _auto_select_turnkey_address(client)
            return TurnkeySigner(client=client, organization_id=client.org_id, sign_with=sign_with)
        except Exception as exc:
            raise ValueError(f"Failed to initialize Turnkey signer: {exc}") from exc

    raise ValueError(
        "No valid signing method configured; set one of "
        f"{', '.join(env_priority)} (plain or ENC:v1) or Turnkey credentials in your .env file."
    )
