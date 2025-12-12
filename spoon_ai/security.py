"""
Environment security helpers: auto-decrypt ENC:v1 env vars at import time.
"""

import getpass
import os
import sys
from typing import Dict, List

try:
    from dotenv import load_dotenv
except Exception:  # pragma: no cover - optional dependency
    load_dotenv = None  # type: ignore

from spoon_ai.wallet.security import ENCRYPTED_PREFIX, decrypt_private_key

_MASTER_PASSWORD_ENV = "SPOON_MASTER_PWD"
_DOTENV_LOADED = False


def _collect_encrypted_vars(env: Dict[str, str]) -> List[str]:
    return [key for key, value in env.items() if isinstance(value, str) and value.startswith(ENCRYPTED_PREFIX)]


def _resolve_master_password() -> str:
    pwd = os.getenv(_MASTER_PASSWORD_ENV)
    if pwd:
        return pwd
    # Fallback: interactive prompt if TTY is available
    try:
        if sys.stdin.isatty():
            return getpass.getpass(f"Enter master password {_MASTER_PASSWORD_ENV} to decrypt environment secrets: ")
    except Exception:
        pass
    return ""


def _load_env_files() -> None:
    """Load .env files if python-dotenv is available."""
    global _DOTENV_LOADED
    if _DOTENV_LOADED or load_dotenv is None:
        return

    candidates = [".env", "../.env", "../../.env"]
    for path in candidates:
        if os.path.exists(path):
            load_dotenv(path, override=False)
            _DOTENV_LOADED = True
            return
    # Fallback to default search
    load_dotenv()
    _DOTENV_LOADED = True


def init_security() -> None:
    """
    Decrypt any environment variables that start with ENC:v1: using SPOON_MASTER_PWD.
    Raises a clear error if encrypted vars exist but no password is available.
    """
    _load_env_files()
    encrypted_keys = _collect_encrypted_vars(os.environ)
    if not encrypted_keys:
        return

    master_pwd = _resolve_master_password()
    if not master_pwd:
        raise RuntimeError(
            f"Found encrypted environment variables {encrypted_keys} but {_MASTER_PASSWORD_ENV} is not set. "
            f"Set {_MASTER_PASSWORD_ENV} or run in an interactive terminal to provide the password."
        )

    for key in encrypted_keys:
        try:
            decrypted = decrypt_private_key(os.environ[key], master_pwd)
            os.environ[key] = decrypted
        except Exception as exc:
            raise RuntimeError(f"Failed to decrypt {key}: {exc}") from exc
