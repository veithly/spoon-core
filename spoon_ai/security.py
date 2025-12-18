"""
Environment security helpers for ENC:v1 secrets.

Security model:
- By default, Spoon does NOT permanently overwrite `os.environ` with decrypted values.
- Instead, tools can opt into a *scoped* decryption context that temporarily exposes
  plaintext env vars for the duration of a single operation, then restores the
  original encrypted values.

This avoids a common footgun where secrets remain printable (and leakable) for the
entire lifetime of a Python process after a one-time decrypt.
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager, contextmanager
import getpass
import os
import sys
import weakref
from typing import Dict, Iterable, Iterator, List, Optional

try:
    from dotenv import load_dotenv
except Exception:  # pragma: no cover - optional dependency
    load_dotenv = None  # type: ignore

from spoon_ai.wallet.security import ENCRYPTED_PREFIX, decrypt_private_key

_MASTER_PASSWORD_ENV = "SPOON_MASTER_PWD"
_DOTENV_LOADED = False
_DECRYPT_ON_IMPORT_ENV = "SPOON_SECURITY_DECRYPT_ON_IMPORT"

_LOOP_LOCKS: "weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, asyncio.Lock]" = weakref.WeakKeyDictionary()


def _truthy(value: Optional[str]) -> bool:
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _get_loop_lock() -> asyncio.Lock:
    """Return a per-event-loop lock to serialize env mutation during scoped decrypt."""
    loop = asyncio.get_running_loop()
    lock = _LOOP_LOCKS.get(loop)
    if lock is None:
        lock = asyncio.Lock()
        _LOOP_LOCKS[loop] = lock
    return lock


def _collect_encrypted_vars(env: Dict[str, str]) -> List[str]:
    return [key for key, value in env.items() if isinstance(value, str) and value.startswith(ENCRYPTED_PREFIX)]


def _resolve_master_password(*, prompt: bool = True) -> str:
    pwd = os.getenv(_MASTER_PASSWORD_ENV)
    if pwd:
        return pwd
    # Fallback: interactive prompt if TTY is available
    try:
        if prompt and sys.stdin.isatty():
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


def _decrypt_env_vars_in_place(keys: Iterable[str], *, password: str) -> None:
    for key in keys:
        value = os.environ.get(key)
        if not (isinstance(value, str) and value.startswith(ENCRYPTED_PREFIX)):
            continue
        os.environ[key] = decrypt_private_key(value, password)


def init_security() -> None:
    """
    Initialize env security features.

    Default behaviour:
    - Load `.env` files (if python-dotenv is installed).
    - Detect presence of ENC:v1 values and set `SPOON_ENC_PRESENT=1`.
    - Do NOT overwrite `os.environ` with plaintext.

    Legacy/compat mode (not recommended):
    - If `SPOON_SECURITY_DECRYPT_ON_IMPORT` is truthy, decrypt all ENC:v1 env vars
      and overwrite `os.environ` (previous behaviour).
    """
    _load_env_files()
    encrypted_keys = _collect_encrypted_vars(os.environ)
    if not encrypted_keys:
        return

    # Mark that encrypted secrets were present (used by wallet selection logic)
    os.environ["SPOON_ENC_PRESENT"] = "1"

    if not _truthy(os.getenv(_DECRYPT_ON_IMPORT_ENV)):
        return

    master_pwd = _resolve_master_password(prompt=True)
    if not master_pwd:
        raise RuntimeError(
            f"Found encrypted environment variables {encrypted_keys} but {_MASTER_PASSWORD_ENV} is not set. "
            f"Set {_MASTER_PASSWORD_ENV} or enable a TTY prompt to provide the password."
        )

    try:
        _decrypt_env_vars_in_place(encrypted_keys, password=master_pwd)
    except Exception as exc:
        raise RuntimeError(f"Failed to decrypt environment secrets on import: {exc}") from exc


@contextmanager
def decrypted_environ(
    keys: Optional[Iterable[str]] = None,
    *,
    password: Optional[str] = None,
    prompt: bool = True,
) -> Iterator[None]:
    """
    Temporarily decrypt selected ENC:v1 env vars for the scope of a `with` block.

    Notes:
    - This mutates `os.environ` temporarily and restores the original values in `finally`.
    - Prefer `async_decrypted_environ()` inside async code.
    """
    _load_env_files()
    env = os.environ
    target_keys = list(keys) if keys is not None else _collect_encrypted_vars(env)
    target_keys = [k for k in target_keys if isinstance(env.get(k), str) and env[k].startswith(ENCRYPTED_PREFIX)]
    if not target_keys:
        yield
        return

    master_pwd = password or _resolve_master_password(prompt=prompt)
    if not master_pwd:
        raise RuntimeError(
            f"Found encrypted environment variables {target_keys} but {_MASTER_PASSWORD_ENV} is not set."
        )

    original = {k: env[k] for k in target_keys}
    try:
        _decrypt_env_vars_in_place(target_keys, password=master_pwd)
        yield
    finally:
        for k, v in original.items():
            env[k] = v


@asynccontextmanager
async def async_decrypted_environ(
    keys: Optional[Iterable[str]] = None,
    *,
    password: Optional[str] = None,
    prompt: bool = True,
) -> Iterator[None]:
    """
    Async version of `decrypted_environ()` with per-event-loop locking to avoid
    concurrent env mutation races.
    """
    _load_env_files()
    env = os.environ
    target_keys = list(keys) if keys is not None else _collect_encrypted_vars(env)
    target_keys = [k for k in target_keys if isinstance(env.get(k), str) and env[k].startswith(ENCRYPTED_PREFIX)]
    if not target_keys:
        yield
        return

    master_pwd = password or _resolve_master_password(prompt=prompt)
    if not master_pwd:
        raise RuntimeError(
            f"Found encrypted environment variables {target_keys} but {_MASTER_PASSWORD_ENV} is not set."
        )

    lock = _get_loop_lock()
    async with lock:
        original = {k: env[k] for k in target_keys}
        try:
            _decrypt_env_vars_in_place(target_keys, password=master_pwd)
            yield
        finally:
            for k, v in original.items():
                env[k] = v
