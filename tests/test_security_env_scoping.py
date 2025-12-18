import os

import pytest

from spoon_ai.security import async_decrypted_environ, init_security
from spoon_ai.tools.base import BaseTool
from spoon_ai.wallet.security import ENCRYPTED_PREFIX, encrypt_private_key


@pytest.mark.asyncio
async def test_init_security_does_not_overwrite_env_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    password = "test-password"
    plaintext = "0x" + "1" * 64
    encrypted = encrypt_private_key(plaintext, password)
    assert encrypted.startswith(ENCRYPTED_PREFIX)

    monkeypatch.setenv("SPOON_MASTER_PWD", password)
    monkeypatch.setenv("EVM_PRIVATE_KEY", encrypted)
    monkeypatch.delenv("SPOON_SECURITY_DECRYPT_ON_IMPORT", raising=False)

    init_security()

    # No persistent plaintext in the process env by default.
    assert os.environ["EVM_PRIVATE_KEY"] == encrypted


@pytest.mark.asyncio
async def test_async_decrypted_environ_decrypts_and_restores(monkeypatch: pytest.MonkeyPatch) -> None:
    password = "test-password"
    plaintext = "0x" + "2" * 64
    encrypted = encrypt_private_key(plaintext, password)

    monkeypatch.setenv("SPOON_MASTER_PWD", password)
    monkeypatch.setenv("EVM_PRIVATE_KEY", encrypted)

    async with async_decrypted_environ(keys=["EVM_PRIVATE_KEY"]):
        assert os.environ["EVM_PRIVATE_KEY"] == plaintext

    # Restored after the scope ends.
    assert os.environ["EVM_PRIVATE_KEY"] == encrypted


@pytest.mark.asyncio
async def test_tool_call_scopes_env_decryption(monkeypatch: pytest.MonkeyPatch) -> None:
    password = "test-password"
    plaintext = "0x" + "3" * 64
    encrypted = encrypt_private_key(plaintext, password)

    monkeypatch.setenv("SPOON_MASTER_PWD", password)
    monkeypatch.setenv("EVM_PRIVATE_KEY", encrypted)

    class EvmDummyTool(BaseTool):
        name: str = "evm_dummy"
        description: str = "Reads EVM_PRIVATE_KEY from env."
        parameters: dict = {"type": "object", "properties": {}, "required": []}

        async def execute(self, **kwargs):
            return os.getenv("EVM_PRIVATE_KEY")

    tool = EvmDummyTool()

    # During the call, the env value is plaintext (so downstream toolkits can read it).
    result = await tool()
    assert result == plaintext

    # After the call, the env value is restored to ciphertext.
    assert os.environ["EVM_PRIVATE_KEY"] == encrypted

