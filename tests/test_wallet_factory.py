import base64

import pytest

from spoon_ai.wallet import factory


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    """Reset relevant env and disable dotenv loading for isolation."""
    for key in (
        "PRIVATE_KEY",
        "SPOON_MASTER_PWD",
        "TURNKEY_API_PUBLIC_KEY",
        "TURNKEY_API_PRIVATE_KEY",
        "TURNKEY_ORG_ID",
        "TURNKEY_DEFAULT_ADDRESS",
        "SPOON_ENC_PRESENT",
        "OTHER_SECRET",
        "EVM_PRIVATE_KEY",
        "SOLANA_PRIVATE_KEY",
        "SPOON_WALLET_ENV_PRIORITY",
    ):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(factory, "load_dotenv", lambda *args, **kwargs: None)
    monkeypatch.setattr(factory, "_auto_select_turnkey_address", lambda client: "0xturnkey")


def _stub_turnkey(monkeypatch, calls):
    class DummyTurnkey:
        def __init__(self):
            calls["turnkey_inits"] = calls.get("turnkey_inits", 0) + 1
            self.org_id = "org123"

    dummy_signer = object()

    def fake_signer(client, organization_id, sign_with):
        calls["signer_args"] = (client, organization_id, sign_with)
        return dummy_signer

    monkeypatch.setattr(factory, "Turnkey", DummyTurnkey)
    monkeypatch.setattr(factory, "TurnkeySigner", fake_signer)
    return dummy_signer


def test_encrypted_private_key_preferred_over_turnkey(monkeypatch):
    calls = {}

    def fake_decrypt(value, pwd):
        calls["decrypt"] = (value, pwd)
        return "0x" + "1" * 64

    class DummyAccount:
        def __init__(self, address):
            self.address = address

    def fake_from_key(key):
        calls["from_key"] = key
        return DummyAccount("0xlocal")

    monkeypatch.setattr(factory, "decrypt_private_key", fake_decrypt)
    monkeypatch.setattr(factory, "Account", type("AccountStub", (), {"from_key": staticmethod(fake_from_key)}))
    monkeypatch.setattr(
        factory,
        "Turnkey",
        lambda: (_ for _ in ()).throw(AssertionError("Turnkey should not be used when PRIVATE_KEY is encrypted")),
    )
    monkeypatch.setattr(
        factory,
        "TurnkeySigner",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("Turnkey signer should not be used")),
    )

    monkeypatch.setenv("PRIVATE_KEY", "ENC:v1:abc")
    monkeypatch.setenv("SPOON_MASTER_PWD", "pwd")

    wallet = factory.load_wallet()

    assert isinstance(wallet, DummyAccount)
    assert calls["decrypt"] == ("ENC:v1:abc", "pwd")
    assert calls["from_key"].startswith("0x")


def test_plain_private_key_preferred_over_turnkey(monkeypatch):
    calls = {}

    class DummyAccount:
        def __init__(self, address):
            self.address = address

    def fake_from_key(key):
        calls["from_key"] = key
        return DummyAccount("0xplain")

    monkeypatch.setattr(factory, "decrypt_private_key", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError))
    monkeypatch.setattr(factory, "Account", type("AccountStub", (), {"from_key": staticmethod(fake_from_key)}))
    monkeypatch.setattr(
        factory,
        "Turnkey",
        lambda: (_ for _ in ()).throw(AssertionError("Turnkey should not be used when PRIVATE_KEY is plaintext")),
    )
    monkeypatch.setattr(
        factory,
        "TurnkeySigner",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("Turnkey signer should not be used")),
    )

    monkeypatch.setenv("PRIVATE_KEY", "0x" + "2" * 64)

    wallet = factory.load_wallet()

    assert isinstance(wallet, DummyAccount)
    assert calls["from_key"].startswith("0x2")


def test_turnkey_used_when_no_private_key_even_with_other_encrypted_vars(monkeypatch):
    calls = {}
    dummy_signer = _stub_turnkey(monkeypatch, calls)
    monkeypatch.setattr(factory, "decrypt_private_key", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError))
    monkeypatch.setattr(
        factory,
        "Account",
        type(
            "AccountStub",
            (),
            {"from_key": staticmethod(lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError))},
        ),
    )

    monkeypatch.setenv("TURNKEY_API_PUBLIC_KEY", "pub")
    monkeypatch.setenv("TURNKEY_API_PRIVATE_KEY", "priv")
    monkeypatch.setenv("TURNKEY_ORG_ID", "org123")
    monkeypatch.setenv("OTHER_SECRET", "ENC:v1:zzz")

    wallet = factory.load_wallet()

    assert wallet is dummy_signer
    assert calls["signer_args"][1] == "org123"
    assert calls["signer_args"][2] == "0xturnkey"


def test_turnkey_used_when_private_key_format_invalid(monkeypatch):
    calls = {}
    dummy_signer = _stub_turnkey(monkeypatch, calls)
    monkeypatch.setattr(factory, "decrypt_private_key", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError))
    monkeypatch.setattr(
        factory,
        "Account",
        type(
            "AccountStub",
            (),
            {"from_key": staticmethod(lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError))},
        ),
    )

    monkeypatch.setenv("PRIVATE_KEY", "not-a-valid-key")
    monkeypatch.setenv("TURNKEY_API_PUBLIC_KEY", "pub")
    monkeypatch.setenv("TURNKEY_API_PRIVATE_KEY", "priv")
    monkeypatch.setenv("TURNKEY_ORG_ID", "org123")

    wallet = factory.load_wallet()

    assert wallet is dummy_signer
    assert calls["signer_args"][1] == "org123"
    assert calls["signer_args"][2] == "0xturnkey"


def test_evm_private_key_used_when_primary_missing(monkeypatch):
    calls = {}

    class DummyAccount:
        def __init__(self, address):
            self.address = address

    def fake_from_key(key):
        calls["from_key"] = key
        return DummyAccount("0xevm")

    monkeypatch.setattr(factory, "decrypt_private_key", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError))
    monkeypatch.setattr(factory, "Account", type("AccountStub", (), {"from_key": staticmethod(fake_from_key)}))
    monkeypatch.setattr(
        factory,
        "Turnkey",
        lambda: (_ for _ in ()).throw(AssertionError("Turnkey should not be used when EVM_PRIVATE_KEY is present")),
    )
    monkeypatch.setattr(
        factory,
        "TurnkeySigner",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("Turnkey signer should not be used")),
    )

    monkeypatch.setenv("EVM_PRIVATE_KEY", "0x" + "3" * 64)

    wallet = factory.load_wallet()

    assert isinstance(wallet, DummyAccount)
    assert calls["from_key"].startswith("0x3")


def test_invalid_format_env_skipped_before_turnkey(monkeypatch):
    calls = {}
    dummy_signer = _stub_turnkey(monkeypatch, calls)
    monkeypatch.setattr(factory, "decrypt_private_key", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError))
    monkeypatch.setattr(
        factory,
        "Account",
        type(
            "AccountStub",
            (),
            {"from_key": staticmethod(lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError))},
        ),
    )

    monkeypatch.setenv("SOLANA_PRIVATE_KEY", "ThisIsNotAHexKey")
    monkeypatch.setenv("TURNKEY_API_PUBLIC_KEY", "pub")
    monkeypatch.setenv("TURNKEY_API_PRIVATE_KEY", "priv")
    monkeypatch.setenv("TURNKEY_ORG_ID", "org123")

    wallet = factory.load_wallet()

    assert wallet is dummy_signer
    assert calls["signer_args"][1] == "org123"
    assert calls["signer_args"][2] == "0xturnkey"


def test_solana_base64_private_key_supported(monkeypatch):
    # Ensure no other keys or Turnkey paths are used
    monkeypatch.setenv("SOLANA_PRIVATE_KEY", base64.b64encode(bytes(range(64))).decode())
    monkeypatch.setattr(
        factory,
        "Turnkey",
        lambda: (_ for _ in ()).throw(AssertionError("Turnkey should not be used for Solana key")),
    )
    monkeypatch.setattr(
        factory,
        "TurnkeySigner",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("Turnkey signer should not be used")),
    )

    wallet = factory.load_wallet()

    assert isinstance(wallet, factory.SolanaLocalWallet)
    assert wallet.private_key_bytes == bytes(range(64))
