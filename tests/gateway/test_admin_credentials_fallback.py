import os

from gateway.control_plane.admin_routes import (
    _effective_masked_credentials,
    apply_stored_credentials_to_environ,
)


class _FakeStore:
    def __init__(self, *, masked=None, decrypted=None):
        self._masked = list(masked or [])
        self._decrypted = dict(decrypted or {})

    def list_masked(self):
        return list(self._masked)

    def decrypt_all(self):
        return dict(self._decrypted)


def test_apply_stored_credentials_db_first_with_env_fallback(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "local-openai-key")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "local-deepseek-key")
    store = _FakeStore(decrypted={"openai_api": "db-openai-key"})

    apply_stored_credentials_to_environ(store)

    assert os.getenv("OPENAI_API_KEY") == "db-openai-key"
    assert os.getenv("DEEPSEEK_API_KEY") == "local-deepseek-key"


def test_effective_masked_credentials_includes_env_when_db_missing(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "local-openai-1234")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "local-deepseek-5678")
    store = _FakeStore(
        masked=[
            {
                "purpose": "openai_api",
                "last4": "…db42",
                "updated_at": 123.0,
            }
        ]
    )

    rows = _effective_masked_credentials(store)
    by_purpose = {row["purpose"]: row for row in rows}

    assert by_purpose["openai_api"]["last4"] == "…db42"
    assert by_purpose["openai_api"]["source"] == "db"
    assert by_purpose["deepseek_api"]["last4"] == "...5678"
    assert by_purpose["deepseek_api"]["source"] == "env"
