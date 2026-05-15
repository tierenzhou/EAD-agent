"""SQLite store for encrypted API keys (single-tenant)."""

from __future__ import annotations

import sqlite3
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from cryptography.hazmat.primitives.ciphers.aead import AESGCM


def _hermes_home() -> Path:
    from hermes_cli.config import get_hermes_home

    return get_hermes_home()


def _db_path() -> Path:
    override = __import__("os").environ.get("EXPLORE_CONTROL_DB", "").strip()
    if override:
        return Path(override)
    return _hermes_home() / "explore_control.db"


def _last4(plaintext: str) -> str:
    s = plaintext.strip()
    if len(s) <= 4:
        return "****"
    return f"…{s[-4:]}"


class ExploreCredentialStore:
    def __init__(self, dek: bytes) -> None:
        if len(dek) != 32:
            raise ValueError("DEK must be 32 bytes")
        self._dek = dek
        self._path = _db_path()
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self._path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute(
            """CREATE TABLE IF NOT EXISTS credentials (
                purpose TEXT PRIMARY KEY,
                ciphertext BLOB NOT NULL,
                nonce BLOB NOT NULL,
                last4 TEXT NOT NULL,
                updated_at REAL NOT NULL
            )"""
        )
        self._conn.commit()

    def _encrypt(self, purpose: str, plaintext: str) -> Tuple[bytes, bytes]:
        aes = AESGCM(self._dek)
        nonce = __import__("os").urandom(12)
        ad = purpose.encode("utf-8")
        ct = aes.encrypt(nonce, plaintext.encode("utf-8"), ad)
        return nonce, ct

    def _decrypt(self, purpose: str, nonce: bytes, ciphertext: bytes) -> str:
        aes = AESGCM(self._dek)
        ad = purpose.encode("utf-8")
        pt = aes.decrypt(nonce, ciphertext, ad)
        return pt.decode("utf-8")

    def upsert(self, purpose: str, plaintext: str) -> None:
        nonce, ct = self._encrypt(purpose, plaintext)
        now = time.time()
        self._conn.execute(
            """INSERT INTO credentials (purpose, ciphertext, nonce, last4, updated_at)
               VALUES (?, ?, ?, ?, ?)
               ON CONFLICT(purpose) DO UPDATE SET
                 ciphertext = excluded.ciphertext,
                 nonce = excluded.nonce,
                 last4 = excluded.last4,
                 updated_at = excluded.updated_at
            """,
            (purpose, ct, nonce, _last4(plaintext), now),
        )
        self._conn.commit()

    def list_masked(self) -> List[Dict[str, Any]]:
        rows = self._conn.execute(
            "SELECT purpose, last4, updated_at FROM credentials ORDER BY purpose"
        ).fetchall()
        return [{"purpose": r["purpose"], "last4": r["last4"], "updated_at": r["updated_at"]} for r in rows]

    def decrypt_all(self) -> Dict[str, str]:
        out: Dict[str, str] = {}
        rows = self._conn.execute(
            "SELECT purpose, ciphertext, nonce FROM credentials"
        ).fetchall()
        for r in rows:
            try:
                out[r["purpose"]] = self._decrypt(r["purpose"], r["nonce"], r["ciphertext"])
            except Exception:
                continue
        return out
