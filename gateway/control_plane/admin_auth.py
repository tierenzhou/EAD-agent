"""PBKDF2 admin password file + HMAC session tokens."""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import secrets
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

_AUTH_NAME = "explore_admin_auth.json"
_SESSION_BYTES = 32


def _hermes_home() -> Path:
    from hermes_cli.config import get_hermes_home

    return get_hermes_home()


def _auth_path() -> Path:
    return _hermes_home() / _AUTH_NAME


def _session_secret_path() -> Path:
    return _hermes_home() / "explore_session_secret.bin"


def load_session_secret() -> bytes:
    env = os.getenv("EXPLORE_SESSION_SECRET", "").strip()
    if env:
        return hashlib.sha256(env.encode("utf-8")).digest()
    p = _session_secret_path()
    if p.is_file():
        data = p.read_bytes()
        if len(data) >= _SESSION_BYTES:
            return data[:_SESSION_BYTES]
    sec = secrets.token_bytes(_SESSION_BYTES)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(sec)
    try:
        p.chmod(0o600)
    except OSError:
        pass
    return sec


def _pbkdf2_hash(password: str, salt: bytes, iterations: int = 600_000) -> bytes:
    return hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iterations, dklen=32)


def auth_file_exists() -> bool:
    return _auth_path().is_file()


def bootstrap_password(password: str) -> None:
    """Create auth file (fails if already exists)."""
    path = _auth_path()
    if path.exists():
        raise FileExistsError("admin auth already bootstrapped")
    salt = secrets.token_bytes(16)
    h = _pbkdf2_hash(password, salt)
    data = {
        "v": 1,
        "pbkdf2": {
            "salt_b64": base64.b64encode(salt).decode("ascii"),
            "hash_b64": base64.b64encode(h).decode("ascii"),
            "iterations": 600_000,
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    try:
        path.chmod(0o600)
    except OSError:
        pass


def verify_password(password: str) -> bool:
    path = _auth_path()
    if not path.is_file():
        return False
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        block = data.get("pbkdf2") or {}
        salt = base64.b64decode(block.get("salt_b64", ""))
        good = base64.b64decode(block.get("hash_b64", ""))
        it = int(block.get("iterations", 600_000))
        trial = _pbkdf2_hash(password, salt, it)
        return hmac.compare_digest(trial, good)
    except Exception:
        return False


def _b64url_encode(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def _b64url_decode(data: str) -> bytes:
    pad = "=" * ((4 - len(data) % 4) % 4)
    return base64.urlsafe_b64decode(data + pad)


def create_session_token(ttl_seconds: int = 86400) -> str:
    secret = load_session_secret()
    payload = {"exp": int(time.time()) + ttl_seconds, "v": 1}
    body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    body_b64 = _b64url_encode(body)
    sig = hmac.new(secret, body_b64.encode("ascii"), hashlib.sha256).hexdigest()
    return f"{body_b64}.{sig}"


def verify_session_token(token: Optional[str]) -> bool:
    if not token:
        return False
    parts = token.split(".", 1)
    if len(parts) != 2:
        return False
    body_b64, sig_hex = parts
    secret = load_session_secret()
    expect = hmac.new(secret, body_b64.encode("ascii"), hashlib.sha256).hexdigest()
    if not hmac.compare_digest(expect, sig_hex):
        return False
    try:
        payload = json.loads(_b64url_decode(body_b64).decode("utf-8"))
        exp = int(payload.get("exp", 0))
        if exp < int(time.time()):
            return False
        return True
    except Exception:
        return False


def extract_session_from_request(headers: Any) -> Optional[str]:
    """Cookie explore_admin_session, Authorization Bearer, or X-Explore-Admin-Session."""
    cookie = headers.get("Cookie", "")
    for part in cookie.split(";"):
        part = part.strip()
        if part.startswith("explore_admin_session="):
            return part.split("=", 1)[1].strip()
    auth = headers.get("Authorization", "")
    if auth.lower().startswith("bearer "):
        return auth[7:].strip()
    return headers.get("X-Explore-Admin-Session", "").strip() or None
