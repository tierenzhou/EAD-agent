"""Load or create the 32-byte AES data-encryption key for Explore control DB."""

from __future__ import annotations

import base64
import os
from pathlib import Path
from typing import Optional

_DEK_BYTES = 32


def _hermes_home() -> Path:
    from hermes_cli.config import get_hermes_home

    return get_hermes_home()


def load_dek() -> bytes:
    """Return 32-byte DEK from EXPLORE_CONTROL_DEK (base64) or ~/.hermes/explore_control_dek.bin."""
    raw = os.getenv("EXPLORE_CONTROL_DEK", "").strip()
    if raw:
        try:
            key = base64.b64decode(raw, validate=True)
        except Exception as exc:
            raise ValueError("EXPLORE_CONTROL_DEK must be valid base64") from exc
        if len(key) != _DEK_BYTES:
            raise ValueError(f"EXPLORE_CONTROL_DEK must decode to {_DEK_BYTES} bytes, got {len(key)}")
        return key

    path = _hermes_home() / "explore_control_dek.bin"
    if path.is_file():
        data = path.read_bytes()
        if len(data) != _DEK_BYTES:
            raise ValueError(f"{path} must be {_DEK_BYTES} bytes")
        return data

    import secrets

    key = secrets.token_bytes(_DEK_BYTES)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(key)
    try:
        path.chmod(0o600)
    except OSError:
        pass
    return key
