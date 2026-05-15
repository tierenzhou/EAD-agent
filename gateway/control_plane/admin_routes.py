"""Minimal /v1/admin/* routes for Explore UI (single-tenant credentials)."""

from __future__ import annotations

import logging
import os
import time
from typing import TYPE_CHECKING, Any, Dict, Optional

if TYPE_CHECKING:
    from aiohttp import web

logger = logging.getLogger(__name__)

_PURPOSE_TO_ENV = {
    "openai_api": "OPENAI_API_KEY",
    "deepseek_api": "DEEPSEEK_API_KEY",
}

_RATE_WINDOW_S = 60.0
_RATE_MAX = 20
_FAILED_LOGINS: Dict[str, list] = {}
_PUT_ATTEMPTS: Dict[str, list] = {}


def _mask_last4(value: str) -> str:
    s = str(value or "").strip()
    if len(s) <= 4:
        return "****" if s else ""
    return f"...{s[-4:]}"


def _effective_masked_credentials(store: Any) -> list[dict[str, Any]]:
    """Return masked credentials with DB precedence and env fallback."""
    rows = []
    try:
        rows = list(store.list_masked() or [])
    except Exception as exc:
        logger.warning("[explore_admin] could not list stored credentials: %s", exc)
        rows = []
    by_purpose: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        purpose = str((row or {}).get("purpose") or "").strip()
        if not purpose:
            continue
        row_copy = dict(row)
        row_copy.setdefault("source", "db")
        by_purpose[purpose] = row_copy
    now = time.time()
    for purpose, env_key in _PURPOSE_TO_ENV.items():
        if purpose in by_purpose:
            continue
        env_val = os.getenv(env_key, "").strip()
        if not env_val:
            continue
        by_purpose[purpose] = {
            "purpose": purpose,
            "last4": _mask_last4(env_val),
            "updated_at": now,
            "source": "env",
        }
    return [by_purpose[k] for k in sorted(by_purpose.keys())]


def apply_stored_credentials_to_environ(store: Any) -> None:
    """Apply credentials with precedence: DB first, then existing env fallback."""
    secrets: Dict[str, str] = {}
    try:
        secrets = store.decrypt_all()
    except Exception as exc:
        logger.warning("[explore_admin] could not decrypt credentials: %s", exc)
    for purpose, env_key in _PURPOSE_TO_ENV.items():
        db_value = str(secrets.get(purpose, "") or "").strip()
        if db_value:
            os.environ[env_key] = db_value
            logger.info("[explore_admin] applied %s from encrypted DB", env_key)
            continue
        env_value = os.getenv(env_key, "").strip()
        if env_value:
            logger.info("[explore_admin] using %s from environment fallback", env_key)


def _failed_login_rate_limited(remote: str) -> bool:
    """True if this remote has too many failed logins in the window."""
    now = time.monotonic()
    times = _FAILED_LOGINS.setdefault(remote, [])
    times[:] = [t for t in times if now - t < _RATE_WINDOW_S]
    return len(times) >= _RATE_MAX


def _record_failed_login(remote: str) -> None:
    now = time.monotonic()
    times = _FAILED_LOGINS.setdefault(remote, [])
    times[:] = [t for t in times if now - t < _RATE_WINDOW_S]
    times.append(now)


def _put_rate_limited(remote: str) -> bool:
    now = time.monotonic()
    times = _PUT_ATTEMPTS.setdefault(remote, [])
    times[:] = [t for t in times if now - t < _RATE_WINDOW_S]
    if len(times) >= _RATE_MAX:
        return True
    times.append(now)
    return False


def _require_admin(request: "web.Request") -> Optional["web.Response"]:
    from aiohttp import web

    from gateway.control_plane import admin_auth

    tok = admin_auth.extract_session_from_request(request.headers)
    if not admin_auth.verify_session_token(tok):
        return web.json_response({"error": "unauthorized"}, status=401)
    return None


def setup_explore_control_plane(app: "web.Application") -> None:
    from aiohttp import web

    from gateway.control_plane.admin_auth import (
        auth_file_exists,
        bootstrap_password,
        create_session_token,
        verify_password,
    )
    from gateway.control_plane.credential_store import ExploreCredentialStore
    from gateway.control_plane.dek import load_dek

    try:
        dek = load_dek()
        store = ExploreCredentialStore(dek)
    except Exception as exc:
        logger.warning("[explore_admin] control plane disabled: %s", exc)
        return

    app["explore_credential_store"] = store
    app["explore_control_plane_enabled"] = True

    async def handle_status(request: web.Request) -> web.Response:
        return web.json_response({"bootstrapped": auth_file_exists()})

    async def handle_bootstrap(request: web.Request) -> web.Response:
        if auth_file_exists():
            return web.json_response({"error": "already_bootstrapped"}, status=400)
        try:
            body = await request.json()
        except Exception:
            return web.json_response({"error": "invalid_json"}, status=400)
        pw = str(body.get("password") or "")
        if len(pw) < 8:
            return web.json_response({"error": "password_too_short"}, status=400)
        try:
            bootstrap_password(pw)
        except FileExistsError:
            return web.json_response({"error": "already_bootstrapped"}, status=400)
        token = create_session_token()
        resp = web.json_response({"ok": True, "session_token": token})
        resp.set_cookie(
            "explore_admin_session",
            token,
            httponly=True,
            max_age=86400,
            samesite="Lax",
            path="/",
        )
        return resp

    async def handle_login(request: web.Request) -> web.Response:
        remote = request.remote or "unknown"
        if _failed_login_rate_limited(remote):
            return web.json_response({"error": "rate_limited"}, status=429)
        try:
            body = await request.json()
        except Exception:
            return web.json_response({"error": "invalid_json"}, status=400)
        pw = str(body.get("password") or "")
        if not verify_password(pw):
            _record_failed_login(remote)
            return web.json_response({"error": "invalid_credentials"}, status=401)
        token = create_session_token()
        resp = web.json_response({"ok": True, "session_token": token})
        resp.set_cookie(
            "explore_admin_session",
            token,
            httponly=True,
            max_age=86400,
            samesite="Lax",
            path="/",
        )
        return resp

    async def handle_logout(request: web.Request) -> web.Response:
        resp = web.json_response({"ok": True})
        resp.del_cookie("explore_admin_session", path="/")
        return resp

    async def handle_get_credentials(request: web.Request) -> web.Response:
        err = _require_admin(request)
        if err is not None:
            return err
        st: ExploreCredentialStore = app["explore_credential_store"]
        rows = _effective_masked_credentials(st)
        return web.json_response({"purposes": rows})

    async def handle_put_credentials(request: web.Request) -> web.Response:
        err = _require_admin(request)
        if err is not None:
            return err
        remote = request.remote or "unknown"
        if _put_rate_limited(remote):
            return web.json_response({"error": "rate_limited"}, status=429)
        try:
            body = await request.json()
        except Exception:
            return web.json_response({"error": "invalid_json"}, status=400)
        keys = body.get("keys") or {}
        if not isinstance(keys, dict):
            return web.json_response({"error": "keys_must_be_object"}, status=400)
        st: ExploreCredentialStore = app["explore_credential_store"]
        for purpose, val in keys.items():
            if purpose not in _PURPOSE_TO_ENV:
                continue
            if val is None or val == "":
                continue
            if not isinstance(val, str):
                continue
            st.upsert(purpose, val)
        apply_stored_credentials_to_environ(st)
        rows = _effective_masked_credentials(st)
        return web.json_response({"purposes": rows})

    app.router.add_get("/v1/admin/status", handle_status)
    app.router.add_post("/v1/admin/bootstrap", handle_bootstrap)
    app.router.add_post("/v1/admin/login", handle_login)
    app.router.add_post("/v1/admin/logout", handle_logout)
    app.router.add_get("/v1/admin/credentials", handle_get_credentials)
    app.router.add_put("/v1/admin/credentials", handle_put_credentials)

    apply_stored_credentials_to_environ(store)
    logger.info("[explore_admin] control plane routes registered")
