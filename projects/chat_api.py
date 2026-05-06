"""
Chat control API handlers for the Hermes API server.

Provides REST endpoints for chat operations that ProjectChat depends on:
- inject: append a message to session transcript without triggering agent
- send: append a message and optionally trigger the agent
- abort: interrupt an in-flight agent run for a session
- history: load session transcript
- status: get session status (running/idle/error)
"""

import asyncio
import json
import logging
import time
import uuid
from typing import Any, Dict, List, Optional

try:
    from aiohttp import web

    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False
    web = None

logger = logging.getLogger(__name__)


def _json_response(data: Any, status: int = 200) -> "web.Response":
    return web.json_response(data, status=status)


def _error_response(message: str, status: int = 400, code: str = "bad_request") -> "web.Response":
    return web.json_response({"error": {"message": message, "code": code}}, status=status)


class ChatControlHandlers:
    """Handles /v1/chat/* API endpoints for project chat control."""

    def __init__(self, session_db=None, agent_pool=None):
        self._session_db = session_db
        self._agent_pool = agent_pool
        self._idempotency_cache: Dict[str, float] = {}

    def _get_session_db(self):
        if self._session_db is not None:
            return self._session_db
        try:
            from hermes_state import SessionDB

            self._session_db = SessionDB()
        except Exception as e:
            raise RuntimeError(f"SessionDB unavailable: {e}")
        return self._session_db

    # ------------------------------------------------------------------
    # POST /v1/chat/inject
    # ------------------------------------------------------------------

    async def handle_inject(self, request: "web.Request") -> "web.Response":
        body = await request.json()
        session_key = body.get("session_key")
        role = body.get("role", "system")
        content = body.get("content", "")
        label = body.get("label", "")

        if not session_key:
            return _error_response("session_key is required")

        db = self._get_session_db()

        session_id = self._resolve_session_id(db, session_key)
        if not session_id:
            return _error_response(f"Session not found for key: {session_key}", 404, "not_found")

        display_content = content
        if label:
            display_content = f"[{label}]\n{content}"

        db.append_message(
            session_id=session_id,
            role=role,
            content=display_content,
        )
        logger.debug("[chat] Injected %s message into session %s", role, session_key)
        return _json_response({"injected": True, "session_key": session_key})

    # ------------------------------------------------------------------
    # POST /v1/chat/send
    # ------------------------------------------------------------------

    async def handle_send(self, request: "web.Request") -> "web.Response":
        body = await request.json()
        session_key = body.get("session_key")
        content = body.get("content", "")
        deliver = body.get("deliver", True)
        idempotency_key = body.get("idempotency_key")

        if not session_key:
            return _error_response("session_key is required")
        if not content:
            return _error_response("content is required")

        if idempotency_key:
            if idempotency_key in self._idempotency_cache:
                return _json_response(
                    {"skipped": True, "reason": "idempotency_key already processed"}
                )
            self._idempotency_cache[idempotency_key] = time.time()
            self._purge_idempotency_cache()

        db = self._get_session_db()

        session_id = self._resolve_session_id(db, session_key)
        if not session_id:
            session_id = self._create_project_session(db, session_key)

        if not deliver:
            db.append_message(
                session_id=session_id,
                role="user",
                content=content,
            )
            logger.info("[chat] Stored user message to session %s (deliver=%s)", session_key, deliver)
            return _json_response({"sent": True, "deliver": False, "session_key": session_key})

        logger.info("[chat] Sent user message to agent for session %s (deliver=%s)", session_key, deliver)

        if self._agent_pool is None:
            return _json_response(
                {
                    "sent": True,
                    "deliver": True,
                    "session_key": session_key,
                    "note": "agent_pool not configured; message appended but agent not triggered",
                }
            )

        try:
            loop = asyncio.get_running_loop()
            result = await asyncio.wait_for(
                loop.run_in_executor(
                    None,
                    lambda: self._agent_pool.send_message(
                        session_key=session_key,
                        user_message=content,
                        session_id=session_id,
                    ),
                ),
                timeout=45.0,
            )
        except asyncio.TimeoutError:
            logger.warning("[chat] Agent timed out after 45s for session %s", session_key)
            self._agent_pool.interrupt_agent(session_key)
            return _json_response(
                {
                    "sent": True,
                    "deliver": True,
                    "session_key": session_key,
                    "response": "(Still working on your request — the agent is busy with a long task. Your message was added and will be addressed.)",
                }
            )
        except Exception as e:
            logger.error(
                "[chat] Agent execution failed for session %s: %s", session_key, e, exc_info=True
            )
            return _json_response(
                {
                    "sent": True,
                    "deliver": True,
                    "session_key": session_key,
                    "error": str(e),
                },
                status=500,
            )

        final_text = ""
        usage = {}
        if isinstance(result, dict):
            final_text = (
                result.get("result", {}).get("final_response", "")
                if isinstance(result.get("result"), dict)
                else result.get("final_response", "")
            )
            if not final_text and isinstance(result.get("result"), dict):
                final_text = result["result"].get("final_response", "") or ""
            usage = result.get("usage", {})

        return _json_response(
            {
                "sent": True,
                "deliver": True,
                "session_key": session_key,
                "response": final_text,
                "usage": usage,
            }
        )

    # ------------------------------------------------------------------
    # POST /v1/chat/abort
    # ------------------------------------------------------------------

    async def handle_abort(self, request: "web.Request") -> "web.Response":
        body = await request.json()
        session_key = body.get("session_key")
        reason = body.get("reason", "operator_abort")

        if not session_key:
            return _error_response("session_key is required")

        if self._agent_pool is None:
            return _error_response("agent_pool not configured", 501, "not_implemented")

        interrupted = self._agent_pool.interrupt_agent(session_key)
        logger.info(
            "[chat] Abort requested for session %s (reason=%s, interrupted=%s)",
            session_key,
            reason,
            interrupted,
        )
        return _json_response(
            {
                "aborted": interrupted,
                "session_key": session_key,
                "reason": reason,
            }
        )

    # ------------------------------------------------------------------
    # GET /v1/chat/history
    # ------------------------------------------------------------------

    async def handle_history(self, request: "web.Request") -> "web.Response":
        session_key = request.query.get("session_key")
        limit = int(request.query.get("limit", "100"))
        offset = int(request.query.get("offset", "0"))

        if not session_key:
            return _error_response("session_key is required")

        db = self._get_session_db()
        session_id = self._resolve_session_id(db, session_key)

        if not session_id:
            return _json_response({"messages": [], "total": 0})

        messages = db.get_messages(session_id)
        total = len(messages)
        sliced = messages[offset : offset + limit]

        return _json_response(
            {
                "messages": sliced,
                "total": total,
                "session_key": session_key,
            }
        )

    # ------------------------------------------------------------------
    # GET /v1/chat/status
    # ------------------------------------------------------------------

    async def handle_status(self, request: "web.Request") -> "web.Response":
        session_key = request.query.get("session_key")

        if not session_key:
            return _error_response("session_key is required")

        agent_active = False
        if self._agent_pool:
            agent_active = self._agent_pool.is_agent_active(session_key)

        status = "running" if agent_active else "idle"

        return _json_response(
            {
                "session_key": session_key,
                "status": status,
                "agent_active": agent_active,
            }
        )

    # ------------------------------------------------------------------
    # POST /v1/chat/sessions
    # ------------------------------------------------------------------

    async def handle_create_session(self, request: "web.Request") -> "web.Response":
        body = await request.json()
        session_key = body.get("session_key")
        display_name = body.get("display_name", "")

        if not session_key:
            return _error_response("session_key is required")

        db = self._get_session_db()
        session_id = self._resolve_session_id(db, session_key)

        if session_id:
            return _json_response(
                {
                    "session_id": session_id,
                    "session_key": session_key,
                    "created": False,
                }
            )

        session_id = self._create_project_session(db, session_key, display_name=display_name)
        return _json_response(
            {
                "session_id": session_id,
                "session_key": session_key,
                "created": True,
            },
            status=201,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _resolve_session_id(self, db, session_key: str) -> Optional[str]:
        session = db.get_session_by_title(session_key)
        if session:
            return session.get("id")
        return None

    def _create_project_session(self, db, session_key: str, display_name: str = "") -> str:
        session_id = f"eadproj-{uuid.uuid4().hex[:12]}"
        db.create_session(
            session_id=session_id,
            source="api_server",
            system_prompt="",
        )
        db.set_session_title(session_id, session_key)
        logger.info("[chat] Created project session %s for key %s", session_id, session_key)
        return session_id

    def _purge_idempotency_cache(self, max_age: float = 300.0) -> None:
        now = time.time()
        expired = [k for k, v in self._idempotency_cache.items() if now - v > max_age]
        for k in expired:
            del self._idempotency_cache[k]

    # ------------------------------------------------------------------
    # Route registration
    # ------------------------------------------------------------------

    def register_routes(self, app: "web.Application") -> None:
        app.router.add_post("/v1/chat/inject", self.handle_inject)
        app.router.add_post("/v1/chat/send", self.handle_send)
        app.router.add_post("/v1/chat/abort", self.handle_abort)
        app.router.add_get("/v1/chat/history", self.handle_history)
        app.router.add_get("/v1/chat/status", self.handle_status)
        app.router.add_post("/v1/chat/sessions", self.handle_create_session)
