"""
Session-bound agent pool for project chat.

Maintains AIAgent instances keyed by session_key so they can be:
- Sent new messages (auto-continue, resume)
- Interrupted by session key (abort)
- Polled for active status

Design: stateless with session persistence (Option B from the migration plan).
Each call creates a fresh AIAgent with the same session_id so it loads
its transcript from SessionDB. This avoids memory leaks from long-lived agents.
"""

import asyncio
import logging
import threading
import time
import uuid
from concurrent.futures import Future
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


class SessionAgentPool:
    """Manages agent execution per session key.

    Thread-safe: agent creation and execution happen via thread executor.
    """

    def __init__(self, adapter=None):
        self._adapter = adapter
        self._running_agents: Dict[str, Any] = {}
        self._running_futures: Dict[str, Future] = {}
        self._lock = threading.Lock()

    def _get_session_db(self):
        try:
            from hermes_state import SessionDB

            return SessionDB()
        except Exception as e:
            logger.error("[agent_pool] SessionDB unavailable: %s", e)
            return None

    def _create_agent(
        self,
        session_id: Optional[str] = None,
        ephemeral_system_prompt: Optional[str] = None,
        stream_delta_callback: Optional[Callable] = None,
        tool_progress_callback: Optional[Callable] = None,
        enabled_toolsets: Optional[List[str]] = None,
    ):
        if self._adapter is not None and hasattr(self._adapter, "_create_agent"):
            return self._adapter._create_agent(
                ephemeral_system_prompt=ephemeral_system_prompt,
                session_id=session_id,
                stream_delta_callback=stream_delta_callback,
                tool_progress_callback=tool_progress_callback,
            )

        from run_agent import AIAgent

        return AIAgent(
            session_id=session_id,
            ephemeral_system_prompt=ephemeral_system_prompt,
            stream_delta_callback=stream_delta_callback,
            tool_progress_callback=tool_progress_callback,
            enabled_toolsets=enabled_toolsets,
        )

    def _get_conversation_history(self, session_id: str) -> List[Dict[str, Any]]:
        db = self._get_session_db()
        if not db:
            return []
        try:
            return db.get_messages_as_conversation(session_id)
        except Exception as e:
            logger.warning("[agent_pool] Failed to load conversation history: %s", e)
            return []

    def _resolve_session_id(self, session_key: str) -> Optional[str]:
        db = self._get_session_db()
        if not db:
            return None
        try:
            session = db.get_session_by_title(session_key)
            if session:
                return session.get("id")
        except Exception:
            pass
        return None

    def send_message(
        self,
        session_key: str,
        user_message: str,
        session_id: Optional[str] = None,
        ephemeral_system_prompt: Optional[str] = None,
        enable_tools: bool = False,
    ) -> Dict[str, Any]:
        """Send a message to the session's agent synchronously. Blocks until complete."""
        if not session_id:
            session_id = self._resolve_session_id(session_key)

        conversation_history = []
        if session_id:
            conversation_history = self._get_conversation_history(session_id)

        tool_sets = ["project"] if enable_tools else []
        agent = self._create_agent(
            session_id=session_id,
            ephemeral_system_prompt=ephemeral_system_prompt,
            enabled_toolsets=tool_sets,
        )

        with self._lock:
            self._running_agents[session_key] = agent

        try:
            result = agent.run_conversation(
                user_message=user_message,
                conversation_history=conversation_history,
                task_id=f"eadproj:{session_key}",
            )
            final_response = ""
            completed = False
            if isinstance(result, dict):
                final_response = str(result.get("final_response") or "").strip()
                completed = bool(result.get("completed"))
                if not final_response:
                    err = str(result.get("error") or "").strip()
                    if err:
                        lower_err = err.lower()
                        # Treat provider output-length truncation as recoverable.
                        # The executor will issue follow-up turns automatically.
                        if "truncated due to output length limit" in lower_err or (
                            "truncated" in lower_err and "output length" in lower_err
                        ):
                            logger.warning(
                                "[agent_pool] Session %s hit output truncation; deferring recovery to next turn",
                                session_key,
                            )
                            usage = {
                                "input_tokens": getattr(agent, "session_prompt_tokens", 0) or 0,
                                "output_tokens": getattr(agent, "session_completion_tokens", 0) or 0,
                                "total_tokens": getattr(agent, "session_total_tokens", 0) or 0,
                            }
                            return {"result": result, "usage": usage}
                        raise RuntimeError(err)
                    if not completed:
                        raise RuntimeError(
                            "Agent run did not complete and produced no final response."
                        )
            usage = {
                "input_tokens": getattr(agent, "session_prompt_tokens", 0) or 0,
                "output_tokens": getattr(agent, "session_completion_tokens", 0) or 0,
                "total_tokens": getattr(agent, "session_total_tokens", 0) or 0,
            }
            return {"result": result, "usage": usage}
        finally:
            with self._lock:
                self._running_agents.pop(session_key, None)

    def send_message_async(
        self,
        session_key: str,
        user_message: str,
        session_id: Optional[str] = None,
        ephemeral_system_prompt: Optional[str] = None,
        enable_tools: bool = False,
    ) -> str:
        """Send a message asynchronously. Returns a run_id for tracking."""
        run_id = f"eadrun_{uuid.uuid4().hex[:12]}"

        def _run():
            try:
                result = self.send_message(
                    session_key=session_key,
                    user_message=user_message,
                    session_id=session_id,
                    ephemeral_system_prompt=ephemeral_system_prompt,
                    enable_tools=enable_tools,
                )
                logger.info(
                    "[agent_pool] Async run %s completed for session %s", run_id, session_key
                )
                return result
            except Exception as e:
                try:
                    sid = session_id or self._resolve_session_id(session_key)
                    db = self._get_session_db()
                    if sid and db:
                        raw_err = str(e).strip()
                        lowered = raw_err.lower()
                        is_credit_error = (
                            "402" in lowered
                            or "insufficient balance" in lowered
                            or "quota" in lowered
                            or "credit" in lowered
                        )
                        no_final_response = (
                            "did not complete and produced no final response" in lowered
                        )
                        # Surface backend model/provider failures to end users with actionable guidance.
                        if is_credit_error:
                            user_msg = (
                                "⚠️ Agent execution error: model/provider call failed.\n"
                                f"Details: {raw_err[:260]}\n"
                                "Likely cause: provider credits/quota exhausted. "
                                "Please verify balance/quota and model access."
                            )
                        elif no_final_response:
                            user_msg = (
                                "⚠️ Agent execution error: provider returned no final response.\n"
                                f"Details: {raw_err[:260]}\n"
                                "Likely cause: transient provider timeout/stream interruption. "
                                "Please retry the run or switch provider/model."
                            )
                        else:
                            user_msg = (
                                "⚠️ Agent execution error: model/provider call failed.\n"
                                f"Details: {raw_err[:260]}\n"
                                "Please retry. If this persists, verify provider endpoint/model configuration."
                            )
                        db.append_message(
                            session_id=sid,
                            role="assistant",
                            content=user_msg,
                        )
                except Exception:
                    logger.warning(
                        "[agent_pool] Failed to append user-facing error message for session %s",
                        session_key,
                    )
                logger.error(
                    "[agent_pool] Async run %s failed for session %s: %s",
                    run_id,
                    session_key,
                    e,
                    exc_info=True,
                )
                raise

        loop = None
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            pass

        if loop:
            future = loop.run_in_executor(None, _run)
        else:
            executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
            future = executor.submit(_run)

        if future:
            with self._lock:
                self._running_futures[session_key] = future

        logger.info("[agent_pool] Started async run %s for session %s", run_id, session_key)
        return run_id

    def interrupt_agent(self, session_key: str) -> bool:
        """Interrupt the running agent for a session. Returns True if agent was active."""
        with self._lock:
            agent = self._running_agents.get(session_key)

        if agent and hasattr(agent, "interrupt"):
            agent.interrupt("Project run aborted")
            logger.info("[agent_pool] Interrupted agent for session %s", session_key)
            return True

        return False

    def is_agent_active(self, session_key: str) -> bool:
        """Check if a session has an actively running agent."""
        with self._lock:
            return session_key in self._running_agents

    def cleanup_session(self, session_key: str) -> None:
        """Remove agent and resources for a session."""
        with self._lock:
            self._running_agents.pop(session_key, None)
            future = self._running_futures.pop(session_key, None)

        if future and not future.done():
            try:
                future.cancel()
            except Exception:
                pass
