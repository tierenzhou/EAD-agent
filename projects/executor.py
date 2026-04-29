"""
Project execution orchestrator.

Background asyncio process that monitors execution state, drives auto-continue,
extracts progress from session transcripts, and handles failure recovery.

Port of src/projects/executor.ts from EAD-EXP.
"""

import asyncio
import json
import logging
import time
from typing import Dict, List, Optional

from .models import ExecutionStatus, ProgressLogEntry, ProjectExecute, StepResult
from .store import ProjectStore
from .agent_pool import SessionAgentPool


def _cleanup_browser_for_session(session_key: str) -> None:
    """Clean up agent-browser daemon and Chrome for a session."""
    try:
        from tools.browser_tool import cleanup_browser

        task_id = f"eadproj:{session_key}"
        cleanup_browser(task_id)
        logger.info("[executor] Browser cleaned up for %s", task_id)
    except Exception as e:
        logger.warning("[executor] Browser cleanup failed for %s: %s", session_key, e)


logger = logging.getLogger(__name__)

_MAX_AUTO_CONTINUES = 200
_COOLDOWN_BASE_S = 2.5
_COOLDOWN_STEP_S = 2.5
_COOLDOWN_MAX_S = 60.0
_SUCCESS_COUNT_TO_DECREASE = 5
_RECOVERY_WINDOW_S = 600
_POLL_INTERVAL_S = 1.2
_PROGRESS_REPORT_INTERVAL_S = 90


class ProjectExecutor:
    """Background executor for project runs."""

    def __init__(
        self,
        project_store: ProjectStore,
        agent_pool: SessionAgentPool,
    ):
        self._store = project_store
        self._pool = agent_pool
        self._running: Dict[str, asyncio.Task] = {}
        self._abort_events: Dict[str, asyncio.Event] = {}
        self._auto_continue_counts: Dict[str, int] = {}
        self._last_continue_ts: Dict[str, float] = {}
        self._last_progress_report_ts: Dict[str, float] = {}
        self._cancelled: set = set()
        self._cooldown_s: Dict[str, float] = {}
        self._success_streak: Dict[str, int] = {}

    async def start_execution(self, execution_id: str) -> None:
        execution = self._store.get_execution(execution_id)
        if not execution:
            logger.error("[executor] Execution %s not found", execution_id)
            return

        self._store.update_execution(
            execution_id,
            status=ExecutionStatus.RUNNING,
            start_time=int(time.time() * 1000),
        )
        self._auto_continue_counts[execution_id] = 0
        self._abort_events[execution_id] = asyncio.Event()
        self._cooldown_s[execution_id] = _COOLDOWN_BASE_S
        self._success_streak[execution_id] = 0

        session_key = execution.run_session_key
        if session_key:
            logger.info(
                "[executor] Sending initial task for execution %s (session=%s)",
                execution_id,
                session_key,
            )
            self._pool.send_message_async(
                session_key=session_key,
                user_message=(
                    "Please review the task above carefully, then confirm your "
                    "understanding by summarizing what you will do step by step. "
                    "Do NOT start navigating or using tools yet — just confirm "
                    "your plan first."
                ),
                enable_tools=False,
            )

        task = asyncio.create_task(self._monitor_loop(execution_id))
        self._running[execution_id] = task
        logger.info("[executor] Started monitoring execution %s", execution_id)

    async def _monitor_loop(self, execution_id: str) -> None:
        while True:
            try:
                execution = self._store.get_execution(execution_id)
                if not execution:
                    logger.warning("[executor] Execution %s disappeared", execution_id)
                    break

                if execution.status in (
                    ExecutionStatus.COMPLETED,
                    ExecutionStatus.CANCELLED,
                    ExecutionStatus.ERROR,
                ):
                    logger.info(
                        "[executor] Execution %s is terminal: %s",
                        execution_id,
                        execution.status.value,
                    )
                    break

                if self._abort_events.get(execution_id, asyncio.Event()).is_set():
                    logger.info("[executor] Abort signal for %s", execution_id)
                    break

                if execution.paused:
                    await asyncio.sleep(_POLL_INTERVAL_S)
                    continue

                if execution_id in self._cancelled:
                    logger.info("[executor] Execution %s was cancelled, skipping", execution_id)
                    break

                session_key = execution.run_session_key
                if session_key:
                    agent_active = self._pool.is_agent_active(session_key)

                    await self._extract_progress(execution_id)

                    if not agent_active and execution.status == ExecutionStatus.RUNNING:
                        await self._auto_continue(execution_id)

                    await self._check_budget(execution_id)
                    await self._report_progress_to_chat(execution_id, session_key)

                await asyncio.sleep(_POLL_INTERVAL_S)

            except asyncio.CancelledError:
                logger.info("[executor] Monitor loop cancelled for %s", execution_id)
                break
            except Exception as e:
                logger.error("[executor] Error in monitor loop for %s: %s", execution_id, e)
                await asyncio.sleep(5)

        execution = self._store.get_execution(execution_id)
        sk = execution.run_session_key if execution else None

        self._running.pop(execution_id, None)
        self._abort_events.pop(execution_id, None)
        self._auto_continue_counts.pop(execution_id, None)
        self._last_continue_ts.pop(execution_id, None)
        self._last_progress_report_ts.pop(execution_id, None)
        self._cancelled.discard(execution_id)
        self._cooldown_s.pop(execution_id, None)
        self._success_streak.pop(execution_id, None)

        if sk:
            _cleanup_browser_for_session(sk)

    def _get_cooldown(self, execution_id: str) -> float:
        return self._cooldown_s.get(execution_id, _COOLDOWN_BASE_S)

    def _on_rate_limit(self, execution_id: str) -> None:
        current = self._cooldown_s.get(execution_id, _COOLDOWN_BASE_S)
        new_cooldown = min(current + _COOLDOWN_STEP_S, _COOLDOWN_MAX_S)
        self._cooldown_s[execution_id] = new_cooldown
        self._success_streak[execution_id] = 0
        logger.info(
            "[executor] Rate limit detected for %s — cooldown increased to %.1fs",
            execution_id,
            new_cooldown,
        )

    def _on_step_success(self, execution_id: str) -> None:
        streak = self._success_streak.get(execution_id, 0) + 1
        self._success_streak[execution_id] = streak
        if streak >= _SUCCESS_COUNT_TO_DECREASE and streak % _SUCCESS_COUNT_TO_DECREASE == 0:
            current = self._cooldown_s.get(execution_id, _COOLDOWN_BASE_S)
            new_cooldown = max(current - _COOLDOWN_STEP_S, _COOLDOWN_BASE_S)
            if new_cooldown < current:
                self._cooldown_s[execution_id] = new_cooldown
                logger.info(
                    "[executor] %d consecutive successes for %s — cooldown reduced to %.1fs",
                    streak,
                    execution_id,
                    new_cooldown,
                )

    def _detect_rate_limit_in_progress_log(self, execution_id: str) -> bool:
        execution = self._store.get_execution(execution_id)
        if not execution or not execution.progress_log:
            return False
        recent = (
            execution.progress_log[-5:]
            if len(execution.progress_log) >= 5
            else execution.progress_log
        )
        for entry in reversed(recent):
            text = (entry.text or "").lower()
            if "rate limit" in text or "ratelimit" in text or "tpm" in text:
                return True
        return False

    async def _auto_continue(self, execution_id: str) -> None:
        if execution_id in self._cancelled:
            return

        count = self._auto_continue_counts.get(execution_id, 0)
        if count >= _MAX_AUTO_CONTINUES:
            logger.info("[executor] Max auto-continues reached for %s", execution_id)
            self._store.update_execution(
                execution_id,
                status=ExecutionStatus.COMPLETED,
                cancel_reason="max_auto_continues_reached",
            )
            return

        cooldown = self._get_cooldown(execution_id)
        last_ts = self._last_continue_ts.get(execution_id, 0)
        if time.time() - last_ts < cooldown:
            return

        execution = self._store.get_execution(execution_id)
        if not execution:
            return

        if execution.first_failed_at:
            elapsed = (time.time() * 1000) - execution.first_failed_at
            if elapsed > _RECOVERY_WINDOW_S * 1000:
                logger.info("[executor] Recovery window expired for %s", execution_id)
                self._store.update_execution(
                    execution_id,
                    status=ExecutionStatus.FAILED,
                    last_error_message="Recovery window expired",
                )
                return

        session_key = execution.run_session_key
        if not session_key:
            return

        if self._detect_rate_limit_in_progress_log(execution_id):
            self._on_rate_limit(execution_id)
        else:
            self._on_step_success(execution_id)

        current_cooldown = self._get_cooldown(execution_id)
        self._auto_continue_counts[execution_id] = count + 1
        self._last_continue_ts[execution_id] = time.time()

        logger.info(
            "[executor] Auto-continue #%d for %s (cooldown=%.1fs)",
            count + 1,
            execution_id,
            current_cooldown,
        )

        pace_hint = ""
        if current_cooldown > _COOLDOWN_BASE_S * 2:
            pace_hint = (
                f" Pace yourself — wait ~{current_cooldown:.0f}s between actions "
                f"to avoid hitting API rate limits."
            )

        self._pool.send_message_async(
            session_key=session_key,
            user_message=(
                "Good, your plan looks correct. Now please start executing — "
                "begin navigating the site and working through your plan." + pace_hint
                if count == 0
                else "Continue exploring. Look for more edge cases and untested areas." + pace_hint
            ),
            enable_tools=True,
        )

    async def _check_budget(self, execution_id: str) -> None:
        execution = self._store.get_execution(execution_id)
        if not execution or not execution.start_time:
            return

        elapsed_ms = int(time.time() * 1000) - execution.start_time

        if execution.time_budget_minutes:
            budget_ms = execution.time_budget_minutes * 60 * 1000
            if elapsed_ms > budget_ms:
                logger.info("[executor] Time budget exceeded for %s", execution_id)
                self._store.update_execution(
                    execution_id,
                    status=ExecutionStatus.COMPLETED,
                    cancel_reason="time_budget_exceeded",
                    duration_ms=elapsed_ms,
                )

    async def _extract_progress(self, execution_id: str) -> None:
        execution = self._store.get_execution(execution_id)
        if not execution or not execution.run_session_key:
            return

        db = self._pool._get_session_db()
        if not db:
            return

        session_id = self._pool._resolve_session_id(execution.run_session_key)
        if not session_id:
            return

        try:
            messages = db.get_messages(session_id)
        except Exception:
            return

        seq = execution.progress_log_seq or 0
        new_messages = messages[seq:]
        if not new_messages:
            return

        progress_entries = []
        for msg in new_messages:
            role = msg.get("role", "")
            content = msg.get("content", "") or ""
            tool_name = msg.get("tool_name")
            tool_input = None

            if role == "assistant" and msg.get("tool_calls"):
                for tc in msg["tool_calls"]:
                    if isinstance(tc, dict) and tc.get("function"):
                        tool_name = tc["function"].get("name", "")
                        try:
                            tool_input = json.loads(tc["function"].get("arguments", "{}"))
                        except (json.JSONDecodeError, TypeError):
                            tool_input = {}
                        progress_entries.append(
                            ProgressLogEntry(
                                kind="tool_use",
                                text=content[:200] if content else tool_name,
                                tool_name=tool_name,
                                tool_input=tool_input,
                            )
                        )
            elif role == "tool":
                progress_entries.append(
                    ProgressLogEntry(
                        kind="tool_result",
                        text=content[:200],
                        tool_name=tool_name,
                    )
                )
            elif role == "assistant":
                progress_entries.append(
                    ProgressLogEntry(
                        kind="assistant",
                        text=content[:200],
                    )
                )

        if progress_entries:
            current_log = execution.progress_log or []
            updated_log = current_log + progress_entries
            msg_count = len(messages)
            progress_pct = min(15 + msg_count * 3, 92)

            self._store.update_execution(
                execution_id,
                progress_log=updated_log,
                progress_log_seq=msg_count,
                progress_percentage=progress_pct,
            )

    async def _report_progress_to_chat(self, execution_id: str, session_key: str) -> None:
        now = time.time()
        last_report = self._last_progress_report_ts.get(execution_id, 0)
        if now - last_report < _PROGRESS_REPORT_INTERVAL_S:
            return

        self._last_progress_report_ts[execution_id] = now
        execution = self._store.get_execution(execution_id)
        if not execution:
            return

        elapsed_min = (now * 1000 - (execution.start_time or 0)) / 60000
        pct = execution.progress_percentage or 0
        steps_total = getattr(execution, "total_test_steps", None) or 0
        steps_done = (
            (getattr(execution, "succeeded_test_steps", None) or 0)
            + (getattr(execution, "failed_test_steps", None) or 0)
            + (getattr(execution, "skipped_test_steps", None) or 0)
        )

        db = self._pool._get_session_db()
        if not db:
            return
        session_id = self._pool._resolve_session_id(session_key)
        if not session_id:
            return

        progress_text = (
            f"📊 Progress Update ({elapsed_min:.1f}m elapsed)\n"
            f"Status: {execution.status.value}\n"
            f"Completion: ~{pct}%\n"
        )
        if steps_total > 0:
            progress_text += f"Steps: {steps_done}/{steps_total} completed\n"
        if execution.progress_log:
            recent = [e for e in execution.progress_log if e.kind == "tool_use"][-3:]
            if recent:
                progress_text += "Recent actions:\n"
                for e in recent:
                    progress_text += f"  • {e.text}\n"

        try:
            db.append_message(
                session_id=session_id,
                role="assistant",
                content=progress_text,
            )
            logger.info("[executor] Progress reported for %s (%.0f%%)", execution_id, pct)
        except Exception as e:
            logger.warning("[executor] Failed to report progress for %s: %s", execution_id, e)

    async def cancel_execution(self, execution_id: str) -> None:
        self._cancelled.add(execution_id)
        event = self._abort_events.get(execution_id)
        if event:
            event.set()

        execution = self._store.get_execution(execution_id)
        if execution and execution.run_session_key:
            sk = execution.run_session_key
            self._pool.interrupt_agent(sk)
            self._pool.cleanup_session(sk)
            _cleanup_browser_for_session(sk)

        self._store.update_execution(
            execution_id,
            status=ExecutionStatus.CANCELLED,
            paused=False,
        )
        logger.info("[executor] Cancelled execution %s", execution_id)

    async def pause_execution(self, execution_id: str) -> None:
        execution = self._store.get_execution(execution_id)
        if execution and execution.run_session_key:
            self._pool.interrupt_agent(execution.run_session_key)

        self._store.update_execution(execution_id, paused=True)
        logger.info("[executor] Paused execution %s", execution_id)

    async def resume_execution(self, execution_id: str) -> None:
        self._store.update_execution(execution_id, paused=False)
        logger.info("[executor] Resumed execution %s", execution_id)

    async def resume_active_projects(self) -> None:
        active = self._store.get_active_executions()
        for execution in active:
            if execution.status == ExecutionStatus.RUNNING:
                logger.info("[executor] Resuming execution %s", execution.id)
                await self.start_execution(execution.id)
