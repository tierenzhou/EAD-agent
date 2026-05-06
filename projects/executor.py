"""
Project execution orchestrator.

Background asyncio process that monitors execution state, drives auto-continue,
extracts progress from session transcripts, and handles failure recovery.

Port of src/projects/executor.ts from EAD-EXP.
"""

import asyncio
import hashlib
import json
import logging
import os
import re
import shutil
import time
from pathlib import Path
from typing import Dict, List, Optional

from .models import ExecutionStatus, ProgressLogEntry, ProjectExecute, StepResult
from .store import ProjectStore
from .agent_pool import SessionAgentPool
from .pfm_artifacts import (
    build_and_persist_pfm_artifacts,
    derive_node_runs_from_progress,
    report_file_path,
)


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
_PROGRESS_REPORT_INTERVAL_S = 15
_STALL_INTERRUPT_S = 90.0

_LOGIN_PHASE_PROMPT = (
    "LOGIN PHASE (MANDATORY, FIRST):\n"
    "1) Your first objective is successful login only. Do NOT start PFM discovery yet.\n"
    "2) Find a viable login path autonomously (UI clues, navigation options, auth hints, fallback routes).\n"
    "3) For each meaningful attempt, call report_running_step and include:\n"
    "   - title: 'Login PFM map (Interactive)'\n"
    "   - login_phase_status: pending | failed | success\n"
    "   - login_success: true/false\n"
    "   - observation, finding, reasoning, decision, next_direction\n"
    "4) Keep iterating until login succeeds or strong evidence indicates a blocker.\n"
    "5) Only after login_phase_status=success may you send pfm_node updates.\n"
    "6) If blocked, explain exactly why and propose the next best login strategy.\n"
)

_ACTIVITY_MESSAGE_PROMPT = (
    "ACTIVITY MESSAGE STYLE (MANDATORY FOR report_running_step):\n"
    "- Write for an end user, not as raw agent/debug output.\n"
    "- Translate your reasoning into clear value language the user can understand and appreciate.\n"
    "- Every screenshot must be attached to a meaningful activity explanation; never attach a screenshot with only 'Screenshot captured', 'Recovered screenshot', or empty text.\n"
    "- Include these ideas naturally in title/observation/finding/reasoning/decision/next_direction:\n"
    "  1) What the agent just did.\n"
    "  2) What value or business/user benefit that proved.\n"
    "  3) Why it matters to the user, business process, or validation flow.\n"
    "  4) What changed, passed, failed, or was confirmed.\n"
    "- Example tone: 'Verified Employee Portal login and dashboard access. This confirms users can reach the core service entry point before deeper feature discovery begins.'\n"
)

_LOGIN_PROGRESS_PROMPT = (
    "LOGIN STEP REPORTING:\n"
    "- State your login plan briefly before/while trying.\n"
    "- For each attempt, report what you saw, what you did, why, and next move.\n"
    "- Clearly report whether login succeeded and what evidence confirms success.\n"
    "- Capture screenshots frequently during meaningful UI transitions so end users can follow progress visually.\n"
    "- When a screenshot is available, include it in report_running_step.thumbnail_urls.\n"
    + _ACTIVITY_MESSAGE_PROMPT
)

_PFM_STRUCTURE_PROMPT = (
    "PFM STRUCTURE RULES (for viewable mindmap):\n"
    "- Return pfm_node in multi-level hierarchy (domain -> feature group -> atomic function -> action).\n"
    "- Keep each hierarchy level under 20 sibling nodes.\n"
    "- Provide brief title and brief description for each node.\n"
    "- Keep title/description concise; avoid long paragraphs.\n"
    "- Include parent_node_key and level when reporting child nodes.\n"
    "- During exploration, capture screenshots frequently and attach available screenshots via report_running_step.thumbnail_urls.\n"
    + _ACTIVITY_MESSAGE_PROMPT
)

_PROGRESS_MESSAGE_PREFIX = "📊 Progress Update"
_IMAGE_EXT_RE = re.compile(r"\.(png|jpe?g|gif|webp)(?:$|[?#])", re.IGNORECASE)
_IMAGE_BASENAME_RE = re.compile(
    r"\b[\w.-]*?(?:screenshot|screen|image)[\w.-]*?\.(?:png|jpe?g|gif|webp)\b",
    re.IGNORECASE,
)
_MIN_BROWSER_SCREENSHOT_BYTES = 10_000


def _hermes_home() -> Path:
    configured = os.getenv("HERMES_HOME", "").strip()
    return Path(configured).expanduser() if configured else Path.home() / ".hermes"


def _resolve_local_image_path(path_text: str) -> Path:
    src = Path(path_text).expanduser()
    if src.exists():
        return src
    if src.is_absolute() or src.name != path_text:
        return src
    for directory in (
        _hermes_home() / "cache" / "screenshots",
        _hermes_home() / "browser_screenshots",
    ):
        candidate = directory / src.name
        if candidate.exists():
            return candidate
    return src


def _is_usable_local_image(src: Path) -> bool:
    if not src.exists() or not src.is_file():
        return False
    if not _looks_like_image_ref(src.name):
        return False
    try:
        size = src.stat().st_size
    except OSError:
        return False
    if "browser_screenshot_" in src.name and size < _MIN_BROWSER_SCREENSHOT_BYTES:
        return False
    return True


def _content_addressed_report_name(src: Path) -> Optional[str]:
    try:
        digest = hashlib.sha256(src.read_bytes()).hexdigest()[:16]
    except OSError:
        return None
    suffix = src.suffix.lower()
    if not suffix or not re.fullmatch(r"\.[a-z0-9]+", suffix):
        suffix = ".png"
    return f"live-shot-{digest}{suffix}"


def _looks_like_image_ref(value: str) -> bool:
    text = str(value or "").strip()
    if not text:
        return False
    if "\n" in text or "\r" in text or len(text) > 2048:
        return False
    if text.startswith(("http://", "https://")):
        return bool(_IMAGE_EXT_RE.search(text))
    if text.startswith("/"):
        return bool(_IMAGE_EXT_RE.search(text))
    return bool(_IMAGE_EXT_RE.search(text))


def _materialize_local_image_as_report(execution_id: str, image_path: str) -> Optional[str]:
    """Copy a local screenshot into the run report folder and return API URL."""
    path_text = str(image_path or "").strip()
    if not path_text:
        return None
    if "\n" in path_text or "\r" in path_text or len(path_text) > 2048:
        return None
    src = _resolve_local_image_path(path_text)
    if not _is_usable_local_image(src):
        return None
    target_name = _content_addressed_report_name(src)
    if not target_name:
        return None
    target = report_file_path(execution_id, target_name)
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            shutil.copy2(src, target)
    except Exception:
        return None
    return f"/v1/projects/executions/{execution_id}/reports/{target_name}"


def _extract_image_reference_from_payload(payload: object) -> Optional[str]:
    """Try to find an image URL/path in nested tool payloads."""
    if isinstance(payload, str):
        text = payload.strip()
        if _looks_like_image_ref(text):
            return text
        local_match = re.search(r"/[^\s\"']+\.(?:png|jpe?g|gif|webp)", text, re.IGNORECASE)
        if local_match:
            return local_match.group(0)
        remote_match = re.search(
            r"https?://[^\s\"']+\.(?:png|jpe?g|gif|webp)(?:[^\s\"']*)?",
            text,
            re.IGNORECASE,
        )
        if remote_match:
            return remote_match.group(0)
        basename_match = _IMAGE_BASENAME_RE.search(text)
        if basename_match:
            return basename_match.group(0)
        return None
    if isinstance(payload, dict):
        preferred_keys = (
            "thumbnail_url",
            "thumbnailUrl",
            "image_url",
            "imageUrl",
            "screenshot_url",
            "screenshotUrl",
            "screenshot_path",
            "screenshotPath",
            "path",
            "url",
        )
        for key in preferred_keys:
            if key in payload:
                hit = _extract_image_reference_from_payload(payload.get(key))
                if hit:
                    return hit
        for value in payload.values():
            hit = _extract_image_reference_from_payload(value)
            if hit:
                return hit
        return None
    if isinstance(payload, list):
        for item in payload:
            hit = _extract_image_reference_from_payload(item)
            if hit:
                return hit
    return None


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
        self._last_pfm_sync_seq: Dict[str, int] = {}
        self._cancelled: set = set()
        self._cooldown_s: Dict[str, float] = {}
        self._success_streak: Dict[str, int] = {}
        self._login_phase_prompt_sent: Dict[str, bool] = {}
        self._discovery_phase_prompt_sent: Dict[str, bool] = {}
        self._last_progress_seq_seen: Dict[str, int] = {}
        self._last_progress_activity_ts: Dict[str, float] = {}

    async def start_execution(self, execution_id: str) -> None:
        execution = self._store.get_execution(execution_id)
        if not execution:
            logger.error("[executor] Execution %s not found", execution_id)
            return

        self._store.update_execution(
            execution_id,
            status=ExecutionStatus.RUNNING,
            # Start timer only after login checkpoint is explicitly successful.
            start_time=None,
            progress_percentage=0,
            executor_hint="Login checkpoint in progress; timer starts after login success.",
        )
        self._auto_continue_counts[execution_id] = 0
        self._abort_events[execution_id] = asyncio.Event()
        self._cooldown_s[execution_id] = _COOLDOWN_BASE_S
        self._success_streak[execution_id] = 0
        self._login_phase_prompt_sent[execution_id] = False
        self._discovery_phase_prompt_sent[execution_id] = False
        self._last_progress_seq_seen[execution_id] = int(execution.progress_log_seq or 0)
        self._last_progress_activity_ts[execution_id] = time.time()

        # Auto-continue will send the one-time phase prompt with tools enabled.

        task = asyncio.create_task(self._monitor_loop(execution_id))
        self._running[execution_id] = task
        logger.info("[executor] Started monitoring execution %s", execution_id)

    async def resume_active_executions(self) -> None:
        """Reattach monitor loops for running/pending executions after process restart."""
        for execution in self._store.get_active_executions():
            execution_id = execution.id
            existing = self._running.get(execution_id)
            if existing and not existing.done():
                continue
            self._abort_events[execution_id] = asyncio.Event()
            self._cooldown_s.setdefault(execution_id, _COOLDOWN_BASE_S)
            self._success_streak.setdefault(execution_id, 0)
            self._login_phase_prompt_sent.setdefault(execution_id, False)
            self._discovery_phase_prompt_sent.setdefault(execution_id, False)
            task = asyncio.create_task(self._monitor_loop(execution_id))
            self._running[execution_id] = task
            logger.info("[executor] Resumed monitoring execution %s", execution_id)

    async def _monitor_loop(self, execution_id: str) -> None:
        while True:
            try:
                execution = self._store.get_execution(execution_id)
                if not execution:
                    logger.warning("[executor] Execution %s disappeared", execution_id)
                    break

                if execution.status in (
                    ExecutionStatus.COMPLETED,
                    ExecutionStatus.FAILED,
                    ExecutionStatus.CANCELLED,
                    ExecutionStatus.ERROR,
                ):
                    await self._sync_pfm_artifacts(execution_id, force=True)
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
                    if not execution.start_time and self._is_login_successful(execution):
                        execution = self._store.update_execution(
                            execution_id,
                            start_time=int(time.time() * 1000),
                            progress_percentage=0,
                            executor_hint="Exploring...",
                        ) or execution
                    agent_active = self._pool.is_agent_active(session_key)
                    execution = self._store.update_execution(
                        execution_id,
                        progress_percentage=self._compute_time_progress_percent(execution),
                    ) or execution

                    await self._extract_progress(execution_id)
                    await self._sync_pfm_artifacts(execution_id)

                    latest = self._store.get_execution(execution_id) or execution
                    current_seq = int(latest.progress_log_seq or 0)
                    prev_seq = int(self._last_progress_seq_seen.get(execution_id, 0))
                    if current_seq > prev_seq:
                        self._last_progress_seq_seen[execution_id] = current_seq
                        self._last_progress_activity_ts[execution_id] = time.time()

                    if agent_active:
                        last_activity = self._last_progress_activity_ts.get(
                            execution_id, time.time()
                        )
                        stalled_for = time.time() - last_activity
                        if stalled_for >= _STALL_INTERRUPT_S:
                            logger.warning(
                                "[executor] Stalled agent detected for %s (%.1fs). Interrupting and retrying phase prompt.",
                                execution_id,
                                stalled_for,
                            )
                            self._pool.interrupt_agent(session_key)
                            if self._is_login_successful(latest):
                                self._discovery_phase_prompt_sent[execution_id] = False
                            else:
                                self._login_phase_prompt_sent[execution_id] = False
                            self._last_progress_activity_ts[execution_id] = time.time()
                            self._store.update_execution(
                                execution_id,
                                executor_hint="No agent activity detected; retrying prompt...",
                            )
                            agent_active = False

                    if not agent_active and execution.status == ExecutionStatus.RUNNING:
                        await self._auto_continue(execution_id)

                    await self._check_budget(execution_id)
                    await self._report_progress_to_chat(execution_id, session_key)
                elif execution.status == ExecutionStatus.RUNNING:
                    # Fail fast if no session key was linked; otherwise run appears
                    # stuck in Initializing forever with no agent bridge.
                    self._store.update_execution(
                        execution_id,
                        status=ExecutionStatus.FAILED,
                        last_error_message="Run session bootstrap failed (missing run_session_key).",
                        executor_hint="AI Failed",
                    )
                    logger.error(
                        "[executor] Execution %s missing run_session_key; marked failed",
                        execution_id,
                    )
                    break

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
        self._last_pfm_sync_seq.pop(execution_id, None)
        self._cancelled.discard(execution_id)
        self._cooldown_s.pop(execution_id, None)
        self._success_streak.pop(execution_id, None)
        self._login_phase_prompt_sent.pop(execution_id, None)
        self._discovery_phase_prompt_sent.pop(execution_id, None)
        self._last_progress_seq_seen.pop(execution_id, None)
        self._last_progress_activity_ts.pop(execution_id, None)

        if sk:
            _cleanup_browser_for_session(sk)

    def _get_cooldown(self, execution_id: str) -> float:
        return self._cooldown_s.get(execution_id, _COOLDOWN_BASE_S)

    def _compute_time_progress_percent(self, execution: ProjectExecute, now_ms: Optional[int] = None) -> int:
        """Compute progress from elapsed runtime versus configured time budget."""
        if not execution.start_time:
            return 0
        if not execution.time_budget_minutes or execution.time_budget_minutes <= 0:
            # No time budget configured: keep existing tracked progress as fallback.
            return int(max(0, min(100, execution.progress_percentage or 0)))
        now_ms = now_ms if now_ms is not None else int(time.time() * 1000)
        elapsed_ms = max(0, now_ms - execution.start_time)
        budget_ms = int(execution.time_budget_minutes * 60 * 1000)
        if budget_ms <= 0:
            return int(max(0, min(100, execution.progress_percentage or 0)))
        return int(max(0, min(100, round((elapsed_ms / budget_ms) * 100))))

    def _format_mmss(self, ms: int) -> str:
        total_sec = max(0, int(ms / 1000))
        mins = total_sec // 60
        secs = total_sec % 60
        return f"{mins:02d}:{secs:02d}"

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

    def _login_phase_status_from_log(self, progress_log: Optional[List[ProgressLogEntry]]) -> str:
        """Return login phase status: missing | pending | success | failed."""
        seen_checkpoint = False
        for entry in reversed(progress_log or []):
            # Broader fallback: infer login success/failure from concrete tool results,
            # even when the agent doesn't call report_running_step.
            if entry.kind == "tool_result":
                txt = str(entry.text or "").strip().lower()
                if txt:
                    success_markers = (
                        "home dashboard",
                        "#/banner",
                        "\"url\": \"http://jiaxing.devsuite.cn/p1/#/banner",
                        "login was successful",
                        "login successful",
                        "\"access_token\"",
                    )
                    failed_markers = (
                        "invalid user name or password",
                        "authentication failed",
                        "failed to login",
                        "captcha failed",
                    )
                    if any(m in txt for m in success_markers):
                        return "success"
                    if any(m in txt for m in failed_markers):
                        seen_checkpoint = True
            if entry.kind == "tool_use":
                txt = str(entry.text or "").strip().lower()
                if txt:
                    success_markers = (
                        "login successful",
                        "login success",
                        "logged in",
                        "successfully logged",
                        "home dashboard",
                        "#/banner",
                        "access_token",
                    )
                    failed_markers = (
                        "login failed",
                        "invalid user name or password",
                        "invalid username",
                        "invalid password",
                        "authentication failed",
                    )
                    if any(m in txt for m in success_markers):
                        return "success"
                    if any(m in txt for m in failed_markers):
                        seen_checkpoint = True
            if entry.kind != "tool_use":
                continue
            if (entry.tool_name or "") != "report_running_step":
                continue
            tool_input = entry.tool_input or {}
            title = str(tool_input.get("title") or "").strip().lower()
            text_blob = " ".join(
                str(tool_input.get(k) or "")
                for k in ("description", "observation", "finding", "reasoning", "decision", "next_direction")
            ).strip().lower()
            is_login_related = (
                "login pfm map (interactive)" in title
                or "login" in title
                or "sign in" in title
                or "auth" in title
                or "login" in text_blob
                or "sign in" in text_blob
                or "authentication" in text_blob
                or "access token" in text_blob
                or "captcha" in text_blob
            )
            if not is_login_related:
                continue
            seen_checkpoint = True
            login_phase_status = str(tool_input.get("login_phase_status") or "").strip().lower()
            if login_phase_status in ("success", "failed", "pending"):
                return login_phase_status
            if isinstance(tool_input.get("login_success"), bool):
                return "success" if bool(tool_input.get("login_success")) else "failed"
            # Heuristic fallback: infer login outcome from narrative fields
            # when the agent forgot to include explicit login_phase_status/login_success.
            if text_blob:
                success_markers = (
                    "login success",
                    "logged in",
                    "successfully logged",
                    "dashboard",
                    "home page after login",
                    "token received",
                    "access token",
                )
                failed_markers = (
                    "login failed",
                    "invalid password",
                    "invalid username",
                    "captcha failed",
                    "user inactive",
                    "authentication failed",
                    "wrong password",
                )
                if any(m in text_blob for m in success_markers):
                    return "success"
                if any(m in text_blob for m in failed_markers):
                    return "failed"
            return "pending"
        return "pending" if seen_checkpoint else "missing"

    def _login_phase_status(self, execution: ProjectExecute) -> str:
        return self._login_phase_status_from_log(execution.progress_log or [])

    def _has_login_checkpoint(self, execution: ProjectExecute) -> bool:
        return self._login_phase_status(execution) != "missing"

    def _is_login_successful(self, execution: ProjectExecute) -> bool:
        return self._login_phase_status(execution) == "success"

    def _extract_phase_instruction(self, ai_prompt: str, phase_no: int) -> str:
        text = str(ai_prompt or "").strip()
        if not text:
            return ""
        lines = text.splitlines()
        phase_aliases = {
            1: ("phase 1", "phase i", "environment & initialization", "browser setup"),
            2: ("phase 2", "phase ii", "explore & discovery", "pfm node identification"),
            3: ("phase 3", "phase iii", "ead story", "quantization"),
            4: ("phase 4", "phase iv", "final deliverable", "report generation"),
        }
        aliases = phase_aliases.get(phase_no, ())
        collect = False
        collected: List[str] = []
        for raw in lines:
            normalized = raw.strip().lower()
            if any(alias in normalized for alias in aliases):
                collect = True
                continue
            if collect and normalized.startswith("phase ") and "phase " in normalized:
                break
            if collect:
                collected.append(raw)
        return "\n".join(collected).strip()

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

        login_success = self._is_login_successful(execution)
        should_send = (
            (not login_success and not self._login_phase_prompt_sent.get(execution_id, False))
            or (login_success and not self._discovery_phase_prompt_sent.get(execution_id, False))
        )
        if not should_send:
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

        if not login_success:
            self._login_phase_prompt_sent[execution_id] = True
            phase1 = self._extract_phase_instruction(execution.ai_prompt or "", 1)
            target_hint = (execution.target_url or "").strip()
            user_message = (
                "Login phase is required first. Execute login now and report real actions + reasoning."
                + "\n"
                + _LOGIN_PROGRESS_PROMPT
                + (f"\n\nTarget URL:\n{target_hint}" if target_hint else "")
                + (f"\n\nPhase I assignment:\n{phase1}" if phase1 else "")
                + pace_hint
            )
        else:
            self._discovery_phase_prompt_sent[execution_id] = True
            phase2 = self._extract_phase_instruction(execution.ai_prompt or "", 2)
            target_hint = (execution.target_url or "").strip()
            user_message = (
                "Login confirmed. Start PFM discovery phase now. "
                "Report real actions/reasoning via report_running_step; include structured pfm_node; "
                "periodically publish_pfm_artifacts."
                + "\n"
                + _PFM_STRUCTURE_PROMPT
                + (f"\n\nTarget URL:\n{target_hint}" if target_hint else "")
                + (f"\n\nPhase II assignment:\n{phase2}" if phase2 else "")
                + pace_hint
            )

        self._pool.send_message_async(
            session_key=session_key,
            user_message=user_message,
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
                if execution.run_session_key:
                    self._pool.interrupt_agent(execution.run_session_key)
                # Surface reporting phase before final completion.
                self._store.update_execution(
                    execution_id,
                    status=ExecutionStatus.RUNNING,
                    executor_hint="Reporting...",
                )
                await self._sync_pfm_artifacts(execution_id, force=True)
                self._store.update_execution(
                    execution_id,
                    status=ExecutionStatus.COMPLETED,
                    cancel_reason="time_budget_exceeded",
                    duration_ms=elapsed_ms,
                    progress_percentage=100,
                    executor_hint="AI Finish",
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
        tool_name_by_call_id: Dict[str, str] = {}
        for msg in new_messages:
            role = msg.get("role", "")
            content = msg.get("content", "") or ""
            msg_ts = float(msg.get("timestamp") or 0.0) or time.time()
            tool_name = msg.get("tool_name")
            tool_input = None

            if role == "assistant" and msg.get("tool_calls"):
                for tc in msg["tool_calls"]:
                    if isinstance(tc, dict) and tc.get("function"):
                        tool_name = tc["function"].get("name", "")
                        call_id = str(tc.get("id") or "").strip()
                        if call_id and tool_name:
                            tool_name_by_call_id[call_id] = tool_name
                        try:
                            tool_input = json.loads(tc["function"].get("arguments", "{}"))
                        except (json.JSONDecodeError, TypeError):
                            tool_input = {}
                        display_text = content[:200] if content else tool_name
                        if tool_name == "report_running_step" and isinstance(tool_input, dict):
                            title = str(tool_input.get("title") or "").strip()
                            finding = str(tool_input.get("finding") or "").strip()
                            next_direction = str(tool_input.get("next_direction") or "").strip()
                            summary_parts = [part for part in [title, finding, next_direction] if part]
                            if summary_parts:
                                display_text = " | ".join(summary_parts)[:200]
                        image_url = None
                        thumbnail_url = None
                        if tool_name == "report_running_step" and isinstance(tool_input, dict):
                            thumbs = tool_input.get("thumbnail_urls")
                            if isinstance(thumbs, list) and thumbs:
                                raw_thumb = str(thumbs[0] or "").strip()
                                if raw_thumb:
                                    if raw_thumb.startswith(("http://", "https://", "/v1/")):
                                        thumbnail_url = raw_thumb
                                    else:
                                        thumbnail_url = _materialize_local_image_as_report(
                                            execution_id, raw_thumb
                                        )
                                    image_url = thumbnail_url
                        progress_entries.append(
                            ProgressLogEntry(
                                ts=msg_ts,
                                kind="tool_use",
                                text=display_text,
                                tool_name=tool_name,
                                tool_input=tool_input,
                                image_url=image_url,
                                thumbnail_url=thumbnail_url,
                            )
                        )
            elif role == "tool":
                resolved_tool_name = str(tool_name or "").strip()
                call_id = str(msg.get("tool_call_id") or "").strip()
                if not resolved_tool_name and call_id:
                    resolved_tool_name = tool_name_by_call_id.get(call_id, "")
                payload = None
                try:
                    payload = json.loads(content) if isinstance(content, str) else None
                except Exception:
                    payload = None
                image_ref = _extract_image_reference_from_payload(payload or content)
                image_url = None
                thumbnail_url = None
                if image_ref:
                    if str(image_ref).startswith(("http://", "https://", "/v1/")):
                        image_url = str(image_ref)
                    else:
                        image_url = _materialize_local_image_as_report(execution_id, str(image_ref))
                    thumbnail_url = image_url
                progress_entries.append(
                    ProgressLogEntry(
                        ts=msg_ts,
                        kind="tool_result",
                        text=content[:200],
                        tool_name=resolved_tool_name or tool_name,
                        image_url=image_url,
                        thumbnail_url=thumbnail_url,
                    )
                )
            elif role == "assistant":
                # Ignore executor-generated progress messages to avoid echo loops in future summaries.
                if content.strip().startswith(_PROGRESS_MESSAGE_PREFIX):
                    continue
                image_ref = _extract_image_reference_from_payload(content)
                image_url = None
                thumbnail_url = None
                if image_ref:
                    if str(image_ref).startswith(("http://", "https://", "/v1/")):
                        image_url = str(image_ref)
                    else:
                        image_url = _materialize_local_image_as_report(execution_id, str(image_ref))
                    thumbnail_url = image_url
                progress_entries.append(
                    ProgressLogEntry(
                        ts=msg_ts,
                        kind="assistant",
                        text=content[:200],
                        image_url=image_url,
                        thumbnail_url=thumbnail_url,
                    )
                )

        if progress_entries:
            current_log = execution.progress_log or []
            updated_log = current_log + progress_entries
            msg_count = len(messages)

            update_fields = {
                "progress_log": updated_log,
                "progress_log_seq": msg_count,
                "progress_percentage": self._compute_time_progress_percent(execution),
            }
            if not execution.start_time and self._login_phase_status_from_log(updated_log) == "success":
                update_fields["start_time"] = int(time.time() * 1000)
                update_fields["progress_percentage"] = 0
                update_fields["executor_hint"] = "Login confirmed; PFM discovery is now active."
            self._store.update_execution(execution_id, **update_fields)

    async def _sync_pfm_artifacts(self, execution_id: str, force: bool = False) -> None:
        execution = self._store.get_execution(execution_id)
        if not execution:
            return

        seq = int(execution.progress_log_seq or 0)
        if not force and self._last_pfm_sync_seq.get(execution_id) == seq:
            return

        login_success = self._is_login_successful(execution)
        if not login_success and not force:
            self._last_pfm_sync_seq[execution_id] = seq
            return

        # If the run did not produce structured node results yet, derive a live
        # placeholder map from transcript progress so the UI mindmap can update in real time.
        if login_success and not execution.results:
            derived = derive_node_runs_from_progress(execution.progress_log or [])
            if derived:
                self._store.update_execution(execution_id, results=derived)
                refreshed = self._store.get_execution(execution_id)
                if refreshed:
                    execution = refreshed

        reports = build_and_persist_pfm_artifacts(execution)
        self._store.update_execution(execution_id, reports=reports)
        self._last_pfm_sync_seq[execution_id] = seq

    async def _report_progress_to_chat(self, execution_id: str, session_key: str) -> None:
        # User requested agent-native updates only; suppress executor-authored chat summaries.
        return
        now = time.time()
        last_report = self._last_progress_report_ts.get(execution_id, 0)
        if now - last_report < _PROGRESS_REPORT_INTERVAL_S:
            return

        self._last_progress_report_ts[execution_id] = now
        execution = self._store.get_execution(execution_id)
        if not execution:
            return

        now_ms = int(now * 1000)
        elapsed_ms = max(0, now_ms - (execution.start_time or now_ms))
        elapsed_min = elapsed_ms / 60000
        pct = self._compute_time_progress_percent(execution, now_ms)
        execution = self._store.update_execution(
            execution_id,
            progress_percentage=pct,
        ) or execution
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

        login_phase_status = self._login_phase_status(execution)
        login_success = login_phase_status == "success"
        recent: List[ProgressLogEntry] = []
        progress_text = f"📊 Progress Update ({elapsed_min:.1f}m elapsed)\n"
        progress_text += f"Status: {execution.status.value}\n"
        if execution.time_budget_minutes and execution.time_budget_minutes > 0 and login_success:
            budget_ms = int(execution.time_budget_minutes * 60 * 1000)
            progress_text += (
                f"Time budget: {self._format_mmss(elapsed_ms)} / {self._format_mmss(budget_ms)}\n"
                f"Completion: {pct}%\n"
            )
        elif execution.time_budget_minutes and execution.time_budget_minutes > 0:
            progress_text += "Time budget: awaiting login success to start timer\n"
        else:
            progress_text += (
                "Time budget: not configured yet\n"
                "Completion: pending budget setup\n"
            )
        if steps_total > 0:
            progress_text += f"Steps: {steps_done}/{steps_total} completed\n"
        if execution.progress_log:
            # Show only concrete execution records, not chat/meta chatter.
            recent = [
                e for e in execution.progress_log
                if (
                    (e.kind == "tool_use" and (e.tool_name or "") == "report_running_step")
                    or e.kind == "tool_result"
                )
            ][-6:]
            if recent:
                progress_text += "Execution trail (latest):\n"
                for e in recent:
                    if e.kind == "tool_use" and (e.tool_name or "") == "report_running_step":
                        ti = e.tool_input or {}
                        title = str(ti.get("title") or "").strip()
                        observation = str(ti.get("observation") or "").strip()
                        finding = str(ti.get("finding") or "").strip()
                        reasoning = str(ti.get("reasoning") or "").strip()
                        decision = str(ti.get("decision") or "").strip()
                        next_direction = str(ti.get("next_direction") or "").strip()
                        parts = [p for p in [title, observation, finding, reasoning, decision, next_direction] if p]
                        if parts:
                            progress_text += f"  • Step: {' | '.join(parts)[:220]}\n"
                        else:
                            progress_text += f"  • Step: {(e.text or '').strip()[:220]}\n"
                    elif e.kind == "tool_result":
                        text = (e.text or "").strip()
                        if text:
                            progress_text += f"  • Tool result: {text[:180]}\n"
                    else:
                        text = (e.text or "").strip()
                        if text:
                            progress_text += f"  • {text[:180]}\n"
        if not recent:
            # Do not post synthetic status-only updates; wait for real execution activity.
            return

        # PFM-oriented summary for EAD Explore runs.
        pfm_nodes = set()
        pfm_features = set()
        pfm_test_cases = set()

        if login_success:
            for node in execution.results or []:
                if node.title:
                    pfm_nodes.add(node.title.strip().lower())
                elif node.node_key:
                    pfm_nodes.add(node.node_key.strip().lower())
                for tc in node.test_case_runs or []:
                    if tc.title:
                        pfm_test_cases.add(tc.title.strip().lower())

            for entry in execution.progress_log or []:
                if entry.kind != "tool_use":
                    continue
                if (entry.tool_name or "") != "report_running_step":
                    continue
                tool_input = entry.tool_input or {}
                pfm_node = tool_input.get("pfm_node")
                if not isinstance(pfm_node, dict):
                    continue
                node_label = str(pfm_node.get("title") or pfm_node.get("node_key") or pfm_node.get("node_id") or "").strip()
                if node_label:
                    pfm_nodes.add(node_label.lower())
                feats = pfm_node.get("features") or []
                if isinstance(feats, list):
                    for f in feats:
                        if isinstance(f, str) and f.strip():
                            pfm_features.add(f.strip().lower())
                test_cases = pfm_node.get("test_cases") or []
                if isinstance(test_cases, list):
                    for tc in test_cases:
                        if not isinstance(tc, dict):
                            continue
                        title = str(tc.get("title") or tc.get("case_id") or "").strip()
                        if title:
                            pfm_test_cases.add(title.lower())

        login_label = (
            "Confirmed" if login_phase_status == "success"
            else "Failed" if login_phase_status == "failed"
            else "Pending"
        )
        if login_success and (pfm_nodes or pfm_features or pfm_test_cases or self._has_login_checkpoint(execution)):
            progress_text += (
                "PFM discovery progress (Interactive):\n"
                f"  • Initial checkpoint (Login PFM map): {login_label}\n"
                f"  • Nodes discovered: {len(pfm_nodes)}\n"
                f"  • Features captured: {len(pfm_features)}\n"
                f"  • Test cases captured: {len(pfm_test_cases)}\n"
            )
        latest_insight = None
        for entry in reversed(execution.progress_log or []):
            if entry.kind != "tool_use" or (entry.tool_name or "") != "report_running_step":
                continue
            tool_input = entry.tool_input or {}
            if not isinstance(tool_input, dict):
                continue
            latest_insight = tool_input
            break

        if latest_insight:
            observation = str(latest_insight.get("observation") or "").strip()
            finding = str(latest_insight.get("finding") or "").strip()
            reasoning = str(latest_insight.get("reasoning") or "").strip()
            decision = str(latest_insight.get("decision") or "").strip()
            next_direction = str(latest_insight.get("next_direction") or "").strip()
            if observation or finding or reasoning or decision or next_direction:
                progress_text += "Latest discovery reasoning:\n"
                if observation:
                    progress_text += f"  • Observation: {observation[:160]}\n"
                if finding:
                    progress_text += f"  • Finding: {finding[:160]}\n"
                if reasoning:
                    progress_text += f"  • Reasoning: {reasoning[:160]}\n"
                if decision:
                    progress_text += f"  • Decision: {decision[:160]}\n"
                if next_direction:
                    progress_text += f"  • Next direction: {next_direction[:160]}\n"

        try:
            db.append_message(
                session_id=session_id,
                role="assistant",
                content=progress_text,
            )
            logger.info("[executor] Progress reported for %s (%.0f%%)", execution_id, pct)
        except Exception as e:
            logger.warning("[executor] Failed to report progress for %s: %s", execution_id, e)

    async def cancel_execution(
        self,
        execution_id: str,
        final_status: ExecutionStatus = ExecutionStatus.CANCELLED,
        operator_stop_kind: Optional[str] = None,
        cancel_reason: Optional[str] = None,
    ) -> None:
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
            status=final_status,
            paused=False,
            operator_stop_kind=operator_stop_kind,
            cancel_reason=cancel_reason,
        )
        logger.info(
            "[executor] Stopped execution %s status=%s kind=%s",
            execution_id,
            final_status.value,
            operator_stop_kind or "ai",
        )

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
