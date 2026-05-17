"""Tests for Phase II discovery kickoff and continue nudges."""

from __future__ import annotations

import asyncio
import time
from unittest.mock import MagicMock

import pytest

from projects.executor import ProjectExecutor
from projects.models import ExecutionStatus, ProgressLogEntry, ProjectExecute


def _login_success_entry() -> ProgressLogEntry:
    return ProgressLogEntry(
        kind="tool_use",
        tool_name="report_running_step",
        tool_input={
            "title": "Login PFM map (interactive)",
            "login_phase_status": "success",
        },
    )


def _init_review_success_entry() -> ProgressLogEntry:
    return ProgressLogEntry(
        kind="tool_use",
        tool_name="report_running_step",
        tool_input={
            "title": "PFM inheritance review (Initialization)",
            "initialization_review_status": "success",
            "ready_to_explore": True,
        },
    )


def _discovery_execution(
    execution_id: str = "exec-disc",
    *,
    minutes_elapsed: float = 1.0,
    budget_minutes: int = 30,
) -> ProjectExecute:
    now_ms = int(time.time() * 1000)
    return ProjectExecute(
        id=execution_id,
        linked_template_id="tpl-1",
        name="Discovery run",
        status=ExecutionStatus.RUNNING,
        run_session_key="sk-disc",
        target_url="https://example.test/app",
        start_time=now_ms - int(minutes_elapsed * 60 * 1000),
        time_budget_minutes=budget_minutes,
        progress_log=[_login_success_entry(), _init_review_success_entry()],
    )


def test_discovery_nudge_kind_kickoff_when_ready():
    executor = ProjectExecutor(MagicMock(), MagicMock())
    ex = _discovery_execution()
    assert executor._discovery_nudge_kind(ex.id, ex) == "kickoff"


def test_discovery_nudge_kind_none_before_continue_interval():
    executor = ProjectExecutor(MagicMock(), MagicMock())
    ex = _discovery_execution()
    executor._discovery_kickoff_sent[ex.id] = True
    executor._last_discovery_nudge_ts[ex.id] = time.time()
    assert executor._discovery_nudge_kind(ex.id, ex) is None


def test_discovery_nudge_kind_continue_after_interval(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr("projects.executor._DISCOVERY_NUDGE_INTERVAL_S", 1.0)
    executor = ProjectExecutor(MagicMock(), MagicMock())
    ex = _discovery_execution()
    executor._discovery_kickoff_sent[ex.id] = True
    executor._last_discovery_nudge_ts[ex.id] = time.time() - 5.0
    assert executor._discovery_nudge_kind(ex.id, ex) == "continue"


def test_discovery_nudge_none_when_budget_exceeded():
    executor = ProjectExecutor(MagicMock(), MagicMock())
    ex = _discovery_execution(minutes_elapsed=31.0, budget_minutes=30)
    assert executor._discovery_nudge_kind(ex.id, ex) is None


def test_auto_continue_sends_kickoff_then_continue(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr("projects.executor._DISCOVERY_NUDGE_INTERVAL_S", 1.0)
    store = MagicMock()
    pool = MagicMock()
    pool.is_agent_active.return_value = False

    ex = _discovery_execution()
    store.get_execution.return_value = ex

    executor = ProjectExecutor(store, pool)
    executor._login_phase_prompt_sent[ex.id] = True
    executor._initialization_review_prompt_sent[ex.id] = True

    asyncio.run(executor._auto_continue(ex.id))
    assert pool.send_message_async.call_count == 1
    kickoff_msg = pool.send_message_async.call_args.kwargs["user_message"]
    assert "Start PFM discovery" in kickoff_msg
    assert "hypothesis only" in kickoff_msg
    assert executor._discovery_kickoff_sent[ex.id] is True

    executor._last_discovery_nudge_ts[ex.id] = time.time() - 5.0
    executor._last_continue_ts[ex.id] = time.time() - 100.0
    asyncio.run(executor._auto_continue(ex.id))
    assert pool.send_message_async.call_count == 2
    continue_msg = pool.send_message_async.call_args.kwargs["user_message"]
    assert "Continue Phase II DFS exploration" in continue_msg
    assert "RUN SCOPE" not in continue_msg
    assert "Do not stop early" in continue_msg


def test_build_discovery_messages_include_budget_remaining():
    executor = ProjectExecutor(MagicMock(), MagicMock())
    ex = _discovery_execution()
    boundary = executor._project_boundary_prompt(ex)
    kickoff = executor._build_discovery_kickoff_message(ex, boundary, "")
    cont = executor._build_discovery_continue_message(ex, boundary, "")
    assert "remaining" in kickoff.lower()
    assert "remaining" in cont.lower()
    assert "Do not treat this run as complete" in kickoff
    assert "RUN SCOPE" in kickoff
    assert "RUN SCOPE" not in cont


def test_continue_message_omits_project_boundary():
    executor = ProjectExecutor(MagicMock(), MagicMock())
    ex = _discovery_execution()
    boundary = executor._project_boundary_prompt(ex)
    cont = executor._build_discovery_continue_message(ex, boundary, "")
    assert "RUN SCOPE" not in cont
    assert "Continue Phase II" in cont
