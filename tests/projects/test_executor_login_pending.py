"""Login pending must not trigger repeated executor login kickoffs."""

from __future__ import annotations

import asyncio
import time
from unittest.mock import MagicMock

from projects.executor import ProjectExecutor
from projects.models import ExecutionStatus, ProgressLogEntry, ProjectExecute


def _pending_login_execution() -> ProjectExecute:
    return ProjectExecute(
        id="exec-login",
        linked_template_id="tpl-1",
        name="Login run",
        status=ExecutionStatus.RUNNING,
        run_session_key="sk-login",
        progress_log=[
            ProgressLogEntry(
                kind="tool_use",
                tool_name="report_running_step",
                tool_input={
                    "title": "Login PFM map (Interactive)",
                    "login_phase_status": "pending",
                },
            ),
        ],
    )


def test_auto_continue_skips_login_kickoff_when_pending():
    store = MagicMock()
    pool = MagicMock()
    ex = _pending_login_execution()
    store.get_execution.return_value = ex

    executor = ProjectExecutor(store, pool)
    executor._login_phase_prompt_sent[ex.id] = True
    executor._last_continue_ts[ex.id] = time.time() - 100.0

    asyncio.run(executor._auto_continue(ex.id))

    pool.send_message_async.assert_not_called()


def test_synthesizes_login_checkpoint_from_authenticated_browser_evidence():
    executor = ProjectExecutor(MagicMock(), MagicMock())
    progress_entries = [
        ProgressLogEntry(
            kind="assistant",
            text="Login successful — redirected to /homepage?isRedirect=1.",
        ),
        ProgressLogEntry(
            kind="tool_result",
            tool_name="browser_snapshot",
            text=(
                '{"success": true, "snapshot": "- document\\n  - text: Home TZ '
                'Management Requirement Development Test Case Test Run Report"}'
            ),
        ),
    ]

    updated = executor._maybe_append_synthesized_login_success_checkpoint(
        "exec-login",
        [],
        progress_entries,
    )

    assert len(updated) == 1
    checkpoint = updated[0]
    assert checkpoint.tool_name == "report_running_step"
    assert checkpoint.tool_input["title"] == "Login PFM map (Interactive)"
    assert checkpoint.tool_input["login_phase_status"] == "success"
    assert checkpoint.tool_input["login_success"] is True
    assert checkpoint.tool_input["recovered_by_gateway"] is True


def test_does_not_synthesize_login_checkpoint_from_narration_only():
    executor = ProjectExecutor(MagicMock(), MagicMock())
    progress_entries = [
        ProgressLogEntry(kind="assistant", text="Login successful!"),
    ]

    updated = executor._maybe_append_synthesized_login_success_checkpoint(
        "exec-login",
        [],
        progress_entries,
    )

    assert updated == []


def test_does_not_synthesize_login_checkpoint_when_back_at_login_form():
    executor = ProjectExecutor(MagicMock(), MagicMock())
    progress_entries = [
        ProgressLogEntry(kind="assistant", text="Session expired — back at login."),
        ProgressLogEntry(
            kind="tool_result",
            tool_name="browser_snapshot",
            text='textbox "User name" textbox "Password" button "Log In"',
        ),
    ]

    updated = executor._maybe_append_synthesized_login_success_checkpoint(
        "exec-login",
        [],
        progress_entries,
    )

    assert updated == []


def test_synthesizes_after_relogin_even_if_earlier_batch_had_login_form():
    executor = ProjectExecutor(MagicMock(), MagicMock())
    progress_entries = [
        ProgressLogEntry(kind="assistant", text="Session expired — back at login."),
        ProgressLogEntry(
            kind="tool_result",
            tool_name="browser_snapshot",
            text='textbox "User name" textbox "Password" button "Log In"',
        ),
        ProgressLogEntry(kind="assistant", text="Login successful — redirected to /homepage?isRedirect=1."),
        ProgressLogEntry(
            kind="tool_result",
            tool_name="browser_snapshot",
            text="- document\n  - text: Home TZ Management Requirement Development Test Case Test Run Report",
        ),
    ]

    updated = executor._maybe_append_synthesized_login_success_checkpoint(
        "exec-login",
        [],
        progress_entries,
    )

    assert len(updated) == 1
    assert updated[0].tool_input["login_phase_status"] == "success"
