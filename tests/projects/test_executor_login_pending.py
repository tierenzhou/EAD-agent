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
