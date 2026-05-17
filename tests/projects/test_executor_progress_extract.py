"""Progress extraction must not mirror internal bootstrap ack messages."""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

from projects.api import EAD_RUN_ACK_MARKER, _execution_ack_message
from projects.executor import ProjectExecutor
from projects.models import ExecutionStatus, ProjectExecute


def test_execution_ack_message_has_marker() -> None:
    ex = ProjectExecute(
        id="exec-1",
        linked_template_id="tpl-1",
        name="Run: demo",
        status=ExecutionStatus.PENDING,
        target_url="https://example.test",
    )
    body = _execution_ack_message(ex)
    assert body.startswith(EAD_RUN_ACK_MARKER)
    assert "## Run Introduction" in body


def test_extract_progress_skips_run_ack_message() -> None:
    store = MagicMock()
    pool = MagicMock()
    db = MagicMock()

    ack = _execution_ack_message(
        ProjectExecute(
            id="exec-ack",
            linked_template_id="tpl-1",
            name="Run: ack",
            status=ExecutionStatus.RUNNING,
            run_session_key="sk-ack",
        )
    )
    user_visible = "Login step report: credentials accepted."

    ex = ProjectExecute(
        id="exec-ack",
        linked_template_id="tpl-1",
        name="Run: ack",
        status=ExecutionStatus.RUNNING,
        run_session_key="sk-ack",
        progress_log=[],
        progress_log_seq=0,
    )
    store.get_execution.return_value = ex
    pool._get_session_db.return_value = db
    pool._resolve_session_id.return_value = "session-ack"
    db.get_messages.return_value = [
        {"role": "assistant", "content": ack, "timestamp": 1.0},
        {"role": "assistant", "content": user_visible, "timestamp": 2.0},
    ]

    captured: dict = {}

    def _update(execution_id: str, **kwargs):
        captured.update(kwargs)

    store.update_execution.side_effect = _update

    executor = ProjectExecutor(store, pool)
    asyncio.run(executor._extract_progress("exec-ack"))

    store.update_execution.assert_called_once()
    log = captured.get("progress_log") or []
    texts = [getattr(e, "text", "") or "" for e in log]
    assert not any(EAD_RUN_ACK_MARKER in t for t in texts)
    assert not any("## Run Introduction" in t for t in texts)
    assert any("Login step report" in t for t in texts)
