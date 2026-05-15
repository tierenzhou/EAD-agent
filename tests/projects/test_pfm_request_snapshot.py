"""Tests for UI-triggered PFM snapshot refresh (commit_pfm_snapshot nudge)."""

from __future__ import annotations

import time
from unittest.mock import MagicMock

from projects.executor import ProjectExecutor, _PFM_REFRESH_NUDGE_MIN_INTERVAL_S
from projects.models import EadFmNodeRun, ExecutionStatus, ProjectExecute


def _minimal_execution(
    execution_id: str = "exec-1",
    *,
    status: ExecutionStatus = ExecutionStatus.RUNNING,
    session_key: str = "sk-test",
    bootstrap_pending: bool = False,
) -> ProjectExecute:
    return ProjectExecute(
        id=execution_id,
        linked_template_id="tpl-1",
        name="Test",
        status=status,
        run_session_key=session_key,
        bootstrap_pending=bootstrap_pending,
    )


def test_request_pfm_snapshot_refresh_queues_when_idle():
    store = MagicMock()
    pool = MagicMock()
    pool.is_agent_active.return_value = False
    pool.send_message_async.return_value = "eadrun_abc123"

    ex = _minimal_execution()
    store.get_execution.return_value = ex

    executor = ProjectExecutor(store, pool)
    out = executor.request_pfm_snapshot_refresh(ex.id)

    assert out["ok"] is True
    assert out["code"] == "queued"
    assert out["run_id"] == "eadrun_abc123"
    pool.send_message_async.assert_called_once()
    kwargs = pool.send_message_async.call_args.kwargs
    assert kwargs["session_key"] == "sk-test"
    assert kwargs["enable_tools"] is True
    assert "commit_pfm_snapshot" in kwargs["user_message"]
    assert "full current PFM tree" in kwargs["user_message"]


def test_request_pfm_snapshot_refresh_agent_busy():
    store = MagicMock()
    pool = MagicMock()
    pool.is_agent_active.return_value = True
    store.get_execution.return_value = _minimal_execution()

    executor = ProjectExecutor(store, pool)
    out = executor.request_pfm_snapshot_refresh("exec-1")

    assert out["ok"] is True
    assert out["code"] == "queued_deferred"
    assert "queued" in str(out.get("message") or "").lower()
    pool.send_message_async.assert_not_called()


def test_request_pfm_snapshot_refresh_throttled():
    store = MagicMock()
    pool = MagicMock()
    pool.is_agent_active.return_value = False
    pool.send_message_async.return_value = "eadrun_first"

    ex = _minimal_execution()
    store.get_execution.return_value = ex

    executor = ProjectExecutor(store, pool)
    assert executor.request_pfm_snapshot_refresh(ex.id)["code"] == "queued"

    out2 = executor.request_pfm_snapshot_refresh(ex.id)
    assert out2["ok"] is True
    assert out2["code"] == "throttled"
    assert out2.get("retry_after_s", 0) > 0
    assert pool.send_message_async.call_count == 1


def test_request_pfm_snapshot_refresh_after_interval():
    store = MagicMock()
    pool = MagicMock()
    pool.is_agent_active.return_value = False
    pool.send_message_async.return_value = "eadrun_x"

    ex = _minimal_execution()
    store.get_execution.return_value = ex

    executor = ProjectExecutor(store, pool)
    executor.request_pfm_snapshot_refresh(ex.id)
    # Pretend the throttle window elapsed without waiting in real time.
    executor._pfm_refresh_nudge_ts[ex.id] = time.time() - _PFM_REFRESH_NUDGE_MIN_INTERVAL_S - 1.0
    executor.request_pfm_snapshot_refresh(ex.id)
    assert pool.send_message_async.call_count == 2


def test_request_pfm_snapshot_refresh_allows_completed_with_session():
    store = MagicMock()
    pool = MagicMock()
    pool.is_agent_active.return_value = False
    pool.send_message_async.return_value = "eadrun_done"

    ex = _minimal_execution(status=ExecutionStatus.COMPLETED)
    store.get_execution.return_value = ex

    executor = ProjectExecutor(store, pool)
    out = executor.request_pfm_snapshot_refresh(ex.id)
    assert out["ok"] is True
    assert out["code"] == "queued"
    pool.send_message_async.assert_called_once()


def test_request_pfm_snapshot_refresh_completed_no_changes_short_circuits():
    store = MagicMock()
    pool = MagicMock()
    pool.is_agent_active.return_value = False
    pool.send_message_async.return_value = "eadrun_done"

    ex = _minimal_execution(status=ExecutionStatus.COMPLETED)
    ex.results = [
        EadFmNodeRun(
            node_key="reporting",
            parent_node_key=None,
            level=1,
            title="Reporting",
            status="Success",
            meta="Summary report generation and visualization",
        ),
        EadFmNodeRun(
            node_key="reporting/summary-report-view",
            parent_node_key="reporting",
            level=2,
            title="Summary Report View",
            status="Success",
            meta="Report View ; Metrics Display",
        ),
    ]
    store.get_execution.return_value = ex
    store.get_committed_pfm_tree.return_value = {
        "version": 9,
        "flat_nodes": [
            {
                "node_key": "reporting",
                "parent_node_key": None,
                "level": 1,
                "title": "Reporting",
                "status": "Success",
                "description": "Summary report generation and visualization",
            },
            {
                "node_key": "reporting/summary-report-view",
                "parent_node_key": "reporting",
                "level": 2,
                "title": "Summary Report View",
                "status": "Success",
                "description": "Report View ; Metrics Display",
            },
        ],
    }

    executor = ProjectExecutor(store, pool)
    out = executor.request_pfm_snapshot_refresh(ex.id)

    assert out["ok"] is True
    assert out["code"] == "no_changes"
    pool.send_message_async.assert_not_called()
