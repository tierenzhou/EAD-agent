"""PFM snapshot nudges after init and materialize fallback on finalize."""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from projects.executor import ProjectExecutor
from projects.models import ExecutionStatus, ProgressLogEntry, ProjectExecute
from projects.store import ProjectStore
from tests.projects.test_pfm_canonical import _commit_minimal_tree


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


def _running_execution(execution_id: str = "exec-nudge") -> ProjectExecute:
    return ProjectExecute(
        id=execution_id,
        linked_template_id="tpl-1",
        name="Nudge run",
        status=ExecutionStatus.RUNNING,
        run_session_key="sk-nudge",
        target_url="https://example.test/app",
        start_time=int(time.time() * 1000),
        time_budget_minutes=30,
        progress_log=[_login_success_entry(), _init_review_success_entry()],
    )


def test_periodic_pfm_snapshot_nudge_queues_and_counts(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("projects.executor._PFM_SNAPSHOT_NUDGE_INTERVAL_S", 0.0)
    store = MagicMock()
    store.has_committed_pfm_tree.return_value = False
    executor = ProjectExecutor(store, MagicMock())
    ex = _running_execution()
    executor._post_init_snapshot_nudged.add(ex.id)
    executor._pfm_snapshot_nudge_count[ex.id] = 1

    with patch.object(executor, "request_pfm_snapshot_refresh", return_value={"ok": True, "code": "queued"}) as mock_refresh:
        executor._maybe_periodic_pfm_snapshot_nudge(ex.id, ex)
        mock_refresh.assert_called_once_with(ex.id)
    assert executor._pfm_snapshot_nudge_count[ex.id] == 2


def test_periodic_pfm_snapshot_nudge_stops_at_max(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("projects.executor._PFM_SNAPSHOT_NUDGE_MAX", 2)
    store = MagicMock()
    store.has_committed_pfm_tree.return_value = False
    executor = ProjectExecutor(store, MagicMock())
    ex = _running_execution("exec-cap")
    executor._post_init_snapshot_nudged.add(ex.id)
    executor._pfm_snapshot_nudge_count[ex.id] = 2

    with patch.object(executor, "request_pfm_snapshot_refresh") as mock_refresh:
        executor._maybe_periodic_pfm_snapshot_nudge(ex.id, ex)
        mock_refresh.assert_not_called()


def test_finalize_materializes_pfm_when_missing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("EAD_REPORT_BASE_DIR", str(tmp_path / "reports"))
    monkeypatch.setenv("EAD_PFM_AGENT_AUTHORED", "true")
    store = ProjectStore(db_path=tmp_path / "projects.db")
    from projects.models import ProjectTemplate

    store.create_template(ProjectTemplate(id="tpl-1", name="T"))
    store.create_execution(
        ProjectExecute(
            id="exec-baseline",
            linked_template_id="tpl-1",
            name="Baseline",
            status=ExecutionStatus.COMPLETED,
            start_time=1000,
        )
    )
    _commit_minimal_tree(store, "exec-baseline")
    store.create_execution(
        ProjectExecute(
            id="exec-done",
            linked_template_id="tpl-1",
            name="Done",
            status=ExecutionStatus.COMPLETED,
            start_time=2000,
            inherited_from_execution_id="exec-baseline",
        )
    )
    baseline = store.get_execution("exec-baseline")
    assert baseline is not None
    store.update_execution("exec-done", results=list(baseline.results or []))

    executor = ProjectExecutor(store, MagicMock())
    with patch("projects.pfm_deliverables.generate_pfm_deliverables", return_value=[]):
        with patch.object(store, "upsert_template_learning_summary"):
            asyncio.run(executor._finalize_completed_execution("exec-done"))

    assert store.has_committed_pfm_tree("exec-done")


def test_finalize_requires_committed_tree_or_sets_hint(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("EAD_REPORT_DIR", str(tmp_path / "reports"))
    monkeypatch.setenv("EAD_PFM_AGENT_AUTHORED", "true")
    store = ProjectStore(db_path=tmp_path / "projects.db")
    from projects.models import ProjectTemplate

    store.create_template(ProjectTemplate(id="tpl-1", name="T"))
    store.create_execution(
        ProjectExecute(
            id="exec-empty",
            linked_template_id="tpl-1",
            name="Empty completed run",
            status=ExecutionStatus.COMPLETED,
            start_time=2000,
        )
    )

    executor = ProjectExecutor(store, MagicMock())
    with patch("projects.pfm_deliverables.generate_pfm_deliverables") as mock_deliver:
        asyncio.run(executor._finalize_completed_execution("exec-empty"))

    mock_deliver.assert_not_called()
    ex = store.get_execution("exec-empty")
    assert ex is not None
    assert "PFM map is not committed yet" in str(ex.executor_hint or "")


def test_cancel_finish_blocks_completed_without_committed_tree(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("EAD_REPORT_DIR", str(tmp_path / "reports"))
    monkeypatch.setenv("EAD_PFM_AGENT_AUTHORED", "true")
    store = ProjectStore(db_path=tmp_path / "projects.db")
    from projects.models import ProjectTemplate

    store.create_template(ProjectTemplate(id="tpl-1", name="T"))
    store.create_execution(
        ProjectExecute(
            id="exec-no-tree",
            linked_template_id="tpl-1",
            name="No tree run",
            status=ExecutionStatus.RUNNING,
            run_session_key="sk-no-tree",
            start_time=2000,
        )
    )

    pool = MagicMock()
    pool.is_agent_active.return_value = False
    executor = ProjectExecutor(store, pool)
    asyncio.run(
        executor.cancel_execution(
            "exec-no-tree",
            final_status=ExecutionStatus.COMPLETED,
            operator_stop_kind="finish",
            cancel_reason="operator requested finish",
        )
    )

    ex = store.get_execution("exec-no-tree")
    assert ex is not None
    assert ex.status == ExecutionStatus.CANCELLED
    assert "completion_blocked:" in str(ex.cancel_reason or "")


def test_monitor_loop_skips_force_sync_for_cancelled_terminal_state() -> None:
    store = MagicMock()
    pool = MagicMock()
    executor = ProjectExecutor(store, pool)
    ex = ProjectExecute(
        id="exec-cancelled",
        linked_template_id="tpl-1",
        name="Cancelled run",
        status=ExecutionStatus.CANCELLED,
        run_session_key="sk-cancelled",
    )
    store.get_execution.return_value = ex

    with patch.object(executor, "_sync_pfm_artifacts", new_callable=MagicMock) as mock_sync:
        with patch.object(executor, "_finalize_completed_execution", new_callable=MagicMock) as mock_finalize:
            asyncio.run(executor._monitor_loop(ex.id))

    mock_sync.assert_not_called()
    mock_finalize.assert_not_called()
    store.set_execution_reporting_activity.assert_called_with(ex.id, active=False)


def test_cancel_execution_closes_reporting_activity() -> None:
    store = MagicMock()
    pool = MagicMock()
    executor = ProjectExecutor(store, pool)
    ex = ProjectExecute(
        id="exec-stop",
        linked_template_id="tpl-1",
        name="Stopped run",
        status=ExecutionStatus.RUNNING,
        run_session_key="sk-stop",
    )
    store.get_execution.return_value = ex

    asyncio.run(
        executor.cancel_execution(
            ex.id,
            final_status=ExecutionStatus.CANCELLED,
            operator_stop_kind="cancel",
            cancel_reason="operator stop",
        )
    )

    store.set_execution_reporting_activity.assert_called_with(ex.id, active=False)


def test_force_sync_inherited_run_persists_after_divergence() -> None:
    store = MagicMock()
    pool = MagicMock()
    executor = ProjectExecutor(store, pool)
    ex = ProjectExecute(
        id="exec-inh",
        linked_template_id="tpl-1",
        name="Inherited",
        status=ExecutionStatus.RUNNING,
        inherited_from_execution_id="exec-base",
        progress_log=[_login_success_entry()],
        results=[{"node_key": "root", "title": "Root"}],  # shape only for mocked signatures
    )
    store.get_execution.return_value = ex
    store.get_committed_pfm_tree.side_effect = lambda eid: (
        None
        if eid == "exec-inh"
        else {"flat_nodes": [{"node_key": "base", "title": "Base", "status": "No Run"}]}
    )
    store._results_signature.return_value = [("root", "Root", "", "no run")]
    store._snapshot_signature.return_value = [("base", "Base", "", "no run")]
    store.persist_pfm_tree_from_execution_state.return_value = {"ok": True, "code": "materialized"}

    with patch("projects.executor.build_and_persist_pfm_artifacts", return_value=[]):
        asyncio.run(executor._sync_pfm_artifacts("exec-inh", force=True))

    store.persist_pfm_tree_from_execution_state.assert_called_once_with("exec-inh")


def test_force_sync_inherited_run_defers_when_still_baseline() -> None:
    store = MagicMock()
    pool = MagicMock()
    executor = ProjectExecutor(store, pool)
    ex = ProjectExecute(
        id="exec-inh",
        linked_template_id="tpl-1",
        name="Inherited",
        status=ExecutionStatus.RUNNING,
        inherited_from_execution_id="exec-base",
        progress_log=[_login_success_entry()],
        results=[{"node_key": "base", "title": "Base"}],  # shape only for mocked signatures
    )
    store.get_execution.return_value = ex
    store.get_committed_pfm_tree.side_effect = lambda eid: (
        None
        if eid == "exec-inh"
        else {"flat_nodes": [{"node_key": "base", "title": "Base", "status": "No Run"}]}
    )
    sig = [("base", "Base", "", "no run")]
    store._results_signature.return_value = sig
    store._snapshot_signature.return_value = sig

    with patch("projects.executor.build_and_persist_pfm_artifacts", return_value=[]):
        asyncio.run(executor._sync_pfm_artifacts("exec-inh", force=True))

    store.persist_pfm_tree_from_execution_state.assert_not_called()


def test_sync_triggers_immediate_refresh_when_new_delivery_without_new_progress() -> None:
    store = MagicMock()
    pool = MagicMock()
    executor = ProjectExecutor(store, pool)
    ex = _running_execution("exec-delivery")
    ex.progress_log_seq = 7
    store.get_execution.return_value = ex
    executor._last_pfm_sync_seq[ex.id] = 7  # no new progress => normal sync body would return early

    with patch(
        "projects.pfm_delivery.compute_canonical_delivery_stamp",
        return_value={"fingerprint": "fp-new"},
    ), patch(
        "projects.pfm_refresh.try_refresh_pfm_from_delivery",
        return_value={"ok": True, "code": "materialized"},
    ) as mock_refresh, patch(
        "projects.executor.build_and_persist_pfm_artifacts",
        return_value=[],
    ) as mock_build:
        asyncio.run(executor._sync_pfm_artifacts(ex.id))

    mock_refresh.assert_called_once_with(store, ex.id, promote_template_canonical=False)
    mock_build.assert_not_called()
    assert executor._last_delivery_refresh_fingerprint.get(ex.id) == "fp-new"


def test_sync_does_not_repeat_refresh_for_same_delivery_fingerprint() -> None:
    store = MagicMock()
    pool = MagicMock()
    executor = ProjectExecutor(store, pool)
    ex = _running_execution("exec-delivery-same-fp")
    ex.progress_log_seq = 9
    store.get_execution.return_value = ex
    executor._last_pfm_sync_seq[ex.id] = 9

    with patch(
        "projects.pfm_delivery.compute_canonical_delivery_stamp",
        return_value={"fingerprint": "fp-stable"},
    ), patch(
        "projects.pfm_refresh.try_refresh_pfm_from_delivery",
        return_value={"ok": True, "code": "no_changes"},
    ) as mock_refresh:
        asyncio.run(executor._sync_pfm_artifacts(ex.id))
        asyncio.run(executor._sync_pfm_artifacts(ex.id))

    assert mock_refresh.call_count == 1


def test_sync_throttles_delivery_scan_when_progress_unchanged() -> None:
    store = MagicMock()
    pool = MagicMock()
    executor = ProjectExecutor(store, pool)
    ex = _running_execution("exec-delivery-throttle")
    ex.progress_log_seq = 13
    store.get_execution.return_value = ex
    executor._last_pfm_sync_seq[ex.id] = 13

    with patch(
        "projects.pfm_delivery.compute_canonical_delivery_stamp",
        return_value={"fingerprint": "fp-a"},
    ) as mock_stamp, patch(
        "projects.pfm_refresh.try_refresh_pfm_from_delivery",
        return_value={"ok": True, "code": "materialized"},
    ) as mock_refresh:
        asyncio.run(executor._sync_pfm_artifacts(ex.id))
        asyncio.run(executor._sync_pfm_artifacts(ex.id))

    assert mock_stamp.call_count == 1
    assert mock_refresh.call_count == 1


def test_periodic_api_commit_nudge_asks_agent_to_call_tools() -> None:
    store = MagicMock()
    pool = MagicMock()
    pool.is_agent_active.return_value = False
    executor = ProjectExecutor(store, pool)
    ex = _running_execution("exec-periodic-api")

    executor._maybe_periodic_api_commit_nudge(ex.id, ex)

    pool.send_message_async.assert_called_once()
    kwargs = pool.send_message_async.call_args.kwargs
    msg = str(kwargs.get("user_message") or "")
    assert "commit_pfm_snapshot" in msg
    assert "publish_pfm_artifacts" in msg


def test_extract_progress_skips_transcript_load_when_no_new_messages() -> None:
    store = MagicMock()
    pool = MagicMock()
    executor = ProjectExecutor(store, pool)

    ex = ProjectExecute(
        id="exec-no-new-messages",
        linked_template_id="tpl-1",
        name="No new messages run",
        status=ExecutionStatus.RUNNING,
        run_session_key="sk-no-new",
        progress_log_seq=5,
        progress_log=[],
    )
    store.get_execution.return_value = ex

    db = MagicMock()
    db.message_count.return_value = 5
    pool._get_session_db.return_value = db
    pool._resolve_session_id.return_value = "session-no-new"

    asyncio.run(executor._extract_progress(ex.id))

    db.get_messages.assert_not_called()
