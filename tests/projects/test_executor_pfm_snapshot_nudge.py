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
