from __future__ import annotations

from pathlib import Path

import pytest

from projects.models import ExecutionStatus, ProjectExecute, ProjectTemplate, ReportingActivityStatus
from projects.store import ProjectStore


@pytest.fixture()
def reporting_store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> ProjectStore:
    db_path = tmp_path / "projects.db"
    monkeypatch.setenv("EAD_REPORT_BASE_DIR", str(tmp_path / "reports"))
    store = ProjectStore(db_path=db_path)
    store.create_template(ProjectTemplate(id="tpl-report", name="Template"))
    return store


def _mk_exec(execution_id: str, start_ms: int, status: ExecutionStatus = ExecutionStatus.RUNNING) -> ProjectExecute:
    return ProjectExecute(
        id=execution_id,
        linked_template_id="tpl-report",
        name=execution_id,
        status=status,
        start_time=start_ms,
        valid_for_data_reporting_training=True,
        contributes_to_learning=True,
    )


def test_single_active_run_enforced_by_recency(reporting_store: ProjectStore):
    reporting_store.create_execution(_mk_exec("exec-1", 1_000))
    reporting_store.create_execution(_mk_exec("exec-2", 2_000))
    latest = reporting_store.create_execution(_mk_exec("exec-3", 3_000))

    assert latest.reporting_activity_status == ReportingActivityStatus.ACTIVE
    assert reporting_store.get_execution("exec-1").reporting_activity_status == ReportingActivityStatus.CLOSED
    assert reporting_store.get_execution("exec-2").reporting_activity_status == ReportingActivityStatus.CLOSED
    assert reporting_store.get_active_reporting_execution_id("tpl-report") == "exec-3"


def test_close_active_does_not_promote_finished_or_old_runs(reporting_store: ProjectStore):
    reporting_store.create_execution(_mk_exec("exec-1", 1_000))
    reporting_store.create_execution(_mk_exec("exec-2", 2_000))
    reporting_store.create_execution(_mk_exec("exec-3", 3_000))

    reporting_store.set_execution_reporting_activity("exec-3", active=False)
    assert reporting_store.get_active_reporting_execution_id("tpl-report") is None
    assert reporting_store.get_execution("exec-3").reporting_activity_status == ReportingActivityStatus.CLOSED


def test_pending_run_can_still_be_selected_as_active(reporting_store: ProjectStore):
    reporting_store.create_execution(_mk_exec("exec-1", 1_000, status=ExecutionStatus.COMPLETED))
    pending = reporting_store.create_execution(
        _mk_exec("exec-2", 2_000, status=ExecutionStatus.PENDING)
    )
    assert pending.reporting_activity_status == ReportingActivityStatus.ACTIVE
    assert reporting_store.get_active_reporting_execution_id("tpl-report") == "exec-2"


def test_terminal_execution_created_closed(reporting_store: ProjectStore):
    completed = reporting_store.create_execution(
        _mk_exec("exec-done", 1_000, status=ExecutionStatus.COMPLETED)
    )

    assert completed.reporting_activity_status == ReportingActivityStatus.CLOSED
    assert reporting_store.get_active_reporting_execution_id("tpl-report") is None


def test_get_active_executions_excludes_reporting_closed(reporting_store: ProjectStore):
    active = reporting_store.create_execution(
        _mk_exec("exec-active", 1_000, status=ExecutionStatus.RUNNING)
    )
    closed = reporting_store.create_execution(
        _mk_exec("exec-closed", 2_000, status=ExecutionStatus.PENDING)
    )
    reporting_store.set_execution_reporting_activity(active.id, active=True)
    reporting_store.set_execution_reporting_activity(closed.id, active=False)

    active_ids = {execution.id for execution in reporting_store.get_active_executions()}
    assert active.id in active_ids
    assert closed.id not in active_ids


def test_close_other_active_executions_is_global(reporting_store: ProjectStore):
    reporting_store.create_template(ProjectTemplate(id="tpl-other", name="Other"))
    first = reporting_store.create_execution(
        _mk_exec("exec-first", 1_000, status=ExecutionStatus.RUNNING)
    )
    second = ProjectExecute(
        id="exec-second",
        linked_template_id="tpl-other",
        name="exec-second",
        status=ExecutionStatus.PENDING,
        start_time=2_000,
        valid_for_data_reporting_training=True,
        contributes_to_learning=True,
    )
    reporting_store.create_execution(second)

    closed_ids = reporting_store.close_other_active_executions(second.id)

    assert first.id in closed_ids
    assert reporting_store.get_execution(first.id).status == ExecutionStatus.CANCELLED
    assert (
        reporting_store.get_execution(first.id).reporting_activity_status
        == ReportingActivityStatus.CLOSED
    )
    assert [execution.id for execution in reporting_store.get_active_executions()] == [
        second.id
    ]

