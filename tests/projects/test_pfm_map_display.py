"""PFM map display: Current vs Previous from DeliveredPFMMapReport."""

from __future__ import annotations

from pathlib import Path

import pytest

from projects.models import ExecutionStatus, ProjectExecute, ProjectTemplate
from projects.pfm_map_display import resolve_pfm_map_display
from projects.store import ProjectStore
from tests.projects.test_pfm_canonical import _commit_minimal_tree
from tests.projects.test_pfm_lineage_context import _paired_delivery_snapshot


@pytest.fixture()
def display_store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> ProjectStore:
    reports = tmp_path / "reports"
    reports.mkdir()
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("EAD_REPORT_BASE_DIR", str(reports))
    monkeypatch.setenv("EAD_REPORT_DIR", str(reports))
    monkeypatch.setenv("EAD_PFM_AGENT_AUTHORED", "true")
    store = ProjectStore(db_path=tmp_path / "projects.db")
    store.create_template(ProjectTemplate(id="tpl-1", name="P1 Test"))
    store.create_execution(
        ProjectExecute(
            id="exec-finished",
            linked_template_id="tpl-1",
            name="Run 1",
            status=ExecutionStatus.COMPLETED,
            start_time=1000,
        )
    )
    _commit_minimal_tree(store, "exec-finished")
    snap = store.get_committed_pfm_tree("exec-finished")
    assert snap is not None
    store.replace_execution_pfm_tree(
        "exec-finished",
        snapshot=_paired_delivery_snapshot(dict(snap), "exec-finished"),
        node_reports=[{"node_key": "root", "title": "Root", "markdown": "# v1\n"}],
    )
    store.create_execution(
        ProjectExecute(
            id="exec-active",
            linked_template_id="tpl-1",
            name="Run 2",
            status=ExecutionStatus.RUNNING,
            inherited_from_execution_id="exec-finished",
            start_time=2000,
        )
    )
    return store


def test_active_run_without_delivery_shows_previous(display_store: ProjectStore) -> None:
    ex = display_store.get_execution("exec-active")
    assert ex is not None
    ctx = resolve_pfm_map_display(display_store, ex)
    assert ctx["pfm_map_display_mode"] == "previous"
    assert ctx["has_delivered_pfm_map_report"] is False
    assert ctx["pfm_has_committed_snapshot"] is False
    assert ctx["pfm_tree_read_execution_id"] == "exec-finished"
    assert ctx["pfm_source_run_number"] >= 1


def test_active_run_with_disk_delivery_shows_current(
    display_store: ProjectStore, tmp_path: Path,
) -> None:
    run_id = "exec-active"
    report_dir = tmp_path / "reports" / run_id
    report_dir.mkdir(parents=True)
    (report_dir / "p1-test-exec-act.pfm").write_text("{}", encoding="utf-8")
    (report_dir / "p1-test-exec-act.FMR").write_text("# ok", encoding="utf-8")

    ex = display_store.get_execution(run_id)
    assert ex is not None
    ctx = resolve_pfm_map_display(display_store, ex)
    assert ctx["pfm_map_display_mode"] == "current"
    assert ctx["has_delivered_pfm_map_report"] is True
    assert ctx["pfm_tree_read_execution_id"] == run_id
