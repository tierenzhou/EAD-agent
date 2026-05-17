"""PFM lineage: V = per-template run number."""

from __future__ import annotations

from pathlib import Path

import pytest

from projects.models import ExecutionStatus, ProjectExecute, ProjectTemplate
from projects.pfm_run_number import get_run_number
from projects.store import ProjectStore
from tests.projects.test_pfm_canonical import _commit_minimal_tree


@pytest.fixture()
def lineage_store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> ProjectStore:
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("EAD_REPORT_BASE_DIR", str(tmp_path / "reports"))
    monkeypatch.setenv("EAD_PFM_AGENT_AUTHORED", "true")
    store = ProjectStore(db_path=tmp_path / "projects.db")
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
    store.update_template("tpl-1", canonical_pfm_execution_id="exec-baseline")
    store.create_execution(
        ProjectExecute(
            id="exec-new",
            linked_template_id="tpl-1",
            name="New run",
            status=ExecutionStatus.RUNNING,
            inherited_from_execution_id="exec-baseline",
            start_time=2000,
        )
    )
    return store


def test_run_numbers_per_template(lineage_store: ProjectStore) -> None:
    base = lineage_store.get_execution("exec-baseline")
    new = lineage_store.get_execution("exec-new")
    assert base is not None and new is not None
    assert get_run_number(lineage_store, base) == 1
    assert get_run_number(lineage_store, new) == 2


def test_lineage_baseline_preview(lineage_store: ProjectStore) -> None:
    ex = lineage_store.get_execution("exec-new")
    assert ex is not None
    ctx = lineage_store.resolve_pfm_lineage_context(ex)
    assert ctx["pfm_snapshot_phase"] == "baseline_preview"
    assert ctx["run_number"] == 2
    assert ctx["pfm_baseline_tree_version"] == 1
    assert ctx["pfm_generation_version"] == 2
    assert ctx["pfm_revision"] == 0
    assert ctx["pfm_has_committed_snapshot"] is False


def test_lineage_evolving_after_first_commit(lineage_store: ProjectStore) -> None:
    snap = lineage_store.get_committed_pfm_tree("exec-baseline")
    assert snap is not None
    first = dict(snap)
    for key in ("generation", "revision", "finalized", "finalized_at"):
        first.pop(key, None)
    first["version"] = 1
    lineage_store.replace_execution_pfm_tree(
        "exec-new",
        snapshot=first,
        node_reports=[{"node_key": "root", "title": "Root", "markdown": "# Root Rev 1\n"}],
    )
    ex = lineage_store.get_execution("exec-new")
    assert ex is not None
    ctx = lineage_store.resolve_pfm_lineage_context(ex)
    assert ctx["pfm_snapshot_phase"] == "evolving"
    assert ctx["pfm_generation_version"] == 2
    assert ctx["pfm_revision"] == 1
    snap = lineage_store.get_committed_pfm_tree("exec-new")
    assert snap is not None
    assert snap.get("generation") == 2
    assert snap.get("revision") == 1


def test_lineage_walks_past_intermediate_runs_without_snapshots(
    lineage_store: ProjectStore,
) -> None:
    store = lineage_store
    store.create_execution(
        ProjectExecute(
            id="exec-mid-a",
            linked_template_id="tpl-1",
            name="Mid A",
            status=ExecutionStatus.COMPLETED,
            inherited_from_execution_id="exec-baseline",
            start_time=1500,
        )
    )
    store.create_execution(
        ProjectExecute(
            id="exec-mid-b",
            linked_template_id="tpl-1",
            name="Mid B",
            status=ExecutionStatus.COMPLETED,
            inherited_from_execution_id="exec-mid-a",
            start_time=1600,
        )
    )
    store.create_execution(
        ProjectExecute(
            id="exec-current",
            linked_template_id="tpl-1",
            name="Current",
            status=ExecutionStatus.COMPLETED,
            inherited_from_execution_id="exec-mid-b",
            start_time=1700,
        )
    )
    snap = store.get_committed_pfm_tree("exec-baseline")
    assert snap is not None
    committed = dict(snap)
    for key in ("generation", "revision", "finalized", "finalized_at"):
        committed.pop(key, None)
    committed["version"] = 1
    store.replace_execution_pfm_tree(
        "exec-current",
        snapshot=committed,
        node_reports=[{"node_key": "root", "title": "Root", "markdown": "# Root\n"}],
    )
    ex = store.get_execution("exec-current")
    assert ex is not None
    ctx = store.resolve_pfm_lineage_context(ex)
    assert ctx["run_number"] == 4
    assert ctx["pfm_baseline_tree_version"] == 3
    assert ctx["pfm_generation_version"] == 4
    assert ctx["pfm_revision"] == 1
    snap_out = store.get_committed_pfm_tree("exec-current")
    assert snap_out is not None
    assert snap_out.get("generation") == 4


def test_third_run_gets_v3(lineage_store: ProjectStore) -> None:
    store = lineage_store
    store.create_execution(
        ProjectExecute(
            id="exec-third",
            linked_template_id="tpl-1",
            name="Third",
            status=ExecutionStatus.COMPLETED,
            inherited_from_execution_id="exec-baseline",
            start_time=3000,
        )
    )
    snap = store.get_committed_pfm_tree("exec-baseline")
    assert snap is not None
    first = dict(snap)
    for key in ("generation", "revision", "finalized"):
        first.pop(key, None)
    first["version"] = 1
    store.replace_execution_pfm_tree(
        "exec-third",
        snapshot=first,
        node_reports=[{"node_key": "root", "title": "Root", "markdown": "# v3\n"}],
    )
    ex = store.get_execution("exec-third")
    assert ex is not None
    ctx = store.resolve_pfm_lineage_context(ex)
    assert ctx["pfm_generation_version"] == 3
    assert ctx["pfm_baseline_tree_version"] == 2
    saved = store.get_committed_pfm_tree("exec-third")
    assert saved is not None
    assert saved.get("generation") == 3


def test_lineage_final_when_completed(lineage_store: ProjectStore) -> None:
    snap = lineage_store.get_committed_pfm_tree("exec-baseline")
    assert snap is not None
    first = dict(snap)
    for key in ("generation", "revision", "finalized", "finalized_at"):
        first.pop(key, None)
    first["version"] = 1
    lineage_store.replace_execution_pfm_tree(
        "exec-new",
        snapshot=first,
        node_reports=[{"node_key": "root", "title": "Root", "markdown": "# Root Rev 1\n"}],
    )
    second = dict(lineage_store.get_committed_pfm_tree("exec-new") or {})
    second["version"] = 2
    lineage_store.replace_execution_pfm_tree(
        "exec-new",
        snapshot=second,
        node_reports=[{"node_key": "root", "title": "Root", "markdown": "# Root Rev 2\n"}],
    )
    lineage_store.update_execution("exec-new", status=ExecutionStatus.COMPLETED)
    ex = lineage_store.get_execution("exec-new")
    assert ex is not None
    ctx = lineage_store.resolve_pfm_lineage_context(ex)
    assert ctx["pfm_snapshot_phase"] == "final"
    assert ctx["pfm_generation_version"] == 2
    assert ctx["pfm_revision"] == 2
    snap = lineage_store.get_committed_pfm_tree("exec-new")
    assert snap is not None
    assert snap.get("generation") == 2
    assert snap.get("revision") == 2
