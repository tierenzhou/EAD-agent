"""Auto-commit PFM tree to DB after artifact delivery when none exists yet."""

from __future__ import annotations

from pathlib import Path

import pytest

from projects.models import ExecutionStatus, ProjectExecute, ProjectTemplate
from projects.pfm_artifacts import build_and_persist_pfm_artifacts
from projects.pfm_materialize import ensure_committed_pfm_snapshot_after_artifact_delivery
from projects.store import ProjectStore
from tests.projects.test_pfm_canonical import _commit_minimal_tree


def test_ensure_materializes_after_publish_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
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
    store.create_execution(
        ProjectExecute(
            id="exec-pub",
            linked_template_id="tpl-1",
            name="Publish test",
            status=ExecutionStatus.COMPLETED,
            start_time=1000,
        )
    )
    _commit_minimal_tree(store, "exec-baseline")
    baseline = store.get_execution("exec-baseline")
    assert baseline is not None
    store.update_execution(
        "exec-pub",
        results=list(baseline.results or []),
        inherited_from_execution_id="exec-baseline",
    )
    execution = store.get_execution("exec-pub")
    assert execution is not None
    reports = build_and_persist_pfm_artifacts(execution, project_store=store)
    store.update_execution("exec-pub", reports=reports)
    store.sync_execution_pfm_artifacts_from_state("exec-pub")

    result = ensure_committed_pfm_snapshot_after_artifact_delivery(store, "exec-pub")
    assert result.get("ok") is True
    assert result.get("code") == "materialized"
    assert store.has_committed_pfm_tree("exec-pub")
