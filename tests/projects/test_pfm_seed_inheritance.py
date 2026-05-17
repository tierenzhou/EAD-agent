"""PFM seed must not copy committed tree artifacts onto a new execution."""

from __future__ import annotations

from pathlib import Path

import pytest

from projects.models import ExecutionStatus, ProjectExecute, ProjectTemplate
from projects.pfm_tree import PFM_TREE_ARTIFACT_KEY, PFM_TREE_ARTIFACT_TYPE, PFM_VIEW_STATE_ARTIFACT_KEY
from projects.store import ProjectStore
from tests.projects.test_pfm_canonical import _commit_minimal_tree


@pytest.fixture()
def seed_store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> ProjectStore:
    db_path = tmp_path / "projects.db"
    monkeypatch.setenv("EAD_REPORT_BASE_DIR", str(tmp_path / "reports"))
    monkeypatch.setenv("EAD_PFM_AGENT_AUTHORED", "true")
    store = ProjectStore(db_path=db_path)
    store.create_template(ProjectTemplate(id="tpl-1", name="T"))
    store.create_execution(
        ProjectExecute(
            id="exec-old",
            linked_template_id="tpl-1",
            name="Old",
            status=ExecutionStatus.COMPLETED,
            contributes_to_learning=True,
            valid_for_data_reporting_training=True,
        )
    )
    store.create_execution(
        ProjectExecute(
            id="exec-new",
            linked_template_id="tpl-1",
            name="New",
            status=ExecutionStatus.PENDING,
        )
    )
    return store


def test_seed_from_prior_run_skips_pfm_tree(seed_store: ProjectStore) -> None:
    _commit_minimal_tree(seed_store, "exec-old")
    assert seed_store.has_committed_pfm_tree("exec-old") is True

    seed_store.upsert_execution_pfm_artifact(
        "exec-old",
        "mindmap-1",
        "pfm_mindmap",
        {"nodes": [{"node_key": "n1", "title": "Node 1"}]},
    )

    seeded = seed_store.seed_execution_from_prior_run("exec-new", "exec-old")
    assert seeded is not None
    assert seeded.inherited_from_execution_id == "exec-old"
    assert seed_store.has_committed_pfm_tree("exec-new") is False
    assert seed_store.get_execution_pfm_artifact("exec-new", PFM_TREE_ARTIFACT_KEY) is None
    assert seed_store.resolve_pfm_baseline_execution_id(seeded) == "exec-old"
    assert seed_store.get_execution_pfm_artifact("exec-new", "mindmap-1") is not None


def test_seed_progress_log_is_one_line_without_report_bodies(seed_store: ProjectStore) -> None:
    seed_store.upsert_execution_pfm_artifact(
        "exec-old",
        "report-login",
        "node_ead_report",
        {
            "title": "Login Portal",
            "content": "# Login Portal\n\nParent: auth-access\n\nLong body " + ("x" * 800),
        },
    )
    seed_store.upsert_execution_pfm_artifact(
        "exec-old",
        "mindmap-1",
        "pfm_mindmap",
        {"nodes": [{"node_key": "n1", "title": "Node 1"}]},
    )

    seeded = seed_store.seed_execution_from_prior_run("exec-new", "exec-old")
    assert seeded is not None
    system_lines = [
        e.text or ""
        for e in (seeded.progress_log or [])
        if getattr(e, "kind", None) == "system"
    ]
    assert len(system_lines) == 1
    line = system_lines[0]
    assert line.startswith("PFM inherited from run exec-old")
    assert "Login Portal" not in line
    assert "# Login Portal" not in line
    assert len(line) < 200


def test_seed_from_prior_run_skips_view_state(seed_store: ProjectStore) -> None:
    seed_store.upsert_execution_pfm_artifact(
        "exec-old",
        PFM_VIEW_STATE_ARTIFACT_KEY,
        "pfm_view_state",
        {"selectedNodeKey": "n1"},
    )
    seed_store.seed_execution_from_prior_run("exec-new", "exec-old")
    assert seed_store.get_execution_pfm_artifact("exec-new", PFM_VIEW_STATE_ARTIFACT_KEY) is None
