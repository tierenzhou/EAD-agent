"""Tests for PFM hierarchy reconciliation from node_ead_report artifacts."""

from __future__ import annotations

from pathlib import Path

import pytest

from projects.models import ExecutionStatus, ProjectExecute, ProjectTemplate
from projects.pfm_hierarchy import (
    build_report_hierarchy_index,
    needs_hierarchy_reconcile,
    persist_reconciled_snapshot_hierarchy,
    reconcile_flat_nodes_from_node_reports,
)
from projects.store import ProjectStore
from tests.projects.test_pfm_canonical import _commit_minimal_tree


@pytest.fixture()
def hierarchy_store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> ProjectStore:
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("EAD_REPORT_DIR", str(tmp_path / "reports"))
    monkeypatch.setenv("EAD_PFM_AGENT_AUTHORED", "true")
    store = ProjectStore(db_path=tmp_path / "projects.db")
    store.create_template(ProjectTemplate(id="tpl-h", name="H"))
    store.create_execution(
        ProjectExecute(
            id="exec-shallow",
            linked_template_id="tpl-h",
            name="Shallow",
            status=ExecutionStatus.COMPLETED,
            start_time=1000,
        )
    )
    _commit_minimal_tree(store, "exec-shallow")
    snap = store.get_committed_pfm_tree("exec-shallow") or {}
    shallow = dict(snap)
    shallow["flat_nodes"] = [
        {
            "node_key": "p1-test",
            "node_id": "p1-test",
            "parent_node_key": None,
            "level": 1,
            "title": "Hub",
            "type": "hub",
            "status": "No Run",
            "description": "Hub",
            "cross_cutting": False,
        },
        {
            "node_key": "p1-test/test-case",
            "node_id": "p1-test/test-case",
            "parent_node_key": "p1-test",
            "level": 1,
            "title": "Test Case",
            "type": "module",
            "status": "No Run",
            "description": "Test Case module",
            "cross_cutting": False,
        },
    ]
    shallow["roots"] = [
        {
            "node_key": "p1-test",
            "title": "Hub",
            "level": 1,
            "type": "hub",
            "status": "No Run",
            "description": "Hub",
            "children": [
                {
                    "node_key": "p1-test/test-case",
                    "title": "Test Case",
                    "level": 1,
                    "type": "module",
                    "status": "No Run",
                    "description": "Test Case module",
                    "children": [],
                }
            ],
        }
    ]
    store.replace_execution_pfm_tree(
        "exec-shallow",
        snapshot=shallow,
        node_reports=[{"node_key": "p1-test", "title": "Hub", "markdown": "# Hub\n"}],
    )
    return store


def test_reconcile_adds_children_under_committed_leaf() -> None:
    flat = [
        {
            "node_key": "p1-test",
            "parent_node_key": None,
            "level": 1,
            "title": "Hub",
        },
        {
            "node_key": "p1-test/test-case",
            "parent_node_key": "p1-test",
            "level": 1,
            "title": "Test Case",
        },
    ]
    artifacts = [
        {
            "artifact_type": "node_ead_report",
            "node_key": "legacy/6-test-case/6-1-list-view",
            "title": "6.1 List View",
            "content": "# Report\n",
        },
        {
            "artifact_type": "node_ead_report",
            "node_key": "legacy/6-test-case/6-2-toggle",
            "title": "6.2 Toggle",
            "content": "# Report\n",
        },
    ]
    key_to_title, direct_by_parent = build_report_hierarchy_index(artifacts)
    assert needs_hierarchy_reconcile(flat, key_to_title, direct_by_parent)
    new_flat, added = reconcile_flat_nodes_from_node_reports(
        flat,
        key_to_title,
        direct_by_parent,
    )
    assert added >= 2
    child_keys = [
        n["node_key"]
        for n in new_flat
        if str(n.get("parent_node_key") or "") == "p1-test/test-case"
    ]
    assert "p1-test/test-case/6-1-list-view" in child_keys
    assert "p1-test/test-case/6-2-toggle" in child_keys


def test_persist_reconciled_snapshot_upgrades_db(hierarchy_store: ProjectStore) -> None:
    hierarchy_store.save_node_ead_report_artifact(
        "exec-shallow",
        node_key="legacy-root/6-test-case/6-1-list-view",
        title="6.1 List View",
        content="# Node report\n\nFeature F-001: List\n",
    )
    hierarchy_store.save_node_ead_report_artifact(
        "exec-shallow",
        node_key="legacy-root/6-test-case/6-2-toggle",
        title="6.2 Toggle",
        content="# Node report\n\nFeature F-002: Toggle\n",
    )
    out = persist_reconciled_snapshot_hierarchy(hierarchy_store, "exec-shallow")
    assert out.get("ok") is True
    assert out.get("code") == "tree_upgraded"
    assert int(out.get("nodes_added") or 0) >= 2
    snap = hierarchy_store.get_committed_pfm_tree("exec-shallow") or {}
    flat = list(snap.get("flat_nodes") or [])
    assert len(flat) >= 3
    child_parents = {
        str(n.get("parent_node_key") or "")
        for n in flat
        if "/" in str(n.get("node_key") or "")
    }
    assert any("test-case" in p for p in child_parents)


def test_persist_reconciled_is_idempotent(hierarchy_store: ProjectStore) -> None:
    hierarchy_store.save_node_ead_report_artifact(
        "exec-shallow",
        node_key="legacy-root/6-test-case/6-1-list-view",
        title="6.1 List View",
        content="# Node report\n",
    )
    first = persist_reconciled_snapshot_hierarchy(hierarchy_store, "exec-shallow")
    second = persist_reconciled_snapshot_hierarchy(hierarchy_store, "exec-shallow")
    assert first.get("code") == "tree_upgraded"
    assert second.get("code") == "no_changes"
    assert int(second.get("nodes_added") or 0) == 0
