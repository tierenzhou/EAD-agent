"""Unit tests for the agent-authored PFM tree pipeline.

Covers:
- `commit_pfm_snapshot` validator (`validate_and_normalize_snapshot`).
- Mermaid scope slicing (`render_mermaid_for_scope`).
- View-state migration across tree versions (`apply_view_state_to_tree`).
- End-to-end persistence and rendering via `ProjectStore.replace_execution_pfm_tree`.
"""

from __future__ import annotations

import copy
import os
import time
from pathlib import Path

import pytest

from projects.models import (
    ExecutionStatus,
    ProjectExecute,
    ProjectTemplate,
)
from projects.pfm_artifacts import resolve_pfm_nodes_for_mindmap
from projects.pfm_tree import (
    PFM_TREE_ARTIFACT_KEY,
    SnapshotValidationError,
    apply_view_state_to_tree,
    render_mermaid_for_scope,
    validate_and_normalize_snapshot,
)
from projects.store import ProjectStore


def _good_payload(version: int = 1) -> dict:
    return {
        "execution_id": "exec-1",
        "version": version,
        "generated_at": 1_700_000_000_000 + version,
        "roots": [
            {
                "node_key": "auth",
                "title": "Auth",
                "level": 1,
                "type": "domain",
                "status": "Success",
                "description": "Login + sessions",
                "children": [
                    {
                        "node_key": "auth/login",
                        "title": "Login",
                        "parent_node_key": "auth",
                        "level": 2,
                        "status": "Success",
                        "description": "Credentials, SSO",
                    },
                    {
                        "node_key": "auth/signup",
                        "title": "Signup",
                        "parent_node_key": "auth",
                        "level": 2,
                        "status": "No Run",
                        "description": "New account",
                    },
                ],
            },
            {
                "node_key": "studio",
                "title": "Studio",
                "level": 1,
                "status": "Success",
                "description": "Editor",
            },
        ],
        "cross_cutting": [
            {
                "node_key": "shared",
                "title": "Shared",
                "level": 1,
                "status": "No Run",
                "description": "Bucket for cross-cutting concerns",
            }
        ],
        "node_reports": [
            {"node_key": "auth", "title": "Auth", "markdown": "Auth report"},
            {"node_key": "auth/login", "title": "Login", "markdown": "Login report"},
            {"node_key": "auth/signup", "title": "Signup", "markdown": "Signup report"},
            {"node_key": "studio", "title": "Studio", "markdown": "Studio report"},
            {"node_key": "shared", "title": "Shared", "markdown": "Shared report"},
        ],
    }


# ---------------------------------------------------------------------------
# Validator
# ---------------------------------------------------------------------------


def test_validator_accepts_well_formed_tree():
    snapshot, runs, reports = validate_and_normalize_snapshot(_good_payload(1))
    assert snapshot["version"] == 1
    flat_keys = [n["node_key"] for n in snapshot["flat_nodes"]]
    assert flat_keys == [
        "auth",
        "auth/login",
        "auth/signup",
        "studio",
        "shared",
    ]
    parents = {n["node_key"]: n["parent_node_key"] for n in snapshot["flat_nodes"]}
    assert parents["auth"] is None
    assert parents["auth/login"] == "auth"
    assert parents["studio"] is None
    assert {r["node_key"] for r in reports} == set(flat_keys)
    assert {r.node_key for r in runs} == set(flat_keys)


def test_validator_rejects_missing_reports():
    payload = _good_payload(1)
    payload["node_reports"] = [
        r for r in payload["node_reports"] if r["node_key"] != "auth/login"
    ]
    with pytest.raises(SnapshotValidationError) as info:
        validate_and_normalize_snapshot(payload)
    assert info.value.code == "missing_node_reports"


def test_validator_rejects_empty_markdown():
    payload = _good_payload(1)
    payload["node_reports"][0]["markdown"] = ""
    with pytest.raises(SnapshotValidationError) as info:
        validate_and_normalize_snapshot(payload)
    assert info.value.code == "report_empty_markdown"


def test_validator_rejects_duplicate_node_keys():
    payload = _good_payload(1)
    payload["roots"][0]["children"].append(
        {"node_key": "auth/login", "title": "Dup", "parent_node_key": "auth", "level": 2}
    )
    payload["node_reports"].append({"node_key": "auth/login", "markdown": "dup"})
    with pytest.raises(SnapshotValidationError) as info:
        validate_and_normalize_snapshot(payload)
    assert info.value.code == "duplicate_node_key"


def test_validator_rejects_parent_prefix_mismatch():
    payload = _good_payload(1)
    payload["roots"][0]["children"][0]["node_key"] = "other/login"
    with pytest.raises(SnapshotValidationError) as info:
        validate_and_normalize_snapshot(payload)
    assert info.value.code == "parent_prefix_mismatch"


def test_validator_rejects_stale_version():
    payload = _good_payload(2)
    with pytest.raises(SnapshotValidationError) as info:
        validate_and_normalize_snapshot(payload, previous_version=5)
    assert info.value.code == "stale_version"


def test_validator_rejects_empty_tree():
    payload = _good_payload(1)
    payload["roots"] = []
    payload["cross_cutting"] = []
    with pytest.raises(SnapshotValidationError) as info:
        validate_and_normalize_snapshot(payload)
    assert info.value.code == "empty_tree"


def test_validator_rejects_unknown_report_node_key():
    payload = _good_payload(1)
    payload["node_reports"].append({"node_key": "ghost", "title": "Ghost", "markdown": "orphan"})
    with pytest.raises(SnapshotValidationError) as info:
        validate_and_normalize_snapshot(payload)
    assert info.value.code == "unknown_report_node_key"


def test_incremental_reports_carried_from_prior_tree():
    snap1, _, reports1 = validate_and_normalize_snapshot(_good_payload(1))
    prev_keys = {n["node_key"] for n in snap1["flat_nodes"]}
    carry_lib = {
        r["node_key"]: {"title": r["title"], "markdown": r["markdown"]} for r in reports1
    }
    payload = copy.deepcopy(_good_payload(2))
    payload["roots"][0]["children"].append(
        {
            "node_key": "auth/oauth",
            "title": "OAuth",
            "parent_node_key": "auth",
            "level": 2,
            "type": "feature-area",
            "status": "No Run",
            "description": "External SSO",
        }
    )
    payload["node_reports"] = [
        {"node_key": "auth/oauth", "title": "OAuth", "markdown": "OAuth-only delta for this commit."}
    ]
    snap2, _, merged = validate_and_normalize_snapshot(
        payload,
        previous_version=1,
        report_carry_source_keys=prev_keys,
        report_carry_library=carry_lib,
    )
    assert len(snap2["flat_nodes"]) == 6
    assert len(merged) == 6
    byk = {r["node_key"]: r["markdown"] for r in merged}
    assert byk["auth/oauth"] == "OAuth-only delta for this commit."
    assert byk["auth"] == "Auth report"
    assert byk["studio"] == "Studio report"


def test_incremental_fails_when_carry_library_missing_node():
    snap1, _, reports1 = validate_and_normalize_snapshot(_good_payload(1))
    prev_keys = {n["node_key"] for n in snap1["flat_nodes"]}
    carry_deficient = {
        r["node_key"]: {"title": r["title"], "markdown": r["markdown"]}
        for r in reports1
        if r["node_key"] != "studio"
    }
    payload = copy.deepcopy(_good_payload(2))
    # Omit full node_reports so unchanged nodes must be satisfied from carry_library.
    payload["node_reports"] = [
        {"node_key": "auth", "title": "Auth", "markdown": "Only an explicit delta in this commit."}
    ]
    with pytest.raises(SnapshotValidationError) as info:
        validate_and_normalize_snapshot(
            payload,
            previous_version=1,
            report_carry_source_keys=prev_keys,
            report_carry_library=carry_deficient,
        )
    assert info.value.code == "missing_node_reports"
    assert "studio" in str(info.value)


def _execution_for_render() -> ProjectExecute:
    return ProjectExecute(id="exec-1", linked_template_id="tpl-1", name="Demo Run")


def test_mermaid_top_scope_emits_top_level_nodes_only():
    snapshot, _, _ = validate_and_normalize_snapshot(_good_payload(1))
    mermaid = render_mermaid_for_scope(snapshot, _execution_for_render(), scope="top")
    assert mermaid.startswith("mindmap")
    assert "Auth [Success]" in mermaid
    assert "Studio [Success]" in mermaid
    assert "Shared [No Run]" in mermaid
    # top scope must not descend past depth 1
    assert "Login [Success]" not in mermaid
    assert "Signup [No Run]" not in mermaid


def test_mermaid_full_scope_emits_entire_tree():
    snapshot, _, _ = validate_and_normalize_snapshot(_good_payload(1))
    mermaid = render_mermaid_for_scope(snapshot, _execution_for_render(), scope="full")
    for label in ("Auth", "Login", "Signup", "Studio", "Shared"):
        assert f"{label} [" in mermaid


def test_mermaid_subtree_scope_anchors_on_node_key():
    snapshot, _, _ = validate_and_normalize_snapshot(_good_payload(1))
    mermaid = render_mermaid_for_scope(
        snapshot,
        _execution_for_render(),
        scope="subtree",
        node_key="auth",
        depth=2,
    )
    assert "Auth [Success]" in mermaid
    assert "Login [Success]" in mermaid
    assert "Signup [No Run]" in mermaid
    assert "Studio" not in mermaid
    assert "Shared" not in mermaid


def test_mermaid_path_scope_includes_only_path_to_node():
    snapshot, _, _ = validate_and_normalize_snapshot(_good_payload(1))
    mermaid = render_mermaid_for_scope(
        snapshot, _execution_for_render(), scope="path", node_key="auth/login"
    )
    assert "Auth [Success]" in mermaid
    assert "Login [Success]" in mermaid
    assert "Signup" not in mermaid
    assert "Studio" not in mermaid


def test_mermaid_subtree_for_unknown_node_returns_diagnostic():
    snapshot, _, _ = validate_and_normalize_snapshot(_good_payload(1))
    mermaid = render_mermaid_for_scope(
        snapshot,
        _execution_for_render(),
        scope="subtree",
        node_key="nonexistent",
        depth=2,
    )
    assert "Unknown node_key" in mermaid


def test_mermaid_same_snapshot_deterministic():
    snapshot, _, _ = validate_and_normalize_snapshot(_good_payload(1))
    a = render_mermaid_for_scope(snapshot, _execution_for_render(), scope="full")
    b = render_mermaid_for_scope(snapshot, _execution_for_render(), scope="full")
    assert a == b


# ---------------------------------------------------------------------------
# View-state migration
# ---------------------------------------------------------------------------


def test_view_state_preserves_selection_when_node_survives():
    snapshot, _, _ = validate_and_normalize_snapshot(_good_payload(1))
    saved = {
        "node_path": ["auth", "auth/login"],
        "selected_node_key": "auth/login",
        "view_scope": "subtree",
        "depth_cap": 3,
    }
    repaired = apply_view_state_to_tree(saved, snapshot)
    assert repaired["selected_node_key"] == "auth/login"
    assert repaired["node_path"] == ["auth", "auth/login"]
    assert repaired["view_scope"] == "subtree"
    assert repaired["depth_cap"] == 3
    assert repaired["pfm_tree_version"] == 1


def test_view_state_migrates_to_nearest_surviving_ancestor():
    # Newer tree drops `auth/login` but keeps `auth`.
    payload = _good_payload(2)
    payload["roots"][0]["children"] = [
        {"node_key": "auth/signup", "title": "Signup", "parent_node_key": "auth", "level": 2}
    ]
    payload["node_reports"] = [
        r for r in payload["node_reports"] if r["node_key"] != "auth/login"
    ]
    snapshot, _, _ = validate_and_normalize_snapshot(payload, previous_version=1)
    saved = {
        "node_path": ["auth", "auth/login"],
        "selected_node_key": "auth/login",
        "view_scope": "subtree",
        "depth_cap": 2,
    }
    repaired = apply_view_state_to_tree(saved, snapshot)
    assert repaired["selected_node_key"] == "auth"
    assert repaired["node_path"] == ["auth"]
    assert repaired["view_scope"] == "subtree"


def test_view_state_falls_back_to_top_when_nothing_survives():
    payload = _good_payload(2)
    # Replace auth subtree entirely.
    payload["roots"] = [
        {"node_key": "new", "title": "New", "level": 1, "status": "No Run", "description": "x"},
    ]
    payload["cross_cutting"] = []
    payload["node_reports"] = [{"node_key": "new", "markdown": "new"}]
    snapshot, _, _ = validate_and_normalize_snapshot(payload, previous_version=1)
    saved = {
        "node_path": ["auth", "auth/login"],
        "selected_node_key": "auth/login",
        "view_scope": "subtree",
        "depth_cap": 2,
    }
    repaired = apply_view_state_to_tree(saved, snapshot)
    assert repaired["selected_node_key"] is None
    assert repaired["node_path"] == []
    assert repaired["view_scope"] == "top"


# ---------------------------------------------------------------------------
# Persistence + strict resolver
# ---------------------------------------------------------------------------


@pytest.fixture()
def isolated_store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> ProjectStore:
    """Build a ProjectStore that persists into the test's tmp_path."""
    db_path = tmp_path / "projects.db"
    monkeypatch.setenv("EAD_REPORT_BASE_DIR", str(tmp_path / "reports"))
    monkeypatch.setenv("EAD_PFM_AGENT_AUTHORED", "true")
    store = ProjectStore(db_path=db_path)
    template = ProjectTemplate(id="tpl-1", name="Test Template")
    store.create_template(template)
    execution = ProjectExecute(
        id="exec-1",
        linked_template_id="tpl-1",
        name="Test Run",
        status=ExecutionStatus.RUNNING,
    )
    store.create_execution(execution)
    return store


def test_replace_execution_pfm_tree_persists_snapshot_and_reports(isolated_store: ProjectStore):
    snapshot, _, reports = validate_and_normalize_snapshot(_good_payload(1))
    result = isolated_store.replace_execution_pfm_tree(
        "exec-1", snapshot=snapshot, node_reports=reports
    )
    assert result["committed"] is True
    assert result["pfm_tree_version"] == 1
    saved_tree = isolated_store.get_committed_pfm_tree("exec-1")
    assert saved_tree is not None
    assert int(saved_tree["version"]) == 1
    assert {n["node_key"] for n in saved_tree["flat_nodes"]} == {
        "auth",
        "auth/login",
        "auth/signup",
        "studio",
        "shared",
    }
    # node_ead_report artifacts persisted for every node.
    artifacts = isolated_store.list_execution_pfm_artifacts("exec-1")
    node_reports = [a for a in artifacts if a.get("artifact_type") == "node_ead_report"]
    assert {a.get("node_key") for a in node_reports} == {
        "auth",
        "auth/login",
        "auth/signup",
        "studio",
        "shared",
    }


def test_strict_resolver_reads_only_committed_tree(isolated_store: ProjectStore):
    snapshot, _, reports = validate_and_normalize_snapshot(_good_payload(1))
    isolated_store.replace_execution_pfm_tree(
        "exec-1", snapshot=snapshot, node_reports=reports
    )
    execution = isolated_store.get_execution("exec-1")
    nodes = resolve_pfm_nodes_for_mindmap(execution)
    keys = {n.node_key for n in nodes}
    assert keys == {"auth", "auth/login", "auth/signup", "studio", "shared"}


def test_strict_resolver_returns_empty_when_no_commit(isolated_store: ProjectStore):
    execution = isolated_store.get_execution("exec-1")
    # No tree committed yet; strict mode returns whatever `execution.results` holds
    # (also empty in this fresh run).
    nodes = resolve_pfm_nodes_for_mindmap(execution)
    assert nodes == []


def test_replace_execution_pfm_tree_rejects_stale_version_via_validator():
    """The validator enforces monotonic version; the persistence layer is
    only called after a successful validation, so this is the operational
    contract callers must obey."""
    snapshot_v3, _, _ = validate_and_normalize_snapshot(_good_payload(3))
    # If a follow-up commit tries to land v2 after v3, the validator must reject.
    payload_v2 = _good_payload(2)
    with pytest.raises(SnapshotValidationError) as info:
        validate_and_normalize_snapshot(payload_v2, previous_version=int(snapshot_v3["version"]))
    assert info.value.code == "stale_version"
