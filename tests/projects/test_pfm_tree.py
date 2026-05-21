"""Unit tests for the agent-authored PFM tree pipeline.

Covers:
- `commit_pfm_snapshot` validator (`validate_and_normalize_snapshot`).
- Mermaid scope slicing (`render_mermaid_for_scope`).
- View-state migration across tree versions (`apply_view_state_to_tree`).
- End-to-end persistence and rendering via `ProjectStore.replace_execution_pfm_tree`.
"""

from __future__ import annotations

import copy
import json
import os
import time
from pathlib import Path

import pytest

from projects.models import (
    EadFmNodeRun,
    ExecutionStatus,
    ProgressLogEntry,
    ProjectExecute,
    ProjectTemplate,
)
from projects.pfm_artifacts import resolve_pfm_nodes_for_mindmap
from projects.pfm_tree import (
    PFM_TREE_ARTIFACT_KEY,
    SnapshotValidationError,
    apply_view_state_to_tree,
    collapse_duplicate_hub_path_segments,
    normalize_operator_flat_nodes,
    operator_top_level_spoke_nodes,
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


def test_validator_rejects_too_many_siblings():
    children = [
        {
            "node_key": f"area/child-{i}",
            "title": f"Child {i}",
            "parent_node_key": "area",
            "level": 2,
            "status": "No Run",
            "description": "x",
        }
        for i in range(16)
    ]
    payload = _good_payload(1)
    payload["roots"] = [
        {
            "node_key": "area",
            "title": "Area",
            "level": 1,
            "status": "No Run",
            "description": "Grouped area",
            "children": children,
        }
    ]
    payload["node_reports"] = [
        {"node_key": "area", "title": "Area", "markdown": "Area report"},
        *[
            {"node_key": c["node_key"], "title": c["title"], "markdown": "child"}
            for c in children
        ],
    ]
    with pytest.raises(SnapshotValidationError) as exc:
        validate_and_normalize_snapshot(payload)
    assert exc.value.code == "too_many_siblings"


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


def test_mermaid_strips_markdown_hr_from_node_titles():
    payload = _good_payload(1)
    payload["roots"][0]["title"] = "Review the current state.\n-----------------------"
    snapshot, _, _ = validate_and_normalize_snapshot(payload)
    mermaid = render_mermaid_for_scope(snapshot, _execution_for_render(), scope="full")
    assert "-----------------------" not in mermaid
    assert "current state" in mermaid


def test_operator_top_level_spokes_are_hub_children_not_hub_row():
    payload = _good_payload(1)
    snapshot, _, _ = validate_and_normalize_snapshot(payload)
    flat = list(snapshot["flat_nodes"])
    hub_key = "product-hub"
    flat.append(
        {
            "node_key": hub_key,
            "node_id": hub_key,
            "parent_node_key": None,
            "level": 1,
            "title": "Internal Hub",
            "type": "feature-area",
            "status": "No Run",
            "description": "",
            "cross_cutting": False,
        }
    )
    for mod, title in (("auth", "Authentication"), ("report", "Report")):
        flat.append(
            {
                "node_key": f"{hub_key}/{mod}",
                "node_id": f"{hub_key}/{mod}",
                "parent_node_key": hub_key,
                "level": 2,
                "title": title,
                "type": "feature-area",
                "status": "No Run",
                "description": "",
                "cross_cutting": False,
            }
        )
    snapshot["flat_nodes"] = flat
    spokes = operator_top_level_spoke_nodes(flat)
    spoke_keys = {n["node_key"] for n in spokes}
    assert hub_key not in spoke_keys
    assert f"{hub_key}/auth" in spoke_keys
    assert f"{hub_key}/report" in spoke_keys
    mermaid = render_mermaid_for_scope(snapshot, _execution_for_render(), scope="top")
    assert "Internal Hub" not in mermaid
    assert "Authentication" in mermaid
    assert "Report" in mermaid


def test_collect_center_hub_keys_empty_after_normalize_promotes_modules():
    flat = [
        {
            "node_key": "hub",
            "node_id": "hub",
            "parent_node_key": None,
            "level": 1,
            "title": "Hub",
            "type": "feature-area",
            "status": "No Run",
            "description": "",
            "cross_cutting": False,
        },
        {
            "node_key": "hub/auth",
            "node_id": "hub/auth",
            "parent_node_key": "hub",
            "level": 2,
            "title": "Authentication",
            "type": "feature-area",
            "status": "No Run",
            "description": "",
            "cross_cutting": False,
        },
        {
            "node_key": "hub/report",
            "node_id": "hub/report",
            "parent_node_key": "hub",
            "level": 2,
            "title": "Report",
            "type": "feature-area",
            "status": "No Run",
            "description": "",
            "cross_cutting": False,
        },
    ]
    from projects.pfm_tree import collect_center_hub_keys

    assert collect_center_hub_keys(flat) == {"hub"}
    normalized = normalize_operator_flat_nodes(flat)
    assert collect_center_hub_keys(normalized) == set()
    spokes = operator_top_level_spoke_nodes(normalized)
    assert {n["node_key"] for n in spokes} == {"hub/auth", "hub/report"}


def test_normalize_operator_flat_nodes_removes_hub_and_collapses_paths():
    flat = [
        {
            "node_key": "hub",
            "node_id": "hub",
            "parent_node_key": None,
            "level": 1,
            "title": "Hub",
            "type": "feature-area",
            "status": "No Run",
            "description": "",
            "cross_cutting": False,
        },
        {
            "node_key": "hub/hub/report",
            "node_id": "hub/hub/report",
            "parent_node_key": "hub",
            "level": 2,
            "title": "Report",
            "type": "feature-area",
            "status": "No Run",
            "description": "",
            "cross_cutting": False,
        },
    ]
    out = normalize_operator_flat_nodes(flat)
    assert len(out) == 1
    assert out[0]["node_key"] == "hub/report"
    assert out[0]["parent_node_key"] is None


def test_mermaid_top_rejects_agent_narration_flat_tree():
    payload = _good_payload(1)
    snapshot, _, _ = validate_and_normalize_snapshot(payload)
    flat = [
        {
            "node_key": "n1",
            "node_id": "n1",
            "parent_node_key": None,
            "level": 1,
            "title": "There was a timeout - let me check the current state.",
            "type": "feature-area",
            "status": "No Run",
            "description": "",
            "cross_cutting": False,
        },
        {
            "node_key": "n2",
            "node_id": "n2",
            "parent_node_key": None,
            "level": 1,
            "title": "Good, screenshot captured! Now let me click Log In.",
            "type": "feature-area",
            "status": "No Run",
            "description": "",
            "cross_cutting": False,
        },
        {
            "node_key": "n3",
            "node_id": "n3",
            "parent_node_key": None,
            "level": 1,
            "title": "## Run Introduction I have received the assignment",
            "type": "feature-area",
            "status": "No Run",
            "description": "",
            "cross_cutting": False,
        },
    ]
    snapshot["flat_nodes"] = flat
    mermaid = render_mermaid_for_scope(snapshot, _execution_for_render(), scope="top")
    assert "Invalid PFM tree" in mermaid
    assert "current state" not in mermaid


def test_mermaid_strips_unicode_hr_from_node_titles():
    payload = _good_payload(1)
    payload["roots"][0]["title"] = "Review the current state.\n\u2014\u2014\u2014\u2014\u2014"
    snapshot, _, _ = validate_and_normalize_snapshot(payload)
    mermaid = render_mermaid_for_scope(snapshot, _execution_for_render(), scope="top")
    assert "\u2014\u2014\u2014" not in mermaid
    assert "current state" in mermaid


def test_mermaid_top_scope_emits_top_level_nodes_only():
    snapshot, _, _ = validate_and_normalize_snapshot(_good_payload(1))
    mermaid = render_mermaid_for_scope(snapshot, _execution_for_render(), scope="top")
    assert mermaid.startswith("mindmap")
    assert "root((Demo Run))" in mermaid
    assert "Auth" in mermaid
    assert "Studio" in mermaid
    assert "Shared" in mermaid
    # top scope must not descend past depth 1
    assert "Login [Success]" not in mermaid
    assert "Signup [No Run]" not in mermaid


def test_collapse_duplicate_hub_path_segments():
    hub = "kloud-1-0-br-p1-test-br-project-one"
    assert collapse_duplicate_hub_path_segments(f"{hub}/{hub}/8-report") == f"{hub}/8-report"
    assert collapse_duplicate_hub_path_segments("auth/login") == "auth/login"


def test_strip_pfm_node_display_title_removes_leading_numbers():
    from projects.pfm_tree import strip_pfm_node_display_title

    assert strip_pfm_node_display_title("1. Authentication") == "Authentication"
    assert strip_pfm_node_display_title("10. Exploration Summary") == "Exploration Summary"
    assert strip_pfm_node_display_title("🔐 2. Home / Task Board") == "Home / Task Board"
    assert strip_pfm_node_display_title("1.2 Login Form") == "Login Form"


def test_mermaid_top_scope_strips_leading_numbers_from_titles():
    payload = {
        "execution_id": "exec-num",
        "version": 1,
        "roots": [
            {
                "node_key": "hub",
                "title": "Product Hub",
                "level": 1,
                "status": "No Run",
                "description": "hub",
                "children": [
                    {
                        "node_key": "hub/auth",
                        "title": "1. Authentication",
                        "parent_node_key": "hub",
                        "level": 2,
                        "status": "Success",
                        "description": "a",
                    },
                ],
            }
        ],
        "cross_cutting": [],
        "node_reports": [
            {"node_key": "hub", "markdown": "# h"},
            {"node_key": "hub/auth", "markdown": "# a"},
        ],
    }
    snapshot, _, _ = validate_and_normalize_snapshot(payload)
    ex = ProjectExecute(id="exec-num", linked_template_id="t", name="P1 Test")
    mermaid = render_mermaid_for_scope(snapshot, ex, scope="top")
    assert "Authentication" in mermaid
    assert "1. Authentication" not in mermaid


def test_mermaid_top_scope_expands_single_hub_children():
    """First screen: project center + modules under the lone null-parent hub."""
    payload = {
        "execution_id": "exec-hub",
        "version": 1,
        "roots": [
            {
                "node_key": "p1-hub",
                "title": "Kloud Product",
                "level": 1,
                "status": "No Run",
                "description": "hub",
                "children": [
                    {
                        "node_key": "p1-hub/auth",
                        "title": "Authentication",
                        "parent_node_key": "p1-hub",
                        "level": 2,
                        "status": "Success",
                        "description": "auth",
                    },
                    {
                        "node_key": "p1-hub/home",
                        "title": "Home",
                        "parent_node_key": "p1-hub",
                        "level": 2,
                        "status": "No Run",
                        "description": "home",
                    },
                ],
            }
        ],
        "cross_cutting": [],
        "node_reports": [
            {"node_key": "p1-hub", "markdown": "# hub"},
            {"node_key": "p1-hub/auth", "markdown": "# auth"},
            {"node_key": "p1-hub/home", "markdown": "# home"},
        ],
    }
    snapshot, _, _ = validate_and_normalize_snapshot(payload)
    ex = ProjectExecute(id="exec-hub", linked_template_id="t", name="Run: P1 Test")
    mermaid = render_mermaid_for_scope(snapshot, ex, scope="top")
    assert "root((P1 Test))" in mermaid
    assert "Authentication" in mermaid
    assert "Home" in mermaid
    assert "Kloud Product" not in mermaid


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


def test_mermaid_focus_resolves_collapsed_hub_path():
    snapshot, _, _ = validate_and_normalize_snapshot(_good_payload(1))
    flat = list(snapshot["flat_nodes"])
    hub = "product-hub"
    module = f"{hub}/{hub}/report"
    flat.append(
        {
            "node_key": module,
            "node_id": module,
            "parent_node_key": hub,
            "level": 2,
            "title": "Report",
            "type": "feature-area",
            "status": "No Run",
            "description": "",
            "cross_cutting": False,
        }
    )
    flat.append(
        {
            "node_key": f"{module}/summary",
            "node_id": f"{module}/summary",
            "parent_node_key": module,
            "level": 3,
            "title": "Summary",
            "type": "feature-area",
            "status": "No Run",
            "description": "",
            "cross_cutting": False,
        }
    )
    snapshot["flat_nodes"] = flat
    mermaid = render_mermaid_for_scope(
        snapshot,
        _execution_for_render(),
        scope="focus",
        node_key=f"{hub}/report",
    )
    assert "Unknown node_key" not in mermaid
    assert "Summary" in mermaid


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


def test_maybe_bootstrap_pfm_snapshot_from_results_writes_v1(isolated_store: ProjectStore):
    nodes = [
        EadFmNodeRun(
            node_key="z_root",
            node_id="z_root",
            title="Root",
            level=1,
            type="domain",
            parent_node_key=None,
        ),
        EadFmNodeRun(
            node_key="z_root/leaf",
            node_id="z_root/leaf",
            title="Leaf",
            level=2,
            type="feature-area",
            parent_node_key="z_root",
        ),
    ]
    isolated_store.update_execution("exec-1", results=nodes)
    assert isolated_store.get_committed_pfm_tree("exec-1") is None
    assert isolated_store.maybe_bootstrap_pfm_snapshot_from_results("exec-1") is True
    snap = isolated_store.get_committed_pfm_tree("exec-1")
    assert snap is not None
    assert int(snap.get("version") or 0) == 1
    keys = {n["node_key"] for n in snap.get("flat_nodes") or []}
    assert keys == {"z_root", "z_root/leaf"}
    assert isolated_store.maybe_bootstrap_pfm_snapshot_from_results("exec-1") is False


def test_active_inherited_run_defers_auto_snapshot_from_results(
    isolated_store: ProjectStore, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setenv("EAD_PFM_SYNC_SNAPSHOT_FROM_RESULTS", "1")
    nodes = [
        EadFmNodeRun(
            node_key="z_root",
            node_id="z_root",
            title="Root",
            level=1,
            type="domain",
            parent_node_key=None,
        ),
        EadFmNodeRun(
            node_key="z_root/leaf",
            node_id="z_root/leaf",
            title="Leaf",
            level=2,
            type="feature-area",
            parent_node_key="z_root",
        ),
    ]
    isolated_store.update_execution(
        "exec-1",
        status=ExecutionStatus.RUNNING,
        inherited_from_execution_id="exec-parent",
        results=nodes,
    )
    assert isolated_store.get_committed_pfm_tree("exec-1") is None
    assert isolated_store.maybe_bootstrap_pfm_snapshot_from_results("exec-1") is False
    assert isolated_store.maybe_incremental_commit_from_execution_results("exec-1") is False
    assert isolated_store.get_committed_pfm_tree("exec-1") is None


def test_maybe_commit_pfm_snapshot_after_publish_merges_progress_log_nodes(
    isolated_store: ProjectStore,
):
    """When results[] is empty but report_running_step logged pfm_node entries, publish
    should still be able to materialize a canonical v1 tree for the strict mindmap."""
    progress = [
        ProgressLogEntry(
            kind="tool_use",
            tool_name="report_running_step",
            tool_input={
                "title": "Root",
                "pfm_node": {
                    "node_key": "prog_root",
                    "title": "Root",
                    "level": 1,
                    "type": "domain",
                    "parent_node_key": None,
                },
            },
        ),
        ProgressLogEntry(
            kind="tool_use",
            tool_name="report_running_step",
            tool_input={
                "title": "Leaf",
                "pfm_node": {
                    "node_key": "leaf",
                    "title": "Leaf",
                    "level": 2,
                    "type": "feature-area",
                    "parent_node_key": "prog_root",
                },
            },
        ),
    ]
    isolated_store.update_execution("exec-1", results=[], progress_log=progress)
    assert isolated_store.get_committed_pfm_tree("exec-1") is None
    assert isolated_store.maybe_commit_pfm_snapshot_after_publish("exec-1") is True
    snap = isolated_store.get_committed_pfm_tree("exec-1")
    assert snap is not None
    assert int(snap.get("version") or 0) == 1
    keys = {n["node_key"] for n in snap.get("flat_nodes") or []}
    assert keys == {"prog_root", "prog_root/leaf"}


def test_maybe_incremental_commit_from_execution_results_bumps_version(
    isolated_store: ProjectStore, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setenv("EAD_PFM_SYNC_SNAPSHOT_FROM_RESULTS", "1")
    nodes = [
        EadFmNodeRun(
            node_key="z_root",
            node_id="z_root",
            title="Root",
            level=1,
            type="domain",
            parent_node_key=None,
        ),
        EadFmNodeRun(
            node_key="z_root/leaf",
            node_id="z_root/leaf",
            title="Leaf",
            level=2,
            type="feature-area",
            parent_node_key="z_root",
        ),
    ]
    isolated_store.update_execution("exec-1", results=nodes)
    assert isolated_store.maybe_bootstrap_pfm_snapshot_from_results("exec-1") is True
    assert int(isolated_store.get_committed_pfm_tree("exec-1").get("version") or 0) == 1

    nodes2 = nodes + [
        EadFmNodeRun(
            node_key="z_root/leaf2",
            node_id="z_root/leaf2",
            title="Leaf 2",
            level=2,
            type="feature-area",
            parent_node_key="z_root",
        ),
    ]
    isolated_store.update_execution("exec-1", results=nodes2)
    assert isolated_store.maybe_incremental_commit_from_execution_results("exec-1") is True
    snap = isolated_store.get_committed_pfm_tree("exec-1")
    assert int(snap.get("version") or 0) == 2
    keys = {n["node_key"] for n in snap.get("flat_nodes") or []}
    assert keys == {"z_root", "z_root/leaf", "z_root/leaf2"}
    assert isolated_store.maybe_incremental_commit_from_execution_results("exec-1") is False


def test_persist_pfm_tree_from_execution_state_materializes_with_sync_env_off(
    isolated_store: ProjectStore, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setenv("EAD_PFM_SYNC_SNAPSHOT_FROM_RESULTS", "0")
    nodes = [
        EadFmNodeRun(
            node_key="p_root",
            node_id="p_root",
            title="Root",
            level=1,
            type="domain",
            parent_node_key=None,
        ),
        EadFmNodeRun(
            node_key="p_root/leaf",
            node_id="p_root/leaf",
            title="Leaf",
            level=2,
            type="feature-area",
            parent_node_key="p_root",
        ),
    ]
    isolated_store.update_execution("exec-1", results=nodes)
    assert isolated_store.maybe_incremental_commit_from_execution_results("exec-1") is False
    assert isolated_store.get_committed_pfm_tree("exec-1") is None

    out = isolated_store.persist_pfm_tree_from_execution_state("exec-1")
    assert out["ok"] is True
    assert out["code"] == "materialized"
    assert int(out.get("version") or 0) == 1
    snap = isolated_store.get_committed_pfm_tree("exec-1")
    assert snap is not None
    assert int(snap.get("version") or 0) == 1

    out2 = isolated_store.persist_pfm_tree_from_execution_state("exec-1")
    assert out2["ok"] is True
    assert out2["code"] == "no_changes"


def test_replace_execution_pfm_tree_satisfies_pending_refresh_request(isolated_store: ProjectStore):
    isolated_store.update_execution("exec-1", pfm_refresh_pending_request_id="rid-pending-1")
    snapshot, _, reports = validate_and_normalize_snapshot(_good_payload(1))
    isolated_store.replace_execution_pfm_tree("exec-1", snapshot=snapshot, node_reports=reports)
    ex = isolated_store.get_execution("exec-1")
    assert ex.pfm_refresh_pending_request_id is None
    assert ex.pfm_refresh_satisfied_request_id == "rid-pending-1"


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


def test_read_ead_execution_includes_results_and_committed_version(isolated_store: ProjectStore):
    """Regression: discovery prompts refer to results[]; the tool must return them."""
    from projects.tools import register_project_tools
    from tools.registry import registry

    register_project_tools(isolated_store)
    nodes = [
        EadFmNodeRun(
            node_key="alpha",
            node_id="alpha",
            title="Alpha",
            level=1,
            type="domain",
            parent_node_key=None,
        ),
        EadFmNodeRun(
            node_key="alpha/beta",
            node_id="alpha/beta",
            title="Beta",
            level=2,
            type="feature-area",
            parent_node_key="alpha",
        ),
    ]
    isolated_store.update_execution("exec-1", results=nodes)
    raw = registry.dispatch("read_ead_execution", {"execution_id": "exec-1"})
    data = json.loads(raw)
    assert data["results_count"] == 2
    assert len(data["results"]) == 2
    assert data["results"][0]["node_key"] == "alpha"
    assert data["results"][1]["node_key"] == "alpha/beta"
    assert data["results_truncated"] is False
    assert data["committed_pfm_tree_version"] == 0


def test_normalize_pfm_snapshot_payload_accepts_camel_case():
    """Models sometimes emit camelCase keys; the commit tool must accept them."""
    from projects.tools import _normalize_pfm_snapshot_payload

    raw = {
        "executionId": "exec-x",
        "version": 1,
        "roots": [
            {
                "nodeKey": "billing",
                "title": "Billing",
                "level": 1,
                "type": "domain",
                "status": "No Run",
                "description": "",
                "children": [
                    {
                        "nodeKey": "billing/invoices",
                        "parentNodeKey": "billing",
                        "title": "Invoices",
                        "level": 2,
                        "type": "feature-area",
                        "status": "No Run",
                        "description": "",
                    },
                ],
            },
        ],
        "nodeReports": [
            {"nodeKey": "billing", "markdown": "# Billing\n"},
            {"nodeKey": "billing/invoices", "markdown": "# Invoices\n"},
        ],
    }
    norm = _normalize_pfm_snapshot_payload(raw)
    snap, _, _ = validate_and_normalize_snapshot(norm)
    assert snap["version"] == 1
    keys = {n["node_key"] for n in snap["flat_nodes"]}
    assert keys == {"billing", "billing/invoices"}
