"""API fallback behavior for inherited runs without own committed PFM tree."""

from __future__ import annotations

import asyncio
import json
from unittest.mock import MagicMock, patch

from projects.api import ProjectHandlers
from projects.models import ExecutionStatus, ProjectExecute


class _GetReq:
    def __init__(self, execution_id: str, query: dict | None = None) -> None:
        self.match_info = {"execution_id": execution_id}
        self.query = query or {}


class _ListReq:
    def __init__(self, query: dict | None = None) -> None:
        self.query = query or {}


def test_mindmap_falls_back_to_baseline_committed_tree() -> None:
    store = MagicMock()
    ex = ProjectExecute(
        id="exec-new",
        linked_template_id="tpl-1",
        name="Run 57",
        status=ExecutionStatus.RUNNING,
        inherited_from_execution_id="exec-old",
    )
    baseline_snapshot = {
        "version": 12,
        "generated_at": 1700000000000,
        "flat_nodes": [{"node_key": "root", "title": "Root", "status": "No Run"}],
    }
    store.get_execution.return_value = ex
    store.resolve_pfm_lineage_context.return_value = {
        "pfm_tree_read_execution_id": "exec-old",
        "pfm_baseline_execution_id": "exec-old",
    }
    store.has_committed_pfm_tree.side_effect = lambda eid: eid != "exec-new"
    store.get_committed_pfm_tree.side_effect = (
        lambda eid: None if eid == "exec-new" else baseline_snapshot
    )

    handlers = ProjectHandlers(store=store, executor=MagicMock())
    with patch("projects.pfm_tree.render_mermaid_for_scope", return_value="mindmap\n  root((Run 57))\n    Root\n"), patch(
        "projects.pfm_tree.list_clickable_nodes_for_scope",
        return_value=[{"node_key": "root", "title": "Root", "label": "Root", "child_count": 0}],
    ):
        resp = asyncio.run(handlers.handle_get_pfm_mindmap(_GetReq("exec-new", {"scope": "top"})))

    payload = json.loads(resp.text)
    assert payload["clickableNodes"][0]["nodeKey"] == "root"
    assert payload["committed"] is False
    assert payload["sourceExecutionId"] == "exec-old"
    assert payload["readExecutionId"] == "exec-old"
    assert payload["baselineExecutionId"] == "exec-old"
    assert payload["version"] == 12
    assert "Root" in payload["mermaid"]


def test_mindmap_uses_previous_when_current_has_bootstrap_tree_but_no_delivery() -> None:
    store = MagicMock()
    ex = ProjectExecute(
        id="exec-new",
        linked_template_id="tpl-1",
        name="Run 69",
        status=ExecutionStatus.RUNNING,
        inherited_from_execution_id="exec-old",
    )
    current_snapshot = {
        "version": 4,
        "generated_at": 1700000001000,
        "flat_nodes": [{"node_key": "current", "title": "Current Bootstrap", "status": "No Run"}],
    }
    baseline_snapshot = {
        "version": 13,
        "generated_at": 1700000000000,
        "flat_nodes": [{"node_key": "previous", "title": "Previous DB Map", "status": "No Run"}],
    }
    baseline = ProjectExecute(
        id="exec-old",
        linked_template_id="tpl-1",
        name="Run 66",
        status=ExecutionStatus.COMPLETED,
        run_number=66,
    )

    store.get_execution.side_effect = lambda eid: baseline if eid == "exec-old" else ex
    store.resolve_pfm_lineage_context.return_value = {
        "pfm_map_display_mode": "previous",
        "has_delivered_pfm_map_report": False,
        "pfm_tree_read_execution_id": "exec-old",
        "pfm_baseline_execution_id": "exec-old",
    }
    store.has_committed_pfm_tree.return_value = True
    store.get_committed_pfm_tree.side_effect = (
        lambda eid: current_snapshot if eid == "exec-new" else baseline_snapshot
    )

    handlers = ProjectHandlers(store=store, executor=MagicMock())
    with patch(
        "projects.pfm_tree.render_mermaid_for_scope",
        return_value="mindmap\n  root((Run 69))\n    Previous DB Map\n",
    ), patch("projects.pfm_tree.list_clickable_nodes_for_scope", return_value=[]):
        resp = asyncio.run(handlers.handle_get_pfm_mindmap(_GetReq("exec-new", {"scope": "top"})))

    payload = json.loads(resp.text)
    assert payload["committed"] is False
    assert payload["sourceExecutionId"] == "exec-old"
    assert payload["readExecutionId"] == "exec-old"
    assert payload["baselineExecutionId"] == "exec-old"
    assert payload["sourceRunNumber"] == 66
    assert payload["version"] == 13
    assert "Previous DB Map" in payload["mermaid"]


def test_persisted_state_falls_back_to_baseline_node_reports_when_current_empty() -> None:
    store = MagicMock()
    ex = ProjectExecute(
        id="exec-new",
        linked_template_id="tpl-1",
        name="Run 57",
        status=ExecutionStatus.RUNNING,
        inherited_from_execution_id="exec-old",
    )
    baseline_snapshot = {
        "version": 12,
        "generated_at": 1700000000000,
        "flat_nodes": [{"node_key": "root", "title": "Root", "status": "No Run"}],
    }
    baseline_report = {
        "artifact_type": "node_ead_report",
        "artifact_key": "node-ead-report-root.md",
        "node_key": "root",
        "title": "Root",
        "content": "# Root report",
        "excerpt": "Root report",
        "created_at": 1700000000001,
    }

    store.get_execution.return_value = ex
    store.resolve_pfm_lineage_context.return_value = {
        "pfm_tree_read_execution_id": "exec-old",
        "pfm_baseline_execution_id": "exec-old",
    }
    store.has_committed_pfm_tree.side_effect = lambda eid: eid != "exec-new"
    store.get_committed_pfm_tree.side_effect = (
        lambda eid: None if eid == "exec-new" else baseline_snapshot
    )
    store.list_execution_pfm_artifacts.side_effect = (
        lambda eid: [] if eid == "exec-new" else [baseline_report]
    )

    handlers = ProjectHandlers(store=store, executor=MagicMock())
    resp = asyncio.run(handlers.handle_get_pfm_persisted_state(_GetReq("exec-new")))
    payload = json.loads(resp.text)
    assert payload["readExecutionId"] == "exec-old"
    assert payload["baselineExecutionId"] == "exec-old"
    assert payload["hasOwnCommittedSnapshot"] is False
    assert payload["committedSnapshot"] is not None
    assert len(payload["nodeReports"]) == 1
    assert payload["nodeReports"][0]["sourceExecutionId"] == "exec-old"


def test_list_executions_enriches_lineage_for_payload_rows() -> None:
    store = MagicMock()
    ex = ProjectExecute(
        id="exec-new",
        linked_template_id="tpl-1",
        name="Run 59",
        status=ExecutionStatus.RUNNING,
        inherited_from_execution_id="exec-old",
    )
    store.list_execution_payloads.return_value = [
        {
            "id": "exec-new",
            "name": "Run 59",
            "status": "running",
            "linked_template_id": "tpl-1",
            "run_number": 59,
            "inherited_from_execution_id": "exec-old",
            "pfm_has_committed_snapshot": False,
            "pfm_tree_read_execution_id": "exec-old",
            "pfm_baseline_execution_id": "exec-old",
            "pfm_snapshot_phase": "baseline_preview",
            "pfm_map_display_mode": "previous",
        }
    ]

    handlers = ProjectHandlers(store=store, executor=MagicMock())
    resp = asyncio.run(handlers.handle_list_executions(_ListReq()))
    payload = json.loads(resp.text)
    row = payload["executions"][0]

    assert row["id"] == "exec-new"
    assert row["pfmHasCommittedSnapshot"] is False
    assert row["pfmTreeReadExecutionId"] == "exec-old"
    assert row["pfmBaselineExecutionId"] == "exec-old"
    assert row["pfmSnapshotPhase"] == "baseline_preview"
