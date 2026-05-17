"""Tests for .FMR parsing and node report backfill."""

from __future__ import annotations

from pathlib import Path

import pytest

from projects.models import ExecutionStatus, ProjectExecute, ProjectTemplate
from projects.pfm_fmr_parse import (
    is_materialized_stub_markdown,
    load_canonical_fmr_reports,
    parse_fmr_node_reports,
    sync_node_reports_from_canonical_fmr,
)
from projects.store import ProjectStore
from tests.projects.test_pfm_tree import _good_payload


@pytest.fixture()
def fmr_store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> ProjectStore:
    report_root = tmp_path / "reports"
    monkeypatch.setenv("EAD_REPORT_DIR", str(report_root))
    store = ProjectStore(db_path=tmp_path / "projects.db")
    store.create_template(ProjectTemplate(id="tpl-fmr", name="Demo"))
    store.create_execution(
        ProjectExecute(
            id="exec-fmr",
            linked_template_id="tpl-fmr",
            name="Run",
            status=ExecutionStatus.COMPLETED,
        )
    )
    p = _good_payload(1)
    p["execution_id"] = "exec-fmr"
    from projects.pfm_tree import validate_and_normalize_snapshot

    snap, _, reps = validate_and_normalize_snapshot(p)
    store.replace_execution_pfm_tree("exec-fmr", snapshot=snap, node_reports=reps)
    out_dir = report_root / "exec-fmr"
    out_dir.mkdir(parents=True, exist_ok=True)
    fmr = out_dir / "demo.FMR"
    fmr.write_text(
        "# PFM Full Node EAD Report: Demo\n\n"
        "## Per-Node EAD Reports\n\n"
        "### Node 1: Login\n"
        "- Node Key: `auth/login`\n"
        "- Status: `Success`\n\n"
        "Description: Login flow\n\n"
        "#### Node EAD Report\n"
        "Node Summary:\n"
        "Purpose: Sign in\n\n"
        "Features:\n\n"
        "Feature F-001: Dashboard\n\n"
        "Test Case TC-001: Open login\n",
        encoding="utf-8",
    )
    return store


def test_parse_fmr_node_reports_extracts_keys():
    text = (
        "## Per-Node EAD Reports\n\n"
        "### Node 1: Reporting\n"
        "- Node Key: `reporting`\n\n"
        "#### Node EAD Report\n"
        "Node Summary:\nPurpose: Analytics\n"
    )
    out = parse_fmr_node_reports(text)
    assert "reporting" in out
    assert "Analytics" in out["reporting"]["markdown"]


def test_sync_replaces_materialized_stub_with_fmr(fmr_store: ProjectStore):
    stub = (
        "# EAD Feature Map — Login\n\n"
        "This markdown was **materialized from run data** when an operator used "
        "**Refresh EAD Feature Map**. It is not a live agent-authored `commit_pfm_snapshot`.\n"
    )
    fmr_store.save_node_ead_report_artifact(
        "exec-fmr",
        node_key="auth/login",
        title="Login",
        content=stub,
    )
    n = sync_node_reports_from_canonical_fmr(fmr_store, "exec-fmr")
    assert n >= 1
    art = fmr_store.get_execution_pfm_artifact(
        "exec-fmr",
        "node-ead-report-auth-login.md",
    )
    content = str(art.get("content") or "")
    assert "Feature F-001" in content
    assert not is_materialized_stub_markdown(content)


def test_load_canonical_fmr_reports_from_disk(fmr_store: ProjectStore):
    reports = load_canonical_fmr_reports("exec-fmr")
    assert "auth/login" in reports
