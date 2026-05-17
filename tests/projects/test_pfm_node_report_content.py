"""Tests for node report normalize/validate and FMR-first delivery."""

from __future__ import annotations

from pathlib import Path

import pytest

from projects.models import ExecutionStatus, ProgressLogEntry, ProjectExecute, ProjectTemplate
from projects.pfm_deliverables import generate_pfm_deliverables
from projects.pfm_node_report_content import (
    backfill_node_reports_from_progress_log,
    is_valid_node_report_markdown,
    normalize_node_report_markdown,
)
from projects.store import ProjectStore
from tests.projects.test_pfm_tree import _good_payload


def test_normalize_strips_preamble_and_marker():
    raw = (
        "[Node-Report-Reply-To: REQ-NODE-ABC]\n"
        "I now have sufficient evidence to compile the report.\n\n"
        "Node Summary:\n"
        "Purpose: User settings area\n"
        "In-scope behaviors:\n"
        "- Toggle preferences\n"
    )
    out = normalize_node_report_markdown(raw)
    assert out.startswith("Node Summary:")
    assert "sufficient evidence" not in out


def test_valid_summary_only_report():
    md = normalize_node_report_markdown(
        "Node Summary:\n"
        "Purpose: Central preferences for the application.\n"
        "In-scope behaviors:\n"
        "- Open profile\n"
        "- Change theme\n"
        "- Log out\n"
    )
    assert is_valid_node_report_markdown(md)


def test_invalid_too_short():
    assert not is_valid_node_report_markdown("Node Summary:\nPurpose: Hi\n")


@pytest.fixture()
def deliverables_store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> ProjectStore:
    report_root = tmp_path / "reports"
    monkeypatch.setenv("EAD_REPORT_DIR", str(report_root))
    store = ProjectStore(db_path=tmp_path / "projects.db")
    store.create_template(ProjectTemplate(id="tpl-del", name="Demo App"))
    store.create_execution(
        ProjectExecute(
            id="exec-del",
            linked_template_id="tpl-del",
            name="Run",
            status=ExecutionStatus.COMPLETED,
        )
    )
    p = _good_payload(1)
    p["execution_id"] = "exec-del"
    from projects.pfm_tree import validate_and_normalize_snapshot

    snap, _, reps = validate_and_normalize_snapshot(p)
    store.replace_execution_pfm_tree("exec-del", snapshot=snap, node_reports=reps)

    rich = (
        "Node Summary:\n"
        "Purpose: Login gateway\n"
        "In-scope behaviors:\n"
        "- Submit credentials\n\n"
        "Features:\n"
        "Feature F-001: Sign in\n"
        "Description: Primary login\n\n"
        "Test Case TC-001: Login\n"
        "Objective: Verify login\n"
        "Preconditions: NONE\n"
        "Test Data: NONE\n\n"
        "Steps:\n"
        "Step 1: Open login\n"
        "   - Expected Result: Form visible\n"
        "   - Evidence Image: NONE\n"
    )
    store.save_node_ead_report_artifact(
        "exec-del",
        node_key="auth/login",
        title="Login",
        content=rich,
    )
    return store


def test_api_rejects_invalid_report(deliverables_store: ProjectStore):
    from projects.pfm_node_report_content import is_valid_node_report_markdown

    assert not is_valid_node_report_markdown("Node Summary:\nPurpose: only one line\n")


def test_save_normalized_via_store(deliverables_store: ProjectStore):
    preamble = (
        "Here is the report.\n\n"
        "Node Summary:\n"
        "Purpose: Area\n"
        "In-scope behaviors:\n"
        "- One\n"
    )
    art = deliverables_store.save_node_ead_report_artifact(
        "exec-del",
        node_key="auth/login",
        title="Login",
        content=preamble,
    )
    content = str(art.get("content") or "")
    assert content.startswith("Node Summary:")
    assert "Here is the report" not in content


def test_generate_fmr_includes_db_report(deliverables_store: ProjectStore, tmp_path: Path):
    generate_pfm_deliverables(deliverables_store, "exec-del")
    fmr_files = list((tmp_path / "reports" / "exec-del").glob("*.FMR"))
    assert fmr_files
    text = fmr_files[0].read_text(encoding="utf-8")
    assert "Feature F-001: Sign in" in text
    assert "Test Case TC-001: Login" in text


def test_backfill_from_progress_log(deliverables_store: ProjectStore):
    execution = deliverables_store.get_execution("exec-del")
    assert execution
    execution.progress_log = [
        ProgressLogEntry(
            kind="user",
            text=(
                "[Node-Report-Request-Id: REQ-NODE-X]\n"
                "PFM node key: user-settings\n"
                "Generate a detailed PFM node report."
            ),
        ),
        ProgressLogEntry(
            kind="assistant",
            text=(
                "Node Summary:\n"
                "Purpose: User settings container\n"
                "In-scope behaviors:\n"
                "- Avatar menu\n"
                "- Preference toggles\n"
            ),
        ),
    ]
    deliverables_store.update_execution("exec-del", progress_log=execution.progress_log)
    n = backfill_node_reports_from_progress_log(deliverables_store, "exec-del")
    assert n >= 1
    art = deliverables_store.get_execution_pfm_artifact(
        "exec-del",
        "node-ead-report-user-settings.md",
    )
    assert art is not None
    assert "User settings container" in str(art.get("content") or "")
