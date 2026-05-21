"""Node report hydration: FMR + progress_log fallbacks into DB."""

from __future__ import annotations

from pathlib import Path

import pytest

from projects.models import ExecutionStatus, ProjectExecute, ProjectTemplate
from projects.pfm_fmr_parse import parse_fmr_node_reports
from projects.pfm_node_report_content import normalize_node_report_markdown
from projects.pfm_node_report_resolve import (
    hydrate_node_reports_from_sources,
    resolve_node_ead_report_artifact,
)
from projects.store import ProjectStore


@pytest.fixture()
def isolated_store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> ProjectStore:
    db_path = tmp_path / "projects.db"
    monkeypatch.setenv("EAD_REPORT_BASE_DIR", str(tmp_path / "reports"))
    store = ProjectStore(db_path=db_path)
    store.create_template(ProjectTemplate(id="tpl-fmr", name="T"))
    return store


def _good_report_md() -> str:
    return normalize_node_report_markdown(
        """
Node Summary:
Purpose: Authentication area for login and session.
In-scope behaviors:
- User can sign in with valid credentials

Features:
Feature F-001: Login
Description: Standard login form

Test Case TC-001: Happy path login
Objective: Verify login works
Preconditions: User exists
Test Data: NONE

Steps:
Step 1: Open login page
   - Expected Result: Form visible
   - Evidence Image: NONE

Explore and improve (for future exploration):
What we documented well:
- Login form layout

Gaps and open questions:
- MFA not tested

Recommended next exploration:
- Password reset flow

How to improve this report next time:
- Add negative test cases
"""
    )


def test_parse_fmr_and_hydrate_into_db(isolated_store: ProjectStore, tmp_path) -> None:
    from projects.pfm_artifacts import reports_root_dir

    ex = isolated_store.create_execution(
        ProjectExecute(
            id="exec-fmr",
            linked_template_id="tpl-fmr",
            name="Run FMR",
            status=ExecutionStatus.COMPLETED,
        )
    )
    eid = ex.id
    isolated_store.replace_execution_pfm_tree(
        eid,
        snapshot={
            "version": 1,
            "generated_at": 1,
            "flat_nodes": [
                {
                    "node_key": "auth",
                    "title": "Authentication",
                    "parent_node_key": None,
                    "level": 1,
                    "status": "No Run",
                }
            ],
            "roots": [],
        },
        node_reports=[],
    )

    md = _good_report_md()
    fmr_text = "\n".join(
        [
            "# PFM Full Node EAD Report: T",
            "",
            "## Per-Node EAD Reports",
            "",
            "### Node 1: Authentication",
            "- Node Key: `auth`",
            "- Status: `No Run`",
            "",
            "Description: Auth module",
            "",
            "#### Node EAD Report",
            md,
        ]
    )
    out_dir = reports_root_dir() / eid
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{eid}.FMR").write_text(fmr_text, encoding="utf-8")

    parsed = parse_fmr_node_reports(fmr_text)
    assert "auth" in parsed

    counts = hydrate_node_reports_from_sources(isolated_store, eid)
    assert counts["fmr_synced"] >= 1

    art = resolve_node_ead_report_artifact(
        isolated_store, eid, "auth", hydrate=False
    )
    assert art is not None
    assert "Authentication" in str(art.get("content") or "")
