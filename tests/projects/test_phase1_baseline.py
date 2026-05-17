"""Phase-1 fallback baseline must stay minimal when skills are unavailable."""

from __future__ import annotations

from pathlib import Path

import pytest

from projects.api import _build_phase1_canonical_baseline_message
from projects.models import ExecutionStatus, ProjectExecute, ProjectTemplate
from projects.store import ProjectStore


@pytest.fixture()
def baseline_store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> ProjectStore:
    monkeypatch.setenv("EAD_REPORT_BASE_DIR", str(tmp_path / "reports"))
    monkeypatch.setenv("EAD_PFM_AGENT_AUTHORED", "true")
    store = ProjectStore(db_path=tmp_path / "projects.db")
    store.create_template(ProjectTemplate(id="tpl-1", name="T"))
    store.create_execution(
        ProjectExecute(
            id="exec-new",
            linked_template_id="tpl-1",
            name="New",
            status=ExecutionStatus.PENDING,
            inherited_from_execution_id="exec-old",
        )
    )
    store.upsert_execution_pfm_artifact(
        "exec-new",
        "report-login",
        "node_ead_report",
        {
            "title": "Login Portal",
            "content": "# Login Portal\n\nLong body " + ("x" * 2000),
        },
    )
    store.upsert_execution_pfm_artifact(
        "exec-new",
        "mindmap-1",
        "pfm_mindmap",
        {"nodes": [{"node_key": "n1", "title": "Node 1"}]},
    )
    return store


def test_phase1_fallback_is_short_without_report_bodies(baseline_store: ProjectStore) -> None:
    ex = baseline_store.get_execution("exec-new")
    assert ex is not None
    body = _build_phase1_canonical_baseline_message(baseline_store, ex)
    assert body is not None
    assert len(body) < 500
    assert "# Login Portal" not in body
    assert "Login Portal" not in body
    assert "PFM nodes available:" in body
    assert "Node reports available: 1" in body
    assert "exec-old" in body
