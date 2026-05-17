"""Template skill resolution for post-login chat injection."""

from __future__ import annotations

from pathlib import Path

import pytest

from projects.models import ExecutionStatus, ProjectExecute, ProjectTemplate
from projects.pfm_skills import (
    PFM_POST_LOGIN_SKILLS_MARKER,
    build_post_login_skills_inject_message,
    resolve_template_skills_for_injection,
    write_local_skill,
)
from projects.store import ProjectStore
from tests.projects.test_pfm_canonical import _commit_minimal_tree


@pytest.fixture()
def skills_store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> ProjectStore:
    skills_root = tmp_path / "skills" / "pfm"
    monkeypatch.setenv("HOME", str(tmp_path))
    db_path = tmp_path / "projects.db"
    monkeypatch.setenv("EAD_REPORT_BASE_DIR", str(tmp_path / "reports"))
    monkeypatch.setenv("EAD_PFM_AGENT_AUTHORED", "true")
    store = ProjectStore(db_path=db_path)
    store.create_template(ProjectTemplate(id="tpl-1", name="T"))
    store.create_execution(
        ProjectExecute(
            id="exec-skills",
            linked_template_id="tpl-1",
            name="Skills run",
            status=ExecutionStatus.COMPLETED,
            start_time=2000,
            contributes_to_learning=True,
            valid_for_data_reporting_training=True,
        )
    )
    _commit_minimal_tree(store, "exec-skills")
    from projects.pfm_skills import create_execution_pfm_skills

    create_execution_pfm_skills(store, "exec-skills")
    return store


def test_resolve_template_skills(skills_store: ProjectStore) -> None:
    resolved = resolve_template_skills_for_injection(skills_store, "tpl-1")
    assert resolved is not None
    assert resolved["source_execution_id"] == "exec-skills"
    assert "kloud-pfm-map" in resolved["map_path"]
    assert "kloud-pfm-reports" in resolved["reports_path"]


def test_build_post_login_message(skills_store: ProjectStore) -> None:
    body = build_post_login_skills_inject_message(skills_store, "tpl-1")
    assert body is not None
    assert PFM_POST_LOGIN_SKILLS_MARKER in body
    assert "kloud-pfm-map" in body
    assert "kloud-pfm-reports" in body
    assert "Read tool" in body
    assert "SKILL.md" in body
    assert "does **not** complete this run" in body
    assert "commit_pfm_snapshot" not in body


def test_resolve_skips_invalid_run(skills_store: ProjectStore) -> None:
    skills_store.create_execution(
        ProjectExecute(
            id="exec-bad",
            linked_template_id="tpl-1",
            name="Bad",
            status=ExecutionStatus.COMPLETED,
            start_time=3000,
            contributes_to_learning=False,
        )
    )
    write_local_skill("kloud-pfm-map", "# map\n")
    write_local_skill("kloud-pfm-reports", "# reports\n")
    resolved = resolve_template_skills_for_injection(skills_store, "tpl-1")
    assert resolved is not None
    assert resolved["source_execution_id"] == "exec-skills"
