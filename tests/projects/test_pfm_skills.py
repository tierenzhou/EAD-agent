"""Tests for PFM skill export (disk + DB artifacts)."""

from __future__ import annotations

from pathlib import Path

import pytest

from projects.models import ExecutionStatus, ProjectExecute, ProjectTemplate
from projects.pfm_skills import (
    SKILL_MAP_NAME,
    create_execution_pfm_skills,
    list_execution_pfm_skills,
    local_skill_md_path,
)
from projects.store import ProjectStore
from tests.projects.test_pfm_tree import _good_payload


@pytest.fixture()
def skill_store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> ProjectStore:
    db_path = tmp_path / "projects.db"
    skills_dir = tmp_path / "skills"
    monkeypatch.setenv("EAD_REPORT_BASE_DIR", str(tmp_path / "reports"))
    monkeypatch.setenv("HOME", str(tmp_path))
    store = ProjectStore(db_path=db_path)
    store.create_template(ProjectTemplate(id="tpl-s", name="Demo App"))
    store.create_execution(
        ProjectExecute(
            id="exec-s",
            linked_template_id="tpl-s",
            name="Run",
            status=ExecutionStatus.COMPLETED,
        )
    )
    p = _good_payload(1)
    p["execution_id"] = "exec-s"
    from projects.pfm_tree import validate_and_normalize_snapshot

    snap, _, reps = validate_and_normalize_snapshot(p)
    store.replace_execution_pfm_tree("exec-s", snapshot=snap, node_reports=reps)
    return store


def test_create_skills_writes_disk_and_db(skill_store: ProjectStore, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    result = create_execution_pfm_skills(skill_store, "exec-s")
    assert result.get("ok") is True
    assert local_skill_md_path(SKILL_MAP_NAME).is_file()
    rows = list_execution_pfm_skills(skill_store, "exec-s")
    map_row = next(r for r in rows if r["name"] == SKILL_MAP_NAME)
    assert map_row["local_exists"] is True
    assert map_row["in_database"] is True
