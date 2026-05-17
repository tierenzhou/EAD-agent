"""Template learning source picks latest completed run with skill files."""

from __future__ import annotations

from pathlib import Path

import pytest

from projects.models import ExecutionStatus, ProjectExecute, ProjectTemplate
from projects.pfm_skills import (
    create_execution_pfm_skills,
    resolve_template_learning_source_execution,
)
from projects.store import ProjectStore
from tests.projects.test_pfm_canonical import _commit_minimal_tree


@pytest.fixture()
def learning_store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> ProjectStore:
    monkeypatch.setenv("HOME", str(tmp_path))
    db_path = tmp_path / "projects.db"
    monkeypatch.setenv("EAD_REPORT_BASE_DIR", str(tmp_path / "reports"))
    monkeypatch.setenv("EAD_PFM_AGENT_AUTHORED", "true")
    store = ProjectStore(db_path=db_path)
    store.create_template(ProjectTemplate(id="tpl-1", name="T"))
    return store


def _add_completed_run(
    store: ProjectStore,
    execution_id: str,
    *,
    start_time: int,
    with_skills: bool,
) -> None:
    store.create_execution(
        ProjectExecute(
            id=execution_id,
            linked_template_id="tpl-1",
            name=execution_id,
            status=ExecutionStatus.COMPLETED,
            start_time=start_time,
            contributes_to_learning=True,
            valid_for_data_reporting_training=True,
        )
    )
    _commit_minimal_tree(store, execution_id)
    if with_skills:
        create_execution_pfm_skills(store, execution_id)


def test_resolve_prefers_latest_completed_with_skills(learning_store: ProjectStore) -> None:
    _add_completed_run(learning_store, "exec-old-skills", start_time=1000, with_skills=True)
    _add_completed_run(learning_store, "exec-new-no-skills", start_time=2000, with_skills=False)

    source = resolve_template_learning_source_execution(learning_store, "tpl-1")
    assert source is not None
    assert source.id == "exec-old-skills"

    inherit = learning_store.resolve_pfm_inheritance_source("tpl-1")
    assert inherit is not None
    assert inherit.id == "exec-old-skills"


def test_resolve_excludes_current_execution(learning_store: ProjectStore) -> None:
    _add_completed_run(learning_store, "exec-only", start_time=1000, with_skills=True)
    excluded = resolve_template_learning_source_execution(
        learning_store, "tpl-1", exclude_execution_id="exec-only"
    )
    assert excluded is None
