"""Post-login inject uses skills only; Phase-1 fallback when skills missing."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from projects.executor import ProjectExecutor
from projects.models import ExecutionStatus, ProjectExecute, ProjectTemplate
from projects.store import ProjectStore
from tests.projects.test_pfm_canonical import _commit_minimal_tree


@pytest.fixture()
def inject_store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> ProjectStore:
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("EAD_REPORT_BASE_DIR", str(tmp_path / "reports"))
    monkeypatch.setenv("EAD_PFM_AGENT_AUTHORED", "true")
    store = ProjectStore(db_path=tmp_path / "projects.db")
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
    store.create_execution(
        ProjectExecute(
            id="exec-new",
            linked_template_id="tpl-1",
            name="New run",
            status=ExecutionStatus.RUNNING,
            run_session_key="eadproj-exec-exec-new",
        )
    )
    return store


def _execution_with_session(store: ProjectStore) -> ProjectExecute:
    ex = store.get_execution("exec-new")
    assert ex is not None
    return ex


def test_post_login_skips_phase1_when_skills_exist(inject_store: ProjectStore) -> None:
    executor = ProjectExecutor(inject_store, MagicMock())
    ex = _execution_with_session(inject_store)

    with patch.object(executor, "_inject_post_login_skills_once") as mock_skills:
        with patch.object(executor, "_inject_phase1_baseline_once") as mock_phase1:
            executor._inject_post_login_context_once(ex)

    mock_skills.assert_called_once_with(ex)
    mock_phase1.assert_not_called()


def test_post_login_injects_phase1_when_no_skills(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EAD_REPORT_BASE_DIR", str(tmp_path / "reports"))
    store = ProjectStore(db_path=tmp_path / "projects.db")
    store.create_template(ProjectTemplate(id="tpl-2", name="Empty"))
    store.create_execution(
        ProjectExecute(
            id="exec-first",
            linked_template_id="tpl-2",
            name="First",
            status=ExecutionStatus.RUNNING,
            run_session_key="eadproj-exec-exec-first",
        )
    )
    executor = ProjectExecutor(store, MagicMock())
    ex = store.get_execution("exec-first")
    assert ex is not None

    with patch.object(executor, "_inject_post_login_skills_once") as mock_skills:
        with patch.object(executor, "_inject_phase1_baseline_once") as mock_phase1:
            executor._inject_post_login_context_once(ex)

    mock_skills.assert_called_once_with(ex)
    mock_phase1.assert_called_once_with(ex)
