"""Tests for execution interactive-learning eligibility toggles."""

from __future__ import annotations

from pathlib import Path

import pytest

from projects.models import ExecutionStatus, ProjectExecute, ProjectTemplate
from projects.store import ProjectStore


@pytest.fixture()
def learning_store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> ProjectStore:
    db_path = tmp_path / "projects.db"
    monkeypatch.setenv("EAD_REPORT_BASE_DIR", str(tmp_path / "reports"))
    store = ProjectStore(db_path=db_path)
    store.create_template(ProjectTemplate(id="tpl-learn", name="Learn"))
    store.create_execution(
        ProjectExecute(
            id="exec-learn",
            linked_template_id="tpl-learn",
            name="Run",
            status=ExecutionStatus.COMPLETED,
            contributes_to_learning=True,
            valid_for_data_reporting_training=True,
        )
    )
    return store


def test_disable_continuous_learning(learning_store: ProjectStore):
    updated = learning_store.set_execution_continuous_learning(
        "exec-learn",
        enabled=False,
        reason="bad run",
    )
    assert updated is not None
    assert updated.contributes_to_learning is False
    assert updated.valid_for_data_reporting_training is False
    assert "bad run" in (updated.learning_exclusion_reason or "")


def test_reenable_continuous_learning(learning_store: ProjectStore):
    learning_store.set_execution_continuous_learning("exec-learn", enabled=False)
    updated = learning_store.set_execution_continuous_learning("exec-learn", enabled=True)
    assert updated is not None
    assert updated.contributes_to_learning is True
    assert updated.valid_for_data_reporting_training is True
    assert not (updated.learning_exclusion_reason or "").strip()
    assert not (updated.invalid_for_data_reporting_training_reason or "").strip()
