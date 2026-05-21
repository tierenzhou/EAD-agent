from __future__ import annotations

from pathlib import Path

import pytest

from projects.models import ExecutionStatus, ProjectExecute, ProjectTemplate
from projects.pfm_run_number import ensure_template_run_numbers
from projects.store import ProjectStore


@pytest.fixture()
def run_number_store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> ProjectStore:
    monkeypatch.setenv("EAD_REPORT_BASE_DIR", str(tmp_path / "reports"))
    store = ProjectStore(db_path=tmp_path / "projects.db")
    store.create_template(ProjectTemplate(id="tpl-run-number", name="Template"))
    return store


def _execution(execution_id: str, start_ms: int, run_number: int | None = None) -> ProjectExecute:
    return ProjectExecute(
        id=execution_id,
        linked_template_id="tpl-run-number",
        name=execution_id,
        status=ExecutionStatus.COMPLETED,
        start_time=start_ms,
        run_number=run_number,
    )


def test_create_execution_allocates_after_highest_persisted_run_number(
    run_number_store: ProjectStore,
) -> None:
    run_number_store.create_execution(_execution("run-66", 1000, run_number=66))

    created = run_number_store.create_execution(
        ProjectExecute(
            id="run-new",
            linked_template_id="tpl-run-number",
            name="New run",
            status=ExecutionStatus.PENDING,
            start_time=2000,
        )
    )

    assert created.run_number == 67


def test_ensure_template_run_numbers_does_not_renumber_existing_values(
    run_number_store: ProjectStore,
) -> None:
    run_number_store.create_execution(_execution("run-66", 1000, run_number=66))
    run_number_store.create_execution(_execution("legacy-missing", 500, run_number=None))

    ensure_template_run_numbers(run_number_store, "tpl-run-number")

    assert run_number_store.get_execution("run-66").run_number == 66
    assert run_number_store.get_execution("legacy-missing").run_number == 67
