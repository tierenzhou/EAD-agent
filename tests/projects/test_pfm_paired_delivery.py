"""Paired .pfm + .FMR delivery must belong to the execution (filename contains run id)."""

from __future__ import annotations

from pathlib import Path

import pytest

from projects.pfm_delivery import (
    execution_has_paired_canonical_delivery,
    has_paired_canonical_delivery_on_disk,
    ingest_workspace_delivery_exports,
)
from projects.models import ExecutionStatus, ProjectExecute, ProjectTemplate
from projects.store import ProjectStore


@pytest.fixture()
def delivery_store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> ProjectStore:
    monkeypatch.setenv("HOME", str(tmp_path))
    reports = tmp_path / "reports"
    reports.mkdir()
    monkeypatch.setenv("EAD_REPORT_BASE_DIR", str(reports))
    monkeypatch.setenv("EAD_REPORT_DIR", str(reports))
    store = ProjectStore(db_path=tmp_path / "projects.db")
    store.create_template(ProjectTemplate(id="tpl-1", name="P1 Test"))
    store.create_execution(
        ProjectExecute(
            id="aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee",
            linked_template_id="tpl-1",
            name="P1 Test",
            status=ExecutionStatus.RUNNING,
        )
    )
    store.create_execution(
        ProjectExecute(
            id="11111111-2222-3333-4444-555555555555",
            linked_template_id="tpl-1",
            name="P1 Test",
            status=ExecutionStatus.RUNNING,
        )
    )
    return store


def test_paired_delivery_requires_execution_id_in_filename(
    delivery_store: ProjectStore, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_a = "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
    run_b = "11111111-2222-3333-4444-555555555555"
    reports = tmp_path / "reports"
    dir_a = reports / run_a
    dir_b = reports / run_b
    dir_a.mkdir(parents=True)
    dir_b.mkdir(parents=True)

    # Prior-run delivery files copied into run B's folder must not count as run B delivery.
    (dir_b / "p1-test-aaaaaaaa.pfm").write_text("{}", encoding="utf-8")
    (dir_b / "p1-test-aaaaaaaa.FMR").write_text("# old", encoding="utf-8")

    assert has_paired_canonical_delivery_on_disk(run_b) is False
    assert execution_has_paired_canonical_delivery(delivery_store, run_b) is False

    (dir_b / "p1-test-11111111.pfm").write_text("{}", encoding="utf-8")
    (dir_b / "p1-test-11111111.FMR").write_text("# current", encoding="utf-8")

    assert has_paired_canonical_delivery_on_disk(run_b) is True
    assert execution_has_paired_canonical_delivery(delivery_store, run_b) is True
    assert has_paired_canonical_delivery_on_disk(run_a) is False


def test_ingest_does_not_copy_unscoped_slug_canonical_files(
    delivery_store: ProjectStore, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_id = "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
    slug_dir = Path(__file__).resolve().parents[2] / "projects" / "p1-test-reports"
    slug_dir.mkdir(parents=True, exist_ok=True)
    generic_pfm = slug_dir / "generic-slug.pfm"
    generic_fmr = slug_dir / "generic-slug.FMR"
    generic_pfm.write_text("{}", encoding="utf-8")
    generic_fmr.write_text("# generic", encoding="utf-8")

    out = ingest_workspace_delivery_exports(delivery_store, run_id)
    assert out.get("ok") is True
    dest = tmp_path / "reports" / run_id
    assert not (dest / "generic-slug.pfm").is_file()
    assert not (dest / "generic-slug.FMR").is_file()
