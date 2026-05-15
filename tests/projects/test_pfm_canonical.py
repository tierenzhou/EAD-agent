"""Tests for canonical PFM pointer, baseline resolution, and judge helpers."""

from __future__ import annotations

from pathlib import Path

import pytest

from projects.models import ExecutionStatus, ProjectExecute, ProjectTemplate
from projects.pfm_canonical_judge import _coerce_judge_payload, _extract_json_object
from projects.pfm_tree import validate_and_normalize_snapshot
from projects.store import ProjectStore


@pytest.fixture()
def isolated_store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> ProjectStore:
    db_path = tmp_path / "projects.db"
    monkeypatch.setenv("EAD_REPORT_BASE_DIR", str(tmp_path / "reports"))
    monkeypatch.setenv("EAD_PFM_AGENT_AUTHORED", "true")
    store = ProjectStore(db_path=db_path)
    store.create_template(ProjectTemplate(id="tpl-1", name="T"))
    store.create_execution(
        ProjectExecute(
            id="exec-old",
            linked_template_id="tpl-1",
            name="Old",
            status=ExecutionStatus.COMPLETED,
        )
    )
    store.create_execution(
        ProjectExecute(
            id="exec-new",
            linked_template_id="tpl-1",
            name="New",
            status=ExecutionStatus.RUNNING,
            inherited_from_execution_id="exec-old",
        )
    )
    return store


def _commit_minimal_tree(store: ProjectStore, execution_id: str) -> None:
    from tests.projects.test_pfm_tree import _good_payload

    p = _good_payload(1)
    p["execution_id"] = execution_id
    snap, _, reps = validate_and_normalize_snapshot(p)
    store.replace_execution_pfm_tree(execution_id, snapshot=snap, node_reports=reps)


def test_has_committed_pfm_tree(isolated_store: ProjectStore):
    assert isolated_store.has_committed_pfm_tree("exec-old") is False
    _commit_minimal_tree(isolated_store, "exec-old")
    assert isolated_store.has_committed_pfm_tree("exec-old") is True


def test_resolve_baseline_uses_template_canonical(isolated_store: ProjectStore):
    _commit_minimal_tree(isolated_store, "exec-old")
    isolated_store.update_template(
        "tpl-1",
        canonical_pfm_execution_id="exec-old",
        canonical_pfm_promoted_by="operator",
    )
    ex_new = isolated_store.get_execution("exec-new")
    assert ex_new is not None
    assert isolated_store.resolve_pfm_baseline_execution_id(ex_new) == "exec-old"


def test_resolve_baseline_falls_back_to_inherited(isolated_store: ProjectStore):
    _commit_minimal_tree(isolated_store, "exec-old")
    ex_new = isolated_store.get_execution("exec-new")
    assert ex_new is not None
    assert isolated_store.resolve_pfm_baseline_execution_id(ex_new) == "exec-old"


def test_operator_promote_updates_template(isolated_store: ProjectStore):
    _commit_minimal_tree(isolated_store, "exec-new")
    out = isolated_store.promote_template_canonical_pfm(
        "tpl-1",
        "exec-new",
        source="operator",
        rationale="test",
        require_eligible=False,
    )
    assert out is not None
    assert out.canonical_pfm_execution_id == "exec-new"
    ex = isolated_store.get_execution("exec-new")
    assert ex is not None
    assert ex.pfm_canonical_promotion_applied is True


def test_invalidate_clears_canonical_when_same(isolated_store: ProjectStore):
    _commit_minimal_tree(isolated_store, "exec-new")
    isolated_store.promote_template_canonical_pfm(
        "tpl-1", "exec-new", source="operator", rationale="x", require_eligible=False
    )
    isolated_store.invalidate_execution_learning("exec-new", reason="bad")
    tpl = isolated_store.get_template("tpl-1")
    assert tpl is not None
    assert not (tpl.canonical_pfm_execution_id or "").strip()


def test_extract_json_object():
    parsed = _extract_json_object(
        'prefix {"replace_canonical": true, "confidence": 0.9, "rationale": "x"}'
    )
    assert parsed is not None
    assert parsed.get("replace_canonical") is True


def test_coerce_judge_payload():
    rep, conf, rat = _coerce_judge_payload({"replace_canonical": True, "confidence": 0.8, "rationale": "ok"})
    assert rep is True and conf == 0.8 and rat == "ok"
