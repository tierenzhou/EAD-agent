"""Tests for bulk PFM backfill across finished runs."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from projects.api import ProjectHandlers
from projects.models import ExecutionStatus, ProjectExecute, ProjectTemplate
from projects.pfm_backfill import backfill_execution_pfm, list_terminal_executions, run_pfm_backfill
from projects.store import ProjectStore
from tests.projects.test_pfm_canonical import _commit_minimal_tree


class _Req:
    def __init__(self, body: dict | None = None) -> None:
        self.can_read_body = True
        self._body = body or {}

    async def json(self) -> dict:
        return self._body


@pytest.fixture()
def backfill_store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> ProjectStore:
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("EAD_REPORT_DIR", str(tmp_path / "reports"))
    monkeypatch.setenv("EAD_PFM_AGENT_AUTHORED", "true")
    store = ProjectStore(db_path=tmp_path / "projects.db")
    store.create_template(ProjectTemplate(id="tpl-b", name="B"))
    for idx, eid in enumerate(("exec-a", "exec-b"), start=1):
        store.create_execution(
            ProjectExecute(
                id=eid,
                linked_template_id="tpl-b",
                name=f"Run {idx}",
                status=ExecutionStatus.COMPLETED,
                start_time=idx * 1000,
            )
        )
        _commit_minimal_tree(store, eid)
    store.create_execution(
        ProjectExecute(
            id="exec-running",
            linked_template_id="tpl-b",
            name="Active",
            status=ExecutionStatus.RUNNING,
            start_time=9999,
        )
    )
    return store


def test_list_terminal_executions_excludes_running(backfill_store: ProjectStore) -> None:
    batch, cursor, has_more = list_terminal_executions(backfill_store, limit=10)
    ids = {ex.id for ex in batch}
    assert "exec-running" not in ids
    assert "exec-a" in ids
    assert "exec-b" in ids
    assert cursor in ("exec-a", "exec-b")
    assert has_more is False


def test_run_pfm_backfill_dry_run(backfill_store: ProjectStore) -> None:
    out = run_pfm_backfill(backfill_store, dry_run=True, limit=10)
    assert out.get("ok") is True
    assert out.get("dry_run") is True
    assert int(out.get("scanned") or 0) == 2
    results = out.get("results") or []
    assert len(results) == 2
    assert all(r.get("code") == "dry_run" for r in results)


def test_backfill_execution_skips_running(backfill_store: ProjectStore) -> None:
    out = backfill_execution_pfm(backfill_store, "exec-running")
    assert out.get("code") == "skipped_not_terminal"


def test_backfill_api_endpoint(backfill_store: ProjectStore) -> None:
    handlers = ProjectHandlers(store=backfill_store, executor=MagicMock())
    resp = asyncio.run(
        handlers.handle_post_pfm_backfill(
            _Req({"dry_run": True, "limit": 5}),
        )
    )
    assert resp.status == 200
    payload = json.loads(resp.text)
    assert payload["ok"] is True
    assert payload["dryRun"] is True
    assert payload["scanned"] == 2


def test_backfill_resumable_cursor(backfill_store: ProjectStore) -> None:
    first = run_pfm_backfill(backfill_store, dry_run=True, limit=1, cursor="")
    assert first.get("has_more") is True
    next_cursor = first.get("next_cursor")
    second = run_pfm_backfill(
        backfill_store,
        dry_run=True,
        limit=1,
        cursor=str(next_cursor or ""),
    )
    assert int(second.get("scanned") or 0) == 1
    first_id = (first.get("results") or [{}])[0].get("execution_id")
    second_id = (second.get("results") or [{}])[0].get("execution_id")
    assert first_id != second_id
