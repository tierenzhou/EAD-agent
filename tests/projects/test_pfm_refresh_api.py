"""Tests for default Refresh EAD Feature Map routing in project API."""

from __future__ import annotations

import asyncio
import json
from unittest.mock import MagicMock

from projects.api import ProjectHandlers
from projects.models import ExecutionStatus, ProjectExecute


class _Req:
    def __init__(self, execution_id: str, body_exists: bool = False, body: dict | None = None) -> None:
        self.match_info = {"execution_id": execution_id}
        self.body_exists = body_exists
        self._body = body or {}

    async def json(self) -> dict:
        return self._body


def test_refresh_default_active_run_prefers_agent_commit_path():
    store = MagicMock()
    executor = MagicMock()
    execution = ProjectExecute(
        id="exec-1",
        linked_template_id="tpl-1",
        name="Run 1",
        status=ExecutionStatus.RUNNING,
    )
    store.get_execution.return_value = execution
    executor.request_pfm_snapshot_refresh.return_value = {"ok": True, "code": "queued"}

    handlers = ProjectHandlers(store=store, executor=executor)
    resp = asyncio.run(handlers.handle_post_pfm_request_snapshot(_Req("exec-1")))

    assert resp.status == 200
    payload = json.loads(resp.text)
    assert payload["ok"] is True
    assert payload["code"] == "queued"
    assert payload["executionId"] == "exec-1"
    executor.request_pfm_snapshot_refresh.assert_called_once_with("exec-1")


def test_refresh_default_completed_run_prefers_agent_commit_path_when_session_exists():
    store = MagicMock()
    executor = MagicMock()
    execution = ProjectExecute(
        id="exec-2",
        linked_template_id="tpl-1",
        name="Run 2",
        status=ExecutionStatus.COMPLETED,
        run_session_key="sk-done",
    )
    store.get_execution.return_value = execution
    executor.request_pfm_snapshot_refresh.return_value = {"ok": True, "code": "queued"}

    handlers = ProjectHandlers(store=store, executor=executor)
    resp = asyncio.run(handlers.handle_post_pfm_request_snapshot(_Req("exec-2")))

    assert resp.status == 200
    payload = json.loads(resp.text)
    assert payload["ok"] is True
    assert payload["code"] == "queued"
    assert payload["executionId"] == "exec-2"
    executor.request_pfm_snapshot_refresh.assert_called_once_with("exec-2")

