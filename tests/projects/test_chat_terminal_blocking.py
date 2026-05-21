from __future__ import annotations

import asyncio
import json

from projects.chat_api import ChatControlHandlers
from projects.models import ExecutionStatus, ProjectExecute


class _Request:
    def __init__(self, body: dict | None = None, query: dict | None = None):
        self._body = body or {}
        self.query = query or {}

    async def json(self) -> dict:
        return self._body


class _FakeSessionDB:
    def __init__(self, session_id: str = "sess-existing"):
        self._session_id = session_id
        self.created = False

    def get_session_by_title(self, session_key: str):
        if session_key == "eadproj-exec-exec-terminal":
            return {"id": self._session_id}
        return None

    def create_session(self, **kwargs):
        self.created = True
        return self._session_id

    def set_session_title(self, session_id: str, title: str) -> None:
        return None


def _mk_execution(execution_id: str, status: ExecutionStatus) -> ProjectExecute:
    return ProjectExecute(
        id=execution_id,
        linked_template_id="tpl-1",
        name=f"Run {execution_id}",
        status=status,
    )


def test_send_is_blocked_for_terminal_run() -> None:
    execution = _mk_execution("exec-terminal", ExecutionStatus.COMPLETED)
    store = type("Store", (), {"get_execution": lambda self, _id: execution})()

    handlers = ChatControlHandlers(session_db=None, agent_pool=None, project_store=store)
    req = _Request(
        body={
            "session_key": "eadproj-exec-exec-terminal",
            "content": "continue",
            "deliver": True,
        }
    )

    response = asyncio.run(handlers.handle_send(req))
    payload = json.loads(response.body.decode("utf-8"))

    assert response.status == 409
    assert payload["error"]["code"] == "run_closed"


def test_create_session_returns_existing_for_terminal_run() -> None:
    execution = _mk_execution("exec-terminal", ExecutionStatus.COMPLETED)
    store = type("Store", (), {"get_execution": lambda self, _id: execution})()
    db = _FakeSessionDB()

    handlers = ChatControlHandlers(session_db=db, agent_pool=None, project_store=store)
    req = _Request(body={"session_key": "eadproj-exec-exec-terminal"})

    response = asyncio.run(handlers.handle_create_session(req))
    payload = json.loads(response.body.decode("utf-8"))

    assert response.status == 200
    assert payload["created"] is False
    assert payload["session_id"] == "sess-existing"
    assert db.created is False


def test_create_session_is_blocked_for_terminal_run_without_history() -> None:
    execution = _mk_execution("exec-failed", ExecutionStatus.FAILED)
    store = type("Store", (), {"get_execution": lambda self, _id: execution})()
    db = _FakeSessionDB()

    handlers = ChatControlHandlers(session_db=db, agent_pool=None, project_store=store)
    req = _Request(body={"session_key": "eadproj-exec-exec-missing"})

    response = asyncio.run(handlers.handle_create_session(req))
    payload = json.loads(response.body.decode("utf-8"))

    assert response.status == 409
    assert payload["error"]["code"] == "run_closed"
    assert db.created is False
