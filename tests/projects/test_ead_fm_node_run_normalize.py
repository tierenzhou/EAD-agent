"""EadFmNodeRun tolerates camelCase and agent status strings in persisted JSON."""

from __future__ import annotations

from projects.models import EadFmNodeRun, TestCaseStepRunStatus


def test_camel_case_node_key_and_verified_status():
    raw = {
        "nodeKey": "auth-access",
        "title": "Authentication",
        "status": "verified",
        "type": "feature-area",
    }
    n = EadFmNodeRun.model_validate(raw)
    assert n.node_key == "auth-access"
    assert n.node_id == "auth-access"
    assert n.status == TestCaseStepRunStatus.SUCCESS


def test_snake_case_unchanged():
    n = EadFmNodeRun.model_validate(
        {
            "node_id": "nid-1",
            "node_key": "k1",
            "status": "Success",
        }
    )
    assert n.node_id == "nid-1"
    assert n.status == TestCaseStepRunStatus.SUCCESS
