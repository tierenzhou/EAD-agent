"""API routing for PFM backfill endpoint."""

from __future__ import annotations

import inspect

from projects.api import ProjectHandlers


def test_backfill_route_registered() -> None:
    source = inspect.getsource(ProjectHandlers.register_routes)
    assert "/v1/projects/pfm/backfill" in source
    assert "handle_post_pfm_backfill" in source
