"""
Operator refresh: rebuild DB PFM when agent delivery changed.

  V = run number (per template). Rev increments within the same run.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Tuple

from .pfm_delivery import compute_delivery_stamp, delivery_changed
from .pfm_tree import snapshot_generation, snapshot_revision

logger = logging.getLogger(__name__)


def _find_other_run_with_new_delivery(store: Any, execution: Any) -> str:
    from .pfm_delivery import compute_delivery_stamp, delivery_changed

    template_id = str(getattr(execution, "linked_template_id", "") or "").strip()
    if not template_id:
        return ""
    eid = str(getattr(execution, "id", "") or "").strip()
    for other in store.list_executions(template_id=template_id) or []:
        oid = str(getattr(other, "id", "") or "").strip()
        if not oid or oid == eid:
            continue
        stamp = compute_delivery_stamp(oid)
        if not str(stamp.get("fingerprint") or "").strip():
            continue
        prev = store._get_pfm_tree_snapshot_raw(oid)
        if delivery_changed(prev, stamp):
            return oid
    return ""


def decide_versioning_for_delivery(
    store: Any,
    execution_id: str,
    prev_snap_raw: Optional[Dict[str, Any]],
    _baseline_execution_id: Optional[str],
) -> Tuple[int, int]:
    ex = store.get_execution(execution_id)
    if not ex:
        return 1, 1
    gen = store.get_pfm_generation_for_run(ex)
    prev_rev = snapshot_revision(prev_snap_raw) if isinstance(prev_snap_raw, dict) else 0
    if prev_rev > 0 and snapshot_generation(prev_snap_raw) == gen:
        return gen, prev_rev + 1
    return gen, 1


def try_refresh_pfm_from_delivery(
    store: Any,
    execution_id: str,
    *,
    promote_template_canonical: bool = True,
) -> Dict[str, Any]:
    from .pfm_materialize import materialize_operator_pfm_snapshot

    execution = store.get_execution(execution_id)
    if not execution:
        return {"ok": False, "code": "not_found", "message": "Execution not found"}

    stamp = compute_delivery_stamp(execution_id)
    fp = str(stamp.get("fingerprint") or "")
    if not fp:
        return {
            "ok": True,
            "code": "no_delivery_files",
            "message": "No agent-delivered PFM files on disk for this run.",
        }

    prev_raw = store._get_pfm_tree_snapshot_raw(execution_id)
    if not delivery_changed(prev_raw, stamp):
        snap = store.get_committed_pfm_tree(execution_id) or {}
        hint_run_id = _find_other_run_with_new_delivery(store, execution)
        message = "Agent delivery unchanged since last DB commit."
        if hint_run_id:
            short = hint_run_id[:8]
            message += f" New agent files are on run {short}… — select that run and refresh."
        return {
            "ok": True,
            "code": "no_changes",
            "message": message,
            "source_fingerprint": fp,
            "pfm_generation": store.get_pfm_generation_for_run(execution),
            "pfm_revision": snapshot_revision(snap),
            "run_number": store.get_pfm_generation_for_run(execution),
            "newer_delivery_execution_id": hint_run_id,
        }

    generation, revision = decide_versioning_for_delivery(
        store, execution_id, prev_raw, None
    )

    logger.info(
        "[pfm_refresh] Rebuild %s → V%s (run %s) Rev %s",
        execution_id,
        generation,
        generation,
        revision,
    )

    return materialize_operator_pfm_snapshot(
        store,
        execution_id,
        promote_template_canonical=promote_template_canonical,
        generation=generation,
        revision=revision,
        source_fingerprint=fp,
        source_delivery_mtime_ms=int(stamp.get("delivery_mtime_ms") or 0),
        source_delivery_files=list(stamp.get("files") or []),
        source_run_id=execution_id,
        delivery_refresh=True,
    )
