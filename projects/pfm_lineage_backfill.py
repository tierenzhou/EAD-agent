"""
Backfill committed PFM snapshots for finished runs in chronological order.

Keeps template generation (V9, V10, V11, …) consistent when older runs were never
materialized from agent-delivered files.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

from .models import ExecutionStatus, ProjectExecute
from .pfm_delivery import compute_delivery_stamp, delivery_changed
from .pfm_tree import baseline_generation_from_snap, snapshot_generation, snapshot_revision

logger = logging.getLogger(__name__)

_TERMINAL = frozenset(
    {
        ExecutionStatus.COMPLETED.value,
        ExecutionStatus.FAILED.value,
        ExecutionStatus.CANCELLED.value,
        "error",
    }
)


def _run_status_value(ex: ProjectExecute) -> str:
    st = ex.status.value if hasattr(ex.status, "value") else ex.status
    return str(st or "").strip().lower()


def _sorted_template_runs(store: Any, template_id: str) -> List[ProjectExecute]:
    runs = list(store.list_executions(template_id=template_id) or [])
    return sorted(runs, key=lambda e: int(e.start_time or 0))


def target_generation_for_run(store: Any, ex: ProjectExecute) -> int:
    """V = run number on this template."""
    return store.get_pfm_generation_for_run(ex)


def run_needs_pfm_backfill(store: Any, ex: ProjectExecute) -> Tuple[bool, str]:
    """
    Return (needs_backfill, reason).
    """
    eid = str(ex.id or "").strip()
    if _run_status_value(ex) not in _TERMINAL:
        return False, "not_finished"

    stamp = compute_delivery_stamp(eid)
    if not str(stamp.get("fingerprint") or "").strip():
        return False, "no_delivery_files"

    prev = store._get_pfm_tree_snapshot_raw(eid)
    target_gen = target_generation_for_run(store, ex)

    if not isinstance(prev, dict):
        return True, "no_snapshot"

    current_gen = snapshot_generation(prev) or baseline_generation_from_snap(prev)
    source_run = str(prev.get("source_run_id") or eid).strip()
    if current_gen <= 0:
        return True, "missing_generation"
    if current_gen != target_gen:
        return True, f"wrong_generation:{current_gen}!={target_gen}"
    if source_run != eid:
        return True, "wrong_source_run"
    if delivery_changed(prev, stamp):
        return True, "delivery_changed"
    return False, "ok"


def list_pfm_lineage_gaps(store: Any, template_id: str) -> Dict[str, Any]:
    """Runs that need backfill, oldest first."""
    tid = str(template_id or "").strip()
    pending: List[Dict[str, Any]] = []
    for ex in _sorted_template_runs(store, tid):
        needs, reason = run_needs_pfm_backfill(store, ex)
        if not needs:
            continue
        pending.append(
            {
                "execution_id": ex.id,
                "name": ex.name,
                "start_time": ex.start_time,
                "status": _run_status_value(ex),
                "reason": reason,
                "target_generation": target_generation_for_run(store, ex),
            }
        )
    return {
        "template_id": tid,
        "count": len(pending),
        "runs": pending,
    }


def backfill_template_pfm_lineage(
    store: Any,
    template_id: str,
    *,
    promote_template_canonical: bool = True,
    force: bool = True,
) -> Dict[str, Any]:
    """
    Materialize PFM snapshots for all pending finished runs, oldest first.
    """
    from .pfm_materialize import materialize_operator_pfm_snapshot

    tid = str(template_id or "").strip()
    if not tid:
        return {"ok": False, "code": "invalid_template", "message": "template_id required"}

    gaps = list_pfm_lineage_gaps(store, tid)
    runs_meta = gaps.get("runs") or []
    if not runs_meta:
        return {
            "ok": True,
            "code": "nothing_to_do",
            "message": "All finished runs with agent files already have consistent PFM versions.",
            "processed": 0,
            "results": [],
        }

    results: List[Dict[str, Any]] = []
    processed = 0
    last_eid = ""

    for item in runs_meta:
        eid = str(item.get("execution_id") or "").strip()
        if not eid:
            continue
        ex = store.get_execution(eid)
        if not ex:
            results.append({"execution_id": eid, "ok": False, "code": "not_found"})
            continue

        stamp = compute_delivery_stamp(eid)
        fp = str(stamp.get("fingerprint") or "")
        prev = store._get_pfm_tree_snapshot_raw(eid)
        target_gen = target_generation_for_run(store, ex)

        if not fp:
            results.append(
                {
                    "execution_id": eid,
                    "ok": True,
                    "code": "skipped_no_files",
                    "target_generation": target_gen,
                }
            )
            continue

        needs, reason = run_needs_pfm_backfill(store, ex)
        if not needs and not force:
            results.append(
                {
                    "execution_id": eid,
                    "ok": True,
                    "code": "skipped_ok",
                    "pfm_generation": snapshot_generation(prev),
                    "pfm_revision": snapshot_revision(prev),
                }
            )
            continue
        if not needs:
            results.append(
                {
                    "execution_id": eid,
                    "ok": True,
                    "code": "skipped_ok",
                    "target_generation": target_gen,
                }
            )
            continue

        current_gen = snapshot_generation(prev) if isinstance(prev, dict) else 0
        if current_gen == target_gen and delivery_changed(prev, stamp):
            revision = snapshot_revision(prev) + 1
        else:
            revision = 1

        logger.info(
            "[pfm_backfill] %s (%s) → V%s Rev %s",
            eid[:8],
            reason,
            target_gen,
            revision,
        )

        out = materialize_operator_pfm_snapshot(
            store,
            eid,
            promote_template_canonical=False,
            generation=target_gen,
            revision=revision,
            source_fingerprint=fp,
            source_delivery_mtime_ms=int(stamp.get("delivery_mtime_ms") or 0),
            source_delivery_files=list(stamp.get("files") or []),
            source_run_id=eid,
            delivery_refresh=True,
        )
        row = {
            "execution_id": eid,
            "ok": bool(out.get("ok")),
            "code": out.get("code"),
            "message": out.get("message"),
            "reason": reason,
            "pfm_generation": out.get("pfm_generation") or target_gen,
            "pfm_revision": out.get("pfm_revision") or revision,
        }
        results.append(row)
        if out.get("ok") and out.get("code") == "materialized":
            processed += 1
            last_eid = eid

    promoted = False
    if promote_template_canonical and last_eid:
        tmpl = store.promote_template_canonical_pfm(
            tid,
            last_eid,
            source="lineage_backfill",
            rationale="Operator backfill — latest chronological run set as template canonical.",
            require_eligible=False,
        )
        promoted = tmpl is not None

    failed = [r for r in results if not r.get("ok")]
    return {
        "ok": len(failed) == 0,
        "code": "backfill_complete" if not failed else "backfill_partial",
        "message": f"Backfilled {processed} run(s); {len(failed)} failed.",
        "processed": processed,
        "failed": len(failed),
        "promoted_template_canonical": promoted,
        "results": results,
    }
