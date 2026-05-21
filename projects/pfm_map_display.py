"""
Operator PFM mindmap display rule (single source of truth).

For any run:
1. If it has its own **DeliveredPFMMapReport** (paired ``.pfm`` + ``.FMR`` for this run)
   → show **Current** EAD Feature Map (read this run).
2. Else find the **most recent finished** run on the same template with a **valid**
   EAD feature map (DeliveredPFMMapReport + committed ``pfm-tree.json``)
   → show **Previous** (read that run).
"""

from __future__ import annotations

from typing import Any, Dict, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .models import ProjectExecute
    from .store import ProjectStore


def _is_terminal_status(status: object) -> bool:
    from .store import _is_terminal_execution_status

    return _is_terminal_execution_status(status)


def _is_active_status(status: object) -> bool:
    from .store import _is_active_execution_status

    return _is_active_execution_status(status)


def has_delivered_pfm_map_report(
    store: "ProjectStore",
    execution_id: str,
    *,
    check_disk: bool = True,
) -> bool:
    """This run delivered its own paired canonical ``.pfm`` + ``.FMR`` files."""
    from .pfm_delivery import execution_has_paired_canonical_delivery

    return execution_has_paired_canonical_delivery(
        store, execution_id, check_disk=check_disk
    )


def has_valid_ead_feature_map(store: "ProjectStore", execution_id: str) -> bool:
    """
    A finished run's map is valid when it has DeliveredPFMMapReport and a committed tree.
    """
    eid = str(execution_id or "").strip()
    if not eid:
        return False
    if not store.has_committed_pfm_tree(eid):
        return False
    return has_delivered_pfm_map_report(store, eid, check_disk=False) or has_delivered_pfm_map_report(
        store, eid, check_disk=True
    )


def find_most_recent_finished_valid_map_run(
    store: "ProjectStore",
    template_id: str,
    *,
    exclude_execution_id: Optional[str] = None,
    before_run_number: Optional[int] = None,
) -> str:
    """
    Highest run-number finished execution on ``template_id`` with a valid EAD feature map.
    """
    tid = str(template_id or "").strip()
    if not tid:
        return ""
    exclude = str(exclude_execution_id or "").strip()
    best_id = ""
    best_run_number = -1
    for ex in store.list_executions(template_id=tid):
        eid = str(ex.id or "").strip()
        if not eid or eid == exclude:
            continue
        if not _is_terminal_status(ex.status):
            continue
        from .pfm_run_number import get_run_number

        run_number = int(get_run_number(store, ex) or 0)
        if before_run_number is not None and run_number >= int(before_run_number):
            continue
        if not has_valid_ead_feature_map(store, eid):
            continue
        if run_number > best_run_number:
            best_run_number = run_number
            best_id = eid
    return best_id


def resolve_pfm_map_display(store: "ProjectStore", ex: "ProjectExecute") -> Dict[str, Any]:
    """
    Resolve operator-facing PFM map pointers and labels for one execution.
    """
    from .models import ExecutionStatus
    from .pfm_run_number import get_run_number
    from .pfm_tree import snapshot_finalized, snapshot_revision

    eid = str(ex.id or "").strip()
    template_id = str(ex.linked_template_id or "").strip()
    run_number = int(get_run_number(store, ex) or 0)
    check_disk = _is_active_status(ex.status)

    has_own_delivery = has_delivered_pfm_map_report(store, eid, check_disk=check_disk)

    if has_own_delivery:
        read_id = eid
        display_mode = "current"
    else:
        read_id = find_most_recent_finished_valid_map_run(
            store,
            template_id,
            exclude_execution_id=eid,
            before_run_number=run_number if run_number > 0 else None,
        )
        display_mode = "previous"

    own_snap = store.get_committed_pfm_tree(eid) if store.has_committed_pfm_tree(eid) else None
    read_snap = store.get_committed_pfm_tree(read_id) if read_id else None

    if display_mode == "current":
        revision = snapshot_revision(own_snap) if own_snap else 0
        source_run_number = run_number
        finalized = snapshot_finalized(own_snap) if own_snap else False
    else:
        revision = snapshot_revision(read_snap) if read_snap else 0
        source_run_number = int(get_run_number(store, store.get_execution(read_id)) or 0) if read_id else 0
        finalized = False

    status = (
        ex.status.value if isinstance(ex.status, ExecutionStatus) else str(ex.status or "")
    ).lower()
    hint = str(ex.executor_hint or "").lower()
    finalizing_hint = any(
        token in hint for token in ("reporting", "finalizing", "deliverable", "ai finish")
    )

    if display_mode == "previous":
        phase = "baseline_preview"
    elif status in ("completed", "failed", "cancelled", "error"):
        phase = "final"
    elif status in ("running", "pending") and finalizing_hint:
        phase = "finalizing"
    elif status in ("running", "pending"):
        phase = "evolving"
    else:
        phase = "final"

    baseline_id = read_id if display_mode == "previous" else ""

    return {
        "run_number": run_number,
        "has_delivered_pfm_map_report": has_own_delivery,
        "pfm_map_display_mode": display_mode,
        "pfm_baseline_execution_id": baseline_id or None,
        "pfm_tree_read_execution_id": read_id or None,
        "pfm_baseline_tree_version": source_run_number if display_mode == "previous" else 0,
        "pfm_source_run_number": source_run_number,
        "pfm_generation_version": run_number,
        "pfm_revision": revision,
        "pfm_lineage_version": run_number,
        "pfm_step_version": revision,
        "pfm_finalized": finalized,
        "pfm_has_committed_snapshot": has_own_delivery,
        "pfm_snapshot_phase": phase,
    }
