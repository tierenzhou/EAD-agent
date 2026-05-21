"""
Bulk backfill for finished runs: delivery refresh, FMR node-report sync, hierarchy reconcile.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from .models import ExecutionStatus, ProjectExecute
from .pfm_delivery import ingest_workspace_delivery_exports
from .pfm_fmr_parse import sync_node_reports_from_canonical_fmr
from .pfm_hierarchy import persist_reconciled_snapshot_hierarchy
from .pfm_refresh import try_refresh_pfm_from_delivery

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


def is_terminal_execution(ex: ProjectExecute) -> bool:
    return _run_status_value(ex) in _TERMINAL


def list_terminal_executions(
    store: Any,
    *,
    cursor: str = "",
    limit: int = 50,
) -> tuple[List[ProjectExecute], str, bool]:
    """
    Return terminal executions oldest-first, optionally starting after ``cursor`` (execution id).

    Returns (batch, next_cursor, has_more).
    """
    runs = [
        ex
        for ex in store.list_executions() or []
        if isinstance(ex, ProjectExecute) and is_terminal_execution(ex)
    ]
    runs.sort(key=lambda e: (int(e.start_time or 0), str(e.id or "")))
    start_idx = 0
    if cursor:
        cursor_val = str(cursor).strip()
        for i, ex in enumerate(runs):
            if str(ex.id or "") == cursor_val:
                start_idx = i + 1
                break
    lim = max(1, min(int(limit or 50), 500))
    batch = runs[start_idx : start_idx + lim]
    next_cursor = str(batch[-1].id or "") if batch else ""
    has_more = (start_idx + len(batch)) < len(runs)
    return batch, next_cursor, has_more


def backfill_execution_pfm(
    store: Any,
    execution_id: str,
    *,
    dry_run: bool = False,
    promote_template_canonical: bool = False,
) -> Dict[str, Any]:
    """
    Refresh delivery, sync FMR node reports, and reconcile shallow trees for one execution.
    """
    eid = str(execution_id or "").strip()
    execution = store.get_execution(eid)
    if not execution:
        return {"ok": False, "code": "not_found", "execution_id": eid}

    if not is_terminal_execution(execution):
        return {
            "ok": True,
            "code": "skipped_not_terminal",
            "execution_id": eid,
        }

    row: Dict[str, Any] = {
        "execution_id": eid,
        "ok": True,
        "refresh_code": None,
        "fmr_node_reports_synced": 0,
        "hierarchy_code": None,
        "nodes_added": 0,
    }

    if dry_run:
        from .pfm_delivery import compute_canonical_delivery_stamp, filter_canonical_delivery_files
        from .pfm_hierarchy import build_report_hierarchy_index, needs_hierarchy_reconcile

        ingest_workspace_delivery_exports(store, eid)
        stamp = compute_canonical_delivery_stamp(eid)
        files = filter_canonical_delivery_files(stamp.get("files") or [])
        snap = store.get_committed_pfm_tree(eid) or {}
        artifacts = store.list_execution_pfm_artifacts(eid) or []
        key_to_title, direct_by_parent = build_report_hierarchy_index(artifacts)
        flat = list(snap.get("flat_nodes") or [])
        would_reconcile = needs_hierarchy_reconcile(flat, key_to_title, direct_by_parent)
        row.update(
            {
                "code": "dry_run",
                "has_delivery": bool(str(stamp.get("fingerprint") or "").strip()),
                "delivery_files": [f.get("name") for f in files],
                "has_committed_snapshot": bool(flat),
                "would_reconcile_hierarchy": would_reconcile,
                "artifact_report_keys": len(key_to_title),
            }
        )
        return row

    ingest_workspace_delivery_exports(store, eid)
    refresh_out = try_refresh_pfm_from_delivery(
        store,
        eid,
        promote_template_canonical=promote_template_canonical,
    )
    row["refresh_code"] = str(refresh_out.get("code") or "")
    row["fmr_node_reports_synced"] = int(refresh_out.get("fmr_node_reports_synced") or 0)
    if refresh_out.get("ok") is not True:
        row["ok"] = False
        row["code"] = row["refresh_code"] or "refresh_failed"
        row["message"] = str(refresh_out.get("message") or "")
        return row

    synced = int(sync_node_reports_from_canonical_fmr(store, eid))
    row["fmr_node_reports_synced"] = max(row["fmr_node_reports_synced"], synced)

    hierarchy_out = persist_reconciled_snapshot_hierarchy(store, eid, dry_run=False)
    row["hierarchy_code"] = str(hierarchy_out.get("code") or "")
    row["nodes_added"] = int(hierarchy_out.get("nodes_added") or 0)
    if hierarchy_out.get("ok") is False:
        row["ok"] = False
        row["code"] = row["hierarchy_code"] or "hierarchy_failed"
        row["message"] = str(hierarchy_out.get("message") or "")
        return row

    if row["nodes_added"] > 0:
        row["code"] = "tree_upgraded"
    elif row["refresh_code"] in ("materialized", "no_changes", "fmr_reports_synced_no_pfm"):
        row["code"] = row["refresh_code"]
    else:
        row["code"] = row["refresh_code"] or row["hierarchy_code"] or "ok"

    return row


def run_pfm_backfill(
    store: Any,
    *,
    dry_run: bool = False,
    limit: int = 50,
    cursor: str = "",
    promote_template_canonical: bool = False,
) -> Dict[str, Any]:
    """
    Process a batch of finished runs. Use returned ``next_cursor`` for resumable batches.
    """
    batch, next_cursor, has_more = list_terminal_executions(store, cursor=cursor, limit=limit)
    stats: Dict[str, Any] = {
        "ok": True,
        "dry_run": bool(dry_run),
        "scanned": 0,
        "refreshed": 0,
        "reports_synced": 0,
        "tree_upgraded": 0,
        "skipped": 0,
        "failed": 0,
        "cursor": str(cursor or ""),
        "next_cursor": next_cursor,
        "has_more": has_more,
        "results": [],
    }

    for ex in batch:
        stats["scanned"] += 1
        result = backfill_execution_pfm(
            store,
            str(ex.id or ""),
            dry_run=dry_run,
            promote_template_canonical=promote_template_canonical,
        )
        stats["results"].append(result)
        code = str(result.get("code") or "")
        if result.get("ok") is not True:
            stats["failed"] += 1
            stats["ok"] = False
            continue
        if code in ("skipped_not_terminal", "no_delivery_files", "awaiting_delivery", "no_snapshot"):
            stats["skipped"] += 1
            continue
        if code == "dry_run":
            if result.get("would_reconcile_hierarchy"):
                stats["tree_upgraded"] += 1
            continue
        stats["reports_synced"] += int(result.get("fmr_node_reports_synced") or 0)
        if code in ("materialized", "no_changes", "fmr_reports_synced_no_pfm"):
            stats["refreshed"] += 1
        if code == "tree_upgraded" or int(result.get("nodes_added") or 0) > 0:
            stats["tree_upgraded"] += 1

    stats["code"] = "backfill_complete" if stats["ok"] else "backfill_partial"
    stats["message"] = (
        f"Scanned {stats['scanned']} run(s): "
        f"{stats['refreshed']} refreshed, "
        f"{stats['tree_upgraded']} hierarchy upgrades, "
        f"{stats['skipped']} skipped, "
        f"{stats['failed']} failed."
    )
    return stats
