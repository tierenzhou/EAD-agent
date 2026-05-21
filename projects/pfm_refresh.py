"""
Operator refresh: rebuild DB PFM when agent delivery changed.

  V = run number (per template). Rev increments within the same run.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Tuple

from .pfm_delivery import (
    compute_canonical_delivery_stamp,
    delivery_changed,
    describe_canonical_delivery_locations,
    filter_canonical_delivery_files,
)
from .pfm_tree import snapshot_generation, snapshot_revision

logger = logging.getLogger(__name__)

# Operator-facing copy when Refresh runs before agent delivery exists.
NO_EAD_EXPLORE_DELIVERY_YET_MESSAGE = (
    "No canonical PFM delivery found on disk (.pfm / .fmr / .FMR) after workspace ingest. "
    "If the agent wrote files elsewhere, copy them into one of the locations below, then retry."
)
PAIRED_DELIVERY_REQUIRED_MESSAGE = (
    "Canonical delivery must include both .pfm and .FMR files for this run. "
    "PFM refresh is skipped until both files are present and .FMR node reports can be parsed."
)


def _reconcile_hierarchy_after_refresh(store: Any, execution_id: str) -> Dict[str, Any]:
    from .pfm_hierarchy import persist_reconciled_snapshot_hierarchy

    return persist_reconciled_snapshot_hierarchy(store, execution_id, dry_run=False)


def _find_other_run_with_new_delivery(store: Any, execution: Any) -> str:
    template_id = str(getattr(execution, "linked_template_id", "") or "").strip()
    if not template_id:
        return ""
    eid = str(getattr(execution, "id", "") or "").strip()
    for other in store.list_executions(template_id=template_id) or []:
        oid = str(getattr(other, "id", "") or "").strip()
        if not oid or oid == eid:
            continue
        stamp = compute_canonical_delivery_stamp(oid)
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
    from .pfm_delivery import ingest_workspace_delivery_exports
    from .pfm_fmr_parse import (
        load_canonical_fmr_reports,
        sync_node_reports_from_canonical_fmr,
    )
    from .pfm_materialize import materialize_operator_pfm_snapshot

    execution = store.get_execution(execution_id)
    if not execution:
        return {"ok": False, "code": "not_found", "message": "Execution not found"}

    ingest_workspace_delivery_exports(store, execution_id)

    stamp = compute_canonical_delivery_stamp(execution_id)
    canonical_files = filter_canonical_delivery_files(stamp.get("files"))
    has_pfm = any(str(row.get("name") or "").lower().endswith(".pfm") for row in canonical_files)
    has_fmr = any(str(row.get("name") or "").lower().endswith(".fmr") for row in canonical_files)
    fp = str(stamp.get("fingerprint") or "")
    if not fp:
        return {
            "ok": True,
            "code": "no_delivery_files",
            "message": NO_EAD_EXPLORE_DELIVERY_YET_MESSAGE
            + "\n\n"
            + describe_canonical_delivery_locations(execution_id),
        }
    if has_fmr and not has_pfm:
        fmr_reports = load_canonical_fmr_reports(execution_id)
        if fmr_reports:
            synced_reports = sync_node_reports_from_canonical_fmr(store, execution_id)
            return {
                "ok": True,
                "code": "fmr_reports_synced_no_pfm",
                "message": (
                    "Canonical .FMR was parsed and per-node reports were saved to DB. "
                    "Waiting for canonical .pfm to materialize/update the PFM tree."
                ),
                "source_fingerprint": fp,
                "source_delivery_files": canonical_files,
                "fmr_node_reports_synced": int(synced_reports),
            }
    if not (has_pfm and has_fmr):
        return {
            "ok": True,
            "code": "awaiting_delivery",
            "message": PAIRED_DELIVERY_REQUIRED_MESSAGE
            + "\n\n"
            + describe_canonical_delivery_locations(execution_id),
            "source_fingerprint": fp,
            "source_delivery_files": canonical_files,
        }
    fmr_reports = load_canonical_fmr_reports(execution_id)
    if not fmr_reports:
        return {
            "ok": True,
            "code": "awaiting_delivery",
            "message": (
                "Canonical .FMR was found but no per-node EAD reports could be parsed. "
                "Keep both .pfm + .FMR delivery files and retry."
            ),
            "source_fingerprint": fp,
            "source_delivery_files": canonical_files,
        }

    prev_raw = store._get_pfm_tree_snapshot_raw(execution_id)
    if not delivery_changed(prev_raw, stamp):
        synced_reports = sync_node_reports_from_canonical_fmr(store, execution_id)
        hierarchy_out = _reconcile_hierarchy_after_refresh(store, execution_id)
        snap = store.get_committed_pfm_tree(execution_id) or {}
        hint_run_id = _find_other_run_with_new_delivery(store, execution)
        message = "Agent delivery unchanged since last DB commit."
        if hint_run_id:
            short = hint_run_id[:8]
            message += f" New agent files are on run {short}… — select that run and refresh."
        code = "no_changes"
        if hierarchy_out.get("code") == "tree_upgraded":
            code = "tree_upgraded"
            message = (
                f"Committed PFM tree upgraded with {int(hierarchy_out.get('nodes_added') or 0)} "
                "node(s) from saved node-report hierarchy."
            )
        return {
            "ok": True,
            "code": code,
            "message": message,
            "source_fingerprint": fp,
            "pfm_generation": store.get_pfm_generation_for_run(execution),
            "pfm_revision": snapshot_revision(snap),
            "run_number": store.get_pfm_generation_for_run(execution),
            "newer_delivery_execution_id": hint_run_id,
            "fmr_node_reports_synced": int(synced_reports),
            "hierarchy_nodes_added": int(hierarchy_out.get("nodes_added") or 0),
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

    out = materialize_operator_pfm_snapshot(
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
    if out.get("ok") is True:
        out["fmr_node_reports_synced"] = int(
            sync_node_reports_from_canonical_fmr(store, execution_id)
        )
        hierarchy_out = _reconcile_hierarchy_after_refresh(store, execution_id)
        out["hierarchy_nodes_added"] = int(hierarchy_out.get("nodes_added") or 0)
        if hierarchy_out.get("code") == "tree_upgraded":
            out["code"] = "tree_upgraded"
            out["message"] = (
                f"PFM materialized and tree upgraded with "
                f"{out['hierarchy_nodes_added']} node(s) from saved node-report hierarchy."
            )
    return out
