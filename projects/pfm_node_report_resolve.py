"""
Resolve per-node EAD reports for the operator UI with layered fallbacks.

Priority when hydrating a run:
1. Existing rich ``node_ead_report`` rows in the DB
2. Canonical agent-delivered ``*.FMR`` on disk (parsed into DB)
3. ``progress_log`` / chat-captured node report replies (paired by node key)
4. Rewrite ``.FMR`` on disk from DB when new content was recovered (keeps delivery in sync)
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

from .pfm_fmr_parse import (
    ensure_node_reports_from_agent_delivery,
    is_fmr_placeholder_report,
    is_materialized_stub_markdown,
)
from .pfm_node_report_content import (
    backfill_node_reports_from_progress_log,
    normalize_node_report_markdown,
)

logger = logging.getLogger(__name__)


def _artifact_has_rich_content(artifact: Optional[Dict[str, Any]]) -> bool:
    if not isinstance(artifact, dict):
        return False
    content = normalize_node_report_markdown(
        str(artifact.get("content") or artifact.get("markdown") or "")
    )
    if not content or is_fmr_placeholder_report(content) or is_materialized_stub_markdown(content):
        return False
    return len(content) >= 80


def hydrate_node_reports_from_sources(store: Any, execution_id: str) -> Dict[str, int]:
    """
    Import missing node reports into the DB from FMR and progress_log.

    Returns counts: ``fmr_synced``, ``progress_log_synced``.
    """
    eid = str(execution_id or "").strip()
    if not eid:
        return {"fmr_synced": 0, "progress_log_synced": 0}
    fmr_synced = 0
    if store.has_committed_pfm_tree(eid):
        fmr_synced = int(ensure_node_reports_from_agent_delivery(store, eid) or 0)
    progress_log_synced = int(backfill_node_reports_from_progress_log(store, eid) or 0)
    if fmr_synced or progress_log_synced:
        try:
            from .pfm_deliverables import rewrite_canonical_fmr_from_store

            rewrite_canonical_fmr_from_store(store, eid)
        except Exception as exc:
            logger.debug("[pfm_node_report_resolve] FMR rewrite skipped for %s: %s", eid, exc)
    return {
        "fmr_synced": fmr_synced,
        "progress_log_synced": progress_log_synced,
    }


def _fuzzy_match_node_report(
    store: Any,
    execution_id: str,
    node_key: str,
) -> Optional[Dict[str, Any]]:
    req_parts = [p for p in node_key.split("/") if p]
    req_leaf = (req_parts[-1] if req_parts else "").lower()
    req_parent = (req_parts[-2] if len(req_parts) >= 2 else "").lower()
    candidates: List[Tuple[int, Dict[str, Any]]] = []
    for row in store.list_execution_pfm_artifacts(execution_id) or []:
        if str(row.get("artifact_type") or "").strip() != "node_ead_report":
            continue
        nk = str(row.get("node_key") or "").strip()
        if not nk:
            continue
        parts = [p for p in nk.split("/") if p]
        if not parts:
            continue
        leaf = parts[-1].lower()
        score = 0
        if leaf == req_leaf:
            score += 5
        if req_leaf and leaf.endswith(req_leaf):
            score += 3
        if req_parent and any(req_parent in p.lower() for p in parts):
            score += 2
        if req_leaf and any(req_leaf in p.lower() for p in parts):
            score += 1
        if score > 0 and _artifact_has_rich_content(row):
            candidates.append((score, row))
    if not candidates:
        return None
    candidates.sort(
        key=lambda it: (it[0], int((it[1] or {}).get("created_at") or 0)),
        reverse=True,
    )
    return candidates[0][1]


def resolve_node_ead_report_artifact(
    store: Any,
    execution_id: str,
    node_key: str,
    *,
    hydrate: bool = True,
) -> Optional[Dict[str, Any]]:
    """
  Return the best ``node_ead_report`` artifact dict for ``node_key``, optionally
  importing from FMR / progress_log first.
    """
    from .pfm_tree import node_report_artifact_key

    eid = str(execution_id or "").strip()
    nk = str(node_key or "").strip()
    if not eid or not nk:
        return None

    if hydrate:
        hydrate_node_reports_from_sources(store, eid)

    artifact_key = node_report_artifact_key(nk)
    artifact = store.get_execution_pfm_artifact(eid, artifact_key)
    if _artifact_has_rich_content(artifact):
        return artifact

    matched = _fuzzy_match_node_report(store, eid, nk)
    if matched:
        return matched

    return artifact if isinstance(artifact, dict) else None
