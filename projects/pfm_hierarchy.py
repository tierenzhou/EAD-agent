"""
Reconcile committed PFM tree hierarchy with saved node_ead_report artifact keys.

When delivery materialization produced a shallow tree (top-level modules only) but
``.FMR`` / DB artifacts contain deeper per-node reports, this module upgrades
``flat_nodes`` and ``roots`` so drill-down works without render-time synthesis.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional, Set, Tuple

from .pfm_tree import snapshot_finalized

logger = logging.getLogger(__name__)

_MAX_RECONCILE_NODES = 500
_MAX_CHILDREN_PER_PARENT = 15


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", str(value or "").lower()).strip("-")


def build_report_hierarchy_index(
    artifacts: List[Dict[str, Any]],
) -> Tuple[Dict[str, str], Dict[str, Set[str]]]:
    """Return (node_key -> title, parent_key -> set of direct child keys)."""
    key_to_title: Dict[str, str] = {}
    direct_by_parent: Dict[str, Set[str]] = {}
    for art in artifacts or []:
        if str(art.get("artifact_type") or "").strip() != "node_ead_report":
            continue
        nk = str(art.get("node_key") or "").strip()
        if not nk:
            continue
        key_to_title[nk] = str(art.get("title") or nk).strip() or nk
        parts = [p for p in nk.split("/") if p]
        if len(parts) < 2:
            continue
        for idx in range(1, len(parts)):
            parent = "/".join(parts[:idx])
            child = "/".join(parts[: idx + 1])
            direct_by_parent.setdefault(parent, set()).add(child)
    return key_to_title, direct_by_parent


def _flat_max_depth(flat_nodes: List[Dict[str, Any]]) -> int:
    if not flat_nodes:
        return 0
    by_key = {str(n.get("node_key") or "").strip(): n for n in flat_nodes if n.get("node_key")}
    depths: List[int] = []
    for nk in by_key:
        depths.append(len([p for p in nk.split("/") if p]))
    return max(depths) if depths else 0


def _artifact_max_depth(direct_by_parent: Dict[str, Set[str]]) -> int:
    if not direct_by_parent:
        return 0
    depths: List[int] = []
    for key in direct_by_parent.keys():
        for child in direct_by_parent.get(key) or []:
            depths.append(len([p for p in child.split("/") if p]))
    return max(depths) if depths else 0


def needs_hierarchy_reconcile(
    flat_nodes: List[Dict[str, Any]],
    key_to_title: Dict[str, str],
    direct_by_parent: Dict[str, Set[str]],
) -> bool:
    """True when artifact hierarchy is deeper than the committed flat tree."""
    if not flat_nodes or not key_to_title:
        return False
    flat_depth = _flat_max_depth(flat_nodes)
    artifact_depth = _artifact_max_depth(direct_by_parent)
    if artifact_depth > flat_depth:
        return True
    flat_keys = {str(n.get("node_key") or "").strip() for n in flat_nodes if n.get("node_key")}
    for parent in flat_keys:
        if any(str(n.get("parent_node_key") or "").strip() == parent for n in flat_nodes):
            continue
        if _find_report_prefix_for_committed_key(parent, direct_by_parent):
            return True
    return False


def _find_report_prefix_for_committed_key(
    focus_key: str,
    direct_by_parent: Dict[str, Set[str]],
) -> Optional[str]:
    focus_key = str(focus_key or "").strip()
    if not focus_key:
        return None
    if focus_key in direct_by_parent and direct_by_parent.get(focus_key):
        return focus_key
    selected_slug = _slug(focus_key.split("/")[-1])
    candidates: List[str] = []
    if selected_slug:
        for parent in direct_by_parent.keys():
            last = parent.split("/")[-1].lower()
            if last == selected_slug or last.endswith(selected_slug) or selected_slug in last:
                candidates.append(parent)
    if not candidates:
        return None
    return max(candidates, key=lambda p: len(direct_by_parent.get(p) or []))


def _map_report_key_to_committed(
    report_key: str,
    report_prefix: str,
    committed_prefix: str,
) -> Optional[str]:
    report_key = str(report_key or "").strip()
    report_prefix = str(report_prefix or "").strip()
    committed_prefix = str(committed_prefix or "").strip()
    if not report_key.startswith(report_prefix):
        return None
    suffix = report_key[len(report_prefix) :].lstrip("/")
    if not suffix:
        return committed_prefix or None
    return f"{committed_prefix}/{suffix}" if committed_prefix else suffix


def reconcile_flat_nodes_from_node_reports(
    flat_nodes: List[Dict[str, Any]],
    key_to_title: Dict[str, str],
    direct_by_parent: Dict[str, Set[str]],
    *,
    max_nodes: int = _MAX_RECONCILE_NODES,
) -> Tuple[List[Dict[str, Any]], int]:
    """
    Expand ``flat_nodes`` using node-report key hierarchy aligned to committed keys.

    Returns (new_flat_nodes, nodes_added).
    """
    flat = list(flat_nodes or [])
    if not flat or not key_to_title:
        return flat, 0

    existing: Dict[str, Dict[str, Any]] = {}
    for node in flat:
        nk = str(node.get("node_key") or "").strip()
        if nk:
            existing[nk] = dict(node)

    added = 0
    seeds = list(existing.keys())
    queue: List[Tuple[str, str]] = []
    for committed_parent in seeds:
        report_prefix = _find_report_prefix_for_committed_key(committed_parent, direct_by_parent)
        if not report_prefix:
            continue
        for report_child in sorted(direct_by_parent.get(report_prefix) or []):
            mapped = _map_report_key_to_committed(report_child, report_prefix, committed_parent)
            if mapped:
                queue.append((mapped, report_child))

    seen_pairs: Set[Tuple[str, str]] = set()
    while queue and len(existing) < max_nodes:
        committed_key, report_key = queue.pop(0)
        pair = (committed_key, report_key)
        if pair in seen_pairs:
            continue
        seen_pairs.add(pair)

        if committed_key not in existing:
            parent_key = str(committed_key.rsplit("/", 1)[0] if "/" in committed_key else "").strip()
            parent_level = int((existing.get(parent_key) or {}).get("level") or 1)
            seg = committed_key.split("/")[-1]
            title = key_to_title.get(
                report_key,
                seg.replace("-", " ").strip().title(),
            )
            existing[committed_key] = {
                "node_key": committed_key,
                "node_id": committed_key,
                "parent_node_key": parent_key or None,
                "level": parent_level + 1 if parent_key else 1,
                "title": title[:240],
                "type": "feature-area",
                "status": "No Run",
                "description": "Derived from saved node report hierarchy.",
                "cross_cutting": False,
            }
            added += 1

        report_prefix = report_key
        committed_prefix = committed_key
        for report_child in sorted(direct_by_parent.get(report_prefix) or []):
            if len(existing) >= max_nodes:
                break
            mapped = _map_report_key_to_committed(report_child, report_prefix, committed_prefix)
            if mapped and (mapped, report_child) not in seen_pairs:
                queue.append((mapped, report_child))

    if added <= 0:
        return flat, 0
    return list(existing.values()), added


def augment_focus_snapshot_from_node_reports(
    snapshot: Dict[str, Any],
    *,
    read_execution_id: str,
    node_key: Optional[str],
    list_artifacts_fn: Any,
) -> Dict[str, Any]:
    """Render-time fallback: one child level under a focused leaf node."""
    focus_key = str(node_key or "").strip()
    if not focus_key or not isinstance(snapshot, dict):
        return snapshot
    flat = list(snapshot.get("flat_nodes") or [])
    if not flat:
        return snapshot
    if any(str(n.get("parent_node_key") or "").strip() == focus_key for n in flat):
        return snapshot

    artifacts = list_artifacts_fn(read_execution_id) or []
    key_to_title, direct_by_parent = build_report_hierarchy_index(artifacts)
    if not direct_by_parent:
        return snapshot

    report_prefix = _find_report_prefix_for_committed_key(focus_key, direct_by_parent)
    if not report_prefix:
        return snapshot
    direct_children = sorted(direct_by_parent.get(report_prefix) or [])
    if not direct_children:
        return snapshot

    by_key = {str(n.get("node_key") or "").strip(): n for n in flat if n.get("node_key")}
    parent_level = int((by_key.get(focus_key) or {}).get("level") or 1)
    existing_keys = set(by_key.keys())
    synthetic_nodes: List[Dict[str, Any]] = []
    for child in direct_children[:_MAX_CHILDREN_PER_PARENT]:
        mapped = _map_report_key_to_committed(child, report_prefix, focus_key)
        if not mapped or mapped in existing_keys:
            continue
        existing_keys.add(mapped)
        seg = mapped.split("/")[-1]
        synthetic_nodes.append(
            {
                "node_key": mapped,
                "node_id": mapped,
                "parent_node_key": focus_key,
                "level": parent_level + 1,
                "title": key_to_title.get(child, seg.replace("-", " ").strip().title())[:240],
                "type": "feature-area",
                "status": "No Run",
                "description": "Derived from saved node report hierarchy.",
            }
        )
    if not synthetic_nodes:
        return snapshot
    out = dict(snapshot)
    out["flat_nodes"] = flat + synthetic_nodes
    return out


def reconcile_snapshot_hierarchy(
    snapshot: Dict[str, Any],
    artifacts: List[Dict[str, Any]],
) -> Tuple[Dict[str, Any], int]:
    """Return upgraded snapshot and count of nodes added (0 if unchanged)."""
    if not isinstance(snapshot, dict):
        return snapshot, 0
    flat = list(snapshot.get("flat_nodes") or [])
    key_to_title, direct_by_parent = build_report_hierarchy_index(artifacts)
    if not needs_hierarchy_reconcile(flat, key_to_title, direct_by_parent):
        return snapshot, 0
    new_flat, added = reconcile_flat_nodes_from_node_reports(
        flat,
        key_to_title,
        direct_by_parent,
    )
    if added <= 0:
        return snapshot, 0
    out = dict(snapshot)
    out["flat_nodes"] = new_flat
    return out, added


def _rebuild_roots_from_flat(flat_nodes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    from .pfm_materialize import _all_parents_resolve, _build_roots_payload, _build_roots_payload_explicit_parents
    from .pfm_tree import flat_nodes_to_ead_runs

    runs = flat_nodes_to_ead_runs(flat_nodes)
    if _all_parents_resolve(runs):
        return _build_roots_payload_explicit_parents(runs)
    return _build_roots_payload(runs)


def persist_reconciled_snapshot_hierarchy(
    store: Any,
    execution_id: str,
    *,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """
    Upgrade committed tree from node-report hierarchy when the tree is shallower.

    Does not regenerate node reports; only expands ``flat_nodes`` / ``roots``.
    """
    eid = str(execution_id or "").strip()
    if not eid:
        return {"ok": False, "code": "invalid_execution", "nodes_added": 0}

    snap = store.get_committed_pfm_tree(eid)
    if not isinstance(snap, dict) or not snap.get("flat_nodes"):
        return {"ok": True, "code": "no_snapshot", "nodes_added": 0}

    if snapshot_finalized(snap):
        return {"ok": True, "code": "finalized_skip", "nodes_added": 0}

    artifacts = store.list_execution_pfm_artifacts(eid) or []
    upgraded, added = reconcile_snapshot_hierarchy(snap, artifacts)
    if added <= 0:
        return {"ok": True, "code": "no_changes", "nodes_added": 0}

    if dry_run:
        return {
            "ok": True,
            "code": "would_upgrade",
            "nodes_added": added,
            "flat_node_count": len(upgraded.get("flat_nodes") or []),
        }

    upgraded["roots"] = _rebuild_roots_from_flat(list(upgraded.get("flat_nodes") or []))
    commit = store.replace_execution_pfm_tree(eid, snapshot=upgraded, node_reports=[])
    if not commit.get("committed"):
        err = str(commit.get("error") or commit.get("message") or "persist_failed")
        return {"ok": False, "code": str(commit.get("code") or "persist_failed"), "message": err, "nodes_added": 0}

    logger.info(
        "[pfm_hierarchy] Reconciled %s: +%s nodes (flat=%s)",
        eid[:8],
        added,
        len(upgraded.get("flat_nodes") or []),
    )
    return {
        "ok": True,
        "code": "tree_upgraded",
        "nodes_added": added,
        "flat_node_count": len(upgraded.get("flat_nodes") or []),
        "pfm_revision": int(upgraded.get("revision") or upgraded.get("version") or 0),
    }
