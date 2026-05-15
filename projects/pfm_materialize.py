"""
Operator-triggered persistence of a committed PFM tree when the agent session
cannot (or has not yet) called ``commit_pfm_snapshot``.

Uses execution ``results[]``, ``report_running_step`` payloads, saved ``pfm_mindmap`` /
``pfm_report`` artifacts (and on-disk ``pfm-mindmap.mmd`` / ``pfm-report.md``), then
generic progress text as a last resort.

While a run is **active**, materialize is allowed only when ``results[]`` is still empty
(so operator refresh can load a tree from saved mindmap text without clobbering agent
commits).
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set

from .models import EadFmNodeRun, ExecutionStatus, ProjectExecute
from .pfm_artifacts import (
    _nodes_by_key,
    _partition_mindmap_tree,
    _pfm_node_dict_to_ead_run,
    derive_node_runs_from_progress,
    derive_pfm_nodes_from_report_running_step_progress,
    derive_pfm_nodes_from_saved_mindmap_or_report_text,
    nodes_from_markdown_fenced_mermaid,
    report_file_path,
)
from .pfm_tree import SnapshotValidationError, validate_and_normalize_snapshot

logger = logging.getLogger(__name__)

_DEGENERATE_TITLES = frozenset(
    {"", "success", "failed", "no run", "verified", "pending", "unknown"},
)


def _status_display(node: EadFmNodeRun) -> str:
    s = getattr(node.status, "value", None) if node.status is not None else node.status
    return str(s if s is not None else "No Run").strip()


def _results_titles_are_degenerate(results: List[EadFmNodeRun]) -> bool:
    """True when persisted ``results[]`` look like status placeholders, not human PFM titles."""
    if len(results) < 3:
        return False
    bad = 0
    for n in results:
        t = (n.title or "").strip()
        st = _status_display(n)
        tl = t.lower()
        if tl in _DEGENERATE_TITLES or t == st:
            bad += 1
    return bad * 3 >= len(results) * 2


def _nodes_are_hierarchical(nodes: List[EadFmNodeRun]) -> bool:
    """
    True when node rows clearly encode a multi-level tree.

    Signals:
    - at least one resolvable ``parent_node_key`` edge
    - or levels deeper than 2
    - or path-like node keys with 2+ separators
    """
    if len(nodes) < 2:
        return False
    keys = {(n.node_key or "").strip() for n in nodes if (n.node_key or "").strip()}
    if not keys:
        return False
    for n in nodes:
        p = (n.parent_node_key or "").strip()
        if p and p in keys:
            return True
        if int(n.level or 0) >= 3:
            return True
        nk = (n.node_key or "").strip()
        if nk.count("/") >= 2:
            return True
    return False


def _max_node_depth(nodes: List[EadFmNodeRun]) -> int:
    """Best-effort depth estimate across ``level`` and path-style ``node_key``."""
    best = 0
    for n in nodes:
        try:
            best = max(best, int(n.level or 0))
        except Exception:
            pass
        nk = (n.node_key or "").strip()
        if nk:
            best = max(best, nk.count("/") + 1)
    return best


def _nodes_from_pfm_artifacts_and_disk(store: Any, execution_id: str) -> List[EadFmNodeRun]:
    """Prefer DB ``pfm_mindmap`` ``nodes[]``, else mindmap/report ``content``, else on-disk files."""
    if not store or not execution_id:
        return []

    if hasattr(store, "list_execution_pfm_artifacts"):
        for art in store.list_execution_pfm_artifacts(execution_id) or []:
            if not isinstance(art, dict):
                continue
            at = str(art.get("artifact_type") or "").strip()
            if at != "pfm_mindmap":
                continue
            raw_nodes = art.get("nodes") or []
            if isinstance(raw_nodes, list) and raw_nodes:
                parsed: List[EadFmNodeRun] = []
                for item in raw_nodes:
                    if not isinstance(item, dict):
                        continue
                    try:
                        parsed.append(EadFmNodeRun.model_validate(item))
                    except Exception:
                        r = _pfm_node_dict_to_ead_run(item, title_fallback=str(art.get("title") or ""))
                        if r:
                            parsed.append(r)
                if parsed:
                    return parsed
            content = str(art.get("content") or "").strip()
            if content:
                nodes = derive_pfm_nodes_from_saved_mindmap_or_report_text(content)
                if nodes:
                    return nodes

        report_md = ""
        for art in store.list_execution_pfm_artifacts(execution_id) or []:
            if not isinstance(art, dict):
                continue
            if str(art.get("artifact_type") or "").strip() != "pfm_report":
                continue
            c = str(art.get("content") or "").strip()
            if c:
                report_md = c
                break
        if report_md:
            nodes = nodes_from_markdown_fenced_mermaid(report_md)
            if nodes:
                return nodes
            nodes = derive_pfm_nodes_from_saved_mindmap_or_report_text(report_md)
            if nodes:
                return nodes

    for fname in ("pfm-mindmap.mmd", "pfm-report.md"):
        path = report_file_path(execution_id, fname)
        if not path.is_file():
            continue
        try:
            txt = path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            logger.warning("[pfm_materialize] could not read %s: %s", path, exc)
            continue
        if fname.endswith(".mmd"):
            nodes = derive_pfm_nodes_from_saved_mindmap_or_report_text(txt)
        else:
            nodes = nodes_from_markdown_fenced_mermaid(txt)
            if not nodes:
                nodes = derive_pfm_nodes_from_saved_mindmap_or_report_text(txt)
        if nodes:
            return nodes

    return []


def _collect_nodes_for_materialize(store: Any, execution: ProjectExecute) -> List[EadFmNodeRun]:
    raw_results = list(execution.results or [])
    progress_nodes = derive_pfm_nodes_from_report_running_step_progress(execution.progress_log or [])
    artifact_nodes = _nodes_from_pfm_artifacts_and_disk(store, execution.id)
    if progress_nodes:
        # report_running_step snapshots can be partial (e.g. top-level only); prefer
        # artifact trees when they are clearly deeper.
        if artifact_nodes and _max_node_depth(artifact_nodes) > _max_node_depth(progress_nodes):
            return artifact_nodes
        return progress_nodes
    degen = _results_titles_are_degenerate(raw_results)
    # Prefer richer artifact trees when results[] are shallow/non-hierarchical.
    if artifact_nodes and raw_results:
        if _max_node_depth(artifact_nodes) > _max_node_depth(raw_results):
            return artifact_nodes
        if not _nodes_are_hierarchical(raw_results) and _nodes_are_hierarchical(artifact_nodes):
            return artifact_nodes
    if artifact_nodes and (not raw_results or degen):
        return artifact_nodes
    if raw_results and not degen:
        return raw_results
    if artifact_nodes:
        return artifact_nodes
    if raw_results:
        return raw_results
    return derive_node_runs_from_progress(execution.progress_log or [])


def _node_run_to_tree_dict(
    node: EadFmNodeRun,
    children_map: Dict[str, List[EadFmNodeRun]],
    visiting: Set[str],
) -> Dict[str, Any]:
    nk = (node.node_key or "").strip() or f"node-{abs(hash((node.title, id(node)))) % 10_000_000}"
    if nk in visiting:
        nk = f"{nk}-dup-{abs(hash(id(node))) % 10_000}"
    visiting.add(nk)
    st = getattr(node.status, "value", None) if node.status is not None else node.status
    st_txt = str(st if st is not None else "No Run")
    desc = (node.meta or "").strip() or (node.title or nk) or ""
    out: Dict[str, Any] = {
        "node_key": nk,
        "title": (node.title or nk)[:240],
        "level": max(1, int(node.level or 1)),
        "type": str(node.type or "feature-area"),
        "status": st_txt,
        "description": desc[:1200],
        "children": [],
    }
    for ch in children_map.get(nk, []) or []:
        out["children"].append(_node_run_to_tree_dict(ch, children_map, visiting))
    return out


def _build_roots_payload(nodes: List[EadFmNodeRun]) -> List[Dict[str, Any]]:
    index = _nodes_by_key(nodes)
    roots, children_map = _partition_mindmap_tree(nodes, index)
    visiting: Set[str] = set()
    roots_sorted = sorted(
        roots,
        key=lambda n: (n.level or 0, (n.title or n.node_key or "").lower()),
    )
    for plist in children_map.values():
        plist.sort(key=lambda n: (n.level or 0, (n.title or n.node_key or "").lower()))
    return [_node_run_to_tree_dict(r, children_map, visiting) for r in roots_sorted]


def _all_parents_resolve(nodes: List[EadFmNodeRun]) -> bool:
    """True when every ``parent_node_key`` is empty or points at another node in the list."""
    keys = {(n.node_key or "").strip() for n in nodes if (n.node_key or "").strip()}
    if not keys:
        return False
    for n in nodes:
        nk = (n.node_key or "").strip()
        if not nk:
            return False
        p = (n.parent_node_key or "").strip()
        if p and p not in keys:
            return False
    return True


def _build_roots_payload_explicit_parents(nodes: List[EadFmNodeRun]) -> List[Dict[str, Any]]:
    """
    Build nested ``roots`` JSON using **only** ``parent_node_key`` edges and **document order**
    for sibling ordering. Avoids ``_partition_mindmap_tree``, which can reshape trees that
    already encode parents explicitly (e.g. ASCII materialization).
    """
    order = {(n.node_key or "").strip(): i for i, n in enumerate(nodes) if (n.node_key or "").strip()}
    keys = set(order.keys())
    children_map: Dict[str, List[EadFmNodeRun]] = defaultdict(list)
    placed: Dict[str, Set[str]] = defaultdict(set)
    roots: List[EadFmNodeRun] = []

    for n in nodes:
        nk = (n.node_key or "").strip()
        if not nk:
            continue
        p = (n.parent_node_key or "").strip()
        if not p or p not in keys:
            roots.append(n)
            continue
        if nk in placed[p]:
            continue
        placed[p].add(nk)
        children_map[p].append(n)

    seen_root: Set[str] = set()
    roots_unique: List[EadFmNodeRun] = []
    for r in roots:
        rk = (r.node_key or "").strip()
        if not rk or rk in seen_root:
            continue
        seen_root.add(rk)
        roots_unique.append(r)
    roots_unique.sort(key=lambda x: order.get((x.node_key or "").strip(), 1_000_000))
    for plist in children_map.values():
        plist.sort(key=lambda x: order.get((x.node_key or "").strip(), 1_000_000))

    visiting: Set[str] = set()
    return [_node_run_to_tree_dict(r, dict(children_map), visiting) for r in roots_unique]


def _default_node_markdown(execution: ProjectExecute, node: EadFmNodeRun, node_key: str) -> str:
    title = (node.title or node_key).replace("\n", " ").strip()
    st = getattr(node.status, "value", None) if node.status is not None else node.status
    st_txt = str(st if st is not None else "No Run")
    meta = (node.meta or "").strip()
    return (
        f"# EAD Feature Map — {title}\n\n"
        f"This markdown was **materialized from run data** (results, `report_running_step`, "
        f"and/or saved PFM mindmap/report text) when an operator used **Refresh EAD Feature Map**. "
        f"It is not a live agent-authored `commit_pfm_snapshot`.\n\n"
        f"- **Execution:** `{execution.id}`\n"
        f"- **Node key:** `{node_key}`\n"
        f"- **Status:** `{st_txt}`\n\n"
        + (f"## Notes from run\n\n{meta}\n" if meta else "")
    )


def _collect_tree_keys(root: Dict[str, Any], out: Set[str]) -> None:
    nk = str(root.get("node_key") or "").strip()
    if nk:
        out.add(nk)
    for ch in root.get("children") or []:
        if isinstance(ch, dict):
            _collect_tree_keys(ch, out)


def materialize_operator_pfm_snapshot(
    store: Any,
    execution_id: str,
    *,
    promote_template_canonical: bool = True,
) -> Dict[str, Any]:
    """
    Build and persist a committed PFM tree from existing run state.

    ``store`` is a :class:`projects.store.ProjectStore` (typed as Any to avoid
    circular imports in thin tooling).
    """
    execution = store.get_execution(execution_id)
    if not execution:
        return {"ok": False, "code": "not_found", "message": "Execution not found"}

    is_active = execution.status in (ExecutionStatus.RUNNING, ExecutionStatus.PENDING)
    if is_active and list(execution.results or []):
        return {
            "ok": False,
            "code": "use_agent_path",
            "message": "Run is active with committed results[]; use the live agent snapshot path.",
        }

    nodes = _collect_nodes_for_materialize(store, execution)
    if not nodes:
        return {
            "ok": False,
            "code": "no_nodes",
            "message": "No PFM nodes found in results, progress, saved mindmap/report artifacts, or disk.",
        }

    prev = store.get_committed_pfm_tree(execution_id)
    prev_ver = int(prev.get("version") or 0) if isinstance(prev, dict) else 0
    new_ver = prev_ver + 1 if prev_ver > 0 else 1

    if _all_parents_resolve(nodes):
        roots_payload = _build_roots_payload_explicit_parents(nodes)
    else:
        roots_payload = _build_roots_payload(nodes)
    if not roots_payload:
        return {"ok": False, "code": "empty_tree", "message": "Could not partition nodes into a tree."}

    tree_keys: Set[str] = set()
    for r in roots_payload:
        _collect_tree_keys(r, tree_keys)
    keyed: Dict[str, EadFmNodeRun] = {}
    for n in nodes:
        k = (n.node_key or "").strip()
        if k:
            keyed[k] = n

    node_reports: List[Dict[str, Any]] = []
    for nk in sorted(tree_keys):
        n = keyed.get(nk)
        if n:
            title = (n.title or nk)[:240]
            md = _default_node_markdown(execution, n, nk)
        else:
            title = nk
            md = (
                f"# EAD Feature Map — {nk}\n\n"
                f"Materialized operator snapshot for execution `{execution.id}`.\n"
            )
        node_reports.append({"node_key": nk, "title": title, "markdown": md})

    payload: Dict[str, Any] = {
        "execution_id": execution_id,
        "version": new_ver,
        "generated_at": int(time.time() * 1000),
        "roots": roots_payload,
        "cross_cutting": [],
        "node_reports": node_reports,
    }

    try:
        snap, _, reps = validate_and_normalize_snapshot(
            payload,
            previous_version=prev_ver,
        )
    except SnapshotValidationError as exc:
        logger.warning("[pfm_materialize] validate failed for %s: %s", execution_id, exc)
        return {"ok": False, "code": getattr(exc, "code", "snapshot_invalid"), "message": str(exc)}

    commit = store.replace_execution_pfm_tree(execution_id, snapshot=snap, node_reports=reps)
    if not commit.get("committed"):
        return {"ok": False, "code": "persist_failed", "message": str(commit.get("error") or "persist_failed")}

    promoted = False
    promotion_error: Optional[str] = None
    if promote_template_canonical:
        tmpl = store.promote_template_canonical_pfm(
            execution.linked_template_id,
            execution_id,
            source="operator_refresh",
            rationale="Operator Refresh EAD Feature Map — materialized snapshot set as template canonical.",
            require_eligible=True,
        )
        promoted = tmpl is not None
        if not promoted:
            promotion_error = "canonical_not_promoted_run_may_be_ineligible_for_training"

    return {
        "ok": True,
        "code": "materialized",
        "materialized": True,
        "promoted_template_canonical": promoted,
        "promotion_note": promotion_error,
        **commit,
    }
