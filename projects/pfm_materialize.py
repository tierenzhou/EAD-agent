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
import re
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

from .pfm_fmr_parse import (
    is_materialized_stub_markdown as _is_materialized_stub_markdown,
    load_canonical_fmr_reports,
    merge_fmr_reports_into_library,
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


def _report_entry(node_key: str, title: str, markdown: str) -> Dict[str, str]:
    return {
        "node_key": node_key,
        "title": (title or node_key)[:240],
        "markdown": str(markdown or "").strip(),
    }


def _merge_report_into_library(
    library: Dict[str, Dict[str, str]],
    node_key: str,
    title: str,
    markdown: str,
) -> None:
    nk = str(node_key or "").strip()
    md = str(markdown or "").strip()
    if not nk or not md or _is_materialized_stub_markdown(md):
        return
    entry = _report_entry(nk, title, md)
    prev = library.get(nk)
    if not prev or _is_materialized_stub_markdown(prev.get("markdown", "")):
        library[nk] = entry


def _load_node_report_library(
    store: Any,
    execution: ProjectExecute,
    *,
    include_fallback_executions: bool = True,
) -> Dict[str, Dict[str, str]]:
    """
    Rich node_ead_report artifacts for carry-forward during operator materialize/refresh.

    Never prefer materialized stubs over agent-authored markdown. Optionally pulls from
    inherited/baseline runs when this execution has no saved report for a node.
    """
    library: Dict[str, Dict[str, str]] = {}
    execution_ids: List[str] = [str(execution.id or "").strip()]
    if include_fallback_executions:
        inh = str(execution.inherited_from_execution_id or "").strip()
        if inh and inh not in execution_ids:
            execution_ids.append(inh)
        if hasattr(store, "resolve_pfm_baseline_execution_id"):
            baseline = str(store.resolve_pfm_baseline_execution_id(execution) or "").strip()
            if baseline and baseline not in execution_ids:
                execution_ids.append(baseline)

    if not hasattr(store, "list_execution_pfm_artifacts"):
        return library

    for eid in execution_ids:
        if not eid:
            continue
        for art in store.list_execution_pfm_artifacts(eid) or []:
            if not isinstance(art, dict):
                continue
            if str(art.get("artifact_type") or "").strip() != "node_ead_report":
                continue
            nk = str(art.get("node_key") or "").strip()
            md = str(art.get("content") or art.get("markdown") or "").strip()
            title = str(art.get("title") or nk)
            _merge_report_into_library(library, nk, title, md)

    _enrich_library_from_progress_log(execution, library)
    merge_fmr_reports_into_library(
        library,
        load_canonical_fmr_reports(str(execution.id or "")),
    )
    return library


def _enrich_library_from_progress_log(
    execution: ProjectExecute,
    library: Dict[str, Dict[str, str]],
) -> None:
    """Recover agent-authored reports still present in progress_log after stub overwrite."""
    for entry in execution.progress_log or []:
        text = str(getattr(entry, "text", None) or "").strip()
        if not text or _is_materialized_stub_markdown(text):
            continue
        if not (
            "[Node-Report-Reply-To:" in text
            or (
                "Node Summary:" in text
                and "Features:" in text
                and "Test Case TC-" in text
            )
        ):
            continue
        m = re.search(r"\[Node-Report-Reply-To:\s*([^\]]+)\]", text, re.IGNORECASE)
        nk = m.group(1).strip() if m else ""
        if not nk:
            continue
        title = nk
        for raw_line in text.splitlines():
            line = raw_line.strip()
            if line.startswith("Purpose:"):
                title = line.replace("Purpose:", "", 1).strip()[:120] or nk
                break
        _merge_report_into_library(library, nk, title, text)


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


def _stamp_delivery_metadata(
    snap: Dict[str, Any],
    *,
    execution_id: str,
    source_fingerprint: str = "",
    source_delivery_mtime_ms: int = 0,
    source_delivery_files: Optional[List[Dict[str, Any]]] = None,
    source_run_id: str = "",
) -> Dict[str, Any]:
    out = dict(snap)
    eid = str(execution_id or "").strip()
    if eid:
        out["source_run_id"] = str(source_run_id or eid).strip() or eid
    fp = str(source_fingerprint or "").strip()
    if fp:
        out["source_fingerprint"] = fp
    if source_delivery_mtime_ms > 0:
        out["source_delivery_mtime_ms"] = int(source_delivery_mtime_ms)
    if source_delivery_files:
        out["source_delivery_files"] = list(source_delivery_files)
    return out


def materialize_operator_pfm_snapshot(
    store: Any,
    execution_id: str,
    *,
    promote_template_canonical: bool = True,
    generation: Optional[int] = None,
    revision: Optional[int] = None,
    source_fingerprint: str = "",
    source_delivery_mtime_ms: int = 0,
    source_delivery_files: Optional[List[Dict[str, Any]]] = None,
    source_run_id: str = "",
    delivery_refresh: bool = False,
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
    if (
        not delivery_refresh
        and is_active
        and list(execution.results or [])
        and store.has_committed_pfm_tree(execution_id)
    ):
        return {
            "ok": False,
            "code": "use_agent_path",
            "message": "Run is active with a committed tree; use commit_pfm_snapshot for updates.",
        }

    nodes = _collect_nodes_for_materialize(store, execution)
    if not nodes:
        return {
            "ok": False,
            "code": "no_nodes",
            "message": "No PFM nodes found in results, progress, saved mindmap/report artifacts, or disk.",
        }

    prev = store.get_committed_pfm_tree(execution_id)
    if generation is not None and revision is not None:
        gen, rev = int(generation), int(revision)
    else:
        gen, rev = store.compute_next_pfm_versioning(execution, prev)

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

    report_library = _load_node_report_library(store, execution)
    carry_keys: Set[str] = set()
    carry_lib: Dict[str, Dict[str, str]] = {}
    node_reports: List[Dict[str, Any]] = []

    def _library_hit(tree_key: str) -> Optional[Dict[str, str]]:
        if tree_key in report_library:
            return report_library[tree_key]
        tl = tree_key.lower()
        for lib_key, entry in report_library.items():
            if lib_key.lower() == tl:
                return entry
        return None

    for nk in sorted(tree_keys):
        kept = _library_hit(nk)
        if kept and not _is_materialized_stub_markdown(kept.get("markdown", "")):
            carry_keys.add(nk)
            carry_lib[nk] = dict(kept)
            continue
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

    from .pfm_tree import snapshot_revision

    prev_rev = snapshot_revision(prev)

    payload: Dict[str, Any] = {
        "execution_id": execution_id,
        "generated_at": int(time.time() * 1000),
        "roots": roots_payload,
        "cross_cutting": [],
        "node_reports": node_reports,
    }

    try:
        snap, _, reps = validate_and_normalize_snapshot(
            payload,
            generation=gen,
            revision=rev,
            previous_revision=prev_rev,
            report_carry_source_keys=carry_keys,
            report_carry_library=carry_lib,
        )
    except SnapshotValidationError as exc:
        logger.warning("[pfm_materialize] validate failed for %s: %s", execution_id, exc)
        return {"ok": False, "code": getattr(exc, "code", "snapshot_invalid"), "message": str(exc)}

    if not source_fingerprint or not source_delivery_files:
        from .pfm_delivery import compute_delivery_stamp

        stamp = compute_delivery_stamp(execution_id)
        if not source_fingerprint:
            source_fingerprint = str(stamp.get("fingerprint") or "")
        if not source_delivery_mtime_ms:
            source_delivery_mtime_ms = int(stamp.get("delivery_mtime_ms") or 0)
        if not source_delivery_files:
            source_delivery_files = list(stamp.get("files") or [])

    snap = _stamp_delivery_metadata(
        snap,
        execution_id=execution_id,
        source_fingerprint=source_fingerprint,
        source_delivery_mtime_ms=source_delivery_mtime_ms,
        source_delivery_files=source_delivery_files,
        source_run_id=source_run_id,
    )
    from .pfm_delivery import apply_delivery_baseline_to_snapshot

    snap = apply_delivery_baseline_to_snapshot(snap, execution_id)

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


def ensure_committed_pfm_snapshot_after_artifact_delivery(
    store: Any,
    execution_id: str,
    *,
    promote_template_canonical: bool = False,
) -> Dict[str, Any]:
    """
    When the agent publishes PFM files (``publish_pfm_artifacts``) but has not committed
    a DB snapshot, materialize from run/artifact data so the operator UI can show this run's map.
    """
    if store.has_committed_pfm_tree(execution_id):
        from .pfm_delivery import compute_delivery_stamp, delivery_changed
        from .pfm_refresh import try_refresh_pfm_from_delivery

        prev_raw = store._get_pfm_tree_snapshot_raw(execution_id)
        stamp = compute_delivery_stamp(execution_id)
        if delivery_changed(prev_raw, stamp):
            return try_refresh_pfm_from_delivery(
                store,
                execution_id,
                promote_template_canonical=promote_template_canonical,
            )
        snap = store.get_committed_pfm_tree(execution_id) or {}
        return {
            "ok": True,
            "code": "already_committed",
            "committed": True,
            "pfm_tree_version": int(snap.get("version") or 0),
            "skipped_materialize": True,
        }
    result = materialize_operator_pfm_snapshot(
        store,
        execution_id,
        promote_template_canonical=promote_template_canonical,
    )
    if result.get("ok"):
        logger.info(
            "[pfm_materialize] Auto-materialized committed PFM tree for %s after artifact delivery (v%s).",
            execution_id,
            result.get("pfm_tree_version"),
        )
    else:
        logger.warning(
            "[pfm_materialize] Auto-materialize after artifact delivery failed for %s: %s",
            execution_id,
            result.get("message") or result.get("code"),
        )
    return result
