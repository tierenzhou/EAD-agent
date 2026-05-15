"""
Agent-authored PFM tree pipeline.

This module owns the single canonical PFM tree per execution. The agent calls
`commit_pfm_snapshot` (registered in projects.tools) with the whole tree plus
Markdown EAD reports. On incremental commits, `node_reports` may list only
**new, deleted-replaced, or materially updated** nodes; unchanged `node_key`s
that already existed in the previous committed tree reuse the persisted
`node_ead_report` artifact from the last snapshot (see
`validate_and_normalize_snapshot` carry-forward). The first snapshot for an
execution still requires a non-empty report for every node.

Anything that displays the PFM mindmap or per-node EAD reports must read from
the saved tree (artifact_type "pfm_tree") -- never from progress-text fallbacks.

Public API:
    validate_and_normalize_snapshot   - validate payload + flatten the tree
    load_committed_tree               - read latest committed tree (or None)
    render_mermaid_for_scope          - slice the tree and produce Mermaid text
    apply_view_state_to_tree          - repair a saved selection against the latest tree
"""

from __future__ import annotations

import re
import time
from typing import Any, Dict, List, Optional, Set, Tuple

from .models import EadFmNodeRun, ProjectExecute, TestCaseStepRunStatus


PFM_TREE_ARTIFACT_KEY = "pfm-tree.json"
PFM_TREE_ARTIFACT_TYPE = "pfm_tree"
PFM_VIEW_STATE_ARTIFACT_KEY = "pfm-view-state.json"
PFM_VIEW_STATE_ARTIFACT_TYPE = "pfm_view_state"

NODE_KEY_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_\-./]*$")
MAX_TREE_DEPTH = 24
MAX_NODES = 5000


def _slugify_for_artifact_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", str(value or "").lower()).strip("-")


def node_report_artifact_key(node_key: str) -> str:
    """Mirror projects.store.save_node_ead_report_artifact key shape."""
    slug = _slugify_for_artifact_key(node_key) or "node"
    return f"node-ead-report-{slug}.md"


def _coerce_status(raw: Any) -> TestCaseStepRunStatus:
    val = str(raw or "No Run").strip().lower()
    if val == "success":
        return TestCaseStepRunStatus.SUCCESS
    if val == "failed":
        return TestCaseStepRunStatus.FAILED
    return TestCaseStepRunStatus.NO_RUN


class SnapshotValidationError(ValueError):
    """Raised when a commit_pfm_snapshot payload fails validation."""

    def __init__(self, message: str, code: str = "snapshot_invalid"):
        super().__init__(message)
        self.code = code


def committed_tree_node_keys(snapshot: Optional[Dict[str, Any]]) -> Set[str]:
    """Return node_key set from a committed snapshot dict, or empty."""
    if not isinstance(snapshot, dict):
        return set()
    flat = snapshot.get("flat_nodes") or []
    if not isinstance(flat, list):
        return set()
    out: Set[str] = set()
    for row in flat:
        if isinstance(row, dict):
            nk = str(row.get("node_key") or "").strip()
            if nk:
                out.add(nk)
    return out


def collect_node_report_markdown_by_key(store: Any, execution_id: str) -> Dict[str, Dict[str, str]]:
    """Build node_key -> {title, markdown} from persisted node_ead_report rows."""
    out: Dict[str, Dict[str, str]] = {}
    try:
        rows = store.list_execution_pfm_artifacts(execution_id)
    except Exception:
        return out
    for art in rows or []:
        if str(art.get("artifact_type") or "") != "node_ead_report":
            continue
        nk = str(art.get("node_key") or "").strip()
        if not nk:
            continue
        md = str(art.get("content") or "").strip()
        if not md:
            continue
        out[nk] = {
            "title": str(art.get("title") or nk),
            "markdown": md,
        }
    return out


def _walk_tree(
    nodes: List[Dict[str, Any]],
    parent_key: Optional[str],
    depth: int,
    seen_keys: Dict[str, Dict[str, Any]],
    flat: List[Dict[str, Any]],
    is_cross_cutting: bool,
) -> None:
    if depth > MAX_TREE_DEPTH:
        raise SnapshotValidationError(
            f"tree exceeds maximum depth of {MAX_TREE_DEPTH}",
            code="tree_too_deep",
        )
    for raw in nodes or []:
        if not isinstance(raw, dict):
            raise SnapshotValidationError("tree node must be an object", code="node_not_object")
        node_key = str(raw.get("node_key") or "").strip()
        title = str(raw.get("title") or "").strip()
        if not node_key:
            raise SnapshotValidationError("every node requires node_key", code="missing_node_key")
        if not NODE_KEY_RE.match(node_key):
            raise SnapshotValidationError(
                f"node_key {node_key!r} contains disallowed characters; allowed: letters, digits, '_', '-', '.', '/'",
                code="invalid_node_key",
            )
        if not title:
            raise SnapshotValidationError(
                f"node {node_key} missing title",
                code="missing_title",
            )
        if node_key in seen_keys:
            raise SnapshotValidationError(
                f"duplicate node_key {node_key}",
                code="duplicate_node_key",
            )

        explicit_parent = str(raw.get("parent_node_key") or "").strip() or None
        if parent_key is not None:
            if explicit_parent and explicit_parent != parent_key:
                raise SnapshotValidationError(
                    f"node {node_key} has parent_node_key {explicit_parent!r} but appears under {parent_key!r}",
                    code="parent_mismatch",
                )
            if "/" in node_key:
                expected_prefix = f"{parent_key}/"
                if not node_key.startswith(expected_prefix):
                    raise SnapshotValidationError(
                        f"node_key {node_key} must start with parent path {expected_prefix!r}",
                        code="parent_prefix_mismatch",
                    )

        try:
            level_raw = raw.get("level")
            node_level = max(0, int(level_raw)) if level_raw is not None else depth
        except Exception:
            node_level = depth

        flat_node = {
            "node_key": node_key,
            "node_id": str(raw.get("node_id") or node_key),
            "parent_node_key": parent_key,
            "level": node_level,
            "title": title,
            "type": str(raw.get("type") or "feature-area"),
            "status": str(raw.get("status") or "No Run").strip() or "No Run",
            "description": str(raw.get("description") or "").strip(),
            "cross_cutting": is_cross_cutting,
        }
        seen_keys[node_key] = flat_node
        flat.append(flat_node)
        if len(flat) > MAX_NODES:
            raise SnapshotValidationError(
                f"tree exceeds maximum size of {MAX_NODES} nodes",
                code="tree_too_large",
            )

        children = raw.get("children") or []
        if children and not isinstance(children, list):
            raise SnapshotValidationError(
                f"node {node_key} children must be a list",
                code="children_not_list",
            )
        _walk_tree(children, node_key, depth + 1, seen_keys, flat, is_cross_cutting)


def validate_and_normalize_snapshot(
    payload: Dict[str, Any],
    *,
    previous_version: int = 0,
    report_carry_source_keys: Optional[Set[str]] = None,
    report_carry_library: Optional[Dict[str, Dict[str, str]]] = None,
) -> Tuple[Dict[str, Any], List[EadFmNodeRun], List[Dict[str, Any]]]:
    """
    Validate a commit_pfm_snapshot payload.

    Returns (normalized_snapshot, flat_node_runs, report_payloads).

    **Incremental reports (2–5 min checkpoints):** when ``report_carry_source_keys``
    lists node_keys that existed in the *previous* committed tree and
    ``report_carry_library`` maps those keys to prior Markdown (typically loaded
    from ``node_ead_report`` artifacts), any current-tree node whose key is in
    that set may omit ``node_reports[]`` — the prior Markdown is carried forward.
    Keys **new** to this tree (not in ``report_carry_source_keys``) must always
    appear in ``node_reports`` with non-empty markdown. First snapshot
    (no prior tree): pass empty carry sets so every node requires a fresh report.

    Raises SnapshotValidationError with `code` set so the agent gets a stable error.
    """
    if not isinstance(payload, dict):
        raise SnapshotValidationError("payload must be an object", code="payload_not_object")

    try:
        version = int(payload.get("version") or 0)
    except Exception:
        raise SnapshotValidationError("version must be an integer", code="invalid_version")
    if version <= 0:
        raise SnapshotValidationError("version must be a positive integer", code="invalid_version")
    if previous_version and version <= previous_version:
        raise SnapshotValidationError(
            f"version {version} must be strictly greater than current pfm_tree_version {previous_version}",
            code="stale_version",
        )

    generated_at = payload.get("generated_at")
    try:
        generated_at_ms = int(generated_at) if generated_at is not None else int(time.time() * 1000)
    except Exception:
        generated_at_ms = int(time.time() * 1000)

    roots_raw = payload.get("roots") or []
    cross_raw = payload.get("cross_cutting") or []
    if not isinstance(roots_raw, list):
        raise SnapshotValidationError("roots must be a list", code="roots_not_list")
    if not isinstance(cross_raw, list):
        raise SnapshotValidationError("cross_cutting must be a list", code="cross_cutting_not_list")
    if not roots_raw and not cross_raw:
        raise SnapshotValidationError("snapshot must include at least one root", code="empty_tree")

    seen_keys: Dict[str, Dict[str, Any]] = {}
    flat: List[Dict[str, Any]] = []
    _walk_tree(roots_raw, None, 1, seen_keys, flat, False)
    _walk_tree(cross_raw, None, 1, seen_keys, flat, True)

    reports_raw = payload.get("node_reports") or []
    if not isinstance(reports_raw, list):
        raise SnapshotValidationError("node_reports must be a list", code="reports_not_list")

    incoming_by_key: Dict[str, Dict[str, Any]] = {}
    for raw in reports_raw:
        if not isinstance(raw, dict):
            raise SnapshotValidationError("each node report must be an object", code="report_not_object")
        node_key = str(raw.get("node_key") or "").strip()
        markdown = str(raw.get("markdown") or raw.get("content") or "").strip()
        if not node_key:
            raise SnapshotValidationError("node report missing node_key", code="report_missing_node_key")
        if not markdown:
            raise SnapshotValidationError(
                f"node report for {node_key} has empty markdown",
                code="report_empty_markdown",
            )
        if node_key not in seen_keys:
            raise SnapshotValidationError(
                f"node_reports references unknown node_key {node_key!r} (not present in roots/cross_cutting)",
                code="unknown_report_node_key",
            )
        incoming_by_key[node_key] = {
            "node_key": node_key,
            "title": str(raw.get("title") or seen_keys.get(node_key, {}).get("title") or node_key),
            "markdown": markdown,
        }

    carry_keys = report_carry_source_keys or set()
    carry_lib = report_carry_library or {}

    reports_by_key: Dict[str, Dict[str, Any]] = {}
    missing_reports: List[str] = []

    for nk in seen_keys.keys():
        if nk in incoming_by_key:
            reports_by_key[nk] = incoming_by_key[nk]
            continue
        if nk in carry_keys:
            prev = carry_lib.get(nk) or {}
            prev_md = str(prev.get("markdown") or "").strip()
            if prev_md:
                reports_by_key[nk] = {
                    "node_key": nk,
                    "title": str(prev.get("title") or seen_keys[nk].get("title") or nk),
                    "markdown": prev_md,
                }
                continue
        missing_reports.append(nk)

    if missing_reports:
        raise SnapshotValidationError(
            "missing EAD report for nodes (include markdown in node_reports for new/changed "
            "keys; unchanged keys may be omitted only if they existed in the prior committed "
            "tree and still have a saved node_ead_report artifact): "
            + ", ".join(missing_reports[:12]),
            code="missing_node_reports",
        )

    flat_runs: List[EadFmNodeRun] = []
    for node in flat:
        flat_runs.append(
            EadFmNodeRun(
                node_id=node["node_id"],
                node_key=node["node_key"],
                parent_node_key=node["parent_node_key"],
                level=node["level"],
                type=node["type"],
                title=node["title"],
                meta=node["description"],
                status=_coerce_status(node["status"]),
                test_case_runs=[],
            )
        )

    normalized = {
        "version": version,
        "generated_at": generated_at_ms,
        "roots": roots_raw,
        "cross_cutting": cross_raw,
        "flat_nodes": flat,
    }
    return normalized, flat_runs, list(reports_by_key.values())


# ---------------------------------------------------------------------------
# Read helpers
# ---------------------------------------------------------------------------


def load_committed_tree(store, execution: ProjectExecute) -> Optional[Dict[str, Any]]:
    """Return the latest committed tree snapshot, or None if the run never committed."""
    artifact = store.get_execution_pfm_artifact(execution.id, PFM_TREE_ARTIFACT_KEY)
    if not isinstance(artifact, dict):
        return None
    snap = artifact.get("snapshot")
    if isinstance(snap, dict) and snap.get("flat_nodes"):
        return snap
    return None


def flat_nodes_to_ead_runs(flat_nodes: List[Dict[str, Any]]) -> List[EadFmNodeRun]:
    out: List[EadFmNodeRun] = []
    for node in flat_nodes or []:
        if not isinstance(node, dict):
            continue
        try:
            out.append(
                EadFmNodeRun(
                    node_id=str(node.get("node_id") or node.get("node_key") or ""),
                    node_key=str(node.get("node_key") or ""),
                    parent_node_key=node.get("parent_node_key"),
                    level=int(node.get("level") or 0),
                    type=str(node.get("type") or "feature-area"),
                    title=str(node.get("title") or node.get("node_key") or ""),
                    meta=str(node.get("description") or ""),
                    status=_coerce_status(node.get("status")),
                    test_case_runs=[],
                )
            )
        except Exception:
            continue
    return out


# ---------------------------------------------------------------------------
# Scope slicing + Mermaid rendering
# ---------------------------------------------------------------------------


def _index_flat(flat: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    return {str(n.get("node_key") or ""): n for n in flat or [] if n.get("node_key")}


def _children_index(flat: List[Dict[str, Any]]) -> Dict[Optional[str], List[Dict[str, Any]]]:
    children: Dict[Optional[str], List[Dict[str, Any]]] = {}
    for n in flat or []:
        parent = n.get("parent_node_key")
        children.setdefault(parent, []).append(n)
    for plist in children.values():
        plist.sort(key=lambda x: (int(x.get("level") or 0), str(x.get("title") or "").lower()))
    return children


def _path_to_root(node_key: str, by_key: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    path: List[Dict[str, Any]] = []
    current_key: Optional[str] = node_key
    seen: set = set()
    while current_key:
        if current_key in seen:
            break
        seen.add(current_key)
        node = by_key.get(current_key)
        if not node:
            break
        path.append(node)
        current_key = node.get("parent_node_key")
    path.reverse()
    return path


def _mermaid_label(node: Dict[str, Any]) -> str:
    """
    Mindmap node text for Mermaid ``mindmap`` output.

    Use **title only** (no trailing ``[status]``): bracketed suffixes confuse Mermaid
    mindmap rendering and often collapse to misleading single-word boxes.
    """
    raw = str(node.get("title") or node.get("node_key") or "node").strip().replace("\n", " ")
    if len(raw) > 92:
        raw = raw[:89] + "..."
    return raw.replace("(", "[").replace(")", "]").replace('"', "'")


def _emit_subtree(
    lines: List[str],
    node: Dict[str, Any],
    depth: int,
    children: Dict[Optional[str], List[Dict[str, Any]]],
    depth_cap: Optional[int],
    emitted: set,
) -> None:
    key = str(node.get("node_key") or "")
    if not key or key in emitted:
        return
    emitted.add(key)
    indent = " " * max(2 + 2 * depth, 4)
    lines.append(f"{indent}{_mermaid_label(node)}")
    if depth_cap is not None and depth >= depth_cap:
        return
    for child in children.get(key, []) or []:
        _emit_subtree(lines, child, depth + 1, children, depth_cap, emitted)


def render_mermaid_for_scope(
    snapshot: Dict[str, Any],
    execution: ProjectExecute,
    *,
    scope: str = "top",
    node_key: Optional[str] = None,
    depth: Optional[int] = None,
) -> str:
    """Render Mermaid mindmap text for the requested scope, using the committed tree only."""
    flat = list(snapshot.get("flat_nodes") or []) if isinstance(snapshot, dict) else []
    by_key = _index_flat(flat)
    children = _children_index(flat)

    if not flat:
        root_title = (execution.name or execution.id or "PFM Run").replace("\n", " ").strip()
        if len(root_title) > 80:
            root_title = root_title[:77] + "..."
        root_title = root_title.replace("(", "[").replace(")", "]")
        lines = ["mindmap", f"  root(({root_title}))"]
        lines.append("    No PFM tree committed yet")
        return "\n".join(lines) + "\n"

    s = (scope or "top").strip().lower()
    if s == "focus" and node_key and node_key in by_key:
        root_title = _mermaid_label(by_key[node_key])
    else:
        root_title = (execution.name or execution.id or "PFM Run").replace("\n", " ").strip()
        if len(root_title) > 80:
            root_title = root_title[:77] + "..."
        root_title = root_title.replace("(", "[").replace(")", "]")
    lines = ["mindmap", f"  root(({root_title}))"]
    emitted: set = set()

    if s == "top":
        roots = children.get(None, [])
        for r in roots:
            _emit_subtree(lines, r, 1, children, 1, emitted)
        return "\n".join(lines) + "\n"

    if s == "focus":
        if node_key:
            if node_key not in by_key:
                return "\n".join(lines + [f"    Unknown node_key: {node_key}"]) + "\n"
            # In focus mode, selected node is the center root; render one child level only.
            for child in children.get(node_key, []) or []:
                _emit_subtree(lines, child, 1, children, 1, emitted)
            return "\n".join(lines) + "\n"
        roots = children.get(None, [])
        for r in roots:
            _emit_subtree(lines, r, 1, children, 1, emitted)
        return "\n".join(lines) + "\n"

    if s == "full":
        roots = children.get(None, [])
        for r in roots:
            _emit_subtree(lines, r, 1, children, None, emitted)
        return "\n".join(lines) + "\n"

    if s == "subtree":
        if not node_key or node_key not in by_key:
            return "\n".join(lines + [f"    Unknown node_key: {node_key}"]) + "\n"
        cap = depth if isinstance(depth, int) and depth >= 0 else 2
        anchor = by_key[node_key]
        _emit_subtree(lines, anchor, 1, children, cap, emitted)
        return "\n".join(lines) + "\n"

    if s == "path":
        if not node_key or node_key not in by_key:
            return "\n".join(lines + [f"    Unknown node_key: {node_key}"]) + "\n"
        path = _path_to_root(node_key, by_key)
        for i, node in enumerate(path):
            indent = " " * max(2 + 2 * (i + 1), 4)
            lines.append(f"{indent}{_mermaid_label(node)}")
        return "\n".join(lines) + "\n"

    lines.append(f"    Unknown scope: {scope}")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Saved-view-state repair
# ---------------------------------------------------------------------------


def apply_view_state_to_tree(
    view_state: Optional[Dict[str, Any]],
    snapshot: Dict[str, Any],
) -> Dict[str, Any]:
    """Repair a saved selection against a possibly-updated tree.

    If a saved node_key no longer exists, walk up the saved node_path to the
    nearest ancestor that does. If nothing survives, fall back to scope=top.
    """
    flat = snapshot.get("flat_nodes") if isinstance(snapshot, dict) else []
    by_key = _index_flat(list(flat or []))
    if not isinstance(view_state, dict):
        view_state = {}
    node_path = [str(k).strip() for k in (view_state.get("node_path") or []) if str(k).strip()]
    selected = str(view_state.get("selected_node_key") or "").strip()
    scope = str(view_state.get("view_scope") or "focus").strip().lower() or "focus"
    if scope not in {"focus", "top", "subtree", "full", "path"}:
        scope = "focus"
    try:
        depth_cap = int(view_state.get("depth_cap") or 2)
    except Exception:
        depth_cap = 2

    survivors = [k for k in node_path if k in by_key]
    if selected and selected in by_key and selected not in survivors:
        survivors.append(selected)
    new_selected = survivors[-1] if survivors else None
    new_path: List[str] = []
    if new_selected:
        new_path = [n["node_key"] for n in _path_to_root(new_selected, by_key)]
        if scope in {"top", "path"}:
            scope = "focus"
    else:
        if scope in {"subtree", "path"}:
            scope = "focus"

    return {
        "node_path": new_path,
        "selected_node_key": new_selected,
        "view_scope": scope,
        "depth_cap": max(0, depth_cap),
        "pfm_tree_version": int(snapshot.get("version") or 0),
        "updated_at": int(time.time() * 1000),
    }
