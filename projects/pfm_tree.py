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
# Operator mindmap readability: UI drills down one level at a time.
MAX_SIBLINGS_PER_PARENT = 15

PFM_HIERARCHY_READABILITY_RULES = (
    "PFM MINDMAP READABILITY (mandatory for commit_pfm_snapshot and report_running_step pfm_node):\n"
    "- Build a **multi-level** tree: product domain → functional area → feature group → specific feature.\n"
    f"- At **each** parent node, include at most **{MAX_SIBLINGS_PER_PARENT}** direct children (siblings). "
    "If you discover more, add an intermediate grouping level (e.g. `home/task_board/columns` "
    "instead of 30 nodes directly under `home`).\n"
    "- Do **not** hang table column headers, filter chips, pagination labels, or other UI chrome as "
    "separate mindmap nodes unless they are true product capabilities. Put that detail in the node's "
    "EAD Markdown report instead.\n"
    "- Operators navigate the UI mindmap level-by-level; design the tree for drill-down, not one giant flat star.\n"
)

PFM_NODE_TITLE_RULES = (
    "PFM NODE TITLES (displayed on the mindmap):\n"
    "- Use the **functional name only** (e.g. Authentication, Home / Task Board).\n"
    "- Do **not** prefix titles with outline numbers (wrong: `1. Authentication`, `2. Home`).\n"
    "- Do **not** prefix with emojis used as list markers; put status in the node status field instead.\n"
)

PFM_PARENT_KEY_RULES = (
    "PFM PARENT / HUB RULES (mandatory for commit_pfm_snapshot):\n"
    "- The operator UI center is the **project/run name** (e.g. P1 Test) — not shown as a breadcrumb step.\n"
    "- The operator UI center is the **project/run name** (e.g. P1 Test) — not a separate mindmap node.\n"
    "- Use **at most one** hub node with parent_node_key null (product root). All functional modules "
    "(Authentication, Home, Management, …) MUST be children of that hub: parent_node_key = hub node_key.\n"
    "- **Never** set parent_node_key null on modules, pages, columns, filters, or UI-detail nodes.\n"
    "- node_key pattern: hub `p1-test` → module `p1-test/authentication`, not `authentication` with null parent.\n"
    "- The first mindmap screen shows the hub's children around the run name; drill-down uses focus on each module.\n"
)


def snapshot_generation(snap: Optional[Dict[str, Any]]) -> int:
    """Template PFM generation (e.g. 10). Zero when absent on legacy snapshots."""
    if not isinstance(snap, dict):
        return 0
    try:
        gen = int(snap.get("generation") or 0)
    except Exception:
        gen = 0
    return gen if gen > 0 else 0


def snapshot_revision(snap: Optional[Dict[str, Any]]) -> int:
    """Within-run revision (Rev 1, Rev 2, …). Legacy snapshots used ``version`` only."""
    if not isinstance(snap, dict):
        return 0
    try:
        rev = int(snap.get("revision") or 0)
    except Exception:
        rev = 0
    try:
        ver = int(snap.get("version") or 0)
    except Exception:
        ver = 0
    if rev > 0:
        return max(rev, ver) if ver > rev else rev
    return ver if ver > 0 else 0


def snapshot_finalized(snap: Optional[Dict[str, Any]]) -> bool:
    if not isinstance(snap, dict):
        return False
    return bool(snap.get("finalized"))


def baseline_generation_from_snap(snap: Optional[Dict[str, Any]]) -> int:
    """
    Generation on a prior run's tree (for v9 → v10).

    Legacy trees stored only ``version``; on canonical/baseline runs that counter
    often equals template generation.
    """
    gen = snapshot_generation(snap)
    if gen > 0:
        return gen
    if not isinstance(snap, dict):
        return 0
    try:
        return int(snap.get("version") or 0)
    except Exception:
        return 0


def snapshot_has_committed_tree(snap: Optional[Dict[str, Any]]) -> bool:
    if not isinstance(snap, dict):
        return False
    flat = snap.get("flat_nodes") or snap.get("flatNodes") or []
    if not isinstance(flat, list) or len(flat) == 0:
        return False
    return snapshot_generation(snap) > 0 or snapshot_revision(snap) > 0


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
        title = strip_pfm_node_display_title(str(raw.get("title") or ""))
        if not title:
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


def _enforce_sibling_limits(flat: List[Dict[str, Any]]) -> None:
    """Reject snapshots where any parent has more than MAX_SIBLINGS_PER_PARENT direct children."""
    from collections import defaultdict

    counts: Dict[Optional[str], int] = defaultdict(int)
    for node in flat:
        parent = node.get("parent_node_key")
        if parent is not None and not str(parent).strip():
            parent = None
        counts[parent] += 1
    violations = [
        (parent, count)
        for parent, count in counts.items()
        if count > MAX_SIBLINGS_PER_PARENT
    ]
    if not violations:
        return
    parent_key, count = max(violations, key=lambda item: item[1])
    parent_label = parent_key if parent_key else "(top level)"
    raise SnapshotValidationError(
        f"parent {parent_label!r} has {count} direct children; maximum is {MAX_SIBLINGS_PER_PARENT}. "
        "Add intermediate grouping levels and move UI-detail text into node EAD reports.",
        code="too_many_siblings",
    )


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
    (no prior tree): ``report_carry_source_keys`` may still list node_keys that
    already have ``node_ead_report`` rows on disk (inheritance seed); those may
    omit fresh markdown. If carry sets are empty, every node requires a new report.

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
    _enforce_sibling_limits(flat)

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


def _execution_project_center_title(execution: ProjectExecute) -> str:
    """Run/project label for the mindmap center (strip leading ``Run:`` from execution name)."""
    raw = str(execution.name or execution.id or "PFM Run").strip()
    raw = re.sub(r"^Run:\s*", "", raw, flags=re.IGNORECASE).strip() or str(
        execution.id or "PFM Run"
    )
    return _sanitize_mermaid_mindmap_text(raw, fallback=execution.id or "PFM Run")


_CENTER_HUB_TITLES = frozenset(
    {"hub", "internal hub", "pfm hub", "center hub"},
)


def _slug_last_segment(node_key: str) -> str:
    key = str(node_key or "").strip()
    if "/" in key:
        return key.rsplit("/", 1)[-1].lower()
    return key.lower()


def _is_likely_center_hub_row(node: Dict[str, Any]) -> bool:
    """True for the hidden internal hub row, not product modules promoted to null parent."""
    key = str(node.get("node_key") or "").strip()
    if not key:
        return False
    title = str(node.get("title") or "").strip().lower()
    slug = _slug_last_segment(key)
    if slug in ("hub", "pfm-hub", "product-hub") or slug.endswith("-hub"):
        return True
    if title in _CENTER_HUB_TITLES:
        return True
    if key.lower() in ("hub", "pfm-hub", "product-hub"):
        return True
    return False


def collect_center_hub_keys(flat: List[Dict[str, Any]]) -> Set[str]:
    """
    Internal center hub row keys (hidden in the UI).

    After ``normalize_operator_flat_nodes``, product modules use null parent and
    are **not** hub rows. Only null-parent nodes that are referenced as parent and
    look like a hub (or the lone null-parent hub in the canonical shape) qualify.
    """
    nodes = list(flat or [])
    null_parent: List[Dict[str, Any]] = []
    child_parent_keys: Set[str] = set()
    for node in nodes:
        key = str(node.get("node_key") or "").strip()
        if not key:
            continue
        parent = str(node.get("parent_node_key") or "").strip()
        if parent:
            child_parent_keys.add(parent)
        else:
            null_parent.append(node)

    if not null_parent:
        return set()

    if len(null_parent) == 1:
        only = null_parent[0]
        only_key = str(only.get("node_key") or "").strip()
        if not only_key:
            return set()
        if only_key in child_parent_keys:
            return {only_key}
        if _is_likely_center_hub_row(only):
            return {only_key}
        return set()

    hubs: Set[str] = set()
    for node in null_parent:
        key = str(node.get("node_key") or "").strip()
        if not key or key not in child_parent_keys:
            continue
        if _is_likely_center_hub_row(node):
            hubs.add(key)
    return hubs


def is_operator_top_level_node(
    node: Dict[str, Any],
    center_hub_keys: Set[str],
) -> bool:
    """
    Operator top-level spokes around the run name (not the hidden hub row).

    A node is top-level when parent is null (legacy) or parent is a center hub key.
    """
    key = str(node.get("node_key") or "").strip()
    if not key or key in center_hub_keys:
        return False
    if _is_junk_mindmap_node_title(str(node.get("title") or "")):
        return False
    parent = str(node.get("parent_node_key") or "").strip()
    if not parent:
        return True
    if parent in center_hub_keys:
        return True
    return False


def operator_top_level_spoke_nodes(flat: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Product modules shown on the first mindmap screen (run name at center)."""
    hub_keys = collect_center_hub_keys(flat)
    spokes = [n for n in flat if is_operator_top_level_node(n, hub_keys)]
    spokes.sort(
        key=lambda n: (int(n.get("level") or 0), str(n.get("title") or n.get("node_key") or "").lower()),
    )
    return spokes


def normalize_operator_flat_nodes(flat: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Remove internal center hub rows and promote hub children to null-parent modules.

    Also collapses duplicate hub path segments in node_key / parent_node_key.
    """
    hub_keys = collect_center_hub_keys(flat)
    if not hub_keys:
        out: List[Dict[str, Any]] = []
        for node in flat or []:
            row = dict(node)
            nk = collapse_duplicate_hub_path_segments(str(row.get("node_key") or ""))
            pk = str(row.get("parent_node_key") or "").strip()
            pk = collapse_duplicate_hub_path_segments(pk) if pk else ""
            row["node_key"] = nk
            row["node_id"] = str(row.get("node_id") or nk)
            if pk:
                row["parent_node_key"] = pk
            else:
                row["parent_node_key"] = None
            out.append(row)
        return out

    out: List[Dict[str, Any]] = []
    for node in flat or []:
        key = str(node.get("node_key") or "").strip()
        if not key or key in hub_keys:
            continue
        row = dict(node)
        nk = collapse_duplicate_hub_path_segments(key)
        parent = str(node.get("parent_node_key") or "").strip()
        if parent in hub_keys:
            parent = ""
        elif parent:
            parent = collapse_duplicate_hub_path_segments(parent)
        row["node_key"] = nk
        row["node_id"] = str(row.get("node_id") or nk)
        row["parent_node_key"] = parent or None
        out.append(row)
    return out


def _top_level_spoke_nodes(
    children: Dict[Optional[str], List[Dict[str, Any]]],
    flat: Optional[List[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    """Backward-compatible wrapper; prefer ``operator_top_level_spoke_nodes(flat)``."""
    if flat is not None:
        return operator_top_level_spoke_nodes(flat)
    null_roots = list(children.get(None) or [])
    hub_keys = {str(n.get("node_key") or "").strip() for n in null_roots if n.get("node_key")}
    spokes: List[Dict[str, Any]] = []
    for parent_key, plist in children.items():
        if parent_key is None or str(parent_key).strip() in hub_keys:
            for n in plist or []:
                nk = str(n.get("node_key") or "").strip()
                if nk and nk not in hub_keys:
                    spokes.append(n)
    if not spokes:
        spokes = null_roots
    return [n for n in spokes if not _is_junk_mindmap_node_title(str(n.get("title") or ""))]


def collapse_duplicate_hub_path_segments(node_key: str) -> str:
    """Collapse consecutive duplicate path segments (``hub/hub/module`` → ``hub/module``)."""
    parts = [p.strip() for p in str(node_key or "").split("/") if p.strip()]
    if len(parts) < 2:
        return str(node_key or "").strip()
    out: List[str] = []
    for part in parts:
        if out and out[-1] == part:
            continue
        out.append(part)
    return "/".join(out)


def resolve_node_key_in_snapshot(
    node_key: Optional[str],
    by_key: Dict[str, Dict[str, Any]],
) -> Optional[str]:
    """
    Map operator UI keys to canonical ``flat_nodes`` node_key values.

    Committed trees may store duplicate hub segments (``hub/hub/module``) while
    breadcrumbs or legacy paths send a collapsed key (``hub/module``).
    """
    raw = str(node_key or "").strip()
    if not raw or not by_key:
        return None
    if raw in by_key:
        return raw
    collapsed = collapse_duplicate_hub_path_segments(raw)
    if collapsed in by_key:
        return collapsed
    collapsed_matches = [
        stored
        for stored in by_key
        if collapse_duplicate_hub_path_segments(stored) == collapsed
    ]
    if len(collapsed_matches) == 1:
        return collapsed_matches[0]
    if len(collapsed_matches) > 1:
        last = collapsed.rsplit("/", 1)[-1].lower()
        suffix_matches = [
            stored
            for stored in collapsed_matches
            if stored.rsplit("/", 1)[-1].lower() == last
        ]
        if len(suffix_matches) == 1:
            return suffix_matches[0]
    last = raw.rsplit("/", 1)[-1].lower()
    if last:
        seg_matches = [
            stored for stored in by_key if stored.rsplit("/", 1)[-1].lower() == last
        ]
        if len(seg_matches) == 1:
            return seg_matches[0]
    return None


def strip_pfm_node_display_title(title: str) -> str:
    """Human-readable PFM title: no leading outline numbers (``1.``, ``1.2.``) or list emojis."""
    raw = re.sub(r"\r?\n+", " ", str(title or "")).strip()
    raw = re.sub(r"<br\s*/?>", " ", raw, flags=re.IGNORECASE)
    prev = ""
    while prev != raw:
        prev = raw
        raw = re.sub(r"^[\s\U0001F300-\U0001FAFF]+", "", raw)
        raw = re.sub(r"^(?:\d+\.)+\s*", "", raw)
        raw = re.sub(r"^\d+\s+", "", raw)
    return re.sub(r"\s{2,}", " ", raw).strip()


def _humanize_node_key_slug(node_key: str) -> str:
    slug = str(node_key or "").strip().split("/")[-1] or "node"
    slug = re.sub(r"[-_]+", " ", slug).strip()
    return slug[:72].strip() or "node"


def _sanitize_mermaid_mindmap_text(value: str, *, fallback: str = "node") -> str:
    """Strip markdown HR / mindmap-breaking punctuation from a single node label."""
    if _is_junk_mindmap_node_title(str(value or "")):
        value = _humanize_node_key_slug(fallback)
    raw = strip_pfm_node_display_title(str(value or ""))
    if not raw:
        raw = re.sub(r"\r?\n+", " ", str(value or "")).strip()
        raw = re.sub(r"<br\s*/?>", " ", raw, flags=re.IGNORECASE)
    raw = re.sub(r"#{1,6}\s*", "", raw)
    raw = re.sub(r"\*\*([^*]+)\*\*", r"\1", raw)
    raw = re.sub(r"`([^`]+)`", r"\1", raw)
    raw = re.sub(r"[\u2010-\u2015\u2212\u2500-\u2503\-]{3,}", " ", raw)
    raw = re.sub(r"_{3,}", " ", raw)
    raw = re.sub(r"={3,}", " ", raw)
    raw = re.sub(r"\s{2,}", " ", raw).strip()
    if not raw or re.fullmatch(r"[-_.= \u2010-\u2015]+", raw):
        raw = _humanize_node_key_slug(fallback)
    if len(raw) > 72:
        raw = raw[:69] + "..."
    return raw.replace("(", "[").replace(")", "]").replace('"', "'")


_AGENT_NARRATION_RE = re.compile(
    r"(?i)\b(let me|i'll|i've|i am|screenshot|captured|login page|timeout|console shows|"
    r"navigating to|proceed with|check the current state|starting now)\b"
)


def _is_junk_mindmap_node_title(title: str) -> bool:
    """True when a flat_nodes title is markdown chrome or agent narration, not a PFM feature name."""
    raw = re.sub(r"\r?\n+", " ", str(title or "")).strip()
    if not raw:
        return True
    if re.fullmatch(r"[-_.= \u2010-\u2015]+", raw):
        return True
    if re.fullmatch(r"[\u2010-\u2015\-]{3,}", raw):
        return True
    if raw.startswith("##") or raw.startswith("# "):
        return True
    if len(raw) > 72:
        return True
    if _AGENT_NARRATION_RE.search(raw):
        return True
    words = [w for w in raw.split() if w]
    if len(words) > 10:
        return True
    return False


def tree_looks_like_agent_narration_flat(flat: List[Dict[str, Any]]) -> bool:
    """True when committed flat_nodes are mostly null-parent agent log lines, not a product tree."""
    if len(flat) < 3:
        return False
    null_parent = sum(1 for n in flat if not str(n.get("parent_node_key") or "").strip())
    if null_parent < max(3, int(len(flat) * 0.75)):
        return False
    junk = sum(1 for n in flat if _is_junk_mindmap_node_title(str(n.get("title") or "")))
    return junk >= max(2, int(len(flat) * 0.6))


def sanitize_mermaid_diagram(definition: str) -> str:
    """Final pass on rendered mindmap text before the UI hands it to Mermaid."""
    out: List[str] = []
    for line in str(definition or "").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if re.fullmatch(r"[-_.=\u2010-\u2015]{3,}", stripped):
            continue
        cleaned = re.sub(r"[\u2010-\u2015\u2212\u2500-\u2503\-]{3,}", " ", line)
        cleaned = re.sub(r"<br\s*/?>", " ", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"_{3,}", " ", cleaned)
        cleaned = re.sub(r"={3,}", " ", cleaned)
        if cleaned.strip():
            out.append(cleaned.rstrip())
    if not out:
        return "mindmap\n  root((PFM))\n    No nodes\n"
    return "\n".join(out) + "\n"


def _mermaid_label(node: Dict[str, Any]) -> str:
    fallback = str(node.get("node_key") or "node").strip() or "node"
    return _sanitize_mermaid_mindmap_text(
        str(node.get("title") or fallback),
        fallback=fallback,
    )


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
    if _is_junk_mindmap_node_title(str(node.get("title") or "")):
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
    center_hubs = collect_center_hub_keys(flat)
    resolved_key = resolve_node_key_in_snapshot(node_key, by_key) if node_key else None
    if resolved_key and resolved_key in center_hubs:
        resolved_key = None

    if not flat:
        root_title = _execution_project_center_title(execution)
        lines = ["mindmap", f"  root(({root_title}))"]
        lines.append("    No PFM tree committed yet")
        return "\n".join(lines) + "\n"

    s = (scope or "top").strip().lower()
    if s == "focus" and resolved_key:
        root_title = _mermaid_label(by_key[resolved_key])
    else:
        root_title = _execution_project_center_title(execution)
    lines = ["mindmap", f"  root(({root_title}))"]
    emitted: set = set()

    if s == "top":
        if tree_looks_like_agent_narration_flat(flat):
            lines.append("    Invalid PFM tree (agent log lines, not features)")
            lines.append("    Use a run with commit_pfm_snapshot or refresh delivery")
            return sanitize_mermaid_diagram("\n".join(lines) + "\n")
        for r in operator_top_level_spoke_nodes(flat):
            _emit_subtree(lines, r, 1, children, 1, emitted)
        if not emitted:
            lines.append("    No displayable PFM modules in committed tree")
        return sanitize_mermaid_diagram("\n".join(lines) + "\n")

    if s == "focus":
        if node_key:
            if not resolved_key:
                return sanitize_mermaid_diagram(
                    "\n".join(lines + [f"    Unknown node_key: {node_key}"]) + "\n"
                )
            for child in children.get(resolved_key, []) or []:
                _emit_subtree(lines, child, 1, children, 1, emitted)
            return sanitize_mermaid_diagram("\n".join(lines) + "\n")
        for r in operator_top_level_spoke_nodes(flat):
            _emit_subtree(lines, r, 1, children, 1, emitted)
        return sanitize_mermaid_diagram("\n".join(lines) + "\n")

    if s == "full":
        roots = children.get(None, [])
        for r in roots:
            _emit_subtree(lines, r, 1, children, None, emitted)
        return sanitize_mermaid_diagram("\n".join(lines) + "\n")

    if s == "subtree":
        if not node_key or not resolved_key:
            return sanitize_mermaid_diagram(
                "\n".join(lines + [f"    Unknown node_key: {node_key}"]) + "\n"
            )
        cap = depth if isinstance(depth, int) and depth >= 0 else 2
        anchor = by_key[resolved_key]
        _emit_subtree(lines, anchor, 1, children, cap, emitted)
        return sanitize_mermaid_diagram("\n".join(lines) + "\n")

    if s == "path":
        if not node_key or not resolved_key:
            return sanitize_mermaid_diagram(
                "\n".join(lines + [f"    Unknown node_key: {node_key}"]) + "\n"
            )
        path = _path_to_root(resolved_key, by_key)
        for i, node in enumerate(path):
            indent = " " * max(2 + 2 * (i + 1), 4)
            lines.append(f"{indent}{_mermaid_label(node)}")
        return sanitize_mermaid_diagram("\n".join(lines) + "\n")

    lines.append(f"    Unknown scope: {scope}")
    return sanitize_mermaid_diagram("\n".join(lines) + "\n")


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
    scope = str(view_state.get("view_scope") or "top").strip().lower() or "top"
    try:
        depth_cap = int(view_state.get("depth_cap") or 2)
    except Exception:
        depth_cap = 2

    survivors: List[str] = []
    for k in node_path:
        resolved = resolve_node_key_in_snapshot(k, by_key)
        if resolved and resolved not in survivors:
            survivors.append(resolved)
    selected_resolved = resolve_node_key_in_snapshot(selected, by_key) if selected else None
    if selected_resolved and selected_resolved not in survivors:
        survivors.append(selected_resolved)
    new_selected = survivors[-1] if survivors else None
    new_path: List[str] = []
    if new_selected:
        new_path = [n["node_key"] for n in _path_to_root(new_selected, by_key)]
        if scope == "top":
            scope = "subtree"
    else:
        scope = "top"

    return {
        "node_path": new_path,
        "selected_node_key": new_selected,
        "view_scope": scope,
        "depth_cap": max(0, depth_cap),
        "pfm_tree_version": int(snapshot.get("version") or 0),
        "updated_at": int(time.time() * 1000),
    }
