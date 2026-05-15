"""
PFM artifact generation for project runs.

Creates a Mermaid mindmap (`.mmd`) and Markdown report (`.md`) for each
execution, with optional upload to S3-compatible storage.
"""

from __future__ import annotations

import os
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Set

from .models import (
    EadFmNodeRun,
    ProjectExecute,
    ProjectReportArtifact,
    ProgressLogEntry,
    TestCaseRun,
    TestCaseStepRunStatus,
)


def reports_root_dir() -> Path:
    configured = os.getenv("EAD_REPORT_DIR", "").strip()
    if configured:
        return Path(configured).expanduser()
    return Path.home() / ".hermes" / "projects" / "reports"


def report_file_path(execution_id: str, filename: str) -> Path:
    safe_name = Path(filename).name
    return reports_root_dir() / execution_id / safe_name


def derive_node_runs_from_progress(progress_log: List[ProgressLogEntry], limit: int = 20) -> List[EadFmNodeRun]:
    seen: set[str] = set()
    titles: List[str] = []

    for entry in progress_log:
        if entry.kind not in ("tool_use", "assistant"):
            continue
        text = (entry.text or "").strip()
        if not text:
            continue
        normalized = re.sub(r"\s+", " ", text)
        if len(normalized) > 120:
            normalized = normalized[:117] + "..."
        if normalized.lower() in seen:
            continue
        seen.add(normalized.lower())
        titles.append(normalized)
        if len(titles) >= limit:
            break

    runs: List[EadFmNodeRun] = []
    for idx, title in enumerate(titles):
        node_key = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-") or f"live-node-{idx + 1}"
        runs.append(
            EadFmNodeRun(
                node_id=f"live-node-{idx + 1}",
                node_key=node_key,
                type="live-signal",
                title=title,
                status="No Run",
                test_case_runs=[],
            )
        )
    return runs


def _pfm_node_dict_to_ead_run(
    pfm_node: dict,
    title_fallback: str = "",
) -> Optional[EadFmNodeRun]:
    """Build EadFmNodeRun from report_running_step pfm_node payload (same shape as projects/tools)."""
    if not isinstance(pfm_node, dict):
        return None
    title = str(title_fallback or "").strip()
    raw_node_key = str(
        pfm_node.get("node_key")
        or pfm_node.get("node_id")
        or re.sub(r"[^a-z0-9]+", "-", str(pfm_node.get("title", "")).lower()).strip("-")
        or ""
    ).strip()
    if not raw_node_key and not pfm_node.get("title"):
        return None
    parent_node_key = str(pfm_node.get("parent_node_key") or "").strip() or None
    node_level_raw = pfm_node.get("level")
    try:
        node_level = max(0, int(node_level_raw)) if node_level_raw is not None else 0
    except Exception:
        node_level = 0
    node_description = str(pfm_node.get("description") or "").strip()
    node_key = raw_node_key or "node"
    if parent_node_key and "/" not in node_key and ">" not in node_key:
        node_key = f"{parent_node_key}/{node_key}"
    node_id = str(pfm_node.get("node_id") or node_key)
    node_title = str(pfm_node.get("title") or title or node_key)
    node_type = str(pfm_node.get("type") or "feature-area")
    node_status_raw = str(pfm_node.get("status") or "No Run").strip().lower()
    node_status = (
        TestCaseStepRunStatus.SUCCESS
        if node_status_raw == "success"
        else TestCaseStepRunStatus.FAILED
        if node_status_raw == "failed"
        else TestCaseStepRunStatus.NO_RUN
    )

    case_runs: List[TestCaseRun] = []
    raw_cases = pfm_node.get("test_cases") or []
    if isinstance(raw_cases, list):
        for idx, raw in enumerate(raw_cases):
            if not isinstance(raw, dict):
                continue
            raw_status = str(raw.get("status") or "No Run").strip().lower()
            case_status = (
                TestCaseStepRunStatus.SUCCESS
                if raw_status == "success"
                else TestCaseStepRunStatus.FAILED
                if raw_status == "failed"
                else TestCaseStepRunStatus.NO_RUN
            )
            case_title = str(raw.get("title") or f"Test case {idx + 1}")
            case_id = str(
                raw.get("case_id")
                or re.sub(r"[^a-z0-9]+", "-", case_title.lower()).strip("-")
                or f"case-{idx+1}"
            )
            case_runs.append(
                TestCaseRun(
                    case_id=case_id,
                    title=case_title,
                    status=case_status,
                    test_case_step_runs=[],
                )
            )

    return EadFmNodeRun(
        node_id=node_id,
        node_key=node_key,
        parent_node_key=parent_node_key,
        level=node_level,
        type=node_type,
        title=node_title,
        meta=node_description,
        status=node_status,
        test_case_runs=case_runs,
    )


def derive_pfm_nodes_from_report_running_step_progress(
    progress_log: List[ProgressLogEntry],
) -> List[EadFmNodeRun]:
    """
    Recover PFM nodes declared in report_running_step tool calls from the progress log.
    Later log entries win for the same node_key (last reported state).
    """
    ordered: List[EadFmNodeRun] = []
    index_by_key: Dict[str, int] = {}
    for entry in progress_log:
        if entry.kind != "tool_use" or (entry.tool_name or "") != "report_running_step":
            continue
        ti = entry.tool_input
        if not isinstance(ti, dict):
            continue
        pfm_node = ti.get("pfm_node")
        if not isinstance(pfm_node, dict):
            continue
        title_fallback = str(ti.get("title") or "")
        run = _pfm_node_dict_to_ead_run(pfm_node, title_fallback=title_fallback)
        if not run or not run.node_key:
            continue
        k = run.node_key
        if k in index_by_key:
            ordered[index_by_key[k]] = run
        else:
            index_by_key[k] = len(ordered)
            ordered.append(run)
    return ordered


def _slug_node_key(title: str, fallback: str) -> str:
    base = re.sub(r"[^a-z0-9]+", "-", (title or "").lower()).strip("-")[:72]
    return (base or fallback)[:96]


def derive_pfm_nodes_from_ascii_tree_text(content: str) -> List[EadFmNodeRun]:
    """
    Recover ``EadFmNodeRun`` rows from a textual tree using ``├──`` / ``└──`` and ``│``.

    Supports multi-level trees (Kloud-style). The first non-empty prose line without
    branch drawing characters becomes the root title.
    """
    lines = [ln.rstrip("\n\r") for ln in content.splitlines()]
    branch_re = re.compile(r"^([\s│|]*)(?:├──|└──)\s*(?:\d+\.\s*)?(.*)$")

    root_title = ""
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("```"):
            continue
        if re.fullmatch(r"[\s│|]+", line):
            continue
        if branch_re.match(line):
            continue
        if "──" in stripped:
            continue
        if stripped.startswith("#") or stripped.lower() == "mindmap":
            continue
        root_title = stripped
        break

    if not root_title:
        root_title = "Product"

    rk = _slug_node_key(root_title, "pfm-root")
    runs: List[EadFmNodeRun] = [
        EadFmNodeRun(
            node_id=rk,
            node_key=rk,
            parent_node_key=None,
            level=1,
            type="product-area",
            title=root_title[:240],
            status=TestCaseStepRunStatus.NO_RUN,
            test_case_runs=[],
        )
    ]
    used: Set[str] = {rk}
    stack: List[tuple[int, str]] = [(0, rk)]

    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("```"):
            continue
        if re.fullmatch(r"[\s│|]+", line):
            continue
        m = branch_re.match(line)
        if not m:
            continue
        prefix, label = m.group(1), m.group(2).strip()
        if not label:
            continue
        norm_prefix = prefix.replace("|", "│")
        depth = 1 + norm_prefix.count("│")
        while stack and stack[-1][0] >= depth:
            stack.pop()
        parent_key = stack[-1][1] if stack else rk
        slug = _slug_node_key(label, f"n{len(runs)}")
        nk = f"{parent_key}/{slug}" if parent_key else slug
        if len(nk) > 200:
            nk = nk[-200:]
        base = nk
        i = 0
        while nk in used:
            i += 1
            nk = f"{base}-d{i}"[-200:]
        used.add(nk)
        runs.append(
            EadFmNodeRun(
                node_id=nk,
                node_key=nk,
                parent_node_key=parent_key,
                level=min(64, depth + 1),
                type="feature-area",
                title=label[:240],
                status=TestCaseStepRunStatus.NO_RUN,
                test_case_runs=[],
            )
        )
        stack.append((depth, nk))

    return runs if len(runs) > 1 else []


def derive_pfm_nodes_from_mermaid_mindmap_text(content: str) -> List[EadFmNodeRun]:
    """
    Recover ``EadFmNodeRun`` rows from Mermaid ``mindmap`` text (``root((...))`` and ``(...)`` nodes).

    Lines without leading node shapes (e.g. ``Status:`` continuations) are skipped.
    """
    runs: List[EadFmNodeRun] = []
    stack: List[tuple[int, str]] = []

    for raw in content.splitlines():
        stripped = raw.strip()
        if not stripped or stripped.startswith("```"):
            continue
        s = raw.rstrip("\n")
        leading = len(s) - len(s.lstrip(" "))
        rest = s.lstrip(" ")
        if rest == "mindmap":
            continue
        if not (rest.startswith("(") or rest.startswith("root((")):
            continue
        label: Optional[str] = None
        m = re.search(r"root\(\((.+?)\)\)", rest)
        if m:
            label = m.group(1).strip()
        else:
            m2 = re.match(r"\(\((.+?)\)\)", rest)
            if m2:
                label = m2.group(1).strip()
            else:
                m3 = re.match(r"\((.+?)\)", rest)
                if m3:
                    label = m3.group(1).strip()
        if not label:
            continue
        if "[" in label and "]" in label:
            label = label.split("[", 1)[0].strip()
        depth = max(0, leading // 2 - 1) if leading >= 2 else 0
        while stack and stack[-1][0] >= depth:
            stack.pop()
        parent_key = stack[-1][1] if stack else None
        nk = _slug_node_key(label, f"mm-{len(runs)}")
        used_keys = {r.node_key for r in runs}
        if nk in used_keys:
            nk = f"{nk}-{len(runs)}"
        runs.append(
            EadFmNodeRun(
                node_id=nk,
                node_key=nk,
                parent_node_key=parent_key,
                level=max(1, depth + 1),
                type="feature-area",
                title=label[:240],
                status=TestCaseStepRunStatus.NO_RUN,
                test_case_runs=[],
            )
        )
        stack.append((depth, nk))
    return runs


def nodes_from_markdown_fenced_mermaid(md: str) -> List[EadFmNodeRun]:
    """Extract `` ```mermaid `` blocks and parse mindmap nodes when possible."""
    if not (md or "").strip():
        return []
    for m in re.finditer(r"```mermaid\s*([\s\S]*?)```", md, re.IGNORECASE):
        inner = (m.group(1) or "").strip()
        nodes = derive_pfm_nodes_from_mermaid_mindmap_text(inner)
        if nodes:
            return nodes
        nodes = derive_pfm_nodes_from_ascii_tree_text(inner)
        if nodes:
            return nodes
    return []


def derive_pfm_nodes_from_saved_mindmap_or_report_text(content: str) -> List[EadFmNodeRun]:
    """
    Parse saved PFM mindmap / report text into ``EadFmNodeRun`` rows.

    Tries ASCII tree lines first, then Mermaid ``mindmap`` syntax.
    """
    if not (content or "").strip():
        return []
    lines = content.splitlines()
    if any("├──" in ln or "└──" in ln for ln in lines):
        nodes = derive_pfm_nodes_from_ascii_tree_text(content)
        if nodes:
            return nodes
    if any(ln.strip() == "mindmap" for ln in lines[:12]) or "root((" in content:
        nodes = derive_pfm_nodes_from_mermaid_mindmap_text(content)
        if nodes:
            return nodes
    return []


_AGENT_AUTHORED_ENV = "EAD_PFM_AGENT_AUTHORED"


def _agent_authored_enabled() -> bool:
    """Strict agent-authored mode: only the committed PFM tree feeds the official mindmap."""
    raw = (os.getenv(_AGENT_AUTHORED_ENV, "true") or "").strip().lower()
    return raw not in ("0", "false", "no", "off")


def _load_committed_tree_runs(execution: ProjectExecute) -> Optional[List[EadFmNodeRun]]:
    """Read the canonical pfm_tree artifact and return its flattened EadFmNodeRun list.

    Returns None when no tree has ever been committed for this execution. An empty
    list indicates an explicit empty tree (treated upstream as "Awaiting first commit").
    """
    try:
        from .pfm_tree import PFM_TREE_ARTIFACT_KEY, flat_nodes_to_ead_runs
        from .store import ProjectStore
    except Exception:
        return None

    try:
        store = ProjectStore()
    except Exception:
        return None
    artifact = store.get_execution_pfm_artifact(execution.id, PFM_TREE_ARTIFACT_KEY)
    if not isinstance(artifact, dict):
        return None
    snap = artifact.get("snapshot")
    if not isinstance(snap, dict):
        return None
    flat = snap.get("flat_nodes")
    if not isinstance(flat, list):
        return None
    return flat_nodes_to_ead_runs(flat)


def resolve_pfm_nodes_for_mindmap(execution: ProjectExecute) -> List[EadFmNodeRun]:
    """
    Nodes shown on the official PFM mindmap.

    Strict (default) mode driven by EAD_PFM_AGENT_AUTHORED:
        Read only the agent-authored, committed pfm_tree artifact. Progress-text
        fallbacks are intentionally excluded so the diagram is a faithful
        rendering of what the agent declared via `commit_pfm_snapshot`.

    Legacy mode (EAD_PFM_AGENT_AUTHORED=false):
        Merge `execution.results` with progress-log inferred nodes, as in the
        previous implementation. Retained for backward compatibility on installs
        that have not yet rolled out the new tool.
    """
    if _agent_authored_enabled():
        committed = _load_committed_tree_runs(execution)
        if committed is not None:
            return committed
        return list(execution.results or [])

    stored = list(execution.results or [])
    derived = derive_pfm_nodes_from_report_running_step_progress(execution.progress_log or [])

    if not stored and not derived:
        return derive_node_runs_from_progress(execution.progress_log or [])

    if not stored:
        return derived

    stored_keys: Set[str] = {n.node_key for n in stored if getattr(n, "node_key", None)}
    out = list(stored)
    for node in derived:
        if node.node_key and node.node_key not in stored_keys:
            out.append(node)
            stored_keys.add(node.node_key)
    return out


def _mindmap_status_text(node: EadFmNodeRun) -> str:
    s = getattr(node.status, "value", None) if node.status is not None else None
    if s is None:
        s = str(node.status if node.status is not None else "No Run")
    return str(s).replace("\n", " ").strip() or "No Run"


def _mindmap_leaf_text(node: EadFmNodeRun) -> str:
    raw = (node.title or node.node_key or node.node_id or "node").replace("\n", " ").strip()
    if len(raw) > 92:
        raw = raw[:89] + "..."
    sts = _mindmap_status_text(node)
    # Avoid parentheses in rendered line (nested shape syntax); use brackets for status.
    raw = raw.replace("(", "[").replace(")", "]").replace('"', "'")
    return f"{raw} [{sts}]"


def _nodes_by_key(nodes: List[EadFmNodeRun]) -> Dict[str, EadFmNodeRun]:
    out: Dict[str, EadFmNodeRun] = {}
    for n in nodes:
        k = (n.node_key or "").strip()
        if k:
            out[k] = n
    return out


def _nearest_path_parent_key(node_key: str, index: Dict[str, EadFmNodeRun]) -> Optional[str]:
    if not node_key or "/" not in node_key.replace("\\", "/"):
        return None
    normalized = node_key.replace("\\", "/")
    parts = normalized.split("/")
    while len(parts) > 1:
        parts.pop()
        cand = "/".join(parts).strip()
        if cand and cand in index:
            return cand
    return None


def _resolved_parent_key(
    node: EadFmNodeRun,
    index: Dict[str, EadFmNodeRun],
    own_key: str,
) -> Optional[str]:
    explicit = str(node.parent_node_key or "").strip()
    if explicit and explicit != own_key and explicit in index:
        return explicit
    return _nearest_path_parent_key(own_key, index)


def _partition_mindmap_tree(
    nodes: List[EadFmNodeRun],
    index: Dict[str, EadFmNodeRun],
) -> tuple[List[EadFmNodeRun], Dict[str, List[EadFmNodeRun]]]:
    roots: List[EadFmNodeRun] = []
    children: Dict[str, List[EadFmNodeRun]] = {}
    for node in nodes:
        own_key = (node.node_key or "").strip()
        if not own_key:
            roots.append(node)
            continue
        parent_key = _resolved_parent_key(node, index, own_key)
        if not parent_key:
            roots.append(node)
        else:
            children.setdefault(parent_key, []).append(node)
    return roots, children


def _build_mermaid_mindmap(execution: ProjectExecute, nodes: List[EadFmNodeRun]) -> str:
    root_title = (execution.name or execution.id or "PFM Run").replace("\n", " ").strip()
    if len(root_title) > 80:
        root_title = root_title[:77] + "..."
    root_title = root_title.replace("(", "[").replace(")", "]")
    lines = ["mindmap", f"  root(({root_title}))"]

    if not nodes:
        lines.append("    No PFM nodes recorded yet")
        return "\n".join(lines) + "\n"

    keyed: Dict[str, EadFmNodeRun] = {}
    orphans: List[EadFmNodeRun] = []
    for n in nodes:
        k = (n.node_key or "").strip()
        if k:
            keyed[k] = n
        else:
            orphans.append(n)
    nodes = list(keyed.values()) + orphans

    index = _nodes_by_key(nodes)
    roots, children_map = _partition_mindmap_tree(nodes, index)
    roots = sorted(
        roots,
        key=lambda n: (n.level or 0, (n.title or n.node_key or "").lower()),
    )
    for plist in children_map.values():
        plist.sort(
            key=lambda n: (n.level or 0, (n.title or n.node_key or "").lower()),
        )

    emitted: Set[str] = set()

    def emit(node: EadFmNodeRun, depth: int) -> None:
        if depth > 24:
            return
        own_key = (node.node_key or "").strip()
        if own_key and own_key in emitted:
            return
        if own_key:
            emitted.add(own_key)
        indent_n = max(2 + 2 * depth, 4)
        indent = " " * indent_n
        lines.append(f"{indent}{_mindmap_leaf_text(node)}")
        nk = own_key or f"_:id-{id(node)}"
        for child in children_map.get(nk, []) or []:
            emit(child, depth + 1)

    for root in roots:
        emit(root, depth=1)

    return "\n".join(lines) + "\n"


def _build_markdown_report(execution: ProjectExecute, nodes: List[EadFmNodeRun]) -> str:
    now_ms = int(time.time() * 1000)
    lines = [
        f"# PFM Report: {execution.name or execution.id}",
        "",
        "## Run Summary",
        f"- Execution ID: `{execution.id}`",
        f"- Status: `{execution.status.value}`",
        f"- Generated At: `{now_ms}`",
        f"- Explore Type: `{execution.explore_type or 'pfm-full'}`",
        f"- Node Count: `{len(nodes)}`",
        "",
        "## PFM Nodes",
    ]

    if not nodes:
        lines.extend(
            [
                "- No PFM nodes captured yet.",
                "",
                "## Recent Activity",
            ]
        )
    else:
        for idx, node in enumerate(nodes, start=1):
            lines.append(f"{idx}. **{node.title or node.node_key or node.node_id}** — `{node.status}`")
        lines.append("")
        lines.append("## Recent Activity")

    recent = execution.progress_log[-8:] if execution.progress_log else []
    if not recent:
        lines.append("- No progress events recorded yet.")
    else:
        for entry in recent:
            text = (entry.text or "").strip() or "(empty)"
            lines.append(f"- `{entry.kind}`: {text}")

    lines.append("")
    return "\n".join(lines)


def _try_upload_to_s3(local_path: Path, execution_id: str, filename: str) -> Optional[str]:
    bucket = os.getenv("EAD_S3_BUCKET") or os.getenv("S3_BUCKET")
    if not bucket:
        return None

    region = os.getenv("EAD_S3_REGION") or os.getenv("S3_REGION") or "us-east-1"
    endpoint = os.getenv("EAD_S3_ENDPOINT") or os.getenv("S3_ENDPOINT")
    access_key = (
        os.getenv("EAD_S3_ACCESS_KEY_ID")
        or os.getenv("S3_ACCESS_KEY")
        or os.getenv("S3_KEY_ID")
    )
    secret_key = (
        os.getenv("EAD_S3_SECRET_ACCESS_KEY")
        or os.getenv("S3_SECRET")
        or os.getenv("S3_KEY_SECRET")
    )
    acl = os.getenv("EAD_S3_ACL") or os.getenv("S3_ACL")

    if not access_key or not secret_key:
        return None

    try:
        import boto3  # type: ignore
    except Exception:
        return None

    session = boto3.session.Session()
    client = session.client(
        "s3",
        region_name=region,
        endpoint_url=endpoint or None,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
    )

    key = f"ead-reports/{execution_id}/{filename}"
    extra: Dict[str, str] = {}
    if acl:
        extra["ACL"] = acl
    content_type = "text/plain" if filename.endswith(".mmd") else "text/markdown"
    extra["ContentType"] = content_type
    client.upload_file(str(local_path), bucket, key, ExtraArgs=extra or None)

    if endpoint:
        return f"{endpoint.rstrip('/')}/{bucket}/{key}"
    return f"https://{bucket}.s3.{region}.amazonaws.com/{key}"


def build_pfm_artifact_payloads(execution: ProjectExecute) -> List[Dict]:
    nodes = resolve_pfm_nodes_for_mindmap(execution)
    created_at = int(time.time() * 1000)
    mindmap_name = "pfm-mindmap.mmd"
    report_name = "pfm-report.md"
    node_payloads = [
        node.model_dump() if hasattr(node, "model_dump") else dict(node)
        for node in nodes
    ]

    return [
        {
            "artifact_key": mindmap_name,
            "artifact_type": "pfm_mindmap",
            "title": "PFM Mindmap (Mermaid)",
            "filename": mindmap_name,
            "format": "mmd",
            "content": _build_mermaid_mindmap(execution, nodes),
            "nodes": node_payloads,
            "created_at": created_at,
        },
        {
            "artifact_key": report_name,
            "artifact_type": "pfm_report",
            "title": "PFM Report (Markdown)",
            "filename": report_name,
            "format": "md",
            "content": _build_markdown_report(execution, nodes),
            "nodes": node_payloads,
            "created_at": created_at,
        },
    ]


def build_and_persist_pfm_artifacts(execution: ProjectExecute) -> List[ProjectReportArtifact]:
    base_dir = reports_root_dir() / execution.id
    base_dir.mkdir(parents=True, exist_ok=True)

    reports = []
    for payload in build_pfm_artifact_payloads(execution):
        filename = payload["filename"]
        path = base_dir / filename
        path.write_text(str(payload.get("content") or ""), encoding="utf-8")
        uploaded_url = _try_upload_to_s3(path, execution.id, filename)
        reports.append(
            ProjectReportArtifact(
                title=payload["title"],
                filename=filename,
                format=payload["format"],
                created_at=int(payload["created_at"]),
                url=uploaded_url or f"/v1/projects/executions/{execution.id}/reports/{filename}",
            )
        )
    return reports
