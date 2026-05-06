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
from typing import Dict, List, Optional

from .models import EadFmNodeRun, ProjectExecute, ProjectReportArtifact, ProgressLogEntry


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


def _build_mermaid_mindmap(execution: ProjectExecute, nodes: List[EadFmNodeRun]) -> str:
    root_title = (execution.name or execution.id or "PFM Run").replace("\n", " ").strip()
    lines = ["mindmap", f"  root(({root_title}))"]

    if not nodes:
        lines.append("    No PFM nodes recorded yet")
        return "\n".join(lines) + "\n"

    for node in nodes:
        label = (node.title or node.node_key or node.node_id).replace("\n", " ").strip()
        if not label:
            label = node.node_id
        status = (node.status or "No Run").strip()
        lines.append(f"    {label} ({status})")

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


def build_and_persist_pfm_artifacts(execution: ProjectExecute) -> List[ProjectReportArtifact]:
    base_dir = reports_root_dir() / execution.id
    base_dir.mkdir(parents=True, exist_ok=True)

    nodes = execution.results or derive_node_runs_from_progress(execution.progress_log or [])
    mindmap_name = "pfm-mindmap.mmd"
    report_name = "pfm-report.md"
    mindmap_path = base_dir / mindmap_name
    report_path = base_dir / report_name

    mindmap_path.write_text(_build_mermaid_mindmap(execution, nodes), encoding="utf-8")
    report_path.write_text(_build_markdown_report(execution, nodes), encoding="utf-8")

    mindmap_url = _try_upload_to_s3(mindmap_path, execution.id, mindmap_name)
    report_url = _try_upload_to_s3(report_path, execution.id, report_name)

    created_at = int(time.time() * 1000)
    return [
        ProjectReportArtifact(
            title="PFM Mindmap (Mermaid)",
            filename=mindmap_name,
            format="mmd",
            created_at=created_at,
            url=mindmap_url or f"/v1/projects/executions/{execution.id}/reports/{mindmap_name}",
        ),
        ProjectReportArtifact(
            title="PFM Report (Markdown)",
            filename=report_name,
            format="md",
            created_at=created_at,
            url=report_url or f"/v1/projects/executions/{execution.id}/reports/{report_name}",
        ),
    ]
