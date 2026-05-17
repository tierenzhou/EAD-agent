"""
Parse agent-delivered ``*.FMR`` portable reports into per-node markdown.

FMR layout is produced by :func:`projects.pfm_deliverables._build_fmr_markdown`.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, Optional

from .pfm_delivery import _canonical_delivery_paths

_MATERIALIZED_STUB_MARKER = "materialized from run data"
_FMR_PLACEHOLDER_PREFIX = "No saved Node EAD report exists"


def is_materialized_stub_markdown(markdown: str) -> bool:
    md = str(markdown or "").strip()
    if not md:
        return True
    return _MATERIALIZED_STUB_MARKER in md and "commit_pfm_snapshot" in md


def is_fmr_placeholder_report(markdown: str) -> bool:
    text = str(markdown or "").strip()
    if not text:
        return True
    return text.startswith(_FMR_PLACEHOLDER_PREFIX)


def parse_fmr_node_reports(fmr_text: str) -> Dict[str, Dict[str, str]]:
    """
    Return ``node_key -> {title, markdown}`` from a full ``.FMR`` file body.
    """
    text = str(fmr_text or "")
    if "## Per-Node EAD Reports" not in text:
        return {}

    body = text.split("## Per-Node EAD Reports", 1)[1]
    out: Dict[str, Dict[str, str]] = {}
    for chunk in re.split(r"\n### Node \d+:", body):
        block = chunk.strip()
        if not block:
            continue
        title_line, _, rest = block.partition("\n")
        title = title_line.strip() or ""
        key_m = re.search(r"^- Node Key:\s*`([^`]+)`", rest, re.MULTILINE)
        if not key_m:
            continue
        node_key = key_m.group(1).strip()
        report_m = re.search(r"#### Node EAD Report\n(.*)\Z", rest, re.DOTALL)
        if not report_m:
            continue
        from .pfm_node_report_content import normalize_node_report_markdown

        markdown = normalize_node_report_markdown(report_m.group(1))
        if is_fmr_placeholder_report(markdown):
            continue
        out[node_key] = {
            "title": title or node_key,
            "markdown": markdown,
        }
    return out


def read_canonical_fmr_text(execution_id: str) -> Optional[str]:
    eid = str(execution_id or "").strip()
    if not eid:
        return None
    for path in _canonical_delivery_paths(eid):
        if path.suffix.lower() != ".fmr":
            continue
        try:
            return path.read_text(encoding="utf-8")
        except OSError:
            continue
    return None


def load_canonical_fmr_reports(execution_id: str) -> Dict[str, Dict[str, str]]:
    text = read_canonical_fmr_text(execution_id)
    if not text:
        return {}
    return parse_fmr_node_reports(text)


def merge_fmr_reports_into_library(
    library: Dict[str, Dict[str, str]],
    fmr_reports: Dict[str, Dict[str, str]],
) -> None:
    """Prefer existing rich DB reports; replace materialized stubs with FMR content."""
    for node_key, entry in fmr_reports.items():
        nk = str(node_key or "").strip()
        md = str(entry.get("markdown") or "").strip()
        title = str(entry.get("title") or nk).strip()
        if not nk or not md or is_fmr_placeholder_report(md):
            continue
        prev = library.get(nk)
        if prev:
            prev_md = str(prev.get("markdown") or "").strip()
            if prev_md and not is_materialized_stub_markdown(prev_md):
                continue
        library[nk] = {"title": title, "markdown": md}


def ensure_node_reports_from_agent_delivery(store: Any, execution_id: str) -> int:
    """
    Backfill ``node_ead_report`` rows from canonical ``.FMR`` on disk.

    Safe to call on every tree/node-report read; only overwrites missing rows or
    operator materialized stubs (never rich agent-authored DB reports).
    """
    if not str(execution_id or "").strip():
        return 0
    if hasattr(store, "has_committed_pfm_tree") and not store.has_committed_pfm_tree(execution_id):
        return 0
    return sync_node_reports_from_canonical_fmr(store, execution_id)


def sync_node_reports_from_canonical_fmr(store: Any, execution_id: str) -> int:
    """
    Upsert ``node_ead_report`` rows from canonical ``.FMR`` when missing or materialized stubs.
    Returns the number of artifacts written/updated.
    """
    reports = load_canonical_fmr_reports(execution_id)
    if not reports:
        return 0

    existing: Dict[str, str] = {}
    for art in store.list_execution_pfm_artifacts(execution_id) or []:
        if str(art.get("artifact_type") or "") != "node_ead_report":
            continue
        nk = str(art.get("node_key") or "").strip()
        if nk:
            existing[nk] = str(art.get("content") or art.get("markdown") or "")

    updated = 0
    for nk, entry in reports.items():
        from .pfm_node_report_content import normalize_node_report_markdown

        md = normalize_node_report_markdown(str(entry.get("markdown") or ""))
        if not md:
            continue
        prev = existing.get(nk, "")
        if prev and not is_materialized_stub_markdown(prev):
            continue
        store.save_node_ead_report_artifact(
            execution_id,
            node_key=nk,
            title=str(entry.get("title") or nk),
            content=md,
        )
        updated += 1
    return updated
