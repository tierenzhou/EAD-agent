"""
Normalize and validate per-node EAD report markdown (operator + agent paths).
"""

from __future__ import annotations

import re
from typing import Any, Dict, Optional

_NODE_REPORT_REPLY_RE = re.compile(
    r"\[Node-Report-Reply-To:\s*[^\]]+\]\s*",
    re.IGNORECASE,
)
_NODE_SUMMARY_RE = re.compile(r"^node\s+summary\s*:\s*$", re.IGNORECASE | re.MULTILINE)
_MIN_REPORT_CHARS = 80


def normalize_node_report_markdown(raw: str) -> str:
    """
    Strip reply markers and conversational preamble before the standard body.
    """
    text = _NODE_REPORT_REPLY_RE.sub("", str(raw or "")).replace("\r\n", "\n").strip()
    if not text:
        return ""

    match = _NODE_SUMMARY_RE.search(text)
    if match and match.start() > 0:
        text = text[match.start() :].strip()

    # Drop leading filler lines before Node Summary when marker search failed.
    lines = text.split("\n")
    for i, line in enumerate(lines):
        if re.match(r"^node\s+summary\s*:\s*$", line.strip(), re.IGNORECASE):
            if i > 0:
                text = "\n".join(lines[i:]).strip()
            break

    return text.strip()


def is_valid_node_report_markdown(markdown: str) -> bool:
    """
    True when content matches the standard node report template (relaxed for
    summary-only reports that are still substantive).
    """
    text = normalize_node_report_markdown(markdown)
    if len(text) < _MIN_REPORT_CHARS:
        return False

    lower = text.lower()
    if "node summary" not in lower:
        return False

    has_features_and_tests = (
        "features:" in lower
        and ("test case" in lower or re.search(r"feature\s+f-\d+", lower))
    )
    has_explore = "explore and improve" in lower and (
        "what we documented well:" in lower
        or "gaps and open questions:" in lower
        or "recommended next exploration:" in lower
        or "how to improve this report next time:" in lower
    )
    has_substantive_summary = (
        "purpose:" in lower
        and "in-scope behaviors:" in lower
        and any(line.strip().startswith("-") for line in text.splitlines())
    )

    return has_features_and_tests or has_explore or has_substantive_summary


_PFM_NODE_KEY_IN_REQUEST_RE = re.compile(
    r"PFM node key:\s*([^\s.`]+)",
    re.IGNORECASE,
)


def backfill_node_reports_from_progress_log(store: Any, execution_id: str) -> int:
    """
    Save valid node reports found in progress_log when not already in DB as rich content.
    Pairs ``PFM node key:`` user prompts with the next valid assistant report body.
    """
    from .pfm_fmr_parse import is_materialized_stub_markdown

    execution = store.get_execution(execution_id)
    if not execution:
        return 0

    existing: Dict[str, str] = {}
    for art in store.list_execution_pfm_artifacts(execution_id) or []:
        if str(art.get("artifact_type") or "") != "node_ead_report":
            continue
        nk = str(art.get("node_key") or "").strip()
        if nk:
            existing[nk] = str(art.get("content") or art.get("markdown") or "")

    updated = 0
    pending_key: Optional[str] = None

    for entry in execution.progress_log or []:
        text = str(getattr(entry, "text", None) or "").strip()
        if not text:
            continue

        key_m = _PFM_NODE_KEY_IN_REQUEST_RE.search(text)
        if key_m:
            pending_key = key_m.group(1).strip()
            continue

        if not pending_key:
            continue

        md = normalize_node_report_markdown(text)
        if not is_valid_node_report_markdown(md):
            continue

        prev = existing.get(pending_key, "")
        if prev and not is_materialized_stub_markdown(prev):
            pending_key = None
            continue

        store.save_node_ead_report_artifact(
            execution_id,
            node_key=pending_key,
            title=pending_key,
            content=md,
        )
        existing[pending_key] = md
        updated += 1
        pending_key = None

    return updated
