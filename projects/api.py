"""
Project API handlers for the Hermes API server.

Provides REST endpoints for EAD project template and execution management,
mirroring the RPC methods from EAD-EXP's gateway.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import logging
import mimetypes
import os
import re
import shutil
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

_RUN_PURPOSE_ALLOWED = frozenset({"live_app_learning", "live_app_testing", "document_analysis"})
_EVIDENCE_SOURCE_ALLOWED = frozenset({"live_app", "document", "hybrid"})

# Executor progress extraction skips messages that start with this marker.
EAD_RUN_ACK_MARKER = "[EAD-RUN-ACK:v1]"

_EXPLORE_TYPE_LABELS: Dict[str, str] = {
    "pfm-full": "Full PFM map (features and test cases)",
    "pfm-spec": "PFM map with features and functional specs",
    "website": "Website exploration and content gathering",
    "other": "Custom exploration goal",
}


def _explore_type_label(explore_type: Optional[str]) -> str:
    key = (explore_type or "").strip()
    if not key:
        return "PFM discovery (default)"
    return _EXPLORE_TYPE_LABELS.get(key, key)


def _coerce_request_bool(value: Any) -> Optional[bool]:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value in (0, 1):
        return bool(int(value))
    if isinstance(value, str):
        v = value.strip().lower()
        if v in ("true", "1", "yes", "on"):
            return True
        if v in ("false", "0", "no", "off"):
            return False
    return None


def _parse_execution_learning_metadata(body: Dict[str, Any]) -> Union[Dict[str, Any], str]:
    """Build ProjectExecute learning fields from POST body. On error returns a message string."""
    rp = body.get("run_purpose")
    if rp is not None:
        rp = str(rp).strip()
        if rp not in _RUN_PURPOSE_ALLOWED:
            return f"Invalid run_purpose: {rp!r} (expected one of {sorted(_RUN_PURPOSE_ALLOWED)})"
    else:
        rp = "live_app_learning"

    contrib = _coerce_request_bool(body.get("contributes_to_learning"))
    if contrib is None:
        contrib = rp != "live_app_testing"

    valid_reporting_training = _coerce_request_bool(body.get("valid_for_data_reporting_training"))
    if valid_reporting_training is None:
        valid_reporting_training = True
    if valid_reporting_training is False:
        contrib = False

    es = body.get("evidence_source")
    if es is not None:
        es = str(es).strip()
        if es not in _EVIDENCE_SOURCE_ALLOWED:
            return f"Invalid evidence_source: {es!r} (expected one of {sorted(_EVIDENCE_SOURCE_ALLOWED)})"
    else:
        es = "document" if rp == "document_analysis" else "live_app"

    reason_raw = body.get("learning_exclusion_reason")
    reason: Optional[str] = None
    if reason_raw is not None:
        s = str(reason_raw).strip()
        if s:
            reason = s[:500]

    return {
        "run_purpose": rp,
        "contributes_to_learning": contrib,
        "evidence_source": es,
        "learning_exclusion_reason": reason,
        "valid_for_data_reporting_training": valid_reporting_training,
        "invalid_for_data_reporting_training_reason": reason if valid_reporting_training is False else None,
    }


def _execution_ack_message(
    execution: ProjectExecute,
    *,
    template: Optional[ProjectTemplate] = None,
) -> str:
    purpose = execution.run_purpose or "live_app_learning"
    evidence = execution.evidence_source or "live_app"
    learning = (
        "will contribute to template learning"
        if execution.contributes_to_learning is not False
        else "will not update the template learning index"
    )
    target = (execution.target_url or "").strip() or "the configured target"
    run_name = (execution.name or "this Explore run").strip()
    template_name = (template.name if template else "").strip()
    about = (execution.description or "").strip()
    if not about and template is not None:
        about = (template.description or "").strip()
    explore = _explore_type_label(execution.explore_type or (template.explore_type if template else None))
    budget = execution.time_budget_minutes
    budget_line = f"{int(budget)} minutes" if budget else "not set"
    about_block = f"\n### What this run is about\n\n{about}\n\n" if about else ""
    template_line = f"- Project template: **{template_name}**\n" if template_name else ""
    return (
        f"{EAD_RUN_ACK_MARKER}\n\n"
        f"## Run Introduction\n\n"
        f"I have received the assignment for **{run_name}** and I am starting now.\n\n"
        f"{about_block}"
        f"{template_line}"
        f"- Explore focus: {explore}\n"
        f"- Purpose: `{purpose}`\n"
        f"- Evidence focus: `{evidence}`\n"
        f"- Learning mode: {learning}\n"
        f"- Time budget: {budget_line}\n"
        f"- Target: {target}\n\n"
        "First I will complete the login/initialization phase. After login is confirmed, "
        "I will begin PFM discovery and report meaningful milestones with evidence.\n\n"
        "I will wait for your messages in chat when you need to steer login or priorities. "
        "Until then I will work through initialization and discovery on the schedule above.\n\n"
        "Login evidence: I will attach at least 3 screenshots (site URL identity, login step, post-login view) "
        "in report_running_step.thumbnail_urls before declaring login success."
    )


def _execution_boundary_message(execution: ProjectExecute) -> str:
    target = (execution.target_url or "").strip() or "the configured target URL"
    return "\n".join(
        [
            "HARD PROJECT TEMPLATE BOUNDARY:",
            f"- Execution ID: {execution.id}.",
            f"- Template ID: {execution.linked_template_id}.",
            f"- Run name: {execution.name or execution.id}.",
            f"- Target URL: {target}.",
            "- Each project template is independent from all other templates.",
            "- Do not use global memory, another template's runs, another session, or another target as evidence.",
            "- Use only this run's live observations plus explicitly injected same-template learning context.",
            "- Start from the Target URL. Do not navigate to remembered URLs or subdomains unless the Target URL itself redirects there.",
            "- If remembered information conflicts with this boundary, ignore the remembered information and report only live evidence.",
        ]
    )


def _template_isolation_block(template: ProjectTemplate, target_url: str) -> str:
    return "\n".join(
        [
            "## Template Isolation Rules",
            f"- Current template ID: `{template.id}`.",
            f"- Current template name: `{template.name}`.",
            f"- Current target URL: `{target_url}`.",
            "- Hard rule: no global knowledge and no cross-template memory. Each project template is independent.",
            "- Do not use facts from other project templates, previous runs, or persistent memory as current evidence unless they appear in the Template Learning Context above and belong to this exact template.",
            "- If persistent memory mentions a different target, subdomain, template, or application, ignore it unless this run rediscovers it from the current target.",
            "- Start from the current target URL. If the browser redirects to a different domain or subdomain, report the redirect as a fresh observation; do not assume it means this run is another project.",
            "- Build the PFM and node EAD reports from evidence collected in this run plus same-template inherited context only.",
        ]
    )


def _assemble_execution_ai_prompt(
    store: ProjectStore,
    template: ProjectTemplate,
    user_prompt: str,
    target_url: str,
    *,
    include_learning_context: bool,
) -> str:
    """Build execution ai_prompt. Learning context is deferred to bootstrap when include_learning_context=False."""
    base_ai_prompt = str(user_prompt or template.ai_prompt or "").strip()
    if include_learning_context:
        inherited_context = store.build_template_learning_context(template.id)
        if inherited_context:
            base_ai_prompt = "\n\n".join(
                [
                    base_ai_prompt,
                    "## Template Learning Context",
                    (
                        "This same-template PFM context is locked until login succeeds. "
                        "Do not read, summarize, mention, display, or rely on it before the login checkpoint "
                        "reports login_phase_status=success."
                    ),
                    inherited_context,
                    (
                        "Instruction after login only: reuse these prior PFM mindmap/node-report findings "
                        "as a starting hypothesis, then improve or correct them with new evidence."
                    ),
                ]
            ).strip()
    return "\n\n".join(
        [
            part
            for part in [base_ai_prompt, _template_isolation_block(template, target_url)]
            if part
        ]
    ).strip()


def enrich_execution_ai_prompt_with_learning(store: ProjectStore, execution_id: str) -> None:
    """Append template learning context during deferred bootstrap (not on POST /executions/run)."""
    execution = store.get_execution(execution_id)
    if not execution:
        return
    template = store.get_template(execution.linked_template_id)
    if not template:
        return
    target_url = str(execution.target_url or template.target_url or "").strip()
    user_prompt = str(execution.ai_prompt or template.ai_prompt or "").strip()
    full_prompt = _assemble_execution_ai_prompt(
        store,
        template,
        user_prompt,
        target_url,
        include_learning_context=True,
    )
    store.update_execution(execution_id, ai_prompt=full_prompt)


def _execution_list_item_dict(
    execution: ProjectExecute | Dict[str, Any],
    store: Optional["ProjectStore"] = None,
) -> Dict[str, Any]:
    """Lightweight execution row for list API (avoids shipping full progress logs for every run)."""
    # Avoid serializing then discarding large nested payloads (progress_log/results),
    # which can block the gateway main loop on large/active runs.
    if isinstance(execution, dict):
        raw = dict(execution)
    else:
        raw = execution.model_dump(mode="json", exclude={"progress_log", "results"})
    raw["progress_log"] = []
    raw["results"] = []
    prompt = str(raw.get("ai_prompt") or "")
    if len(prompt) > 800:
        raw["ai_prompt"] = prompt[:800] + "\n...(truncated in list view)"
    if store is not None and not isinstance(execution, dict):
        raw.update(store.resolve_pfm_lineage_context(execution))
    return _to_camel_dict(raw)


def _execution_detail_dict(store: "ProjectStore", execution: ProjectExecute) -> Dict[str, Any]:
    """Full execution payload for GET + UI, including PFM lineage flags."""
    raw = json.loads(execution.model_dump_json())
    raw.update(store.resolve_pfm_lineage_context(execution))
    return _to_camel_dict(raw)

try:
    from aiohttp import web

    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False
    web = None

from projects.models import (
    ExecutionStatus,
    ProgressLogEntry,
    ProjectAuthMode,
    ProjectExecute,
    ProjectTemplate,
    ReportingActivityStatus,
)
from projects.store import ProjectStore
from projects.pfm_artifacts import report_file_path, resolve_pfm_nodes_for_mindmap
from projects.pfm_tree import (
    PFM_HIERARCHY_READABILITY_RULES,
    PFM_NODE_TITLE_RULES,
    PFM_PARENT_KEY_RULES,
)

logger = logging.getLogger(__name__)
_IMAGE_PATH_RE = re.compile(r"(?:MEDIA:)?(?P<path>/[^\s`\"']+\.(?:png|jpe?g|gif|webp))", re.IGNORECASE)
_IMAGE_BASENAME_RE = re.compile(
    r"\b(?P<path>[\w.-]*?(?:screenshot|screen|image)[\w.-]*?\.(?:png|jpe?g|gif|webp))\b",
    re.IGNORECASE,
)
_IMAGE_EXT_RE = re.compile(r"\.(png|jpe?g|gif|webp)(?:$|[?#])", re.IGNORECASE)
_MIN_BROWSER_SCREENSHOT_BYTES = 10_000

# Align with EAD_ExpUI src/ui/pfm-mindmap-remote-cache.ts
_PFM_MINDMAP_REMOTE_CACHE_MARKER = "[PFM-MINDMAP-CACHE:v1]"
_PFM_PHASE1_BASELINE_MARKER = "[PFM-PHASE1-CANONICAL-BASELINE:v1]"


def _build_phase1_canonical_baseline_message(
    store: ProjectStore,
    execution: ProjectExecute,
) -> Optional[str]:
    """
    One authoritative Phase-1 baseline for agents: persisted PFM nodes, mindmap excerpt,
    and per-node EAD reports so Initialization (post-login) can reconcile accurately.
    """
    nodes = resolve_pfm_nodes_for_mindmap(execution)
    artifacts = store.list_execution_pfm_artifacts(execution.id)
    committed_tree = store.get_committed_pfm_tree(execution.id)

    mind_excerpt = ""
    for art in artifacts:
        if str(art.get("artifact_type") or "") == "pfm_mindmap":
            mc = str(art.get("content") or "").strip()
            if mc:
                me = mc[:4500]
                if len(mc) > 4500:
                    me += "\n... [mindmap truncated] ..."
                mind_excerpt = me
                break

    ereports_lines: List[str] = []
    for art in artifacts:
        if str(art.get("artifact_type") or "") != "node_ead_report":
            continue
        title = str(art.get("title") or art.get("artifact_key") or "EAD report").strip()
        nk = str(art.get("node_key") or "").strip()
        raw_ex = art.get("excerpt") if art.get("excerpt") is not None else art.get("content")
        excerpt = str(raw_ex or "")[:900].strip().replace("\n", " ")
        ereports_lines.append(f"- **{title}** (`node_key`: `{nk or 'n/a'}`): {excerpt}")

    if not nodes and not mind_excerpt and not ereports_lines:
        return None

    chunks: List[str] = [
        _PFM_PHASE1_BASELINE_MARKER,
        "",
        f"- **Execution ID:** `{execution.id}`",
    ]
    if isinstance(committed_tree, dict):
        tree_version = int(committed_tree.get("version") or 0)
        generated_at_ms = int(committed_tree.get("generated_at") or 0)
        chunks.append(f"- **PFM tree version:** `v{tree_version}`")
        if generated_at_ms:
            chunks.append(f"- **PFM tree generated_at (ms):** `{generated_at_ms}`")
    else:
        chunks.append("- **PFM tree version:** `none` (no committed snapshot yet)")
    if execution.inherited_from_execution_id:
        chunks.append(
            f"- **Inherited / seeded from execution:** `{execution.inherited_from_execution_id}`"
        )
    chunks.extend(
        [
            "",
            "**Phase 1 (post-login Initialization) commitments:**",
            "- This snapshot is authoritative **prior** product-structure state for this Execution ID ",
            "(not live evidence yet). Confirm with targeted navigation and screenshots.",
            "- Call **read_ead_execution** for this Execution ID and reconcile counters, artifact links, ",
            "and `results[]` against this baseline and the listed Node EAD reports.",
            "- In Initialization step 2, summarize this baseline as a concise mindmap-style outline ",
            "(including coverage of **each** EAD note report listed here), cite gaps vs the live ",
            "target URL, then set `initialization_review_status=success` and `ready_to_explore=true`.",
            "",
            PFM_HIERARCHY_READABILITY_RULES,
            PFM_PARENT_KEY_RULES,
            PFM_NODE_TITLE_RULES,
            "",
        ]
    )
    if nodes:
        chunks.append(f"### Canonical PFM nodes ({len(nodes)})")
        chunks.append("| # | Title | Node key | Status |")
        chunks.append("|---|--------|----------|--------|")
        for i, n in enumerate(nodes[:120], start=1):
            t = str(n.title or "").replace("|", " ").strip()[:120]
            k = str(n.node_key or "").replace("|", " ").strip()[:100]
            st = getattr(n.status, "value", None) if n.status is not None else n.status
            st_txt = str(st if st is not None else "").replace("|", " ")
            chunks.append(f"| {i} | {t} | `{k}` | {st_txt} |")
        if len(nodes) > 120:
            chunks.append("")
            chunks.append(f"... truncated: {len(nodes) - 120} more nodes omitted from snapshot.")
        chunks.append("")
    if ereports_lines:
        chunks.append("### Persisted Node EAD reports (must appear in Initialization outline)")
        chunks.extend(ereports_lines[:40])
        if len(ereports_lines) > 40:
            chunks.append(f"... +{len(ereports_lines) - 40} more reports not listed here.")
        chunks.append("")
    if mind_excerpt:
        chunks.append("### Persisted Mermaid mindmap excerpt (structure reference)")
        chunks.append("```")
        chunks.append(mind_excerpt)
        chunks.append("```")

    body = "\n".join(chunks).strip()
    max_len = 14_800
    if len(body) > max_len:
        body = body[: max_len - 40].rstrip() + "\n\n... [snapshot truncated]"
    return body


def _try_inject_inherited_pfm_mindmap_cache(
    new_session_id: str,
    new_run_session_key: str,
    source_execution_id: str,
) -> bool:
    """Copy latest PFM mindmap cache system message from source run's chat into the new session."""
    try:
        from hermes_state import SessionDB
    except ImportError:
        return False

    db = SessionDB()
    source_title = f"eadproj-exec-{source_execution_id}"
    src_sid = db.resolve_session_by_title(source_title)
    if not src_sid:
        return False

    messages = db.get_messages(src_sid)
    for msg in reversed(messages):
        if (msg.get("role") or "").lower() != "system":
            continue
        content = str(msg.get("content") or "").strip()
        if not content.startswith(_PFM_MINDMAP_REMOTE_CACHE_MARKER):
            continue
        json_part = content[len(_PFM_MINDMAP_REMOTE_CACHE_MARKER) :].strip()
        if not json_part:
            continue
        try:
            envelope = json.loads(json_part)
        except json.JSONDecodeError:
            continue
        if not isinstance(envelope, dict) or envelope.get("cleared") is True:
            continue
        mm = envelope.get("mindmap")
        if not isinstance(mm, dict):
            continue
        mm["sessionKey"] = new_run_session_key
        envelope["cacheKey"] = new_run_session_key
        envelope["savedAt"] = int(time.time() * 1000)
        new_content = _PFM_MINDMAP_REMOTE_CACHE_MARKER + "\n" + json.dumps(
            envelope, ensure_ascii=False
        )
        db.append_message(new_session_id, "system", new_content)
        return True
    return False


def _hermes_home() -> Path:
    configured = os.getenv("HERMES_HOME", "").strip()
    return Path(configured).expanduser() if configured else Path.home() / ".hermes"


def _resolve_local_image_path(path_text: str) -> Path:
    src = Path(path_text).expanduser()
    try:
        if src.exists():
            return src
    except OSError:
        return src
    if src.is_absolute() or src.name != path_text:
        return src
    for directory in (
        _hermes_home() / "cache" / "screenshots",
        _hermes_home() / "browser_screenshots",
    ):
        candidate = directory / src.name
        try:
            if candidate.exists():
                return candidate
        except OSError:
            continue
    return src


def _looks_like_image_ref(value: str) -> bool:
    return bool(_IMAGE_EXT_RE.search(str(value or "").strip()))


def _is_usable_local_image(src: Path) -> bool:
    if not src.exists() or not src.is_file():
        return False
    if not _looks_like_image_ref(src.name):
        return False
    try:
        size = src.stat().st_size
    except OSError:
        return False
    if "browser_screenshot_" in src.name and size < _MIN_BROWSER_SCREENSHOT_BYTES:
        return False
    return True


def _report_url_to_file(execution_id: str, url: str) -> Optional[Path]:
    marker = f"/v1/projects/executions/{execution_id}/reports/"
    text = str(url or "")
    if marker not in text:
        return None
    filename = text.split(marker, 1)[1].split("?", 1)[0].split("#", 1)[0]
    if not filename:
        return None
    return report_file_path(execution_id, filename)


def _entry_has_unusable_report_image(execution_id: str, entry: Any) -> bool:
    if not isinstance(entry, dict):
        return False
    for key in ("image_url", "imageUrl", "thumbnail_url", "thumbnailUrl"):
        path = _report_url_to_file(execution_id, str(entry.get(key) or ""))
        if path and not _is_usable_local_image(path):
            return True
    return False


def _content_addressed_report_name(src: Path) -> Optional[str]:
    try:
        digest = hashlib.sha256(src.read_bytes()).hexdigest()[:16]
    except OSError:
        return None
    suffix = src.suffix.lower()
    if not suffix or not re.fullmatch(r"\.[a-z0-9]+", suffix):
        suffix = ".png"
    return f"live-shot-{digest}{suffix}"


def _json_response(data: Any, status: int = 200) -> "web.Response":
    return web.json_response(data, status=status)


def _to_snake_dict(data: Dict[str, Any]) -> Dict[str, Any]:
    """Convert camelCase dict keys to snake_case for Pydantic model compatibility."""
    out: Dict[str, Any] = {}
    for k, v in data.items():
        import re

        snake = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", k).lower()
        if isinstance(v, dict):
            out[snake] = _to_snake_dict(v)
        elif isinstance(v, list):
            out[snake] = [_to_snake_dict(item) if isinstance(item, dict) else item for item in v]
        else:
            out[snake] = v
    return out


def _error_response(message: str, status: int = 400, code: str = "bad_request") -> "web.Response":
    return web.json_response({"error": {"message": message, "code": code}}, status=status)


def _to_camel_dict(data: Dict[str, Any]) -> Dict[str, Any]:
    """Convert snake_case dict keys to camelCase for UI compatibility."""
    out: Dict[str, Any] = {}
    for k, v in data.items():
        components = k.split("_")
        camel = components[0] + "".join(w.capitalize() for w in components[1:])
        if isinstance(v, dict):
            out[camel] = _to_camel_dict(v)
        elif isinstance(v, list):
            out[camel] = [_to_camel_dict(item) if isinstance(item, dict) else item for item in v]
        else:
            out[camel] = v
    return out


def _materialize_local_image_as_report(execution_id: str, image_path: str) -> Optional[str]:
    path_text = str(image_path or "").strip()
    if not path_text:
        return None
    if (
        "\n" in path_text
        or "\r" in path_text
        or "\\n" in path_text
        or "\\r" in path_text
        or len(path_text) > 2048
    ):
        return None
    src = _resolve_local_image_path(path_text)
    if not _is_usable_local_image(src):
        return None
    target_name = _content_addressed_report_name(src)
    if not target_name:
        return None
    target = report_file_path(execution_id, target_name)
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            shutil.copy2(src, target)
    except Exception:
        return None
    return f"/v1/projects/executions/{execution_id}/reports/{target_name}"


def _extract_media_refs(content: Any) -> list[str]:
    blobs: list[str] = []
    if isinstance(content, str):
        blobs.append(content)
    elif content is not None:
        try:
            blobs.append(json.dumps(content, ensure_ascii=False))
        except Exception:
            return []
    refs: list[str] = []
    for blob in blobs:
        for pattern in (_IMAGE_PATH_RE, _IMAGE_BASENAME_RE):
            for match in pattern.finditer(blob):
                candidate = (match.group("path") or "").strip().strip("`'\"),.;")
                if candidate:
                    refs.append(candidate)
    return refs


def _image_basename(value: str) -> str:
    match = _IMAGE_BASENAME_RE.search(str(value or ""))
    return match.group("path") if match else Path(str(value or "")).name


def _has_image_named(progress_log: list[Any], name: str) -> bool:
    if not name:
        return False
    for entry in progress_log:
        if not isinstance(entry, dict):
            continue
        for key in ("image_url", "imageUrl", "thumbnail_url", "thumbnailUrl"):
            if name in str(entry.get(key) or ""):
                return True
    return False


def _backfill_execution_screenshots(raw: Dict[str, Any]) -> Dict[str, Any]:
    progress_log = raw.get("progress_log") or []
    execution_id = str(raw.get("id") or "").strip()
    run_session_key = str(raw.get("run_session_key") or "").strip()
    if execution_id and progress_log:
        filtered_log = [
            entry for entry in progress_log
            if not _entry_has_unusable_report_image(execution_id, entry)
        ]
        if len(filtered_log) != len(progress_log):
            progress_log = filtered_log
            raw["progress_log"] = progress_log

    if not execution_id or not run_session_key:
        return raw

    try:
        from hermes_state import SessionDB

        db = SessionDB()
        session = db.get_session_by_title(run_session_key)
        if not session or not session.get("id"):
            return raw
        messages = db.get_messages(session["id"]) or []
    except Exception:
        return raw

    seen_names: set[str] = set()
    recovered_urls: list[str] = []
    for msg in messages:
        refs = _extract_media_refs(msg.get("content"))
        for ref in refs:
            name = _image_basename(ref)
            if not name or name in seen_names or _has_image_named(progress_log, name):
                continue
            seen_names.add(name)
            url = _materialize_local_image_as_report(execution_id, ref)
            if not url:
                continue
            recovered_urls.append(url)

    if not recovered_urls:
        return raw

    ts = time.time()
    for url in recovered_urls[-30:]:
        progress_log.append(
            {
                "ts": ts,
                "kind": "assistant",
                "text": "Recovered screenshot",
                "thumbnail_url": url,
                "image_url": url,
            }
        )
        ts += 0.001
    raw["progress_log"] = progress_log
    return raw


def _template_json(template) -> "web.Response":
    """Serialize a ProjectTemplate with camelCase keys for the UI."""
    raw = json.loads(template.model_dump_json())
    return _json_response(_to_camel_dict(raw))


def _templates_json(templates) -> "web.Response":
    """Serialize a list of ProjectTemplates with camelCase keys for the UI."""
    raw_list = [json.loads(t.model_dump_json()) for t in templates]
    return _json_response(
        {"templates": [_to_camel_dict(t) for t in raw_list], "activeTemplateId": None}
    )


class ProjectHandlers:
    """Handles /v1/projects/* API endpoints."""

    def __init__(self, store: Optional[ProjectStore] = None, executor=None):
        self._store = store or ProjectStore()
        self._executor = executor
        self._fmr_sync_inflight: set[str] = set()

    def _schedule_execution_bootstrap(self, execution_id: str) -> None:
        from projects.run_bootstrap import run_execution_bootstrap

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            logger.error("[projects] No running event loop; cannot bootstrap execution %s", execution_id)
            self._store.update_execution(
                execution_id,
                status=ExecutionStatus.FAILED,
                last_error_message="Internal error: no event loop for deferred run bootstrap.",
                executor_hint="AI Failed",
                bootstrap_pending=False,
            )
            return
        loop.create_task(run_execution_bootstrap(self._store, self._executor, execution_id))

    def _schedule_fmr_node_report_sync(self, execution_id: str) -> None:
        eid = str(execution_id or "").strip()
        if not eid or eid in self._fmr_sync_inflight:
            return
        self._fmr_sync_inflight.add(eid)
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            self._fmr_sync_inflight.discard(eid)
            return

        async def _runner() -> None:
            try:
                from .pfm_fmr_parse import sync_node_reports_from_canonical_fmr

                updated = await asyncio.to_thread(sync_node_reports_from_canonical_fmr, self._store, eid)
                if updated > 0:
                    logger.info("[projects] Background FMR node-report sync updated %s rows for %s", updated, eid)
            except Exception:
                logger.exception("[projects] Background FMR node-report sync failed for %s", eid)
            finally:
                self._fmr_sync_inflight.discard(eid)

        loop.create_task(_runner())

    # ------------------------------------------------------------------
    # Template endpoints
    # ------------------------------------------------------------------

    async def handle_list_templates(self, request: "web.Request") -> "web.Response":
        templates = self._store.list_templates()
        active_id = self._store.get_active_template_id()
        raw_list = [json.loads(t.model_dump_json()) for t in templates]
        return _json_response(
            {
                "templates": [_to_camel_dict(t) for t in raw_list],
                "activeTemplateId": active_id,
            }
        )

    async def handle_get_template(self, request: "web.Request") -> "web.Response":
        template_id = request.match_info["template_id"]
        template = self._store.get_template(template_id)
        if not template:
            return _error_response(f"Template {template_id} not found", 404, "not_found")
        return _template_json(template)

    async def handle_create_template(self, request: "web.Request") -> "web.Response":
        try:
            body = _to_snake_dict(await request.json())
        except Exception:
            return _error_response("Invalid JSON")

        try:
            template = ProjectTemplate(**body)
        except Exception as e:
            return _error_response(f"Invalid template data: {e}")

        created = self._store.create_template(template)
        return _template_json(created)

    async def handle_update_template(self, request: "web.Request") -> "web.Response":
        template_id = request.match_info["template_id"]
        try:
            body = _to_snake_dict(await request.json())
        except Exception:
            return _error_response("Invalid JSON")

        updated = self._store.update_template(template_id, **body)
        if not updated:
            return _error_response(f"Template {template_id} not found", 404, "not_found")
        return _template_json(updated)

    async def handle_delete_template(self, request: "web.Request") -> "web.Response":
        template_id = request.match_info["template_id"]
        deleted = self._store.delete_template(template_id)
        if not deleted:
            return _error_response(f"Template {template_id} not found", 404, "not_found")
        return _json_response({"deleted": True})

    async def handle_activate_template(self, request: "web.Request") -> "web.Response":
        template_id = request.match_info["template_id"]
        template = self._store.get_template(template_id)
        if not template:
            return _error_response(f"Template {template_id} not found", 404, "not_found")
        self._store.set_active_template_id(template_id)
        return _json_response({"active_template_id": template_id})

    # ------------------------------------------------------------------
    # Execution endpoints
    # ------------------------------------------------------------------

    async def _enforce_single_active_reporting_run(
        self,
        template_id: str,
        preferred_execution_id: str,
    ) -> None:
        if not template_id:
            return
        runs = list(self._store.list_executions(template_id=template_id) or [])
        for run in runs:
            if run.id == preferred_execution_id:
                continue
            raw_status = getattr(run, "status", "")
            status = str(getattr(raw_status, "value", raw_status) or "").strip().lower()
            if status in (ExecutionStatus.RUNNING.value, ExecutionStatus.PENDING.value):
                if self._executor:
                    await self._executor.cancel_execution(
                        run.id,
                        final_status=ExecutionStatus.CANCELLED,
                        operator_stop_kind="cancel",
                        cancel_reason=(
                            "Closed automatically: only one active run is allowed "
                            "per template for reporting and knowledge retrieval."
                        ),
                    )
                else:
                    self._store.update_execution(
                        run.id,
                        status=ExecutionStatus.CANCELLED,
                        paused=False,
                        operator_stop_kind="cancel",
                        cancel_reason=(
                            "Closed automatically: only one active run is allowed "
                            "per template for reporting and knowledge retrieval."
                        ),
                    )
            self._store.set_execution_reporting_activity(run.id, active=False)

        self._store.set_execution_reporting_activity(preferred_execution_id, active=True)

    async def handle_list_executions(self, request: "web.Request") -> "web.Response":
        template_id = request.query.get("template_id")
        status = request.query.get("status")
        execution_payloads = self._store.list_execution_payloads(
            template_id=template_id or None,
            status=status or None,
        )
        return _json_response(
            {
                "executions": [_execution_list_item_dict(e, None) for e in execution_payloads],
            }
        )

    async def handle_get_execution(self, request: "web.Request") -> "web.Response":
        execution_id = request.match_info["execution_id"]
        execution = self._store.get_execution(execution_id)
        if not execution:
            return _error_response(f"Execution {execution_id} not found", 404, "not_found")
        raw = json.loads(execution.model_dump_json())
        before_progress_len = len(raw.get("progress_log") or [])
        raw = _backfill_execution_screenshots(raw)
        after_progress = raw.get("progress_log") or []
        if len(after_progress) > before_progress_len:
            try:
                self._store.update_execution(
                    execution_id,
                    progress_log=[
                        ProgressLogEntry.model_validate(entry)
                        for entry in after_progress
                        if isinstance(entry, dict)
                    ],
                )
            except Exception as exc:
                logger.warning("[projects] Failed to persist recovered screenshots for %s: %s", execution_id, exc)
        return _json_response(_execution_detail_dict(self._store, execution))

    async def handle_run_execution(self, request: "web.Request") -> "web.Response":
        try:
            body = await request.json()
        except Exception:
            return _error_response("Invalid JSON")

        body = _to_snake_dict(body)

        template_id = body.get("template_id") or body.get("templateId")
        if not template_id:
            return _error_response("template_id is required")

        template = self._store.get_template(template_id)
        if not template:
            return _error_response(f"Template {template_id} not found", 404, "not_found")

        learning_meta = _parse_execution_learning_metadata(body)
        if isinstance(learning_meta, str):
            return _error_response(learning_meta)

        inherit_pfm = _coerce_request_bool(body.get("inherit_pfm"))
        if inherit_pfm is None:
            inherit_pfm = True

        raw_explicit = body.get("inherit_from_execution_id") or body.get("inheritFromExecutionId")
        explicit_source_id = str(raw_explicit).strip() if raw_explicit else None

        resolved_inherit_source: Optional[ProjectExecute] = None
        if inherit_pfm and explicit_source_id:
            resolved_inherit_source = self._store.resolve_pfm_inheritance_source(
                template.id,
                explicit_source_id=explicit_source_id,
            )
            if not resolved_inherit_source:
                return _error_response(
                    "inherit_from_execution_id was not found, is not for this template, "
                    "or is not eligible for PFM inheritance",
                    400,
                )

        target_url = str(body.get("target_url", template.target_url) or "").strip()
        user_prompt = str(body.get("ai_prompt", template.ai_prompt) or "").strip()
        # Fast POST: template learning context + isolation rules are applied in deferred bootstrap.
        base_ai_prompt = user_prompt

        execution = ProjectExecute(
            linked_template_id=template.id,
            name=body.get("name", f"Run - {template.name}"),
            description=body.get("description", template.description),
            target_url=body.get("target_url", template.target_url),
            ai_prompt=base_ai_prompt,
            explore_type=body.get("explore_type", template.explore_type),
            auth_mode=template.auth_mode,
            auth_login_url=template.auth_login_url,
            auth_session_profile=template.auth_session_profile,
            auth_instructions=template.auth_instructions,
            time_budget_minutes=body.get("time_budget_minutes", template.time_budget_minutes),
            cost_budget_dollars=body.get("cost_budget_dollars", template.cost_budget_dollars),
            show_local_browser=body.get("show_local_browser", False),
            status=ExecutionStatus.PENDING,
            start_time=int(time.time() * 1000),
            run_purpose=learning_meta["run_purpose"],
            contributes_to_learning=learning_meta["contributes_to_learning"],
            evidence_source=learning_meta["evidence_source"],
            learning_exclusion_reason=learning_meta["learning_exclusion_reason"],
            valid_for_data_reporting_training=learning_meta["valid_for_data_reporting_training"],
            invalid_for_data_reporting_training_reason=learning_meta[
                "invalid_for_data_reporting_training_reason"
            ],
            executor_hint="Preparing run (PFM inheritance, artifacts, and chat session)…",
            bootstrap_pending=True,
            bootstrap_inherit_pfm=inherit_pfm,
            bootstrap_explicit_inherit_from_execution_id=(
                explicit_source_id if inherit_pfm and explicit_source_id else None
            ),
        )

        created = self._store.create_execution(execution)
        await self._enforce_single_active_reporting_run(template.id, created.id)

        logger.info(
            "[projects] Created execution %s for template %s (inherit_pfm=%s); deferred bootstrap",
            created.id,
            template.id,
            inherit_pfm,
        )

        if self._executor:
            self._schedule_execution_bootstrap(created.id)
        else:
            logger.error(
                "[projects] ProjectExecutor not wired; execution %s will not start automatically",
                created.id,
            )

        raw = json.loads(created.model_dump_json())
        return _json_response(_to_camel_dict(raw), status=201)

    async def handle_set_reporting_activity(self, request: "web.Request") -> "web.Response":
        execution_id = request.match_info["execution_id"]
        execution = self._store.get_execution(execution_id)
        if not execution:
            return _error_response(f"Execution {execution_id} not found", 404, "not_found")
        try:
            body = _to_snake_dict(await request.json()) if request.can_read_body else {}
        except Exception:
            body = {}

        active_raw = body.get("active")
        status_raw = str(body.get("status") or "").strip().lower()
        if active_raw is None:
            if status_raw == ReportingActivityStatus.ACTIVE.value:
                active = True
            elif status_raw == ReportingActivityStatus.CLOSED.value:
                active = False
            else:
                return _error_response("Provide either active=true/false or status=active|closed")
        else:
            active = bool(active_raw)

        if not active:
            raw_status = getattr(execution, "status", "")
            status = str(getattr(raw_status, "value", raw_status) or "").strip().lower()
            if status in (ExecutionStatus.RUNNING.value, ExecutionStatus.PENDING.value):
                reason = (
                    "Closed by operator for reporting/knowledge retrieval. "
                    "Run auto-stopped to enforce single active run."
                )
                if self._executor:
                    await self._executor.cancel_execution(
                        execution_id,
                        final_status=ExecutionStatus.CANCELLED,
                        operator_stop_kind="cancel",
                        cancel_reason=reason,
                    )
                else:
                    self._store.update_execution(
                        execution_id,
                        status=ExecutionStatus.CANCELLED,
                        paused=False,
                        operator_stop_kind="cancel",
                        cancel_reason=reason,
                    )

        updated = self._store.set_execution_reporting_activity(execution_id, active=active)
        if not updated:
            return _error_response(f"Execution {execution_id} not found", 404, "not_found")

        template_active_id = self._store.get_active_reporting_execution_id(updated.linked_template_id)
        out = _to_camel_dict(json.loads(updated.model_dump_json()))
        out["activeReportingExecutionId"] = template_active_id
        return _json_response(out)

    async def handle_cancel_execution(self, request: "web.Request") -> "web.Response":
        execution_id = request.match_info["execution_id"]
        try:
            body = await request.json()
        except Exception:
            body = {}

        execution = self._store.get_execution(execution_id)
        if not execution:
            return _error_response(f"Execution {execution_id} not found", 404, "not_found")

        stop_kind = body.get("stop_kind", "cancel")
        cancel_reason = body.get("reason", "")

        final_status = (
            ExecutionStatus.COMPLETED if stop_kind == "finish" else ExecutionStatus.CANCELLED
        )
        duration_ms = int(time.time() * 1000) - (execution.start_time or int(time.time() * 1000))

        if self._executor:
            await self._executor.cancel_execution(
                execution_id,
                final_status=final_status,
                operator_stop_kind=stop_kind,
                cancel_reason=cancel_reason,
            )
            updated = self._store.update_execution(execution_id, duration_ms=duration_ms)
        else:
            updated = self._store.update_execution(
                execution_id,
                status=final_status,
                paused=False,
                operator_stop_kind=stop_kind,
                cancel_reason=cancel_reason,
                duration_ms=duration_ms,
            )

        logger.info("[projects] Cancelled execution %s (kind=%s)", execution_id, stop_kind)
        raw = json.loads(updated.model_dump_json())
        return _json_response(_to_camel_dict(raw))

    async def handle_delete_execution(self, request: "web.Request") -> "web.Response":
        execution_id = request.match_info["execution_id"]
        execution = self._store.get_execution(execution_id)
        if not execution:
            return _error_response(f"Execution {execution_id} not found", 404, "not_found")

        if execution.status == ExecutionStatus.RUNNING:
            if self._executor:
                await self._executor.cancel_execution(
                    execution_id,
                    final_status=ExecutionStatus.CANCELLED,
                    operator_stop_kind="cancel",
                    cancel_reason="Deleted by operator",
                )
            else:
                self._store.update_execution(
                    execution_id,
                    status=ExecutionStatus.CANCELLED,
                    paused=False,
                    operator_stop_kind="cancel",
                    cancel_reason="Deleted by operator",
                )

        deleted = self._store.delete_execution(execution_id)
        if not deleted:
            return _error_response(f"Execution {execution_id} not found", 404, "not_found")
        logger.info("[projects] Deleted execution %s", execution_id)
        return _json_response({"deleted": True, "execution_id": execution_id})

    async def handle_invalidate_execution(self, request: "web.Request") -> "web.Response":
        execution_id = request.match_info["execution_id"]
        try:
            body = await request.json()
        except Exception:
            body = {}

        execution = self._store.get_execution(execution_id)
        if not execution:
            return _error_response(f"Execution {execution_id} not found", 404, "not_found")

        reason = str(body.get("reason") or "Not valid for data reporting and training").strip()
        updated = self._store.invalidate_execution_learning(execution_id, reason=reason)
        if not updated:
            return _error_response(f"Execution {execution_id} not found", 404, "not_found")

        logger.info("[projects] Marked execution %s invalid for data reporting and training", execution_id)
        raw = json.loads(updated.model_dump_json())
        return _json_response(_to_camel_dict(raw))

    async def handle_get_execution_report(self, request: "web.Request") -> "web.Response":
        execution_id = request.match_info["execution_id"]
        filename = Path(request.match_info["filename"]).name
        execution = self._store.get_execution(execution_id)
        if not execution:
            return _error_response(f"Execution {execution_id} not found", 404, "not_found")

        target = report_file_path(execution_id, filename)
        if not target.exists() or not target.is_file():
            artifact = self._store.get_execution_pfm_artifact(execution_id, filename)
            if artifact:
                content_type, _ = mimetypes.guess_type(filename)
                if artifact.get("content_encoding") == "base64" and artifact.get("content_base64"):
                    try:
                        body = base64.b64decode(str(artifact.get("content_base64")))
                    except Exception:
                        body = b""
                    return web.Response(
                        body=body,
                        headers={"Content-Type": content_type or "application/octet-stream"},
                    )
                return web.Response(
                    text=str(artifact.get("content") or ""),
                    headers={"Content-Type": content_type or "text/plain"},
                )
            return _error_response(f"Report {filename} not found", 404, "not_found")

        content_type, _ = mimetypes.guess_type(str(target))
        return web.FileResponse(path=target, headers={"Content-Type": content_type or "text/plain"})

    async def handle_get_node_report_artifact(self, request: "web.Request") -> "web.Response":
        from .pfm_tree import node_report_artifact_key

        execution_id = request.match_info["execution_id"]
        execution = self._store.get_execution(execution_id)
        if not execution:
            return _error_response(f"Execution {execution_id} not found", 404, "not_found")
        node_key = (request.query.get("node_key") or request.query.get("nodeKey") or "").strip()
        if not node_key:
            return _error_response("node_key is required")
        artifact_key = node_report_artifact_key(node_key)
        artifact = self._store.get_execution_pfm_artifact(execution_id, artifact_key)
        snapshot = self._store.get_committed_pfm_tree(execution_id)
        tree_version = int((snapshot or {}).get("version") or 0)
        tree_generated_at = int((snapshot or {}).get("generated_at") or 0)
        if not artifact:
            return _json_response(
                {
                    "executionId": execution_id,
                    "nodeKey": node_key,
                    "artifact": None,
                    "pfmTreeVersion": tree_version,
                    "pfmTreeGeneratedAt": tree_generated_at,
                }
            )
        return _json_response(
            {
                "executionId": execution_id,
                "nodeKey": node_key,
                "artifact": _to_camel_dict(artifact),
                "pfmTreeVersion": tree_version,
                "pfmTreeGeneratedAt": tree_generated_at,
                "isStale": tree_generated_at > 0
                    and int(artifact.get("created_at") or 0) < tree_generated_at,
            }
        )

    async def handle_save_node_report_artifact(self, request: "web.Request") -> "web.Response":
        execution_id = request.match_info["execution_id"]
        execution = self._store.get_execution(execution_id)
        if not execution:
            return _error_response(f"Execution {execution_id} not found", 404, "not_found")

        try:
            body = _to_snake_dict(await request.json())
        except Exception:
            return _error_response("Invalid JSON")

        node_key = str(body.get("node_key") or "").strip()
        title = str(body.get("title") or "").strip()
        content = str(body.get("content") or "").strip()
        if not node_key:
            return _error_response("node_key is required")
        if not content:
            return _error_response("content is required")

        artifact = self._store.save_node_ead_report_artifact(
            execution_id,
            node_key=node_key,
            title=title,
            content=content,
        )
        if not artifact:
            return _error_response("Unable to save node report artifact")
        return _json_response(_to_camel_dict(artifact))

    # ------------------------------------------------------------------
    # Agent-authored PFM tree endpoints (commit_pfm_snapshot pipeline)
    # ------------------------------------------------------------------

    async def handle_get_pfm_tree(self, request: "web.Request") -> "web.Response":
        execution_id = request.match_info["execution_id"]
        execution = self._store.get_execution(execution_id)
        if not execution:
            return _error_response(f"Execution {execution_id} not found", 404, "not_found")
        satisfied = (
            getattr(execution, "pfm_refresh_satisfied_request_id", None) or ""
        ).strip() or None
        snapshot = self._store.get_committed_pfm_tree(execution_id)
        if snapshot is None:
            return _json_response(
                {
                    "executionId": execution_id,
                    "snapshot": None,
                    "satisfiedRefreshRequestId": satisfied,
                }
            )
        return _json_response(
            {
                "executionId": execution_id,
                "snapshot": snapshot,
                "version": int(snapshot.get("version") or 0),
                "generatedAt": int(snapshot.get("generated_at") or 0),
                "satisfiedRefreshRequestId": satisfied,
            }
        )

    async def handle_get_pfm_persisted_state(self, request: "web.Request") -> "web.Response":
        """Return DB-backed canonical PFM tree + all node EAD reports for this execution (read-only)."""
        execution_id = request.match_info["execution_id"]
        execution = self._store.get_execution(execution_id)
        if not execution:
            return _error_response(f"Execution {execution_id} not found", 404, "not_found")

        snapshot = self._store.get_committed_pfm_tree(execution_id)
        node_reports: List[Dict[str, Any]] = []
        summaries_only = str(
            request.query.get("summaries_only") or request.query.get("summariesOnly") or ""
        ).strip().lower() in ("1", "true", "yes", "on")
        for art in self._store.list_execution_pfm_artifacts(execution_id):
            if str(art.get("artifact_type") or "") != "node_ead_report":
                continue
            content = str(art.get("content") or "")
            row: Dict[str, Any] = {
                "node_key": str(art.get("node_key") or "").strip(),
                "artifact_key": str(art.get("artifact_key") or "").strip(),
                "title": str(art.get("title") or "").strip(),
                "created_at": art.get("created_at"),
                "excerpt": str(art.get("excerpt") or "")[:800],
                "content_length": len(content),
            }
            if not summaries_only:
                row["content"] = content
            node_reports.append(row)

        payload = {
            "execution_id": execution_id,
            "committed_snapshot": snapshot,
            "node_reports": node_reports,
        }
        return _json_response(_to_camel_dict(payload))

    async def handle_post_pfm_request_snapshot(self, request: "web.Request") -> "web.Response":
        """Refresh PFM from disk: rebuild DB only when .FMR/.pfm are newer than the committed baseline."""
        execution_id = request.match_info["execution_id"]
        execution = self._store.get_execution(execution_id)
        if not execution:
            return _error_response(f"Execution {execution_id} not found", 404, "not_found")
        if not self._executor:
            return _error_response("Project executor is not available", 503, "unavailable")

        body: Dict[str, Any] = {}
        if request.can_read_body:
            try:
                body = _to_snake_dict(await request.json())
            except Exception:
                body = {}

        refresh_delivery = body.get("refresh_from_delivery") in (True, "true", 1, "1")
        if refresh_delivery:
            from .pfm_delivery import ingest_workspace_delivery_exports
            from .pfm_refresh import try_refresh_pfm_from_delivery

            ingest_workspace_delivery_exports(self._store, execution_id)
            delivery_result = dict(
                try_refresh_pfm_from_delivery(
                    self._store,
                    execution_id,
                    promote_template_canonical=False,
                )
            )
            if delivery_result.get("ok") is True and delivery_result.get("code") != "no_delivery_files":
                # Keep Refresh responsive: import per-node reports from .FMR asynchronously.
                self._schedule_fmr_node_report_sync(execution_id)
            delivery_result["execution_id"] = execution_id
            return _json_response(_to_camel_dict(delivery_result))

        try:
            result = self._executor.request_pfm_snapshot_refresh(execution_id)
        except Exception as exc:
            logger.exception(
                "[projects] request_pfm_snapshot_refresh failed for %s", execution_id
            )
            return _json_response(
                _to_camel_dict(
                    {
                        "ok": False,
                        "code": "error",
                        "message": str(exc)[:240],
                        "execution_id": execution_id,
                    }
                ),
                status=200,
            )

        raw = dict(result)
        raw["execution_id"] = execution_id
        return _json_response(_to_camel_dict(raw))

    async def handle_get_pfm_mindmap(self, request: "web.Request") -> "web.Response":
        from .pfm_tree import render_mermaid_for_scope

        execution_id = request.match_info["execution_id"]
        execution = self._store.get_execution(execution_id)
        if not execution:
            return _error_response(f"Execution {execution_id} not found", 404, "not_found")

        snapshot = self._store.get_committed_pfm_tree(execution_id)
        scope = request.query.get("scope", "top")
        node_key = request.query.get("node_key") or request.query.get("nodeKey")
        depth_raw = request.query.get("depth")
        try:
            depth = int(depth_raw) if depth_raw is not None else None
        except Exception:
            depth = None

        if not snapshot:
            mermaid = (
                "mindmap\n"
                f"  root(({(execution.name or execution.id).replace('(', '[').replace(')', ']')}))\n"
                "    Awaiting first committed PFM tree\n"
            )
            return _json_response(
                {
                    "executionId": execution_id,
                    "scope": scope,
                    "nodeKey": node_key,
                    "depth": depth,
                    "version": 0,
                    "generatedAt": 0,
                    "mermaid": mermaid,
                    "committed": False,
                }
            )

        mermaid = render_mermaid_for_scope(
            snapshot,
            execution,
            scope=scope,
            node_key=node_key,
            depth=depth,
        )
        return _json_response(
            {
                "executionId": execution_id,
                "scope": scope,
                "nodeKey": node_key,
                "depth": depth,
                "version": int(snapshot.get("version") or 0),
                "generatedAt": int(snapshot.get("generated_at") or 0),
                "mermaid": mermaid,
                "committed": True,
            }
        )

    async def handle_get_pfm_view(self, request: "web.Request") -> "web.Response":
        from .pfm_tree import apply_view_state_to_tree

        execution_id = request.match_info["execution_id"]
        execution = self._store.get_execution(execution_id)
        if not execution:
            return _error_response(f"Execution {execution_id} not found", 404, "not_found")
        snapshot = self._store.get_committed_pfm_tree(execution_id)
        saved = self._store.get_pfm_view_state(execution_id)
        if snapshot:
            repaired = apply_view_state_to_tree(saved, snapshot)
        else:
            repaired = {
                "node_path": [],
                "selected_node_key": None,
                "view_scope": "top",
                "depth_cap": 2,
                "pfm_tree_version": 0,
                "updated_at": int(time.time() * 1000),
            }
        return _json_response(
            {
                "executionId": execution_id,
                "state": _to_camel_dict(repaired),
            }
        )

    async def handle_put_pfm_view(self, request: "web.Request") -> "web.Response":
        from .pfm_tree import apply_view_state_to_tree

        execution_id = request.match_info["execution_id"]
        execution = self._store.get_execution(execution_id)
        if not execution:
            return _error_response(f"Execution {execution_id} not found", 404, "not_found")
        try:
            body = _to_snake_dict(await request.json())
        except Exception:
            return _error_response("Invalid JSON")

        snapshot = self._store.get_committed_pfm_tree(execution_id) or {"flat_nodes": []}
        repaired = apply_view_state_to_tree(body, snapshot)
        self._store.set_pfm_view_state(execution_id, repaired)
        return _json_response(
            {"executionId": execution_id, "state": _to_camel_dict(repaired)}
        )

    async def handle_pause_execution(self, request: "web.Request") -> "web.Response":
        execution_id = request.match_info["execution_id"]
        execution = self._store.get_execution(execution_id)
        if not execution:
            return _error_response(f"Execution {execution_id} not found", 404, "not_found")

        if execution.status != ExecutionStatus.RUNNING:
            return _error_response(f"Cannot pause execution in status {execution.status.value}")

        updated = self._store.update_execution(execution_id, paused=True)
        logger.info("[projects] Paused execution %s", execution_id)
        raw = json.loads(updated.model_dump_json())
        return _json_response(_to_camel_dict(raw))

    async def handle_resume_execution(self, request: "web.Request") -> "web.Response":
        execution_id = request.match_info["execution_id"]
        execution = self._store.get_execution(execution_id)
        if not execution:
            return _error_response(f"Execution {execution_id} not found", 404, "not_found")

        if not execution.paused:
            return _error_response("Execution is not paused")

        updated = self._store.update_execution(execution_id, paused=False)
        logger.info("[projects] Resumed execution %s", execution_id)
        raw = json.loads(updated.model_dump_json())
        return _json_response(_to_camel_dict(raw))

    # ------------------------------------------------------------------
    # Route registration
    # ------------------------------------------------------------------

    def register_routes(self, app: "web.Application") -> None:
        app.router.add_get("/v1/projects/templates", self.handle_list_templates)
        app.router.add_get("/v1/projects/templates/{template_id}", self.handle_get_template)
        app.router.add_post("/v1/projects/templates", self.handle_create_template)
        app.router.add_patch("/v1/projects/templates/{template_id}", self.handle_update_template)
        app.router.add_delete("/v1/projects/templates/{template_id}", self.handle_delete_template)
        app.router.add_post(
            "/v1/projects/templates/{template_id}/activate", self.handle_activate_template
        )

        app.router.add_get("/v1/projects/executions", self.handle_list_executions)
        app.router.add_get("/v1/projects/executions/{execution_id}", self.handle_get_execution)
        app.router.add_delete("/v1/projects/executions/{execution_id}", self.handle_delete_execution)
        app.router.add_post(
            "/v1/projects/executions/{execution_id}/invalidate",
            self.handle_invalidate_execution,
        )
        app.router.add_post("/v1/projects/executions/run", self.handle_run_execution)
        app.router.add_post(
            "/v1/projects/executions/{execution_id}/cancel", self.handle_cancel_execution
        )
        app.router.add_post(
            "/v1/projects/executions/{execution_id}/pause", self.handle_pause_execution
        )
        app.router.add_post(
            "/v1/projects/executions/{execution_id}/resume", self.handle_resume_execution
        )
        app.router.add_post(
            "/v1/projects/executions/{execution_id}/reporting-activity",
            self.handle_set_reporting_activity,
        )
        app.router.add_get(
            "/v1/projects/executions/{execution_id}/reports/{filename}",
            self.handle_get_execution_report,
        )
        app.router.add_post(
            "/v1/projects/executions/{execution_id}/artifacts/node-report",
            self.handle_save_node_report_artifact,
        )
        app.router.add_get(
            "/v1/projects/executions/{execution_id}/artifacts/node-report",
            self.handle_get_node_report_artifact,
        )
        app.router.add_get(
            "/v1/projects/executions/{execution_id}/pfm/tree",
            self.handle_get_pfm_tree,
        )
        app.router.add_get(
            "/v1/projects/executions/{execution_id}/pfm/persisted-state",
            self.handle_get_pfm_persisted_state,
        )
        app.router.add_post(
            "/v1/projects/executions/{execution_id}/pfm/request-snapshot",
            self.handle_post_pfm_request_snapshot,
        )
        app.router.add_get(
            "/v1/projects/executions/{execution_id}/pfm/mindmap",
            self.handle_get_pfm_mindmap,
        )
        app.router.add_get(
            "/v1/projects/executions/{execution_id}/pfm/view",
            self.handle_get_pfm_view,
        )
        app.router.add_put(
            "/v1/projects/executions/{execution_id}/pfm/view",
            self.handle_put_pfm_view,
        )
