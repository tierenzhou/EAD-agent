"""
AI-assisted promotion of a template's canonical PFM execution pointer.

Runs after a successful terminal completion when the execution has a committed
PFM tree. Uses the auxiliary provider router (same stack as compression/summary).
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from typing import Any, Dict, Optional, Tuple

from projects.models import ExecutionStatus, ProjectExecute
from projects.store import ProjectStore

logger = logging.getLogger(__name__)

_JSON_OBJECT_RE = re.compile(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", re.DOTALL)


def _tree_summary(store: ProjectStore, execution_id: str, max_lines: int = 100) -> str:
    snap = store.get_committed_pfm_tree(execution_id)
    if not isinstance(snap, dict):
        return "(no committed tree)"
    flat = snap.get("flat_nodes") or snap.get("flatNodes") or []
    if not isinstance(flat, list) or not flat:
        return "(empty flat_nodes)"
    ver = int(snap.get("version") or 0)
    lines = [f"version={ver}, node_count={len(flat)}"]
    for row in flat[:max_lines]:
        if not isinstance(row, dict):
            continue
        nk = str(row.get("node_key") or "").strip()
        title = str(row.get("title") or "").strip().replace("\n", " ")[:120]
        st = str(row.get("status") or "").strip()
        par = str(row.get("parent_node_key") or "").strip()
        parent = f" parent={par}" if par else ""
        lines.append(f"- {nk}: {title} [{st}]{parent}")
    if len(flat) > max_lines:
        lines.append(f"... ({len(flat) - max_lines} more nodes omitted)")
    return "\n".join(lines)


def _extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    raw = str(text or "").strip()
    if not raw:
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    start = raw.find("{")
    end = raw.rfind("}")
    if start == -1 or end <= start:
        return None
    chunk = raw[start : end + 1]
    try:
        return json.loads(chunk)
    except json.JSONDecodeError:
        m = _JSON_OBJECT_RE.search(raw)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                return None
    return None


def _coerce_judge_payload(data: Optional[Dict[str, Any]]) -> Tuple[bool, float, str]:
    if not isinstance(data, dict):
        return False, 0.0, "invalid judge payload"
    replace = data.get("replace_canonical")
    if replace is None:
        replace = data.get("replaceCanonical")
    conf_raw = data.get("confidence", data.get("score", 0))
    try:
        confidence = float(conf_raw)
    except (TypeError, ValueError):
        confidence = 0.0
    rationale = str(data.get("rationale") or data.get("reason") or "").strip()
    return bool(replace), max(0.0, min(1.0, confidence)), rationale[:4000]


async def _call_llm_judge(
    *,
    prior_summary: str,
    candidate_summary: str,
    prior_id: str,
    candidate_id: str,
) -> Tuple[Optional[Dict[str, Any]], str]:
    """Returns (parsed_json_or_none, error_message)."""
    try:
        from agent.auxiliary_client import resolve_provider_client
    except ImportError:
        return None, "auxiliary_client_unavailable"

    provider = (os.getenv("EAD_PFM_CANONICAL_JUDGE_PROVIDER") or "auto").strip() or "auto"
    model = (os.getenv("EAD_PFM_CANONICAL_JUDGE_MODEL") or "").strip() or None

    client, resolved_model = resolve_provider_client(provider, model, async_mode=True)
    if client is None or not resolved_model:
        return None, "no_auxiliary_client"

    system = (
        "You compare two committed PFM (product feature map) tree summaries for the SAME product template. "
        "Decide whether the candidate run should become the new canonical reference map for the template "
        "(better structure, coverage, clarity, or evidence of deeper exploration). "
        "Reply with a single JSON object only, no markdown fences, keys exactly: "
        '{"replace_canonical": boolean, "confidence": number between 0 and 1, "rationale": string}. '
        "Use replace_canonical=true only when the candidate is clearly better or there is no prior map. "
        "Be conservative when the prior map is already strong."
    )
    user = (
        f"prior_execution_id={prior_id or 'none'}\n"
        f"candidate_execution_id={candidate_id}\n\n"
        "## Prior canonical summary\n"
        f"{prior_summary}\n\n"
        "## Candidate run summary\n"
        f"{candidate_summary}\n"
    )

    try:
        resp = await client.chat.completions.create(
            model=resolved_model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0.1,
            max_tokens=600,
        )
    except Exception as exc:
        logger.warning("[pfm_canonical_judge] LLM call failed: %s", exc)
        return None, str(exc)[:500]

    choice = resp.choices[0] if resp and getattr(resp, "choices", None) else None
    msg = choice.message if choice else None
    content = getattr(msg, "content", None) or ""
    parsed = _extract_json_object(str(content))
    if not parsed:
        return None, "unparseable_llm_response"
    return parsed, ""


def _eligible_for_auto_promotion(ex: ProjectExecute) -> bool:
    if ex.valid_for_data_reporting_training is False:
        return False
    if ex.contributes_to_learning is False:
        return False
    return True


async def evaluate_and_maybe_promote_after_completion(
    store: ProjectStore,
    execution_id: str,
    *,
    force: bool = False,
) -> Dict[str, Any]:
    """
    Idempotent post-completion step: judge whether ``execution_id`` should
    replace the template's canonical PFM pointer, then maybe promote.

    Returns a small dict suitable for JSON logging / API responses.
    """
    ex = store.get_execution(execution_id)
    out: Dict[str, Any] = {"execution_id": execution_id, "ok": False}
    if not ex:
        out["error"] = "execution_not_found"
        return out
    if ex.status != ExecutionStatus.COMPLETED:
        out["error"] = "execution_not_completed"
        return out

    if (
        not force
        and ex.pfm_canonical_evaluation_status
        in (
            "skipped_no_tree",
            "skipped_ineligible",
            "judge_unavailable",
            "judged",
            "promoted",
        )
    ):
        out["ok"] = True
        out["skipped"] = "already_evaluated"
        out["status"] = ex.pfm_canonical_evaluation_status
        return out

    template = store.get_template(ex.linked_template_id)
    if not template:
        out["error"] = "template_not_found"
        return out

    if not store.has_committed_pfm_tree(execution_id):
        store.update_execution(
            execution_id,
            pfm_canonical_evaluation_status="skipped_no_tree",
            pfm_canonical_replace_recommended=False,
            pfm_canonical_evaluation_confidence=0.0,
            pfm_canonical_evaluation_rationale=None,
            pfm_canonical_evaluation_at_ms=int(time.time() * 1000),
            pfm_canonical_promotion_applied=False,
        )
        out["ok"] = True
        out["status"] = "skipped_no_tree"
        return out

    if not _eligible_for_auto_promotion(ex):
        store.update_execution(
            execution_id,
            pfm_canonical_evaluation_status="skipped_ineligible",
            pfm_canonical_replace_recommended=False,
            pfm_canonical_evaluation_confidence=0.0,
            pfm_canonical_evaluation_rationale="Run not eligible (reporting/training or learning flags).",
            pfm_canonical_evaluation_at_ms=int(time.time() * 1000),
            pfm_canonical_promotion_applied=False,
        )
        out["ok"] = True
        out["status"] = "skipped_ineligible"
        return out

    prior_id = (template.canonical_pfm_execution_id or "").strip() or None
    if prior_id == execution_id:
        store.update_execution(
            execution_id,
            pfm_canonical_evaluation_status="promoted",
            pfm_canonical_replace_recommended=False,
            pfm_canonical_evaluation_confidence=1.0,
            pfm_canonical_evaluation_rationale="Already the template canonical execution.",
            pfm_canonical_evaluation_at_ms=int(time.time() * 1000),
            pfm_canonical_promotion_applied=True,
        )
        out["ok"] = True
        out["status"] = "already_canonical"
        return out

    candidate_summary = _tree_summary(store, execution_id)
    prior_summary = (
        _tree_summary(store, prior_id)
        if prior_id and store.has_committed_pfm_tree(prior_id)
        else "(no prior canonical committed tree)"
    )

    threshold = 0.65
    try:
        threshold = float(os.getenv("EAD_PFM_CANONICAL_CONFIDENCE_THRESHOLD", "0.65"))
    except ValueError:
        threshold = 0.65

    replace = False
    confidence = 0.0
    rationale = ""
    judge_status = "judge_unavailable"

    parsed: Optional[Dict[str, Any]] = None
    err = ""
    if (os.getenv("EAD_PFM_CANONICAL_JUDGE_DISABLED") or "").strip().lower() in ("1", "true", "yes"):
        # No LLM: first canonical heuristic only
        if not prior_id:
            replace, confidence, rationale = True, 0.9, "judge_disabled_first_canonical_heuristic"
            judge_status = "judged"
        else:
            replace, confidence, rationale = False, 0.0, "judge_disabled_with_existing_canonical"
            judge_status = "judged"
    else:
        parsed, err = await _call_llm_judge(
            prior_summary=prior_summary,
            candidate_summary=candidate_summary,
            prior_id=prior_id or "",
            candidate_id=execution_id,
        )
        if parsed is None:
            if not prior_id:
                # Safe default when nothing to compare against
                replace, confidence, rationale = True, 0.75, f"first_canonical_no_judge: {err}"
                judge_status = "judged"
            else:
                replace, confidence, rationale = False, 0.0, f"judge_failed: {err}"
                judge_status = "judge_unavailable"
        else:
            replace, confidence, rationale = _coerce_judge_payload(parsed)
            judge_status = "judged"

    now_ms = int(time.time() * 1000)
    promoted = False
    if judge_status == "judged" and replace and confidence >= threshold:
        pt = store.promote_template_canonical_pfm(
            ex.linked_template_id,
            execution_id,
            source="ai",
            rationale=rationale or "ai_canonical_promotion",
            require_eligible=True,
        )
        promoted = pt is not None

    store.update_execution(
        execution_id,
        pfm_canonical_evaluation_status="promoted" if promoted else judge_status,
        pfm_canonical_replace_recommended=replace,
        pfm_canonical_evaluation_confidence=confidence,
        pfm_canonical_evaluation_rationale=rationale or None,
        pfm_canonical_evaluation_at_ms=now_ms,
        pfm_canonical_promotion_applied=promoted,
    )

    out["ok"] = True
    out["status"] = "promoted" if promoted else judge_status
    out["promoted"] = promoted
    out["replace_recommended"] = replace
    out["confidence"] = confidence
    return out
