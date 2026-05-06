"""
EAD project agent tools.

Registers read_ead_execution and report_running_step with the Hermes tool registry.
These tools allow the agent to read execution state and report milestones
during a project run.
"""

import json
import logging
import re
from typing import Optional

from .pfm_artifacts import build_and_persist_pfm_artifacts
from .store import ProjectStore

logger = logging.getLogger(__name__)

_store: Optional[ProjectStore] = None

_MIN_LOGIN_SUCCESS_SCREENSHOTS = 3


def _login_checkpoint_title(title: str) -> bool:
    return "login pfm map (interactive)" in str(title or "").strip().lower()


def _nonempty_thumbnail_urls(thumbnail_urls) -> list:
    if not isinstance(thumbnail_urls, list):
        return []
    return [u for u in thumbnail_urls if str(u).strip()]


def _count_prior_login_steps_with_screenshots(progress_log) -> int:
    n = 0
    for entry in progress_log or []:
        if entry.kind != "tool_use" or (entry.tool_name or "") != "report_running_step":
            continue
        ti = entry.tool_input or {}
        if not _login_checkpoint_title(str(ti.get("title") or "")):
            continue
        if _nonempty_thumbnail_urls(ti.get("thumbnail_urls")):
            n += 1
    return n


def _count_prior_login_evidence_screenshots(progress_log) -> int:
    urls = set()
    for entry in progress_log or []:
        for attr in ("thumbnail_url", "image_url"):
            url = getattr(entry, attr, None)
            if url:
                urls.add(str(url))
        ti = entry.tool_input or {}
        thumbs = ti.get("thumbnail_urls")
        if isinstance(thumbs, list):
            urls.update(str(u) for u in thumbs if str(u).strip())
    return len(urls)


def _get_store() -> ProjectStore:
    global _store
    if _store is None:
        _store = ProjectStore()
    return _store


def register_project_tools(store: Optional[ProjectStore] = None) -> None:
    global _store
    if store:
        _store = store

    try:
        from tools.registry import registry
    except ImportError:
        logger.warning("[ead_tools] Tool registry not available, skipping registration")
        return

    def _read_ead_execution(args: dict, context: dict = None) -> str:
        s = _get_store()
        execution_id = args.get("execution_id", "")
        execution = s.get_execution(execution_id)
        if not execution:
            return json.dumps({"error": f"Execution {execution_id} not found"})

        screenshots_recorded = sum(
            1 for entry in (execution.progress_log or [])
            if entry.kind == "tool_result" and entry.thumbnail_url
        )

        return json.dumps({
            "id": execution.id,
            "template_id": execution.linked_template_id,
            "name": execution.name,
            "target_url": execution.target_url,
            "status": execution.status.value,
            "progress_percentage": execution.progress_percentage,
            "paused": execution.paused,
            "steps_count": len(execution.steps),
            "results_count": len(execution.results),
            "screenshots_recorded": screenshots_recorded,
            "executor_hint": execution.executor_hint,
            "cancel_reason": execution.cancel_reason,
        })

    registry.register(
        name="read_ead_execution",
        toolset="project",
        schema={
            "type": "object",
            "description": "Read the current status and results of an EAD project execution. "
                           "Use this to check progress, view step results, and see screenshots recorded.",
            "properties": {
                "execution_id": {
                    "type": "string",
                    "description": "The execution ID to read",
                },
            },
            "required": ["execution_id"],
        },
        handler=_read_ead_execution,
        description="Read EAD project execution status and results",
    )

    def _report_running_step(args: dict, context: dict = None) -> str:
        from .models import (
            EadFmNodeRun,
            StepArtifact,
            StepResult,
            StepStatus,
            TestCaseRun,
            TestCaseStepRunStatus,
        )

        s = _get_store()
        execution_id = args.get("execution_id", "")
        title = args.get("title", "")
        description = args.get("description", "")
        thumbnail_urls = args.get("thumbnail_urls", [])
        pfm_node = args.get("pfm_node")
        login_phase_status_raw = str(args.get("login_phase_status") or "").strip().lower()
        login_success_arg = args.get("login_success")
        initialization_review_status = str(args.get("initialization_review_status") or "").strip().lower()
        ready_to_explore = args.get("ready_to_explore")
        observation = str(args.get("observation") or "").strip()
        finding = str(args.get("finding") or "").strip()
        reasoning = str(args.get("reasoning") or "").strip()
        decision = str(args.get("decision") or "").strip()
        next_direction = str(args.get("next_direction") or "").strip()

        execution = s.get_execution(execution_id)
        if not execution:
            return json.dumps({"error": f"Execution {execution_id} not found"})

        thumbs_now = _nonempty_thumbnail_urls(thumbnail_urls)
        claims_login_success = login_phase_status_raw == "success" or login_success_arg is True
        if claims_login_success and _login_checkpoint_title(title):
            login_text = " ".join(
                str(part or "")
                for part in (title, description, observation, finding, reasoning, decision, next_direction)
            ).lower()
            signup_markers = (
                "sign up",
                "signup",
                "register",
                "create account",
                "new account",
                "start trial",
            )
            signin_markers = ("sign in", "signin", "log in", "login", "logged in")
            if any(marker in login_text for marker in signup_markers) and not any(
                marker in login_text for marker in signin_markers
            ):
                return json.dumps(
                    {
                        "error": "login_success_must_be_signin_only",
                        "recorded": False,
                        "message": (
                            "Do not complete sign-up/register/create-account flows. "
                            "Use existing-account sign-in only, or ask the end user for the correct sign-in route "
                            "and credentials."
                        ),
                    }
                )
            prior_shot_steps = _count_prior_login_steps_with_screenshots(execution.progress_log)
            prior_evidence_shots = _count_prior_login_evidence_screenshots(execution.progress_log)
            steps_with_shots = prior_shot_steps + (1 if thumbs_now else 0)
            enough = (
                len(thumbs_now) >= _MIN_LOGIN_SUCCESS_SCREENSHOTS
                or steps_with_shots >= _MIN_LOGIN_SUCCESS_SCREENSHOTS
                or prior_evidence_shots + len(thumbs_now) >= _MIN_LOGIN_SUCCESS_SCREENSHOTS
            )
            if not enough:
                return json.dumps(
                    {
                        "error": "login_success_requires_screenshots",
                        "recorded": False,
                        "message": (
                            f"Before login_phase_status=success (or login_success=true), attach at least "
                            f"{_MIN_LOGIN_SUCCESS_SCREENSHOTS} screenshots: either include "
                            f"{_MIN_LOGIN_SUCCESS_SCREENSHOTS}+ distinct URLs in thumbnail_urls on this call, "
                            f"or accumulate {_MIN_LOGIN_SUCCESS_SCREENSHOTS} separate "
                            "'Login PFM map (Interactive)' steps each with at least one screenshot. "
                            "Typical proof: (1) URL/site identity, (2) login step, (3) post-login view."
                        ),
                        "login_steps_with_screenshots_so_far": prior_shot_steps,
                        "login_evidence_screenshots_so_far": prior_evidence_shots,
                        "thumbnails_in_this_call": len(thumbs_now),
                    }
                )

        step_num = len(execution.steps) + 1
        artifacts = [
            StepArtifact(
                type="screenshot",
                path=url,
                captured_at=__import__("time").strftime("%Y-%m-%dT%H:%M:%S"),
            )
            for url in thumbnail_urls
        ]

        detail_lines = []
        if observation:
            detail_lines.append(f"Observation: {observation}")
        if finding:
            detail_lines.append(f"Finding: {finding}")
        if reasoning:
            detail_lines.append(f"Reasoning: {reasoning}")
        if decision:
            detail_lines.append(f"Decision: {decision}")
        if next_direction:
            detail_lines.append(f"Next direction: {next_direction}")
        enriched_description = description
        if detail_lines:
            enriched_description = (description + "\n" if description else "") + "\n".join(detail_lines)

        new_step = StepResult(
            step_id=f"step-{step_num}",
            title=title,
            status=StepStatus.COMPLETED,
            summary=enriched_description,
            artifacts=artifacts,
        )

        current_steps = list(execution.steps) + [new_step]
        update_fields = {
            "steps": current_steps,
            "current_step_id": new_step.step_id,
        }

        login_success_before_step = False
        for entry in execution.progress_log or []:
            if entry.kind != "tool_use" or (entry.tool_name or "") != "report_running_step":
                continue
            ti = entry.tool_input or {}
            title_i = str(ti.get("title") or "").strip().lower()
            if "login pfm map (interactive)" not in title_i:
                continue
            status_i = str(ti.get("login_phase_status") or "").strip().lower()
            if status_i == "success":
                login_success_before_step = True
                break
            if isinstance(ti.get("login_success"), bool) and bool(ti.get("login_success")):
                login_success_before_step = True
                break

        should_record_pfm_node = isinstance(pfm_node, dict) and login_success_before_step
        if should_record_pfm_node and not thumbs_now:
            return json.dumps(
                {
                    "error": "pfm_confirmation_requires_screenshot",
                    "recorded": False,
                    "message": (
                        "Whenever you identify, update, or re-confirm a feature, function, "
                        "functional area, or PFM node, capture a fresh screenshot and include it in "
                        "thumbnail_urls on the same report_running_step call. This applies "
                        "to inherited/existing nodes as well as new nodes."
                    ),
                }
            )
        if should_record_pfm_node:
            node_title_for_validation = str(
                pfm_node.get("title")
                or pfm_node.get("node_key")
                or pfm_node.get("node_id")
                or ""
            ).strip().lower()
            action_node_prefixes = (
                "click ",
                "tap ",
                "press ",
                "type ",
                "enter ",
                "input ",
                "select ",
                "choose ",
                "open dropdown",
                "scroll ",
                "navigate to ",
            )
            if any(node_title_for_validation.startswith(prefix) for prefix in action_node_prefixes):
                return json.dumps(
                    {
                        "error": "pfm_node_must_be_functional_area",
                        "recorded": False,
                        "message": (
                            "A PFM node must represent a functional area/capability, not a simple UI action. "
                            "Record clicks, typing, navigation, and button presses as test steps under "
                            "test_case_runs for the relevant functional node."
                        ),
                    }
                )
        if should_record_pfm_node:
            existing_results = list(execution.results or [])
            raw_node_key = str(
                pfm_node.get("node_key")
                or pfm_node.get("node_id")
                or re.sub(r"[^a-z0-9]+", "-", str(pfm_node.get("title", "")).lower()).strip("-")
                or f"node-{len(existing_results) + 1}"
            )
            parent_node_key = str(pfm_node.get("parent_node_key") or "").strip() or None
            node_level_raw = pfm_node.get("level")
            try:
                node_level = max(0, int(node_level_raw)) if node_level_raw is not None else 0
            except Exception:
                node_level = 0
            node_description = str(pfm_node.get("description") or "").strip()
            node_key = raw_node_key
            if parent_node_key and "/" not in node_key and ">" not in node_key:
                # Keep hierarchical context in node_key so UI can render parent-child tree
                # even when parent references are incomplete in intermediate updates.
                node_key = f"{parent_node_key}/{node_key}"
            node_id = str(pfm_node.get("node_id") or node_key)
            node_title = str(pfm_node.get("title") or title or node_key)
            node_type = str(pfm_node.get("type") or "feature-area")
            node_status_raw = str(pfm_node.get("status") or "No Run").strip().lower()
            node_status = (
                TestCaseStepRunStatus.SUCCESS if node_status_raw == "success"
                else TestCaseStepRunStatus.FAILED if node_status_raw == "failed"
                else TestCaseStepRunStatus.NO_RUN
            )

            case_runs = []
            raw_cases = pfm_node.get("test_cases") or []
            if isinstance(raw_cases, list):
                for idx, raw in enumerate(raw_cases):
                    if not isinstance(raw, dict):
                        continue
                    raw_status = str(raw.get("status") or "No Run").strip().lower()
                    case_status = (
                        TestCaseStepRunStatus.SUCCESS if raw_status == "success"
                        else TestCaseStepRunStatus.FAILED if raw_status == "failed"
                        else TestCaseStepRunStatus.NO_RUN
                    )
                    case_title = str(raw.get("title") or f"Test case {idx + 1}")
                    case_id = str(raw.get("case_id") or re.sub(r"[^a-z0-9]+", "-", case_title.lower()).strip("-") or f"case-{idx+1}")
                    case_runs.append(
                        TestCaseRun(
                            case_id=case_id,
                            title=case_title,
                            status=case_status,
                            test_case_step_runs=[],
                        )
                    )

            merged = False
            for i, node in enumerate(existing_results):
                if node.node_key == node_key or node.node_id == node_id:
                    existing_results[i] = EadFmNodeRun(
                        node_id=node_id,
                        node_key=node_key,
                        parent_node_key=parent_node_key,
                        level=node_level,
                        type=node_type,
                        title=node_title,
                        meta=node_description or node.meta,
                        status=node_status,
                        test_case_runs=case_runs or node.test_case_runs,
                    )
                    merged = True
                    break
            if not merged:
                existing_results.append(
                    EadFmNodeRun(
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
                )
            update_fields["results"] = existing_results

        if (
            "login pfm map (interactive)" in str(title).strip().lower()
            and login_phase_status_raw in ("pending", "success", "failed")
        ):
            update_fields["executor_hint"] = (
                "Login checkpoint pending"
                if login_phase_status_raw == "pending"
                else "Login checkpoint failed"
                if login_phase_status_raw == "failed"
                else "Login confirmed; PFM discovery started"
            )

        if "pfm inheritance review (initialization)" in str(title).strip().lower():
            update_fields["executor_hint"] = (
                "Running..."
                if initialization_review_status == "success" or ready_to_explore is True
                else "Reviewing previous PFM results before exploration..."
            )

        s.update_execution(execution_id, **update_fields)

        logger.info("[ead_tools] Reported step %s for execution %s: %s", new_step.step_id, execution_id, title)
        return json.dumps(
            {
                "recorded": True,
                "step_id": new_step.step_id,
                "pfm_node_recorded": should_record_pfm_node,
                "pfm_node_ignored_reason": (
                    "login_not_confirmed_yet"
                    if isinstance(pfm_node, dict) and not should_record_pfm_node
                    else None
                ),
            }
        )

    registry.register(
        name="report_running_step",
        toolset="project",
        schema={
            "type": "object",
            "description": "Report a milestone step during an EAD project execution. "
                           "Call this when you complete a significant action like navigating to a page, "
                           "filling a form, or discovering a new PFM node. "
                           "For PFM discovery runs, the first checkpoint should use title "
                           "'Login PFM map (Interactive)' to confirm login readiness/success.",
            "properties": {
                "execution_id": {
                    "type": "string",
                    "description": "The execution ID this step belongs to",
                },
                "title": {
                    "type": "string",
                    "description": "Short title for this step (e.g. 'Navigated to login page')",
                },
                "description": {
                    "type": "string",
                    "description": "Description of what happened in this step",
                },
                "thumbnail_urls": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "Screenshot URLs captured during this step. Required whenever this "
                        "step identifies, updates, or re-confirms a feature, function, "
                        "functional area, or PFM node, including inherited/existing nodes."
                    ),
                },
                "observation": {
                    "type": "string",
                    "description": "What the agent directly observed in UI/system during this step.",
                },
                "finding": {
                    "type": "string",
                    "description": "Important discovery extracted from the observation.",
                },
                "reasoning": {
                    "type": "string",
                    "description": "Why this finding matters and how it informs strategy.",
                },
                "decision": {
                    "type": "string",
                    "description": "The decision taken for the next exploration move.",
                },
                "next_direction": {
                    "type": "string",
                    "description": "Immediate next direction or target area to explore.",
                },
                "login_phase_status": {
                    "type": "string",
                    "description": "For login checkpoint only: pending | success | failed. "
                    "Existing-account sign-in only: never use sign-up, register, create-account, trial, "
                    "or new-user onboarding flows. "
                    "Use pending while asking the user for credentials/OTP/captcha help or waiting for UI response. "
                    "Use success only after attaching at least 3 screenshots total (see thumbnail_urls): "
                    "either 3+ URLs on one step, or 3+ separate login steps each with a screenshot.",
                },
                "login_success": {
                    "type": "boolean",
                    "description": "Alternative explicit login success flag for the checkpoint step. "
                    "Only set true for existing-account sign-in, never sign-up/register/create-account flows. "
                    "Same screenshot evidence rules as login_phase_status=success.",
                },
                "initialization_review_status": {
                    "type": "string",
                    "description": (
                        "For post-login initialization review only: pending | success. "
                        "Use success after reviewing inherited previous PFM results, summarizing the prior mindmap, "
                        "and stating the plan for the assigned active exploration duration."
                    ),
                },
                "ready_to_explore": {
                    "type": "boolean",
                    "description": (
                        "For post-login initialization review only. Set true when sign-in is complete, "
                        "previous results have been analyzed, and the next-run plan is ready."
                    ),
                },
                "pfm_node": {
                    "type": "object",
                    "description": (
                        "Optional structured PFM node update for mindmap progress tracking "
                        "(allowed only after login success checkpoint). A PFM node must be "
                        "a functional area/capability, not a simple UI action like a click, "
                        "typing into a field, opening a dropdown, or pressing a button. If provided, "
                        "thumbnail_urls must include fresh screenshot evidence from this run."
                    ),
                    "properties": {
                        "node_id": {"type": "string"},
                        "node_key": {"type": "string"},
                        "parent_node_key": {
                            "type": "string",
                            "description": "Parent node key for hierarchy (required for child nodes).",
                        },
                        "level": {
                            "type": "integer",
                            "description": (
                                "Hierarchy level: 1=domain, 2=functional area/PFM node, "
                                "3=feature group or feature. Do not use action-level nodes; "
                                "record clicks/form fills/navigation as test steps under test_case_runs."
                            ),
                        },
                        "title": {"type": "string"},
                        "description": {
                            "type": "string",
                            "description": "Brief node description (keep concise, mindmap-friendly).",
                        },
                        "type": {"type": "string"},
                        "status": {"type": "string", "description": "Success | Failed | No Run"},
                        "features": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Feature names discovered for this node",
                        },
                        "test_cases": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "case_id": {"type": "string"},
                                    "title": {"type": "string"},
                                    "status": {"type": "string", "description": "Success | Failed | No Run"},
                                },
                            },
                        },
                    },
                },
            },
            "required": ["execution_id", "title", "description"],
        },
        handler=_report_running_step,
        description="Report a milestone step during EAD project execution",
    )

    def _publish_pfm_artifacts(args: dict, context: dict = None) -> str:
        s = _get_store()
        execution_id = args.get("execution_id", "")
        execution = s.get_execution(execution_id)
        if not execution:
            return json.dumps({"error": f"Execution {execution_id} not found"})

        reports = build_and_persist_pfm_artifacts(execution)
        s.update_execution(execution_id, reports=reports)
        s.sync_execution_pfm_artifacts_from_state(execution_id)
        return json.dumps(
            {
                "published": True,
                "execution_id": execution_id,
                "reports": [r.model_dump() for r in reports],
            }
        )

    registry.register(
        name="publish_pfm_artifacts",
        toolset="project",
        schema={
            "type": "object",
            "description": (
                "Generate and publish PFM mindmap/report artifacts for the current "
                "EAD project execution, then attach them to the run."
            ),
            "properties": {
                "execution_id": {
                    "type": "string",
                    "description": "The execution ID to publish PFM artifacts for",
                },
            },
            "required": ["execution_id"],
        },
        handler=_publish_pfm_artifacts,
        description="Generate and attach PFM mindmap/report artifacts for a run",
    )

    logger.info(
        "[ead_tools] Registered read_ead_execution, report_running_step, publish_pfm_artifacts"
    )
