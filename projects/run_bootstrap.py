"""
Deferred execution bootstrap: PFM seed, artifact sync, Hermes session, executor start.

Runs asynchronously after POST /v1/projects/executions/run returns 201 so the UI
does not block on heavy SQLite + SessionDB work.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from typing import Any, Optional

from projects.models import ExecutionStatus, ProjectExecute
from projects.store import ProjectStore

logger = logging.getLogger(__name__)


def _should_abort_bootstrap(ex: Optional[ProjectExecute]) -> bool:
    if not ex:
        return True
    if ex.status not in (ExecutionStatus.PENDING,):
        return True
    return False


async def run_execution_bootstrap(
    store: ProjectStore,
    executor: Any,
    execution_id: str,
) -> None:
    """Copy PFM artifacts, sync mindmaps, create chat session, link run_session_key, start executor."""
    from projects.api import (
        _build_phase1_canonical_baseline_message,
        _execution_ack_message,
        _execution_boundary_message,
        _try_inject_inherited_pfm_mindmap_cache,
        enrich_execution_ai_prompt_with_learning,
    )

    t0 = time.perf_counter()

    def _log_phase(name: str, start: float) -> None:
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        logger.info(
            "[projects.bootstrap] execution=%s phase=%s elapsed_ms=%.1f total_ms=%.1f",
            execution_id,
            name,
            elapsed_ms,
            (time.perf_counter() - t0) * 1000.0,
        )

    ex = store.get_execution(execution_id)
    if not ex:
        logger.info("[projects.bootstrap] skip execution=%s (missing)", execution_id)
        return

    if (ex.run_session_key or "").strip():
        if executor and not executor.has_active_monitor(execution_id):
            try:
                asyncio.get_running_loop().create_task(executor.start_execution(execution_id))
                logger.info(
                    "[projects.bootstrap] linked session exists; started monitor for %s",
                    execution_id,
                )
            except Exception as exc:
                logger.error("[projects.bootstrap] executor start failed %s: %s", execution_id, exc)
        return

    if _should_abort_bootstrap(ex):
        logger.info("[projects.bootstrap] skip execution=%s (not pending)", execution_id)
        return

    template_id = ex.linked_template_id
    inherit_pfm = getattr(ex, "bootstrap_inherit_pfm", True)
    if inherit_pfm is None:
        inherit_pfm = True
    explicit_source_id = getattr(ex, "bootstrap_explicit_inherit_from_execution_id", None) or None
    if explicit_source_id is not None:
        explicit_source_id = str(explicit_source_id).strip() or None

    created = ex
    session_key = f"eadproj-exec-{execution_id}"

    phase_start = time.perf_counter()
    if inherit_pfm:
        resolved_src: Optional[ProjectExecute] = None
        if explicit_source_id:
            resolved_src = store.resolve_pfm_inheritance_source(
                template_id,
                explicit_source_id=explicit_source_id,
            )
        src = resolved_src
        if not src:
            src = store.resolve_pfm_inheritance_source(
                template_id,
                exclude_execution_id=created.id,
                explicit_source_id=None,
            )
        if src:
            try:
                seeded_run = store.seed_execution_from_prior_run(created.id, src.id)
                if seeded_run:
                    created = seeded_run
            except Exception as exc:
                logger.exception(
                    "[projects.bootstrap] seed_execution_from_prior_run failed exec=%s from=%s: %s",
                    created.id,
                    src.id,
                    exc,
                )
                try:
                    seeded_fallback = store.seed_execution_from_template_artifacts(created.id)
                    if seeded_fallback:
                        created = seeded_fallback
                except Exception as exc2:
                    logger.exception("[projects.bootstrap] template seed fallback failed: %s", exc2)
        else:
            seeded = store.seed_execution_from_template_artifacts(created.id)
            if seeded:
                created = seeded
    else:
        seeded = store.seed_execution_from_template_artifacts(created.id)
        if seeded:
            created = seeded

    _log_phase("pfm_seed", phase_start)

    ex = store.get_execution(execution_id)
    if _should_abort_bootstrap(ex):
        logger.info("[projects.bootstrap] aborted after seed execution=%s", execution_id)
        return

    phase_start = time.perf_counter()
    try:
        store.sync_execution_pfm_artifacts_from_state(created.id)
        refreshed_ex = store.get_execution(created.id)
        if refreshed_ex:
            created = refreshed_ex
    except Exception as exc:
        logger.warning(
            "[projects.bootstrap] sync_execution_pfm_artifacts_from_state failed for %s: %s",
            created.id,
            exc,
        )
    _log_phase("pfm_sync", phase_start)

    ex = store.get_execution(execution_id)
    if _should_abort_bootstrap(ex):
        logger.info("[projects.bootstrap] aborted before session execution=%s", execution_id)
        return

    phase_start = time.perf_counter()
    try:
        enrich_execution_ai_prompt_with_learning(store, created.id)
        refreshed = store.get_execution(created.id)
        if refreshed:
            created = refreshed
    except Exception as exc:
        logger.warning(
            "[projects.bootstrap] enrich_execution_ai_prompt_with_learning failed for %s: %s",
            created.id,
            exc,
        )
    _log_phase("ai_prompt", phase_start)

    phase_start = time.perf_counter()
    template = store.get_template(template_id) if template_id else None
    session_bootstrap_ok = False
    try:
        from hermes_state import SessionDB

        db = SessionDB()
        session_id = f"eadproj-{uuid.uuid4().hex[:12]}"
        boundary_message = _execution_boundary_message(created)
        db.create_session(
            session_id=session_id,
            source="api_server",
            system_prompt=boundary_message,
        )
        db.set_session_title(session_id, session_key)
        db.append_message(
            session_id=session_id,
            role="system",
            content=boundary_message,
        )
        phase1_snap = _build_phase1_canonical_baseline_message(store, created)
        if phase1_snap:
            db.append_message(
                session_id=session_id,
                role="system",
                content=phase1_snap,
            )
            logger.info("[projects.bootstrap] Phase-1 baseline injected for execution %s", created.id)
        inh = getattr(created, "inherited_from_execution_id", None)
        if inh and _try_inject_inherited_pfm_mindmap_cache(session_id, session_key, str(inh)):
            logger.info("[projects.bootstrap] Injected mindmap cache from prior execution %s", inh)

        db.append_message(
            session_id=session_id,
            role="assistant",
            content=_execution_ack_message(created, template=template),
        )
        logger.info("[projects.bootstrap] Bootstrapped session %s for execution %s", session_id, created.id)

        store.update_execution(
            created.id,
            run_session_key=session_key,
            bootstrap_pending=False,
        )
        session_bootstrap_ok = True
    except Exception as e:
        logger.warning("[projects.bootstrap] Session bootstrap failed for execution %s: %s", created.id, e)
        store.update_execution(
            created.id,
            status=ExecutionStatus.FAILED,
            last_error_message=f"Run session bootstrap failed: {str(e)[:180]}",
            executor_hint="AI Failed",
            bootstrap_pending=False,
        )
    _log_phase("session_db", phase_start)

    if executor and session_bootstrap_ok:
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(executor.start_execution(execution_id))
            logger.info("[projects.bootstrap] executor started for execution %s", execution_id)
        except Exception as e:
            logger.error("[projects.bootstrap] Failed to start executor for %s: %s", execution_id, e)

    logger.info(
        "[projects.bootstrap] execution=%s done total_ms=%.1f",
        execution_id,
        (time.perf_counter() - t0) * 1000.0,
    )
