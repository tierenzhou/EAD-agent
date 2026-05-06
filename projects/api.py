"""
Project API handlers for the Hermes API server.

Provides REST endpoints for EAD project template and execution management,
mirroring the RPC methods from EAD-EXP's gateway.
"""

import asyncio
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
from typing import Any, Dict, Optional

try:
    from aiohttp import web

    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False
    web = None

from projects.models import (
    ExecutionStatus,
    ProjectAuthMode,
    ProjectExecute,
    ProjectTemplate,
)
from projects.store import ProjectStore
from projects.pfm_artifacts import report_file_path

logger = logging.getLogger(__name__)
_IMAGE_PATH_RE = re.compile(r"(?:MEDIA:)?(?P<path>/[^\s`\"']+\.(?:png|jpe?g|gif|webp))", re.IGNORECASE)
_IMAGE_BASENAME_RE = re.compile(
    r"\b(?P<path>[\w.-]*?(?:screenshot|screen|image)[\w.-]*?\.(?:png|jpe?g|gif|webp))\b",
    re.IGNORECASE,
)
_IMAGE_EXT_RE = re.compile(r"\.(png|jpe?g|gif|webp)(?:$|[?#])", re.IGNORECASE)
_MIN_BROWSER_SCREENSHOT_BYTES = 10_000


def _hermes_home() -> Path:
    configured = os.getenv("HERMES_HOME", "").strip()
    return Path(configured).expanduser() if configured else Path.home() / ".hermes"


def _resolve_local_image_path(path_text: str) -> Path:
    src = Path(path_text).expanduser()
    if src.exists():
        return src
    if src.is_absolute() or src.name != path_text:
        return src
    for directory in (
        _hermes_home() / "cache" / "screenshots",
        _hermes_home() / "browser_screenshots",
    ):
        candidate = directory / src.name
        if candidate.exists():
            return candidate
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
    if "\n" in path_text or "\r" in path_text or len(path_text) > 2048:
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

    async def handle_list_executions(self, request: "web.Request") -> "web.Response":
        template_id = request.query.get("template_id")
        status = request.query.get("status")
        executions = self._store.list_executions(
            template_id=template_id or None,
            status=status or None,
        )
        return _json_response(
            {
                "executions": [_to_camel_dict(json.loads(e.model_dump_json())) for e in executions],
            }
        )

    async def handle_get_execution(self, request: "web.Request") -> "web.Response":
        execution_id = request.match_info["execution_id"]
        execution = self._store.get_execution(execution_id)
        if not execution:
            return _error_response(f"Execution {execution_id} not found", 404, "not_found")
        raw = json.loads(execution.model_dump_json())
        raw = _backfill_execution_screenshots(raw)
        return _json_response(_to_camel_dict(raw))

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

        execution = ProjectExecute(
            linked_template_id=template.id,
            name=body.get("name", f"Run - {template.name}"),
            description=body.get("description", template.description),
            target_url=body.get("target_url", template.target_url),
            ai_prompt=body.get("ai_prompt", template.ai_prompt),
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
        )

        created = self._store.create_execution(execution)
        logger.info("[projects] Created execution %s for template %s", execution.id, template.id)

        session_key = f"eadproj-exec-{created.id}"
        session_bootstrap_ok = False
        try:
            from hermes_state import SessionDB

            db = SessionDB()
            session_id = f"eadproj-{uuid.uuid4().hex[:12]}"
            db.create_session(session_id=session_id, source="api_server", system_prompt="")
            db.set_session_title(session_id, session_key)

            logger.info(
                "[projects] Bootstrapped session %s for execution %s", session_id, created.id
            )

            self._store.update_execution(created.id, run_session_key=session_key)
            session_bootstrap_ok = True
        except Exception as e:
            logger.warning(
                "[projects] Session bootstrap failed for execution %s: %s", created.id, e
            )
            updated = self._store.update_execution(
                created.id,
                status=ExecutionStatus.FAILED,
                last_error_message=f"Run session bootstrap failed: {str(e)[:180]}",
                executor_hint="AI Failed",
            )
            if updated:
                created = updated

        if self._executor and session_bootstrap_ok:
            try:
                loop = asyncio.get_running_loop()
                loop.create_task(self._executor.start_execution(created.id))
                logger.info("[projects] Executor started for execution %s", created.id)
            except Exception as e:
                logger.error("[projects] Failed to start executor for %s: %s", created.id, e)

        raw = json.loads(created.model_dump_json())
        return _json_response(_to_camel_dict(raw), status=201)

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

    async def handle_get_execution_report(self, request: "web.Request") -> "web.Response":
        execution_id = request.match_info["execution_id"]
        filename = Path(request.match_info["filename"]).name
        execution = self._store.get_execution(execution_id)
        if not execution:
            return _error_response(f"Execution {execution_id} not found", 404, "not_found")

        target = report_file_path(execution_id, filename)
        if not target.exists() or not target.is_file():
            return _error_response(f"Report {filename} not found", 404, "not_found")

        content_type, _ = mimetypes.guess_type(str(target))
        return web.FileResponse(path=target, headers={"Content-Type": content_type or "text/plain"})

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
        app.router.add_get(
            "/v1/projects/executions/{execution_id}/reports/{filename}",
            self.handle_get_execution_report,
        )
