"""
Export committed PFM run knowledge to Hermes skill files (disk + DB artifacts).
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from .pfm_tree import PFM_TREE_ARTIFACT_KEY, collect_node_report_markdown_by_key

if TYPE_CHECKING:
    from .store import ProjectStore

logger = logging.getLogger(__name__)

SKILL_MAP_NAME = "kloud-pfm-map"
SKILL_REPORTS_NAME = "kloud-pfm-reports"
SKILL_CATEGORY = "pfm"
PFM_SKILL_ARTIFACT_TYPE = "pfm_skill"

KNOWN_SKILLS = (SKILL_MAP_NAME, SKILL_REPORTS_NAME)

PFM_POST_LOGIN_SKILLS_MARKER = "[PFM-POST-LOGIN-SKILLS:v1]"


def skills_root_dir() -> Path:
    return Path.home() / ".hermes" / "skills" / SKILL_CATEGORY


def local_skill_md_path(skill_name: str) -> Path:
    return skills_root_dir() / skill_name / "SKILL.md"


def skill_artifact_key(skill_name: str) -> str:
    return f"skill-{skill_name}"


def _frontmatter(name: str, description: str) -> str:
    return (
        "---\n"
        f"name: {name}\n"
        f"description: {description}\n"
        f"category: {SKILL_CATEGORY}\n"
        "trigger: EAD PFM explore — template-scoped product knowledge.\n"
        "---\n\n"
    )


def build_map_skill_markdown(
    *,
    template_name: str,
    execution_id: str,
    snapshot: Dict[str, Any],
) -> str:
    version = int(snapshot.get("version") or 0)
    flat_nodes = list(snapshot.get("flat_nodes") or [])
    lines = [
        _frontmatter(
            SKILL_MAP_NAME,
            f"PFM feature map for {template_name} (committed tree v{version}).",
        ),
        f"# PFM map — {template_name}",
        "",
        f"- **Execution:** `{execution_id}`",
        f"- **Tree version:** v{version}",
        "",
        "## Nodes",
        "",
    ]
    sorted_nodes = sorted(
        flat_nodes,
        key=lambda n: (
            int(n.get("level") or 0) if isinstance(n, dict) else 0,
            str(n.get("title") or "") if isinstance(n, dict) else "",
        ),
    )
    for row in sorted_nodes:
        if not isinstance(row, dict):
            continue
        title = str(row.get("title") or row.get("node_key") or "Node").strip()
        node_key = str(row.get("node_key") or "").strip()
        level = int(row.get("level") or 0)
        indent = "  " * max(0, level)
        status = str(row.get("status") or "No Run").strip()
        lines.append(f"{indent}- **{title}** (`{node_key}`) — {status}")
    if len(lines) <= 8:
        lines.append("- _(no nodes in committed tree)_")
    return "\n".join(lines) + "\n"


def build_reports_skill_markdown(
    *,
    template_name: str,
    execution_id: str,
    reports_by_key: Dict[str, Dict[str, str]],
    snapshot: Optional[Dict[str, Any]] = None,
) -> str:
    version = int((snapshot or {}).get("version") or 0)
    lines = [
        _frontmatter(
            SKILL_REPORTS_NAME,
            f"Per-node EAD reports for {template_name} (tree v{version}).",
        ),
        f"# PFM node reports — {template_name}",
        "",
        f"- **Execution:** `{execution_id}`",
        f"- **Report count:** {len(reports_by_key)}",
        "",
    ]
    for node_key in sorted(reports_by_key.keys()):
        entry = reports_by_key[node_key]
        title = str(entry.get("title") or node_key).strip()
        md = str(entry.get("markdown") or "").strip()
        preview = md[:1200] + ("…" if len(md) > 1200 else "")
        lines.extend([f"## {title}", "", f"`{node_key}`", "", preview, ""])
    if not reports_by_key:
        lines.append("_No node reports committed for this run yet._")
    return "\n".join(lines) + "\n"


def write_local_skill(skill_name: str, content: str) -> Path:
    target = local_skill_md_path(skill_name)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")
    return target


def _skills_ready_for_injection(skills: List[Dict[str, Any]]) -> bool:
    """True when both map and reports skill artifacts exist for this execution in DB."""
    by_name = {str(s.get("name") or ""): s for s in skills}
    for name in KNOWN_SKILLS:
        row = by_name.get(name)
        if not row:
            return False
        if not row.get("in_database"):
            return False
    return True


def _execution_eligible_for_template_learning(execution: Any) -> bool:
    if execution.valid_for_data_reporting_training is False:
        return False
    if execution.contributes_to_learning is False:
        return False
    return True


def resolve_template_learning_source_execution(
    store: "ProjectStore",
    template_id: str,
    *,
    exclude_execution_id: Optional[str] = None,
) -> Optional[Any]:
    """
    Latest completed, learning-eligible execution for this template with both PFM skills.
    Used for bootstrap seeding and post-login skill inject so both pick the same source.
    """
    from .models import ExecutionStatus

    tid = str(template_id or "").strip()
    if not tid:
        return None

    candidates: List[tuple] = []
    for execution in store.list_executions(template_id=tid):
        if exclude_execution_id and execution.id == exclude_execution_id:
            continue
        if execution.status != ExecutionStatus.COMPLETED:
            continue
        if not _execution_eligible_for_template_learning(execution):
            continue
        skills = list_execution_pfm_skills(store, execution.id)
        if not _skills_ready_for_injection(skills):
            continue
        sort_key = int(execution.start_time or 0)
        candidates.append((sort_key, execution))

    if not candidates:
        return None

    candidates.sort(key=lambda row: row[0], reverse=True)
    return candidates[0][1]


def resolve_template_skills_for_injection(
    store: "ProjectStore",
    template_id: str,
) -> Optional[Dict[str, Any]]:
    """Latest completed eligible execution with both PFM skills (post-login inject)."""
    source = resolve_template_learning_source_execution(store, template_id)
    if not source:
        return None

    skills = list_execution_pfm_skills(store, source.id)
    by_name = {str(s.get("name") or ""): s for s in skills}
    return {
        "source_execution_id": source.id,
        "map_path": str(by_name[SKILL_MAP_NAME].get("local_path") or local_skill_md_path(SKILL_MAP_NAME)),
        "reports_path": str(
            by_name[SKILL_REPORTS_NAME].get("local_path") or local_skill_md_path(SKILL_REPORTS_NAME)
        ),
        "skill_names": list(KNOWN_SKILLS),
    }


def build_post_login_skills_inject_message(
    store: "ProjectStore",
    template_id: str,
) -> Optional[str]:
    resolved = resolve_template_skills_for_injection(store, template_id)
    if not resolved:
        return None
    source_id = resolved["source_execution_id"]
    map_path = resolved["map_path"]
    reports_path = resolved["reports_path"]
    return "\n".join(
        [
            PFM_POST_LOGIN_SKILLS_MARKER,
            "",
            "Read these Hermes PFM skills before Phase II:",
            f"- **kloud-pfm-map:** `{map_path}`",
            f"- **kloud-pfm-reports:** `{reports_path}`",
            "",
            f"Use the Read tool on both SKILL.md files now. "
            f"They are baseline structure and per-node reports from run `{source_id}`.",
            "Reading them does **not** complete this run.",
        ]
    ).strip()


def list_execution_pfm_skills(store: "ProjectStore", execution_id: str) -> List[Dict[str, Any]]:
    rows: Dict[str, Dict[str, Any]] = {}
    for art in store.list_execution_pfm_artifacts(execution_id):
        if str(art.get("artifact_type") or "") != PFM_SKILL_ARTIFACT_TYPE:
            continue
        key = str(art.get("skill_name") or art.get("artifact_key") or "").replace("skill-", "", 1)
        if key:
            rows[key] = art

    out: List[Dict[str, Any]] = []
    for name in KNOWN_SKILLS:
        local_path = local_skill_md_path(name)
        art = rows.get(name)
        artifact_key = skill_artifact_key(name)
        out.append(
            {
                "name": name,
                "artifact_key": artifact_key,
                "local_path": str(local_path),
                "local_exists": local_path.is_file(),
                "in_database": bool(art and str(art.get("content") or "").strip()),
                "updated_at": int(art.get("updated_at") or 0) if art else 0,
                "download_path": f"/v1/projects/executions/{execution_id}/reports/{artifact_key}",
            }
        )
    return out


def _format_bytes(size: Optional[int]) -> str:
    if size is None or size < 0:
        return ""
    if size < 1024:
        return f"{size} B"
    if size < 1024 * 1024:
        return f"{size / 1024:.1f} KB"
    return f"{size / (1024 * 1024):.1f} MB"


def _resolve_byte_size(disk_path: Path, art: Optional[Dict[str, Any]]) -> Optional[int]:
    if disk_path.is_file():
        try:
            return int(disk_path.stat().st_size)
        except OSError:
            pass
    if art:
        raw = str(art.get("content") or "")
        if raw:
            return len(raw.encode("utf-8"))
        b64 = art.get("content_base64")
        if b64:
            return int(len(str(b64)) * 3 / 4)
    return None


def _download_item(
    *,
    name: str,
    description: str,
    download_path: str,
    artifact_key: str = "",
    local_path: str = "",
    in_database: bool = False,
    local_exists: bool = False,
    size_bytes: Optional[int] = None,
) -> Dict[str, Any]:
    return {
        "name": name,
        "description": description,
        "artifact_key": artifact_key,
        "download_path": download_path,
        "local_path": local_path,
        "in_database": in_database,
        "local_exists": local_exists,
        "size_bytes": size_bytes,
        "size_label": _format_bytes(size_bytes),
    }


def list_execution_download_catalog(
    store: "ProjectStore", execution_id: str
) -> List[Dict[str, Any]]:
    """Grouped downloadable files for the Explore UI download modal."""
    from .pfm_artifacts import report_file_path

    execution = store.get_execution(execution_id)
    if not execution:
        return []

    skill_items = list_execution_pfm_skills(store, execution_id)
    skills_cat = {
        "id": "skills",
        "title": "Skill files",
        "subtitle": "For future agent injection",
        "items": [
            _download_item(
                name=str(row["name"]),
                description="Hermes skill (SKILL.md)",
                download_path=str(row["download_path"]),
                artifact_key=str(row.get("artifact_key") or ""),
                local_path=str(row.get("local_path") or ""),
                in_database=bool(row.get("in_database")),
                local_exists=bool(row.get("local_exists")),
                size_bytes=_resolve_byte_size(
                    Path(str(row.get("local_path") or "")),
                    store.get_execution_pfm_artifact(execution_id, str(row.get("artifact_key") or "")),
                ),
            )
            for row in skill_items
            if row.get("in_database") or row.get("local_exists")
        ],
    }

    report_type_labels = {
        "pfm_report": "Full PFM report (Markdown)",
        "pfm_mindmap": "Mermaid mindmap",
        "pfm_tree": "Committed PFM tree (JSON)",
        "pfm_schema_file": "Structured schema (.pfm)",
        "pfm_report_file": "Node EAD findings (.FMR)",
        "pfm_schema_pdf": "PFM mindmap (PDF)",
        "pfm_report_pdf": "Full report (PDF)",
    }
    deliverable_types = frozenset(report_type_labels.keys())

    report_items: List[Dict[str, Any]] = []
    node_report_items: List[Dict[str, Any]] = []
    seen_keys: set[str] = set()

    for art in store.list_execution_pfm_artifacts(execution_id):
        artifact_type = str(art.get("artifact_type") or "").strip()
        artifact_key = str(art.get("artifact_key") or art.get("filename") or "").strip()
        if not artifact_key or artifact_key in seen_keys:
            continue
        disk_path = report_file_path(execution_id, artifact_key)
        in_db = bool(str(art.get("content") or "").strip() or art.get("content_base64"))
        on_disk = disk_path.is_file()
        if not in_db and not on_disk:
            continue
        seen_keys.add(artifact_key)
        item = _download_item(
            name=artifact_key,
            description=report_type_labels.get(artifact_type, str(art.get("title") or artifact_type)),
            download_path=f"/v1/projects/executions/{execution_id}/reports/{artifact_key}",
            artifact_key=artifact_key,
            local_path=str(disk_path),
            in_database=in_db,
            local_exists=on_disk,
            size_bytes=_resolve_byte_size(disk_path, art),
        )
        if artifact_type == "node_ead_report":
            title = str(art.get("title") or art.get("node_key") or artifact_key)
            item["name"] = title
            item["description"] = "Per-node EAD report"
            node_report_items.append(item)
        elif artifact_type in deliverable_types or artifact_type == "pfm_skill":
            if artifact_type != "pfm_skill":
                report_items.append(item)

    for rep in execution.reports or []:
        filename = str(getattr(rep, "filename", None) or (rep.get("filename") if isinstance(rep, dict) else "") or "").strip()
        if not filename or filename in seen_keys:
            continue
        url = str(getattr(rep, "url", None) or (rep.get("url") if isinstance(rep, dict) else "") or "").strip()
        title = str(getattr(rep, "title", None) or (rep.get("title") if isinstance(rep, dict) else "") or filename)
        disk_path = report_file_path(execution_id, filename)
        on_disk = disk_path.is_file()
        art = store.get_execution_pfm_artifact(execution_id, filename)
        in_db = bool(art)
        if not on_disk and not in_db and not url:
            continue
        seen_keys.add(filename)
        report_items.append(
            _download_item(
                name=filename,
                description=title,
                download_path=url or f"/v1/projects/executions/{execution_id}/reports/{filename}",
                artifact_key=filename,
                local_path=str(disk_path) if on_disk else "",
                in_database=in_db,
                local_exists=on_disk,
                size_bytes=_resolve_byte_size(disk_path, art),
            )
        )

    report_items.sort(key=lambda row: row.get("name") or "")
    node_report_items.sort(key=lambda row: row.get("name") or "")

    screenshot_items: List[Dict[str, Any]] = []
    seen_shots: set[str] = set()
    for entry in execution.progress_log or []:
        image_url = str(getattr(entry, "image_url", None) or getattr(entry, "imageUrl", "") or "").strip()
        if not image_url:
            image_url = str(getattr(entry, "thumbnail_url", None) or getattr(entry, "thumbnailUrl", "") or "").strip()
        if not image_url or image_url in seen_shots:
            continue
        seen_shots.add(image_url)
        label = str(getattr(entry, "text", None) or getattr(entry, "text", "") or "Screenshot").strip()
        if len(label) > 80:
            label = label[:77] + "…"
        name = Path(image_url.split("?")[0]).name or "screenshot.png"
        screenshot_items.append(
            _download_item(
                name=name,
                description=label or "Run screenshot",
                download_path=image_url,
                local_path=image_url if image_url.startswith("/") else "",
                local_exists=image_url.startswith("http") or image_url.startswith("/"),
                in_database=False,
                size_bytes=None,
            )
        )

    categories: List[Dict[str, Any]] = []
    if skills_cat["items"]:
        categories.append(skills_cat)
    if report_items:
        categories.append(
            {
                "id": "reports",
                "title": "Reports",
                "subtitle": "For human reading",
                "items": report_items,
            }
        )
    if node_report_items:
        categories.append(
            {
                "id": "node_reports",
                "title": "Node EAD reports",
                "subtitle": f"{len(node_report_items)} per-node files",
                "items": node_report_items,
            }
        )
    if screenshot_items:
        categories.append(
            {
                "id": "screenshots",
                "title": "Screenshots",
                "subtitle": f"{len(screenshot_items)} captures",
                "items": screenshot_items,
            }
        )
    return categories


def list_execution_pfm_report_downloads(
    store: "ProjectStore", execution_id: str
) -> List[Dict[str, Any]]:
    """Downloadable PFM report / tree artifacts (not skill files)."""
    from .pfm_artifacts import report_file_path

    allowed_types = frozenset({"pfm_report", "pfm_mindmap", "pfm_tree", "node_ead_report"})
    out: List[Dict[str, Any]] = []
    for art in store.list_execution_pfm_artifacts(execution_id):
        artifact_type = str(art.get("artifact_type") or "").strip()
        if artifact_type not in allowed_types:
            continue
        artifact_key = str(art.get("artifact_key") or art.get("filename") or "").strip()
        if not artifact_key:
            continue
        disk_path = report_file_path(execution_id, artifact_key)
        in_db = bool(str(art.get("content") or "").strip() or art.get("content_base64"))
        on_disk = disk_path.is_file()
        if not in_db and not on_disk:
            continue
        label = str(art.get("title") or artifact_key).strip()
        if artifact_type == "node_ead_report":
            node_key = str(art.get("node_key") or "").strip()
            if node_key:
                label = f"Node report: {art.get('title') or node_key}"
        out.append(
            {
                "name": label,
                "artifact_key": artifact_key,
                "artifact_type": artifact_type,
                "local_exists": on_disk,
                "in_database": in_db,
                "download_path": f"/v1/projects/executions/{execution_id}/reports/{artifact_key}",
            }
        )
    out.sort(key=lambda row: (row.get("artifact_type") or "", row.get("name") or ""))
    return out


def create_execution_pfm_skills(store: "ProjectStore", execution_id: str) -> Dict[str, Any]:
    execution = store.get_execution(execution_id)
    if not execution:
        return {"ok": False, "error": "execution_not_found"}

    snapshot = store.get_committed_pfm_tree(execution_id)
    if not snapshot:
        return {
            "ok": False,
            "error": "no_committed_tree",
            "message": "Commit a PFM tree snapshot before creating skill files.",
        }

    template = store.get_template(execution.linked_template_id)
    template_name = (template.name if template else None) or execution.name or execution.linked_template_id

    reports_by_key = collect_node_report_markdown_by_key(store, execution_id)
    now_ms = int(time.time() * 1000)

    payloads = [
        (
            SKILL_MAP_NAME,
            build_map_skill_markdown(
                template_name=template_name,
                execution_id=execution_id,
                snapshot=snapshot,
            ),
        ),
        (
            SKILL_REPORTS_NAME,
            build_reports_skill_markdown(
                template_name=template_name,
                execution_id=execution_id,
                reports_by_key=reports_by_key,
                snapshot=snapshot,
            ),
        ),
    ]

    created: List[Dict[str, Any]] = []
    for skill_name, content in payloads:
        local_path = write_local_skill(skill_name, content)
        artifact_key = skill_artifact_key(skill_name)
        data = {
            "artifact_key": artifact_key,
            "artifact_type": PFM_SKILL_ARTIFACT_TYPE,
            "skill_name": skill_name,
            "filename": f"{skill_name}-SKILL.md",
            "format": "markdown",
            "content": content,
            "title": f"{skill_name} SKILL.md",
            "local_path": str(local_path),
            "updated_at": now_ms,
            "execution_id": execution_id,
            "template_id": execution.linked_template_id,
        }
        store.upsert_execution_pfm_artifact(
            execution_id,
            artifact_key,
            PFM_SKILL_ARTIFACT_TYPE,
            data,
        )
        created.append(
            {
                "name": skill_name,
                "artifact_key": artifact_key,
                "local_path": str(local_path),
            }
        )

    if (
        execution.contributes_to_learning is not False
        and execution.valid_for_data_reporting_training is not False
    ):
        store.publish_execution_artifacts_to_template(execution_id)

    return {
        "ok": True,
        "execution_id": execution_id,
        "skills": created,
        "skills_status": list_execution_pfm_skills(store, execution_id),
    }
