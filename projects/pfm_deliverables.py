"""
Portable PFM/FMR deliverable generation.

Creates four run-finish artifacts:
- <base>.pfm: machine-readable PFM schema JSON
- <base>.FMR: structured full report Markdown
- <base>_PFM.pdf: readable mindmap PDF
- <base>_FMR.pdf: readable full node EAD report PDF
"""

from __future__ import annotations

import base64
import json
import logging
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from .models import EadFmNodeRun, ProjectExecute, ProjectReportArtifact
from .pfm_artifacts import derive_node_runs_from_progress, reports_root_dir

logger = logging.getLogger(__name__)


def generate_pfm_deliverables(store, execution_id: str) -> List[ProjectReportArtifact]:
    """Generate and persist portable deliverables for a finished execution."""
    execution = store.get_execution(execution_id)
    if not execution:
        return []

    template = store.get_template(execution.linked_template_id)
    nodes = execution.results or derive_node_runs_from_progress(execution.progress_log or [])
    artifacts = store.list_execution_pfm_artifacts(execution.id)
    node_reports = [
        artifact for artifact in artifacts if artifact.get("artifact_type") == "node_ead_report"
    ]

    base_name = _export_base_name(template.name if template else execution.name, execution.id)
    out_dir = reports_root_dir() / execution.id
    out_dir.mkdir(parents=True, exist_ok=True)

    schema = _build_pfm_schema(execution, template, nodes, artifacts)
    fmr = _build_fmr_markdown(execution, template, nodes, node_reports)

    pfm_path = out_dir / f"{base_name}.pfm"
    fmr_path = out_dir / f"{base_name}.FMR"
    pfm_pdf_path = out_dir / f"{base_name}_PFM.pdf"
    fmr_pdf_path = out_dir / f"{base_name}_FMR.pdf"

    pfm_path.write_text(json.dumps(schema, ensure_ascii=False, indent=2), encoding="utf-8")
    fmr_path.write_text(fmr, encoding="utf-8")

    _write_pfm_pdf(pfm_pdf_path, execution, template, nodes)
    _write_fmr_pdf(fmr_pdf_path, execution, template, nodes, node_reports)

    created_at = int(time.time() * 1000)
    report_artifacts = [
        ProjectReportArtifact(
            title="PFM Schema",
            filename=pfm_path.name,
            format="pfm",
            created_at=created_at,
            url=f"/v1/projects/executions/{execution.id}/reports/{pfm_path.name}",
        ),
        ProjectReportArtifact(
            title="PFM Full Report",
            filename=fmr_path.name,
            format="FMR",
            created_at=created_at,
            url=f"/v1/projects/executions/{execution.id}/reports/{fmr_path.name}",
        ),
        ProjectReportArtifact(
            title="PFM Mindmap PDF",
            filename=pfm_pdf_path.name,
            format="pdf",
            created_at=created_at,
            url=f"/v1/projects/executions/{execution.id}/reports/{pfm_pdf_path.name}",
        ),
        ProjectReportArtifact(
            title="PFM Full Node EAD Report PDF",
            filename=fmr_pdf_path.name,
            format="pdf",
            created_at=created_at,
            url=f"/v1/projects/executions/{execution.id}/reports/{fmr_pdf_path.name}",
        ),
    ]

    for path, artifact_type, title, fmt in [
        (pfm_path, "pfm_schema_file", "PFM Schema", "pfm"),
        (fmr_path, "pfm_report_file", "PFM Full Report", "FMR"),
    ]:
        store.upsert_execution_pfm_artifact(
            execution.id,
            path.name,
            artifact_type,
            {
                "filename": path.name,
                "format": fmt,
                "title": title,
                "content": path.read_text(encoding="utf-8"),
                "created_at": created_at,
            },
        )

    for path, artifact_type, title in [
        (pfm_pdf_path, "pfm_schema_pdf", "PFM Mindmap PDF"),
        (fmr_pdf_path, "pfm_report_pdf", "PFM Full Node EAD Report PDF"),
    ]:
        store.upsert_execution_pfm_artifact(
            execution.id,
            path.name,
            artifact_type,
            {
                "filename": path.name,
                "format": "pdf",
                "title": title,
                "content_encoding": "base64",
                "content_base64": base64.b64encode(path.read_bytes()).decode("ascii"),
                "byte_size": path.stat().st_size,
                "created_at": created_at,
            },
        )

    if execution.contributes_to_learning is not False:
        store.publish_execution_artifacts_to_template(execution.id)

    return report_artifacts


def _build_pfm_schema(
    execution: ProjectExecute,
    template,
    nodes: List[EadFmNodeRun],
    artifacts: List[Dict[str, Any]],
) -> Dict[str, Any]:
    return {
        "schemaVersion": "1.0",
        "artifactType": "pfm_schema",
        "fileExtension": ".pfm",
        "generatedAt": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "template": {
            "templateId": execution.linked_template_id,
            "templateName": getattr(template, "name", "") if template else "",
            "templateDescription": getattr(template, "description", "") if template else "",
        },
        "sourceExecution": {
            "executionId": execution.id,
            "name": execution.name,
            "status": execution.status.value,
            "runPurpose": execution.run_purpose,
            "evidenceSource": execution.evidence_source,
            "contributesToLearning": execution.contributes_to_learning,
            "targetUrl": execution.target_url,
            "exploreType": execution.explore_type,
            "startTime": execution.start_time,
            "durationMs": execution.duration_ms,
        },
        "nodes": [_node_to_schema(node) for node in nodes],
        "artifacts": [_artifact_ref(artifact) for artifact in artifacts],
        "metrics": {
            "nodeCount": len(nodes),
            "testCaseCount": sum(len(node.test_case_runs or []) for node in nodes),
            "testStepCount": sum(
                len(tc.test_case_step_runs or [])
                for node in nodes
                for tc in (node.test_case_runs or [])
            ),
            "progressEventCount": len(execution.progress_log or []),
            "nodeEadReportCount": len(
                [artifact for artifact in artifacts if artifact.get("artifact_type") == "node_ead_report"]
            ),
        },
    }


def _build_fmr_markdown(
    execution: ProjectExecute,
    template,
    nodes: List[EadFmNodeRun],
    node_reports: List[Dict[str, Any]],
) -> str:
    reports_by_node_key = {
        str(report.get("node_key") or ""): report
        for report in node_reports
        if report.get("node_key")
    }
    template_name = getattr(template, "name", "") if template else execution.name
    lines = [
        f"# PFM Full Node EAD Report: {template_name}",
        "",
        "## Summary",
        f"- File Format: `.FMR`",
        f"- Schema Version: `1.0`",
        f"- Template ID: `{execution.linked_template_id}`",
        f"- Source Execution: `{execution.id}`",
        f"- Execution Status: `{execution.status.value}`",
        f"- Node Count: `{len(nodes)}`",
        f"- Node EAD Report Count: `{len(node_reports)}`",
        "",
        "## Per-Node EAD Reports",
    ]

    for idx, node in enumerate(nodes, start=1):
        title = _clean_text(node.title or node.node_key or node.node_id, 140)
        lines.extend(
            [
                "",
                f"### Node {idx}: {title}",
                f"- Node Key: `{node.node_key}`",
                f"- Status: `{_status_value(node.status)}`",
                "",
                f"Description: {_node_description(node)}",
                "",
                "#### Node EAD Report",
            ]
        )
        report = reports_by_node_key.get(node.node_key)
        content = str((report or {}).get("content") or "").strip()
        if content:
            lines.append(content)
        else:
            lines.append(
                "No saved Node EAD report exists for this PFM node in this run yet. "
                "Hermes will populate this section after the per-node EAD report is generated."
            )

    return "\n".join(lines).rstrip() + "\n"


def _write_pfm_pdf(path: Path, execution: ProjectExecute, template, nodes: List[EadFmNodeRun]) -> None:
    doc, styles = _pdf_doc(path)
    story = _cover_story(
        styles,
        "PFM Mindmap",
        getattr(template, "name", "") if template else execution.name,
        [
            ("Nodes", str(len(nodes))),
            ("Run", execution.status.value),
            ("Format", "PFM"),
            ("Theme", "UI"),
        ],
        "Mindmap-focused PDF companion for the portable .pfm file. Leaf nodes do not show artificial submaps.",
    )
    story.extend([_h2(styles, "Main PFM Mindmap"), _mindmap_table(styles, nodes), Spacer(1, 0.18 * inch)])
    story.append(_h2(styles, "Node Descriptions"))
    for idx, node in enumerate(nodes, start=1):
        story.extend(_node_pdf_story(styles, idx, node, include_report=False))
    doc.build(story)


def _write_fmr_pdf(
    path: Path,
    execution: ProjectExecute,
    template,
    nodes: List[EadFmNodeRun],
    node_reports: List[Dict[str, Any]],
) -> None:
    reports_by_node_key = {
        str(report.get("node_key") or ""): report
        for report in node_reports
        if report.get("node_key")
    }
    doc, styles = _pdf_doc(path)
    story = _cover_story(
        styles,
        "FMR Full Node EAD Report",
        getattr(template, "name", "") if template else execution.name,
        [
            ("Nodes", str(len(nodes))),
            ("Node Reports", str(len(node_reports))),
            ("Run", execution.status.value),
            ("Format", "FMR"),
        ],
        "Human-readable companion for the portable .FMR file. Each PFM node has a Node EAD Report section.",
    )
    story.extend([_h2(styles, "PFM Mindmap Reference"), _mindmap_table(styles, nodes), PageBreak()])
    story.append(_h2(styles, "Per-Node EAD Reports"))
    for idx, node in enumerate(nodes, start=1):
        report = reports_by_node_key.get(node.node_key)
        story.extend(_node_pdf_story(styles, idx, node, include_report=True, report=report))
    doc.build(story)


def _pdf_doc(path: Path):
    styles = getSampleStyleSheet()
    styles.add(
        ParagraphStyle(
            name="CoverEyebrow",
            parent=styles["Normal"],
            fontSize=9,
            leading=12,
            textColor=colors.HexColor("#9aa6bd"),
            spaceAfter=8,
        )
    )
    styles.add(
        ParagraphStyle(
            name="CoverTitle",
            parent=styles["Title"],
            fontSize=26,
            leading=31,
            textColor=colors.HexColor("#f4f4f5"),
            spaceAfter=8,
        )
    )
    styles.add(
        ParagraphStyle(
            name="BodyDark",
            parent=styles["BodyText"],
            fontSize=10,
            leading=14,
            textColor=colors.HexColor("#c6d0e1"),
        )
    )
    styles.add(
        ParagraphStyle(
            name="H2Dark",
            parent=styles["Heading2"],
            fontSize=16,
            leading=20,
            textColor=colors.HexColor("#f4f4f5"),
            spaceBefore=10,
            spaceAfter=8,
        )
    )
    styles.add(
        ParagraphStyle(
            name="H3Dark",
            parent=styles["Heading3"],
            fontSize=12,
            leading=15,
            textColor=colors.HexColor("#f4f4f5"),
            spaceBefore=8,
            spaceAfter=4,
        )
    )
    doc = SimpleDocTemplate(
        str(path),
        pagesize=letter,
        rightMargin=0.55 * inch,
        leftMargin=0.55 * inch,
        topMargin=0.55 * inch,
        bottomMargin=0.55 * inch,
    )
    return doc, styles


def _cover_story(styles, label: str, title: str, metrics: List[tuple[str, str]], subtitle: str):
    story = [
        Paragraph(label.upper(), styles["CoverEyebrow"]),
        Paragraph(_escape(title), styles["CoverTitle"]),
        Paragraph(_escape(subtitle), styles["BodyDark"]),
        Spacer(1, 0.2 * inch),
    ]
    cells = [
        [
            Paragraph(f"<b>{_escape(name)}</b><br/>{_escape(value)}", styles["BodyDark"])
            for name, value in metrics
        ]
    ]
    table = Table(cells, colWidths=[1.55 * inch] * len(metrics))
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#1b1f2a")),
                ("BOX", (0, 0), (-1, -1), 0.7, colors.HexColor("#303a4d")),
                ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#303a4d")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 10),
                ("RIGHTPADDING", (0, 0), (-1, -1), 10),
                ("TOPPADDING", (0, 0), (-1, -1), 10),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
            ]
        )
    )
    story.extend([table, Spacer(1, 0.22 * inch)])
    return story


def _mindmap_table(styles, nodes: List[EadFmNodeRun]):
    rows = [[Paragraph("<b>PFM Node</b>", styles["BodyDark"]), Paragraph("<b>Description</b>", styles["BodyDark"])]]
    for node in nodes[:60]:
        rows.append(
            [
                Paragraph(_escape(_clean_text(node.title or node.node_key, 90)), styles["BodyDark"]),
                Paragraph(_escape(_node_description(node)), styles["BodyDark"]),
            ]
        )
    table = Table(rows, colWidths=[2.35 * inch, 4.1 * inch], repeatRows=1)
    table.setStyle(_dark_table_style())
    return table


def _node_pdf_story(styles, idx: int, node: EadFmNodeRun, include_report: bool, report: Optional[Dict[str, Any]] = None):
    title = _clean_text(node.title or node.node_key or node.node_id, 140)
    story = [
        Paragraph(f"{idx:02d}. {_escape(title)}", styles["H3Dark"]),
        Paragraph(f"<b>Node Key:</b> {_escape(node.node_key)}", styles["BodyDark"]),
        Paragraph(f"<b>Description:</b> {_escape(_node_description(node))}", styles["BodyDark"]),
        Spacer(1, 0.08 * inch),
    ]
    if include_report:
        content = str((report or {}).get("content") or "").strip()
        story.append(Paragraph("<b>Node EAD Report</b>", styles["BodyDark"]))
        if content:
            for paragraph in _markdownish_paragraphs(content):
                story.append(Paragraph(_escape(paragraph), styles["BodyDark"]))
        else:
            story.append(
                Paragraph(
                    "No saved Node EAD report exists for this PFM node in this run yet. "
                    "Hermes will populate this section after the per-node EAD report is generated.",
                    styles["BodyDark"],
                )
            )
    story.append(Spacer(1, 0.12 * inch))
    return story


def _dark_table_style() -> TableStyle:
    return TableStyle(
        [
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#304b73")),
            ("BACKGROUND", (0, 1), (-1, -1), colors.HexColor("#161920")),
            ("TEXTCOLOR", (0, 0), (-1, -1), colors.HexColor("#e6ebf4")),
            ("BOX", (0, 0), (-1, -1), 0.7, colors.HexColor("#303a4d")),
            ("INNERGRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#303a4d")),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LEFTPADDING", (0, 0), (-1, -1), 8),
            ("RIGHTPADDING", (0, 0), (-1, -1), 8),
            ("TOPPADDING", (0, 0), (-1, -1), 7),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
        ]
    )


def _h2(styles, text: str):
    return Paragraph(_escape(text), styles["H2Dark"])


def _node_to_schema(node: EadFmNodeRun) -> Dict[str, Any]:
    return {
        "nodeId": node.node_id,
        "nodeKey": node.node_key,
        "parentNodeKey": node.parent_node_key,
        "level": node.level,
        "type": node.type,
        "title": node.title,
        "meta": node.meta,
        "status": _status_value(node.status),
        "testCases": [
            {
                "caseId": tc.case_id,
                "title": tc.title,
                "status": _status_value(tc.status),
                "steps": [
                    {
                        "stepId": step.step_id,
                        "sortOrder": step.sort_order,
                        "action": step.procedure_text,
                        "expectedResult": step.expected_result,
                        "status": _status_value(step.status),
                        "actualResult": step.actual_result,
                        "screenshotUrl": step.screenshot_url,
                    }
                    for step in (tc.test_case_step_runs or [])
                ],
            }
            for tc in (node.test_case_runs or [])
        ],
    }


def _artifact_ref(artifact: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "artifactKey": artifact.get("artifact_key"),
        "artifactType": artifact.get("artifact_type"),
        "title": artifact.get("title"),
        "filename": artifact.get("filename"),
        "format": artifact.get("format"),
        "nodeKey": artifact.get("node_key"),
        "inherited": bool(artifact.get("inherited")),
        "sourceExecutionId": artifact.get("source_execution_id"),
        "inheritedFromExecutionId": artifact.get("inherited_from_execution_id"),
    }


def _export_base_name(name: str, execution_id: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", str(name or "pfm").lower()).strip("-")
    return f"{slug or 'pfm'}-{execution_id[:8]}"


def _clean_text(text: Any, limit: int = 120) -> str:
    value = re.sub(r"\s+", " ", str(text or "")).strip()
    value = re.sub(r"[`*_#|<>\\[\\]{}]+", " ", value)
    value = re.sub(r"\s+", " ", value).strip()
    if len(value) > limit:
        value = value[: limit - 3].rstrip() + "..."
    return value or "Untitled PFM Node"


def _node_description(node: EadFmNodeRun) -> str:
    if node.meta:
        return _clean_text(node.meta, 260)
    return (
        "PFM node identified from run evidence. Future learning runs can verify, "
        f"refine, or replace this node: {_clean_text(node.title or node.node_key, 160)}."
    )


def _markdownish_paragraphs(content: str) -> List[str]:
    paragraphs: List[str] = []
    for raw in str(content or "").splitlines():
        line = raw.strip()
        if not line:
            continue
        line = re.sub(r"^#{1,6}\s*", "", line)
        line = re.sub(r"^[-*]\s+", "", line)
        paragraphs.append(line)
        if len(paragraphs) >= 28:
            paragraphs.append("...")
            break
    return paragraphs


def _status_value(value: Any) -> str:
    return str(getattr(value, "value", value) or "")


def _escape(value: Any) -> str:
    return (
        str(value or "")
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )
