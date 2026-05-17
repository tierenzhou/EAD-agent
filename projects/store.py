"""
SQLite-backed project template and execution storage.

Follows the same patterns as hermes_state.SessionDB:
- WAL mode for concurrent readers + single writer
- Application-level retry with jitter on write contention
- Thread-safe via lock
"""

import json
import logging
import random
import re
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from .models import (
    EadFmNodeRun,
    ExecutionStatus,
    ProgressLogEntry,
    ProjectExecute,
    ProjectReportArtifact,
    ProjectTemplate,
)

logger = logging.getLogger(__name__)


def _eligible_for_pfm_run_inheritance(ex: ProjectExecute) -> bool:
    """Match EAD_ExpUI pfm-run-eligibility: invalid or non-learning runs are not sources."""
    if ex.valid_for_data_reporting_training is False:
        return False
    if ex.contributes_to_learning is False:
        return False
    return True


_DEFAULT_DB_DIR = Path.home() / ".hermes" / "projects"
_DEFAULT_DB_PATH = _DEFAULT_DB_DIR / "projects.db"

_SCHEMA_VERSION = 1

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS schema_version (
    version INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS templates (
    id TEXT PRIMARY KEY,
    data TEXT NOT NULL,
    created_at INTEGER NOT NULL,
    last_modified_at INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS executions (
    id TEXT PRIMARY KEY,
    linked_template_id TEXT NOT NULL,
    data TEXT NOT NULL,
    status TEXT NOT NULL,
    created_at INTEGER NOT NULL,
    FOREIGN KEY (linked_template_id) REFERENCES templates(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_executions_template ON executions(linked_template_id);
CREATE INDEX IF NOT EXISTS idx_executions_status ON executions(status);

CREATE TABLE IF NOT EXISTS template_learning_summaries (
    template_id TEXT NOT NULL,
    execution_id TEXT NOT NULL,
    data TEXT NOT NULL,
    created_at INTEGER NOT NULL,
    updated_at INTEGER NOT NULL,
    PRIMARY KEY (template_id, execution_id),
    FOREIGN KEY (template_id) REFERENCES templates(id) ON DELETE CASCADE,
    FOREIGN KEY (execution_id) REFERENCES executions(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_template_learning_summaries_template
    ON template_learning_summaries(template_id, updated_at DESC);

CREATE TABLE IF NOT EXISTS execution_pfm_artifacts (
    execution_id TEXT NOT NULL,
    artifact_key TEXT NOT NULL,
    template_id TEXT NOT NULL,
    artifact_type TEXT NOT NULL,
    data TEXT NOT NULL,
    inherited_from_execution_id TEXT,
    created_at INTEGER NOT NULL,
    updated_at INTEGER NOT NULL,
    PRIMARY KEY (execution_id, artifact_key),
    FOREIGN KEY (execution_id) REFERENCES executions(id) ON DELETE CASCADE,
    FOREIGN KEY (template_id) REFERENCES templates(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_execution_pfm_artifacts_template
    ON execution_pfm_artifacts(template_id, updated_at DESC);

CREATE TABLE IF NOT EXISTS template_pfm_artifacts (
    template_id TEXT NOT NULL,
    artifact_key TEXT NOT NULL,
    artifact_type TEXT NOT NULL,
    data TEXT NOT NULL,
    source_execution_id TEXT,
    created_at INTEGER NOT NULL,
    updated_at INTEGER NOT NULL,
    PRIMARY KEY (template_id, artifact_key),
    FOREIGN KEY (template_id) REFERENCES templates(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_template_pfm_artifacts_updated
    ON template_pfm_artifacts(template_id, updated_at DESC);

CREATE TABLE IF NOT EXISTS active_state (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""

_WRITE_MAX_RETRIES = 10
_WRITE_RETRY_MIN_S = 0.020
_WRITE_RETRY_MAX_S = 0.150


class ProjectStore:
    """SQLite-backed project template and execution storage."""

    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = Path(db_path) if db_path else _DEFAULT_DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        self._lock = threading.Lock()
        self._conn = sqlite3.connect(
            str(self.db_path),
            check_same_thread=False,
            timeout=1.0,
            isolation_level=None,
        )
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._init_schema()

    def _init_schema(self) -> None:
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                self._conn.executescript(_SCHEMA_SQL)
                row = self._conn.execute(
                    "SELECT version FROM schema_version"
                ).fetchone()
                if row is None:
                    self._conn.execute(
                        "INSERT INTO schema_version (version) VALUES (?)",
                        (_SCHEMA_VERSION,),
                    )
                self._conn.commit()
            except BaseException:
                try:
                    self._conn.rollback()
                except Exception:
                    pass
                raise

    def _execute_write(self, fn):
        last_err = None
        for attempt in range(_WRITE_MAX_RETRIES):
            try:
                with self._lock:
                    self._conn.execute("BEGIN IMMEDIATE")
                    try:
                        result = fn(self._conn)
                        self._conn.commit()
                    except BaseException:
                        try:
                            self._conn.rollback()
                        except Exception:
                            pass
                        raise
                return result
            except sqlite3.OperationalError as exc:
                err_msg = str(exc).lower()
                if "locked" in err_msg or "busy" in err_msg:
                    last_err = exc
                    delay = random.uniform(_WRITE_RETRY_MIN_S, _WRITE_RETRY_MAX_S)
                    time.sleep(delay)
                    continue
                raise
        raise last_err

    # ------------------------------------------------------------------
    # Template CRUD
    # ------------------------------------------------------------------

    def list_templates(self) -> List[ProjectTemplate]:
        rows = self._conn.execute(
            "SELECT data FROM templates ORDER BY last_modified_at DESC"
        ).fetchall()
        return [ProjectTemplate.model_validate_json(r["data"]) for r in rows]

    def get_template(self, template_id: str) -> Optional[ProjectTemplate]:
        row = self._conn.execute(
            "SELECT data FROM templates WHERE id = ?", (template_id,)
        ).fetchone()
        if not row:
            return None
        return ProjectTemplate.model_validate_json(row["data"])

    def create_template(self, template: ProjectTemplate) -> ProjectTemplate:
        data = template.model_dump_json()

        def _write(conn):
            conn.execute(
                "INSERT INTO templates (id, data, created_at, last_modified_at) VALUES (?, ?, ?, ?)",
                (template.id, data, template.created_at, template.last_modified_at),
            )

        self._execute_write(_write)
        return template

    def update_template(self, template_id: str, **fields) -> Optional[ProjectTemplate]:
        template = self.get_template(template_id)
        if not template:
            return None

        for key, value in fields.items():
            if hasattr(template, key):
                setattr(template, key, value)

        template.last_modified_at = int(time.time() * 1000)
        data = template.model_dump_json()

        def _write(conn):
            conn.execute(
                "UPDATE templates SET data = ?, last_modified_at = ? WHERE id = ?",
                (data, template.last_modified_at, template_id),
            )

        self._execute_write(_write)
        return template

    def delete_template(self, template_id: str) -> bool:
        template = self.get_template(template_id)
        if not template:
            return False

        def _write(conn):
            conn.execute("DELETE FROM template_pfm_artifacts WHERE template_id = ?", (template_id,))
            conn.execute("DELETE FROM template_learning_summaries WHERE template_id = ?", (template_id,))
            conn.execute("DELETE FROM execution_pfm_artifacts WHERE template_id = ?", (template_id,))
            conn.execute("DELETE FROM executions WHERE linked_template_id = ?", (template_id,))
            conn.execute("DELETE FROM active_state WHERE key = 'active_template_id' AND value = ?", (template_id,))
            conn.execute("DELETE FROM templates WHERE id = ?", (template_id,))

        self._execute_write(_write)
        return True

    def get_active_template_id(self) -> Optional[str]:
        row = self._conn.execute(
            "SELECT value FROM active_state WHERE key = 'active_template_id'"
        ).fetchone()
        return row["value"] if row else None

    def set_active_template_id(self, template_id: Optional[str]) -> None:
        def _write(conn):
            if template_id is None:
                conn.execute("DELETE FROM active_state WHERE key = 'active_template_id'")
            else:
                conn.execute(
                    "INSERT OR REPLACE INTO active_state (key, value) VALUES (?, ?)",
                    ("active_template_id", template_id),
                )

        self._execute_write(_write)

    # ------------------------------------------------------------------
    # Execution CRUD
    # ------------------------------------------------------------------

    def list_executions(
        self, template_id: Optional[str] = None, status: Optional[str] = None
    ) -> List[ProjectExecute]:
        query = "SELECT data FROM executions WHERE 1=1"
        params = []
        if template_id:
            query += " AND linked_template_id = ?"
            params.append(template_id)
        if status:
            query += " AND status = ?"
            params.append(status)
        query += " ORDER BY created_at DESC"
        rows = self._conn.execute(query, params).fetchall()
        return [ProjectExecute.model_validate_json(r["data"]) for r in rows]

    def get_execution(self, execution_id: str) -> Optional[ProjectExecute]:
        row = self._conn.execute(
            "SELECT data FROM executions WHERE id = ?", (execution_id,)
        ).fetchone()
        if not row:
            return None
        return ProjectExecute.model_validate_json(row["data"])

    def create_execution(self, execution: ProjectExecute) -> ProjectExecute:
        data = execution.model_dump_json()
        created_at = execution.start_time or int(time.time() * 1000)

        def _write(conn):
            conn.execute(
                "INSERT INTO executions (id, linked_template_id, data, status, created_at) VALUES (?, ?, ?, ?, ?)",
                (execution.id, execution.linked_template_id, data, execution.status.value, created_at),
            )

        self._execute_write(_write)
        from .pfm_run_number import ensure_template_run_numbers

        ensure_template_run_numbers(self, execution.linked_template_id)
        return self.get_execution(execution.id) or execution

    def update_execution(self, execution_id: str, **fields) -> Optional[ProjectExecute]:
        execution = self.get_execution(execution_id)
        if not execution:
            return None

        for key, value in fields.items():
            if hasattr(execution, key):
                setattr(execution, key, value)

        data = execution.model_dump_json()

        def _write(conn):
            conn.execute(
                "UPDATE executions SET data = ?, status = ? WHERE id = ?",
                (data, execution.status.value, execution_id),
            )

        self._execute_write(_write)
        return execution

    def delete_execution(self, execution_id: str) -> bool:
        execution = self.get_execution(execution_id)
        if not execution:
            return False
        template_id = execution.linked_template_id

        def _write(conn):
            conn.execute(
                "DELETE FROM template_pfm_artifacts WHERE source_execution_id = ?",
                (execution_id,),
            )
            conn.execute(
                "DELETE FROM template_learning_summaries WHERE execution_id = ?",
                (execution_id,),
            )
            conn.execute(
                "DELETE FROM execution_pfm_artifacts WHERE execution_id = ?",
                (execution_id,),
            )
            conn.execute("DELETE FROM executions WHERE id = ?", (execution_id,))
            remaining = conn.execute(
                "SELECT COUNT(*) AS count FROM executions WHERE linked_template_id = ?",
                (template_id,),
            ).fetchone()
            if int(remaining["count"] if remaining else 0) == 0:
                conn.execute(
                    "DELETE FROM template_pfm_artifacts WHERE template_id = ?",
                    (template_id,),
                )
                conn.execute(
                    "DELETE FROM template_learning_summaries WHERE template_id = ?",
                    (template_id,),
                )

        self._execute_write(_write)
        if self.list_executions(template_id=template_id):
            self.rebuild_template_learning_from_latest_good_run(template_id)
        return True

    def set_execution_continuous_learning(
        self,
        execution_id: str,
        *,
        enabled: bool,
        reason: Optional[str] = None,
    ) -> Optional[ProjectExecute]:
        if not enabled:
            exclusion = (
                reason
                or "Excluded from interactive learning — this run did not represent good knowledge gain"
            ).strip()
            return self.invalidate_execution_learning(execution_id, reason=exclusion)

        execution = self.get_execution(execution_id)
        if not execution:
            return None

        updated = self.update_execution(
            execution_id,
            contributes_to_learning=True,
            valid_for_data_reporting_training=True,
            learning_exclusion_reason=None,
            invalid_for_data_reporting_training_reason=None,
        )
        if execution.linked_template_id:
            self.rebuild_template_learning_from_latest_good_run(execution.linked_template_id)
        return updated

    def invalidate_execution_learning(
        self, execution_id: str, reason: str = "Not valid for data reporting and training"
    ) -> Optional[ProjectExecute]:
        execution = self.get_execution(execution_id)
        if not execution:
            return None
        reason_text = reason or "Not valid for data reporting and training"
        updated = self.update_execution(
            execution_id,
            contributes_to_learning=False,
            learning_exclusion_reason=reason_text,
            valid_for_data_reporting_training=False,
            invalid_for_data_reporting_training_reason=reason_text,
        )
        tmpl = self.get_template(execution.linked_template_id)
        if tmpl and (tmpl.canonical_pfm_execution_id or "").strip() == execution_id:
            self.update_template(
                execution.linked_template_id,
                canonical_pfm_execution_id=None,
                canonical_pfm_promoted_at=None,
                canonical_pfm_promoted_by=None,
                canonical_pfm_promotion_rationale=None,
            )
        self.remove_execution_learning_contribution(execution_id, execution.linked_template_id)
        self.rebuild_template_learning_from_latest_good_run(execution.linked_template_id)
        return updated

    def remove_execution_learning_contribution(
        self, execution_id: str, template_id: Optional[str] = None
    ) -> None:
        if not template_id:
            execution = self.get_execution(execution_id)
            template_id = execution.linked_template_id if execution else None

        def _write(conn):
            conn.execute(
                "DELETE FROM template_pfm_artifacts WHERE source_execution_id = ?",
                (execution_id,),
            )
            conn.execute(
                "DELETE FROM template_learning_summaries WHERE execution_id = ?",
                (execution_id,),
            )
            if template_id:
                remaining = conn.execute(
                    "SELECT COUNT(*) AS count FROM executions WHERE linked_template_id = ?",
                    (template_id,),
                ).fetchone()
                if int(remaining["count"] if remaining else 0) == 0:
                    conn.execute(
                        "DELETE FROM template_pfm_artifacts WHERE template_id = ?",
                        (template_id,),
                    )
                    conn.execute(
                        "DELETE FROM template_learning_summaries WHERE template_id = ?",
                        (template_id,),
                    )

        self._execute_write(_write)

    def rebuild_template_learning_from_latest_good_run(self, template_id: str) -> None:
        if not template_id:
            return
        candidates = [
            execution
            for execution in self.list_executions(template_id=template_id)
            if execution.valid_for_data_reporting_training is not False
            and execution.contributes_to_learning is not False
        ]

        def _clear(conn):
            conn.execute(
                "DELETE FROM template_pfm_artifacts WHERE template_id = ?",
                (template_id,),
            )
            conn.execute(
                "DELETE FROM template_learning_summaries WHERE template_id = ?",
                (template_id,),
            )

        self._execute_write(_clear)
        if not candidates:
            return

        # The latest valid run becomes the template's current training source.
        latest = candidates[0]
        self.sync_execution_pfm_artifacts_from_state(latest.id)
        self.upsert_template_learning_summary(latest.id)

    # ------------------------------------------------------------------
    # PFM artifacts (DB-backed per execution and per template)
    # ------------------------------------------------------------------

    def upsert_execution_pfm_artifact(
        self,
        execution_id: str,
        artifact_key: str,
        artifact_type: str,
        data: Dict[str, Any],
        inherited_from_execution_id: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        execution = self.get_execution(execution_id)
        if not execution:
            return None

        payload = dict(data)
        payload.setdefault("execution_id", execution.id)
        payload.setdefault("template_id", execution.linked_template_id)
        payload.setdefault("artifact_key", artifact_key)
        payload.setdefault("artifact_type", artifact_type)
        if inherited_from_execution_id:
            payload["inherited_from_execution_id"] = inherited_from_execution_id

        now_ms = int(time.time() * 1000)
        serialized = json.dumps(payload, ensure_ascii=False)

        def _write(conn):
            conn.execute(
                """
                INSERT INTO execution_pfm_artifacts
                    (execution_id, artifact_key, template_id, artifact_type, data,
                     inherited_from_execution_id, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(execution_id, artifact_key) DO UPDATE SET
                    template_id = excluded.template_id,
                    artifact_type = excluded.artifact_type,
                    data = excluded.data,
                    inherited_from_execution_id = excluded.inherited_from_execution_id,
                    updated_at = excluded.updated_at
                """,
                (
                    execution.id,
                    artifact_key,
                    execution.linked_template_id,
                    artifact_type,
                    serialized,
                    inherited_from_execution_id,
                    now_ms,
                    now_ms,
                ),
            )

        self._execute_write(_write)
        return payload

    def list_execution_pfm_artifacts(self, execution_id: str) -> List[Dict[str, Any]]:
        rows = self._conn.execute(
            """
            SELECT data FROM execution_pfm_artifacts
            WHERE execution_id = ?
            ORDER BY updated_at DESC
            """,
            (execution_id,),
        ).fetchall()
        return self._decode_artifact_rows(rows)

    def get_execution_pfm_artifact(
        self, execution_id: str, artifact_key: str
    ) -> Optional[Dict[str, Any]]:
        row = self._conn.execute(
            """
            SELECT data FROM execution_pfm_artifacts
            WHERE execution_id = ? AND artifact_key = ?
            """,
            (execution_id, artifact_key),
        ).fetchone()
        if not row:
            return None
        try:
            payload = json.loads(row["data"])
        except Exception:
            return None
        return payload if isinstance(payload, dict) else None

    def upsert_template_pfm_artifact(
        self,
        template_id: str,
        artifact_key: str,
        artifact_type: str,
        data: Dict[str, Any],
        source_execution_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        payload = dict(data)
        payload["template_id"] = template_id
        payload["artifact_key"] = artifact_key
        payload["artifact_type"] = artifact_type
        if source_execution_id:
            payload["source_execution_id"] = source_execution_id

        now_ms = int(time.time() * 1000)
        serialized = json.dumps(payload, ensure_ascii=False)

        def _write(conn):
            conn.execute(
                """
                INSERT INTO template_pfm_artifacts
                    (template_id, artifact_key, artifact_type, data,
                     source_execution_id, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(template_id, artifact_key) DO UPDATE SET
                    artifact_type = excluded.artifact_type,
                    data = excluded.data,
                    source_execution_id = excluded.source_execution_id,
                    updated_at = excluded.updated_at
                """,
                (
                    template_id,
                    artifact_key,
                    artifact_type,
                    serialized,
                    source_execution_id,
                    now_ms,
                    now_ms,
                ),
            )

        self._execute_write(_write)
        return payload

    def list_template_pfm_artifacts(self, template_id: str) -> List[Dict[str, Any]]:
        rows = self._conn.execute(
            """
            SELECT data FROM template_pfm_artifacts
            WHERE template_id = ?
            ORDER BY updated_at DESC
            """,
            (template_id,),
        ).fetchall()
        return self._decode_artifact_rows(rows)

    def resolve_pfm_inheritance_source(
        self,
        template_id: str,
        *,
        exclude_execution_id: Optional[str] = None,
        explicit_source_id: Optional[str] = None,
    ) -> Optional[ProjectExecute]:
        """
        Pick a prior execution to copy PFM artifacts from.

        Prefer the latest completed run with PFM skill files (same source as post-login
        skill inject). Fall back to latest completed eligible run, then any eligible run.
        """
        if explicit_source_id:
            sid = str(explicit_source_id).strip()
            if not sid:
                return None
            ex = self.get_execution(sid)
            if (
                not ex
                or ex.linked_template_id != template_id
                or (exclude_execution_id and ex.id == exclude_execution_id)
            ):
                return None
            if not _eligible_for_pfm_run_inheritance(ex):
                return None
            return ex

        from .pfm_skills import resolve_template_learning_source_execution

        learned = resolve_template_learning_source_execution(
            self,
            template_id,
            exclude_execution_id=exclude_execution_id,
        )
        if learned:
            return learned

        for ex in self.list_executions(template_id=template_id):
            if exclude_execution_id and ex.id == exclude_execution_id:
                continue
            if ex.status != ExecutionStatus.COMPLETED:
                continue
            if _eligible_for_pfm_run_inheritance(ex):
                return ex

        for ex in self.list_executions(template_id=template_id):
            if exclude_execution_id and ex.id == exclude_execution_id:
                continue
            if _eligible_for_pfm_run_inheritance(ex):
                return ex
        return None

    def seed_execution_from_prior_run(
        self, new_execution_id: str, source_execution_id: str
    ) -> Optional[ProjectExecute]:
        """
        Copy PFM artifacts from a prior execution into a new run and set
        inherited_from_execution_id. Mirrors seed_execution_from_template_artifacts
        but uses execution-scoped artifact rows (latest run state, any status).
        """
        execution = self.get_execution(new_execution_id)
        source = self.get_execution(source_execution_id)
        if not execution or not source:
            return None
        if source.linked_template_id != execution.linked_template_id:
            logger.warning(
                "[projects] Refusing cross-template inheritance: %s -> %s",
                source_execution_id,
                new_execution_id,
            )
            return None
        if not _eligible_for_pfm_run_inheritance(source):
            return None
        if source_execution_id == new_execution_id:
            return execution

        artifacts = self.list_execution_pfm_artifacts(source_execution_id)
        inherited_reports: List[ProjectReportArtifact] = []
        inherited_results: Optional[List[EadFmNodeRun]] = None
        inherited_report_count = 0

        from .pfm_tree import NON_INHERITABLE_PFM_ARTIFACT_TYPES

        for artifact in artifacts:
            artifact_key = str(artifact.get("artifact_key") or "").strip()
            artifact_type = str(artifact.get("artifact_type") or "").strip()
            if not artifact_key:
                continue
            if artifact_type in NON_INHERITABLE_PFM_ARTIFACT_TYPES:
                continue

            execution_payload = dict(artifact)
            execution_payload["execution_id"] = execution.id
            execution_payload["template_id"] = execution.linked_template_id
            execution_payload["inherited"] = True
            execution_payload["inherited_from_execution_id"] = source_execution_id

            self.upsert_execution_pfm_artifact(
                execution.id,
                artifact_key,
                artifact_type,
                execution_payload,
                inherited_from_execution_id=source_execution_id,
            )

            if artifact_type == "pfm_mindmap":
                nodes_payload = artifact.get("nodes") or []
                if isinstance(nodes_payload, list) and nodes_payload:
                    parsed_nodes: List[EadFmNodeRun] = []
                    for node in nodes_payload:
                        if not isinstance(node, dict):
                            continue
                        try:
                            parsed_nodes.append(EadFmNodeRun.model_validate(node))
                        except Exception as exc:
                            logger.warning(
                                "[projects] Skipping invalid PFM node in artifact %s: %s",
                                artifact_key,
                                exc,
                            )
                    if parsed_nodes:
                        inherited_results = parsed_nodes

            if artifact_type in ("pfm_mindmap", "pfm_report"):
                filename = str(artifact.get("filename") or artifact_key).strip()
                if filename:
                    inherited_reports.append(
                        ProjectReportArtifact(
                            title=str(artifact.get("title") or filename),
                            filename=filename,
                            format=str(artifact.get("format") or "md"),
                            created_at=int(time.time() * 1000),
                            url=f"/v1/projects/executions/{execution.id}/reports/{filename}",
                        )
                    )

            if artifact_type == "node_ead_report":
                inherited_report_count += 1

        update_fields: Dict[str, Any] = {"inherited_from_execution_id": source_execution_id}
        if inherited_results:
            update_fields["results"] = inherited_results
        if inherited_reports:
            update_fields["reports"] = inherited_reports
        if inherited_results or inherited_reports or inherited_report_count:
            mindmap_seeded = "mindmap seeded" if inherited_results else "no mindmap"
            progress_log = list(execution.progress_log or [])
            progress_log.append(
                ProgressLogEntry(
                    ts=time.time(),
                    kind="system",
                    text=(
                        f"PFM inherited from run {source_execution_id} "
                        f"({inherited_report_count} node reports, {mindmap_seeded})."
                    ),
                )
            )
            update_fields["progress_log"] = progress_log

        return self.update_execution(execution.id, **update_fields)

    def seed_execution_from_template_artifacts(self, execution_id: str) -> Optional[ProjectExecute]:
        execution = self.get_execution(execution_id)
        if not execution:
            return None

        template_artifacts = []
        for artifact in self.list_template_pfm_artifacts(execution.linked_template_id):
            source_execution_id = str(artifact.get("source_execution_id") or "").strip()
            if source_execution_id:
                source_execution = self.get_execution(source_execution_id)
                if (
                    not source_execution
                    or source_execution.linked_template_id != execution.linked_template_id
                    or source_execution.valid_for_data_reporting_training is False
                    or source_execution.contributes_to_learning is False
                ):
                    continue
            template_artifacts.append(artifact)
        if not template_artifacts:
            return execution

        inherited_reports = []
        inherited_results = None
        inherited_report_count = 0

        from .pfm_tree import NON_INHERITABLE_PFM_ARTIFACT_TYPES

        for artifact in template_artifacts:
            artifact_key = str(artifact.get("artifact_key") or "").strip()
            artifact_type = str(artifact.get("artifact_type") or "").strip()
            if not artifact_key:
                continue
            if artifact_type in NON_INHERITABLE_PFM_ARTIFACT_TYPES:
                continue

            source_execution_id = str(artifact.get("source_execution_id") or "").strip() or None
            execution_payload = dict(artifact)
            execution_payload["execution_id"] = execution.id
            execution_payload["template_id"] = execution.linked_template_id
            execution_payload["inherited"] = True
            if source_execution_id:
                execution_payload["inherited_from_execution_id"] = source_execution_id

            self.upsert_execution_pfm_artifact(
                execution.id,
                artifact_key,
                artifact_type,
                execution_payload,
                inherited_from_execution_id=source_execution_id,
            )

            if artifact_type == "pfm_mindmap":
                nodes_payload = artifact.get("nodes") or []
                if isinstance(nodes_payload, list) and nodes_payload:
                    parsed_nodes: List[EadFmNodeRun] = []
                    for node in nodes_payload:
                        if not isinstance(node, dict):
                            continue
                        try:
                            parsed_nodes.append(EadFmNodeRun.model_validate(node))
                        except Exception as exc:
                            logger.warning(
                                "[projects] Skipping invalid PFM node in artifact %s: %s",
                                artifact_key,
                                exc,
                            )
                    if parsed_nodes:
                        inherited_results = parsed_nodes

            if artifact_type in ("pfm_mindmap", "pfm_report"):
                filename = str(artifact.get("filename") or artifact_key).strip()
                if filename:
                    inherited_reports.append(
                        ProjectReportArtifact(
                            title=str(artifact.get("title") or filename),
                            filename=filename,
                            format=str(artifact.get("format") or "md"),
                            created_at=int(time.time() * 1000),
                            url=f"/v1/projects/executions/{execution.id}/reports/{filename}",
                        )
                    )

            if artifact_type == "node_ead_report":
                inherited_report_count += 1

        update_fields: Dict[str, Any] = {}
        if inherited_results:
            update_fields["results"] = inherited_results
        if inherited_reports:
            update_fields["reports"] = inherited_reports
        if inherited_results or inherited_reports or inherited_report_count:
            mindmap_seeded = "mindmap seeded" if inherited_results else "no mindmap"
            progress_log = list(execution.progress_log or [])
            progress_log.append(
                ProgressLogEntry(
                    ts=time.time(),
                    kind="system",
                    text=(
                        f"PFM template context loaded ({inherited_report_count} node reports, "
                        f"{mindmap_seeded})."
                    ),
                )
            )
            update_fields["progress_log"] = progress_log

        if not update_fields:
            return execution
        return self.update_execution(execution.id, **update_fields)

    def publish_execution_artifacts_to_template(self, execution_id: str) -> None:
        execution = self.get_execution(execution_id)
        if (
            not execution
            or execution.valid_for_data_reporting_training is False
            or execution.contributes_to_learning is False
        ):
            return

        for artifact in self.list_execution_pfm_artifacts(execution_id):
            artifact_key = str(artifact.get("artifact_key") or "").strip()
            artifact_type = str(artifact.get("artifact_type") or "").strip()
            if not artifact_key or not artifact_type:
                continue
            if artifact.get("inherited") and artifact.get("source_execution_id"):
                # Do not let a purely inherited row overwrite the template source.
                continue
            self.upsert_template_pfm_artifact(
                execution.linked_template_id,
                artifact_key,
                artifact_type,
                artifact,
                source_execution_id=execution.id,
            )

    def sync_execution_pfm_artifacts_from_state(self, execution_id: str) -> None:
        execution = self.get_execution(execution_id)
        if not execution:
            return

        try:
            from .pfm_artifacts import build_pfm_artifact_payloads

            for payload in build_pfm_artifact_payloads(execution):
                artifact_key = str(payload.get("artifact_key") or payload.get("filename") or "").strip()
                artifact_type = str(payload.get("artifact_type") or "").strip()
                if not artifact_key or not artifact_type:
                    continue
                self.upsert_execution_pfm_artifact(
                    execution.id,
                    artifact_key,
                    artifact_type,
                    payload,
                )
        except Exception as exc:
            logger.warning(
                "[projects] Failed to sync PFM mindmap/report artifacts for %s: %s",
                execution_id,
                exc,
            )

        for payload in self._extract_node_ead_report_artifacts(execution):
            artifact_key = str(payload.get("artifact_key") or "").strip()
            if not artifact_key:
                continue
            self.upsert_execution_pfm_artifact(
                execution.id,
                artifact_key,
                "node_ead_report",
                payload,
            )

        if execution.status == ExecutionStatus.COMPLETED:
            self.publish_execution_artifacts_to_template(execution.id)

    # ------------------------------------------------------------------
    # Agent-authored PFM tree (commit_pfm_snapshot pipeline)
    # ------------------------------------------------------------------

    def replace_execution_pfm_tree(
        self,
        execution_id: str,
        *,
        snapshot: Dict[str, Any],
        node_reports: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """
        Atomically replace the canonical agent-authored PFM tree for this execution.

        - Writes pfm-tree.json (artifact_type "pfm_tree").
        - Upserts node_ead_report artifacts (one per node_key) so the UI / Phase 1
          baseline always read pre-saved Markdown.
        - Replaces `executions.results` with the flattened tree so all legacy
          consumers (reports, deliverables, exporters) see the same data.
        - Rebuilds pfm-mindmap.mmd / pfm-report.md / per-execution side artifacts.
        """
        from .pfm_artifacts import build_and_persist_pfm_artifacts
        from .pfm_tree import (
            PFM_TREE_ARTIFACT_KEY,
            PFM_TREE_ARTIFACT_TYPE,
            flat_nodes_to_ead_runs,
        )

        execution = self.get_execution(execution_id)
        if not execution:
            return {"committed": False, "error": "execution_not_found"}

        flat_nodes = list(snapshot.get("flat_nodes") or [])
        flat_runs = flat_nodes_to_ead_runs(flat_nodes)
        from .pfm_tree import (
            snapshot_finalized,
            snapshot_generation,
            snapshot_has_committed_tree,
            snapshot_revision,
        )

        prev_snap = self.get_committed_pfm_tree(execution.id)
        if snapshot_finalized(prev_snap):
            return {
                "committed": False,
                "error": "pfm_finalized",
                "code": "pfm_finalized",
                "message": "PFM is finalized; no further revisions are allowed on this run.",
            }

        from .pfm_delivery import (
            apply_delivery_baseline_to_snapshot,
            compute_delivery_stamp,
            restore_delivery_file_mtimes,
        )

        # Record agent on-disk delivery (filename + mtime) *before* we rewrite report/mindmap files.
        pre_build_stamp = compute_delivery_stamp(execution.id)

        snapshot_payload = self._enrich_snapshot_versioning(execution, dict(snapshot), prev_snap)
        if not str(snapshot_payload.get("source_run_id") or "").strip():
            snapshot_payload["source_run_id"] = execution.id
        snapshot_payload = apply_delivery_baseline_to_snapshot(
            snapshot_payload,
            execution.id,
            stamp=pre_build_stamp,
        )
        generation = snapshot_generation(snapshot_payload)
        revision = snapshot_revision(snapshot_payload)
        version = revision
        generated_at = int(snapshot_payload.get("generated_at") or int(time.time() * 1000))

        # Persist node reports before the tree artifact so the committed snapshot never
        # appears without its EAD node reports on disk/DB.
        committed_report_keys: List[str] = []
        for report in node_reports or []:
            if not isinstance(report, dict):
                continue
            node_key = str(report.get("node_key") or "").strip()
            markdown = str(report.get("markdown") or report.get("content") or "").strip()
            if not node_key or not markdown:
                continue
            self.save_node_ead_report_artifact(
                execution.id,
                node_key=node_key,
                title=str(report.get("title") or node_key),
                content=markdown,
            )
            committed_report_keys.append(node_key)

        if committed_report_keys:
            snapshot_payload["node_reports_committed_at"] = generated_at
            snapshot_payload["node_reports_committed_keys"] = list(committed_report_keys)

        artifact_payload = {
            "artifact_key": PFM_TREE_ARTIFACT_KEY,
            "artifact_type": PFM_TREE_ARTIFACT_TYPE,
            "title": "PFM Tree Snapshot",
            "filename": PFM_TREE_ARTIFACT_KEY,
            "format": "json",
            "snapshot": snapshot_payload,
            "version": version,
            "generation": generation,
            "revision": revision,
            "finalized": bool(snapshot_payload.get("finalized")),
            "generated_at": generated_at,
            "created_at": generated_at,
            "node_count": len(flat_nodes),
            "node_report_count": len(committed_report_keys),
        }
        self.upsert_execution_pfm_artifact(
            execution.id,
            PFM_TREE_ARTIFACT_KEY,
            PFM_TREE_ARTIFACT_TYPE,
            artifact_payload,
        )

        updated = self.update_execution(execution.id, results=flat_runs)
        execution = updated or execution

        report_artifacts = []
        try:
            report_artifacts = build_and_persist_pfm_artifacts(execution)
            self.update_execution(execution.id, reports=report_artifacts)
        except Exception as exc:
            logger.warning(
                "[projects] build_and_persist_pfm_artifacts failed after commit_pfm_snapshot for %s: %s",
                execution.id,
                exc,
            )

        try:
            self.sync_execution_pfm_artifacts_from_state(execution.id)
        except Exception as exc:
            logger.warning(
                "[projects] sync_execution_pfm_artifacts_from_state failed after commit for %s: %s",
                execution.id,
                exc,
            )

        # Our export rewrites pfm-report.md / mindmap; restore mtimes so Refresh is not fooled.
        restore_delivery_file_mtimes(execution.id, pre_build_stamp)

        # No agent files at commit time: record post-export delivery as baseline (once).
        if not (pre_build_stamp.get("files") or []):
            post_stamp = compute_delivery_stamp(execution.id)
            if post_stamp.get("files"):
                snapshot_payload = apply_delivery_baseline_to_snapshot(
                    snapshot_payload,
                    execution.id,
                    stamp=post_stamp,
                )
                restore_delivery_file_mtimes(execution.id, post_stamp)
                artifact_payload["snapshot"] = snapshot_payload
                self.upsert_execution_pfm_artifact(
                    execution.id,
                    PFM_TREE_ARTIFACT_KEY,
                    PFM_TREE_ARTIFACT_TYPE,
                    artifact_payload,
                )

        return {
            "committed": True,
            "execution_id": execution.id,
            "pfm_tree_version": version,
            "pfm_generation": generation,
            "pfm_revision": revision,
            "generated_at": generated_at,
            "node_count": len(flat_nodes),
            "report_count": len(node_reports or []),
            "reports": [r.model_dump() if hasattr(r, "model_dump") else dict(r) for r in report_artifacts],
        }

    def repair_pfm_snapshot_delivery_baseline(self, execution_id: str) -> bool:
        """
        Backfill per-file delivery baseline on snapshots that only have a legacy hash.
        Returns True when the artifact was updated.
        """
        from .pfm_delivery import apply_delivery_baseline_to_snapshot
        from .pfm_tree import PFM_TREE_ARTIFACT_KEY, PFM_TREE_ARTIFACT_TYPE, snapshot_has_committed_tree

        raw = self._get_pfm_tree_snapshot_raw(execution_id)
        if not snapshot_has_committed_tree(raw):
            return False
        from .pfm_delivery import snapshot_delivery_baseline_for_api

        from .pfm_delivery import filter_canonical_delivery_files

        raw_dict = dict(raw)
        has_canonical = bool(
            filter_canonical_delivery_files(snapshot_delivery_baseline_for_api(raw_dict).get("baseline_files"))
        )
        has_explicit = bool(
            str(raw_dict.get("pfm_fmr_based_on_file") or "").strip()
            and str(raw_dict.get("pfm_pfm_based_on_file") or "").strip()
        )
        if has_canonical and has_explicit:
            return False
        patched = apply_delivery_baseline_to_snapshot(raw_dict, execution_id)
        if not filter_canonical_delivery_files(snapshot_delivery_baseline_for_api(patched).get("baseline_files")):
            return False
        artifact = self.get_execution_pfm_artifact(execution_id, PFM_TREE_ARTIFACT_KEY)
        if not isinstance(artifact, dict):
            return False
        payload = dict(artifact)
        payload["snapshot"] = patched
        self.upsert_execution_pfm_artifact(
            execution_id,
            PFM_TREE_ARTIFACT_KEY,
            PFM_TREE_ARTIFACT_TYPE,
            payload,
        )
        return True

    def _get_pfm_tree_snapshot_raw(self, execution_id: str) -> Optional[Dict[str, Any]]:
        """Read committed tree artifact without lineage backfill (avoids recursion)."""
        from .pfm_tree import PFM_TREE_ARTIFACT_KEY

        artifact = self.get_execution_pfm_artifact(execution_id, PFM_TREE_ARTIFACT_KEY)
        if not isinstance(artifact, dict):
            return None
        snap = artifact.get("snapshot")
        return snap if isinstance(snap, dict) else None

    def get_committed_pfm_tree(self, execution_id: str) -> Optional[Dict[str, Any]]:
        """Return the latest committed PFM tree snapshot, with generation/revision backfill when missing."""
        from .pfm_run_number import get_run_number
        from .pfm_tree import snapshot_generation, snapshot_revision

        snap = self._get_pfm_tree_snapshot_raw(execution_id)
        if not isinstance(snap, dict):
            return None
        ex = self.get_execution(execution_id)
        if not ex:
            return snap

        out = dict(snap)
        gen = get_run_number(self, ex)
        rev = snapshot_revision(snap) or 1
        out["generation"] = gen
        out["revision"] = rev
        out["version"] = rev
        return out

    def get_pfm_generation_for_run(self, ex: ProjectExecute) -> int:
        """PFM V = run number on this template."""
        from .pfm_run_number import get_run_number

        return get_run_number(self, ex)

    def has_committed_pfm_tree(self, execution_id: str) -> bool:
        from .pfm_tree import snapshot_has_committed_tree

        return snapshot_has_committed_tree(self._get_pfm_tree_snapshot_raw(execution_id))

    def compute_next_pfm_versioning(
        self, ex: ProjectExecute, prev_snap: Optional[Dict[str, Any]]
    ) -> tuple[int, int]:
        """
        V = run number; Rev increments within the same run.
        """
        from .pfm_tree import snapshot_revision

        gen = self.get_pfm_generation_for_run(ex)
        prev_rev = snapshot_revision(prev_snap)
        if prev_rev > 0:
            return gen, prev_rev + 1
        return gen, 1

    def _enrich_snapshot_versioning(
        self,
        ex: ProjectExecute,
        snapshot: Dict[str, Any],
        prev_snap: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        from .pfm_tree import snapshot_generation, snapshot_revision

        gen = snapshot_generation(snapshot)
        rev = snapshot_revision(snapshot)
        try:
            ver_hint = int(snapshot.get("version") or 0)
        except Exception:
            ver_hint = 0
        if ver_hint > rev:
            rev = ver_hint
        run_gen = self.get_pfm_generation_for_run(ex)
        if gen > 0 and rev > 0:
            snapshot["generation"] = run_gen
            snapshot["revision"] = rev
            snapshot["version"] = rev
            return snapshot
        next_gen, next_rev = self.compute_next_pfm_versioning(ex, prev_snap)
        snapshot["generation"] = next_gen
        snapshot["revision"] = next_rev
        snapshot["version"] = next_rev
        if "finalized" not in snapshot:
            snapshot["finalized"] = False
        return snapshot

    def resolve_pfm_baseline_execution_id(self, ex: ProjectExecute) -> Optional[str]:
        """Execution id to read committed PFM from when ``ex`` has no snapshot yet."""
        if self.has_committed_pfm_tree(ex.id):
            return None
        from .pfm_run_number import prior_run_execution_id

        chron_prior = prior_run_execution_id(self, ex)
        if chron_prior and chron_prior != ex.id and self.has_committed_pfm_tree(chron_prior):
            return chron_prior
        tmpl = self.get_template(ex.linked_template_id)
        canon = (tmpl.canonical_pfm_execution_id or "").strip() if tmpl else ""
        if canon and canon != ex.id and self.has_committed_pfm_tree(canon):
            return canon
        inh = (ex.inherited_from_execution_id or "").strip()
        if inh and inh != ex.id and self.has_committed_pfm_tree(inh):
            return inh
        return None

    def _resolve_lineage_baseline_tree_version(self, ex: ProjectExecute) -> tuple[str, int]:
        """Find prior run with a committed tree for template lineage display (v9 → v10)."""
        seen: Set[str] = set()
        candidates: List[str] = []

        def _add(candidate: Optional[str]) -> None:
            cid = str(candidate or "").strip()
            if not cid or cid == ex.id or cid in seen:
                return
            seen.add(cid)
            candidates.append(cid)

        _add(self.resolve_pfm_baseline_execution_id(ex))
        tmpl = self.get_template(ex.linked_template_id)
        if tmpl:
            _add(tmpl.canonical_pfm_execution_id)
        walk = ex
        walk_seen: Set[str] = set()
        for _ in range(8):
            parent_id = str(getattr(walk, "inherited_from_execution_id", "") or "").strip()
            if not parent_id or parent_id == ex.id:
                break
            _add(parent_id)
            if parent_id in walk_seen:
                break
            walk_seen.add(parent_id)
            parent = self.get_execution(parent_id)
            if not parent:
                break
            walk = parent

        from .pfm_tree import baseline_generation_from_snap, snapshot_has_committed_tree

        best_id, best_gen = "", 0
        for cid in candidates:
            snap = self._get_pfm_tree_snapshot_raw(cid)
            if not snapshot_has_committed_tree(snap):
                continue
            gen = baseline_generation_from_snap(snap)
            if gen > best_gen:
                best_id, best_gen = cid, gen
        return best_id, best_gen

    def resolve_pfm_lineage_context(self, ex: ProjectExecute) -> Dict[str, Any]:
        """
        Operator-facing PFM labels: V = run_number, Rev = within-run revision.
        """
        from .models import ExecutionStatus
        from .pfm_run_number import get_run_number, prior_run_execution_id, prior_run_number
        from .pfm_tree import snapshot_finalized, snapshot_revision

        has_own = self.has_committed_pfm_tree(ex.id)
        own_snap = self.get_committed_pfm_tree(ex.id) if has_own else None
        finalized = snapshot_finalized(own_snap)

        generation = get_run_number(self, ex)
        baseline_ver = prior_run_number(self, ex)
        baseline_id = prior_run_execution_id(self, ex) or ""
        if not baseline_id:
            baseline_id = self.resolve_pfm_baseline_execution_id(ex) or ""

        tree_read_id = ex.id if has_own else (self.resolve_pfm_baseline_execution_id(ex) or baseline_id or "")

        revision = snapshot_revision(own_snap) if has_own else 0

        status = (
            ex.status.value
            if isinstance(ex.status, ExecutionStatus)
            else str(ex.status or "")
        ).lower()
        hint = str(ex.executor_hint or "").lower()
        finalizing_hint = any(
            token in hint
            for token in ("reporting", "finalizing", "deliverable", "ai finish")
        )

        if not has_own:
            phase = "baseline_preview"
        elif status in ("completed", "failed", "cancelled", "error"):
            phase = "final"
        elif status in ("running", "pending") and finalizing_hint:
            phase = "finalizing"
        elif status in ("running", "pending"):
            phase = "evolving"
        else:
            phase = "final"

        return {
            "run_number": generation,
            "pfm_baseline_execution_id": baseline_id or None,
            "pfm_tree_read_execution_id": tree_read_id or None,
            "pfm_baseline_tree_version": baseline_ver,
            "pfm_generation_version": generation,
            "pfm_revision": revision,
            "pfm_lineage_version": generation,
            "pfm_step_version": revision,
            "pfm_finalized": finalized,
            "pfm_has_committed_snapshot": has_own,
            "pfm_snapshot_phase": phase,
        }

    def promote_template_canonical_pfm(
        self,
        template_id: str,
        execution_id: str,
        *,
        source: str,
        rationale: Optional[str] = None,
        require_eligible: bool = True,
    ) -> Optional[ProjectTemplate]:
        """
        Set the template's canonical PFM pointer to ``execution_id``.

        ``require_eligible``: when True (AI path), refuse if the run is marked
        invalid for reporting/training or opted out of learning.
        """
        template = self.get_template(template_id)
        if not template:
            return None
        execution = self.get_execution(execution_id)
        if not execution or execution.linked_template_id != template_id:
            return None
        if not self.has_committed_pfm_tree(execution_id):
            return None
        if require_eligible:
            if execution.valid_for_data_reporting_training is False:
                return None
            if execution.contributes_to_learning is False:
                return None
        now_ms = int(time.time() * 1000)
        rationale_text = (rationale or "").strip()[:8000] or None
        updated = self.update_template(
            template_id,
            canonical_pfm_execution_id=execution_id,
            canonical_pfm_promoted_at=now_ms,
            canonical_pfm_promoted_by=(source or "operator")[:32],
            canonical_pfm_promotion_rationale=rationale_text,
        )
        if updated:
            self.update_execution(
                execution_id,
                pfm_canonical_promotion_applied=True,
            )
        return updated

    def get_pfm_view_state(self, execution_id: str) -> Optional[Dict[str, Any]]:
        from .pfm_tree import PFM_VIEW_STATE_ARTIFACT_KEY

        artifact = self.get_execution_pfm_artifact(execution_id, PFM_VIEW_STATE_ARTIFACT_KEY)
        if not isinstance(artifact, dict):
            return None
        state = artifact.get("state")
        return state if isinstance(state, dict) else None

    def set_pfm_view_state(
        self, execution_id: str, state: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        from .pfm_tree import (
            PFM_VIEW_STATE_ARTIFACT_KEY,
            PFM_VIEW_STATE_ARTIFACT_TYPE,
        )

        execution = self.get_execution(execution_id)
        if not execution:
            return None
        payload = {
            "artifact_key": PFM_VIEW_STATE_ARTIFACT_KEY,
            "artifact_type": PFM_VIEW_STATE_ARTIFACT_TYPE,
            "title": "PFM View State",
            "filename": PFM_VIEW_STATE_ARTIFACT_KEY,
            "format": "json",
            "state": state,
            "updated_at": int(time.time() * 1000),
        }
        self.upsert_execution_pfm_artifact(
            execution.id,
            PFM_VIEW_STATE_ARTIFACT_KEY,
            PFM_VIEW_STATE_ARTIFACT_TYPE,
            payload,
        )
        return payload

    def save_node_ead_report_artifact(
        self,
        execution_id: str,
        node_key: str,
        title: str,
        content: str,
    ) -> Optional[Dict[str, Any]]:
        from .pfm_node_report_content import normalize_node_report_markdown

        execution = self.get_execution(execution_id)
        if not execution:
            return None

        content = normalize_node_report_markdown(content)

        safe_node_key = re.sub(
            r"[^a-z0-9]+",
            "-",
            str(node_key or title or "node").lower(),
        ).strip("-")
        artifact_key = f"node-ead-report-{safe_node_key or 'node'}.md"
        payload = {
            "artifact_key": artifact_key,
            "artifact_type": "node_ead_report",
            "title": title or node_key or "PFM node EAD report",
            "node_key": node_key,
            "filename": artifact_key,
            "format": "md",
            "content": content,
            "excerpt": str(content or "")[:1600],
            "created_at": int(time.time() * 1000),
        }

        saved = self.upsert_execution_pfm_artifact(
            execution.id,
            artifact_key,
            "node_ead_report",
            payload,
        )
        if (
            execution.valid_for_data_reporting_training is not False
            and execution.contributes_to_learning is not False
        ):
            self.upsert_template_pfm_artifact(
                execution.linked_template_id,
                artifact_key,
                "node_ead_report",
                payload,
                source_execution_id=execution.id,
            )
        return saved

    def _extract_node_ead_report_artifacts(
        self, execution: ProjectExecute
    ) -> List[Dict[str, Any]]:
        artifacts = []
        seen_keys: set[str] = set()
        for entry in execution.progress_log or []:
            text = (entry.text or "").strip()
            if not self._looks_like_node_report(text):
                continue

            req_match = re.search(r"\[Node-Report-Reply-To:\s*([^\]]+)\]", text)
            req_id = req_match.group(1).strip() if req_match else ""
            title = self._extract_node_report_title(text)
            slug_source = req_id or title or f"node-report-{len(artifacts) + 1}"
            slug = re.sub(r"[^a-z0-9]+", "-", slug_source.lower()).strip("-")
            artifact_key = f"node-ead-report-{slug or len(artifacts) + 1}.md"
            if artifact_key in seen_keys:
                continue
            seen_keys.add(artifact_key)

            artifacts.append(
                {
                    "artifact_key": artifact_key,
                    "artifact_type": "node_ead_report",
                    "title": title,
                    "filename": artifact_key,
                    "format": "md",
                    "content": text,
                    "excerpt": text[:1600],
                    "created_at": int(time.time() * 1000),
                }
            )
        return artifacts

    def _decode_artifact_rows(self, rows) -> List[Dict[str, Any]]:
        artifacts = []
        for row in rows:
            try:
                payload = json.loads(row["data"])
            except Exception:
                continue
            if isinstance(payload, dict):
                artifacts.append(payload)
        return artifacts

    # ------------------------------------------------------------------
    # Template learning summaries
    # ------------------------------------------------------------------

    def upsert_template_learning_summary(self, execution_id: str) -> Optional[Dict[str, Any]]:
        execution = self.get_execution(execution_id)
        if not execution:
            return None
        if execution.status != ExecutionStatus.COMPLETED:
            return None
        if (
            execution.valid_for_data_reporting_training is False
            or execution.contributes_to_learning is False
        ):
            return None

        summary = self._build_template_learning_summary(execution)
        if not summary:
            return None

        data = json.dumps(summary, ensure_ascii=False)
        now_ms = int(time.time() * 1000)

        def _write(conn):
            conn.execute(
                """
                INSERT INTO template_learning_summaries
                    (template_id, execution_id, data, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(template_id, execution_id) DO UPDATE SET
                    data = excluded.data,
                    updated_at = excluded.updated_at
                """,
                (
                    execution.linked_template_id,
                    execution.id,
                    data,
                    now_ms,
                    now_ms,
                ),
            )

        self._execute_write(_write)
        return summary

    def list_template_learning_summaries(
        self, template_id: str, limit: int = 5
    ) -> List[Dict[str, Any]]:
        rows = self._conn.execute(
            """
            SELECT data FROM template_learning_summaries
            WHERE template_id = ?
            ORDER BY updated_at DESC
            LIMIT ?
            """,
            (template_id, max(1, limit)),
        ).fetchall()
        summaries = []
        for row in rows:
            try:
                payload = json.loads(row["data"])
            except Exception:
                continue
            if isinstance(payload, dict):
                execution_id = str(payload.get("execution_id") or "").strip()
                if execution_id:
                    execution = self.get_execution(execution_id)
                    if (
                        not execution
                        or execution.linked_template_id != template_id
                        or execution.valid_for_data_reporting_training is False
                        or execution.contributes_to_learning is False
                    ):
                        continue
                summaries.append(payload)
        return summaries

    def build_template_learning_context(self, template_id: str, limit: int = 3) -> str:
        summaries = self.list_template_learning_summaries(template_id, limit=limit)
        template_artifacts = []
        for artifact in self.list_template_pfm_artifacts(template_id):
            source_execution_id = str(artifact.get("source_execution_id") or "").strip()
            if source_execution_id:
                execution = self.get_execution(source_execution_id)
                if (
                    not execution
                    or execution.linked_template_id != template_id
                    or execution.valid_for_data_reporting_training is False
                    or execution.contributes_to_learning is False
                ):
                    continue
            template_artifacts.append(artifact)
        if not summaries:
            # Legacy runs may predate the summaries table. Use them as read-only
            # inheritance context so the next run can still benefit immediately.
            for execution in self.list_executions(template_id=template_id):
                if len(summaries) >= limit:
                    break
                if execution.status != ExecutionStatus.COMPLETED:
                    continue
                if (
                    execution.valid_for_data_reporting_training is False
                    or execution.contributes_to_learning is False
                ):
                    continue
                summary = self._build_template_learning_summary(execution)
                if summary:
                    summaries.append(summary)

        if not summaries and not template_artifacts:
            return ""

        lines = [
            "Template-scoped knowledge inherited from previous runs:",
            "Use this as compact context. Verify it against the live app before relying on it.",
        ]

        mindmap_artifact = next(
            (
                artifact
                for artifact in template_artifacts
                if artifact.get("artifact_type") == "pfm_mindmap"
            ),
            None,
        )
        if mindmap_artifact:
            nodes = mindmap_artifact.get("nodes") or []
            if isinstance(nodes, list) and nodes:
                node_bits = []
                for node in nodes[:20]:
                    if not isinstance(node, dict):
                        continue
                    title = str(node.get("title") or node.get("node_key") or "").strip()
                    status = str(node.get("status") or "").strip()
                    if title:
                        node_bits.append(f"{title}{f' [{status}]' if status else ''}")
                if node_bits:
                    lines.append(f"- Current template PFM mindmap nodes: {'; '.join(node_bits)}")

        template_node_reports = [
            artifact
            for artifact in template_artifacts
            if artifact.get("artifact_type") == "node_ead_report"
        ][:8]
        if template_node_reports:
            lines.append("- Current template per-node EAD reports:")
            for artifact in template_node_reports:
                title = str(artifact.get("title") or artifact.get("node_key") or "PFM node").strip()
                excerpt = str(artifact.get("content") or artifact.get("excerpt") or "").strip()
                lines.append(f"  {title}: {excerpt[:700]}")

        for summary in summaries[:limit]:
            run_name = summary.get("name") or summary.get("execution_id")
            evidence = summary.get("evidence_source") or "unknown"
            purpose = summary.get("run_purpose") or "unknown"
            lines.append(f"- Prior run: {run_name} ({purpose}, {evidence})")
            nodes = summary.get("pfm_nodes") or []
            if nodes:
                node_bits = []
                for node in nodes[:12]:
                    title = str(node.get("title") or node.get("node_key") or "").strip()
                    status = str(node.get("status") or "").strip()
                    if title:
                        node_bits.append(f"{title}{f' [{status}]' if status else ''}")
                if node_bits:
                    lines.append(f"  PFM nodes: {'; '.join(node_bits)}")
            metrics = summary.get("metrics") or {}
            if metrics:
                lines.append(
                    "  Metrics: "
                    f"nodes={metrics.get('node_count', 0)}, "
                    f"test_steps={metrics.get('total_test_steps', 0)}, "
                    f"succeeded={metrics.get('succeeded_test_steps', 0)}, "
                    f"failed={metrics.get('failed_test_steps', 0)}"
                )
            reports = summary.get("reports") or []
            if reports:
                report_bits = [
                    str(report.get("url") or report.get("filename") or "").strip()
                    for report in reports[:3]
                ]
                report_bits = [bit for bit in report_bits if bit]
                if report_bits:
                    lines.append(f"  Reports: {'; '.join(report_bits)}")
            node_reports = summary.get("node_reports") or []
            if node_reports:
                report_titles = [
                    str(report.get("title") or "PFM node report").strip()
                    for report in node_reports[:5]
                ]
                lines.append(f"  Node reports captured: {'; '.join(report_titles)}")
                for report in node_reports[:2]:
                    excerpt = str(report.get("excerpt") or "").strip()
                    if excerpt:
                        lines.append(f"  Node report excerpt: {excerpt[:900]}")

        return "\n".join(lines)

    def _build_template_learning_summary(
        self, execution: ProjectExecute
    ) -> Optional[Dict[str, Any]]:
        nodes = []
        for node in (execution.results or [])[:40]:
            nodes.append(
                {
                    "node_id": node.node_id,
                    "node_key": node.node_key,
                    "title": node.title,
                    "status": node.status,
                    "test_case_count": len(node.test_case_runs or []),
                }
            )

        reports = []
        for report in (execution.reports or [])[:10]:
            reports.append(
                {
                    "title": report.title,
                    "filename": report.filename,
                    "format": report.format,
                    "url": report.url,
                    "created_at": report.created_at,
                }
            )

        progress_highlights = []
        node_reports = []
        for entry in (execution.progress_log or []):
            text = (entry.text or "").strip()
            if self._looks_like_node_report(text):
                node_reports.append(
                    {
                        "title": self._extract_node_report_title(text),
                        "excerpt": text[:1600],
                    }
                )

        for entry in (execution.progress_log or [])[-12:]:
            text = (entry.text or "").strip()
            if text:
                progress_highlights.append({"kind": entry.kind, "text": text[:240]})

        if not nodes and not reports and not progress_highlights and not node_reports:
            return None

        return {
            "template_id": execution.linked_template_id,
            "execution_id": execution.id,
            "name": execution.name,
            "run_purpose": execution.run_purpose,
            "evidence_source": execution.evidence_source,
            "explore_type": execution.explore_type,
            "completed_at": int(time.time() * 1000),
            "pfm_nodes": nodes,
            "reports": reports,
            "node_reports": node_reports[-8:],
            "progress_highlights": progress_highlights,
            "metrics": {
                "node_count": len(nodes),
                "total_test_steps": getattr(execution, "total_test_steps", None) or 0,
                "succeeded_test_steps": getattr(execution, "succeeded_test_steps", None) or 0,
                "failed_test_steps": getattr(execution, "failed_test_steps", None) or 0,
                "skipped_test_steps": getattr(execution, "skipped_test_steps", None) or 0,
                "duration_ms": execution.duration_ms or 0,
            },
        }

    def _looks_like_node_report(self, text: str) -> bool:
        from .pfm_node_report_content import is_valid_node_report_markdown

        normalized = str(text or "")
        if "[Node-Report-Reply-To:" in normalized:
            from .pfm_node_report_content import normalize_node_report_markdown

            return is_valid_node_report_markdown(normalize_node_report_markdown(normalized))
        return is_valid_node_report_markdown(normalized)

    def _extract_node_report_title(self, text: str) -> str:
        for raw_line in str(text or "").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("[Node-Report-Reply-To:"):
                continue
            if line.startswith("Purpose:"):
                return line.replace("Purpose:", "", 1).strip()[:120] or "PFM node report"
            if line.startswith("Feature F-"):
                return line[:120]
        return "PFM node report"

    # ------------------------------------------------------------------
    # Bulk / utility
    # ------------------------------------------------------------------

    def get_active_executions(self) -> List[ProjectExecute]:
        return self.list_executions(status=ExecutionStatus.RUNNING.value) + \
               self.list_executions(status=ExecutionStatus.PENDING.value)
