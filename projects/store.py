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
from typing import Any, Dict, List, Optional

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
        return execution

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
        Ordering: same as list_executions (created_at DESC — most recent first).
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
        inherited_notes: List[str] = []

        for artifact in artifacts:
            artifact_key = str(artifact.get("artifact_key") or "").strip()
            artifact_type = str(artifact.get("artifact_type") or "").strip()
            if not artifact_key:
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
                title = str(artifact.get("title") or artifact_key).strip()
                excerpt = str(artifact.get("content") or artifact.get("excerpt") or "").strip()
                inherited_notes.append(f"{title}: {excerpt[:500]}")

        update_fields: Dict[str, Any] = {"inherited_from_execution_id": source_execution_id}
        if inherited_results:
            update_fields["results"] = inherited_results
        if inherited_reports:
            update_fields["reports"] = inherited_reports
        if inherited_notes:
            progress_log = list(execution.progress_log or [])
            progress_log.append(
                ProgressLogEntry(
                    ts=time.time(),
                    kind="system",
                    text=(
                        "Inherited PFM node reports from prior execution "
                        f"{source_execution_id}: "
                        + " | ".join(inherited_notes[:5])
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
        inherited_notes = []

        for artifact in template_artifacts:
            artifact_key = str(artifact.get("artifact_key") or "").strip()
            artifact_type = str(artifact.get("artifact_type") or "").strip()
            if not artifact_key:
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
                title = str(artifact.get("title") or artifact_key).strip()
                excerpt = str(artifact.get("content") or artifact.get("excerpt") or "").strip()
                inherited_notes.append(f"{title}: {excerpt[:500]}")

        update_fields: Dict[str, Any] = {}
        if inherited_results:
            update_fields["results"] = inherited_results
        if inherited_reports:
            update_fields["reports"] = inherited_reports
        if inherited_notes:
            progress_log = list(execution.progress_log or [])
            progress_log.append(
                ProgressLogEntry(
                    ts=time.time(),
                    kind="system",
                    text=(
                        "Inherited template PFM node EAD reports loaded for this run: "
                        + " | ".join(inherited_notes[:5])
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

    def save_node_ead_report_artifact(
        self,
        execution_id: str,
        node_key: str,
        title: str,
        content: str,
    ) -> Optional[Dict[str, Any]]:
        execution = self.get_execution(execution_id)
        if not execution:
            return None

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
        normalized = str(text or "")
        return (
            "[Node-Report-Reply-To:" in normalized
            or (
                "Node Summary:" in normalized
                and "Features:" in normalized
                and "Test Case TC-" in normalized
            )
        )

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
