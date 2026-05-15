-- EAD projects store (templates, executions, PFM artifacts, active state)
-- Schema version: 1 (see projects/store.py _SCHEMA_VERSION)
-- Default path: ~/.hermes/projects/projects.db
-- WAL + foreign_keys are enabled at runtime by ProjectStore.

PRAGMA foreign_keys = ON;

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
