"""
EAD project data models.

Port of src/projects/types.ts from EAD-EXP into Pydantic v2 models.
"""

import time
import uuid
from enum import Enum
from typing import Any, List, Literal, Optional

from pydantic import BaseModel, Field, model_validator

RunPurpose = Literal["live_app_learning", "live_app_testing", "document_analysis"]
EvidenceSource = Literal["live_app", "document", "hybrid"]


class ProjectType(str, Enum):
    EAD_AUTO_TEST = "ead-auto-test"


class ProjectAuthMode(str, Enum):
    NONE = "none"
    REUSE_SESSION = "reuse-session"
    MANUAL_BOOTSTRAP = "manual-bootstrap"
    CREDENTIALS = "credentials"


class ExecutionStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    ERROR = "error"


class StepStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class TestCaseStepRunStatus(str, Enum):
    SUCCESS = "Success"
    FAILED = "Failed"
    NO_RUN = "No Run"


class TestStep(BaseModel):
    step_id: str
    sort_order: int
    procedure_text: str
    expected_result: str
    must_pass: bool = True


class TestCase(BaseModel):
    case_id: str
    title: str
    test_steps: List[TestStep] = Field(default_factory=list)


class EadFmNode(BaseModel):
    node_id: str
    node_key: str
    type: str = ""
    title: str = ""
    meta: str = ""
    test_cases: List[TestCase] = Field(default_factory=list)


class TestCaseStepRun(BaseModel):
    step_id: str
    sort_order: int
    procedure_text: str
    expected_result: str
    must_pass: bool = True
    status: TestCaseStepRunStatus = TestCaseStepRunStatus.NO_RUN
    actual_result: Optional[str] = None
    screenshot_url: Optional[str] = None
    execution_time_ms: Optional[int] = None


class TestCaseRun(BaseModel):
    case_id: str
    title: str
    status: TestCaseStepRunStatus = TestCaseStepRunStatus.NO_RUN
    test_case_step_runs: List[TestCaseStepRun] = Field(default_factory=list)


class EadFmNodeRun(BaseModel):
    """
    PFM node row persisted on ``ProjectExecute.results``.

    Agents sometimes emit camelCase keys (``nodeKey``) and free-form status
    strings such as ``verified``; normalize before strict enum validation so
    SQLite rows and gateway JSON stay loadable.
    """

    node_id: str = ""
    node_key: str = ""
    parent_node_key: Optional[str] = None
    level: int = 0
    type: str = ""
    title: str = ""
    meta: str = ""
    status: TestCaseStepRunStatus = TestCaseStepRunStatus.NO_RUN
    test_case_runs: List[TestCaseRun] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def _normalize_agent_payload(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        out = dict(data)
        if "node_key" not in out and "nodeKey" in out:
            out["node_key"] = out.get("nodeKey")
        if "node_id" not in out and "nodeId" in out:
            out["node_id"] = out.get("nodeId")
        if "parent_node_key" not in out and "parentNodeKey" in out:
            out["parent_node_key"] = out.get("parentNodeKey")
        st = out.get("status")
        if isinstance(st, str):
            low = st.strip().lower()
            if low in ("verified", "passed", "complete", "completed", "success", "ok", "true"):
                out["status"] = TestCaseStepRunStatus.SUCCESS.value
            elif low in ("failed", "error", "blocked", "false"):
                out["status"] = TestCaseStepRunStatus.FAILED.value
            elif low in ("no_run", "pending", "unknown", "not_run", "skipped", ""):
                out["status"] = TestCaseStepRunStatus.NO_RUN.value
        nk = str(out.get("node_key") or "").strip()
        nid = str(out.get("node_id") or "").strip()
        if not nid and nk:
            out["node_id"] = nk
        if not str(out.get("node_id") or "").strip():
            out["node_id"] = "pfm-node"
        return out


class StepArtifact(BaseModel):
    type: str = "screenshot"
    path: str = ""
    thumbnail_path: Optional[str] = None
    captured_at: str = ""
    description: Optional[str] = None


class StepResult(BaseModel):
    step_id: str = ""
    title: str = ""
    status: StepStatus = StepStatus.PENDING
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    duration_ms: Optional[int] = None
    artifacts: List[StepArtifact] = Field(default_factory=list)
    summary: Optional[str] = None
    error: Optional[dict] = None


class ProgressLogEntry(BaseModel):
    ts: float = Field(default_factory=time.time)
    kind: str = "system"
    text: str = ""
    thumbnail_url: Optional[str] = None
    image_url: Optional[str] = None
    tool_name: Optional[str] = None
    tool_input: Optional[dict] = None


class ProjectReportArtifact(BaseModel):
    title: str
    filename: str
    format: str
    created_at: int
    url: str


class ProjectTemplate(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    description: str = ""
    explore_type: Optional[str] = None
    target_url: Optional[str] = None
    ai_prompt: str = ""
    auth_mode: Optional[ProjectAuthMode] = None
    auth_login_url: Optional[str] = None
    auth_session_profile: Optional[str] = None
    auth_instructions: Optional[str] = None
    time_budget_minutes: Optional[int] = None
    cost_budget_dollars: Optional[float] = None
    total_test_steps: int = 0
    failed_test_steps: int = 0
    pfm_nodes: List[EadFmNode] = Field(default_factory=list)
    created_at: int = Field(default_factory=lambda: int(time.time() * 1000))
    created_by: str = ""
    last_modified_at: int = Field(default_factory=lambda: int(time.time() * 1000))
    last_modified_by: str = ""
    # Execution whose committed PFM is the template-wide "most accurate" baseline for UI + new runs.
    canonical_pfm_execution_id: Optional[str] = None
    canonical_pfm_promoted_at: Optional[int] = None
    canonical_pfm_promoted_by: Optional[str] = None  # "ai" | "operator"
    canonical_pfm_promotion_rationale: Optional[str] = None


class ProjectExecute(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    linked_template_id: str = ""
    name: str = ""
    description: str = ""
    target_url: Optional[str] = None
    ai_prompt: str = ""
    auth_mode: Optional[ProjectAuthMode] = None
    auth_login_url: Optional[str] = None
    auth_session_profile: Optional[str] = None
    auth_instructions: Optional[str] = None
    time_budget_minutes: Optional[int] = None
    cost_budget_dollars: Optional[float] = None
    show_local_browser: bool = False
    explore_type: Optional[str] = None
    run_session_key: Optional[str] = None
    agent_run_id: Optional[str] = None
    status: ExecutionStatus = ExecutionStatus.PENDING
    steps: List[StepResult] = Field(default_factory=list)
    current_step_id: Optional[str] = None
    paused: bool = False
    progress_percentage: int = 0
    start_time: Optional[int] = None
    duration_ms: Optional[int] = None
    log_tokens: Optional[int] = None
    executor_hint: Optional[str] = None
    last_error_message: Optional[str] = None
    cancel_reason: Optional[str] = None
    operator_stop_kind: Optional[str] = None
    results: List[EadFmNodeRun] = Field(default_factory=list)
    reports: List[ProjectReportArtifact] = Field(default_factory=list)
    progress_log: List[ProgressLogEntry] = Field(default_factory=list)
    progress_log_seq: Optional[int] = None
    first_failed_at: Optional[int] = None
    # Template knowledge subsystem (Phase 1). None = legacy row before these fields.
    run_purpose: Optional[RunPurpose] = None
    contributes_to_learning: Optional[bool] = None
    evidence_source: Optional[EvidenceSource] = None
    learning_exclusion_reason: Optional[str] = None
    # Operator-controlled validity. False means this run must not be used for
    # data reporting, inherited training context, or template current reports.
    valid_for_data_reporting_training: Optional[bool] = None
    invalid_for_data_reporting_training_reason: Optional[str] = None
    # When this run was seeded from another execution's PFM artifacts / chat cache.
    inherited_from_execution_id: Optional[str] = None
    # Deferred bootstrap (POST /executions/run returns before seed/session complete).
    bootstrap_pending: bool = False
    bootstrap_inherit_pfm: bool = True
    bootstrap_explicit_inherit_from_execution_id: Optional[str] = None
    # Canonical PFM promotion (filled after terminal completion + optional AI judge).
    pfm_canonical_evaluation_status: Optional[str] = None
    pfm_canonical_replace_recommended: Optional[bool] = None
    pfm_canonical_evaluation_confidence: Optional[float] = None
    pfm_canonical_evaluation_rationale: Optional[str] = None
    pfm_canonical_evaluation_at_ms: Optional[int] = None
    pfm_canonical_promotion_applied: Optional[bool] = None


class ProjectsStoreFile(BaseModel):
    version: int = 3
    templates: List[ProjectTemplate] = Field(default_factory=list)
    executions: List[ProjectExecute] = Field(default_factory=list)
    active_template_id: Optional[str] = None
