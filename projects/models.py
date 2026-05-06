"""
EAD project data models.

Port of src/projects/types.ts from EAD-EXP into Pydantic v2 models.
"""

import time
import uuid
from enum import Enum
from typing import List, Literal, Optional

from pydantic import BaseModel, Field

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
    node_id: str
    node_key: str = ""
    parent_node_key: Optional[str] = None
    level: int = 0
    type: str = ""
    title: str = ""
    meta: str = ""
    status: TestCaseStepRunStatus = TestCaseStepRunStatus.NO_RUN
    test_case_runs: List[TestCaseRun] = Field(default_factory=list)


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


class ProjectsStoreFile(BaseModel):
    version: int = 3
    templates: List[ProjectTemplate] = Field(default_factory=list)
    executions: List[ProjectExecute] = Field(default_factory=list)
    active_template_id: Optional[str] = None
