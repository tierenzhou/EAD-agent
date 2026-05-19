"""
Standard structure and quality bar for per-node EAD reports (agent-authored).

Used in commit_pfm_snapshot guidance and operator node-report regeneration prompts.
"""

from __future__ import annotations

# Plain-text template (no markdown # headings) — matches Explore UI node-report format.
PFM_NODE_REPORT_BODY_TEMPLATE = """
Node Summary:
Purpose: <one short paragraph describing what this PFM node is for>
In-scope behaviors:
- <behavior 1>
- <behavior 2>

Features:
Feature F-001: <Feature title>
Description: <short description>

Test Case TC-001: <Test case title>
Objective: <objective>
Preconditions: <preconditions>
Test Data: <test data or NONE>

Steps:
Step 1: <action>
   - Expected Result: <expected outcome>
   - Evidence Image: <URL or MEDIA:/path or NONE>
Step 2: <action>
   - Expected Result: <expected outcome>
   - Evidence Image: <URL or MEDIA:/path or NONE>

Feature F-002: <Feature title>
(same Feature / Test Case / Steps structure as above)

Explore and improve (for future exploration):
What we documented well:
- <areas backed by screenshots or clear UI evidence from this run>

Gaps and open questions:
- <behaviors, edge cases, or roles not yet verified>

Recommended next exploration:
- <specific screens, flows, or data states the next run should visit for this node only>

How to improve this report next time:
- <concrete actions: add test cases, capture missing evidence, validate error paths, etc.>
""".strip()


PFM_FMR_DELIVERY_AGENT_INSTRUCTIONS = (
    "**Canonical delivery — paired `.pfm` + `.FMR` (full node reports):**\n"
    "- The operator UI and future runs rely on **full** per-node EAD reports stored in the DB "
    "and mirrored in the run's **`.FMR`** file.\n"
    "- **Primary path (always):** call **commit_pfm_snapshot** with complete `node_reports[]` "
    "markdown for every new or updated `node_key` (standard structure below). That saves to the DB.\n"
    "- **Delivery file (required at finalize):** write or update the paired **`.FMR`** under this "
    "run's reports folder (`~/.hermes/projects/reports/<execution_id>/`) with the same full "
    "markdown in each `#### Node EAD Report` section — **no placeholders**, no \"awaiting report\" stubs.\n"
    "- **FMR section shape:** `## Per-Node EAD Reports` → `### Node N: <title>` → "
    "`- Node Key: \\`<node_key>\\`` → `#### Node EAD Report` → full markdown body.\n"
    "- If a report exists only in chat, still **commit_pfm_snapshot** (or ensure the chat reply "
    "uses `[Node-Report-Reply-To: <node_key>]` and the standard sections) so the gateway can "
    "recover it from progress history into the DB and `.FMR`.\n"
    "- Do **not** rely on chat-only prose: uncommitted chat text does not populate the operator "
    "node report panel or `.FMR`.\n"
)

PFM_NODE_REPORT_AGENT_INSTRUCTIONS = (
    "Per-node EAD reports (node_reports[] and standalone node-report replies) must be "
    "**detailed** and follow the standard structure below. Write for operators and testers "
    "(plain language, no jargon about tools or snapshots).\n"
    "- Cover every in-scope behavior you can support with evidence from **this run**.\n"
    "- Each feature needs at least one test case with Steps; every step needs Expected Result "
    "and Evidence Image (use NONE only when no screenshot exists).\n"
    "- End every report with **Explore and improve (for future exploration):** so the next "
    "exploration run knows what is solid, what is missing, and how to make the report better.\n"
    "- Do not use markdown tables or # heading lines; use the exact section labels shown.\n"
    "- Scope is **this node only** — never describe child or grandchild PFM nodes here.\n\n"
    + PFM_FMR_DELIVERY_AGENT_INSTRUCTIONS
    + "\n\nStandard report structure:\n"
    + PFM_NODE_REPORT_BODY_TEMPLATE
)
