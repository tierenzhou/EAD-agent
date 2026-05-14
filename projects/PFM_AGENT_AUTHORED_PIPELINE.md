# Agent-authored PFM tree pipeline

This pipeline makes the LLM agent the single source of truth for the PFM
mindmap and per-node EAD reports. Earlier code derived the mindmap from
heterogeneous sources (`execution.results`, progress-log `pfm_node` payloads,
and text-only fallbacks), which produced mindmaps that did not match the
hierarchical tree the agent described in its chat answer. The new pipeline
fixes that by routing all knowledge through a single atomic tool.

## Components

| Layer    | Artifact                                    | Purpose                                                       |
| -------- | ------------------------------------------- | ------------------------------------------------------------- |
| Tool     | `commit_pfm_snapshot` (in `tools.py`)       | Atomic commit of the full tree; per-node EAD reports are merged with prior artifacts on later commits (see below). |
| Schema   | `pfm-tree.json` (`execution_pfm_artifacts`) | Canonical snapshot keyed by `(execution_id, artifact_key)`.   |
| Schema   | `pfm-view-state.json`                       | Per-run saved UI selection (node path, scope, depth cap).     |
| API      | `GET pfm/tree`                              | Read the latest committed snapshot for an execution.          |
| API      | `GET pfm/mindmap?scope=...`                 | Render Mermaid for `top` / `subtree` / `path` / `full`.       |
| API      | `GET/PUT pfm/view`                          | Get/set the saved UI navigation state.                        |
| API      | `GET artifacts/node-report?node_key=...`    | Fetch the pre-saved Markdown EAD report for a node.           |
| Resolver | `resolve_pfm_nodes_for_mindmap` (strict)    | Reads only the committed tree; no progress-text fallback.     |
| UI       | `pfm-mindmap-mermaid` + `pfm-node-report-panel` | Render from gateway, persist view state, show staleness.  |

## Invariants

1. Mindmap and per-node reports are derived only from the latest snapshot.
2. The same input always produces the same Mermaid for any scope. Refresh
   cadence does not change semantics.
3. Until a run sends its first snapshot, the UI shows an explicit "Awaiting
   first committed PFM tree (vN expected)" placeholder.
4. **EAD reports on commit:** the **first** snapshot for a run must include
   non-empty Markdown for **every** node. On **later** checkpoints (typically
   every 2–5 minutes), the agent may send reports only for **new, changed, or
   deleted** work: unchanged nodes are **carried forward** from the prior
   committed tree’s saved `node_ead_report` artifacts when their `node_key` is
   unchanged. Empty markdown is rejected; unknown `node_key` entries in
   `node_reports` are rejected.

## Feature flag

The strict resolver is gated by the `EAD_PFM_AGENT_AUTHORED` environment
variable, read by `projects/pfm_artifacts.py:_agent_authored_enabled`.

| `EAD_PFM_AGENT_AUTHORED` | Behaviour                                                                                                            |
| ------------------------ | -------------------------------------------------------------------------------------------------------------------- |
| `true` (default)         | Strict mode. `resolve_pfm_nodes_for_mindmap` reads only the committed `pfm-tree.json`. Old runs show the placeholder. |
| `false`                  | Legacy mode. Merge `execution.results` with progress-log inferred nodes.                                              |

Set to `false` to fall back to the old resolver while live-debugging or to
service installs that have not adopted `commit_pfm_snapshot` yet. The default
should remain `true` in production so the UI mindmap always matches what the
agent declared.

## Rollout

1. **Ship behind the default-on flag.** All new runs will start prompting the
   agent for `commit_pfm_snapshot` (see `_PFM_SNAPSHOT_COMMIT_PROMPT` in
   `executor.py`). Old runs that have no committed tree will display the
   placeholder until the agent authors a snapshot.
2. **Verify with a single run.** Start an Explore run, watch for at least one
   `commit_pfm_snapshot` invocation, then refresh the UI mindmap. It must
   match the tree the agent described in chat.
3. **Operator option.** If a specific install needs the old behaviour while
   downstream consumers migrate, set `EAD_PFM_AGENT_AUTHORED=false` in the
   gateway process environment.
4. **Backfill (optional).** Old runs without a snapshot can be re-authored by
   asking the agent to call `commit_pfm_snapshot` on the existing chat
   session; no automatic backfill is performed.

## Persistence layout

- `execution_pfm_artifacts(execution_id, artifact_key, …)`:
  - `pfm-tree.json` (`artifact_type="pfm_tree"`) — the snapshot.
  - `pfm-view-state.json` (`artifact_type="pfm_view_state"`) — saved UI state.
  - `node-ead-report-<slug>.md` (`artifact_type="node_ead_report"`) — pre-saved
    Markdown reports written by `replace_execution_pfm_tree`.
  - `pfm-mindmap.mmd` / `pfm-report.md` — rebuilt deterministically from the
    snapshot by `build_pfm_artifact_payloads`.

## Phase 1 inheritance

`_build_phase1_canonical_baseline_message` now includes the snapshot's
`pfm_tree_version` and `generated_at` in the baseline injected into Phase 1
of subsequent runs. The agent reconciles against that version explicitly when
producing its own next snapshot.
