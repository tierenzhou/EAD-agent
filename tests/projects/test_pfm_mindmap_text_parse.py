"""Tests for recovering PFM nodes from saved mindmap / ASCII tree text."""

from __future__ import annotations

from projects.pfm_artifacts import (
    derive_pfm_nodes_from_ascii_tree_text,
    derive_pfm_nodes_from_mermaid_mindmap_text,
    derive_pfm_nodes_from_saved_mindmap_or_report_text,
    nodes_from_markdown_fenced_mermaid,
)
from projects.models import ProgressLogEntry, ProjectExecute
from projects.pfm_materialize import (
    _build_roots_payload,
    _build_roots_payload_explicit_parents,
    _collect_nodes_for_materialize,
)


def test_ascii_kloud_domains():
    text = """
Kloud 1.0 (10 domains, 40+ features)
├── 1. Authentication & Access — Login, Session, Localization
├── 2. Management — Task Board (3 categories, 33 items)
└── 10. Notifications — Center with badge count
"""
    nodes = derive_pfm_nodes_from_ascii_tree_text(text)
    assert len(nodes) == 4
    assert nodes[0].parent_node_key is None
    assert nodes[1].parent_node_key == nodes[0].node_key
    assert "Authentication" in (nodes[1].title or "")
    assert "/" in nodes[1].node_key


def test_ascii_nested_children():
    text = """
Acme App
├── Area One
│   ├── Sub A
│   └── Sub B
└── Area Two
"""
    nodes = derive_pfm_nodes_from_ascii_tree_text(text)
    assert len(nodes) == 5
    titles = {n.title for n in nodes}
    assert "Sub A" in titles
    assert "Sub B" in titles
    assert "Area Two" in titles


def test_ascii_pipe_in_prefix_same_depth_as_unicode():
    """ASCII ``|`` in the prefix must count like ``│`` so parent links stay correct."""
    text = """
Acme App
├── Area One
|   ├── Sub A
|   └── Sub B
"""
    nodes = derive_pfm_nodes_from_ascii_tree_text(text)
    assert len(nodes) == 4
    area = next(n for n in nodes if (n.title or "") == "Area One")
    subs = [n for n in nodes if (n.title or "").startswith("Sub")]
    assert len(subs) == 2
    for s in subs:
        assert s.parent_node_key == area.node_key


def test_materialize_explicit_roots_match_document_order_not_alpha_sort():
    """
    Operator materialize must use ``parent_node_key`` + source order, not
    ``_partition_mindmap_tree`` alphabetical sibling sort (which reshapes the mindmap).
    """
    text = """
Acme App
├── Zebra Domain
│   ├── Child A
│   └── Child B
└── Alpha Domain
"""
    nodes = derive_pfm_nodes_from_ascii_tree_text(text)
    explicit = _build_roots_payload_explicit_parents(nodes)
    partition = _build_roots_payload(nodes)
    root_ex = explicit[0]
    assert [c["title"] for c in root_ex["children"]] == ["Zebra Domain", "Alpha Domain"]
    zebra = root_ex["children"][0]
    assert [c["title"] for c in zebra["children"]] == ["Child A", "Child B"]
    root_part = partition[0]
    assert [c["title"] for c in root_part["children"]] == ["Alpha Domain", "Zebra Domain"]


def test_materialize_prefers_artifact_when_results_are_shallow():
    class _Store:
        def __init__(self, artifact_nodes: list[dict]) -> None:
            self._artifact_nodes = artifact_nodes

        def list_execution_pfm_artifacts(self, _execution_id: str) -> list[dict]:
            return [{"artifact_type": "pfm_mindmap", "nodes": self._artifact_nodes}]

    shallow = derive_pfm_nodes_from_ascii_tree_text(
        """
Acme App
├── Reporting
└── Notifications
"""
    )
    deep = derive_pfm_nodes_from_ascii_tree_text(
        """
Acme App
├── Reporting
│   ├── Summary Report View
│   └── Report Settings
└── Notifications
"""
    )
    execution = ProjectExecute(
        id="exec-1",
        linked_template_id="tpl-1",
        name="Run 1",
        results=shallow,
        progress_log=[],
    )
    picked = _collect_nodes_for_materialize(
        _Store([n.model_dump() for n in deep]),
        execution,
    )
    titles = [n.title for n in picked]
    assert "Summary Report View" in titles
    assert "Report Settings" in titles


def test_materialize_prefers_artifact_when_progress_nodes_are_shallow():
    class _Store:
        def __init__(self, artifact_nodes: list[dict]) -> None:
            self._artifact_nodes = artifact_nodes

        def list_execution_pfm_artifacts(self, _execution_id: str) -> list[dict]:
            return [{"artifact_type": "pfm_mindmap", "nodes": self._artifact_nodes}]

    deep = derive_pfm_nodes_from_ascii_tree_text(
        """
Acme App
├── Reporting
│   ├── Summary Report View
│   └── Report Settings
└── Notifications
"""
    )
    progress_log = [
        ProgressLogEntry(
            kind="tool_use",
            tool_name="report_running_step",
            tool_input={
                "title": "Reporting",
                "pfm_node": {
                    "node_key": "reporting",
                    "title": "Reporting",
                    "parent_node_key": None,
                    "level": 1,
                    "type": "feature-area",
                },
            },
        )
    ]
    execution = ProjectExecute(
        id="exec-1",
        linked_template_id="tpl-1",
        name="Run 1",
        progress_log=progress_log,
    )
    picked = _collect_nodes_for_materialize(
        _Store([n.model_dump() for n in deep]),
        execution,
    )
    titles = [n.title for n in picked]
    assert "Summary Report View" in titles
    assert "Report Settings" in titles


def test_mermaid_prior_baseline():
    text = """mindmap
  root((Prior Baseline — p1 Test))
    (Run 88c10391)
      ((Status: completed))
"""
    nodes = derive_pfm_nodes_from_mermaid_mindmap_text(text)
    assert len(nodes) >= 2
    assert any("Prior Baseline" in (n.title or "") for n in nodes)


def test_saved_prefers_ascii_over_noise():
    blob = "App Root\n├── a\n"
    nodes = derive_pfm_nodes_from_saved_mindmap_or_report_text(blob)
    assert len(nodes) == 2


def test_fenced_mermaid_in_markdown():
    md = """# Report

```mermaid
mindmap
  root((App))
    ((Area One))
```
"""
    nodes = nodes_from_markdown_fenced_mermaid(md)
    assert len(nodes) >= 2
