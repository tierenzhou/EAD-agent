"""Delivery fingerprint / refresh rebuild (V vs Rev)."""

from __future__ import annotations

import os
import time
from pathlib import Path

import pytest

from projects.models import ExecutionStatus, ProjectExecute, ProjectTemplate
from projects.pfm_delivery import compute_delivery_stamp, delivery_changed
from projects.pfm_refresh import try_refresh_pfm_from_delivery
from projects.store import ProjectStore
from tests.projects.test_pfm_canonical import _commit_minimal_tree


@pytest.fixture()
def refresh_store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> ProjectStore:
    report_root = tmp_path / "reports"
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("EAD_REPORT_DIR", str(report_root))
    monkeypatch.setenv("EAD_PFM_AGENT_AUTHORED", "true")
    store = ProjectStore(db_path=tmp_path / "projects.db")
    store.create_template(ProjectTemplate(id="tpl-1", name="T"))
    store.create_execution(
        ProjectExecute(
            id="exec-v9",
            linked_template_id="tpl-1",
            name="V9",
            status=ExecutionStatus.COMPLETED,
            start_time=1000,
        )
    )
    _commit_minimal_tree(store, "exec-v9")
    snap = store.get_committed_pfm_tree("exec-v9")
    assert snap
    v9 = dict(snap)
    v9["generation"] = 9
    v9["revision"] = 1
    v9["version"] = 1
    v9["source_run_id"] = "exec-v9"
    v9["source_fingerprint"] = "old-fp"
    store.replace_execution_pfm_tree(
        "exec-v9",
        snapshot=v9,
        node_reports=[{"node_key": "root", "title": "Root", "markdown": "# v9\n"}],
    )
    store.update_template("tpl-1", canonical_pfm_execution_id="exec-v9")
    store.create_execution(
        ProjectExecute(
            id="exec-v10",
            linked_template_id="tpl-1",
            name="V10",
            status=ExecutionStatus.COMPLETED,
            inherited_from_execution_id="exec-v9",
            start_time=2000,
        )
    )
    return store


def _write_delivery(
    report_dir: Path,
    execution_id: str,
    content: str,
    *,
    mtime_epoch: float | None = None,
) -> None:
    """Write canonical .FMR / .pfm plus export files; optional fixed mtime."""
    d = report_dir / execution_id
    d.mkdir(parents=True, exist_ok=True)
    fmr = d / f"{execution_id}.FMR"
    pfm = d / f"{execution_id}.pfm"
    mmd = d / "pfm-mindmap.mmd"
    report = d / "pfm-report.md"
    fmr.write_text(f"# FMR {content}\n", encoding="utf-8")
    pfm.write_text(f"PFM {content}\n", encoding="utf-8")
    mmd.write_text(f"mindmap\n  root(({content}))\n", encoding="utf-8")
    report.write_text(f"# Report {content}\n", encoding="utf-8")
    if mtime_epoch is not None:
        os.utime(fmr, (mtime_epoch, mtime_epoch))
        os.utime(pfm, (mtime_epoch + 0.0005, mtime_epoch + 0.0005))
        os.utime(mmd, (mtime_epoch, mtime_epoch))
        os.utime(report, (mtime_epoch + 0.001, mtime_epoch + 0.001))
    else:
        time.sleep(0.02)


def test_missing_local_file_does_not_trigger(refresh_store: ProjectStore) -> None:
    """DB has canonical files; local FMR older → no rebuild."""
    prev = {
        "source_delivery_files": [
            {"name": "exec-v10.FMR", "mtime_ms": 5000},
            {"name": "exec-v10.pfm", "mtime_ms": 5000},
        ],
    }
    stamp = {
        "files": [{"name": "exec-v10.FMR", "mtime_ms": 4000}],
        "delivery_mtime_ms": 4000,
    }
    assert delivery_changed(prev, stamp) is False


def test_older_local_mtime_does_not_trigger(refresh_store: ProjectStore) -> None:
    prev = {
        "source_delivery_files": [
            {"name": "exec-v10.FMR", "mtime_ms": 9000},
            {"name": "exec-v10.pfm", "mtime_ms": 9000},
        ],
    }
    stamp = {
        "files": [
            {"name": "exec-v10.FMR", "mtime_ms": 8000},
            {"name": "exec-v10.pfm", "mtime_ms": 8500},
        ],
        "delivery_mtime_ms": 8500,
    }
    assert delivery_changed(prev, stamp) is False


def test_no_local_files_never_triggers(refresh_store: ProjectStore) -> None:
    prev = {
        "source_delivery_files": [{"name": "exec-v10.FMR", "mtime_ms": 1000}],
    }
    assert delivery_changed(prev, {"files": [], "fingerprint": "", "delivery_mtime_ms": 0}) is False


def test_delivery_changed_by_filename_and_time(refresh_store: ProjectStore) -> None:
    prev = {
        "source_fingerprint": "exec-v10.FMR:100|exec-v10.pfm:200",
        "source_delivery_files": [
            {"name": "exec-v10.FMR", "mtime_ms": 100},
            {"name": "exec-v10.pfm", "mtime_ms": 200},
        ],
        "source_run_id": "exec-v10",
    }
    same = {
        "fingerprint": "exec-v10.FMR:100|exec-v10.pfm:200",
        "delivery_mtime_ms": 200,
        "files": [
            {"name": "exec-v10.FMR", "mtime_ms": 100},
            {"name": "exec-v10.pfm", "mtime_ms": 200},
        ],
    }
    newer = {
        "fingerprint": "exec-v10.FMR:150|exec-v10.pfm:200",
        "delivery_mtime_ms": 200,
        "files": [
            {"name": "exec-v10.FMR", "mtime_ms": 150},
            {"name": "exec-v10.pfm", "mtime_ms": 200},
        ],
    }
    assert delivery_changed(prev, same) is False
    assert delivery_changed(prev, newer) is True


def test_delivery_changed_legacy_mtime(refresh_store: ProjectStore) -> None:
    """Legacy content-hash snapshots fall back to max mtime when per-file list is absent."""
    prev = {
        "source_run_id": "exec-v10",
        "source_delivery_mtime_ms": 100,
        "source_fingerprint": "a" * 64,
    }
    newer = {
        "files": [{"name": "exec-v10.FMR", "mtime_ms": 200}],
        "delivery_mtime_ms": 200,
    }
    older = {
        "files": [{"name": "exec-v10.FMR", "mtime_ms": 50}],
        "delivery_mtime_ms": 50,
    }
    assert delivery_changed(prev, newer) is True
    assert delivery_changed(prev, older) is False


def test_refresh_first_commit_run2_rev1(
    refresh_store: ProjectStore, tmp_path: Path
) -> None:
    report_root = tmp_path / "reports"
    _write_delivery(report_root, "exec-v10", "new-run")
    out = try_refresh_pfm_from_delivery(refresh_store, "exec-v10", promote_template_canonical=False)
    assert out.get("ok") is True
    assert out.get("code") == "materialized"
    snap = refresh_store.get_committed_pfm_tree("exec-v10")
    assert snap is not None
    assert snap.get("generation") == 2
    assert snap.get("revision") == 1
    assert snap.get("source_run_id") == "exec-v10"
    assert str(snap.get("source_fingerprint") or "")


def test_refresh_same_run_bumps_rev(
    refresh_store: ProjectStore, tmp_path: Path
) -> None:
    report_root = tmp_path / "reports"
    base = time.time()
    _write_delivery(report_root, "exec-v10", "rev1", mtime_epoch=base)
    try_refresh_pfm_from_delivery(refresh_store, "exec-v10", promote_template_canonical=False)
    _write_delivery(report_root, "exec-v10", "rev2", mtime_epoch=base + 10)
    out = try_refresh_pfm_from_delivery(refresh_store, "exec-v10", promote_template_canonical=False)
    assert out.get("code") == "materialized"
    snap = refresh_store.get_committed_pfm_tree("exec-v10")
    assert snap.get("generation") == 2
    assert snap.get("revision") == 2


def test_refresh_preserves_agent_node_reports(
    refresh_store: ProjectStore, tmp_path: Path
) -> None:
    """Delivery refresh must not replace rich agent reports with materialized stubs."""
    _commit_minimal_tree(refresh_store, "exec-v10")
    rich_md = (
        "# Login\n\n"
        "Node Summary: Login flow\n\n"
        "## Features\n\n"
        "- Feature F-001: Dashboard widgets\n\n"
        "## Test Case TC-001\n\n"
        "Verify export loads.\n"
    )
    refresh_store.save_node_ead_report_artifact(
        "exec-v10",
        node_key="auth/login",
        title="Login",
        content=rich_md,
    )
    report_root = tmp_path / "reports"
    _write_delivery(report_root, "exec-v10", "rev-with-reports")
    out = try_refresh_pfm_from_delivery(refresh_store, "exec-v10", promote_template_canonical=False)
    assert out.get("code") == "materialized"
    art = refresh_store.get_execution_pfm_artifact(
        "exec-v10",
        "node-ead-report-auth-login.md",
    )
    assert art is not None
    content = str(art.get("content") or "")
    assert "Features" in content
    assert "Test Case TC-001" in content
    assert "materialized from run data" not in content.lower()


def test_refresh_no_changes_when_files_unchanged(
    refresh_store: ProjectStore, tmp_path: Path
) -> None:
    report_root = tmp_path / "reports"
    _write_delivery(report_root, "exec-v10", "stable")
    try_refresh_pfm_from_delivery(refresh_store, "exec-v10", promote_template_canonical=False)
    out = try_refresh_pfm_from_delivery(refresh_store, "exec-v10", promote_template_canonical=False)
    assert out.get("code") == "no_changes"


def test_commit_with_legacy_hash_still_saves_delivery_files(
    refresh_store: ProjectStore, tmp_path: Path,
) -> None:
    """Snapshots with only a legacy SHA fingerprint must still store per-file baseline."""
    report_root = tmp_path / "reports"
    _write_delivery(report_root, "exec-v9", "agent-v1")
    snap = refresh_store.get_committed_pfm_tree("exec-v9")
    assert snap
    legacy = dict(snap)
    legacy["source_fingerprint"] = "a" * 64
    legacy.pop("source_delivery_files", None)
    legacy.pop("pfm_fmr_based_on_file", None)
    legacy.pop("pfm_fmr_based_on_mtime_ms", None)
    legacy.pop("pfm_pfm_based_on_file", None)
    legacy.pop("pfm_pfm_based_on_mtime_ms", None)
    refresh_store.replace_execution_pfm_tree(
        "exec-v9",
        snapshot=legacy,
        node_reports=[{"node_key": "root", "title": "Root", "markdown": "# v9\n"}],
    )
    raw = refresh_store._get_pfm_tree_snapshot_raw("exec-v9")
    assert raw
    files = raw.get("source_delivery_files") or []
    assert any(str(f.get("name") or "").endswith(".FMR") for f in files)
    assert any(str(f.get("name") or "").endswith(".pfm") for f in files)
    assert int(raw.get("pfm_fmr_based_on_mtime_ms") or 0) > 0
    assert int(raw.get("pfm_pfm_based_on_mtime_ms") or 0) > 0
    fp = str(raw.get("source_fingerprint") or "")
    assert len(fp) != 64 or not all(c in "0123456789abcdef" for c in fp.lower())


def test_compute_delivery_stamp_uses_filename_not_content(
    refresh_store: ProjectStore, tmp_path: Path
) -> None:
    """Same filename + timestamp → same stamp even when file bytes change."""
    import os

    report_root = tmp_path / "reports"
    d = report_root / "exec-v10"
    d.mkdir(parents=True, exist_ok=True)
    fmr = d / "exec-v10.FMR"
    fmr.write_text("# FMR A\n", encoding="utf-8")
    stamp_a = compute_delivery_stamp("exec-v10")
    atime = fmr.stat().st_atime
    mtime = fmr.stat().st_mtime
    fmr.write_text("# FMR completely different\n", encoding="utf-8")
    os.utime(fmr, (atime, mtime))
    stamp_b = compute_delivery_stamp("exec-v10")
    assert stamp_a.get("fingerprint") == stamp_b.get("fingerprint")


def test_export_mmd_touch_does_not_trigger_rebuild() -> None:
    """Touching pfm-mindmap.mmd alone must not rebuild when .FMR/.pfm are unchanged."""
    prev = {
        "source_delivery_files": [
            {"name": "exec-v10.FMR", "mtime_ms": 5000},
            {"name": "exec-v10.pfm", "mtime_ms": 5000},
        ],
    }
    stamp = {
        "fingerprint": "exec-v10.FMR:5000|exec-v10.pfm:5000",
        "delivery_mtime_ms": 9000,
        "files": [
            {"name": "exec-v10.FMR", "mtime_ms": 5000},
            {"name": "exec-v10.pfm", "mtime_ms": 5000},
            {"name": "pfm-mindmap.mmd", "mtime_ms": 9000},
        ],
    }
    assert delivery_changed(prev, stamp) is False
