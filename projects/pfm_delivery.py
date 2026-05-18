"""
Agent-delivered PFM files on disk — detect changes for refresh / rebuild.

**Canonical delivery** for rebuild decisions: ``*.FMR`` and ``*.pfm`` only
(filename + last-modified time). Other files on disk (e.g. pfm-mindmap.mmd) are
not used to trigger a DB rebuild.

Multi-user safe: rebuild only when a canonical file exists locally and its mtime
is strictly newer than the DB baseline.
"""

from __future__ import annotations

import logging
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .pfm_artifacts import report_file_path

logger = logging.getLogger(__name__)

# Legacy / export files — not used for refresh-vs-baseline comparison.
_PFM_EXPORT_FILENAMES = (
    "pfm-mindmap.mmd",
    "pfm-report.md",
)

_CANONICAL_GLOBS = ("*.FMR", "*.pfm", "*.fmr")

PFM_REPORT_FILE = "pfm-report.md"
PFM_MINDMAP_FILE = "pfm-mindmap.mmd"


def is_canonical_delivery_filename(name: str) -> bool:
    n = str(name or "").strip().lower()
    return n.endswith(".fmr") or n.endswith(".pfm")


def _canonical_delivery_paths(execution_id: str) -> List[Path]:
    root = report_file_path(execution_id, PFM_REPORT_FILE).parent
    paths: List[Path] = []
    if root.is_dir():
        for pattern in _CANONICAL_GLOBS:
            paths.extend(sorted(root.glob(pattern)))
    paths.extend(_canonical_output_paths(execution_id))
    return sorted({p.resolve() for p in paths}, key=lambda p: str(p))


def _canonical_output_paths(execution_id: str) -> List[Path]:
    """
    Extra canonical delivery location used by some agents:
    ``$HERMES_HOME/output/*.pfm|*.fmr``.

    Match by full execution id first, then short id prefix.
    """
    eid = str(execution_id or "").strip()
    if not eid:
        return []
    short = eid[:8]
    hermes_home = Path(os.getenv("HERMES_HOME", str(Path.home() / ".hermes"))).expanduser()
    out_dir = hermes_home / "output"
    if not out_dir.is_dir():
        return []

    patterns: List[str] = []
    for token in (eid, short):
        if not token:
            continue
        patterns.extend(
            [
                f"*{token}*.pfm",
                f"*{token}*.FMR",
                f"*{token}*.fmr",
            ]
        )

    paths: List[Path] = []
    for pattern in patterns:
        paths.extend(sorted(out_dir.glob(pattern)))
    return sorted({p.resolve() for p in paths}, key=lambda p: str(p))


def _export_delivery_paths(execution_id: str) -> List[Path]:
    """Operator/agent export files under the execution reports directory."""
    root = report_file_path(execution_id, PFM_REPORT_FILE).parent
    paths: List[Path] = []
    for fname in _PFM_EXPORT_FILENAMES:
        path = root / fname
        if path.is_file():
            paths.append(path.resolve())
    return paths


def _slugify_run_label(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", (value or "").lower()).strip("-")


def discover_workspace_delivery_dirs(execution: Any) -> List[Path]:
    """
    Agents sometimes write ``projects/<run>-reports/pfm-mindmap.mmd`` in the repo
    instead of the Hermes reports directory. Collect those folders for ingest.
    """
    projects_dir = Path(__file__).resolve().parent
    if not projects_dir.is_dir():
        return []

    slugs: List[str] = []
    for raw in (
        getattr(execution, "name", None),
        getattr(execution, "id", None),
    ):
        slug = _slugify_run_label(str(raw or ""))
        if slug and slug not in slugs:
            slugs.append(slug)
    name_lower = str(getattr(execution, "name", "") or "").lower()
    if "p1" in name_lower and "p1-test" not in slugs:
        slugs.append("p1-test")

    found: List[Path] = []
    seen: set[str] = set()
    for slug in slugs:
        candidate = projects_dir / f"{slug}-reports"
        if candidate.is_dir():
            key = str(candidate.resolve())
            if key not in seen:
                seen.add(key)
                found.append(candidate)
    for candidate in sorted(projects_dir.glob("*-reports")):
        if not candidate.is_dir():
            continue
        if not any((candidate / fname).is_file() for fname in _PFM_EXPORT_FILENAMES):
            continue
        key = str(candidate.resolve())
        if key not in seen:
            seen.add(key)
            found.append(candidate)
    return found


def ingest_workspace_delivery_exports(store: Any, execution_id: str) -> Dict[str, Any]:
    """
    Copy agent workspace exports into ``reports/<execution_id>/`` and upsert DB artifacts
    so operator Refresh can materialize the committed PFM tree.
    """
    execution = store.get_execution(execution_id)
    if not execution:
        return {"ok": False, "ingested": False, "files": []}

    dest_dir = report_file_path(execution_id, PFM_MINDMAP_FILE).parent
    dest_dir.mkdir(parents=True, exist_ok=True)
    copied: List[str] = []
    now_ms = int(time.time() * 1000)

    for src_dir in discover_workspace_delivery_dirs(execution):
        for fname in _PFM_EXPORT_FILENAMES:
            src = src_dir / fname
            if not src.is_file():
                continue
            try:
                text = src.read_text(encoding="utf-8", errors="replace")
            except OSError as exc:
                logger.warning("[pfm_delivery] could not read %s: %s", src, exc)
                continue
            dest = dest_dir / fname
            dest.write_text(text, encoding="utf-8")
            if fname not in copied:
                copied.append(fname)
            artifact_type = "pfm_mindmap" if fname.endswith(".mmd") else "pfm_report"
            if hasattr(store, "upsert_execution_pfm_artifact"):
                store.upsert_execution_pfm_artifact(
                    execution_id,
                    fname,
                    artifact_type,
                    {
                        "artifact_key": fname,
                        "filename": fname,
                        "format": fname.rsplit(".", 1)[-1],
                        "title": "PFM Mindmap" if "mindmap" in fname else "PFM Report",
                        "content": text,
                        "created_at": now_ms,
                    },
                )
        if copied:
            logger.info(
                "[pfm_delivery] Ingested workspace delivery for %s from %s: %s",
                execution_id,
                src_dir,
                ", ".join(copied),
            )

    return {"ok": True, "ingested": bool(copied), "files": copied}


def _file_time_ms(path: Path) -> int:
    try:
        return int(path.stat().st_mtime * 1000)
    except OSError:
        return 0


def _files_from_paths(paths: List[Path]) -> List[Dict[str, Any]]:
    files: List[Dict[str, Any]] = []
    for path in paths:
        try:
            files.append({"name": path.name, "mtime_ms": _file_time_ms(path)})
        except OSError:
            continue
    files.sort(key=lambda row: str(row.get("name") or "").lower())
    return files


def filter_canonical_delivery_files(files: Any) -> List[Dict[str, Any]]:
    """Keep only .FMR / .pfm rows from a delivery file list."""
    return [row for row in _normalize_files_list(files) if is_canonical_delivery_filename(row.get("name"))]


def _fingerprint_from_files(files: List[Dict[str, Any]]) -> str:
    parts: List[str] = []
    for row in sorted(filter_canonical_delivery_files(files), key=lambda r: str(r.get("name") or "").lower()):
        name = str(row.get("name") or "").strip()
        if not name:
            continue
        try:
            ts = int(row.get("mtime_ms") or 0)
        except Exception:
            ts = 0
        parts.append(f"{name}:{ts}")
    return "|".join(parts)


def _normalize_files_list(raw: Any) -> List[Dict[str, Any]]:
    if not isinstance(raw, list):
        return []
    out: List[Dict[str, Any]] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or "").strip()
        if not name:
            continue
        try:
            ts = int(item.get("mtime_ms") or 0)
        except Exception:
            ts = 0
        out.append({"name": name, "mtime_ms": ts})
    out.sort(key=lambda row: row["name"].lower())
    return out


def _stored_canonical_index(prev_snap: Dict[str, Any]) -> Dict[str, int]:
    """Map canonical filename → stored mtime_ms from snapshot metadata."""
    stored = filter_canonical_delivery_files(prev_snap.get("source_delivery_files"))
    if stored:
        return {row["name"]: int(row["mtime_ms"]) for row in stored}

    # Explicit FMR / pfm fields (newer snapshots)
    out: Dict[str, int] = {}
    fmr_name = str(prev_snap.get("pfm_fmr_based_on_file") or "").strip()
    if fmr_name:
        try:
            out[fmr_name] = int(prev_snap.get("pfm_fmr_based_on_mtime_ms") or 0)
        except Exception:
            pass
    pfm_name = str(prev_snap.get("pfm_pfm_based_on_file") or "").strip()
    if pfm_name:
        try:
            out[pfm_name] = int(prev_snap.get("pfm_pfm_based_on_mtime_ms") or 0)
        except Exception:
            pass
    if out:
        return out

    fp = str(prev_snap.get("source_fingerprint") or "").strip()
    if fp and not _looks_like_legacy_content_hash(fp):
        parsed = filter_canonical_delivery_files(_parse_fingerprint(fp))
        if parsed:
            return {row["name"]: int(row["mtime_ms"]) for row in parsed}

    return {}


def _looks_like_legacy_content_hash(fp: str) -> bool:
    return len(fp) == 64 and all(c in "0123456789abcdef" for c in fp.lower())


def _lookup_stored_mtime(stored_index: Dict[str, int], filename: str) -> Optional[int]:
    if filename in stored_index:
        return stored_index[filename]
    fl = filename.lower()
    for name, ms in stored_index.items():
        if name.lower() == fl:
            return ms
    return None


def has_newer_local_delivery(
    prev_snap: Optional[Dict[str, Any]],
    stamp: Dict[str, Any],
) -> bool:
    """
    True when a canonical delivery file (.FMR or .pfm on disk) is newer than the DB baseline.
    Export files (pfm-mindmap.mmd, pfm-report.md) are ignored for rebuild decisions.
    """
    current_files = filter_canonical_delivery_files(_normalize_files_list(stamp.get("files")))
    if not current_files:
        fp = str(stamp.get("fingerprint") or "").strip()
        if fp and not _looks_like_legacy_content_hash(fp):
            current_files = filter_canonical_delivery_files(_parse_fingerprint(fp))
    if not current_files:
        return False

    if not isinstance(prev_snap, dict):
        return True

    stored_index = _stored_canonical_index(prev_snap)
    if not stored_index:
        stored_fp = str(prev_snap.get("source_fingerprint") or "").strip()
        if stored_fp and _looks_like_legacy_content_hash(stored_fp):
            try:
                stored_mtime = int(prev_snap.get("source_delivery_mtime_ms") or 0)
            except Exception:
                stored_mtime = 0
            if stored_mtime > 0 and current_files:
                current_max = max(int(f.get("mtime_ms") or 0) for f in current_files)
                return current_max > stored_mtime
            return False
        # Committed tree in DB but canonical .FMR/.pfm baseline never recorded — ingest on refresh.
        if current_files and (prev_snap.get("flat_nodes") or prev_snap.get("generated_at")):
            return True
        return False

    for cur in current_files:
        name = str(cur.get("name") or "").strip()
        if not name:
            continue
        try:
            cur_ms = int(cur.get("mtime_ms") or 0)
        except Exception:
            cur_ms = 0
        if cur_ms <= 0:
            continue
        stored_ms = _lookup_stored_mtime(stored_index, name)
        if stored_ms is None:
            return True
        if cur_ms > stored_ms:
            return True
    return False


def delivery_changed(
    prev_snap: Optional[Dict[str, Any]],
    stamp: Dict[str, Any],
) -> bool:
    return has_newer_local_delivery(prev_snap, stamp)


def compute_canonical_delivery_stamp(execution_id: str) -> Dict[str, Any]:
    """Delivery stamp for rebuild decisions: .FMR and .pfm only (name + mtime)."""
    eid = str(execution_id or "").strip()
    if not eid:
        return {"fingerprint": "", "delivery_mtime_ms": 0, "files": []}

    files = _files_from_paths(_canonical_delivery_paths(eid))
    if not files:
        return {"fingerprint": "", "delivery_mtime_ms": 0, "files": []}

    max_mtime_ms = max(int(f.get("mtime_ms") or 0) for f in files)
    return {
        "fingerprint": _fingerprint_from_files(files),
        "delivery_mtime_ms": max_mtime_ms,
        "files": files,
    }


def compute_delivery_stamp(execution_id: str) -> Dict[str, Any]:
    """Full on-disk listing (canonical + export files) for diagnostics only."""
    eid = str(execution_id or "").strip()
    if not eid:
        return {"fingerprint": "", "delivery_mtime_ms": 0, "files": []}

    paths = _canonical_delivery_paths(eid) + _export_delivery_paths(eid)
    files = _files_from_paths(paths)
    if not files:
        return {"fingerprint": "", "delivery_mtime_ms": 0, "files": []}

    canonical = filter_canonical_delivery_files(files)
    max_mtime_ms = max(int(f.get("mtime_ms") or 0) for f in files)
    return {
        "fingerprint": _fingerprint_from_files(canonical) if canonical else "",
        "delivery_mtime_ms": max_mtime_ms,
        "files": files,
    }


def compute_delivery_fingerprint(execution_id: str) -> str:
    return str(compute_delivery_stamp(execution_id).get("fingerprint") or "")


def _parse_fingerprint(fp: str) -> List[Dict[str, Any]]:
    files: List[Dict[str, Any]] = []
    for part in str(fp or "").split("|"):
        part = part.strip()
        if not part or ":" not in part:
            continue
        name, _, ts = part.rpartition(":")
        name = name.strip()
        if not name:
            continue
        try:
            mtime_ms = int(ts)
        except Exception:
            mtime_ms = 0
        files.append({"name": name, "mtime_ms": mtime_ms})
    return _normalize_files_list(files)


def delivery_file_summary(execution_id: str) -> List[Tuple[str, int]]:
    stamp = compute_delivery_stamp(execution_id)
    return [(str(f["name"]), int(f["mtime_ms"])) for f in stamp.get("files") or []]


def _canonical_rows_from_files(files: List[Dict[str, Any]]) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    fmr_row: Optional[Dict[str, Any]] = None
    pfm_row: Optional[Dict[str, Any]] = None
    for row in filter_canonical_delivery_files(files):
        name = str(row.get("name") or "")
        low = name.lower()
        if low.endswith(".fmr") and fmr_row is None:
            fmr_row = row
        elif low.endswith(".pfm") and pfm_row is None:
            pfm_row = row
    return fmr_row, pfm_row


def apply_delivery_baseline_to_snapshot(
    snapshot: Dict[str, Any],
    execution_id: str,
    *,
    stamp: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Record .FMR / .pfm name + mtime used as the DB delivery baseline."""
    out = dict(snapshot)
    eid = str(execution_id or "").strip()
    if stamp is None and eid:
        stamp = compute_delivery_stamp(eid)
    stamp = stamp if isinstance(stamp, dict) else {}

    disk_files = filter_canonical_delivery_files(stamp.get("files"))
    stored_files = filter_canonical_delivery_files(out.get("source_delivery_files"))
    legacy_fp = _looks_like_legacy_content_hash(str(out.get("source_fingerprint") or "").strip())

    if disk_files and (not stored_files or legacy_fp):
        out["source_delivery_files"] = disk_files
        out["source_fingerprint"] = _fingerprint_from_files(disk_files)
        mtime = int(stamp.get("delivery_mtime_ms") or 0)
        if mtime > 0:
            out["source_delivery_mtime_ms"] = mtime

    baseline_files = filter_canonical_delivery_files(out.get("source_delivery_files")) or disk_files
    fmr_row, pfm_row = _canonical_rows_from_files(baseline_files)
    if fmr_row:
        out["pfm_fmr_based_on_file"] = str(fmr_row["name"])
        out["pfm_fmr_based_on_mtime_ms"] = int(fmr_row["mtime_ms"])
    if pfm_row:
        out["pfm_pfm_based_on_file"] = str(pfm_row["name"])
        out["pfm_pfm_based_on_mtime_ms"] = int(pfm_row["mtime_ms"])

    return out


def restore_delivery_file_mtimes(execution_id: str, stamp: Dict[str, Any]) -> None:
    """Restore .FMR / .pfm mtimes after export so Refresh is not fooled."""
    import os

    eid = str(execution_id or "").strip()
    if not eid:
        return
    root = report_file_path(eid, PFM_REPORT_FILE).parent
    if not root.is_dir():
        return
    for row in filter_canonical_delivery_files(stamp.get("files")):
        name = str(row.get("name") or "").strip()
        if not name:
            continue
        path = root / name
        if not path.is_file():
            continue
        try:
            ms = int(row.get("mtime_ms") or 0)
        except Exception:
            ms = 0
        if ms <= 0:
            continue
        sec = ms / 1000.0
        try:
            os.utime(path, (sec, sec))
        except OSError:
            continue


def snapshot_delivery_baseline_for_api(snapshot: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    snap = snapshot if isinstance(snapshot, dict) else {}
    files = filter_canonical_delivery_files(snap.get("source_delivery_files"))

    fmr_file = str(snap.get("pfm_fmr_based_on_file") or "").strip()
    fmr_mtime = int(snap.get("pfm_fmr_based_on_mtime_ms") or 0)
    pfm_file = str(snap.get("pfm_pfm_based_on_file") or "").strip()
    pfm_mtime = int(snap.get("pfm_pfm_based_on_mtime_ms") or 0)

    if not fmr_file or not pfm_file:
        fmr_row, pfm_row = _canonical_rows_from_files(files)
        if fmr_row and not fmr_file:
            fmr_file = str(fmr_row["name"])
            fmr_mtime = int(fmr_row["mtime_ms"])
        if pfm_row and not pfm_file:
            pfm_file = str(pfm_row["name"])
            pfm_mtime = int(pfm_row["mtime_ms"])

    return {
        "baseline_files": files,
        "pfm_fmr_based_on_file": fmr_file or None,
        "pfm_fmr_based_on_mtime_ms": fmr_mtime if fmr_mtime > 0 else None,
        "pfm_pfm_based_on_file": pfm_file or None,
        "pfm_pfm_based_on_mtime_ms": pfm_mtime if pfm_mtime > 0 else None,
        "baseline_fingerprint": str(snap.get("source_fingerprint") or ""),
        "baseline_built_at_ms": int(snap.get("generated_at") or 0),
        "baseline_delivery_mtime_ms": int(snap.get("source_delivery_mtime_ms") or 0),
    }
