"""
Run number per template (1, 2, 3, … by start_time). PFM generation V = run number.
"""

from __future__ import annotations

from typing import Any, List, Optional

from .models import ProjectExecute


def _sorted_runs(store: Any, template_id: str) -> List[ProjectExecute]:
    tid = str(template_id or "").strip()
    if not tid:
        return []
    runs = list(store.list_executions(template_id=tid) or [])
    return sorted(runs, key=lambda e: (int(e.start_time or 0), str(e.id or "")))


def ensure_template_run_numbers(store: Any, template_id: str) -> None:
    """Backfill missing run_number values without renumbering existing runs."""
    runs = _sorted_runs(store, template_id)
    next_number = 1
    for ex in runs:
        existing = int(getattr(ex, "run_number", 0) or 0)
        if existing > 0:
            next_number = max(next_number, existing + 1)
            continue
        store.update_execution(ex.id, run_number=next_number)
        next_number += 1


def next_run_number(store: Any, template_id: str) -> int:
    """Allocate the next persistent run_number for a new run."""
    highest = 0
    for ex in _sorted_runs(store, template_id):
        highest = max(highest, int(getattr(ex, "run_number", 0) or 0))
    return highest + 1


def get_run_number(store: Any, ex: ProjectExecute) -> int:
    """PFM V for this run."""
    tid = str(ex.linked_template_id or "").strip()
    if tid:
        ensure_template_run_numbers(store, tid)
        fresh = store.get_execution(ex.id)
        if fresh is not None:
            ex = fresh
    rn = int(getattr(ex, "run_number", 0) or 0)
    if rn > 0:
        return rn
    for idx, row in enumerate(_sorted_runs(store, tid), start=1):
        if row.id == ex.id:
            return idx
    return 1


def prior_run_execution_id(store: Any, ex: ProjectExecute) -> Optional[str]:
    """Previous run on the same template by start_time."""
    tid = str(ex.linked_template_id or "").strip()
    if not tid:
        return None
    prev_id: Optional[str] = None
    for row in _sorted_runs(store, tid):
        if row.id == ex.id:
            return prev_id
        prev_id = row.id
    return None


def prior_run_number(store: Any, ex: ProjectExecute) -> int:
    rn = get_run_number(store, ex)
    return rn - 1 if rn > 1 else 0
