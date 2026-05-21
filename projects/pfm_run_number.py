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
    """Assign run_number 1..N on this template (oldest first)."""
    runs = _sorted_runs(store, template_id)
    for idx, ex in enumerate(runs, start=1):
        if int(getattr(ex, "run_number", 0) or 0) == idx:
            continue
        store.update_execution(ex.id, run_number=idx)


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
