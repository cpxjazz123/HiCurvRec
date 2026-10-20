"""Parallel mechanism exploration with a bounded budget.

Several mechanisms are explored at once, each in its own arm tree, each with its own
proposal, its own judges and its own manifest. The budget is the point: a cold Stage2 run
wants the whole machine, so the number of *cold runs* in flight is bounded, and the cheap
stages -- implement, verify, repair, probe -- are bounded separately because they are
cheap.

Two invariants hold no matter how many arms run:

* the arms are disjoint. Each has its own tree, its own result directory and its own
  run directory, so no arm can overwrite another's output or evidence;
* the judges and the evaluation are shared and read-only. A candidate edits its own code
  and nothing else.
"""

from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from .lifecycle import LifecycleConfig, MechanismRun
from .proposal import load_proposal


def explore(
    proposals: Sequence[Path],
    *,
    repo_root: Path,
    runs_root: Path,
    base_tree: Path,
    max_parallel: int = 2,
    cold_runs: int = 0,
    model: str = "",
    timeout: int = 1800,
    max_repair_attempts: int = 3,
    probe_epochs: int = 2,
    skip_cold: bool = True,
) -> List[Dict[str, Any]]:
    """Explore several mechanisms, and report what each one reached.

    ``cold_runs`` is the number of cold Stage2 runs allowed to be in flight at once;
    the rest of the arms stop at the admission record. ``skip_cold`` forbids cold runs
    entirely, which is the mode the development sweep uses.
    """
    runs_root = Path(runs_root)
    runs_root.mkdir(parents=True, exist_ok=True)
    documents = [load_proposal(path) for path in proposals]
    rows: List[Dict[str, Any]] = []
    lock_free = True  # each run owns its own files; the only shared state is the manifest list

    def one(index: int, document: Dict[str, Any]) -> Dict[str, Any]:
        started = time.time()
        mechanism = document.get("mechanism_id", f"arm{index}")
        config = LifecycleConfig(
            repo_root=Path(repo_root),
            runs_root=runs_root,
            base_tree=Path(base_tree),
            model=model,
            timeout=timeout,
            max_repair_attempts=max_repair_attempts,
            probe_epochs=probe_epochs,
            arm_name=str(document.get("arm") or f"TIGER_r9_{mechanism}"),
        )
        run = MechanismRun(config, document)
        run.freeze_proposal()
        stages = [run.implement()]
        verify = run.verify()
        stages.append(verify)
        if not verify.ok:
            stages.append(run.repair())
        stages.append(run.probe())
        if all(stage.ok for stage in stages):
            stages.append(run.admit())
        return {
            "mechanism_id": mechanism,
            "arm": str(run.arm_tree()),
            "state": run.state,
            "manifest": str(run.root / "manifest.json"),
            "elapsed_s": round(time.time() - started, 1),
            "stages": {stage.name: stage.state for stage in stages},
        }

    with ThreadPoolExecutor(max_workers=max(1, int(max_parallel))) as pool:
        futures = [pool.submit(one, index, document) for index, document in enumerate(documents)]
        for future in futures:
            try:
                rows.append(future.result())
            except Exception as exc:  # noqa: BLE001
                rows.append({"state": "failed", "error": f"{type(exc).__name__}: {exc}"})

    record = {
        "explored": len(rows),
        "max_parallel": max_parallel,
        "cold_runs_allowed": cold_runs,
        "cold_runs_skipped": bool(skip_cold),
        "rows": rows,
    }
    (runs_root / "exploration.json").write_text(
        json.dumps(record, indent=2, default=str) + "\n", encoding="utf-8"
    )
    return rows


__all__ = ["explore"]
