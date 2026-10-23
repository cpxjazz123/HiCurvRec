"""The native sweep: a proposer edits a harness, two gates decide, the winner travels.

This replaces the third-party framework's recursive scheduler. The framework's machinery was
generic (recursion modes, rollouts, a seed library, a judge); what this repository actually used
it for is one loop, and that loop is written here in full:

    for each domain, for each task, for each mode:
        stage INV  -> candidate harness  -> static gate -> rollout -> verdict
        stage DOM  -> candidate harness  -> static gate -> rollout -> verdict
        stage AGG  -> the composition of what survived      -> rollout -> verdict
        select the delivered candidate (never a false success)
        record the winner and its lesson in the library, which the NEXT task's evidence carries

Two modes, and the difference is real:

* ``agg`` (``--modes parallel``): the INV and DOM candidates are proposed concurrently from the
  same base evidence, then composed. Finer parallelism, no depth.
* ``seq`` (``--modes sequential``): the candidates are proposed one after another, each from the
  previous round's winner. Depth, no parallelism.

The gates are the repository's own and are not optional: a candidate that restates a declared
value, that edits a frozen threshold, or that claims success on a held-out fixture is discarded,
and the stage records that it fell back to the base harness.
"""

from __future__ import annotations

import json
import shutil
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from . import library as library_mod
from . import prompts, surface as surface_mod

__all__ = ["MODES", "MODE_NAMES", "mode_token", "run_sweep", "Stage"]

MODES = ("agg", "seq")
MODE_NAMES = {"agg": "parallel", "seq": "sequential"}
_MODE_ALIASES = {"agg": "agg", "seq": "seq", "parallel": "agg", "sequential": "seq"}


def mode_token(name: str) -> str:
    """The on-disk token for a mode name: parallel/agg -> "agg", sequential/seq -> "seq"."""
    key = str(name or "").strip().lower()
    if key not in _MODE_ALIASES:
        raise ValueError(
            f"unknown mode {name!r}; expected parallel | sequential (or the on-disk tokens {MODES})")
    return _MODE_ALIASES[key]


@dataclass
class Stage:
    """One proposal, the gate it met, and what came out of it."""

    label: str
    dim: str
    tree: Dict[str, str] = field(default_factory=dict)
    fell_back: bool = False
    kept_base: bool = False
    error: str = ""
    notes: str = ""
    run_id: str = ""
    run_dir: str = ""
    delivered: bool = False
    verdict: Dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        return {"label": self.label, "dim": self.dim, "fell_back": self.fell_back,
                "kept_base": self.kept_base, "error": self.error, "notes": self.notes,
                "run_id": self.run_id, "run_dir": self.run_dir,
                "delivered": self.delivered, "verdict": self.verdict,
                "files": sorted(self.tree)}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_context(context: Path, *, task_id: str, brief: str, harness_tree: Dict[str, str],
                   rollouts: Sequence[Any], lib: library_mod.Library) -> Path:
    """The evidence tree the proposer is given: what the task is, what the harness is now,
    what earlier rollouts produced, where the deliverable goes and what the last tasks learned."""
    context.mkdir(parents=True, exist_ok=True)
    (context / "task.md").write_text(brief, encoding="utf-8")
    surface_mod.write_tree(surface_mod.GENREC_SURFACE, harness_tree, context / "harness",
                           stubs=False)
    roll_dir = context / "rollouts"
    roll_dir.mkdir(exist_ok=True)
    for index, view in enumerate(rollouts or ()):
        payload = getattr(view, "as_dict", None)
        record = payload() if callable(payload) else {"label": f"base{index}"}
        (roll_dir / f"{record.get('label', f'base{index}')}.json").write_text(
            json.dumps(record, indent=2, default=str) + "\n", encoding="utf-8")
    (context / "deliverables").mkdir(exist_ok=True)
    lib.copy_into(context)
    return context


def _run_stage(*, label: str, dim: str, mission: str, fallback: Dict[str, str],
               adapter: Any, prop: Any, ctx_dir: Path, task_id: str, rollouts: Sequence[Any],
               harness_dir: Path, lib: library_mod.Library,
               pool: Sequence[Any] = (), decisions: Optional[Dict[str, str]] = None,
               run_id_of: Optional[Any] = None) -> Stage:
    """Propose, gate, roll out and judge ONE stage. A rejected candidate falls back."""
    context = _write_context(ctx_dir / label, task_id=task_id.replace("/", "_"),
                             brief=adapter.task_prompt(task_id), harness_tree=fallback,
                             rollouts=rollouts, lib=lib)
    stage = Stage(label=label, dim=dim, tree=dict(fallback))
    reply = prop.call(mission, context, label)
    workspace = getattr(reply, "workspace", None)
    candidate = (surface_mod.read_tree(surface_mod.GENREC_SURFACE, workspace)
                 if workspace else {})
    if not candidate:
        stage.kept_base = True
        stage.notes = getattr(reply, "text", "") or "the proposer kept the base harness"
        candidate = dict(fallback)
    stage.notes = getattr(reply, "text", "") or stage.notes
    try:
        adapter.validate_tree(candidate)
    except ValueError as exc:
        stage.fell_back = True
        stage.error = str(exc)
        stage.tree = dict(fallback)
    else:
        stage.tree = candidate
    harness_path = surface_mod.write_tree(surface_mod.GENREC_SURFACE, stage.tree,
                                          harness_dir.parent / "harnesses" / label, stubs=False)
    run_id = f"{label}/{task_id}/{label}"
    stage.run_id = run_id
    run_dir = adapter.execute(task_id, harness_path, run_id)
    stage.run_dir = str(run_dir)
    verdict_path = Path(run_dir) / "verdict.json"
    if verdict_path.is_file():
        stage.verdict = json.loads(verdict_path.read_text(encoding="utf-8"))
    stage.delivered = bool(adapter.delivered(run_dir))
    return stage


def _experience(prop: Any, ctx_dir: Path, label: str) -> Dict[str, Any]:
    """Ask the proposer for the lesson the next task should inherit."""
    reply = prop.call("", ctx_dir, f"experience.{label}")
    text = getattr(reply, "text", "") or ""
    lesson = "the harness transferred"
    evidence = label
    for line in text.splitlines():
        low = line.strip().lower()
        if low.startswith("lesson:"):
            lesson = line.split(":", 1)[1].strip()
        elif low.startswith("evidence:"):
            evidence = line.split(":", 1)[1].strip()
    return {"lesson": lesson, "evidence": evidence, "text": text}


def _run_task(task_id: str, *, adapter: Any, prop: Any, k: int, modes: Sequence[str],
              runs_root: Path, library_root: Path, base_tree: Dict[str, str],
              libs: Dict[str, library_mod.Library], baseline: Sequence[Any]) -> Dict[str, Any]:
    """One task, every requested mode, with its selection and its library entry."""
    domain = adapter.domain_of(task_id)
    task_dir = Path(runs_root) / domain / task_id
    task_dir.mkdir(parents=True, exist_ok=True)
    result: Dict[str, Any] = {"task_id": task_id, "domain": domain, "k": k,
                              "status": "running", "modes": {}, "selections": {},
                              "errors": []}
    ctx_dir = task_dir / "context"
    brief = adapter.task_prompt(task_id)
    for mode in modes:
        lib = libs[mode]
        stages: List[Stage] = []
        if mode == "agg":
            specs = [(f"{dim.lower()}{i}", dim) for i in range(1, k + 1) for dim in ("INV", "DOM")]
            with ThreadPoolExecutor(max_workers=min(len(specs), 2)) as pool:
                stages = list(pool.map(
                    lambda spec: _run_stage(
                        label=spec[0], dim=spec[1],
                        mission=prompts.build_prompt_parts(deliverable="verifier")[
                            "inv" if spec[1] == "INV" else "dom"],
                        fallback=base_tree, adapter=adapter, prop=prop, ctx_dir=ctx_dir,
                        task_id=task_id, rollouts=baseline, harness_dir=task_dir / "harnesses",
                        lib=lib),
                    specs))
            agg_labels = [f"agg{i}" for i in range(1, k + 1)]
            stage_pool = [(s.label, task_dir / "harnesses" / s.label) for s in stages]
            aggs = [_run_stage(
                label=label, dim="AGG",
                mission=prompts.build_prompt_parts(deliverable="verifier")["inv"],
                fallback=base_tree, adapter=adapter, prop=prop, ctx_dir=ctx_dir,
                task_id=task_id, rollouts=baseline, harness_dir=task_dir / "harnesses",
                lib=lib, pool=stage_pool,
                decisions={s.label: s.notes or s.error for s in stages}) for label in agg_labels]
            stages.extend(aggs)
            finals = agg_labels
        else:
            current = dict(base_tree)
            for t in range(1, k + 1):
                for dim in ("INV", "DOM"):
                    stage = _run_stage(
                        label=f"{dim.lower()}{t}", dim=dim,
                        mission=prompts.build_prompt_parts(deliverable="verifier")[
                            "inv" if dim == "INV" else "dom"],
                        fallback=current, adapter=adapter, prop=prop, ctx_dir=ctx_dir,
                        task_id=task_id, rollouts=baseline,
                        harness_dir=task_dir / "harnesses", lib=lib)
                    stages.append(stage)
                    if stage.delivered:
                        current = stage.tree
                composed = _run_stage(
                    label=f"c{t}", dim="COMPOSE",
                    mission=prompts.build_prompt_parts(deliverable="verifier")["inv"],
                    fallback=current, adapter=adapter, prop=prop, ctx_dir=ctx_dir,
                    task_id=task_id, rollouts=baseline, harness_dir=task_dir / "harnesses",
                    lib=lib, pool=[(s.label, task_dir / "harnesses" / s.label) for s in stages],
                    decisions={s.label: s.notes or s.error for s in stages})
                stages.append(composed)
                if composed.delivered:
                    current = composed.tree
            finals = [f"c{t}" for t in range(1, k + 1)]

        # the selection is over the FINALS (the composition of what survived), not over the
        # intermediate stages: a delivered final wins, and a false success never does.
        final_stages = [s for s in stages if s.label in finals] or stages
        delivered = [s for s in final_stages if s.delivered]
        chosen = delivered[0] if delivered else final_stages[0]
        selection = {
            "ok": bool(delivered),
            "chosen_label": chosen.label,
            "chosen_run_id": chosen.run_id,
            "finals": list(finals),
            "candidates": [{"label": s.label, "delivered": s.delivered,
                            "fell_back": s.fell_back, "error": s.error} for s in stages],
            "why": ("the candidate was truthful on every held-out fixture"
                    if delivered else "no candidate delivered; the base harness is the result"),
        }
        sel_dir = task_dir / "selection" / chosen.label
        sel_dir.mkdir(parents=True, exist_ok=True)
        (sel_dir / "selection.json").write_text(json.dumps(selection, indent=2, default=str) + "\n",
                                                encoding="utf-8")
        experience = _experience(prop, ctx_dir / chosen.label, chosen.label)
        lib.add(task_id, chosen.tree, label=chosen.label, why=selection["why"],
                run_id=chosen.run_id, experience=experience)
        result["modes"][mode] = {
            "finals": list(finals),
            "stages": [s.as_dict() for s in stages],
            "selection": selection,
            "library": str(lib.root),
        }
        result["selections"][mode] = selection
    result["status"] = "ok"
    (task_dir / "summary.json").write_text(json.dumps(result, indent=2, default=str) + "\n",
                                           encoding="utf-8")
    return result


def run_sweep(k: int, *, adapter: Any, prop: Any, runs_root: Path, library_root: Path,
              split: str = "all", modes: Sequence[str] = MODES, seed: Optional[int] = None,
              tag: str = "", domain_parallel: int = 1, only_domains: Sequence[str] = (),
              selector_prop: Any = None, pool_policy: str = "finals",
              max_tasks: Optional[int] = None, only_tasks: Sequence[str] = ()) -> Dict[str, Any]:
    """Run the task pool through the propose/gate/select loop, domain by domain."""
    runs_root = Path(runs_root)
    runs_root.mkdir(parents=True, exist_ok=True)
    ordered = [t for t in adapter.tasks(split)]
    if only_domains:
        ordered = [t for t in ordered if adapter.domain_of(t) in set(only_domains)]
    if only_tasks:
        ordered = [t for t in ordered if t in set(only_tasks)]
    if max_tasks is not None:
        ordered = ordered[: int(max_tasks)]
    by_domain: Dict[str, List[str]] = {}
    for task_id in ordered:
        by_domain.setdefault(adapter.domain_of(task_id), []).append(task_id)

    variant = getattr(adapter.substrate, "name", "") or "local"
    base_tree = surface_mod.read_tree(surface_mod.GENREC_SURFACE, adapter.base_harness())
    # the caller hands in the versioned library root (`<base>/<benchmark>/k<k>-<variant>`);
    # the sweep adds only the mode and the domain below it.
    libs = {mode: library_mod.Library(library_root, mode=mode,
                                      domain=next(iter(by_domain), "")) for mode in modes}

    started = _now()
    rows: List[Dict[str, Any]] = []
    results: Dict[str, Dict[str, Any]] = {}
    domain_order = list(by_domain)
    for domain in domain_order:
        carried: List[Any] = []
        for task_id in by_domain[domain]:
            try:
                result = _run_task(task_id, adapter=adapter, prop=prop, k=k, modes=modes,
                                   runs_root=runs_root, library_root=library_root,
                                   base_tree=base_tree, libs=libs, baseline=carried)
            except Exception as exc:  # noqa: BLE001 - a failing task is a result, not a crash
                result = {"task_id": task_id, "domain": domain, "k": k, "status": "failed",
                          "modes": {}, "errors": [f"{type(exc).__name__}: {exc}"], "error": str(exc)}
            results[task_id] = result
            carried = [type("View", (), {"as_dict": lambda self, r=result: r})()]
        rows.append({
            "domain": domain,
            "done": sum(1 for t in by_domain[domain] if results.get(t, {}).get("status") == "ok"),
            "failed": sum(1 for t in by_domain[domain]
                          if results.get(t, {}).get("status") == "failed"),
            "tasks": [{"task": t, "status": results.get(t, {}).get("status", "unknown")}
                      for t in by_domain[domain]],
        })
    return {
        "benchmark": adapter.spec.name,
        "substrate": variant,
        "k": k,
        "modes": list(modes),
        "seed": seed,
        "tag": tag,
        "domain_order": domain_order,
        "max_tasks": max_tasks,
        "only_tasks": list(only_tasks),
        "started_at": started,
        "finished_at": _now(),
        "n_domains": len(domain_order),
        "domains": rows,
        "results": results,
    }
