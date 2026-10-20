"""Command line for the GeneRec Harness² integration.

    python -m tools.harness2_genrec.cli tasks
    python -m tools.harness2_genrec.cli sweep --k 1 --modes parallel --offline

``--offline`` runs the whole pipeline with the scripted proposer: no model, no
network, no credentials, no GPU. It is the P0 acceptance path (issue #101 §D).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from harness2.core import library as LIB
from harness2.core import schedule as SCHED
from harness2.core.spec import ChannelSpec

from . import tasks
from .adapter import GenrecAdapter
from .proposer import GenrecProposer, ScriptedProposer

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUNS_ROOT = Path(
    os.environ.get(
        "HARNESS2_GENREC_RUNS", "/home/wlia0047/ar57_scratch/wenyu/harness2_runs/genrec"
    )
)
OFFLINE_CHANNEL = ChannelSpec("opencode", "local/deterministic")


def _modes(raw: str) -> tuple[str, ...]:
    tokens = {SCHED.mode_token(name) for name in raw.split(",") if name.strip()}
    return tuple(t for t in SCHED.MODES if t in tokens)


def _library_root(runs_root: Path, adapter: GenrecAdapter, k: int, seed: int | None, tag: str) -> Path:
    variant = adapter.substrate.name
    if seed is not None:
        variant += f"-s{int(seed)}"
    if tag:
        variant += f"-{tag}"
    return LIB.root_for(runs_root / "library", benchmark=adapter.spec.name, k=k, variant=variant)


def cmd_tasks(args: argparse.Namespace) -> int:
    for task_id in tasks.all_tasks():
        task = tasks.task(task_id)
        served = len(tasks.served_fixtures(task_id))
        heldout = len(tasks.heldout_files(task_id))
        print(f"{task_id}\tdomain={task.domain}\tserved={served}\theldout={heldout}")
    return 0


def run_sweep(
    adapter: GenrecAdapter,
    prop,
    runs_root: Path,
    *,
    k: int = 1,
    modes: tuple[str, ...] = ("agg",),
    split: str = "all",
    seed: int | None = None,
    tag: str = "",
    workers: int = 1,
    only_domains: tuple[str, ...] = (),
) -> dict:
    """Run the framework's scheduler over the adapter's task pool."""
    return SCHED.run_sweep(
        k,
        adapter=adapter,
        prop=prop,
        runs_root=runs_root / "adapt",
        library_root=_library_root(runs_root, adapter, k, seed, tag),
        split=split,
        modes=modes,
        seed=seed,
        tag=tag,
        domain_parallel=workers,
        only_domains=only_domains,
        pool_policy="finals",
    )


def cmd_sweep(args: argparse.Namespace) -> int:
    runs_root = Path(args.runs_dir).expanduser().resolve()
    runs_root.mkdir(parents=True, exist_ok=True)

    if args.offline:
        channel = OFFLINE_CHANNEL
        prop = ScriptedProposer(OFFLINE_CHANNEL)
    else:
        if not args.model.strip():
            print(
                "error: --model is required without --offline (or set HARNESS2_GENREC_MODEL)",
                file=sys.stderr,
            )
            return 1
        channel = ChannelSpec(args.channel, args.model)
        prop = GenrecProposer(channel)
    adapter = GenrecAdapter(
        REPO_ROOT,
        runs_root,
        task_ids=[t.strip() for t in args.tasks.split(",") if t.strip()] or None,
        channel=channel,
    )

    report = run_sweep(
        adapter,
        prop,
        runs_root,
        k=args.k,
        modes=_modes(args.modes),
        split=args.split,
        seed=args.seed,
        tag=args.tag,
        workers=args.workers,
        only_domains=tuple(d.strip() for d in args.only_domains.split(",") if d.strip()),
    )

    record = runs_root / "sweep.json"
    record.write_text(json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8")

    done = sum(int(d.get("done", 0)) for d in report["domains"])
    failed = sum(int(d.get("failed", 0)) for d in report["domains"])
    print(f"\nsweep: {done} task(s) ok, {failed} failed")
    print(f"record: {record}")
    print(f"runs:   {runs_root / 'adapt'}")
    print(f"library:{_library_root(runs_root, adapter, args.k, args.seed, args.tag)}")
    for domain in report["domains"]:
        for row in domain.get("tasks", []):
            print(f"  [{domain['domain']}] {row['task']}: {row['status']}")
    return 1 if failed else 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="tools.harness2_genrec.cli", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("tasks", help="list the task pool").set_defaults(func=cmd_tasks)

    sweep = sub.add_parser("sweep", help="run the recursion over the task pool")
    sweep.add_argument("--k", type=int, default=1, help="task-level recursion budget")
    sweep.add_argument("--modes", default="parallel", help="parallel | sequential, comma list")
    sweep.add_argument("--split", default="all", choices=("all", "train", "test"))
    sweep.add_argument("--tasks", default="", help="comma list of task ids")
    sweep.add_argument("--only-domains", default="", help="comma list of domains")
    sweep.add_argument("--workers", type=int, default=1, help="domains run concurrently")
    sweep.add_argument("--seed", type=int, default=None)
    sweep.add_argument("--tag", default="")
    sweep.add_argument("--runs-dir", default=str(DEFAULT_RUNS_ROOT))
    sweep.add_argument("--offline", action="store_true", help="scripted proposer, no model")
    sweep.add_argument("--channel", default="claude_cli", help="proposer channel")
    sweep.add_argument("--model", default=os.environ.get("HARNESS2_GENREC_MODEL", ""))
    sweep.set_defaults(func=cmd_sweep)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        return int(args.func(args))
    except (RuntimeError, ValueError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
