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
from .opencode import preflight as opencode_preflight
from .proposer import GenrecProposer, OpenCodeProposer, ScriptedProposer

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
        model = args.model.strip() or os.environ.get("HARNESS2_OPENCODE_MODEL", "")
        if not model:
            from . import opencode as _opencode

            model = _opencode.model_name()
        channel = ChannelSpec("opencode", model)
        prop = OpenCodeProposer(
            channel,
            audit_dir=runs_root / "_audit",
            model=model,
            timeout=args.timeout,
            repo_root=REPO_ROOT,
        )
    adapter = GenrecAdapter(
        REPO_ROOT,
        runs_root,
        task_ids=[t.strip() for t in args.tasks.split(",") if t.strip()] or None,
        channel=channel,
        harness_root=Path(args.base_harness) if args.base_harness else None,
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


def cmd_preflight(args: argparse.Namespace) -> int:
    """Check that the real agent backend can be used on this machine."""
    report = opencode_preflight(args.model, probe=not args.no_probe)
    print(json.dumps(report.as_dict(), indent=2, default=str))
    return 0 if report.ok else 1


def cmd_mcp_check(args: argparse.Namespace) -> int:
    """Prove that the genrec MCP is really reachable from an OpenCode session."""
    from . import mcp_check

    report = mcp_check.run(
        Path(args.out),
        model=args.model,
        timeout=args.timeout,
        tool=args.tool,
    )
    (Path(args.out) / "mcp_check_report.json").write_text(
        json.dumps(report, indent=2, default=str), encoding="utf-8"
    )
    print(json.dumps(report, indent=2, default=str)[:4000])
    return 0 if report.get("ok") else 1


def cmd_repair_demo(args: argparse.Namespace) -> int:
    """Reject a defective implementation with the judges, repair it with OpenCode, recheck."""
    from . import repair_demo

    report = repair_demo.run(
        Path(args.tree) if args.tree else REPO_ROOT / "stage2_RQ-VAE" / "TIGER_r9_mvg_reference",
        Path(args.spec),
        args.mutant,
        Path(args.out),
        model=args.model,
        timeout=args.timeout,
    )
    (Path(args.out) / "repair_report.json").write_text(
        json.dumps(report, indent=2, default=str), encoding="utf-8"
    )
    print(f"defect       : {report['mutant']}")
    print(f"judged before: {report['judged_before']['verdict']}")
    print(f"judged after : {report['judged_after']['verdict']}")
    print(f"repaired     : {report['rejected_then_repaired']}")
    print(f"report       : {args.out}/repair_report.json")
    return 0 if report["rejected_then_repaired"] else 1


def cmd_lifecycle(args: argparse.Namespace) -> int:
    """Run one mechanism through the whole lifecycle."""
    from .lifecycle import LifecycleConfig, MechanismRun
    from .proposal import load_proposal

    proposal = load_proposal(Path(args.proposal))
    runs_root = Path(args.runs_dir)
    runs_root.mkdir(parents=True, exist_ok=True)
    config = LifecycleConfig(
        repo_root=REPO_ROOT,
        runs_root=runs_root,
        base_tree=Path(args.base_tree) if args.base_tree else REPO_ROOT
        / "stage2_RQ-VAE"
        / "TIGER_r9_mvg_reference",
        model=args.model,
        timeout=args.timeout,
        max_repair_attempts=args.max_repair_attempts,
        probe_epochs=args.probe_epochs,
        arm_name=args.arm,
        probe_checkpoint=args.probe_checkpoint,
    )
    run = MechanismRun(config, proposal)
    run.freeze_proposal()
    stages = [run.implement()]
    verify = run.verify()
    stages.append(verify)
    if not verify.ok:
        stages.append(run.repair())
    stages.append(run.probe())
    if all(stage.ok for stage in stages):
        stages.append(run.admit())
        if not args.skip_stage2:
            stages.append(run.stage2())
            stages.append(run.evaluate())
    for stage in stages:
        print(f"  {stage.name:<16} {stage.state:<22} {stage.detail[:90]}")
    print(f"\nstate    : {run.state}")
    print(f"manifest : {run.root / 'manifest.json'}")
    return 0 if run.state in ("er_evaluated", "stage3_complete") else 1


def cmd_explore(args: argparse.Namespace) -> int:
    """Explore several mechanisms in parallel, each in its own arm."""
    from .parallel import explore

    proposals = [Path(p.strip()) for p in args.proposals.split(",") if p.strip()]
    rows = explore(
        proposals,
        repo_root=REPO_ROOT,
        runs_root=Path(args.runs_dir),
        base_tree=Path(args.base_tree) if args.base_tree else REPO_ROOT
        / "stage2_RQ-VAE"
        / "TIGER_r9_mvg_reference",
        max_parallel=args.max_parallel,
        cold_runs=args.cold_runs,
        model=args.model,
        timeout=args.timeout,
        skip_cold=not args.allow_cold,
    )
    for row in rows:
        print(f"  {row.get('mechanism_id', '?'):<40} {row.get('state', '?'):<22} {row.get('arm', '')}")
    print(f"\nmanifest: {Path(args.runs_dir) / 'exploration.json'}")
    return 0 if all(row.get("state") not in ("failed",) for row in rows) else 1


def cmd_campaign(args: argparse.Namespace) -> int:
    """Run the autonomous research campaign."""
    from .campaign import CampaignConfig, execute

    config = CampaignConfig.load(Path(args.config))
    state = execute(config)
    print()
    print(f"state      : {state.get('state', 'finished')}")
    print(f"arms       : {len(state.get('arms', []))}")
    print(f"qualified  : {state.get('qualified', [])}")
    print(f"leaderboard: {Path(config.out_dir) / 'leaderboard.json'}")
    print(f"report     : {Path(config.out_dir) / 'report.md'}")
    return 0 if state.get("errors") == [] else 1


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="tools.harness2_genrec.cli", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("tasks", help="list the task pool").set_defaults(func=cmd_tasks)

    pre = sub.add_parser("preflight", help="check the OpenCode backend on this machine")
    pre.add_argument("--model", default="", help="the model to probe")
    pre.add_argument("--no-probe", action="store_true", help="only check the CLI, do not call it")
    pre.set_defaults(func=cmd_preflight)

    mcp = sub.add_parser("mcp-check", help="prove the genrec MCP is callable from a session")
    mcp.add_argument("--out", default="/tmp/h2_mcp_check", help="where the evidence is written")
    mcp.add_argument("--model", default="")
    mcp.add_argument("--timeout", type=int, default=900)
    mcp.add_argument("--tool", default="stage2_status", help="the read-only MCP tool to call")
    mcp.set_defaults(func=cmd_mcp_check)

    repair = sub.add_parser(
        "repair-demo", help="reject a defective implementation, repair it, recheck it"
    )
    repair.add_argument("--tree", default="", help="the arm tree to copy and break")
    repair.add_argument(
        "--spec",
        default=str(
            REPO_ROOT / "tools" / "mvg" / "specs" / "masked_directed_codebook_margin.json"
        ),
    )
    repair.add_argument("--mutant", default="symmetric_for_directed", help="the defect to inject")
    repair.add_argument("--out", default="/tmp/h2_repair_demo")
    repair.add_argument("--model", default="")
    repair.add_argument("--timeout", type=int, default=1800)
    repair.set_defaults(func=cmd_repair_demo)

    life = sub.add_parser("lifecycle", help="run one mechanism through the whole lifecycle")
    life.add_argument("--proposal", required=True, help="the frozen proposal (JSON)")
    life.add_argument("--base-tree", default="", help="the tree the mechanism is built on")
    life.add_argument("--arm", default="", help="the arm tree the mechanism lives in")
    life.add_argument("--runs-dir", default="/tmp/h2_lifecycle")
    life.add_argument("--model", default="")
    life.add_argument("--timeout", type=int, default=1800)
    life.add_argument("--max-repair-attempts", type=int, default=3)
    life.add_argument("--probe-epochs", type=int, default=2)
    life.add_argument(
        "--probe-checkpoint",
        default="",
        help="the parent checkpoint the launch's smoke probe uses; it is archived before the "
        "cold run and never becomes a warm start",
    )
    life.add_argument("--skip-stage2", action="store_true", help="stop before the cold run")
    life.set_defaults(func=cmd_lifecycle)

    explore = sub.add_parser("explore", help="explore several mechanisms in parallel")
    explore.add_argument("--proposals", required=True, help="comma list of proposal files")
    explore.add_argument("--base-tree", default="")
    explore.add_argument("--runs-dir", default="/tmp/h2_explore")
    explore.add_argument("--max-parallel", type=int, default=2)
    explore.add_argument("--cold-runs", type=int, default=0, help="cold runs allowed in flight")
    explore.add_argument("--allow-cold", action="store_true")
    explore.add_argument("--model", default="")
    explore.add_argument("--timeout", type=int, default=1800)
    explore.set_defaults(func=cmd_explore)

    camp = sub.add_parser("campaign", help="run the autonomous research campaign")
    camp.add_argument("--config", required=True, help="the campaign configuration (JSON)")
    camp.set_defaults(func=cmd_campaign)

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
    sweep.add_argument(
        "--offline", action="store_true", help="the scripted proposer, no model: the CI path"
    )
    sweep.add_argument(
        "--agent",
        default="opencode",
        choices=("opencode", "scripted"),
        help="the agent that edits the harness; opencode is the default (issue #102)",
    )
    sweep.add_argument("--model", default="", help="the model the agent uses")
    sweep.add_argument("--timeout", type=int, default=1800, help="the agent's timeout, seconds")
    sweep.add_argument(
        "--base-harness",
        default="",
        help="a directory whose surface is the base harness; the repository by default",
    )
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
