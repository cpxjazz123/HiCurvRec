"""Command line for the MVG gates.

    python -m tools.mvg.cli run --tree stage2_RQ-VAE/TIGER_r9_mvg_reference \
        --spec tools/mvg/specs/masked_directed_codebook_margin.json --out /tmp/mvg

The gates run on CPU in the project's own Python 3.9 environment (they need torch); no GPU
is used and no training is started.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import runner
from . import spec as SPEC

REPO_ROOT = SPEC.REPO_ROOT


def _pin_threads() -> None:
    """One thread for the gates.

    The checks run on tensors a few elements wide. With the default thread pool every one
    of those tiny operations pays a fan-out, and the suite spends its time in the
    scheduler instead of in the arithmetic.
    """
    try:
        import torch

        torch.set_num_threads(1)
    except Exception:  # pragma: no cover - torch is present in every environment that runs this
        pass


def cmd_run(args: argparse.Namespace) -> int:
    tree = Path(args.tree)
    if not tree.is_absolute():
        tree = REPO_ROOT / tree
    spec_path = Path(args.spec)
    if not spec_path.is_absolute():
        spec_path = REPO_ROOT / spec_path
    out_dir = Path(args.out) if args.out else None

    report = runner.run(tree, spec_path, out_dir, skip_m5=args.skip_m5)

    print(f"mechanism : {report['mechanism']['id']} v{report['mechanism']['version']}")
    print(f"tree      : {report['mechanism']['tree']}")
    print(f"spec      : {report['mechanism']['spec_sha256'][:16]}")
    print(f"tree code : {report['mechanism']['tree_code_sha256'][:16]}")
    print(f"commit    : {report['mechanism']['commit'][:16]}")
    print()
    for gate in ("M0", "M1", "M2", "M3", "M4", "M5"):
        result = report["gates"][gate]
        print(f"  {gate}  {result['state']:<8} {result['detail']}")
        for problem in result["diagnostics"][:6]:
            print(f"        - {problem}")
    print()
    print(f"verdict: {report['verdict']}")
    if report.get("report_path"):
        print(f"report : {report['report_path']}")
    return 0 if report["verdict"] == "passed" else 1


def cmd_spec(args: argparse.Namespace) -> int:
    doc = SPEC.load_spec(Path(args.spec))
    print(f"mechanism : {doc['mechanism']['id']} v{doc['mechanism']['version']}")
    print(f"digest    : {SPEC.spec_digest(doc)}")
    print(f"cases     : {sum(len(v) for v in doc['cases'].values())}")
    print(f"forbidden : {len(doc['forbidden_substitutions'])}")
    print(f"invariants: {len(doc['invariants'])}")
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="tools.mvg.cli", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="run the gates over a mechanism tree")
    run.add_argument("--tree", required=True, help="the arm tree that hosts the mechanism")
    run.add_argument("--spec", required=True, help="the frozen MechanismSpec")
    run.add_argument("--out", default="", help="directory for mvg_report.json")
    run.add_argument("--skip-m5", action="store_true", help="skip the mutation suite")
    run.set_defaults(func=cmd_run)

    spec = sub.add_parser("spec", help="validate a specification and print its digest")
    spec.add_argument("--spec", required=True)
    spec.set_defaults(func=cmd_spec)
    return parser


def main(argv: list[str] | None = None) -> int:
    _pin_threads()
    args = _build_parser().parse_args(argv)
    try:
        return int(args.func(args))
    except (SPEC.SpecError, RuntimeError, ValueError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
