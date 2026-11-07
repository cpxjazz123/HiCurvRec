"""The process that starts the four ranks of one Stage 2 candidate.

The fork has to happen in a process that never imported torch. A candidate that
forks ``torchrun`` from its own trainer does it after the trainer imported torch,
and a fork from a process that holds torch's threads can deadlock before the
child reaches ``exec``: the ranks never start, the log stays empty, the GPUs
stay idle, and the job looks alive while nothing is training. That failure was
measured on this host -- three launches in a row, each with a zero-byte log and
no GPU use -- while the same tree trained normally whenever torchrun was the
parent process instead.

So the framework forks, from here, and the candidate's trainer only ever runs as
one of the ranks. The candidate's own ``_launch_via_torchrun`` stays in the tree
because the frozen contract asks for it, but it is reached only when a person
starts the trainer by hand, and even then it defers to the same launcher.

This module must stay free of torch, and it must stay in the repository: it is
the one process that has to be able to fork safely, so a tree that cannot find
it cannot be trained at all.

Usage:
    python3 -m tools.research.stage2.bootstrap --entry <trainer> \\
        --result-dir <dir> [--python <interpreter>] [--visible-devices 0,1,2,3]
"""
from __future__ import annotations

import argparse
from pathlib import Path

from . import contract
from . import launcher


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tools.research.stage2.bootstrap",
        description="Start the four ranks of one Stage 2 candidate.",
    )
    parser.add_argument("--entry", required=True,
                        help="the candidate's trainer, the file torchrun will run")
    parser.add_argument("--result-dir", required=True,
                        help="where the run writes; also the ranks' working directory")
    parser.add_argument("--python", default=contract.STAGE2_PYTHON,
                        help="the interpreter the ranks run under")
    parser.add_argument("--port", type=int, default=contract.STAGE2_PORT,
                        help="the frozen port of the four-rank job")
    parser.add_argument("--world-size", type=int, default=contract.WORLD_SIZE,
                        help="how many ranks the job has")
    parser.add_argument("--visible-devices", default="",
                        help="the physical cards the job may use")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    env_extra: dict[str, str] = {}
    if args.visible_devices:
        env_extra["CUDA_VISIBLE_DEVICES"] = args.visible_devices
    return launcher.launch(
        entry=Path(args.entry),
        result_dir=Path(args.result_dir),
        port=int(args.port),
        world_size=int(args.world_size),
        python=str(args.python),
        env_extra=env_extra,
    )


if __name__ == "__main__":
    raise SystemExit(main())
