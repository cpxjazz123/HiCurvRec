"""The last step of the Stage 2 launch: turn a probed layout into one process.

`tools/genrec-mcp/server.py` owns the decision -- which tree, which card, which mode --
and then runs this module, which owns nothing but the exec.  It was deleted with the
`tools/research` cleanup in 9ba91c7b while the server still invoked it, so every formal
Stage 2 launch after that commit died with `No module named
tools.research.stage2.bootstrap` *after* a passing smoke, i.e. every candidate burned a
smoke and then started nothing.

The two layouts are the ones CLAUDE.md 19d0b describes:

* one card -- `python train_rqvae.py`, with the trainer's own single-card branch keeping the
  global batch at 4096;
* four cards -- `torchrun --standalone --nproc_per_node=4 --master_port=<probe>`.

The rendezvous port is passed in by the server (per launch, from the contract) instead of
being a constant here, so two arms can never collide on it.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def build_command(entry: Path, python: str, world_size: int, port: int) -> list[str]:
    """The argv for one layout. Pure, so the launch recipe can be asserted in a test."""
    if int(world_size) > 1:
        return [
            python, "-m", "torch.distributed.run",
            "--standalone",
            "--nproc_per_node=" + str(int(world_size)),
            "--master_port=" + str(int(port)),
            str(entry),
        ]
    return [python, str(entry)]


def build_env(base: dict | None, visible_devices: str | None) -> dict:
    env = dict(os.environ if base is None else base)
    if visible_devices:
        env["CUDA_VISIBLE_DEVICES"] = visible_devices
    return env


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="tools.research.stage2.bootstrap")
    parser.add_argument("--entry", required=True, help="the tree's train_rqvae.py")
    parser.add_argument("--result-dir", required=True, help="working directory of the run")
    parser.add_argument("--python", required=True, help="the tree's interpreter")
    parser.add_argument("--visible-devices", default="",
                        help="comma-separated card indices this run may see")
    parser.add_argument("--world-size", type=int, required=True)
    parser.add_argument("--port", type=int, required=True, help="rendezvous port")
    args = parser.parse_args(argv)

    entry = Path(args.entry).resolve()
    result_dir = Path(args.result_dir).resolve()
    if not entry.is_file():
        print(f"bootstrap: no entry at {entry}", file=sys.stderr)
        return 2
    result_dir.mkdir(parents=True, exist_ok=True)

    command = build_command(entry, args.python, args.world_size, args.port)
    env = build_env(None, args.visible_devices)
    print("bootstrap: " + " ".join(command), flush=True)
    print("bootstrap: cuda_visible_devices=" + str(env.get("CUDA_VISIBLE_DEVICES")), flush=True)
    done = subprocess.run(command, cwd=str(result_dir), env=env)
    return int(done.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
