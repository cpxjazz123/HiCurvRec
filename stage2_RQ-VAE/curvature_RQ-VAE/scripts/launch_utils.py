"""Launch one arm under torchrun without ending the calling process.

``train_rqvae._launch_via_torchrun`` and its Stage3 twin both finish with
``sys.exit(rc)``, which is right for the single-run production entry points they
were written for and wrong for an arm runner: the first arm to finish takes the
loop with it, and the remaining arms silently never run. Every arm runner here
therefore builds the same command itself and returns the exit code instead.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

NCCL_ENV = {
    "NCCL_IB_DISABLE": "1",
    "NCCL_P2P_DISABLE": "1",
    "NCCL_SHM_DISABLE": "1",
    "NCCL_TIMEOUT": "3600",
    "TORCH_NCCL_BLOCKING_WAIT": "1",
}


def launch_arm(trainer, extra_env: dict[str, str] | None = None) -> int:
    """Run one arm under torchrun, blocking until it finishes.

    The child re-executes the calling script, so the caller is responsible for
    putting whatever selects the arm into ``extra_env`` and for having the child
    branch configure itself from it.
    """
    launcher = trainer._LAUNCHER
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = launcher["visible_dev"]
    env.update(NCCL_ENV)
    if extra_env:
        env.update({key: str(value) for key, value in extra_env.items()})
    command = [
        launcher["torchrun"],
        "--standalone",
        f"--nproc_per_node={launcher['nproc']}",
        f"--master_port={launcher['master_port']}",
        launcher["script"],
    ]
    log_path = Path(launcher["log"])
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as handle:
        return subprocess.call(
            command, stdout=handle, stderr=subprocess.STDOUT, env=env
        )