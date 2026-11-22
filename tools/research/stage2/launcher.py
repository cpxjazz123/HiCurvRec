"""The one way a Stage 2 candidate is started.

A candidate does not fork ``torchrun`` itself. It calls :func:`launch`, which
owns the port, the world size, the working directory and the log. That removes
three classes of failure at once: a candidate that hardcodes a port collides with
another job, a candidate that forks from a directory the framework later replaces
dies mid-run, and a candidate that logs next to its source writes outside the
result directory that everything else reads from.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from . import contract
from . import graph_batches
from .anticollapse import patch_tree


def launch(entry: Path | str, result_dir: Path | str, *, port: int | None = None,
           world_size: int = contract.WORLD_SIZE, env_extra: dict | None = None,
           python: str | None = None, log_name: str = "train_migrated.log") -> int:
    """Start the four ranks of one candidate and wait for them.

    The child processes run with the RESULT directory as their working directory
    and with the frozen port, world size and thread limits. The return value is
    the exit code of the launcher, so a caller can decide what to do with a
    failure without guessing.
    """
    entry = Path(entry).resolve()
    result_dir = Path(result_dir).resolve()
    if not entry.is_file():
        raise contract.ContractError("the entry point does not exist: " + str(entry))
    result_dir.mkdir(parents=True, exist_ok=True)
    logs = result_dir / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    # The floor is installed here, at the only permitted launch path, so no candidate can
    # reach training without it -- no cooperation from the mechanism's own code is needed.
    # Fails closed on a quantiser this repository does not know.
    installed = patch_tree.apply_floor(entry.parent)
    (result_dir / "assignment_floor.json").write_text(
        json.dumps(installed, indent=1, sort_keys=True), encoding="utf-8")
    # Issue #230: hand the trainer an epoch order that carries whole graphlet instances.
    # The framework owns batching (§66); a tree that does not read the file is unaffected.
    order = graph_batches.install(result_dir, universe=_universe(entry))
    if order is not None:
        (result_dir / "graph_batch_order.json").write_text(
            json.dumps({"path": str(order), "seed": graph_batches.DEFAULT_SEED}, indent=1),
            encoding="utf-8")
    interpreter = python or sys.executable
    torchrun = Path(interpreter).with_name("torchrun")
    if not torchrun.is_file():
        raise contract.ContractError("torchrun is not next to " + interpreter)
    command = [str(torchrun), "--standalone", "--nproc_per_node=" + str(int(world_size)),
               "--master_port=" + str(int(port if port is not None else contract.STAGE2_PORT)),
               str(entry)]
    if order is not None:
        env_extra = dict(env_extra or {})
        env_extra["GENREC_GRAPH_BATCH_ORDER"] = str(order)
    env = contract.launch_env(env_extra)
    log_path = logs / log_name
    with log_path.open("w", encoding="utf-8") as sink:
        completed = subprocess.run(command, cwd=str(result_dir), env=env,
                                   stdout=sink, stderr=subprocess.STDOUT)
    return int(completed.returncode)


def in_single_card_mode() -> bool:
    """Whether the MCP asked for the frozen single-card branch."""
    return os.environ.get(contract.SINGLE_GPU_ENV) == "1"


def is_child_rank() -> bool:
    """Whether this process is one of the ranks the launcher started."""
    return int(os.environ.get("RANK", -1)) >= 0


def _universe(entry: Path) -> int:
    """The number of items the trainer will iterate: its own embedding row count."""
    import numpy as np
    root = entry.parent
    for parent in entry.parents:
        if (parent / "stage1_GeneEmbedding").is_dir():
            root = parent
            break
    array = np.load(Path(root) / "stage1_GeneEmbedding" / "output" / "sentence_t5.npy",
                    mmap_mode="r")
    return int(array.shape[0])
