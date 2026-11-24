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


class LaunchRefused(contract.ContractError):
    """The host cannot satisfy the frozen launch contract; nothing was started."""


#: Owner rule (2026-11-22): Stage2 is always four-card DDP and always serial. The two properties
#: are enforced here, at the only launch path, so no candidate and no operator flag can pick a
#: different regime: a single-card Stage2 changes the optimisation (one rank against a four-rank
#: batch) and two concurrent Stage2 runs change what the GPU timings and the comparisons mean.
SERIAL_LOCK = Path(__file__).resolve().parents[3] / "results" / "research" / "stage2_serial.lock"


def required_world_size() -> int:
    return int(contract.WORLD_SIZE)


def assert_four_cards(visible: int, required: int | None = None) -> None:
    required = required_world_size() if required is None else int(required)
    visible = int(visible)
    if visible < required:
        raise LaunchRefused(
            "Stage2 needs %d visible cards for four-rank DDP and this host shows %d; "
            "a single-card run is not an accepted Stage2 regime" % (required, visible))


def _pid_alive(pid: int) -> bool:
    if int(pid) <= 0:
        return False
    try:
        os.kill(int(pid), 0)
    except OSError:
        return False
    return True


def acquire_serial(lock: Path, pid: int, alive=_pid_alive) -> Path:
    """Take the one Stage2 slot; refuse when a live run already holds it.

    A lock whose pid is gone is stale (a killed or crashed run) and is replaced rather than
    blocking the host forever.
    """
    lock = Path(lock)
    lock.parent.mkdir(parents=True, exist_ok=True)
    if lock.is_file():
        text = lock.read_text(encoding="utf-8").strip()
        try:
            holder = int(text.split()[0])
        except (ValueError, IndexError):
            holder = 0
        if holder and alive(holder):
            raise LaunchRefused(
                "a Stage2 run is already live (pid %d, lock %s); Stage2 is serial by owner rule"
                % (holder, lock))
    lock.write_text("%d\n" % int(pid), encoding="utf-8")
    return lock


def release_serial(lock: Path, pid: int) -> None:
    lock = Path(lock)
    if not lock.is_file():
        return
    text = lock.read_text(encoding="utf-8").strip()
    try:
        holder = int(text.split()[0])
    except (ValueError, IndexError):
        holder = -1
    if holder in (int(pid), -1):
        lock.unlink()


def prepare_floor(tree: Path, *, frozen: bool = False) -> dict:
    """Repair a live tree, but only verify an immutable admitted generation."""
    tree = Path(tree).resolve()
    if not frozen:
        return patch_tree.apply_floor(tree)
    for family, spec in patch_tree.FAMILIES.items():
        module = tree / spec.module
        if module.is_file():
            break
    else:
        raise LaunchRefused("immutable source has no known assignment-floor quantiser")
    text = module.read_text(encoding="utf-8")
    installed = tree / patch_tree.TARGET_NAME
    if (patch_tree._migrate(text) != text or not patch_tree._already_patched(text)
            or not installed.is_file() or installed.is_symlink()
            or installed.read_bytes() != patch_tree.FLOOR_MODULE.read_bytes()):
        raise LaunchRefused("immutable source requires an assignment-floor patch; re-admit live source")
    patch_tree.floor_constant(tree, family)
    if patch_tree.unguarded_sites(tree, family):
        raise LaunchRefused("immutable source has unguarded assignment sites")
    return {"family": family, "module": spec.module, "changed": False,
            "reason": "immutable assignment floor verified"}


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
    # Owner rule: four-card DDP, and one Stage2 at a time. Both are checked before anything is
    # written into the result directory, so a refused launch leaves no partial state.
    assert_four_cards(_visible_cards(env_extra))
    if int(world_size) != required_world_size():
        raise LaunchRefused(
            "Stage2 runs at world_size=%d; %d was requested" % (required_world_size(), int(world_size)))
    frozen = entry.parent.name == "src" and "_runs" in entry.parts
    if frozen:
        from tools.issue252.run_workspace import check_run_admission, load_run
        workspace = entry.parent.parent / "workspace.json"
        identity = json.loads(workspace.read_text(encoding="utf-8"))
        repo = contract.resolve_root(entry)
        run = load_run(repo, identity["run_id"], identity["arm"])
        if Path(run["entry"]) != entry or Path(run["result"]) != result_dir:
            raise LaunchRefused("immutable launcher paths cross run generations")
        check_run_admission(repo, run)
    installed = prepare_floor(entry.parent, frozen=frozen)
    # Diagnostic entries collect the credential; only the formal trainer consumes it.
    if not frozen and entry.name == "train_rqvae.py":
        from tools.issue252.training_admission import check_receipt, required
        if required(entry.parent):
            check_receipt(contract.resolve_root(entry), entry.parent)
    lock = acquire_serial(SERIAL_LOCK, os.getpid())
    result_dir.mkdir(parents=True, exist_ok=True)
    logs = result_dir / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    # Never repair the snapshot after its admission/source inventory was frozen.
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
    command = [str(torchrun), "--master_addr=127.0.0.1", "--nproc_per_node=" + str(int(world_size)),
               "--master_port=" + str(int(port if port is not None else contract.STAGE2_PORT)),
               str(entry)]
    if order is not None:
        env_extra = dict(env_extra or {})
        env_extra["GENREC_GRAPH_BATCH_ORDER"] = str(order)
    env = contract.launch_env(env_extra)
    log_path = logs / log_name
    try:
        if frozen:
            run = load_run(repo, identity["run_id"], identity["arm"])
            check_run_admission(repo, run)
            prepare_floor(entry.parent, frozen=True)
        elif entry.name == "train_rqvae.py" and required(entry.parent):
            check_receipt(contract.resolve_root(entry), entry.parent)
        with log_path.open("w", encoding="utf-8") as sink:
            completed = subprocess.run(command, cwd=str(result_dir), env=env,
                                       stdout=sink, stderr=subprocess.STDOUT)
    finally:
        release_serial(SERIAL_LOCK, os.getpid())
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


def _visible_cards(env_extra: dict | None) -> int:
    """How many cards this launch will actually see, honouring CUDA_VISIBLE_DEVICES."""
    override = (env_extra or {}).get("CUDA_VISIBLE_DEVICES", os.environ.get("CUDA_VISIBLE_DEVICES"))
    if override is not None and str(override).strip() != "":
        return len([part for part in str(override).split(",") if part.strip() != ""])
    try:
        import torch
        return int(torch.cuda.device_count())
    except Exception:  # noqa: BLE001 - a host without torch cannot give four cards
        return 0
