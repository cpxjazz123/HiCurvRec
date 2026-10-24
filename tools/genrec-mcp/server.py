#!/usr/bin/env python3
"""GeneRec MCP Stage2/Stage3 process service (stdlib only).

Stage2 requires source-bound independent Agent-authored executable tests via
verification_id, followed by a real one-epoch trainer gradient/DDP smoke;
no historical admission path grants launch permission.
Stage3 retains its frozen evaluation protocol and existing status tools.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

PROTOCOL_VERSION = "2024-11-05"
SERVER_NAME = "genrec-mcp"
SERVER_VERSION = "1.0.0"

REPO = Path("/home/wlia0047/ar57/wenyu/GeneRec")
STAGE2_ROOT = REPO / "stage2_RQ-VAE"
RESULT2_ROOT = REPO / "results/stage2_RQ-VAE"
STAGE3_DIR = REPO / "stage3_T5Train"
RESULT3_ROOT = REPO / "results/stage3_T5Train"

PY2 = "/home/wlia0047/ar57_scratch/wenyu/genrec_env_v2/bin/python3.9"
PY3 = "/home/wlia0047/ar57_scratch/wenyu/genrec_env/bin/python3.10"

STAGE2_ARTIFACTS = ("rqvae_best.pth", "sids_raw.npy", "sids_for_hgrec.npy", "item_sids.json")
# Where a retrain's previous artifacts are copied before they are overwritten.
STAGE2_ARCHIVE_ROOT = Path("/home/wlia0047/ar57/wenyu/artifact_archive")
# What every Stage2 tree must leave behind for Stage3 to run; sids_raw.npy only
# exists for the trees that dump the raw SID tensor, so it is not part of this set.
STAGE2_REQUIRED_ARTIFACTS = ("rqvae_best.pth", "sids_for_hgrec.npy", "item_sids.json")
# One real epoch on the tree's own data with the ramps active: long enough to see
# every active loss term reach an optimizer step, short enough to fail fast.
STAGE2_SMOKE_TIMEOUT_S = 3600

STATE_DIR = Path(os.path.expanduser("~/.cache/genrec-mcp"))
# Generated per-launch Stage3 entrypoints live outside the repository: the run's
# SID path arrives as a tool argument, so it cannot be hardcoded in any repo file.
HARNESS_DIR = Path("/tmp/genrec-mcp")

# Per-user shard dumps are a protocol-side constant, not a caller switch: the arm
# this SID belongs to was evaluated with them on, and they do not change any
# metric (the reducer uses recall_sums/ndcg_sums/n_eval only).
STAGE3_PER_USER_DUMP = True

STAGE3_HEADER_KEYS = (
    "n_eval",
    "test_recall@5",
    "test_recall@10",
    "test_ndcg@5",
    "test_ndcg@10",
    "best_checkpoint",
)


# --------------------------------------------------------------------------
# process / filesystem helpers
# --------------------------------------------------------------------------
def proc_table() -> dict[int, str]:
    """pid -> cmdline for every readable process, without pgrep self-matching."""
    table: dict[int, str] = {}
    for entry in os.listdir("/proc"):
        if not entry.isdigit():
            continue
        pid = int(entry)
        try:
            with open(f"/proc/{pid}/cmdline", "rb") as handle:
                raw = handle.read()
        except OSError:
            continue
        if not raw:
            continue
        parts = [part for part in raw.decode("utf-8", "replace").split("\0") if part]
        if parts:
            table[pid] = " ".join(parts)
    return table


def stage_processes(script: Path) -> list[dict]:
    """Every live process whose command line contains this absolute script path.

    The path is matched in full, so a shell that merely mentions the name does
    not count, and our own pid is excluded.
    """
    needle = str(script)
    me = os.getpid()
    hits = []
    for pid, cmd in proc_table().items():
        if pid == me or needle not in cmd:
            continue
        if "/torchrun" in cmd:
            role = "torchrun"
        elif re.search(r"\bpython[0-9.]*\b\s+-u\s", cmd):
            role = "rank"
        else:
            role = "launcher"
        hits.append({"pid": pid, "role": role, "cmd": cmd})
    return sorted(hits, key=lambda hit: hit["pid"])





def pid_alive(pid: int) -> bool:
    try:
        os.kill(int(pid), 0)
    except (OSError, ValueError, TypeError):
        return False
    return True


def state_path(stage: str) -> Path:
    return STATE_DIR / f"{stage}.json"


def load_state(stage: str) -> dict:
    path = state_path(stage)
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def save_state(stage: str, payload: dict) -> None:
    """Write the state file atomically.

    A detached worker updates this file while stage2_status reads it, so a plain
    overwrite could hand a reader a half-written document.
    """
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    path = state_path(stage)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(temporary, path)

















def logical_env(cwd: Path) -> dict:
    """Child environment with PWD pinned to the directory the child runs in.

    /home/wlia0047/ar57 is a symlink to /fs04/ar57, so Path.cwd() resolves to
    the physical /fs04 path while $PWD keeps the logical /home path. Tools that
    derive expected paths from $PWD or that print them need the logical form.
    Pinning PWD is not a hyperparameter override: it only makes the child's view
    of its own directory match the hardcoded paths.
    """
    env = dict(os.environ)
    env["PWD"] = str(cwd)
    return env


# --------------------------------------------------------------------------
# GPU layout probe: how many cards a launched trainer can actually see
# --------------------------------------------------------------------------
STAGE2_NPROC = 4  # the multi-card layout every Stage2 tree is written for


def physical_gpu_indices() -> tuple[list[str], str]:
    """GPU indices present on this host, and where the list came from.

    ``nvidia-smi`` is the authoritative view; the ``/dev/nvidiaN`` device nodes
    are the fallback for hosts where the tool is not on PATH. Standard library
    only: the MCP server never imports torch.
    """
    try:
        done = subprocess.run(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if done.returncode == 0 and done.stdout.strip():
            indices = [line.strip() for line in done.stdout.splitlines() if line.strip()]
            if indices:
                return indices, "nvidia-smi"
    except (OSError, subprocess.SubprocessError):
        pass
    nodes = sorted(
        node.name[len("nvidia"):] for node in Path("/dev").glob("nvidia[0-9]*")
    )
    return nodes, "dev-nodes"


def probe_gpu_layout(required: int = STAGE2_NPROC) -> dict:
    """Count the CUDA devices a launched trainer sees, and pick the run mode.

    ``CUDA_VISIBLE_DEVICES`` is the contract CUDA itself follows, so it is what
    gets interpreted here: unset means every physical card, an empty value means
    none, and a list means exactly those entries (indexes that exist, or GPU
    UUIDs, which are taken at face value because they cannot be resolved to a
    node name cheaply). The mode is a launch fact, not a hyperparameter: the
    trainer's own batch scaling and launcher branch follow it.
    """
    present, source = physical_gpu_indices()
    csv = os.environ.get("CUDA_VISIBLE_DEVICES")
    if csv is None:
        requested = None
        visible = present
    elif csv.strip() == "":
        requested = []
        visible = []
    else:
        requested = [entry.strip() for entry in csv.split(",") if entry.strip()]
        known = set(present)
        visible = [entry for entry in requested if entry in known or entry.startswith("GPU-")]
    count = len(visible)
    return {
        "physical": present,
        "probe_source": source,
        "cuda_visible_devices": csv,
        "requested": requested,
        "visible": visible,
        "visible_count": count,
        "required": int(required),
        "mode": "ddp" if count >= required else ("single-gpu" if count else "none"),
    }


def stage2_child_env(cwd: Path, layout: dict) -> dict:
    """Child environment: ``logical_env`` plus the mode this host was probed as.

    Only the mode travels; ``CUDA_VISIBLE_DEVICES`` is left exactly as the caller
    set it, so an operator who pinned one specific card is not overridden.
    """
    env = logical_env(cwd)
    env["GENREC_STAGE2_MODE"] = str(layout.get("mode") or "none")
    env["GENREC_SINGLE_GPU"] = "1" if layout.get("mode") == "single-gpu" else "0"
    return env


# The single-card launch contract a trainer has to declare: it reads this variable
# and skips the fixed 4-rank torchrun launcher when it says 1.
SINGLE_GPU_MARKER = re.compile(r'SINGLE_GPU_ENV\s*=\s*["\']GENREC_SINGLE_GPU["\']')


def tree_supports_single_gpu(tree: Path) -> bool:
    """Whether this tree's own trainer honours the single-card launch contract.

    A tree without the marker keeps a launcher that unconditionally forks
    ``torchrun --nproc_per_node=4`` over ``CUDA_VISIBLE_DEVICES=0,1,2,3``, so
    promising it a single card would only produce a doomed launch; the caller has
    to be told instead.
    """
    entry = tree / "train_rqvae.py"
    if not entry.is_file():
        return False
    try:
        return bool(
            SINGLE_GPU_MARKER.search(
                entry.read_text(encoding="utf-8", errors="replace")
            )
        )
    except OSError:
        return False


def stage2_launch_decision(gpu_probe: dict) -> str | None:
    """``None`` -> dispatch, otherwise the refusal reason for this probed layout.

    Pure on purpose: the decision is a function of the probe and the tree's
    declared capability, so it can be exercised without launching anything.
    """
    mode = gpu_probe.get("mode")
    if mode == "none":
        return (
            "no CUDA device is visible to this server "
            f"(CUDA_VISIBLE_DEVICES={gpu_probe.get('cuda_visible_devices')!r}, "
            f"physical={gpu_probe.get('physical')}); both the 4-rank and the "
            "single-card layouts need at least one card"
        )
    if mode == "single-gpu" and not gpu_probe.get("entrypoint_supports_single_gpu"):
        return (
            f"{gpu_probe.get('visible_count')} CUDA device(s) visible, but this tree's "
            "trainer has no single-card launch path (it forks 4-rank torchrun); run it "
            "on a 4-card allocation, or add the single-card contract to the tree's "
            "trainer first"
        )
    return None


def stage3_supports_single_gpu() -> bool:
    """Whether the Stage3 trainer honours the single-card launch contract.

    Stage3 has one shared trainer instead of per-tree copies, so the capability is
    a property of that one file: the same marker the Stage2 trees carry, so a
    trainer that loses the branch is refused rather than promised a single card.
    """
    entry = STAGE3_DIR / "train_HG-Rec.py"
    if not entry.is_file():
        return False
    try:
        return bool(
            SINGLE_GPU_MARKER.search(
                entry.read_text(encoding="utf-8", errors="replace")
            )
        )
    except OSError:
        return False


def stage3_launch_decision(gpu_probe: dict) -> str | None:
    """``None`` -> dispatch, otherwise why Stage3 must not start on this layout.

    Both layouts are the trainer's own business: the fixed 4-rank fork when the
    host offers 4 cards, and the single-card branch (same trainer, no torchrun,
    smaller infer batch) when it does not. Only a host with no card at all, or a
    trainer that has lost the single-card contract, is refused.
    """
    mode = gpu_probe.get("mode")
    if mode == "none":
        return (
            "no CUDA device is visible to this server "
            f"(CUDA_VISIBLE_DEVICES={gpu_probe.get('cuda_visible_devices')!r}, "
            f"physical={gpu_probe.get('physical')}); both the 4-rank and the "
            "single-card layouts need at least one card"
        )
    if mode == "single-gpu" and not gpu_probe.get("entrypoint_supports_single_gpu"):
        return (
            f"{gpu_probe.get('visible_count')} CUDA device(s) visible, but the Stage3 "
            "trainer has no single-card launch path (it forks 4-rank torchrun); run it "
            "on a 4-card allocation, or add the single-card contract to the trainer first"
        )
    return None


def stage3_child_env(cwd: Path, layout: dict) -> dict:
    """Child environment: ``logical_env`` plus the layout this host was probed as.

    The trainer reads ``GENREC_SINGLE_GPU`` to skip the 4-rank fork. The device
    list is left exactly as the caller set it, so an operator who pinned one
    specific card is not overridden.
    """
    env = logical_env(cwd)
    env["GENREC_STAGE3_MODE"] = str(layout.get("mode") or "none")
    env["GENREC_SINGLE_GPU"] = "1" if layout.get("mode") == "single-gpu" else "0"
    return env


# --------------------------------------------------------------------------
# Stage2 tree resolution: which tree to run is a parameter, not a constant
# --------------------------------------------------------------------------
def stage2_trees() -> list[Path]:
    """Every Stage2 source tree: a directory under stage2_RQ-VAE/ holding a trainer."""
    if not STAGE2_ROOT.is_dir():
        return []
    return sorted(
        (
            item
            for item in STAGE2_ROOT.iterdir()
            if item.is_dir() and (item / "train_rqvae.py").is_file()
        ),
        key=lambda item: item.name,
    )


def resolve_stage2_tree(source_dir: str | None) -> tuple[Path | None, str | None]:
    """Allowlist a Stage2 tree; only directories directly under stage2_RQ-VAE/ qualify.

    The returned path keeps the logical /home form the repository hardcodes, not
    the /fs04 form that Path.resolve() produces, so every command this server
    reports matches what a human would run.
    """
    trees = stage2_trees()
    names = [tree.name for tree in trees]
    if source_dir:
        given = Path(source_dir)
        if not given.is_absolute():
            given = STAGE2_ROOT / given
        try:
            resolved = given.resolve()
        except OSError as exc:
            return None, f"cannot resolve {source_dir}: {exc}"
        for tree in trees:
            if tree.resolve() == resolved:
                return tree, None
        return None, f"source_dir must name one of {names}"
    remembered = load_state("stage2").get("tree")
    if remembered:
        remembered_name = Path(remembered).name
        for tree in trees:
            if tree.name == remembered_name:
                return tree, None
    if len(trees) == 1:
        return trees[0], None
    return None, f"source_dir is required when several trees exist; available: {names}"


def tree_entrypoint(tree: Path) -> tuple[Path | None, list[dict]]:
    """The runnable Stage2 entrypoint of a tree, plus the verdict for each candidate.

    A candidate is accepted only when it can actually train:
      * the tree's own train_rqvae.py, when it defines both ``__main__`` and
        ``_launch_via_torchrun`` (the pattern every runnable tree uses);
      * a scripts/run_stage2*.py that imports that trainer and does not carry an
        unwired-hook guard -- run_stage2_arm2.py, for example, raises
        "trainer integration is not wired" at runtime, so it must never be run.
    """
    verdicts: list[dict] = []
    trainer = tree / "train_rqvae.py"
    accepted: Path | None = None
    if trainer.is_file():
        text = trainer.read_text(encoding="utf-8", errors="replace")
        runnable = ('if __name__ == "__main__"' in text) and ("_launch_via_torchrun" in text)
        verdicts.append(
            {
                "candidate": str(trainer),
                "accepted": runnable,
                "reason": "has __main__ + _launch_via_torchrun" if runnable else "no runnable __main__",
            }
        )
        if runnable:
            accepted = trainer
    scripts = tree / "scripts"
    if scripts.is_dir():
        for path in sorted(scripts.glob("run_stage2*.py")):
            text = path.read_text(encoding="utf-8", errors="replace")
            unwired = ("not wired" in text) or re.search(r'getattr\(\s*train_rqvae\s*,\s*"run_', text) is not None
            imports_trainer = ("import train_rqvae" in text) or ("spec_from_file_location" in text)
            ok = imports_trainer and not unwired
            verdicts.append(
                {
                    "candidate": str(path),
                    "accepted": ok,
                    "reason": (
                        "unwired trainer hook -- must not be run"
                        if unwired
                        else ("launcher imports the tree trainer" if ok else "does not import the tree trainer")
                    ),
                }
            )
            if ok and accepted is None:
                accepted = path
    return accepted, verdicts


def tree_python(tree: Path) -> str:
    """Interpreter the tree itself launches with, else the default Stage2 env."""
    sources = [tree / "train_rqvae.py"]
    scripts = tree / "scripts"
    if scripts.is_dir():
        sources.extend(sorted(scripts.glob("*.py")))
    for path in sources:
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        found = re.findall(r"(/[^\s\"']*genrec_env[^\s\"']*)/bin/(?:torchrun|python[0-9.]*)", text)
        if found:
            binary_dir = Path(found[0]) / "bin"
            for name in ("python3.9", "python3.10", "python3"):
                if (binary_dir / name).is_file():
                    return str(binary_dir / name)
    return PY2


def tree_arm(tree: Path) -> str | None:
    """The arm this tree would train, read from its hardcoded EXPERIMENT_ARM switch."""
    try:
        text = (tree / "train_rqvae.py").read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    found = re.search(r'^EXPERIMENT_ARM\s*=\s*"([^"]+)"', text, re.M)
    return found.group(1) if found else None


def tree_result_dir(tree: Path) -> Path:
    """Where this tree's artifacts land.

    The trainer is the authority: a tree may declare RESULT_DIR_NAME because its arm name feeds
    the anti-collapse and geometry configuration and therefore cannot be changed to move output.
    Reading that declaration keeps the server from archiving and probing a directory the trainer
    never writes to.
    """
    try:
        text = (tree / "train_rqvae.py").read_text(encoding="utf-8", errors="replace")
        found = re.search(r'^RESULT_DIR_NAME\s*=\s*"([^"]+)"', text, re.M)
        if found:
            return RESULT2_ROOT / found.group(1)
    except OSError:
        pass
    base = RESULT2_ROOT / tree.name
    arm = tree_arm(tree)
    return base / arm if arm else base


def tree_log_paths(tree: Path) -> dict:
    """This tree's logs, arm-aware.

    Every tree that has arms keeps them under ``logs/<EXPERIMENT_ARM>/`` (the
    trainer sets ``LOG_DIR`` from the arm), so the arm directory is searched
    first and the tree-level ``logs/`` is only a fallback. Only non-empty files
    count: an empty stub must not shadow a populated log from a sibling arm.
    """
    logs = tree / "logs"
    arm = tree_arm(tree)
    search_dirs = []
    if arm:
        arm_dir = logs / arm
        if arm_dir.is_dir():
            search_dirs.append(arm_dir)
    if logs.is_dir():
        search_dirs.append(logs)

    def newest_non_empty(directory: Path, pattern: str) -> Path | None:
        candidates = [
            path
            for path in directory.glob(pattern)
            if path.is_file() and path.stat().st_size > 0
        ]
        if not candidates:
            return None
        return max(candidates, key=lambda path: path.stat().st_mtime)

    native = metrics = None
    for directory in search_dirs:
        native = newest_non_empty(directory, "*.log")
        if native:
            break
    for directory in search_dirs:
        metrics = newest_non_empty(directory, "*.jsonl")
        if metrics:
            break
    canonical = logs / arm if arm else logs
    run_log_dir = logs / arm if (arm and (logs / arm).is_dir()) else logs
    return {
        "run_log": str(run_log_dir / "train_run.log"),
        # A tree that has never run has no log files yet, so report where they will
        # be written instead of None: a None here breaks Path() in the status tool,
        # which is exactly the state of a freshly copied arm.
        "native_log": str(native) if native else str(canonical / "train_migrated.log"),
        "metrics": str(metrics) if metrics else str(canonical / "training_metrics.jsonl"),
    }


# --------------------------------------------------------------------------
# Stage3: launcher resolution from a Stage2 SID path
# --------------------------------------------------------------------------









def frozen_protocol_check() -> dict:
    """Manual replacement for the deleted stage3_protocol_gate.py."""
    trainer = STAGE3_DIR / "train_HG-Rec.py"
    if not trainer.is_file():
        return {"trainer": str(trainer), "ok": False, "error": "trainer missing"}
    text = trainer.read_text(encoding="utf-8", errors="replace")
    expected = {"NUM_EPOCHS": "150", "NO_EVAL": "True", "SKIP_TEST": "False", "SEED": "42", "BEAM_SIZE": "20"}
    found = {}
    for key in expected:
        match = re.search(rf"^{key}\s*=\s*(\S+)", text, re.M)
        found[key] = match.group(1).rstrip(",").strip() if match else None
    mismatched = {key: found[key] for key, want in expected.items() if found[key] != want}
    early_stop = bool(re.search(r"(?i)early[ _]?stop", text))
    return {
        "trainer": str(trainer),
        "expected": expected,
        "found": found,
        "mismatched": mismatched,
        "early_stop_present": early_stop,
        "ok": not mismatched and not early_stop,
    }


def derive_stage3_variant(sid: Path) -> str:
    """Variant directory name derived from the SID's own location, never hardcoded.

    ``results/stage2_RQ-VAE/<tree>/<arm>/item_sids.json`` becomes
    ``<tree>__<arm>`` with every character outside ``[A-Za-z0-9._-]`` replaced by
    ``_``. Both the logical (/home) and the physical (/fs04) forms of the roots
    are tried, because Path.resolve() returns the physical path while callers
    usually pass the logical one.
    """
    logical = Path(sid)
    relative = None
    for candidate, root in (
        (logical.parent, RESULT2_ROOT),
        (logical.resolve().parent, RESULT2_ROOT.resolve()),
    ):
        try:
            relative = candidate.relative_to(root)
            break
        except (ValueError, OSError):
            continue
    if relative is None:
        relative = Path(logical.parent.name)
    raw = "__".join(part for part in relative.parts if part) or logical.parent.name
    return re.sub(r"[^A-Za-z0-9._-]+", "_", raw)


HARNESS_TEMPLATE = '''"""Generated by genrec-mcp for one Stage3 run.

The Stage2 SID path and the variant name below are the values passed to the
stage3_start tool call; nothing here is hardcoded in the repository, and the
trainer itself still receives no command-line arguments.
"""
import importlib.util
import os
import sys

SID = {sid!r}
VARIANT = {variant!r}
STAGE3_DIR = {stage3_dir!r}
RESULT_ROOT = {result_root!r}
PER_USER_DUMP = {per_user_dump!r}

# The harness lives outside the repository, so its own directory would be
# sys.path[0] and the trainer's script-relative imports (``data.*``) would fail.
# cwd is set by the server but cwd is not an import root.
if STAGE3_DIR not in sys.path:
    sys.path.insert(0, STAGE3_DIR)

_spec = importlib.util.spec_from_file_location(
    "genrec_stage3_param", os.path.join(STAGE3_DIR, "train_HG-Rec.py")
)
if _spec is None or _spec.loader is None:
    raise RuntimeError("could not load the Stage3 trainer")
trainer = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(trainer)
trainer.CODE_PATH = SID
trainer.RQVAE_VARIANT = VARIANT
trainer.LOG_PATH = os.path.join(RESULT_ROOT, VARIANT, "logs") + os.sep
trainer.SAVE_PATH = os.path.join(RESULT_ROOT, VARIANT, "ckpt") + os.sep
trainer.PER_USER_DUMP = PER_USER_DUMP
trainer._LAUNCHER["script"] = os.path.abspath(__file__)
trainer._LAUNCHER["log"] = os.path.join(trainer.LOG_PATH, "_stage3_launcher.log")
os.environ.update(
    {{
        "NCCL_IB_DISABLE": "1",
        "NCCL_P2P_DISABLE": "1",
        "NCCL_SHM_DISABLE": "1",
        "NCCL_TIMEOUT": "3600",
        "TORCH_NCCL_BLOCKING_WAIT": "1",
    }}
)
if "RANK" in os.environ or os.environ.get(trainer.LAUNCHED_ENV):
    # RANK set: a torchrun rank. LAUNCHED_ENV set: the trainer's single-card branch
    # already re-executed this harness once, so it must run main() rather than
    # launch again (it has no RANK to stop the recursion).
    trainer.main()
else:
    trainer._launch_via_torchrun()
'''


def write_stage3_harness(sid: Path, variant: str, per_user_dump: bool) -> Path:
    """Write the per-launch Stage3 entrypoint outside the repository.

    The file name starts with ``run_stage3_`` so the duplicate-run guard (which
    matches that prefix) also sees processes started from these harnesses.
    """
    HARNESS_DIR.mkdir(parents=True, exist_ok=True)
    target = HARNESS_DIR / f"run_stage3_param_{re.sub(r'[^A-Za-z0-9._-]+', '_', variant)}.py"
    target.write_text(
        HARNESS_TEMPLATE.format(
            sid=str(sid),
            variant=variant,
            stage3_dir=str(STAGE3_DIR),
            result_root=str(RESULT3_ROOT),
            per_user_dump=bool(per_user_dump),
        ),
        encoding="utf-8",
    )
    return target


def stage3_processes() -> list[dict]:
    """Any live Stage3 process, whichever launcher started it.

    Matched on the interpreter plus the launcher/trainer name only: the launcher
    is often started with a relative path, so requiring the absolute directory in
    the command line silently misses a live run (observed: 22 live processes
    reported as zero).
    """
    me = os.getpid()
    hits = []
    for pid, cmd in proc_table().items():
        if pid == me:
            continue
        if not re.search(r"\bpython[0-9.]*", cmd):
            continue
        if "grep" in cmd or "pgrep" in cmd:
            continue
        if "run_stage3_" not in cmd and "train_HG-Rec.py" not in cmd:
            continue
        if "/torchrun" in cmd:
            role = "torchrun"
        elif re.search(r"\bpython[0-9.]*\b\s+-u\s", cmd):
            role = "rank"
        else:
            role = "launcher"
        hits.append({"pid": pid, "role": role, "cmd": cmd})
    return sorted(hits, key=lambda hit: hit["pid"])
























# --------------------------------------------------------------------------
# tools
# --------------------------------------------------------------------------
STAGE2_SMOKE_TEMPLATE = '''"""One-epoch gradient smoke for one Stage2 tree, run by genrec-mcp.

Runs the tree's own trainer for a single epoch on its real data with the
behaviour ramps activated and every output redirected out of the repository.
It records, for every plug-in loss the trainer calls, the loss value and the
gradient that reaches that call's tensor arguments: a term that reports a
non-zero loss while contributing no gradient is a dead term, which is the
failure this smoke exists to catch.
"""
import json
import sys
import os
import math
import types
from pathlib import Path

import torch

TREE = Path(sys.argv[1])
SCRATCH = Path(sys.argv[2])
sys.path.insert(0, str(TREE))
import train_rqvae as T  # noqa: E402

STDLIB = {"json", "sys", "os", "re", "time", "math", "random", "shutil", "threading",
          "subprocess", "copy", "argparse", "dataclasses", "pathlib", "typing",
          "collections", "itertools", "hashlib", "traceback", "warnings", "logging"}

# Smoke-only budget: one epoch with the ramps already active, so the arm's term
# is actually in the objective. Production constants stay untouched on disk.
if hasattr(T, "EPOCHS"):
    T.EPOCHS = 1
for _name in list(vars(T)):
    if _name.startswith("RAMP_START") or _name == "WARMUP_EPOCHS":
        setattr(T, _name, 0)

# Nothing this run writes may land in the repository or in the arm's result dir.
SCRATCH.mkdir(parents=True, exist_ok=True)
for _name, _value in (
    ("_OUT", SCRATCH),
    ("OUTPUT_SID", SCRATCH / "sids_for_hgrec.npy"),
    ("OUTPUT_JSON", SCRATCH / "item_sids.json"),
    ("CHECKPOINT", SCRATCH / "rqvae_best.pth"),
    ("LOG_DIR", SCRATCH),
    ("METRICS_PATH", SCRATCH / "training_metrics.jsonl"),
):
    if hasattr(T, _name):
        setattr(T, _name, _value)

probes = {}
applied_norms = []


def _probe(key, out, args):
    tensor = out[0] if isinstance(out, tuple) and torch.is_tensor(out[0]) else (out if torch.is_tensor(out) else None)
    if tensor is None or not tensor.requires_grad or key in probes:
        return
    best = 0.0
    # ``torch.autograd.grad`` refuses a non-scalar output ("grad can be implicitly
    # created only for scalar outputs"), and that RuntimeError was being swallowed
    # into grad=0.0, which ``dead_terms`` then read as a dead term. Differentiating the
    # sum asks the same question -- is this output wired into the graph -- and for a
    # one-element tensor the sum is the tensor, so scalar terms probe exactly as before.
    surrogate = tensor.sum()
    for arg in args:
        if not torch.is_tensor(arg) or not arg.requires_grad:
            continue
        try:
            grad = torch.autograd.grad(surrogate, arg, retain_graph=True, allow_unused=True)[0]
        except RuntimeError:
            continue
        if grad is not None:
            best = max(best, float(grad.abs().sum()))
    magnitude = tensor.detach().abs()
    if magnitude.numel() == 0:
        return
    # ``_wrap_public`` wraps every public callable the tree defines, so this also sees
    # helpers that hand back a tensor rather than a reduced loss -- residuals, a
    # projection, per-sample terms. Those carry no scalar loss, and ``float()`` on them
    # raised "only one element tensors can be converted to Python scalars", killing
    # the whole smoke. The gradient is still the thing worth asserting, so reduce the
    # magnitude to a number and say plainly that it was not a scalar term.
    record = {"loss": float(magnitude) if magnitude.numel() == 1 else float(magnitude.mean()), "grad": best}
    if magnitude.numel() != 1:
        record["reduction"] = "mean_abs"
        record["shape"] = list(magnitude.shape)
    probes[key] = record


def _wrap(fn, key):
    def inner(*args, **kwargs):
        out = fn(*args, **kwargs)
        _probe(key, out, args)
        return out
    return inner


TREE_ROOT = TREE.resolve()


def _module_inside_tree(module):
    """True when a module object's file lives inside this Stage2 tree."""
    path = getattr(module, "__file__", None)
    if not path:
        return False
    try:
        return TREE_ROOT in Path(path).resolve().parents
    except OSError:
        return False


def _fn_inside_tree(fn):
    """True when a function is defined by a file inside this Stage2 tree."""
    module = getattr(fn, "__module__", None)
    if not module:
        return False
    return _module_inside_tree(sys.modules.get(module))


def _wrap_public(obj, module_name):
    """Wrap the tree's own public callables so each loss term is probed."""
    wrapped = 0
    for name in list(vars(obj)):
        if name.startswith("_"):
            continue
        fn = getattr(obj, name)
        if not callable(fn) or not hasattr(fn, "__code__"):
            continue
        if not _fn_inside_tree(fn):
            continue
        try:
            setattr(obj, name, _wrap(fn, f"{module_name}.{name}"))
            wrapped += 1
        except (AttributeError, TypeError):
            pass
    return wrapped


# The trainer's own helpers ...
_wrapped = _wrap_public(T, T.__name__)
# ... and the tree's own modules it holds, including private ones such as
# ``_l2cond``: the arm's term lives behind that reference and must be probed.
for _attr, _value in list(vars(T).items()):
    if not isinstance(_value, types.ModuleType) or _value.__name__ == T.__name__:
        continue
    if not _module_inside_tree(_value):
        continue
    _wrapped += _wrap_public(_value, _value.__name__)
summary_wrapped = _wrapped

# A candidate whose mechanism lives in the module geometry rather than in an
# auxiliary loss term has no wrapped term to probe, and the smoke used to fail
# with an empty ``probed_terms`` for exactly that shape of tree. Probe the tree's
# OWN base losses against its OWN parameters instead: the total loss and the
# reconstruction term must both carry gradient into the model, or the geometry is
# learning nothing. This is measured on the real training batch, with the real
# trainer call, and it never weakens the auxiliary-term probe above.
_model_cls = getattr(T, "RQVAE", None)
if isinstance(_model_cls, type) and hasattr(_model_cls, "compute_loss"):
    _original_compute_loss = _model_cls.compute_loss

    def _probed_compute_loss(self, *args, **kwargs):
        total, recon = _original_compute_loss(self, *args, **kwargs)
        params = [p for p in self.parameters() if p.requires_grad]
        for _key, _term in (("RQVAE.compute_loss.total", total),
                            ("RQVAE.compute_loss.recon", recon)):
            if _key in probes or not torch.is_tensor(_term) or not _term.requires_grad:
                continue
            try:
                _grads = torch.autograd.grad(_term.sum(), params, retain_graph=True,
                                             allow_unused=True)
            except RuntimeError:
                continue
            _best = max((float(g.abs().sum()) for g in _grads if g is not None), default=0.0)
            probes[_key] = {"loss": float(_term.detach()) if _term.numel() == 1
                            else float(_term.detach().abs().mean()), "grad": _best,
                            "source": "base_loss_parameter_gradient"}
        return total, recon

    _model_cls.compute_loss = _probed_compute_loss

# A candidate may return its loss from forward() instead of exposing
# compute_loss. The same base-loss question then has to be asked of the forward
# tuple: the total loss must carry gradient into the model parameters.
if isinstance(_model_cls, type) and not hasattr(_model_cls, "compute_loss"):
    _original_forward = _model_cls.forward

    def _probed_forward(self, *args, **kwargs):
        out = _original_forward(self, *args, **kwargs)
        total = out[0] if isinstance(out, tuple) and out else out
        key = "RQVAE.forward.total"
        if key not in probes and torch.is_tensor(total) and total.requires_grad:
            params = [p for p in self.parameters() if p.requires_grad]
            try:
                grads = torch.autograd.grad(total.sum(), params, retain_graph=True,
                                            allow_unused=True)
            except RuntimeError:
                return out
            best = max((float(g.abs().sum()) for g in grads if g is not None), default=0.0)
            probes[key] = {"loss": float(total.detach()) if total.numel() == 1
                           else float(total.detach().abs().mean()), "grad": best,
                           "source": "forward_loss_parameter_gradient"}
        return out

    _model_cls.forward = _probed_forward

_original_clip = torch.nn.utils.clip_grad_norm_


def _clip(params, max_norm, *args, **kwargs):
    total = _original_clip(params, max_norm, *args, **kwargs)
    applied_norms.append(float(total))
    return total


torch.nn.utils.clip_grad_norm_ = _clip


def _crash(exc_type, exc, tb):
    """Never lose a crash: report it in the machine-readable summary."""
    import traceback as _traceback

    payload = {
        "passed": False,
        "crashed": True,
        "error": f"{exc_type.__name__}: {exc}",
        "traceback": "".join(_traceback.format_exception(exc_type, exc, tb))[-4000:],
        "probed_terms": dict(globals().get("probes", {})),
        "applied_grad_norms": list(globals().get("applied_norms", []))[:8],
    }
    print("SMOKE_SUMMARY " + json.dumps(payload), flush=True)
    print("SMOKE_FAIL", flush=True)
    sys.exit(1)


sys.excepthook = _crash

T.main()

# Phase two, the check the rules ask for by name: a real checkpoint and a real
# batch, so the terms are probed against trained weights rather than only against
# a cold start. This does not touch the production run, which stays cold start.
checkpoint_probe = {}
checkpoint_arg = Path(sys.argv[3]) if len(sys.argv) > 3 else None
if checkpoint_arg is None or not checkpoint_arg.is_file():
    # A tree that has never trained has no checkpoint to probe, and that is not a failure: there is
    # nothing to check. A checkpoint that EXISTS but cannot be loaded is still reported as
    # missing, and still fails closed below.
    if checkpoint_arg is None:
        checkpoint_probe = {
            "absent": True,
            "note": "this tree has no checkpoint yet; there is nothing to probe",
        }
    else:
        checkpoint_probe = {
            "error": f"checkpoint named but unreadable: {checkpoint_arg}",
            "missing": True,
        }
else:
    import numpy as np
    import pandas as pd
    from torch.utils.data import DataLoader, TensorDataset

    from model import RQVAE as _RQVAE

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    config = T._tokenizer_config()
    embedding = T.maybe_apply_pca(T.load_embeddings(T.EMBEDDING_FILE), T.PCA_DIM, T.SEED)
    frame = pd.read_parquet(T.TRAIN_FILE)
    item_ids = np.unique(frame["target"].to_numpy(dtype=np.int64))
    rows = torch.from_numpy(embedding)[torch.from_numpy(item_ids)]
    loader = DataLoader(
        TensorDataset(rows), batch_size=T.BATCH_SIZE_PER_RANK, shuffle=True, num_workers=T.NUM_WORKERS
    )
    probe_model = _RQVAE(config, in_dim=embedding.shape[1]).to(device)
    payload = torch.load(checkpoint_arg, map_location=device)
    probe_model.load_state_dict(payload.get("state_dict", payload), strict=True)
    probe_model.train()
    batch = next(iter(loader))[0].to(device, non_blocking=True)
    reconstructed, quant_loss, _unused_codes, _tokens = probe_model(batch)
    total, recon_only = probe_model.compute_loss(batch, reconstructed, quant_loss)
    parameters = [p for p in probe_model.parameters() if p.requires_grad]

    # The batch carries no gradient, so the terms are differentiated against the
    # parameters: that is where the trained weights actually have to respond.
    def _param_grad_norm(term):
        grads = torch.autograd.grad(term, parameters, retain_graph=True, allow_unused=True)
        return max([float(g.abs().sum()) for g in grads if g is not None] or [0.0])

    checkpoint_probe = {
        "checkpoint": str(checkpoint_arg),
        "batch_shape": list(batch.shape),
        "total_loss": float(total.detach()),
        "recon_loss": float(recon_only.detach()),
        "total_param_grad_norm_max": _param_grad_norm(total),
        "recon_param_grad_norm_max": _param_grad_norm(recon_only),
    }
    del probe_model

dead_terms = sorted(k for k, v in probes.items() if v["loss"] > 1e-12 and v["grad"] <= 0.0)
nonfinite = sorted(k for k, v in probes.items() if v["grad"] != v["grad"] or v["loss"] != v["loss"])
summary = {
    "arm": getattr(T, "EXPERIMENT_ARM", None),
    "launch_mode": os.environ.get("GENREC_STAGE2_MODE"),
    "steps": len(applied_norms),
    "applied_grad_norms": applied_norms[:8],
    "probed_terms": probes,
    "checkpoint_probe": checkpoint_probe,
    "dead_terms": dead_terms,
    "nonfinite_terms": nonfinite,
    "all_step_norms_positive": bool(applied_norms) and all(g > 0 for g in applied_norms),
    "scratch_files": sorted(p.name for p in SCRATCH.iterdir()),
}
checkpoint_ok = (
    # A brand-new tree has no checkpoint anywhere, and that is not a defect: there is nothing to
    # probe. The probe stays MANDATORY whenever a checkpoint does exist, so an arm resuming from
    # one can never skip it.
    (checkpoint_probe.get("absent", False) is True)
    or (
        not checkpoint_probe.get("missing", False)
        and all(
            math.isfinite(float(checkpoint_probe.get(key, float("nan"))))
            for key in (
                "total_loss",
                "recon_loss",
                "total_param_grad_norm_max",
                "recon_param_grad_norm_max",
            )
        )
        and checkpoint_probe.get("total_param_grad_norm_max", 0.0) > 0.0
        and checkpoint_probe.get("recon_param_grad_norm_max", 0.0) > 0.0
    )
)
passed = (
    summary["steps"] > 0
    and summary["all_step_norms_positive"]
    and not dead_terms
    and not nonfinite
    and any(v["grad"] > 0 for v in probes.values())
    and checkpoint_ok
)
summary["passed"] = bool(passed)
print("SMOKE_SUMMARY " + json.dumps(summary))
print("SMOKE_PASS" if passed else "SMOKE_FAIL")
sys.exit(0 if passed else 1)
'''


def state_process_live(pid: int | None, script: Path) -> bool:
    """True only when the recorded pid is still this tree's own trainer.

    A bare pid check is unsound: the pid from an old run can have been recycled by
    an unrelated process, which would make the server refuse a fresh launch with
    a live-run guard while nothing of ours is running.
    """
    if not pid:
        return False
    try:
        with open(f"/proc/{int(pid)}/cmdline", "rb") as handle:
            cmdline = handle.read().decode("utf-8", "replace")
    except (OSError, ValueError):
        return False
    return str(script) in cmdline


def run_subprocess(cmd: list[str], cwd: Path, timeout: int, env: dict | None = None) -> dict:
    """Run a child process to completion, capturing its output for inspection."""
    try:
        done = subprocess.run(
            cmd, cwd=str(cwd), capture_output=True, text=True, timeout=timeout, env=env
        )
        return {"cmd": " ".join(cmd), "rc": done.returncode,
                "stdout": done.stdout, "stderr": done.stderr[-2000:]}
    except subprocess.TimeoutExpired:
        return {"cmd": " ".join(cmd), "rc": None, "stdout": "", "stderr": "timeout"}
    except OSError as exc:
        return {"cmd": " ".join(cmd), "rc": None, "stdout": "", "stderr": str(exc)}


def probe_checkpoint(tree: Path, result_dir: Path) -> tuple[Path | None, str]:
    """The trained weights the smoke probe needs.

    The arm's own checkpoint is preferred. After an interrupted run the result
    directory is empty (the previous worker removed those artifacts once its
    smoke passed), so fall back to the newest archived copy instead of leaving the
    probe unsatisfiable and every restart refused.
    """
    live = result_dir / "rqvae_best.pth"
    if live.is_file():
        return live, "result_dir"
    archived = sorted(
        STAGE2_ARCHIVE_ROOT.glob(f"{tree.name}_*/rqvae_best.pth"),
        key=lambda item: item.parent.name,
    )
    if archived:
        return archived[-1], f"archive:{archived[-1].parent.name}"
    return None, "none"


def run_stage2_smoke(
    tree: Path,
    python: str,
    timeout: int,
    checkpoint: Path | None = None,
    log_path: Path | None = None,
    env: dict | None = None,
) -> dict:
    """Run the one-epoch gradient smoke for a Stage2 tree; never writes into the repo.

    With ``log_path`` the child's stdout and stderr stream into that file while it
    runs, so a monitor can watch the smoke instead of waiting blind for its verdict.
    ``env`` carries the probed run mode, so the smoke exercises the same single-GPU
    or 4-rank layout the real run will use.
    """
    HARNESS_DIR.mkdir(parents=True, exist_ok=True)
    scratch = HARNESS_DIR / f"smoke_{re.sub(r'[^A-Za-z0-9._-]+', '_', tree.name)}"
    driver = HARNESS_DIR / f"smoke_driver_{re.sub(r'[^A-Za-z0-9._-]+', '_', tree.name)}.py"
    driver.write_text(STAGE2_SMOKE_TEMPLATE, encoding="utf-8")
    argv = [python, str(driver), str(tree), str(scratch)]
    if checkpoint is not None:
        argv.append(str(checkpoint))
    if log_path is None:
        result = run_subprocess(argv, cwd=tree, timeout=timeout, env=env)
    else:
        # Stream, because a captive pipe shows nothing until the smoke ends. The
        # smoke merges stderr into stdout so one file carries the whole story.
        log_path.parent.mkdir(parents=True, exist_ok=True)
        # -u so the smoke's progress reaches the log while it runs, not at exit.
        argv.insert(1, "-u")
        with log_path.open("wb") as sink:
            child = subprocess.Popen(
                argv,
                cwd=str(tree),
                stdout=sink,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                start_new_session=True,
                env=env,
            )
            try:
                rc = child.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                # The smoke forks its own torchrun ranks, so killing only the
                # leader would leave them running: take the whole group down.
                try:
                    os.killpg(os.getpgid(child.pid), signal.SIGKILL)
                except (OSError, ProcessLookupError):
                    child.kill()
                child.wait()
                rc = None
        text = log_path.read_text(encoding="utf-8", errors="replace") if log_path.is_file() else ""
        result = {"cmd": " ".join(argv), "rc": rc, "stdout": text, "stderr": text[-2000:]}
    summary = None
    for line in result["stdout"].splitlines():
        if line.startswith("SMOKE_SUMMARY "):
            try:
                summary = json.loads(line[len("SMOKE_SUMMARY "):])
            except ValueError:
                summary = None
    # The smoke's weights and SIDs are throwaway: keep only the summary.
    shutil.rmtree(scratch, ignore_errors=True)
    driver.unlink(missing_ok=True)
    return {
        "passed": result["rc"] == 0 and bool(summary and summary.get("passed")),
        "summary": summary,
        "rc": result["rc"],
        "stderr_tail": result["stderr"][-800:],
        "scratch": str(scratch),
        "log": str(log_path) if log_path is not None else None,
    }


def tool_stage2_start(source_dir: str, verification_id: str) -> dict:
    """Launch only after independent Agent tests and real trainer smoke.

    Historical admission records and direct trainer calls do not grant access.
    No agent or model can approve training merely by writing a verdict string.
    """
    tree, problem = resolve_stage2_tree(source_dir)
    if tree is None:
        return {
            "started": False,
            "refused": problem,
            "available_trees": [item.name for item in stage2_trees()],
        }
    entry, verdicts = tree_entrypoint(tree)
    python = tree_python(tree)
    arm = tree_arm(tree)
    result_dir = tree_result_dir(tree)
    logs = tree_log_paths(tree)
    # Probe the host before anything else: a Stage2 run is either the 4-rank DDP
    # layout or the single-card layout, and which one applies is a fact about this
    # machine, not a per-tree choice. Zero visible cards means no layout works, and
    # saying so up front beats a crash inside the trainer.
    gpu_probe = probe_gpu_layout()
    # Only the source tree's trainer carries the single-card launch contract so far;
    # the other trees keep a launcher that unconditionally forks 4 ranks, so a
    # single-card promise has to be refused rather than attempted.
    gpu_probe["entrypoint_supports_single_gpu"] = tree_supports_single_gpu(tree)
    context = {
        "tree": str(tree),
        "arm": arm,
        "result_dir": str(result_dir),
        "logs": logs,
        "entrypoint": str(entry) if entry else None,
        "entrypoint_verdicts": verdicts,
        "gpu_probe": gpu_probe,
    }
    if entry is None:
        return {
            "started": False,
            "refused": "no runnable Stage2 entrypoint in this tree",
            **context,
        }
    # Deterministic test gate runs BEFORE any archive or dispatch side effect.
    verification = run_agent_verification(tree, verification_id)
    context["agent_verification"] = verification
    if not verification.get("passed"):
        return {"started": False,
                "refused": "independent executable verification: "
                           + str(verification.get("refused") or "no passing evidence"),
                **context}
    refusal = stage2_launch_decision(gpu_probe)
    if refusal is not None:
        return {"started": False, "refused": refusal, **context}
    cmd = [python, str(entry)]
    existing = stage_processes(entry)
    state_key = f"stage2__{tree.name}"
    state = load_state(state_key)
    worker_live = state_process_live(state.get("worker_pid"), HARNESS_DIR / "stage2_worker.py")
    # A pending smoke owns the tree even though no trainer is running yet, so a
    # second start must not archive and smoke it a second time.
    pending_worker = worker_live and state.get("status") in {"starting", "smoke_pending"}
    if existing or state_process_live(state.get("pid"), entry) or pending_worker:
        return {
            "started": False,
            "refused": (
                "a stage2 smoke worker for this tree is already live; poll stage2_status"
                if pending_worker and not existing
                else "a stage2 process for this tree is already live"
            ),
            "cmd": " ".join(cmd),
            "cwd": str(tree),
            "processes": existing,
            "state": state,
            **context,
        }
    # Existing artifacts must not block a run and must not be lost: copy them to
    # the archive first (copy, because the archive may live on another device),
    # keep the originals until the smoke has passed, and only then remove them so
    # the trainer writes fresh ones.
    existing_artifacts = [name for name in STAGE2_ARTIFACTS if (result_dir / name).is_file()]
    archived: list[str] = []
    if existing_artifacts:
        archive = STAGE2_ARCHIVE_ROOT / f"{tree.name}_{int(time.time())}"
        try:
            archive.mkdir(parents=True, exist_ok=False)
            for name in existing_artifacts:
                source = result_dir / name
                target = archive / name
                shutil.copy2(source, target)
                if target.stat().st_size != source.stat().st_size:
                    raise OSError(f"archive copy of {name} does not match the original size")
                archived.append(str(target))
        except OSError as exc:
            return {
                "started": False,
                "refused": f"could not archive the existing artifacts before retraining: {exc}",
                "cmd": " ".join(cmd),
                "cwd": str(tree),
                "existing_artifacts": existing_artifacts,
                "archived_artifacts": archived,
                "result_dir": str(result_dir),
                **context,
            }
    run_log = Path(logs["run_log"])
    run_log.parent.mkdir(parents=True, exist_ok=True)
    # The one-epoch gradient smoke forks a real 4-rank DDP run, so it costs a GPU
    # and tens of seconds: doing it inline would outlast the MCP client's request
    # deadline and lose the reply. A detached worker owns the rest of the
    # sequence instead -- smoke first, and only on a pass remove the archived
    # originals and launch -- and reports through the state file.
    worker = write_stage2_worker()
    worker_log = HARNESS_DIR / f"stage2_worker_{tree.name}.log"
    worker_spec = json.dumps(
        {
            "state_key": state_key,
            "python": python,
            "entry": str(entry),
            "arm": arm,
            "result_dir": str(result_dir),
            "existing_artifacts": existing_artifacts,
            "archived_artifacts": archived,
            "logs": logs,
            "gpu_probe": gpu_probe,
            "verification_id": verification_id,
        }
    )
    claim = {
        "status": "starting",
        "cmd": " ".join(cmd),
        "cwd": str(tree),
        "tree": str(tree),
        "arm": arm,
        "result_dir": str(result_dir),
        "archived_artifacts": archived,
        "gpu_probe": gpu_probe,
        **logs,
    }
    # Claim the tree before spawning so a worker that finishes early cannot have
    # its own result overwritten by the parent's stale pending state.
    save_state(state_key, claim)
    with worker_log.open("wb") as sink:
        worker_process = subprocess.Popen(
            [sys.executable, str(worker), str(tree), worker_spec],
            cwd=str(tree),
            stdout=sink,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
            env=logical_env(tree),
        )
    pending = {
        **claim,
        "status": "smoke_pending",
        "worker_pid": worker_process.pid,
        "worker_log": str(worker_log),
    }
    if load_state(state_key).get("status") == "starting":
        save_state(state_key, pending)
    return {
        "started": False,
        **pending,
        **context,
        "poll": {"tool": "stage2_status", "arguments": {"source_dir": str(tree)}},
    }


STAGE2_WORKER_TEMPLATE = '''"""Detached Stage2 worker: run the smoke, then launch when it passes.

Started by genrec-mcp's stage2_start so the tool call returns immediately. This
process owns nothing but the sequence after the archive step; the verdict is
written back into the server's state file and read with stage2_status.
"""
import importlib.util
import json
import sys

_spec = importlib.util.spec_from_file_location("genrec_server", {server!r})
server = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(server)

sys.exit(server.stage2_worker_main(sys.argv[1], json.loads(sys.argv[2])))
'''


def write_stage2_worker() -> Path:
    """Write the detached worker next to the smoke driver, outside the repository."""
    HARNESS_DIR.mkdir(parents=True, exist_ok=True)
    worker = HARNESS_DIR / "stage2_worker.py"
    worker.write_text(
        STAGE2_WORKER_TEMPLATE.format(server=str(Path(__file__).resolve())), encoding="utf-8"
    )
    return worker


def stage2_worker_main(tree_arg: str, spec: dict) -> int:
    """Smoke, then launch. Returns 0 when training was started, 1 otherwise."""
    tree = Path(tree_arg)
    state_key = spec["state_key"]
    entry = Path(spec["entry"])
    result_dir = Path(spec["result_dir"])
    logs = spec["logs"]
    pending = load_state(state_key)
    smoke_log = HARNESS_DIR / f"smoke_{tree.name}.log"
    checkpoint, checkpoint_source = probe_checkpoint(tree, result_dir)
    gpu_probe = spec.get("gpu_probe") or probe_gpu_layout()
    child_env = stage2_child_env(tree, gpu_probe)
    # Fail closed before touching the trainer: a worker can outlive its MCP call.
    verification_id = spec.get("verification_id")
    before = validate_agent_evidence_only(tree, verification_id)
    if not before.get("passed"):
        save_state(state_key, {"status": "refused", "worker_pid": pending.get("worker_pid"),
                               "tree": str(tree), "result_dir": str(result_dir),
                               "refused": "source/test verification drift before smoke: "
                                          + str(before.get("refused")), **logs})
        return 1
    smoke = run_stage2_smoke(
        tree,
        spec["python"],
        STAGE2_SMOKE_TIMEOUT_S,
        checkpoint,
        log_path=smoke_log,
        env=child_env,
    )
    summary = smoke["summary"] or {}
    smoke_detail = {
        "passed": smoke["passed"],
        "launch_mode": summary.get("launch_mode"),
        "steps": summary.get("steps"),
        "applied_grad_norms": (summary.get("applied_grad_norms") or [])[:5],
        "live_terms": sorted(
            key for key, value in (summary.get("probed_terms") or {}).items()
            if value.get("grad", 0) > 0
        ),
        "dead_terms": summary.get("dead_terms"),
        "nonfinite_terms": summary.get("nonfinite_terms"),
        "checkpoint_probe": summary.get("checkpoint_probe"),
        "checkpoint_source": checkpoint_source,
        "scratch": smoke["scratch"],
        "log": smoke["log"],
    }
    refused = None
    if not smoke["passed"]:
        refused = "the one-epoch gradient smoke failed, so no training was started"
    else:
        after = validate_agent_evidence_only(tree, verification_id)
        if not after.get("passed"):
            refused = "source/test verification drift after smoke: " + str(after.get("refused"))
    if refused is None:
        # The smoke passed, so the archived copies are the only backup needed: drop
        # the originals now so the run writes a fresh artifact set instead of a
        # mixture. A failed removal is fatal.
        undeleted = []
        for name in spec.get("existing_artifacts") or []:
            try:
                (result_dir / name).unlink()
            except OSError as exc:
                undeleted.append(f"{name}: {exc}")
        if undeleted:
            refused = f"could not remove the archived originals before retraining: {undeleted}"
    if refused is not None:
        save_state(
            state_key,
            {
                "status": "refused",
                "refused": refused,
                "smoke": smoke_detail,
                "traceback": summary.get("traceback"),
                "worker_pid": pending.get("worker_pid"),
                "tree": str(tree),
                "arm": spec.get("arm"),
                "result_dir": str(result_dir),
                "archived_artifacts": spec.get("archived_artifacts"),
                "gpu_probe": gpu_probe,
                **logs,
            },
        )
        return 1
    cmd = [spec["python"], str(entry)]
    with Path(logs["run_log"]).open("wb") as sink:
        child = subprocess.Popen(
            cmd,
            cwd=str(tree),
            stdout=sink,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
            env=child_env,
        )
    save_state(
        state_key,
        {
            "status": "running",
            "pid": child.pid,
            "launched_at": time.time(),
            "cmd": " ".join(cmd),
            "cwd": str(tree),
            "tree": str(tree),
            "arm": spec.get("arm"),
            "result_dir": str(result_dir),
            "smoke": smoke_detail,
            "archived_artifacts": spec.get("archived_artifacts"),
            "gpu_probe": gpu_probe,
            **logs,
        },
    )
    return 0


def _tail_text(path: Path, limit: int = 4000) -> str:
    """The last ``limit`` characters of a text file, read from the end."""
    try:
        with path.open("rb") as handle:
            size = handle.seek(0, os.SEEK_END)
            handle.seek(max(0, size - limit))
            return handle.read().decode("utf-8", "replace")
    except OSError:
        return ""


def _last_record(path: Path, event: str | None = None) -> dict | None:
    """The last parseable JSONL record, optionally restricted to one event."""
    for line in reversed(_tail_text(path, 65536).splitlines()):
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if event is None or record.get("event") == event:
            return record
    return None


def _gpu_processes() -> list[dict]:
    """Per-process GPU memory, so the monitor needs no second tool."""
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,used_memory",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return []
    entries = []
    for line in result.stdout.splitlines():
        pid, _, memory = line.partition(",")
        if pid.strip().isdigit():
            entries.append({"pid": int(pid.strip()), "used_memory_mib": memory.strip()})
    return entries


def tool_stage2_status(source_dir: str) -> dict:
    """Read-only: where the Stage2 run for one tree stands. Starts nothing."""
    tree, problem = resolve_stage2_tree(source_dir)
    if tree is None:
        return {
            "status": "unknown",
            "refused": problem,
            "available_trees": [item.name for item in stage2_trees()],
        }
    entry, _ = tree_entrypoint(tree)
    result_dir = tree_result_dir(tree)
    state = load_state(f"stage2__{tree.name}")
    live = stage_processes(entry) if entry else []
    worker_alive = state_process_live(state.get("worker_pid"), HARNESS_DIR / "stage2_worker.py")
    artifacts = {
        name: {
            "bytes": (result_dir / name).stat().st_size,
            "mtime": (result_dir / name).stat().st_mtime,
        }
        for name in STAGE2_ARTIFACTS
        if (result_dir / name).is_file()
    }
    logs = state.get("logs") or tree_log_paths(tree)
    latest_train = _last_record(Path(logs["metrics"]), "train")
    train_end = _last_record(Path(logs["metrics"]), "train_end")
    launched_at = state.get("launched_at")
    # A run is complete when this launch recorded its own train_end and the three
    # artifacts every tree writes are on disk. STAGE2_ARTIFACTS also carries
    # sids_raw.npy, which only some trees emit, so it cannot gate completeness.
    complete = all(name in artifacts for name in STAGE2_REQUIRED_ARTIFACTS) and bool(
        train_end and (launched_at is None or train_end.get("timestamp", 0) >= launched_at)
    )
    status = state.get("status")
    if not status:
        # Older state files carry no status: derive it from the recorded run and
        # the artifacts, so a completed run is never reported as never started.
        if live:
            status = "running"
        elif state.get("pid") or launched_at:
            status = "finished" if complete else "exited_incomplete"
        else:
            status = "idle"
    worker_log = state.get("worker_log")
    worker_log_tail = ""
    if worker_log:
        try:
            worker_log_tail = Path(worker_log).read_text(encoding="utf-8", errors="replace")[-2000:]
        except OSError:
            worker_log_tail = ""
    if status in {"smoke_pending", "starting"} and not worker_alive and not live:
        # The worker died before it could report; its log tail holds the reason.
        status = "worker_gone"
    elif status == "running" and not live:
        # The recorded pid is gone, so the state file alone must not keep claiming
        # a live run: the run either finished and wrote its artifacts, or died.
        status = "finished" if complete else "exited_incomplete"
    return {
        "status": status,
        "tree": str(tree),
        "arm": tree_arm(tree),
        "result_dir": str(result_dir),
        # Two different facts, kept apart on purpose: ``gpu_probe`` is the layout this
        # run was dispatched with (null when the tree never ran), ``current_gpu_probe``
        # is what this host looks like right now. A stale launch record must not be
        # mistaken for present-day availability, or the other way round.
        "gpu_probe": state.get("gpu_probe"),
        "current_gpu_probe": probe_gpu_layout(),
        "worker_pid": state.get("worker_pid"),
        "worker_alive": worker_alive,
        "worker_log": worker_log,
        "worker_log_tail": worker_log_tail,
        "pid": state.get("pid"),
        "launched_at": state.get("launched_at"),
        "live_processes": live,
        "gpu_processes": _gpu_processes(),
        "latest_train": latest_train,
        "train_end": train_end,
        "smoke_log": str(HARNESS_DIR / f"smoke_{tree.name}.log"),
        "smoke_log_tail": _tail_text(HARNESS_DIR / f"smoke_{tree.name}.log", 4000),
        "run_log_tail": _tail_text(Path(logs["run_log"]), 4000),
        "native_log_tail": _tail_text(Path(logs["native_log"]), 4000),
        "smoke": state.get("smoke"),
        "refused": state.get("refused"),
        "traceback": state.get("traceback"),
        "artifacts": artifacts,
        **logs,
    }





def stage3_paths(variant: str) -> dict:
    """Every path one Stage3 variant writes into."""
    root = RESULT3_ROOT / variant
    logs = root / "logs"
    return {
        "result_dir": root,
        "run_log": logs / "_stage3_run.log",
        "native_log": logs / "_stage3_launcher.log",
        "ckpt": root / "ckpt",
    }


def stage3_newest(pattern: str, variant: str, since: float | None = None) -> Path | None:
    """Newest file matching a variant-relative glob, optionally only if it was
    written at or after ``since``.

    Stage3 leaves its outputs in variant-wide directories, so a prior run's
    ``test_final.json`` or metrics stream would otherwise be mistaken for this
    launch's evidence.
    """
    found = [item for item in (RESULT3_ROOT / variant).glob(pattern) if item.is_file()]
    if since is not None:
        found = [item for item in found if item.stat().st_mtime >= since - 5.0]
    if not found:
        return None
    return max(found, key=lambda item: item.stat().st_mtime)


def stage3_variant_processes(variant: str) -> list[dict]:
    """Live processes of *this* variant only: the harness name carries it, and
    another variant's ranks must not be reported as this run's."""
    harness = f"run_stage3_param_{re.sub(r'[^A-Za-z0-9._-]+', '_', variant)}.py"
    return [item for item in stage3_processes() if harness in item.get("cmd", "")]


def tool_stage3_status(item_sids: str | None = None) -> dict:
    """Read-only: where the Stage3 run for one Stage2 SID stands. Starts nothing."""
    if not item_sids:
        return {"status": "unknown", "refused": "item_sids is required: pass the Stage2 item_sids.json path"}
    given = Path(item_sids)
    if given.is_dir():
        given = given / "item_sids.json"
    try:
        sid = given.resolve()
    except OSError as exc:
        return {"status": "unknown", "refused": f"cannot resolve {item_sids}: {exc}"}
    if RESULT2_ROOT.resolve() not in sid.parents:
        return {
            "status": "unknown",
            "refused": f"item_sids must live under {RESULT2_ROOT}",
            "given": str(given),
        }
    try:
        relative_dir = sid.parent.relative_to(RESULT2_ROOT.resolve())
        sid_logical = RESULT2_ROOT / relative_dir / sid.name
    except ValueError:
        sid_logical = given
    variant = derive_stage3_variant(sid_logical)
    paths = stage3_paths(variant)
    state = load_state(f"stage3__{variant}")
    launched_at = state.get("launched_at")
    # Variant directories are reused across launches, so only evidence written at
    # or after this launch's own timestamp counts as this run's.
    live = stage3_variant_processes(variant)
    metrics = stage3_newest("**/training_metrics.jsonl", variant, since=launched_at)
    test_final = stage3_newest("**/test_final.json", variant, since=launched_at)
    stale_test = stage3_newest("**/test_final.json", variant)
    if stale_test is not None and test_final is None:
        test_final = None
    latest_train = _last_record(metrics, "train") if metrics else None
    train_start = _last_record(metrics, "train_start") if metrics else None
    test_result = None
    if test_final is not None:
        try:
            test_result = json.loads(test_final.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            test_result = None
    if live:
        status = "running"
    elif launched_at or state.get("pid"):
        # The recorded pid is gone: this launch either wrote its final test or died.
        status = "finished" if test_final is not None else "exited_incomplete"
    elif state:
        status = "exited_incomplete"
    else:
        status = "idle"
    native_tail = _tail_text(paths["native_log"], 4000)
    run_tail = _tail_text(paths["run_log"], 4000)
    haystack = (native_tail + run_tail).lower()
    # Only real failure modes: a shutdown traceback from an operator SIGTERM is
    # normal output and must not be reported as an error.
    error_markers = sorted(
        marker
        for marker in ("cuda out of memory", "nccl error", "runtimeerror", "cuda error")
        if marker in haystack
    )
    protocol = frozen_protocol_check()
    total_epochs = (protocol.get("found") or {}).get("NUM_EPOCHS")
    progress = None
    if latest_train and latest_train.get("epoch") and total_epochs:
        progress = f"{latest_train['epoch']}/{total_epochs}"
    return {
        "status": status,
        "variant": variant,
        "sid": str(sid_logical),
        "result_dir": str(paths["result_dir"]),
        # Same split as stage2_status: the layout this run was dispatched with, and
        # what the host offers right now.
        "gpu_probe": state.get("gpu_probe"),
        "current_gpu_probe": probe_gpu_layout(),
        "pid": state.get("pid"),
        "launched_at": state.get("launched_at"),
        "live_processes": live,
        "gpu_processes": _gpu_processes(),
        # Which layout produced these numbers: the probe this run was dispatched
        # with, and the trainer's own train_start record. A single-card run has a
        # global train batch of 4096 against the 4-card 16384, so the two are not
        # interchangeable when comparing results.
        "launch": {
            "dispatch_mode": state.get("mode"),
            "reported_mode": (train_start or {}).get("launch_mode"),
            "world_size": (train_start or {}).get("world_size"),
            "train_batch_per_process": (train_start or {}).get("batch_size"),
            "global_train_batch": (train_start or {}).get("global_train_batch"),
            "infer_batch_per_process": (train_start or {}).get("infer_size"),
            "cuda_visible_devices": (train_start or {}).get("cuda_visible_devices"),
        },
        "latest_train": latest_train,
        "progress": progress,
        "total_epochs": total_epochs,
        "protocol_ok": protocol.get("ok"),
        "metrics": str(metrics) if metrics else None,
        "test_final": str(test_final) if test_final else None,
        "test_final_stale": str(stale_test) if (stale_test is not None and test_final is None) else None,
        "test_result": test_result,
        "ckpt_dir": str(paths["ckpt"]),
        "ckpt_files": sorted(item.name for item in paths["ckpt"].glob("**/*") if item.is_file())[:20]
        if paths["ckpt"].is_dir()
        else [],
        "run_log": str(paths["run_log"]),
        "native_log": str(paths["native_log"]),
        "run_log_tail": run_tail,
        "native_log_tail": native_tail,
        "error_markers": error_markers,
    }


def stage2_validity_admission(sid: Path) -> dict:
    """Check that the exported Stage2 SID is technically usable.

    Issue #120 removed every quality floor from admission: ER, level-two prefix
    counts, collisions and usage are measured as diagnostics and never refuse a
    Stage 3 launch. Only an unreadable, malformed or ambiguous SID does. The
    check is read-only and runs before GPU probing; a user-authored PASS record
    is never trusted.
    """
    import importlib.util
    src = REPO / "tools" / "genrec-mcp" / "stage2_quality_gate.py"
    try:
        spec = importlib.util.spec_from_file_location("stage2_quality_gate", src)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.evaluate(
            sid,
            RESULT2_ROOT / "TIGER_BEHAVIOR_SAMPLE_RQ-VAE" / "resample" / "item_sids.json",
            RESULT3_ROOT / "monotonicity" / "behavior_pairs_valid_s42.npz",
            RESULT3_ROOT / "monotonicity" / "behavior_pairs_valid_s42.meta.json",
        )
    except Exception as exc:
        return {"passed": False, "refused": "quality verifier unavailable: " +
                type(exc).__name__ + ": " + str(exc)}


def tool_stage3_start(item_sids: str | None = None) -> dict:
    """Start Stage3 for the SID path given by the caller -- the only argument.

    Nothing about the SID is hardcoded and nothing else is configurable: the path
    arrives as the single argument, must live under results/stage2_RQ-VAE/, is
    written into a per-launch entrypoint outside the repository, and the variant
    directory name is derived from the SID's own location. The trainer still
    receives no command-line arguments.
    """
    if not item_sids:
        return {"started": False, "refused": "item_sids is required: pass the Stage2 item_sids.json path"}
    given = Path(item_sids)
    if given.is_dir():
        given = given / "item_sids.json"
    if given.name != "item_sids.json":
        return {"started": False, "refused": "item_sids must name an item_sids.json file"}
    try:
        sid = given.resolve()
    except OSError as exc:
        return {"started": False, "refused": f"cannot resolve {item_sids}: {exc}"}
    if RESULT2_ROOT.resolve() not in sid.parents:
        return {
            "started": False,
            "refused": f"item_sids must live under {RESULT2_ROOT}",
            "given": str(given),
            "stage2_result_root": str(RESULT2_ROOT),
        }
    if not sid.is_file():
        return {"started": False, "refused": f"no item_sids file at {given}"}
    validity = stage2_validity_admission(sid)
    if not validity["passed"]:
        return {"started": False,
                "refused": "Stage2 SID validity: " + str(validity.get("refused")),
                "validity": validity}

    # Canonical logical (/home) form: Path.resolve() would embed the physical
    # /fs04 path, while the repository and Stage3 record /home paths.
    try:
        relative_dir = sid.parent.relative_to(RESULT2_ROOT.resolve())
        sid_logical = RESULT2_ROOT / relative_dir / sid.name
    except ValueError:
        sid_logical = given
    try:
        payload = json.loads(sid.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return {"started": False, "refused": f"unreadable SID json: {exc}"}
    if not isinstance(payload, dict) or not payload:
        return {"started": False, "refused": "SID json is not a non-empty object"}
    widths = sorted({len(row) for row in payload.values() if hasattr(row, "__len__")})
    sid_info = {
        "path": str(sid_logical),
        "physical_path": str(sid),
        "bytes": sid.stat().st_size,
        "items": len(payload),
        "row_widths": widths,
    }
    chosen_variant = derive_stage3_variant(sid_logical)
    context = {
        "sid": sid_info,
        "variant": chosen_variant,
        "per_user_dump": STAGE3_PER_USER_DUMP,
    }
    protocol = frozen_protocol_check()
    context["protocol"] = protocol
    if not protocol["ok"]:
        return {"started": False, "refused": "frozen Stage3 protocol mismatch", **context}
    # The trainer owns both layouts: 4-rank torchrun on a 4-card host, its own
    # single-card branch otherwise. The probe decides which one, and the trainer
    # declares whether it can do the second.
    stage3_probe = probe_gpu_layout()
    stage3_probe["entrypoint_supports_single_gpu"] = stage3_supports_single_gpu()
    context["gpu_probe"] = stage3_probe
    stage3_refusal = stage3_launch_decision(stage3_probe)
    if stage3_refusal is not None:
        return {"started": False, "refused": stage3_refusal, **context}
    launch_mode = stage3_probe.get("mode")
    harness = HARNESS_DIR / f"run_stage3_param_{re.sub(r'[^A-Za-z0-9._-]+', '_', chosen_variant)}.py"
    cmd = [PY3, str(harness)]
    run_log = RESULT3_ROOT / chosen_variant / "logs" / "_stage3_run.log"
    live = stage3_processes()
    context.update({"cmd": " ".join(cmd), "cwd": str(STAGE3_DIR), "run_log": str(run_log), "harness": str(harness), "launch_mode": launch_mode})
    if live:
        return {
            "started": False,
            "refused": "a stage3 process is already live",
            "processes": live,
            **context,
        }
    harness = write_stage3_harness(sid_logical, chosen_variant, STAGE3_PER_USER_DUMP)
    run_log.parent.mkdir(parents=True, exist_ok=True)
    with run_log.open("wb") as sink:
        child = subprocess.Popen(
            cmd,
            cwd=str(STAGE3_DIR),
            stdout=sink,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
            env=stage3_child_env(STAGE3_DIR, stage3_probe),
        )
    state = {
        "pid": child.pid,
        "launched_at": time.time(),
        "cmd": " ".join(cmd),
        "cwd": str(STAGE3_DIR),
        "variant": chosen_variant,
        "sid": str(sid),
        "stage2_validity": validity,
        "run_log": str(run_log),
        "native_log": str(RESULT3_ROOT / chosen_variant / "logs/_stage3_launcher.log"),
        "harness": str(harness),
        "mode": launch_mode,
        "gpu_probe": stage3_probe,
    }
    save_state(f"stage3__{chosen_variant}", state)
    return {"started": True, **state, "protocol": protocol}














# --------------------------------------------------------------------------
# Independent executable verification -- active Stage2 admission
# --------------------------------------------------------------------------
def run_agent_verification(tree: Path, verification_id: str) -> dict:
    """Read the current independent evidence for this tree.

    Issue #117 candidates are checked by the reviewed four-suite engine; Issue
    #120 round candidates by their three-verifier implementation record. Either
    way the Stage2 worker re-checks the evidence before and after its smoke.
    """
    import importlib.util
    if str(verification_id).startswith("round:"):
        return round_admission(tree, verification_id)
    verifier = REPO / "tools" / "research" / "verification" / "admission.py"
    if not verifier.is_file():
        return {"passed": False, "refused": "new native verifier is missing"}
    try:
        spec = importlib.util.spec_from_file_location("genrec_native_verification", verifier)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.run_preflight(tree, verification_id, PY2)
    except Exception as exc:
        return {"passed": False,
                "refused": "verification engine error: " + type(exc).__name__ + ": " + str(exc)}


def round_admission(tree: Path, verification_id: str) -> dict:
    """Admission for an Issue #120 round candidate.

    The round's own evidence is the implementation record: three independent
    verifiers, no unresolved blocking finding, the source digest the verifiers
    saw, and a real smoke result. Scientific paperwork is not a gate here; a
    correct, numerically stable, runnable trainer is.
    """
    import importlib.util
    from tools.research.rounds import pipeline as round_pipeline

    spec = importlib.util.spec_from_file_location("genrec_round_pipeline",
                                                  REPO / "tools/research/rounds/pipeline.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    parts = verification_id.split(":", 2)
    if len(parts) != 3 or parts[0] != "round":
        return {"passed": False, "refused": "not a round verification id"}
    _, round_id, idea_id = parts
    directory = REPO / "results" / "research" / "issue120" / "rounds" / round_id
    state = module.load_state(directory)
    records = [record for record in state.get("implementations", [])
               if record.get("idea_id") == idea_id]
    if not records:
        return {"passed": False, "refused": "the round has no implementation for " + idea_id}
    record = records[-1]
    if record.get("status") != "verified":
        return {"passed": False, "refused": "implementation status is " +
                str(record.get("status"))}
    if len(record.get("verdicts", [])) != 3:
        return {"passed": False, "refused": "the implementation was not reviewed by three agents"}
    if record.get("blocking_findings"):
        return {"passed": False, "refused": "unresolved blocking findings remain"}
    if Path(record.get("tree", "")).resolve() != Path(tree).resolve():
        return {"passed": False, "refused": "the record points at a different tree"}
    if record.get("source_digest") != module._source_digest(Path(tree)):
        return {"passed": False, "refused": "the tree changed after the verifiers saw it"}
    if not record.get("smoke"):
        return {"passed": False, "refused": "the implementation recorded no smoke result"}
    return {"passed": True, "round_id": round_id, "idea_id": idea_id,
            "revision": record.get("revision", 0),
            "source_digest": record.get("source_digest"),
            "verifiers": [verdict.get("verifier") for verdict in record.get("verdicts", [])],
            "smoke": record.get("smoke", {})}


def validate_agent_evidence_only(tree: Path, verification_id: str) -> dict:
    """Recheck immutable bytes and independent sessions without re-running GPU tests."""
    import importlib.util
    if str(verification_id).startswith("round:"):
        return round_admission(tree, verification_id)
    path = REPO / "tools" / "research" / "verification" / "admission.py"
    try:
        spec = importlib.util.spec_from_file_location("genrec_native_evidence", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _, _, record = module.validate_evidence(tree, verification_id)
        return {"passed": True, **record}
    except Exception as exc:
        return {"passed": False, "refused": type(exc).__name__ + ": " + str(exc)}


def tool_verification_check(source_dir: str, verification_id: str) -> dict:
    """Re-run independent Agent-authored tests without scheduling formal training."""
    tree, problem = resolve_stage2_tree(source_dir)
    if tree is None:
        return {"passed": False, "refused": problem}
    return run_agent_verification(tree, verification_id)


# --------------------------------------------------------------------------
# Thin MCP transport for the single native research DAG and verification Agents
# --------------------------------------------------------------------------
def _research_endpoints():
    repo = Path(__file__).resolve().parents[2]
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    from tools.research.mcp import handlers, verification_handlers
    return handlers, verification_handlers


def tool_research_start() -> dict:
    return _research_endpoints()[0].research_start()


def tool_research_resume(run_id: str) -> dict:
    return _research_endpoints()[0].research_resume(run_id)


def tool_research_status(run_id: str) -> dict:
    return _research_endpoints()[0].research_status(run_id)


def tool_research_results(run_id: str) -> dict:
    return _research_endpoints()[0].research_results(run_id)


def tool_research_cancel(run_id: str) -> dict:
    return _research_endpoints()[0].research_cancel(run_id)


def tool_research_verify_start(run_id: str, source_dir: str) -> dict:
    return _research_endpoints()[1].research_verify_start(run_id, source_dir)


def tool_research_verify_review(run_id: str, source_dir: str) -> dict:
    return _research_endpoints()[1].research_verify_review(run_id, source_dir)


def tool_research_verify_status(run_id: str) -> dict:
    return _research_endpoints()[1].research_verify_status(run_id)


TOOLS: dict[str, dict] = {
    "research_start": {
        "fn": tool_research_start, "args": {},
        "description": "Create a fresh Issue117 research run with a real run_id and "
        "start detached native DAG workers when source-configured safety permits; "
        "DATA and LITERATURE can work in parallel, and no GPU is launched.",
    },
    "research_resume": {
        "fn": tool_research_resume, "args": {"run_id": "string"},
        "description": "Resume eligible tasks for an existing research run using "
        "the same immutable root DAG and measured source evidence.",
    },
    "research_status": {
        "fn": tool_research_status, "args": {"run_id": "string"},
        "description": "Read actual run/task/worker status, attempts, failures, "
        "evidence hashes and whether detached workers are really alive.",
    },
    "research_results": {
        "fn": tool_research_results, "args": {"run_id": "string"},
        "description": "Return bounded, hash-checked previews of train-only behavior, "
        "literature, geometry proposal and scientific review artifacts.",
    },
    "research_cancel": {
        "fn": tool_research_cancel, "args": {"run_id": "string"},
        "description": "Cooperatively stop new tasks and running OpenCode subprocesses "
        "in a research run, without deleting evidence or stopping GPU jobs.",
    },
    "research_verify_start": {
        "fn": tool_research_verify_start,
        "args": {"run_id": "string", "source_dir": "string"},
        "description": "Start three independent OpenCode verification authors in a "
        "detached worker for an approved new mechanism; produce drafts only.",
    },
    "research_verify_review": {
        "fn": tool_research_verify_review,
        "args": {"run_id": "string", "source_dir": "string"},
        "description": "After reviewable test suites are committed, start a separate "
        "verification Reviewer and freeze source/test/proposal hashes; no GPU.",
    },
    "research_verify_status": {
        "fn": tool_research_verify_status, "args": {"run_id": "string"},
        "description": "Check real draft and independent review Worker status and "
        "whether an executable-verification manifest exists.",
    },

    "stage2_start": {
        "fn": tool_stage2_start,
        "args": {"source_dir": "string", "verification_id": "string"},
        "description": "Start one Stage2 candidate only after the source-bound "
        "independent math/gradient/DDP/entrypoint Agent tests have ACTUALLY executed "
        "and passed, then perform the existing real one-epoch gradient smoke before "
        "3000-epoch training; historical admissions are not accepted.",
    },
    "verification_check": {
        "fn": tool_verification_check,
        "args": {"source_dir": "string", "verification_id": "string"},
        "description": "Read current candidate/test/Agent identities, reject stale or "
        "missing evidence, and execute all four independently authored reviewed "
        "Python test suites with the project Stage2 interpreter; does not launch "
        "the formal Stage2 trainer.",
    },
    "stage2_status": {
        "fn": tool_stage2_status,
        "args": {"source_dir": "string"},
        "description": "Read-only: report where the Stage2 run of one tree stands -- "
        "smoke_pending / running / refused / idle / worker_gone, the smoke verdict, the launched "
        "pid and launched_at, the live process list and which artifacts are present. Starts "
        "nothing.",
    },
    "stage3_status": {
        "fn": tool_stage3_status,
        "args": {"item_sids": "string"},
        "description": "Read-only: report where the Stage3 run for one Stage2 SID stands -- "
        "running / finished / exited_incomplete / idle, the live processes, GPU usage, the latest "
        "train record from the run's metrics stream, the final test result when it exists, the "
        "checkpoint files, and bounded run/native log tails. Starts nothing.",
    },
    "stage3_start": {
        "fn": tool_stage3_start,
        "args": {"item_sids": "string"},
        "description": "Start the Stage3 run for one Stage2 SID. The tool takes exactly one "
        "argument: the item_sids.json path (a directory containing it is also accepted). The path "
        "must live under results/stage2_RQ-VAE/; the variant directory name is derived from its own "
        "location, and the launcher is generated for that call outside the repository, so nothing "
        "about the SID is hardcoded and the trainer receives no command-line arguments. The visible "
        "cards decide the layout: 4 or more run the frozen 4-rank DDP, fewer run the trainer's own "
        "single-card branch (train batch stays 4096 per process and the infer batch shrinks, so the "
        "global train batch is 4096 against the 4-card 16384). Refuses when the path is outside the "
        "Stage2 results root, when the frozen protocol differs, when no card is visible, or when a "
        "stage3 process is already live.",
    },
}


def tool_schema(name: str) -> dict:
    spec = TOOLS[name]
    properties = {}
    required = []
    for arg, kind in spec["args"].items():
        if kind.startswith("boolean"):
            properties[arg] = {"type": "boolean", "default": False}
        elif kind.startswith("integer"):
            properties[arg] = {"type": "integer"}
        elif kind.startswith("object"):
            # the mechanism declaration is structured data, not prose. Typing it as a string would
            # invite passing a path, which is exactly how a claim ends up read from a file.
            properties[arg] = {
                "type": "object",
                "description": "the mechanism declaration as the seven contract groups",
            }
        else:
            properties[arg] = {"type": "string"}
        if "optional" not in kind:
            required.append(arg)
    return {
        "name": name,
        "description": spec["description"],
        "inputSchema": {"type": "object", "properties": properties, "required": required},
    }


def dispatch_tool(name: str, arguments: dict) -> tuple[str, bool]:
    if name not in TOOLS:
        return json.dumps({"error": f"unknown tool {name}"}), True
    spec = TOOLS[name]
    kwargs = {}
    for arg in spec["args"]:
        if arg in arguments:
            kwargs[arg] = arguments[arg]
        elif spec["args"][arg].startswith("boolean"):
            kwargs[arg] = False
        elif spec["args"][arg].startswith("integer"):
            kwargs[arg] = 40
        else:
            kwargs[arg] = None
    try:
        return json.dumps(spec["fn"](**kwargs), indent=2, default=str), False
    except Exception as exc:  # a tool failure must not kill the server
        return json.dumps({"error": f"{type(exc).__name__}: {exc}"}), True


# --------------------------------------------------------------------------
# MCP stdio loop
# --------------------------------------------------------------------------
def respond(message_id, result=None, error=None) -> None:
    payload = {"jsonrpc": "2.0", "id": message_id}
    if error is not None:
        payload["error"] = error
    else:
        payload["result"] = result
    sys.stdout.write(json.dumps(payload) + "\n")
    sys.stdout.flush()


def handle(message: dict) -> None:
    method = message.get("method")
    message_id = message.get("id")
    if method == "initialize":
        respond(
            message_id,
            {
                "protocolVersion": PROTOCOL_VERSION,
                "capabilities": {"tools": {"listChanged": False}},
                "serverInfo": {"name": SERVER_NAME, "version": SERVER_VERSION},
            },
        )
    elif method == "notifications/initialized":
        return
    elif method == "ping":
        respond(message_id, {})
    elif method == "tools/list":
        respond(message_id, {"tools": [tool_schema(name) for name in TOOLS]})
    elif method == "tools/call":
        params = message.get("params") or {}
        text, is_error = dispatch_tool(params.get("name", ""), params.get("arguments") or {})
        respond(message_id, {"content": [{"type": "text", "text": text}], "isError": is_error})
    elif method in ("resources/list", "prompts/list"):
        key = "resources" if method.startswith("resources") else "prompts"
        respond(message_id, {key: []})
    elif message_id is not None:
        respond(message_id, error={"code": -32601, "message": f"method not found: {method}"})


def main() -> None:
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            message = json.loads(line)
        except ValueError as exc:
            print(f"[genrec-mcp] bad json: {exc}", file=sys.stderr, flush=True)
            continue
        try:
            handle(message)
        except Exception as exc:  # never die on one bad request
            print(f"[genrec-mcp] handler error: {exc}", file=sys.stderr, flush=True)
            if message.get("id") is not None:
                respond(message["id"], error={"code": -32603, "message": str(exc)})


if __name__ == "__main__":
    main()

# --------------------------------------------------------------------------
# Issue #120: multi-agent research rounds
# --------------------------------------------------------------------------
ROUND_WORKER_TEMPLATE = '''"""Detached round worker: run the agent stages of one round.

Started by genrec-mcp's research_round_start so the tool call returns
immediately. The worker runs the ideation, scoring, implementation and
verification agents and writes the round state; it never trains anything.
"""
import importlib.util
import json
import sys

_spec = importlib.util.spec_from_file_location("genrec_server", {server!r})
server = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(server)
sys.exit(server.round_worker_main(json.loads(sys.argv[1])))
'''


def round_worker_main(spec: dict) -> int:
    """Advance one round through its agent stages."""
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))
    from tools.research.rounds import pipeline as round_pipeline
    from tools.research.rounds import worker as round_worker

    round_id = spec["round_id"]
    directory = REPO / "results" / "research" / "issue120" / "rounds" / round_id
    try:
        summary = round_worker.advance(REPO, round_id)
    except Exception as exc:                               # noqa: BLE001
        state = round_pipeline.load_state(directory)
        round_pipeline.record_event(state, "round_failed", error=str(exc))
        round_pipeline.save_state(directory, state)
        return 1
    state = round_pipeline.load_state(directory)
    state["worker_summary"] = summary
    round_pipeline.save_state(directory, state)
    return 0


def write_round_worker() -> Path:
    HARNESS_DIR.mkdir(parents=True, exist_ok=True)
    worker = HARNESS_DIR / "round_worker.py"
    worker.write_text(ROUND_WORKER_TEMPLATE.format(server=str(Path(__file__).resolve())),
                      encoding="utf-8")
    return worker


def tool_research_round_start(objective: str = "") -> dict:
    """Start one multi-agent round: three ideation agents, then three scorers.

    The round is asynchronous; research_round_status reports where it is.
    """
    import time
    from tools.research.rounds import pipeline as round_pipeline

    round_id = "round_" + time.strftime("%Y%m%d_%H%M%S") + "_" + str(int(time.time() * 1000) % 1000)
    directory = round_pipeline.new_round(REPO, round_id, objective=objective)
    worker = write_round_worker()
    log = HARNESS_DIR / ("round_" + round_id + ".log")
    with log.open("wb") as sink:
        process = subprocess.Popen(
            [sys.executable, str(worker), json.dumps({"round_id": round_id})],
            cwd=str(REPO),
            stdout=sink,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
        )
    state = round_pipeline.load_state(directory)
    state["worker_pid"] = process.pid
    state["worker_log"] = str(log)
    round_pipeline.record_event(state, "round_worker_started", pid=process.pid)
    round_pipeline.save_state(directory, state)
    return {"started": True, "round_id": round_id, "directory": str(directory),
            "worker_pid": process.pid, "worker_log": str(log), "gpu_launched": False}


def _round_directory(round_id: str) -> Path:
    from tools.research.rounds import pipeline as round_pipeline
    if not round_id or "/" in round_id:
        raise ValueError("invalid round id")
    return round_pipeline.round_root(REPO) / round_id


def tool_research_round_status(round_id: str) -> dict:
    """Where the round is, what it produced and what is still open."""
    from tools.research.rounds import jobs as round_jobs
    from tools.research.rounds import pipeline as round_pipeline

    directory = _round_directory(round_id)
    state = round_pipeline.load_state(directory)
    queue = round_jobs.JobQueue(directory)
    summary = round_pipeline.summary(state)
    summary["worker_pid"] = state.get("worker_pid")
    summary["worker_alive"] = bool(state.get("worker_pid")) and state_process_live(
        state.get("worker_pid"), HARNESS_DIR / ("round_" + round_id + ".log"))
    summary["jobs"] = queue.report()
    summary["selection"] = {
        "selected": [entry["idea_id"] for entry in (state.get("selection") or {}).get("selected", [])],
        "rejected": (state.get("selection") or {}).get("rejected", []),
        "rule": (state.get("selection") or {}).get("rule", {})}
    summary["implementations"] = [
        {"idea_id": record["idea_id"], "status": record.get("status"),
         "revision": record.get("revision", 0), "tree": record.get("tree"),
         "verdicts": [verdict.get("verdict") for verdict in record.get("verdicts", [])],
         "failure_signature": record.get("failure_signature")}
        for record in state.get("implementations", [])]
    return summary


def tool_research_round_results(round_id: str) -> dict:
    """The full audit record of a round: ideas, matrix, selection, jobs, results."""
    from tools.research.rounds import jobs as round_jobs
    from tools.research.rounds import pipeline as round_pipeline

    directory = _round_directory(round_id)
    state = round_pipeline.load_state(directory)
    queue = round_jobs.JobQueue(directory)
    return {"round_id": round_id, "objective": state.get("objective"),
            "ideas": state.get("ideas", []), "duplicates": state.get("duplicates", []),
            "scores": state.get("scores", {}), "matrix": state.get("matrix"),
            "selection": state.get("selection"),
            "implementations": state.get("implementations", []),
            "repairs": state.get("repairs", []), "blocked": state.get("blocked", []),
            "jobs": queue.report(), "stage3_results": queue.stage3_results(),
            "events": state.get("events", [])}


def _promote_round_tree(tree: Path, idea_id: str) -> Path:
    """Place a round implementation inside the Stage2 allowlist.

    resolve_stage2_tree only accepts a direct child of stage2_RQ-VAE/, so a tree
    the round built in its own workspace is copied there under a name derived
    from the idea. The copy keeps the source digest the verifiers reviewed.
    """
    import shutil

    target = STAGE2_ROOT / ("TIGER_" + idea_id)
    source = Path(tree)
    if source.resolve() == target.resolve():
        return target
    if target.exists():
        return target
    ignore = shutil.ignore_patterns("logs", "__pycache__", "results", "dataset")
    shutil.copytree(source, target, ignore=ignore, dirs_exist_ok=True)
    return target


def tool_research_round_launch(round_id: str, parallel: int = 2) -> dict:
    """Enqueue every verified implementation and start up to `parallel` Stage 2 jobs.

    On a four-card host two jobs of one card each run concurrently: each launch is
    pinned to its own CUDA_VISIBLE_DEVICES and its tree and result directory are
    distinct, so the jobs cannot observe or overwrite each other.
    """
    import os
    from tools.research.rounds import jobs as round_jobs
    from tools.research.rounds import pipeline as round_pipeline

    directory = _round_directory(round_id)
    state = round_pipeline.load_state(directory)
    queue = round_jobs.JobQueue(directory, max_parallel=max(1, int(parallel)))
    queued = []
    for record in round_pipeline.verified_implementations(state):
        tree = _promote_round_tree(Path(record["tree"]), str(record["idea_id"]))
        target = tree_result_dir(tree)
        if not any(job["idea_id"] == record["idea_id"] and job["kind"] == "stage2"
                   for job in queue.jobs["jobs"]):
            queue.enqueue_stage2(idea_id=record["idea_id"], tree=tree, result_dir=target,
                                 round_id=round_id)
            queued.append(record["idea_id"])
    launched, refused = [], []
    free_gpus = probe_gpu_layout().get("physical") or []
    batch = queue.next_batch("stage2")
    for slot, job in enumerate(batch):
        gpu = free_gpus[slot % len(free_gpus)] if free_gpus else None
        previous = os.environ.get("CUDA_VISIBLE_DEVICES")
        if gpu is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
        try:
            queue.start(job["job_id"])
            verification = "round:" + round_id + ":" + str(job["idea_id"])
            outcome = tool_stage2_start(str(job["tree"]), verification)
        finally:
            if previous is None:
                os.environ.pop("CUDA_VISIBLE_DEVICES", None)
            else:
                os.environ["CUDA_VISIBLE_DEVICES"] = previous
        if outcome.get("started") is False and outcome.get("refused"):
            queue.fail(job["job_id"], str(outcome["refused"]),
                       evidence={"refused": outcome["refused"]})
            refused.append({"idea_id": job["idea_id"], "refused": outcome["refused"]})
        else:
            launched.append({"idea_id": job["idea_id"], "tree": job["tree"],
                             "gpu": gpu, "worker_pid": outcome.get("worker_pid"),
                             "status": outcome.get("status")})
    state["stage2_jobs"] = queue.report()["jobs"]
    round_pipeline.save_state(directory, state)
    return {"round_id": round_id, "queued": queued, "launched": launched,
            "refused": refused, "jobs": queue.report()}


def tool_research_round_collect(round_id: str) -> dict:
    """Collect finished Stage 2 jobs and hand every valid SID to Stage 3.

    A Stage 2 job that exported a technically valid SID becomes a Stage 3 job
    automatically; a job that failed keeps its evidence and the round continues.
    """
    from tools.research.rounds import jobs as round_jobs
    from tools.research.rounds import pipeline as round_pipeline

    directory = _round_directory(round_id)
    state = round_pipeline.load_state(directory)
    queue = round_jobs.JobQueue(directory)
    started, refused = [], []
    for job in list(queue.jobs["jobs"]):
        if job["kind"] != "stage2" or job["status"] not in ("pending", "running"):
            continue
        result_dir = Path(job["result_dir"])
        sid_path = result_dir / "item_sids.json"
        if not sid_path.is_file():
            if job["status"] == "running":
                continue
            queue.start(job["job_id"])
            tree = _promote_round_tree(Path(job["tree"]), str(job["idea_id"]))
            job["stage2_tree"] = str(tree)
            verification = "round:" + str(job.get("round_id", "")) + ":" + str(job["idea_id"])
            outcome = tool_stage2_start(str(tree), verification)
            if outcome.get("started") is False and outcome.get("refused"):
                queue.fail(job["job_id"], str(outcome["refused"]),
                           evidence={"refused": outcome["refused"]})
                refused.append({"idea_id": job["idea_id"], "refused": outcome["refused"]})
            else:
                started.append({"idea_id": job["idea_id"], "tree": job["tree"]})
            continue
        validity = round_jobs.sid_is_technically_valid(sid_path, repo=REPO)
        if not validity["valid"]:
            queue.fail(job["job_id"], "exported SID is not technically valid: " +
                       str(validity["refused"]), evidence=validity)
            refused.append({"idea_id": job["idea_id"], "refused": validity["refused"]})
            continue
        outcome = queue.complete_stage2(job["job_id"], sid_path=sid_path,
                                        metrics=validity["diagnostics"])
        stage3 = outcome.get("stage3")
        if stage3 is not None:
            result = tool_stage3_start(str(sid_path))
            stage3["started"] = bool(result.get("started"))
            if not result.get("started"):
                stage3["refused"] = result.get("refused")
    pipeline_module_state = state
    pipeline_module_state["stage2_jobs"] = queue.report()["jobs"]
    pipeline_module_state["stage3_jobs"] = [job for job in queue.report()["jobs"]
                                            if job["kind"] == "stage3"]
    round_pipeline.save_state(directory, pipeline_module_state)
    return {"round_id": round_id, "stage2_started": started, "stage2_refused": refused,
            "jobs": queue.report()}

TOOLS.update({
    "research_round_start": {
        "fn": tool_research_round_start,
        "args": {"objective": "string"},
        "description": "Start one multi-agent research round: three independent "
                       "ideation agents, three independent scorers that judge every "
                       "idea, an explicit multi-candidate selection, one isolated "
                       "implementation per selected idea and three verifiers per "
                       "implementation; no GPU work is started by this call.",
    },
    "research_round_status": {
        "fn": tool_research_round_status,
        "args": {"round_id": "string"},
        "description": "Report where a round stands: ideas, duplicates, score matrix, "
                       "selection, implementations, verifications, repairs and jobs.",
    },
    "research_round_results": {
        "fn": tool_research_round_results,
        "args": {"round_id": "string"},
        "description": "The full audit record of a round, including failed and blocked "
                       "branches and every Stage 3 result.",
    },
    "research_round_launch": {
        "fn": tool_research_round_launch,
        "args": {"round_id": "string", "parallel": "integer"},
        "description": "Enqueue the verified implementations and start up to "
                       "`parallel` Stage 2 jobs, each pinned to its own GPU.",
    },
    "research_round_collect": {
        "fn": tool_research_round_collect,
        "args": {"round_id": "string"},
        "description": "Start the queued Stage 2 jobs and hand every technically valid "
                       "exported SID to Stage 3 automatically.",
    },
})
