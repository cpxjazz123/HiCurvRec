"""The parts of a Stage 2 candidate that are not research.

A candidate is a research question: which behaviour signal, which curved
operation, how the two are connected, and what the quantizer does with them.
Everything else — where the data is, where the results go, how the four ranks
are started, which port, which frozen hyperparameters, how an epoch is timed and
how the SID is exported — is the same for every candidate and is owned here.

Nothing in this module is allowed to be overridden by a command line flag, an
environment variable or the candidate's own code. A candidate that needs a
different training procedure asks for it explicitly; it does not re-implement the
launcher.
"""
from __future__ import annotations

import os
from pathlib import Path

#: The frozen recipe. These are the values every published arm was trained with;
#: a candidate may not change them, and the conformance check compares the
#: candidate's own constants against this table.
FROZEN = {
    "epochs": 3000,
    "batch_size_per_rank": 1024,
    "learning_rate": 1e-3,
    "weight_decay": 1e-4,
    "hidden_sizes": (512, 256, 128),
    "codebook_num": 3,
    "codebook_size": (256, 256, 256),
    "codebook_dim": 32,
    "gradient_clip_norm": 1.0,
    "seed": 42,
}

#: The port of the four-rank Stage 2 job. Stage 3 owns 50201, and the two must
#: never collide.
STAGE2_PORT = 50202
STAGE3_PORT = 50201
WORLD_SIZE = 4

#: The interpreter the Stage 2 candidates run under. It is the one with torch,
#: and an agent that has to execute the trainer needs it as well.
STAGE2_PYTHON = "/home/wlia0047/ar57_scratch/wenyu/genrec_env_v2/bin/python3.9"

#: The environment variable that tells a trainer to run in this process instead
#: of forking the four-rank job. It is part of the frozen contract: the MCP
#: single-card smoke sets it, and a trainer that ignores it breaks the smoke.
SINGLE_GPU_ENV = "GENREC_SINGLE_GPU"

DATA_FILES = {
    "train": ("results", "stage0_build_parquet", "train.parquet"),
    "valid": ("results", "stage0_build_parquet", "valid.parquet"),
    "test": ("results", "stage0_build_parquet", "test.parquet"),
    "embeddings": ("stage1_GeneEmbedding", "output", "sentence_t5.npy"),
}

RESULT_ROOT = ("results", "stage2_RQ-VAE")


class ContractError(RuntimeError):
    """The tree does not satisfy the frozen contract."""


def resolve_root(start: Path | str | None = None) -> Path:
    """The repository root, found by an unambiguous marker.

    ``stage2_RQ-VAE`` alone is ambiguous: ``results/stage2_RQ-VAE`` also exists,
    and it sits closer to a candidate tree in the round workspace, so a marker on
    that name alone resolves to ``results/`` and every data path is wrong. The
    root is the directory that holds the source trees, the results and the Stage 1
    output at once.
    """
    here = Path(start or __file__).resolve()
    for parent in here.parents:
        if ((parent / "stage2_RQ-VAE").is_dir()
                and (parent / "results").is_dir()
                and (parent / "stage1_GeneEmbedding").is_dir()):
            return parent
    raise ContractError("cannot locate the repository root above " + str(here))


def data_path(root: Path | str, name: str) -> Path:
    """One frozen data file, resolved from the root."""
    root = Path(root)
    try:
        parts = DATA_FILES[name]
    except KeyError as exc:
        raise ContractError("unknown data file: " + str(name)) from exc
    path = root.joinpath(*parts)
    if not path.is_file():
        raise ContractError("the frozen data file is missing: " + str(path))
    return path


def run_workspace(root: Path | str, run_id: str, arm: str) -> dict:
    """Where one run of one arm writes.

    The workspace is derived from the run id and the arm name, and it is never
    shared with another run: a new revision of the same arm gets its own
    directory, so a running job cannot have its directory replaced under it. The
    layout is

        runs/<run_id>/<arm>/src      the immutable snapshot the run reads
        runs/<run_id>/<arm>/result   the SIDs, the metrics and the logs

    and both are created once. ``source_digest`` of the snapshot is what the
    review and the admission compare, so the snapshot and not the live tree is
    the thing a run is bound to.
    """
    if not run_id or not arm or "/" in run_id or "/" in arm:
        raise ContractError("a run workspace needs a safe run id and arm name")
    root = Path(root)
    base = root.joinpath("results", *RESULT_ROOT[1:], "_runs", run_id, arm)
    source = base / "src"
    result = base / "result"
    for directory in (source, result):
        directory.mkdir(parents=True, exist_ok=True)
    return {"source": source, "result": result, "base": base,
            "sid": result / "sids_for_hgrec.npy",
            "item_sids": result / "item_sids.json",
            "checkpoint": result / "rqvae_best.pth",
            "logs": result / "logs",
            "metrics": result / "logs" / "training_metrics.jsonl"}


def launch_env(extra: dict | None = None) -> dict:
    """The environment of a training subprocess.

    A stable working directory and a fixed set of thread limits: the working
    directory of the four ranks is the RESULT directory, because the candidate
    tree is a snapshot that a later run replaces and a process whose working
    directory disappears dies on the next ``os.getcwd()`` call, which torch makes
    while it is imported.
    """
    env = dict(os.environ)
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        env.setdefault(key, "4")
    env.pop(SINGLE_GPU_ENV, None)
    if extra:
        env.update(extra)
    return env
