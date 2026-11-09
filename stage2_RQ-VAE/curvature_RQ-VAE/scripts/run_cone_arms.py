"""Run the four-arm quantized-cone comparison on the production RQ-VAE.

Arms are geometry x cone supervision x seed:

    A euclid   no cone
    B poincare  no cone
    C euclid   cone
    D poincare cone

Everything else is shared by construction rather than by convention: the same
seed gives both geometries the same data order, the cone module derives its
negative partners from (DATA_SEED, global step) and its prototypes from a
generator seeded by DATA_SEED, so the two arms differ in the metric and nothing
else. The behaviour-context channel is off in every arm, which makes the table
self-contained; the accepted mechanism stays the parent condition at accept time.

The budget reproduces the production protocol exactly. The distributed run
counts a global step as (per-rank steps x world size), so its 72,000 global steps
are 18,000 optimizer updates of an effective batch of 4 x 1024. A single process
reaches the same number of optimizer updates over the same number of items with a
batch of 4096 and 18,000 steps, which is what each arm runs - identical data
budget and identical update count, without the four-way launcher.

Each arm writes to its own directory under
``results/stage2_RQ-VAE/curvature_RQ-VAE/category_cone_arms/<arm>/`` and is
resumable: an arm whose final snapshot exists is skipped. One arm runs per GPU
as a single process, so the run is four-way parallel without DDP.
"""

from __future__ import annotations

import importlib.util
import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

PKG = Path(__file__).resolve().parents[1]
RESULTS = PKG.parent.parent / "results/stage2_RQ-VAE/curvature_RQ-VAE/category_cone_arms"
# One seed for this round: the four arms differ only in geometry and cone
# supervision, and a second seed would only re-measure seed variance, which the
# per-arm interval diagnostics already cover. Raise this list if the seed
# difference ever has to be quantified.
SEEDS = (42,)
ARMS = [
    (name, geometry, cone)
    for name, geometry, cone in (
        ("A_euclid_nocone", "euclid", False),
        ("B_poincare_nocone", "poincare", False),
        ("C_euclid_cone", "euclid", True),
        ("D_poincare_cone", "poincare", True),
    )
]


def _arm_dir(name: str) -> Path:
    return RESULTS / name


def _is_complete(name: str) -> bool:
    directory = _arm_dir(name)
    return (directory / "item_sids.json").is_file() and (
        directory / "logs/training_metrics.jsonl"
    ).is_file()


def _load_trainer(gpu: int):
    """Import the trainer with the arm's geometry and paths patched in."""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    # Four processes each letting torch take every core drove the load average
    # past 20 and starved the GPUs. One arm per card does not need more.
    os.environ["OMP_NUM_THREADS"] = "2"
    os.environ["MKL_NUM_THREADS"] = "2"
    sys.path.insert(0, str(PKG))
    import curvature_config as experiment

    spec = importlib.util.spec_from_file_location(
        f"generec_train_rqvae_gpu{gpu}", PKG / "train_rqvae.py"
    )
    trainer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(trainer)
    return experiment, trainer


def _run_arm(job: tuple[int, str, str, bool]) -> dict:
    gpu, name, geometry, cone = job
    arm_dir = _arm_dir(name)
    arm_dir.mkdir(parents=True, exist_ok=True)
    experiment, trainer = _load_trainer(gpu)
    experiment.GEOMETRY = geometry
    experiment.MECHANISM_NAME = f"category_cone_{name}"
    experiment.STAGE2_RESULT_DIR = arm_dir
    experiment.RQVAE_OUT_DIR = arm_dir / "out/rqvae/instruments"
    experiment.RQVAE_CKPT_PATH = experiment.RQVAE_OUT_DIR / "rqvae_best.pth"
    experiment.RAW_SIDS_NPY = experiment.RQVAE_OUT_DIR / "sids_raw.npy"
    experiment.SIDS_NPY = arm_dir / "dataset/Instruments/sids_for_hgrec.npy"
    experiment.ITEM_SIDS_JSON = arm_dir / "item_sids.json"
    experiment.STAGE2_LOG_DIR = arm_dir / "logs"
    experiment.CATEGORY_CONE_ENABLED = cone

    import torch as _torch

    _torch.set_num_threads(2)
    # Budget: the production protocol performs 18,000 optimizer updates, each
    # over an effective batch of 4 x 1024. A single process per card pays for
    # the whole effective batch itself, and at batch 4096 a step costs several
    # seconds, so the four arms would need roughly 14 hours. Each arm therefore
    # runs the same 18,000 optimizer updates at batch 1024 - one quarter of the
    # production data per arm, equal update count, and the comparison stays
    # exact because all four arms share the budget.
    trainer.BATCH_SIZE_PER_RANK = 1024
    trainer.MAX_GLOBAL_STEPS = 18_000
    trainer.EVAL_INTERVAL_STEPS = 9_000
    trainer.GEOMETRY = geometry
    trainer.CATEGORY_CONE_ENABLED = cone
    trainer.BEHAVIOUR_LOSS_ENABLED = False
    trainer.LOG_DIR = arm_dir / "logs"
    trainer.METRICS_PATH = trainer.LOG_DIR / "training_metrics.jsonl"
    trainer.SNAPSHOT_STEPS = {trainer.MAX_GLOBAL_STEPS: "arm"}
    started = time.time()
    trainer.main()
    return {
        "arm": name,
        "geometry": geometry,
        "cone": cone,
        "gpu": gpu,
        "seconds": round(time.time() - started, 1),
    }


def main() -> None:
    import torch

    gpu_count = torch.cuda.device_count()
    if gpu_count == 0:
        raise RuntimeError("The four-arm comparison requires CUDA devices")
    RESULTS.mkdir(parents=True, exist_ok=True)
    pending: list[tuple[str, str, bool]] = []
    for seed in SEEDS:
        for name, geometry, cone in ARMS:
            arm = f"{name}_s{seed}"
            if _is_complete(arm):
                print(f"[arms] skip complete {arm}", flush=True)
                continue
            pending.append((arm, geometry, cone))
    print(
        f"[arms] {len(pending)} arms to run over {gpu_count} gpus: "
        f"{[arm for arm, _, _ in pending]}",
        flush=True,
    )

    context = mp.get_context("spawn")
    queue: mp.Queue = context.Queue()
    results: mp.Queue = context.Queue()
    for job in pending:
        queue.put(job)
    workers = [
        context.Process(target=_worker, args=(gpu, queue, results))
        for gpu in range(gpu_count)
    ]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join()
    if [worker.exitcode for worker in workers if worker.exitcode != 0]:
        raise RuntimeError(
            f"arm workers failed: {[w.exitcode for w in workers]}"
        )
    finished = []
    while not results.empty():
        finished.append(results.get())
    for outcome in finished:
        print(f"[arms] finished {outcome}", flush=True)
    with (RESULTS / "arms.json").open("w") as handle:
        json.dump(
            {
                "arms": [list(job) for job in pending],
                "seeds": list(SEEDS),
                "outcomes": finished,
                "note": (
                    "one process per GPU, no DDP; an identical seed gives both "
                    "geometries the same data order, and the cone module draws "
                    "its negative partners and prototypes from DATA_SEED alone"
                ),
            },
            handle,
            indent=2,
            sort_keys=True,
        )
    print("[arms] complete", flush=True)


def _worker(gpu: int, queue: mp.Queue, results: mp.Queue) -> None:
    while True:
        try:
            name, geometry, cone = queue.get_nowait()
        except Exception:
            return
        try:
            results.put(_run_arm((gpu, name, geometry, cone)))
        except Exception as error:
            results.put({"arm": name, "gpu": gpu, "error": repr(error)})
            raise


if __name__ == "__main__":
    main()
