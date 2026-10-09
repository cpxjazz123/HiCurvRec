"""The bar's own configuration at other training seeds.

Every mechanism this program has tried has come in at or below the plain
euclidean no-cone arm, and the differences that decide acceptance are a few
percent. Before any of those comparisons can be trusted, the spread the
configuration has against itself has to be known, which is what this measures:
the same euclidean, 256-codes-per-level, no-cone recipe that produced 0.059158,
run at two further seeds.

Only the training seed moves. The cone's held-out split keeps its own seed so
the measuring device is identical across runs, and Stage3's seed is frozen by
the project protocol. Note that the training seed also drives the PCA basis used
on the Stage1 embeddings, so the seeds differ in that pre-processing rotation as
well as in initialization and data order.
"""

import importlib.util
import os
import sys
from pathlib import Path

PKG = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from launch_utils import launch_arm  # noqa: E402
RESULTS = PKG.parent.parent / "results/stage2_RQ-VAE/curvature_RQ-VAE"
SEEDS = (43, 44)
GEOMETRY = "euclid"
ARM_PREFIX = "euclid_l2_256_72k"


def _arm_dir(seed: int) -> Path:
    return RESULTS / f"{ARM_PREFIX}_s{seed}"


def _is_complete(seed: int) -> bool:
    """True when this seed already exported its SIDs."""
    directory = _arm_dir(seed)
    return (directory / "item_sids.json").is_file() and (
        directory / "logs/training_metrics.jsonl"
    ).is_file()


def _load_trainer():
    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
    os.environ["OMP_NUM_THREADS"] = "2"
    os.environ["MKL_NUM_THREADS"] = "2"
    sys.path.insert(0, str(PKG))
    import curvature_config as experiment

    spec = importlib.util.spec_from_file_location(
        "generec_train_rqvae_baseline_seed", PKG / "train_rqvae.py"
    )
    trainer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(trainer)
    return experiment, trainer


def configure(seed: int):
    arm_dir = _arm_dir(seed)
    arm_dir.mkdir(parents=True, exist_ok=True)
    experiment, trainer = _load_trainer()
    experiment.GEOMETRY = GEOMETRY
    experiment.MECHANISM_NAME = f"baseline_seed{seed}"
    experiment.STAGE2_RESULT_DIR = arm_dir
    experiment.RQVAE_OUT_DIR = arm_dir / "out/rqvae/instruments"
    experiment.RQVAE_CKPT_PATH = experiment.RQVAE_OUT_DIR / "rqvae_best.pth"
    experiment.RAW_SIDS_NPY = experiment.RQVAE_OUT_DIR / "sids_raw.npy"
    experiment.SIDS_NPY = arm_dir / "dataset/Instruments/sids_for_hgrec.npy"
    experiment.ITEM_SIDS_JSON = arm_dir / "item_sids.json"
    experiment.STAGE2_LOG_DIR = arm_dir / "logs"
    experiment.CATEGORY_CONE_ENABLED = False

    trainer.GEOMETRY = GEOMETRY
    trainer.SEED = int(seed)
    trainer.CATEGORY_CONE_ENABLED = False
    trainer.CODEBOOK_SIZE = (256, 256, 256)
    trainer.BATCH_SIZE_PER_RANK = 1024
    trainer.MAX_GLOBAL_STEPS = 72_000
    trainer.EVAL_INTERVAL_STEPS = 36_000
    trainer.LOG_DIR = arm_dir / "logs"
    trainer.METRICS_PATH = trainer.LOG_DIR / "training_metrics.jsonl"
    trainer.SNAPSHOT_STEPS = {trainer.MAX_GLOBAL_STEPS: "arm"}
    trainer._LAUNCHER["script"] = os.path.abspath(__file__)
    trainer._LAUNCHER["log"] = str(trainer.LOG_DIR / "train_baseline.log")
    return trainer


if "RANK" in os.environ:
    seed = int(os.environ.get("BASELINE_SEED", "0"))
    if seed not in SEEDS:
        raise RuntimeError(f"BASELINE_SEED must be one of {SEEDS}, got {seed}")
    trainer = configure(seed)
    print(f"[baseline-seed] seed={seed} geometry={GEOMETRY} cone=off", flush=True)
    trainer.main()
else:
    for seed in SEEDS:
        if _is_complete(seed):
            print(f"[baseline-seed] seed {seed} already complete, skipping", flush=True)
            continue
        print(f"[baseline-seed] ===== seed {seed} =====", flush=True)
        trainer = configure(seed)
        # launch_arm blocks and returns; the trainer's own launcher ends the
        # process, which would leave the later seeds unrun.
        launch_arm(trainer, {"BASELINE_SEED": seed})
    print("[baseline-seed] all seeds finished", flush=True)