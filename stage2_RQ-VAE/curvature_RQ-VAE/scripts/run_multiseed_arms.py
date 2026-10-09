"""Repeat the surviving mechanism across training seeds, in both geometries.

The final claim has to hold across seeds, not at one draw. Two things stay fixed
on purpose. The cone's held-out split keeps its own data seed, because it is the
measuring device and a split that moved with the training seed would make the
seeds incomparable. Stage3's seed is frozen by the project protocol, so only the
Stage2 training seed varies here, which is the thing whose variance matters.

Set MECHANISM below to the flags of the mechanism that survived its round; the
defaults are the level-aligned supervision this program is currently testing.
"""

import importlib.util
import os
import sys
from pathlib import Path

PKG = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from launch_utils import launch_arm  # noqa: E402
RESULTS = PKG.parent.parent / "results/stage2_RQ-VAE/curvature_RQ-VAE/multiseed_arms"
SEEDS = (42, 43, 44)
# Only the hyperbolic arm: the claim to confirm is that the hyperbolic model
# beats the euclidean baseline, and that baseline's own seeds are measured
# separately by run_euclid_baseline_seeds.
GEOMETRIES = {"poincare": "poincare"}

# The mechanism under test. One entry per arm pair, so a later round can point
# this at whatever survived without touching the loop below.
# The mechanism that survived its round: metric-smooth quantization, which was
# the first arm in this program to come in above the 0.059158 bar (0.060499 at
# seed 42). Weights are the per-arm values normalized from the measured ratio of
# the term to the quantiser's own loss at initialization.
MECHANISM = {
    "name": "smooth",
    "cone": False,
    "coarse_level": 2,
    "level_aligned": False,
    "metric_margin": False,
    "separation_weight": 0.0,
    "separation_gap": 0.0,
    "smoothness": {"poincare": 1.0, "euclid": 0.34},
    "l2_codes": 256,
}


def _arm_dir(name: str, seed: int) -> Path:
    return RESULTS / f"{name}_s{seed}"


def _load_trainer():
    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
    os.environ["OMP_NUM_THREADS"] = "2"
    os.environ["MKL_NUM_THREADS"] = "2"
    sys.path.insert(0, str(PKG))
    import curvature_config as experiment

    spec = importlib.util.spec_from_file_location(
        "generec_train_rqvae_multiseed", PKG / "train_rqvae.py"
    )
    trainer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(trainer)
    return experiment, trainer


def configure(geometry: str, seed: int):
    name = f"{MECHANISM['name']}_{geometry}"
    arm_dir = _arm_dir(name, seed)
    arm_dir.mkdir(parents=True, exist_ok=True)
    experiment, trainer = _load_trainer()
    experiment.GEOMETRY = geometry
    experiment.MECHANISM_NAME = f"multiseed_{name}"
    experiment.STAGE2_RESULT_DIR = arm_dir
    experiment.RQVAE_OUT_DIR = arm_dir / "out/rqvae/instruments"
    experiment.RQVAE_CKPT_PATH = experiment.RQVAE_OUT_DIR / "rqvae_best.pth"
    experiment.RAW_SIDS_NPY = experiment.RQVAE_OUT_DIR / "sids_raw.npy"
    experiment.SIDS_NPY = arm_dir / "dataset/Instruments/sids_for_hgrec.npy"
    experiment.ITEM_SIDS_JSON = arm_dir / "item_sids.json"
    experiment.STAGE2_LOG_DIR = arm_dir / "logs"
    experiment.CATEGORY_CONE_ENABLED = MECHANISM["cone"]
    experiment.CATEGORY_CONE_METRIC_MARGIN = MECHANISM["metric_margin"]
    experiment.CATEGORY_CONE_SEPARATION_WEIGHT = MECHANISM["separation_weight"]
    experiment.CATEGORY_CONE_SEPARATION_GAP = MECHANISM["separation_gap"]
    experiment.CATEGORY_CONE_LEVEL_ALIGNED = MECHANISM["level_aligned"]
    experiment.CATEGORY_CONE_COARSE_LEVEL = MECHANISM["coarse_level"]
    experiment.QUANT_SMOOTHNESS_WEIGHT = MECHANISM["smoothness"][geometry]

    trainer.GEOMETRY = geometry
    trainer.SEED = int(seed)
    trainer.CATEGORY_CONE_ENABLED = MECHANISM["cone"]
    trainer.CATEGORY_CONE_METRIC_MARGIN = MECHANISM["metric_margin"]
    trainer.CATEGORY_CONE_SEPARATION_WEIGHT = MECHANISM["separation_weight"]
    trainer.CATEGORY_CONE_SEPARATION_GAP = MECHANISM["separation_gap"]
    trainer.CATEGORY_CONE_LEVEL_ALIGNED = MECHANISM["level_aligned"]
    trainer.CATEGORY_CONE_COARSE_LEVEL = MECHANISM["coarse_level"]
    trainer.QUANT_SMOOTHNESS_WEIGHT = MECHANISM["smoothness"][geometry]
    trainer.CODEBOOK_SIZE = (256, MECHANISM["l2_codes"], 256)
    trainer.BATCH_SIZE_PER_RANK = 1024
    trainer.MAX_GLOBAL_STEPS = 72_000
    trainer.EVAL_INTERVAL_STEPS = 9_000
    # On, so a multi-seed confirmation of any mechanism is comparable to the bar.
    trainer.BEHAVIOUR_LOSS_ENABLED = True
    trainer.LOG_DIR = arm_dir / "logs"
    trainer.METRICS_PATH = trainer.LOG_DIR / "training_metrics.jsonl"
    trainer.SNAPSHOT_STEPS = {trainer.MAX_GLOBAL_STEPS: "arm"}
    trainer._LAUNCHER["script"] = os.path.abspath(__file__)
    trainer._LAUNCHER["log"] = str(trainer.LOG_DIR / "train_multiseed.log")
    return trainer


if "RANK" in os.environ:
    geometry = os.environ.get("MULTISEED_GEOMETRY")
    seed = int(os.environ.get("MULTISEED_SEED", "0"))
    if geometry not in GEOMETRIES or seed not in SEEDS:
        raise RuntimeError(f"bad arm selector: {geometry!r} seed={seed}")
    trainer = configure(geometry, seed)
    print(
        f"[multiseed] geometry={geometry} seed={seed} "
        f"l2={MECHANISM['l2_codes']} cone={MECHANISM['cone']} "
        f"coarse_level={MECHANISM['coarse_level']} "
        f"smoothness={MECHANISM['smoothness'][geometry]}",
        flush=True,
    )
    trainer.main()
else:
    for seed in SEEDS:
        for geometry in GEOMETRIES:
            print(f"[multiseed] ===== {geometry} seed {seed} =====", flush=True)
            trainer = configure(geometry, seed)
            # launch_arm blocks and returns; _launch_via_torchrun would end the
            # process here and the remaining arms would never run.
            launch_arm(trainer, {"MULTISEED_GEOMETRY": geometry, "MULTISEED_SEED": seed})
    print("[multiseed] all arms finished", flush=True)