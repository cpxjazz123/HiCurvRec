"""Move the coarse cone off level 2, the level whose merging cost the program measured.

Level 1 already carries the category (fine-category AMI 0.32) and level 2 carries item
identity, which is what keeps the L1L2 prefix unique. The coarse cone on level 2
forces every fine category under one coarse category into a single cone, and that
merge is the transfer failure every earlier cone arm paid for. This arm puts the
coarse cone on level 1 instead and leaves level 2 to the reconstruction objective, so
nothing new forces sharing and the failure mode is "no change" rather than collapse.

The two arms differ in one thing: whether the cone margin is measured in the
arm's own metric. The Poincare ball is conformal, so a cone of a given aperture
selects the same points in either arm, and the geometries only disagree about
how far a point sits from the cone boundary once distance is measured with that
arm's metric. Near the boundary the hyperbolic factor grows without bound, which
is the geometric statement that a fixed angular gap buys an exponentially larger
separation there. The Euclidean arm cannot reproduce that: its factor is
identically one, so its objective is the plain angular cone loss and serves as
the control.

Both arms use 64 codes at level 2, where level-2 codes are shared across fine
categories and the cone has something to do, and both run the accepted model's
72,000 updates so the budget matches the Stage3 bar.
"""

import importlib.util
import json
import os
import sys
from pathlib import Path

PKG = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from launch_utils import launch_arm  # noqa: E402
RESULTS = PKG.parent.parent / "results/stage2_RQ-VAE/curvature_RQ-VAE/coarse_l1_arms"
SEED = 42
ARMS = {
    "A_euclid_coarse_l1": "euclid",
    "B_poincare_coarse_l1": "poincare",
}
# Frozen before the run, from the measured geometry of the saved arms: the
# cones of different fine categories already overlap (worst pair by 0.53 rad),
# and at this gap the requirement is 0.019 rad in the hyperbolic arm against
# 0.050 rad in the euclidean one, so the same demand costs the hyperbolic arm
# less angle. The weight keeps that term comparable to the containment terms
# instead of dominating them.
# Frozen before the run. Level 2 is supervised by the fine cone instead of the
# coarse one, so the level Stage3 reads for fine discrimination carries the fine
# category; the coarse cone moves to level 1. Everything else stays at the
# production protocol that produced the 0.059158 bar, including 256 codes per
# level, where the L1L2 prefix currently carries no fine-category structure at
# all (fine AMI 0.0001), which is the headroom this mechanism targets.
ALIGNED = False
COARSE_LEVEL = 1
L2_CODES = 256


def _arm_dir(name: str) -> Path:
    return RESULTS / f"{name}_s{SEED}"


def _is_complete(name: str) -> bool:
    """True when this arm already exported its SIDs; a rerun must not repeat it."""
    directory = _arm_dir(name)
    return (directory / "item_sids.json").is_file() and (
        directory / "logs/training_metrics.jsonl"
    ).is_file()


def _load_trainer(gpu: int = 0):
    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    os.environ["OMP_NUM_THREADS"] = "2"
    os.environ["MKL_NUM_THREADS"] = "2"
    sys.path.insert(0, str(PKG))
    import curvature_config as experiment

    spec = importlib.util.spec_from_file_location(
        "generec_train_rqvae_coarse_l1", PKG / "train_rqvae.py"
    )
    trainer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(trainer)
    return experiment, trainer


def configure(name: str) -> None:
    geometry = ARMS[name]
    arm_dir = _arm_dir(name)
    arm_dir.mkdir(parents=True, exist_ok=True)
    experiment, trainer = _load_trainer()
    experiment.GEOMETRY = geometry
    experiment.MECHANISM_NAME = f"metric_cone_{name}"
    experiment.STAGE2_RESULT_DIR = arm_dir
    experiment.RQVAE_OUT_DIR = arm_dir / "out/rqvae/instruments"
    experiment.RQVAE_CKPT_PATH = experiment.RQVAE_OUT_DIR / "rqvae_best.pth"
    experiment.RAW_SIDS_NPY = experiment.RQVAE_OUT_DIR / "sids_raw.npy"
    experiment.SIDS_NPY = arm_dir / "dataset/Instruments/sids_for_hgrec.npy"
    experiment.ITEM_SIDS_JSON = arm_dir / "item_sids.json"
    experiment.STAGE2_LOG_DIR = arm_dir / "logs"
    experiment.CATEGORY_CONE_ENABLED = True
    experiment.CATEGORY_CONE_METRIC_MARGIN = False
    experiment.CATEGORY_CONE_LEVEL_ALIGNED = ALIGNED
    experiment.CATEGORY_CONE_COARSE_LEVEL = COARSE_LEVEL

    trainer.GEOMETRY = geometry
    trainer.CATEGORY_CONE_ENABLED = True
    trainer.CATEGORY_CONE_METRIC_MARGIN = False
    trainer.CATEGORY_CONE_LEVEL_ALIGNED = ALIGNED
    trainer.CATEGORY_CONE_COARSE_LEVEL = COARSE_LEVEL
    trainer.CODEBOOK_SIZE = (256, L2_CODES, 256)
    trainer.BATCH_SIZE_PER_RANK = 1024
    trainer.MAX_GLOBAL_STEPS = 72_000
    trainer.EVAL_INTERVAL_STEPS = 9_000
    trainer.BEHAVIOUR_LOSS_ENABLED = False
    trainer.LOG_DIR = arm_dir / "logs"
    trainer.METRICS_PATH = trainer.LOG_DIR / "training_metrics.jsonl"
    trainer.SNAPSHOT_STEPS = {trainer.MAX_GLOBAL_STEPS: "arm"}
    trainer._LAUNCHER["script"] = os.path.abspath(__file__)
    trainer._LAUNCHER["log"] = str(trainer.LOG_DIR / "train_coarse_l1.log")
    return trainer


if "RANK" in os.environ:
    name = os.environ.get("CONE_ARM")
    if name not in ARMS:
        raise RuntimeError(f"CONE_ARM must name an arm, got {name!r}")
    trainer = configure(name)
    print(
        f"[coarse-l1] arm={name} geometry={ARMS[name]} "
        f"codebook={trainer.CODEBOOK_SIZE} steps={trainer.MAX_GLOBAL_STEPS} "
        f"metric_margin={trainer.CATEGORY_CONE_METRIC_MARGIN} "
        f"level_aligned={trainer.CATEGORY_CONE_LEVEL_ALIGNED} "
        f"coarse_level={trainer.CATEGORY_CONE_COARSE_LEVEL} "
        f"l2={L2_CODES} metric_margin={trainer.CATEGORY_CONE_METRIC_MARGIN}",
        flush=True,
    )
    trainer.main()
else:
    for name in ARMS:
        if _is_complete(name):
            print(f"[coarse-l1] arm {name} already complete, skipping", flush=True)
            continue
        print(f"[coarse-l1] ===== arm {name} =====", flush=True)
        trainer = configure(name)
        # launch_arm blocks and returns; _launch_via_torchrun would end the
        # process here and the remaining arms would never run.
        launch_arm(trainer, {"CONE_ARM": name})
    print("[coarse-l1] both arms finished", flush=True)
