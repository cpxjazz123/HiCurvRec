"""Metric-smooth quantization at the production operating point.

Stage3 has to learn content -> SID from the item text, and the SID is the
quantiser's output. This round asks the quantiser to be a smooth function of the
input in the arm's own metric: two latents the metric calls close should not be
sent to code vectors the metric calls far apart. No partition changes and no
sharing is created, so this is the one lever that does not trade against the
uniqueness the Stage3 numbers reward. Cones are off; the parent is the production
protocol, 256 codes per level and 72,000 updates.

The weight is normalized per arm from the measured ratio of the term to the
quantiser's own loss at initialization (0.248 hyperbolic, 0.744 euclidean), so
both arms start with the term at the same quarter of that loss instead of the
euclidean arm being pushed three times harder by the same constant.

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
RESULTS = PKG.parent.parent / "results/stage2_RQ-VAE/curvature_RQ-VAE/smoothness_arms"
SEED = 42
ARMS = {
    "A_euclid_smooth": "euclid",
    "B_poincare_smooth": "poincare",
}
# Normalized from the measured init ratios so both arms carry the same relative
# pressure; see the module docstring.
SMOOTHNESS_WEIGHT = {"poincare": 1.0, "euclid": 0.34}


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
        "generec_train_rqvae_smooth", PKG / "train_rqvae.py"
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
    experiment.CATEGORY_CONE_ENABLED = False
    experiment.CATEGORY_CONE_METRIC_MARGIN = False

    trainer.GEOMETRY = geometry
    trainer.CATEGORY_CONE_ENABLED = False
    trainer.CATEGORY_CONE_METRIC_MARGIN = False
    experiment.QUANT_SMOOTHNESS_WEIGHT = SMOOTHNESS_WEIGHT[geometry]
    trainer.QUANT_SMOOTHNESS_WEIGHT = SMOOTHNESS_WEIGHT[geometry]
    trainer.CODEBOOK_SIZE = (256, 256, 256)
    trainer.BATCH_SIZE_PER_RANK = 1024
    trainer.MAX_GLOBAL_STEPS = 72_000
    trainer.EVAL_INTERVAL_STEPS = 9_000
    # Keep the behaviour loss on. The bar and the accepted model both run it, so
    # turning it off here would make this arm differ from the bar in two ways and
    # leave the comparison unable to say which one moved Stage3.
    trainer.BEHAVIOUR_LOSS_ENABLED = True
    trainer.LOG_DIR = arm_dir / "logs"
    trainer.METRICS_PATH = trainer.LOG_DIR / "training_metrics.jsonl"
    trainer.SNAPSHOT_STEPS = {trainer.MAX_GLOBAL_STEPS: "arm"}
    trainer._LAUNCHER["script"] = os.path.abspath(__file__)
    trainer._LAUNCHER["log"] = str(trainer.LOG_DIR / "train_smooth.log")
    return trainer


if "RANK" in os.environ:
    name = os.environ.get("CONE_ARM")
    if name not in ARMS:
        raise RuntimeError(f"CONE_ARM must name an arm, got {name!r}")
    trainer = configure(name)
    print(
        f"[smooth] arm={name} geometry={ARMS[name]} "
        f"codebook={trainer.CODEBOOK_SIZE} steps={trainer.MAX_GLOBAL_STEPS} "
        f"smoothness={trainer.QUANT_SMOOTHNESS_WEIGHT} cone={trainer.CATEGORY_CONE_ENABLED}",
        flush=True,
    )
    trainer.main()
else:
    for name in ARMS:
        if _is_complete(name):
            print(f"[smooth] arm {name} already complete, skipping", flush=True)
            continue
        print(f"[smooth] ===== arm {name} =====", flush=True)
        trainer = configure(name)
        # launch_arm blocks and returns; _launch_via_torchrun would end the
        # process here and the remaining arms would never run.
        launch_arm(trainer, {"CONE_ARM": name})
    print("[smooth] both arms finished", flush=True)
