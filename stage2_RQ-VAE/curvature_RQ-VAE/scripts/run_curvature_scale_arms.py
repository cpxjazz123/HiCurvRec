"""Does the hyperbolic geometry do anything at the operating point.

Every geometry comparison this project has run sits at curvature 1.0 with the
code vectors at working radius 0.2, and those two numbers together decide how
much hyperbolic structure the model actually sees. The conformal factor of the
Poincare ball is 2 / (1 - c r^2), so at c = 1 and r = 0.2 the factor is 1.04:
a four percent correction. The "hyperbolic" arms have therefore been running in
a region that is very nearly euclidean, which is a plausible reading of why the
geometry lever has come back neutral or negative every time it was pulled.

This round moves the operating point instead of adding another term. The code
radius is fixed by the protocol at 0.2, so the curvature is the only free
quantity, and it is set to make the conformal factor order one rather than order
one percent: c = 25 puts c r^2 at 0.58 and the factor at 2.38. The exponential
and logarithmic maps stay numerically clean there - the tangent-to-ball round
trip is accurate to 3e-8 and the largest radius reached is 0.152, well inside
the ball.

The curvature stays closed form, fixed before training, and untrainable, so the
FCCR-1 contract is untouched. The parent is the production protocol at 256 codes
per level and 72,000 updates, and the two arms differ in one thing: whether the
encoder, the quantiser and the residual transport are evaluated in the euclidean
metric or in the Poincare ball at that curvature. The euclidean arm is the
control, since its metric has no curvature to set.
"""

import importlib.util
import os
import sys
from pathlib import Path

PKG = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from launch_utils import launch_arm  # noqa: E402
RESULTS = PKG.parent.parent / "results/stage2_RQ-VAE/curvature_RQ-VAE/curvature_scale_arms"
SEED = 42
ARMS = {
    "A_euclid_c25": "euclid",
    "B_poincare_c25": "poincare",
}
# Derived, not swept: c * r^2 = 25 * 0.04 = 0.58, so the conformal factor is 2.38
# instead of 1.04. One value, fixed before the run, identical at every level.
LAYER_CURVATURES = (25.0, 25.0, 25.0)


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
    spec = importlib.util.spec_from_file_location("train_rqvae", PKG / "train_rqvae.py")
    trainer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(trainer)
    experiment = sys.modules["curvature_config"]
    return experiment, trainer


def configure(name: str) -> None:
    geometry = ARMS[name]
    arm_dir = _arm_dir(name)
    experiment, trainer = _load_trainer()
    experiment.MECHANISM_NAME = f"curvature_scale_{name}"
    experiment.STAGE2_RESULT_DIR = arm_dir
    experiment.RQVAE_OUT_DIR = arm_dir / "out/rqvae/instruments"
    experiment.RQVAE_CKPT_PATH = experiment.RQVAE_OUT_DIR / "rqvae_best.pth"
    experiment.RAW_SIDS_NPY = experiment.RQVAE_OUT_DIR / "sids_raw.npy"
    experiment.SIDS_NPY = arm_dir / "dataset/Instruments/sids_for_hgrec.npy"
    experiment.ITEM_SIDS_JSON = arm_dir / "item_sids.json"
    experiment.STAGE2_LOG_DIR = arm_dir / "logs"
    experiment.GEOMETRY = geometry
    experiment.CATEGORY_CONE_ENABLED = False
    experiment.CATEGORY_CONE_METRIC_MARGIN = False

    trainer.GEOMETRY = geometry
    trainer.CATEGORY_CONE_ENABLED = False
    trainer.CATEGORY_CONE_METRIC_MARGIN = False
    # The mechanism, and the only difference from the parent.
    trainer.LAYER_CURVATURES = LAYER_CURVATURES
    # Everything below matches the parent protocol exactly, so a moved Stage3
    # number can only come from the curvature.
    trainer.QUANT_SMOOTHNESS_WEIGHT = 0.0
    trainer.CODEBOOK_SIZE = (256, 256, 256)
    trainer.BATCH_SIZE_PER_RANK = 1024
    trainer.MAX_GLOBAL_STEPS = 72_000
    trainer.EVAL_INTERVAL_STEPS = 9_000
    trainer.BEHAVIOUR_LOSS_ENABLED = True
    trainer.LOG_DIR = arm_dir / "logs"
    trainer.METRICS_PATH = trainer.LOG_DIR / "training_metrics.jsonl"
    trainer.SNAPSHOT_STEPS = {trainer.MAX_GLOBAL_STEPS: "arm"}
    trainer._LAUNCHER["script"] = os.path.abspath(__file__)
    trainer._LAUNCHER["log"] = str(trainer.LOG_DIR / "train_curvscale.log")
    return trainer


if "RANK" in os.environ:
    name = os.environ.get("CURVSCALE_ARM")
    if name not in ARMS:
        raise RuntimeError(f"CURVSCALE_ARM must name an arm, got {name!r}")
    trainer = configure(name)
    print(
        f"[curvscale] arm={name} geometry={ARMS[name]} "
        f"curvatures={trainer.LAYER_CURVATURES} codebook={trainer.CODEBOOK_SIZE} "
        f"steps={trainer.MAX_GLOBAL_STEPS}",
        flush=True,
    )
    trainer.main()
else:
    for name in ARMS:
        if _is_complete(name):
            print(f"[curvscale] arm {name} already complete, skipping", flush=True)
            continue
        print(f"[curvscale] ===== arm {name} =====", flush=True)
        trainer = configure(name)
        # launch_arm blocks and returns; _launch_via_torchrun would end the
        # process here and the second arm would never run.
        launch_arm(trainer, {"CURVSCALE_ARM": name})
    print("[curvscale] both arms finished", flush=True)
