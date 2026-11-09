"""Metric smoothness, restricted to pairs that share a preceding code.

The smoothness term is the only mechanism this project has that comes back
positive: at seed 42 the euclidean arm reads 0.060499 against the bar's 0.059158
and the hyperbolic arm 0.059507. Both are stable - neither loss diverges - so the
hyperbolic geometry's one demonstrated advantage, staying stable under
supervision strong enough to blow the euclidean arm up, is dormant here.

This round changes where the pairs come from rather than how hard the term
pushes. The global term draws its pairs from the batch at large, so most of them
cross preceding codes and constrain nothing about the hierarchy. The prefix term
draws every pair from inside one preceding-code bucket, so the same weight is
spent on the comparisons that decide whether two items sharing a prefix keep
sharing structure further down. The weights are exactly the arms' existing ones
(0.34 euclidean, 1.0 hyperbolic), so the only change is the scope.

Two contrasts come out of it:

    B_poincare - A_euclid   the geometry's effect, scope held fixed
    A_euclid   - SE         the scope's effect, geometry held fixed

The bar to beat is SE's 0.060499. A hyperbolic arm above it would be the first
time the geometry carried a result this project would keep.
"""

import importlib.util
import os
import sys
from pathlib import Path

PKG = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from launch_utils import launch_arm  # noqa: E402
RESULTS = PKG.parent.parent / "results/stage2_RQ-VAE/curvature_RQ-VAE/prefix_smooth_arms"
SEED = 42
ARMS = {
    "A_euclid_prefixsmooth": "euclid",
    "B_poincare_prefixsmooth": "poincare",
}
# Identical to the smoothness round's weights: the scope is the only change.
SMOOTHNESS_WEIGHT = {"poincare": 1.0, "euclid": 0.34}
SMOOTHNESS_SCOPE = "prefix"


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
        "generec_train_rqvae_prefix_smooth", PKG / "train_rqvae.py"
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
    experiment.MECHANISM_NAME = f"prefix_smooth_{name}"
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
    # The mechanism, and the only difference from the smoothness round.
    trainer.QUANT_SMOOTHNESS_SCOPE = SMOOTHNESS_SCOPE
    trainer.QUANT_SMOOTHNESS_WEIGHT = SMOOTHNESS_WEIGHT[geometry]
    # Everything below matches the smoothness arms exactly, so a moved Stage3
    # number can only come from the scope.
    trainer.CODEBOOK_SIZE = (256, 256, 256)
    trainer.BATCH_SIZE_PER_RANK = 1024
    trainer.MAX_GLOBAL_STEPS = 72_000
    trainer.EVAL_INTERVAL_STEPS = 9_000
    trainer.BEHAVIOUR_LOSS_ENABLED = True
    trainer.LOG_DIR = arm_dir / "logs"
    trainer.METRICS_PATH = trainer.LOG_DIR / "training_metrics.jsonl"
    trainer.SNAPSHOT_STEPS = {trainer.MAX_GLOBAL_STEPS: "arm"}
    trainer._LAUNCHER["script"] = os.path.abspath(__file__)
    trainer._LAUNCHER["log"] = str(trainer.LOG_DIR / "train_prefix_smooth.log")
    return trainer


if "RANK" in os.environ:
    name = os.environ.get("PREFIXSMOOTH_ARM")
    if name not in ARMS:
        raise RuntimeError(f"PREFIXSMOOTH_ARM must name an arm, got {name!r}")
    trainer = configure(name)
    print(
        f"[prefix-smooth] arm={name} geometry={ARMS[name]} "
        f"scope={trainer.QUANT_SMOOTHNESS_SCOPE} "
        f"weight={trainer.QUANT_SMOOTHNESS_WEIGHT} "
        f"codebook={trainer.CODEBOOK_SIZE} steps={trainer.MAX_GLOBAL_STEPS}",
        flush=True,
    )
    trainer.main()
else:
    for name in ARMS:
        if _is_complete(name):
            print(f"[prefix-smooth] arm {name} already complete, skipping", flush=True)
            continue
        print(f"[prefix-smooth] ===== arm {name} =====", flush=True)
        trainer = configure(name)
        # launch_arm blocks and returns; _launch_via_torchrun would end the
        # process here and the second arm would never run.
        launch_arm(trainer, {"PREFIXSMOOTH_ARM": name})
    print("[prefix-smooth] both arms finished", flush=True)
