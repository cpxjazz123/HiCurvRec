"""The metric cone at the bar's own operating point.

The metric-cone round found the one large, unambiguous geometry effect this
project has produced: at 64 level-2 codes the euclidean arm's cone collapsed -
its held-out positive containment fell from 0.0903 at step 9000 to exactly zero
by step 27000, its loss diverged from 0.133 to 1.359, and its level-2 assignment
entropy swung between 0.16 and 0.99 - while the hyperbolic arm held 0.8551
positive containment at a stable loss of 0.1016 and finished at 0.045561 against
the euclidean arm's 0.032191. That is +41.5% on Stage3 from the metric alone,
thirteen times the three-seed spread.

Both of those arms, though, ran with 64 level-2 codes and the behaviour loss
off, and both finished below the no-cone baseline at the same codebook
(0.053918). So the geometry's advantage is established but has never been asked
to carry a result the project would keep.

This round asks exactly that. It is the same metric-cone mechanism, moved to the
production codebook (256, 256, 256) and with the behaviour loss back on, so the
two arms sit one cone away from the bar instead of two settings away. Two
contrasts come out of it and each isolates one thing:

    B_poincare - A_euclid   the geometry's effect, cone held fixed
    A_euclid   - bar        the cone's effect, geometry held fixed

If the hyperbolic advantage survives at this operating point the project has a
mechanism that beats the bar; if it does not, the advantage is confined to the
regime where the objective is unstable and that is worth knowing too.
"""

import importlib.util
import os
import sys
from pathlib import Path

PKG = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from launch_utils import launch_arm  # noqa: E402
RESULTS = PKG.parent.parent / "results/stage2_RQ-VAE/curvature_RQ-VAE/metric_cone_256_arms"
SEED = 42
ARMS = {
    "A_euclid_metriccone256": "euclid",
    "B_poincare_metriccone256": "poincare",
}


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
        "generec_train_rqvae_metric_cone_256", PKG / "train_rqvae.py"
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
    experiment.MECHANISM_NAME = f"metric_cone_256_{name}"
    experiment.STAGE2_RESULT_DIR = arm_dir
    experiment.RQVAE_OUT_DIR = arm_dir / "out/rqvae/instruments"
    experiment.RQVAE_CKPT_PATH = experiment.RQVAE_OUT_DIR / "rqvae_best.pth"
    experiment.RAW_SIDS_NPY = experiment.RQVAE_OUT_DIR / "sids_raw.npy"
    experiment.SIDS_NPY = arm_dir / "dataset/Instruments/sids_for_hgrec.npy"
    experiment.ITEM_SIDS_JSON = arm_dir / "item_sids.json"
    experiment.STAGE2_LOG_DIR = arm_dir / "logs"
    experiment.CATEGORY_CONE_ENABLED = True
    experiment.CATEGORY_CONE_METRIC_MARGIN = True

    trainer.GEOMETRY = geometry
    trainer.CATEGORY_CONE_ENABLED = True
    trainer.CATEGORY_CONE_METRIC_MARGIN = True
    # The bar's own codebook, so the arms sit one cone away from it rather than
    # two settings away.
    trainer.CODEBOOK_SIZE = (256, 256, 256)
    trainer.BATCH_SIZE_PER_RANK = 1024
    trainer.MAX_GLOBAL_STEPS = 72_000
    trainer.EVAL_INTERVAL_STEPS = 9_000
    # Back on: the bar runs it, and turning it off here would make this round
    # differ from the bar in two ways instead of one.
    trainer.BEHAVIOUR_LOSS_ENABLED = True
    trainer.LOG_DIR = arm_dir / "logs"
    trainer.METRICS_PATH = trainer.LOG_DIR / "training_metrics.jsonl"
    trainer.SNAPSHOT_STEPS = {trainer.MAX_GLOBAL_STEPS: "arm"}
    trainer._LAUNCHER["script"] = os.path.abspath(__file__)
    trainer._LAUNCHER["log"] = str(trainer.LOG_DIR / "train_metric_cone_256.log")
    return trainer


if "RANK" in os.environ:
    name = os.environ.get("CONE256_ARM")
    if name not in ARMS:
        raise RuntimeError(f"CONE256_ARM must name an arm, got {name!r}")
    trainer = configure(name)
    print(
        f"[metric-cone-256] arm={name} geometry={ARMS[name]} "
        f"codebook={trainer.CODEBOOK_SIZE} steps={trainer.MAX_GLOBAL_STEPS} "
        f"behaviour={trainer.BEHAVIOUR_LOSS_ENABLED}",
        flush=True,
    )
    trainer.main()
else:
    for name in ARMS:
        if _is_complete(name):
            print(f"[metric-cone-256] arm {name} already complete, skipping", flush=True)
            continue
        print(f"[metric-cone-256] ===== arm {name} =====", flush=True)
        trainer = configure(name)
        # launch_arm blocks and returns; _launch_via_torchrun would end the
        # process here and the second arm would never run.
        launch_arm(trainer, {"CONE256_ARM": name})
    print("[metric-cone-256] both arms finished", flush=True)
