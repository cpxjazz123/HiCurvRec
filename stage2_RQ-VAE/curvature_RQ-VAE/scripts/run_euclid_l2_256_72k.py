"""Euclidean RQ-VAE Stage2 run on the accepted model's own protocol.

The accepted model is hyperbolic with 256 codes per level, no category cone
and 72,000 optimizer updates.  Every euclidean arm on record was trained for
18,000 updates, so none of them can be compared to the accepted model without a
budget confound.  This run changes exactly one thing against that model, the
geometry, and leaves the budget, codebook sizes, seed, data, batch size and cone
setting untouched, so the two differ by geometry alone.
"""

import os
import sys
from pathlib import Path

SOURCE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, SOURCE_DIR)

import curvature_config as experiment

from train_rqvae import configure_run, _launch_via_torchrun, main

ARM_NAME = "euclid_l2_256_72k_s42"
ARM_DIR = Path(experiment.REPO_ROOT) / "results/stage2_RQ-VAE/curvature_RQ-VAE" / ARM_NAME

experiment.GEOMETRY = "euclid"
experiment.MECHANISM_NAME = "euclid_behaviour_context_channel"
experiment.STAGE2_RESULT_DIR = ARM_DIR
experiment.RQVAE_OUT_DIR = ARM_DIR / "out/rqvae/instruments"
experiment.RQVAE_CKPT_PATH = experiment.RQVAE_OUT_DIR / "rqvae_best.pth"
experiment.RAW_SIDS_NPY = experiment.RQVAE_OUT_DIR / "sids_raw.npy"
experiment.SIDS_NPY = ARM_DIR / "dataset/Instruments/sids_for_hgrec.npy"
experiment.ITEM_SIDS_JSON = ARM_DIR / "item_sids.json"
experiment.STAGE2_LOG_DIR = ARM_DIR / "logs"
experiment.CATEGORY_CONE_ENABLED = False

ARM_DIR.mkdir(parents=True, exist_ok=True)
configure_run(__file__)

# configure_run names its files after the production hyperbolic run; this arm is
# euclidean, so the names and the snapshot version follow the arm instead. The
# parent and the torchrun child both execute this file, so the patches below
# apply to whichever process actually trains.
trainer = sys.modules["train_rqvae"]
trainer.GEOMETRY = "euclid"
trainer.CATEGORY_CONE_ENABLED = False
trainer.LOG_DIR = ARM_DIR / "logs"
trainer.METRICS_PATH = trainer.LOG_DIR / "training_metrics.jsonl"
trainer.SNAPSHOT_STEPS = {trainer.MAX_GLOBAL_STEPS: "euclid"}
trainer._LAUNCHER["log"] = str(trainer.LOG_DIR / "train_euclid.log")

print(
    f"[euclid] arm={ARM_NAME} geometry={experiment.GEOMETRY} "
    f"codebook={trainer.CODEBOOK_SIZE} steps={trainer.MAX_GLOBAL_STEPS} "
    f"cone={experiment.CATEGORY_CONE_ENABLED} "
    f"behaviour={experiment.BEHAVIOUR_LOSS_ENABLED} out={ARM_DIR}",
    flush=True,
)

if __name__ == "__main__":
    _launch_via_torchrun()
    main()