"""Run the locked iter80 Stage2 condition."""
import os
import sys

SOURCE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, SOURCE_DIR)

import curvature_config as experiment
import train_rqvae as training

EXPECTED_OUT_DIR = (
    "/home/wlia0047/ar57/wenyu/GeneRec/results/stage2_RQ-VAE/"
    "curvature_RQ-VAE_iter80/out/rqvae/instruments"
)
if experiment.RQVAE_OUT_DIR != EXPECTED_OUT_DIR:
    raise RuntimeError(f"RQVAE_OUT_DIR must be {EXPECTED_OUT_DIR}")
if experiment.MAX_GLOBAL_STEPS != 72_000:
    raise RuntimeError("Stage2 budget must remain 72,000 global steps")
if (
    training.LAYER_CURVATURES != (1.0, 1.0, 1.0)
    or training.LAYER_WORKING_RADII != (0.2, 0.2, 0.2)
):
    raise RuntimeError("iter80 must retain parent curvature and all three pins")

training.configure_run(__file__)

if __name__ == "__main__":
    training._launch_via_torchrun()
    training.main()
