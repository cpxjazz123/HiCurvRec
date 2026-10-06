from pathlib import Path

REPO_ROOT = Path("/home/wlia0047/ar57/wenyu/GeneRec")
MECHANISM_NAME = "iter95_behavior_separation_curvature_estimator"

EMBEDDING_FILE = REPO_ROOT / "stage1_GeneEmbedding/output/sentence_t5.npy"
TRAIN_FILE = REPO_ROOT / "results/stage0_build_parquet/train.parquet"

# Iter95 is a frozen probe: nothing is trained, so the parent's checkpoint is
# the read-only subject and the only output is the estimator's own report.
PARENT_CKPT = (
    REPO_ROOT
    / "results/stage2_RQ-VAE/curvature_RQ-VAE/out/rqvae/instruments/rqvae_best.pth"
)
STAGE2_RESULT_DIR = REPO_ROOT / "results/stage2_RQ-VAE/curvature_RQ-VAE_iter95"
STAGE2_LOG_DIR = REPO_ROOT / "stage2_RQ-VAE/curvature_RQ-VAE_iter95/logs"
ESTIMATOR_JSON = STAGE2_LOG_DIR / "curvature_estimate.json"
ESTIMATOR_LOG = STAGE2_LOG_DIR / "curvature_estimate.log"

STAGE3_RESULT_DIR = REPO_ROOT / "results/stage3_T5Train/curvature_RQ-VAE_iter95"

# === Frozen-probe search (iter95 phase 1) ===
# Three points around the parent's curvature. With these the first derivative
# over the 0.2-wide bracket and the second derivative over the 0.1 half-steps
# both divide out exactly, so the reported g and h are unbiased central
# differences at the parent's curvature rather than finite-difference leftovers.
ESTIMATOR_CURVATURES = (0.9, 1.0, 1.1)
ESTIMATOR_BOOTSTRAP_SAMPLES = 2000
ESTIMATOR_BOOTSTRAP_SEED = 42
# 1.96 * standard error, so a candidate must beat the parent by more than the
# sampling noise of the very pair population that supports the claim.
ESTIMATOR_CONFIDENCE_Z = 1.96
# Quantization safety gate. A candidate is refused when the collision rate rises
# or the mean assignment margin falls by more than this fraction relative to the
# parent; behaviour separation never overrides those.
ESTIMATOR_MAX_COLLISION_INCREASE = 0.0
ESTIMATOR_MIN_MARGIN_RETENTION = 1.0
