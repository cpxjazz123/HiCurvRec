from pathlib import Path

REPO_ROOT = Path("/home/wlia0047/ar57/wenyu/GeneRec")
MECHANISM_NAME = "iter96_behavior_ranking_curvature_estimator"

EMBEDDING_FILE = REPO_ROOT / "stage1_GeneEmbedding/output/sentence_t5.npy"
TRAIN_FILE = REPO_ROOT / "results/stage0_build_parquet/train.parquet"

# Iter96 is a frozen probe: nothing is trained, so the parent's checkpoint is
# the read-only subject and the only output is the estimator's own report.
PARENT_CKPT = (
    REPO_ROOT
    / "results/stage2_RQ-VAE/curvature_RQ-VAE/out/rqvae/instruments/rqvae_best.pth"
)
STAGE2_RESULT_DIR = REPO_ROOT / "results/stage2_RQ-VAE/curvature_RQ-VAE_iter96"
STAGE2_LOG_DIR = REPO_ROOT / "stage2_RQ-VAE/curvature_RQ-VAE_iter96/logs"
ESTIMATOR_JSON = STAGE2_LOG_DIR / "curvature_estimate.json"
ESTIMATOR_LOG = STAGE2_LOG_DIR / "curvature_estimate.log"

STAGE3_RESULT_DIR = REPO_ROOT / "results/stage3_T5Train/curvature_RQ-VAE_iter96"

# === Frozen-probe search (iter96) ===
# Same three-point bracket as iter95 so the two estimates are comparable.
ESTIMATOR_CURVATURES = (0.9, 1.0, 1.1)
# Negatives per positive pair. Held fixed across curvature candidates so the
# only thing that changes between them is the metric.
ESTIMATOR_NEGATIVES_PER_PAIR = 20
ESTIMATOR_BOOTSTRAP_SAMPLES = 2000
ESTIMATOR_BOOTSTRAP_SEED = 42
# 1.96 * standard error, so a candidate must win by more than the sampling noise
# of the very pair population that supports the claim.
ESTIMATOR_CONFIDENCE_Z = 1.96
