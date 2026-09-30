from pathlib import Path

REPO_ROOT = Path("/home/wlia0047/ar57/wenyu/GeneRec")
MECHANISM_NAME = "iter63_pure_iter51_hyperbolic_rqvae"

EMBEDDING_FILE = REPO_ROOT / "stage1_GeneEmbedding/output/sentence_t5.npy"
TRAIN_FILE = REPO_ROOT / "results/stage0_build_parquet/train.parquet"
STAGE2_RESULT_DIR = REPO_ROOT / "results/stage2_RQ-VAE/curvature_RQ-VAE_iter63"
RQVAE_OUT_DIR = str(STAGE2_RESULT_DIR / "out/rqvae/instruments")
STAGE2_LOG_DIR = REPO_ROOT / "stage2_RQ-VAE/curvature_RQ-VAE_iter63/logs"
STAGE3_RESULT_DIR = REPO_ROOT / "results/stage3_T5Train/curvature_RQ-VAE_iter63"

MAX_GLOBAL_STEPS = 40_000
EVAL_INTERVAL_STEPS = 10_000
