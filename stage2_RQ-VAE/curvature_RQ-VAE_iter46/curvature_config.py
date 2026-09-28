from pathlib import Path

REPO_ROOT = Path("/home/wlia0047/ar57/wenyu/GeneRec")
MECHANISM_NAME = "iter46_geodesic_sinkhorn"

EMBEDDING_FILE = REPO_ROOT / "stage1_GeneEmbedding/output/sentence_t5.npy"
TRAIN_FILE = REPO_ROOT / "results/stage0_build_parquet/train.parquet"

STAGE2_RESULT_DIR = Path(
    "/home/wlia0047/ar57/wenyu/GeneRec/results/stage2_RQ-VAE/"
    "curvature_RQ-VAE_iter46"
)
RQVAE_OUT_DIR = (
    "/home/wlia0047/ar57/wenyu/GeneRec/results/stage2_RQ-VAE/"
    "curvature_RQ-VAE_iter46/out/rqvae/instruments"
)
RQVAE_CKPT_PATH = Path(RQVAE_OUT_DIR) / "rqvae_best.pth"
RAW_SIDS_NPY = Path(RQVAE_OUT_DIR) / "sids_raw.npy"
SIDS_NPY = STAGE2_RESULT_DIR / "dataset/Instruments/sids_for_hgrec.npy"
ITEM_SIDS_JSON = STAGE2_RESULT_DIR / "item_sids.json"
STAGE2_LOG_DIR = REPO_ROOT / "stage2_RQ-VAE/curvature_RQ-VAE_iter46/logs"

# Match the other iterations' fixed 100k distributed-step budget.
MAX_GLOBAL_STEPS = 100_000
EVAL_INTERVAL_STEPS = 10_000

STAGE3_RESULT_DIR = REPO_ROOT / "results/stage3_T5Train/curvature_RQ-VAE_iter46"

