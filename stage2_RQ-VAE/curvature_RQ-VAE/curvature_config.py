from pathlib import Path

REPO_ROOT = Path("/home/wlia0047/ar57/wenyu/GeneRec")
MECHANISM_NAME = "poincare_behaviour_context_channel"

EMBEDDING_FILE = REPO_ROOT / "stage1_GeneEmbedding/output/sentence_t5.npy"
TRAIN_FILE = REPO_ROOT / "results/stage0_build_parquet/train.parquet"

STAGE2_RESULT_DIR = Path(
    "/home/wlia0047/ar57/wenyu/GeneRec/results/stage2_RQ-VAE/"
    "curvature_RQ-VAE"
)
RQVAE_OUT_DIR = (
    "/home/wlia0047/ar57/wenyu/GeneRec/results/stage2_RQ-VAE/"
    "curvature_RQ-VAE/out/rqvae/instruments"
)
RQVAE_CKPT_PATH = Path(RQVAE_OUT_DIR) / "rqvae_best.pth"
RAW_SIDS_NPY = Path(RQVAE_OUT_DIR) / "sids_raw.npy"
SIDS_NPY = STAGE2_RESULT_DIR / "dataset/Instruments/sids_for_hgrec.npy"
ITEM_SIDS_JSON = STAGE2_RESULT_DIR / "item_sids.json"
STAGE2_LOG_DIR = REPO_ROOT / "stage2_RQ-VAE/curvature_RQ-VAE/logs"

# Match the TIGER baseline Stage2 budget exactly: 3000 epochs x 6 optimizer
# steps per epoch (24556 train target items / 4 ranks / batch 1024) = 18000
# optimizer updates. This trainer counts global steps as
# (single-rank step x world_size), so the equivalent budget is 18000 x 4.
MAX_GLOBAL_STEPS = 72_000
# The full-corpus SID pass is a descriptive diagnostic: it re-quantizes every
# item and recomputes the usage/collision statistics, and it never feeds a
# gradient or a gate. 10k steps ran it eight times per 72k-step budget; 36k
# keeps a mid-run reading and the final snapshot point for a quarter of the
# cost. The budget, the loss and the assignment rule are untouched.
EVAL_INTERVAL_STEPS = 36_000

STAGE3_RESULT_DIR = REPO_ROOT / "results/stage3_T5Train/curvature_RQ-VAE"

