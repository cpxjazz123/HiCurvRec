from pathlib import Path

REPO_ROOT = Path("/home/wlia0047/ar57/wenyu/GeneRec")
MECHANISM_NAME = "iter98_sdepth_l1_high"

EMBEDDING_FILE = REPO_ROOT / "stage1_GeneEmbedding/output/sentence_t5.npy"
TRAIN_FILE = REPO_ROOT / "results/stage0_build_parquet/train.parquet"

STAGE2_RESULT_DIR = Path(
    "/home/wlia0047/ar57/wenyu/GeneRec/results/stage2_RQ-VAE/"
    "curvature_RQ-VAE_iter98"
)
RQVAE_OUT_DIR = (
    "/home/wlia0047/ar57/wenyu/GeneRec/results/stage2_RQ-VAE/"
    "curvature_RQ-VAE_iter98/out/rqvae/instruments"
)
RQVAE_CKPT_PATH = Path(RQVAE_OUT_DIR) / "rqvae_best.pth"
RAW_SIDS_NPY = Path(RQVAE_OUT_DIR) / "sids_raw.npy"
SIDS_NPY = STAGE2_RESULT_DIR / "dataset/Instruments/sids_for_hgrec.npy"
ITEM_SIDS_JSON = STAGE2_RESULT_DIR / "item_sids.json"
STAGE2_LOG_DIR = REPO_ROOT / "stage2_RQ-VAE/curvature_RQ-VAE_iter98/logs"

# Match the TIGER baseline Stage2 budget exactly: 3000 epochs x 6 optimizer
# steps per epoch (24556 train target items / 4 ranks / batch 1024) = 18000
# optimizer updates. This trainer counts global steps as
# (single-rank step x world_size), so the equivalent budget is 18000 x 4.
MAX_GLOBAL_STEPS = 72_000
EVAL_INTERVAL_STEPS = 10_000

STAGE3_RESULT_DIR = REPO_ROOT / "results/stage3_T5Train/curvature_RQ-VAE_iter98"

