from pathlib import Path

REPO_ROOT = Path("/home/wlia0047/ar57/wenyu/GeneRec")
MECHANISM_NAME = "iter61_topology_gated_mixed_geometry"

EMBEDDING_FILE = REPO_ROOT / "stage1_GeneEmbedding/output/sentence_t5.npy"
TRAIN_FILE = REPO_ROOT / "results/stage0_build_parquet/train.parquet"
STAGE2_RESULT_DIR = REPO_ROOT / "results/stage2_RQ-VAE/curvature_RQ-VAE_iter61"
RQVAE_OUT_DIR = str(STAGE2_RESULT_DIR / "out/rqvae/instruments")
STAGE2_LOG_DIR = REPO_ROOT / "stage2_RQ-VAE/curvature_RQ-VAE_iter61/logs"
STAGE3_RESULT_DIR = REPO_ROOT / "results/stage3_T5Train/curvature_RQ-VAE_iter61"
ITEM_CURVATURES_PATH = STAGE2_RESULT_DIR / "item_curvatures.npy"
ITEM_SIGNALS_PATH = STAGE2_RESULT_DIR / "item_behavior_ricci_signals.npy"
PARENT_ITEM_CURVATURES_PATH = REPO_ROOT / "results/stage2_RQ-VAE/curvature_RQ-VAE_iter53/item_curvatures.npy"
PARENT_ITEM_SIGNALS_PATH = REPO_ROOT / "results/stage2_RQ-VAE/curvature_RQ-VAE_iter53/item_behavior_ricci_signals.npy"

MAX_GLOBAL_STEPS = 40_000
EVAL_INTERVAL_STEPS = 10_000
CURVATURE_MIN = 0.05
CURVATURE_MAX = 1.5
RICCI_WEIGHT_R = 1.0 / 3.0
RICCI_WEIGHT_G = 1.0 / 3.0
RICCI_WEIGHT_H = 1.0 / 3.0
