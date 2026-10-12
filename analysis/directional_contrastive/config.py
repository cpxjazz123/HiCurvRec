"""Hard-coded paths and constants for the directional-contrastive study."""
from pathlib import Path

REPO = Path("/home/wlia0047/ar57/wenyu/GeneRec")
TRAIN_PARQUET = REPO / "results/stage0_build_parquet/train.parquet"
VALID_PARQUET = REPO / "results/stage0_build_parquet/valid.parquet"
EMBEDDING_FILE = REPO / "stage1_GeneEmbedding/output/sentence_t5.npy"
STAGE2 = REPO / "results/stage2_RQ-VAE/TIGER_BEHAVIOR_SAMPLE_RQ-VAE"
OUT_DIR = REPO / "results/directional_contrastive"
TEMPERATURE = 0.07
LEVEL_NAMES = ("L1", "L2", "L3")
BATCH_SIZE = 512
SEED = 42
ARMS = {
    "A": {"name": "resample", "formula": "resample"},
    "B": {"name": "resample_proj_sym", "formula": "resample_proj_sym"},
    "C": {"name": "resample_proj_dir", "formula": "resample_proj_dir"},
}
