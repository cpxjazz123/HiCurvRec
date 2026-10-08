"""Fixed real-behavior Stage2 comparison settings."""

from pathlib import Path

REPO_ROOT = Path("/home/wlia0047/ar57/wenyu/GeneRec")
EMBEDDING_FILE = REPO_ROOT / "stage1_GeneEmbedding/output/sentence_t5.npy"
TRAIN_FILE = REPO_ROOT / "results/stage0_build_parquet/train.parquet"
VALID_FILE = REPO_ROOT / "results/stage0_build_parquet/valid.parquet"
TEST_FILE = REPO_ROOT / "results/stage0_build_parquet/test.parquet"
RESULT_DIR = (
    REPO_ROOT
    / "results/stage2_RQ-VAE/curvature_RQ-VAE/behavior_node_prototype_cones"
)

SEEDS = (42, 43, 44)
GEOMETRY_ARMS = ("euclid_rq", "hyp_rq", "euclid_cone", "hyp_cone")
HIDDEN_SIZES = (512, 256, 128)
CODEBOOK_DIM = 32
CODEBOOK_SIZES = (256, 256, 256)
TRAIN_STEPS = 4_800
BATCH_SIZE = 1_024
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
GRADIENT_CLIP_NORM = 1.0

BEHAVIOR_USER_FRACTION = 0.8
COARSE_INTERESTS = 16
FINE_INTERESTS_PER_COARSE = 16
BEHAVIOR_SVD_DIM = 64
HIERARCHY_EDGES_PER_LEVEL = 32
PROTOTYPE_ITEMS = 8
HIERARCHY_LOSS_WEIGHT = 0.5
CONE_TRAIN_COVERAGE = 0.8
CONE_ANGLE_MARGIN = 0.05
CONE_RADIAL_MARGIN = 0.02
CONE_RADIAL_WEIGHT = 1.0
BEHAVIOR_SCORE_ALPHA = 5.0

BOOTSTRAP_SAMPLES = 2_000
BOOTSTRAP_SEED = 2026
