from pathlib import Path

REPO_ROOT = Path("/home/wlia0047/ar57/wenyu/GeneRec")
MECHANISM_NAME = "iter122_angular_vs_radial_decomposition"

EMBEDDING_FILE = REPO_ROOT / "stage1_GeneEmbedding/output/sentence_t5.npy"
TRAIN_FILE = REPO_ROOT / "results/stage0_build_parquet/train.parquet"

PARENT_CKPT = (
    REPO_ROOT
    / "results/stage2_RQ-VAE/curvature_RQ-VAE/out/rqvae/instruments/rqvae_best.pth"
)
STAGE2_LOG_DIR = REPO_ROOT / "stage2_RQ-VAE/curvature_RQ-VAE_iter122/logs"
DECOMP_JSON = STAGE2_LOG_DIR / "angular_vs_radial.json"
DECOMP_LOG = STAGE2_LOG_DIR / "angular_vs_radial.log"

LAYER_CURVATURES = (1.0, 1.0, 1.0)
LAYER_WORKING_RADII = (0.2, 0.2, 0.2)
PIN_IN_S_COORDINATES = True
CODEBOOK_SIZE = (256, 256, 256)
CODEBOOK_DIM = 32
HIDDEN_SIZES = (512, 256, 128)
BETA = 0.25
VQ_TYPE = "vq"
EMA_DECAY = 0.99
SK_EPSILON = 0.003
SK_ITERS = 50
BEHAVIOUR_MARGIN = 0.4
BEHAVIOUR_LOSS_WEIGHT = 0.1

DECOMP_PAIRS = 20000
DECOMP_SEED = 42
NEGATIVE_SAMPLER = "batch_shuffled_source"
PAIR_BATCH_SIZE = 1024
NEGATIVE_DRAWS = 8
# Random item pairs, so the positive and negative cosine distributions can be read
# against what an arbitrary pair looks like in the same space. A cosine near 1 is
# only informative relative to that reference, not on its own.
RANDOM_PAIRS = 20000