from pathlib import Path

REPO_ROOT = Path("/home/wlia0047/ar57/wenyu/GeneRec")
MECHANISM_NAME = "iter123_gradient_conflict"

EMBEDDING_FILE = REPO_ROOT / "stage1_GeneEmbedding/output/sentence_t5.npy"
TRAIN_FILE = REPO_ROOT / "results/stage0_build_parquet/train.parquet"

PARENT_CKPT = (
    REPO_ROOT
    / "results/stage2_RQ-VAE/curvature_RQ-VAE/out/rqvae/instruments/rqvae_best.pth"
)
STAGE2_LOG_DIR = REPO_ROOT / "stage2_RQ-VAE/curvature_RQ-VAE_iter123/logs"
DECOMP_JSON = STAGE2_LOG_DIR / "gradient_conflict.json"
DECOMP_LOG = STAGE2_LOG_DIR / "gradient_conflict.log"

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

# The parent trainer's weighting, read here so the two gradients are separated
# exactly as training combines them rather than as an unweighted ideal would.
BETA_QUANT = 0.25

DECOMP_PAIRS = 20000
DECOMP_SEED = 42
NEGATIVE_SAMPLER = "batch_shuffled_source"
PAIR_BATCH_SIZE = 1024
NEGATIVE_DRAWS = 8
# Number of separate reconstruction batches whose gradient is averaged, so the
# reconstruction side is not a single noisy batch of 20000.
RECON_BATCHES = 8
RECON_BATCH = 1024