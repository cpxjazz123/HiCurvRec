from pathlib import Path

REPO_ROOT = Path("/home/wlia0047/ar57/wenyu/GeneRec")
MECHANISM_NAME = "iter125_multi_negative_counterfactual"

EMBEDDING_FILE = REPO_ROOT / "stage1_GeneEmbedding/output/sentence_t5.npy"
TRAIN_FILE = REPO_ROOT / "results/stage0_build_parquet/train.parquet"

PARENT_CKPT = (
    REPO_ROOT
    / "results/stage2_RQ-VAE/curvature_RQ-VAE/out/rqvae/instruments/rqvae_best.pth"
)
STAGE2_LOG_DIR = REPO_ROOT / "stage2_RQ-VAE/curvature_RQ-VAE_iter125/logs"
DECOMP_JSON = STAGE2_LOG_DIR / "multi_negative.json"
DECOMP_LOG = STAGE2_LOG_DIR / "multi_negative.log"

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
BETA_QUANT = 0.25

DECOMP_PAIRS = 20000
DECOMP_SEED = 42
NEGATIVE_SAMPLER = "batch_shuffled_source"
PAIR_BATCH_SIZE = 1024
NEGATIVE_DRAWS = 8
MIN_TRANSITIONS_PER_ANCHOR = 2
CONFLICT_ANCHORS = 6000
CONFLICT_NEGATIVES_PER_ANCHOR = 2

# One-step counterfactual settings.
# Negatives per anchor for the InfoNCE denominator. Drawn with the parent's own
# shuffled-source logic so the negative sampling is not changed at the same time
# as the objective.
NCE_NEGATIVES = 16
NCE_TEMPERATURE = 0.1
# Both objective gradients are normalised to unit length before the probe step, so
# the comparison measures the direction each objective proposes rather than its
# magnitude. Without this the larger-gradient objective would win trivially.
STEP_SIZE = 0.01