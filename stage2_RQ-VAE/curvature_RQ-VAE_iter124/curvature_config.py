from pathlib import Path

REPO_ROOT = Path("/home/wlia0047/ar57/wenyu/GeneRec")
MECHANISM_NAME = "iter124_behaviour_self_conflict"

EMBEDDING_FILE = REPO_ROOT / "stage1_GeneEmbedding/output/sentence_t5.npy"
TRAIN_FILE = REPO_ROOT / "results/stage0_build_parquet/train.parquet"

PARENT_CKPT = (
    REPO_ROOT
    / "results/stage2_RQ-VAE/curvature_RQ-VAE/out/rqvae/instruments/rqvae_best.pth"
)
STAGE2_LOG_DIR = REPO_ROOT / "stage2_RQ-VAE/curvature_RQ-VAE_iter124/logs"
DECOMP_JSON = STAGE2_LOG_DIR / "self_conflict.json"
DECOMP_LOG = STAGE2_LOG_DIR / "self_conflict.log"

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

# Anchors that carry more than one behaviour transition are where two
# constraints can act on the same latent at once. The parent's mean out-degree is
# 0.813, so this is a minority of anchors but a large group of pairs.
MIN_TRANSITIONS_PER_ANCHOR = 2
# Anchors sampled for the conflict measurement. Only anchors with enough
# transitions contribute, so this bounds the work, not the usable population.
CONFLICT_ANCHORS = 6000
# The parent's negative for a pair is another pair's source from the same batch,
# so each transition gets its own negative here in the same way.
CONFLICT_NEGATIVES_PER_ANCHOR = 2