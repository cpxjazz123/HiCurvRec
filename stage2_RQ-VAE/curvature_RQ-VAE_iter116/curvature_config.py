from pathlib import Path

REPO_ROOT = Path("/home/wlia0047/ar57/wenyu/GeneRec")
MECHANISM_NAME = "iter116_l2_conflict_with_matched_negatives"

EMBEDDING_FILE = REPO_ROOT / "stage1_GeneEmbedding/output/sentence_t5.npy"
TRAIN_FILE = REPO_ROOT / "results/stage0_build_parquet/train.parquet"

# Read-only subject: the parent fixed-curvature checkpoint. No training.
PARENT_CKPT = (
    REPO_ROOT
    / "results/stage2_RQ-VAE/curvature_RQ-VAE/out/rqvae/instruments/rqvae_best.pth"
)
STAGE2_LOG_DIR = REPO_ROOT / "stage2_RQ-VAE/curvature_RQ-VAE_iter114/logs"
DIAGNOSTIC_JSON = STAGE2_LOG_DIR / "behavior_retention.json"
DIAGNOSTIC_LOG = STAGE2_LOG_DIR / "behavior_retention.log"

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

# Behaviour neighbours are read off the raw encoder latent, because that is where
# the behaviour ranking loss lives and where no s-pin sits between the pair and
# the metric. Retention is then measured after each stage of the encoder ->
# pin -> RQ pipeline, so the drop is attributed to a specific stage.
NEIGHBOURS_PER_ITEM = 10
# Anchors sampled for the retention read. Every anchor contributes its own
# neighbour set, so this bounds the cost directly.
ANCHOR_ROWS = 4000
# A neighbour counts as retained at level l when the two items agree on all of
# the first l codes. Agreement on a prefix is the same criterion the RQ stack
# itself uses to decide that a pair no longer needs a deeper level.
# Pairs whose neighbour cosine is at or below this are treated as weak behaviour
# and reported separately, so a retention drop can be read against how strong the
# behaviour was to begin with.
WEAK_COSINE_THRESHOLD = 0.5
DIAGNOSTIC_SEED = 42
# Real behaviour pairs, not nearest neighbours. The raw encoder latent is too
# concentrated for a nearest-neighbour definition to mean anything (every pair's
# raw cosine is ~0.998), so behaviour relevance has to come from the transition
# itself rather than from geometric proximity.
BEHAVIOUR_ROWS = 50000
SEPARATION_JSON = STAGE2_LOG_DIR / "behavior_prefix_separation.json"
SEPARATION_LOG = STAGE2_LOG_DIR / "behavior_prefix_separation.log"

SEPARATION_JSON = STAGE2_LOG_DIR / "l2_bucket_balance.json"
SEPARATION_LOG = STAGE2_LOG_DIR / "l2_bucket_balance.log"
# Pairs per L1 bucket sampled when measuring behaviour density inside a bucket.
BUCKET_PAIR_ROWS = 0  # 0 means every available pair

SEPARATION_JSON = STAGE2_LOG_DIR / "l2_conflict.json"
SEPARATION_LOG = STAGE2_LOG_DIR / "l2_conflict.log"
# How many same-bucket non-partners to draw per anchor. Two is enough to show a
# rate difference without turning the search into the dominant cost, and it keeps
# the negative independent of the anchor's own behaviour partners.
MATCHED_NEGATIVES = 2
