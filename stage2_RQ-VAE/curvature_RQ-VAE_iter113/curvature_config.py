from pathlib import Path

REPO_ROOT = Path("/home/wlia0047/ar57/wenyu/GeneRec")
MECHANISM_NAME = "iter113_behavior_hard_curvature_test"

EMBEDDING_FILE = REPO_ROOT / "stage1_GeneEmbedding/output/sentence_t5.npy"
TRAIN_FILE = REPO_ROOT / "results/stage0_build_parquet/train.parquet"

# Read-only subject: the parent fixed-curvature checkpoint. No training.
PARENT_CKPT = (
    REPO_ROOT
    / "results/stage2_RQ-VAE/curvature_RQ-VAE/out/rqvae/instruments/rqvae_best.pth"
)
STAGE2_LOG_DIR = REPO_ROOT / "stage2_RQ-VAE/curvature_RQ-VAE_iter113/logs"
DIAGNOSTIC_JSON = STAGE2_LOG_DIR / "behaviour_hard_curvature.json"
DIAGNOSTIC_LOG = STAGE2_LOG_DIR / "behaviour_hard_curvature.log"

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

# This is a behaviour test, not an assignment test, so it runs on the raw
# encoder latent: that is where the behaviour ranking loss lives, it carries no
# s-pin to cancel the metric factor (iter96 measured 0.13% on the pinned
# residual), and it is the only place a behaviour-hard case can be defined.
#
# Both endpoints of a pair use the same curvature. An asymmetric scheme would
# let curvature manufacture the gap rather than amplify one, which is the failure
# mode iter72 already paid for.
PROBE_CURVATURES = (0.25, 0.5, 2.0, 4.0)
# A behaviour-hard case has a positive but small margin: the model already ranks
# the true successor ahead of a random item, just not by much. Rows with a
# non-positive margin are excluded because there is no existing separation to
# amplify, and the band is the lowest quartile of the positive margins.
BEHAVIOUR_MARGIN_FLOOR_QUANTILE = 0.25
BEHAVIOUR_MARGIN_MIN = 1e-9
# Rows sampled for the test. Behaviour needs an anchor, a positive and a
# negative, so this is a separate pass from anything assignment-shaped.
BEHAVIOUR_ROWS = 50000
DIAGNOSTIC_SEED = 42
# The candidate must raise the hard band's mean margin by at least this factor
# while leaving the easy band within this factor of the parent.
HARD_MIN_GAIN_RATIO = 1.10
EASY_MAX_DRIFT_RATIO = 1.10