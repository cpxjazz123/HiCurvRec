from pathlib import Path

REPO_ROOT = Path("/home/wlia0047/ar57/wenyu/GeneRec")
MECHANISM_NAME = "iter110_residual_curvature_diagnostic"

EMBEDDING_FILE = REPO_ROOT / "stage1_GeneEmbedding/output/sentence_t5.npy"
TRAIN_FILE = REPO_ROOT / "results/stage0_build_parquet/train.parquet"

# Read-only subject: the parent fixed-curvature checkpoint. No training.
PARENT_CKPT = (
    REPO_ROOT
    / "results/stage2_RQ-VAE/curvature_RQ-VAE/out/rqvae/instruments/rqvae_best.pth"
)
STAGE2_LOG_DIR = REPO_ROOT / "stage2_RQ-VAE/curvature_RQ-VAE_iter110/logs"
DIAGNOSTIC_JSON = STAGE2_LOG_DIR / "residual_curvature.json"
DIAGNOSTIC_LOG = STAGE2_LOG_DIR / "residual_curvature.log"

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

DIAGNOSTIC_PAIRS = 20000
DIAGNOSTIC_SEED = 42
# One residual transition moves at a time. c_res,1 controls the L1 subtraction
# that produces r_2, c_res,2 the L2 subtraction that produces r_3. The L3
# subtraction is not swept: nothing downstream consumes r_4, so its effect could
# not be measured here.
SWEEP_VALUES = (0.5, 0.75, 1.0, 1.5, 2.0)
SWEEP_TARGETS = (0, 1)
# A candidate must clear all three to reach training. Behaviour may not drop by
# more than this, and recomposition may not worsen by more than this fraction.
BEHAVIOUR_MAX_DROP = 0.02
RECOMPOSITION_MAX_GROWTH = 0.05
# Above this the assignment is churning rather than responding.
MAX_ASSIGNMENT_FLIP = 0.30
