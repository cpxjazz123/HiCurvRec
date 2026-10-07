from pathlib import Path

REPO_ROOT = Path("/home/wlia0047/ar57/wenyu/GeneRec")
MECHANISM_NAME = "iter106_codeword_radial_hierarchy"

EMBEDDING_FILE = REPO_ROOT / "stage1_GeneEmbedding/output/sentence_t5.npy"
TRAIN_FILE = REPO_ROOT / "results/stage0_build_parquet/train.parquet"

# Read-only subject: the parent fixed-curvature checkpoint. No training.
PARENT_CKPT = (
    REPO_ROOT
    / "results/stage2_RQ-VAE/curvature_RQ-VAE/out/rqvae/instruments/rqvae_best.pth"
)
STAGE2_LOG_DIR = REPO_ROOT / "stage2_RQ-VAE/curvature_RQ-VAE_iter106/logs"
DIAGNOSTIC_JSON = STAGE2_LOG_DIR / "radial_hierarchy.json"
DIAGNOSTIC_LOG = STAGE2_LOG_DIR / "radial_hierarchy.log"

# Parent geometry, reproduced here so the diagnostic reads what it measures.
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

# Behaviour pairs used for every frozen comparison, so the parent and every
# swept curvature see identical evidence.
DIAGNOSTIC_PAIRS = 20000
DIAGNOSTIC_SEED = 42
# Step 2 sweeps one level's curvature with the others pinned at the parent.
SWEEP_LEVEL = 1
SWEEP_CURVATURES = (0.5, 0.75, 1.0, 1.5, 2.0)
