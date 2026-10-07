from pathlib import Path

REPO_ROOT = Path("/home/wlia0047/ar57/wenyu/GeneRec")
MECHANISM_NAME = "iter112_selective_curvature_hard_cases"

EMBEDDING_FILE = REPO_ROOT / "stage1_GeneEmbedding/output/sentence_t5.npy"
TRAIN_FILE = REPO_ROOT / "results/stage0_build_parquet/train.parquet"

# Read-only subject: the parent fixed-curvature checkpoint. No training.
PARENT_CKPT = (
    REPO_ROOT
    / "results/stage2_RQ-VAE/curvature_RQ-VAE/out/rqvae/instruments/rqvae_best.pth"
)
STAGE2_LOG_DIR = REPO_ROOT / "stage2_RQ-VAE/curvature_RQ-VAE_iter112/logs"
DIAGNOSTIC_JSON = STAGE2_LOG_DIR / "selective_curvature.json"
DIAGNOSTIC_LOG = STAGE2_LOG_DIR / "selective_curvature.log"

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

# L2 carries the only codebook with real radial freedom: L1's KMeans fit pins its
# codewords onto the 0.2 shell (spread 0.0009) and L3's residual curvature was
# measured inert (iter110). So the probe runs at L2.
PROBE_LEVEL = 1
# Curvatures to compare. 4 and 8 are left out on purpose: c = 2 already moves the
# hard-case gaps by an order of magnitude, so there is no reason to reach for
# more before that is understood.
PROBE_CURVATURES = (0.5, 1.0, 2.0)
# Hard cases are the rows whose parent top1-top2 gap is at or below this
# quantile of the corpus at that level. 0.10 was chosen as the smallest band
# that still holds enough rows to measure a behaviour gap on.
HARD_QUANTILE = 0.10
# Sub-sample size for the behaviour read. Behaviour needs row ids for both
# endpoints, so it runs on a separate, smaller pass.
BEHAVIOUR_ROWS = 20000
DIAGNOSTIC_SEED = 42