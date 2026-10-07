from pathlib import Path

REPO_ROOT = Path("/home/wlia0047/ar57/wenyu/GeneRec")
MECHANISM_NAME = "iter117_l3_behavior_enrichment"

EMBEDDING_FILE = REPO_ROOT / "stage1_GeneEmbedding/output/sentence_t5.npy"
TRAIN_FILE = REPO_ROOT / "results/stage0_build_parquet/train.parquet"

PARENT_CKPT = (
    REPO_ROOT
    / "results/stage2_RQ-VAE/curvature_RQ-VAE/out/rqvae/instruments/rqvae_best.pth"
)
STAGE2_LOG_DIR = REPO_ROOT / "stage2_RQ-VAE/curvature_RQ-VAE_iter117/logs"
REPORT_JSON = STAGE2_LOG_DIR / "l3_enrichment.json"
REPORT_LOG = STAGE2_LOG_DIR / "l3_enrichment.log"

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
DIAGNOSTIC_SEED = 42

# Matched negatives are drawn from the anchor's own prefix bucket at every depth.
# At L3 this is severe: the L1+L2 prefix groups are almost singletons, so the
# script reports the feasible pool before measuring anything rather than
# substituting a weaker control to get a number.
MATCHED_NEGATIVES = 4
# Below this many pairs on either side the enrichment is reported but flagged as
# not separable from noise.
MIN_PAIRS_FOR_VERDICT = 30