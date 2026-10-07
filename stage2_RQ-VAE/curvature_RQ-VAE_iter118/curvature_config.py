from pathlib import Path

REPO_ROOT = Path("/home/wlia0047/ar57/wenyu/GeneRec")
MECHANISM_NAME = "iter118_virtual_coarsening"

EMBEDDING_FILE = REPO_ROOT / "stage1_GeneEmbedding/output/sentence_t5.npy"
TRAIN_FILE = REPO_ROOT / "results/stage0_build_parquet/train.parquet"

PARENT_CKPT = (
    REPO_ROOT
    / "results/stage2_RQ-VAE/curvature_RQ-VAE/out/rqvae/instruments/rqvae_best.pth"
)
STAGE2_LOG_DIR = REPO_ROOT / "stage2_RQ-VAE/curvature_RQ-VAE_iter117/logs"
REPORT_JSON = STAGE2_LOG_DIR / "coarsening.json"
REPORT_LOG = STAGE2_LOG_DIR / "coarsening.log"

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
# Virtual L2 capacities. The codewords are not retrained or re-initialised: the
# parent's 256 trained codewords are merged into groups by geometric similarity
# and each group is replaced by its member centroid, so the only thing that
# changes is how many distinct codes L2 can emit.
COARSEN_LEVEL = 1
COARSEN_CAPACITIES = (256, 128, 64, 32)
# Merge by greedy agglomeration on the tangent-frame cosine between codewords,
# which is the frame the assignment happens in. Cluster count is exactly
# COARSEN_CAPACITY; the initial grouping is by k-means on that cosine so the merge
# does not depend on codebook ordering.
COARSEN_SEED = 42
