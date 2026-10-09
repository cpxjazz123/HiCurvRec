from pathlib import Path

REPO_ROOT = Path("/home/wlia0047/ar57/wenyu/GeneRec")
MECHANISM_NAME = "poincare_behaviour_context_channel"

# Geometry plug-in for the RQ-VAE: "poincare" is the frozen protocol, "euclid"
# is its flat limit (identity maps, Euclidean distances, tangent subtraction).
# Everything else in the model and trainer is shared, so a two-arm comparison
# differs only in this switch.
GEOMETRY = "poincare"

# ---------------------------------------------------------------------------
# Quantized category cone supervision. Round one: category-balanced random
# negatives shared by both arms, no prefix loss, no Stage3.
# ---------------------------------------------------------------------------
# Per-level assignment rule. "bucket" is the frozen protocol: the previous
# level's code buckets the balancing. Setting the second entry to "global" or
# "argmin" keeps L1 and L3 untouched and only relaxes L2, which is the level
# that decides how many items share an L1+L2 prefix.
LAYER_ASSIGNMENT_MODES = ("bucket", "bucket", "bucket")

# Category prototype radius band, relative to the initial radius, measured in
# the arm's own point space. (0.6, 1.4) is what rounds four and five ran with
# and the Euclidean arm drove straight to the upper edge; (1.0, 1.0) pins the
# radius so that only the direction of a prototype can move, which is the
# controlled comparison against that escape.
CATEGORY_CONE_RADIUS_BAND = (0.6, 1.4)

CATEGORY_CONE_ENABLED = False
CATEGORY_CONE_WEIGHT = 0.5
CATEGORY_CONE_MARGIN = 0.01
# Initial cone aperture, in degrees, measured at the median apex radius. The
# target cannot be expressed as an initial coverage share: at initialisation a
# prototype sits nowhere near the items it must contain, so no aperture reaches a
# high share and the fit pins to its cap, leaving a half space that constrains
# nothing. Fixing the angle puts both arms at the same initial cone width; K, the
# achieved coverage and max(K*g) are reported per run so a saturated or
# degenerate cone is visible.
CATEGORY_CONE_APERTURE_DEG = 20.0
CATEGORY_CONE_RADIAL_WEIGHT = 1.0
CATEGORY_CONE_RADIAL_MARGIN = 0.02
CATEGORY_CONE_HOLDOUT_FRACTION = 0.10
CATEGORY_CONE_DATA_SEED = 2026
# Measure the cone margin in the arm's own metric instead of in radians. A cone
# is an angular object and the Poincare ball is conformal, so the same aperture
# selects the same points in either arm; the distance from a point to the cone
# boundary is the angular gap times the conformal factor, which grows without
# bound near the ball's boundary. Weighting each hinge by that factor is what
# makes the two arms differ, and for euclid the factor is identically one.
CATEGORY_CONE_METRIC_MARGIN = False
# A gap required between different fine categories' cones, measured in the
# arm's own metric. Containment alone is satisfied by widening cones until
# they overlap, and overlapping cones share codes; this term is what asks
# for the fine-category discriminability Stage3 depends on. Zero disables it.
CATEGORY_CONE_SEPARATION_WEIGHT = 0.0
CATEGORY_CONE_SEPARATION_GAP = 0.0
# Give each level the cone of the granularity it has to discriminate: level 1
# the coarse category, level 2 the fine one. The default gives level 2 the
# coarse cone, which merges every fine category under one coarse category
# into a single cone and is what cost the round-four arms their fine-code
# discriminability.
CATEGORY_CONE_LEVEL_ALIGNED = False
# Which level the coarse cone supervises. 2 is the historical setting and the
# source of the merge this program measured; 1 moves the coarse cone onto the
# level that already carries the category, leaving level 2 to the reconstruction
# objective so the L1L2 prefix keeps the item identity that makes it unique.
CATEGORY_CONE_COARSE_LEVEL = 2
# Weight on keeping the quantiser a smooth function of the input in the arm's
# metric: latents the metric calls close should not be sent to code vectors the
CATEGORY_CONE_COARSE_LEVEL = 2
# Weight on keeping the quantiser a smooth function of the input in the arm's
# metric, so that Stage3's content-to-SID map is easier to fit. It changes no
# partition and creates no sharing. Zero disables it.
QUANT_SMOOTHNESS_WEIGHT = 0.0
CATEGORY_CONE_CALIBRATION_ITEMS = 8_192
# The behaviour-context channel is the previous mechanism. The four-arm cone
# comparison is self-contained, so it is switched off in every arm; the cone
# arms are still compared against it as the parent condition at accept time.
BEHAVIOUR_LOSS_ENABLED = True

EMBEDDING_FILE = REPO_ROOT / "stage1_GeneEmbedding/output/sentence_t5.npy"
TRAIN_FILE = REPO_ROOT / "results/stage0_build_parquet/train.parquet"

STAGE2_RESULT_DIR = Path(
    "/home/wlia0047/ar57/wenyu/GeneRec/results/stage2_RQ-VAE/"
    "curvature_RQ-VAE"
)
RQVAE_OUT_DIR = (
    "/home/wlia0047/ar57/wenyu/GeneRec/results/stage2_RQ-VAE/"
    "curvature_RQ-VAE/out/rqvae/instruments"
)
RQVAE_CKPT_PATH = Path(RQVAE_OUT_DIR) / "rqvae_best.pth"
RAW_SIDS_NPY = Path(RQVAE_OUT_DIR) / "sids_raw.npy"
SIDS_NPY = STAGE2_RESULT_DIR / "dataset/Instruments/sids_for_hgrec.npy"
ITEM_SIDS_JSON = STAGE2_RESULT_DIR / "item_sids.json"
STAGE2_LOG_DIR = REPO_ROOT / "stage2_RQ-VAE/curvature_RQ-VAE/logs"

# Match the TIGER baseline Stage2 budget exactly: 3000 epochs x 6 optimizer
# steps per epoch (24556 train target items / 4 ranks / batch 1024) = 18000
# optimizer updates. This trainer counts global steps as
# (single-rank step x world_size), so the equivalent budget is 18000 x 4.
MAX_GLOBAL_STEPS = 72_000
# The full-corpus SID pass is a descriptive diagnostic: it re-quantizes every
# item and recomputes the usage/collision statistics, and it never feeds a
# gradient or a gate. 10k steps ran it eight times per 72k-step budget; 36k
# keeps a mid-run reading and the final snapshot point for a quarter of the
# cost. The budget, the loss and the assignment rule are untouched.
EVAL_INTERVAL_STEPS = 36_000

STAGE3_RESULT_DIR = REPO_ROOT / "results/stage3_T5Train/curvature_RQ-VAE"

