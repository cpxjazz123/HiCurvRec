from pathlib import Path

REPO_ROOT = Path("/home/wlia0047/ar57/wenyu/GeneRec")
MECHANISM_NAME = "iter54_learnable_nonlinear_curvature_controller"

EMBEDDING_FILE = REPO_ROOT / "stage1_GeneEmbedding/output/sentence_t5.npy"
TRAIN_FILE = REPO_ROOT / "results/stage0_build_parquet/train.parquet"
STAGE2_RESULT_DIR = REPO_ROOT / "results/stage2_RQ-VAE/curvature_RQ-VAE_iter54"
RQVAE_OUT_DIR = str(STAGE2_RESULT_DIR / "out/rqvae/instruments")
STAGE2_LOG_DIR = REPO_ROOT / "stage2_RQ-VAE/curvature_RQ-VAE_iter54/logs"
STAGE3_RESULT_DIR = REPO_ROOT / "results/stage3_T5Train/curvature_RQ-VAE_iter54"
ITEM_CURVATURES_PATH = STAGE2_RESULT_DIR / "item_curvatures.npy"
ITEM_SIGNALS_PATH = STAGE2_RESULT_DIR / "item_behavior_ricci_signals.npy"
ITEM_CONTROLLER_PATH = STAGE2_RESULT_DIR / "curvature_controller.pt"

MAX_GLOBAL_STEPS = 40_000
EVAL_INTERVAL_STEPS = 10_000
CURVATURE_MIN = 0.05
CURVATURE_MAX = 1.5
CURVATURE_WORKERS = 16

# The linear Iter53 reference used an equal-weight mean of the three z-scored
# signals.  These are frozen only to define the controller's initialization and
# the curvature regularizer target; the learned map supersedes the formula.
REFERENCE_CURVATURE_MEAN = 0.782913
REFERENCE_CURVATURE_STD = 0.187458

CONTROLLER_HIDDEN = 8
CONTROLLER_LR = 1e-3
CONTROLLER_BIAS_INIT = 4.9
CONTROLLER_DIVERSITY = 0.01
CONTROLLER_WEIGHT_INIT = 1.0 / 3.0

# lambda_1 anchors mean curvature; lambda_2 anchors its population standard
# deviation to the validated Iter53 distribution.
CURVATURE_MEAN_REG = 0.1
CURVATURE_STD_REG = 0.1
