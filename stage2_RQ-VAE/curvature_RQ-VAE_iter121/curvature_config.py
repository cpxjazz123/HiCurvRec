from pathlib import Path

REPO_ROOT = Path("/home/wlia0047/ar57/wenyu/GeneRec")
MECHANISM_NAME = "iter121_persistent_pair_decomposition"

EMBEDDING_FILE = REPO_ROOT / "stage1_GeneEmbedding/output/sentence_t5.npy"
TRAIN_FILE = REPO_ROOT / "results/stage0_build_parquet/train.parquet"

# Read-only reference: the parent's own checkpoint, used to judge whether the
# replay reproduced it and to seed the trajectory comparison.
PARENT_CKPT = (
    REPO_ROOT
    / "results/stage2_RQ-VAE/curvature_RQ-VAE/out/rqvae/instruments/rqvae_best.pth"
)

# Output paths for the replay, isolated from the parent's subtree so nothing here
# can overwrite the baseline being compared against.
REPLAY_ROOT = REPO_ROOT / "results/stage2_RQ-VAE/curvature_RQ-VAE_iter120"
STAGE2_RESULT_DIR = REPLAY_ROOT
RQVAE_OUT_DIR = REPLAY_ROOT / "out/rqvae/instruments"
RQVAE_CKPT_PATH = Path(RQVAE_OUT_DIR) / "rqvae_best.pth"
RAW_SIDS_NPY = Path(RQVAE_OUT_DIR) / "sids_raw.npy"
SIDS_NPY = STAGE2_RESULT_DIR / "dataset/Instruments/sids_for_hgrec.npy"
ITEM_SIDS_JSON = STAGE2_RESULT_DIR / "item_sids.json"
STAGE2_LOG_DIR = REPO_ROOT / "stage2_RQ-VAE/curvature_RQ-VAE_iter120/logs"
TRAJECTORY_JSON = STAGE2_LOG_DIR / "active_trajectory.json"
TRAJECTORY_LOG = STAGE2_LOG_DIR / "active_trajectory.log"

# Parent geometry, reproduced so the replay is the same condition and the
# trajectory script reads the values it measures.
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

# Unchanged TIGER-aligned budget, as the parent ran it.
MAX_GLOBAL_STEPS = 72_000
EVAL_INTERVAL_STEPS = 10_000

# The behaviour margin the parent trains against. It is the rounded median of
# d_H(A,X-) - d_H(A,B+) on the accepted parent, so roughly half the pairs begin
# inside the active region by construction rather than by failure to converge.
BEHAVIOUR_MARGIN = 0.4
BEHAVIOUR_LOSS_WEIGHT = 0.1

# The parent trainer only ever wrote its final checkpoint, so the behaviour margin
# cannot be followed over training from its artefacts. This run replays the
# identical condition with the same seed and saves the encoder at these steps. The
# last point reuses the parent's export path so the two final artefacts are
# directly comparable.
SNAPSHOT_STEP_LIST = (6_000, 18_000, 36_000, 54_000, 72_000)

# One fixed pair sample, followed through every checkpoint. Tracking identity needs
# the same pairs throughout, so this is drawn once and never resampled.
TRAJECTORY_PAIRS = 20000
TRAJECTORY_SEED = 42
# Bitwise equality is not required of the replay: GPU reductions and the Sinkhorn
# solve are not guaranteed deterministic across launches, so agreement is judged on
# the loss, the code assignment and the SID statistics landing within tolerance.
REPLAY_TOLERANCE = 1e-4
# The parent's negative is NOT a catalogue-wide draw. It is
#     negatives = encoded_source[randperm(batch)]
# so the negative for an anchor is another pair's source item from the same
# batch of loader_batch_size pairs. Every negative is therefore an item that
# somebody acted on, which is a much harder and more meaningful distractor than
# a uniformly random item, and the persistent-pair question has to be asked
# against that sampler rather than against a catalogue draw.
NEGATIVE_SAMPLER = "batch_shuffled_source"
PAIR_BATCH_SIZE = 1024
# Replicate the training shuffle this many times per anchor and average, since the
# training negative for a given pair changes every epoch.
NEGATIVE_DRAWS = 8

MECHANISM_NAME = "iter121_persistent_pair_decomposition"
REPLAY_ROOT = REPO_ROOT / "results/stage2_RQ-VAE/curvature_RQ-VAE_iter120"
DECOMP_JSON = STAGE2_LOG_DIR / "persistent_decomposition.json"
DECOMP_LOG = STAGE2_LOG_DIR / "persistent_decomposition.log"
# Pairs tracked, matching the trajectory sample so the persistent set can be
# recovered from it rather than redefined.
DECOMP_PAIRS = 20000
DECOMP_SEED = 42
# A pair counts as a weak positive or a hard negative when its distance sits this
# many robust standard deviations from the population, rather than at a fixed
# absolute distance that would mean nothing across levels.
OUTLIER_Z = 2.0
