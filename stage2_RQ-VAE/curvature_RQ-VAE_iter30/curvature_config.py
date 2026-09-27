"""Hardcoded iter30 Stage2 inputs and six immutable execution routes."""
import os

REPO_ROOT = "/home/wlia0047/ar57/wenyu/GeneRec"
SOURCE_ROOT = os.path.join(REPO_ROOT, "stage2_RQ-VAE", "curvature_RQ-VAE_iter30")
STAGE2_RESULTS_ROOT = os.path.join(
    REPO_ROOT, "results", "stage2_RQ-VAE", "curvature_RQ-VAE_iter30"
)
STAGE3_RESULTS_ROOT = os.path.join(
    REPO_ROOT, "results", "stage3_T5Train", "curvature_RQ-VAE_iter30"
)
PROFILE_ORDER = (
    "iter26_mapping_seed43",
    "iter29_mapping_seed43",
    "iter26_mapping_seed44",
    "iter29_mapping_seed44",
    "iter26_mapping_seed45",
    "iter29_mapping_seed45",
)


def _make_profile(label, mapping_id, curvature, seed):
    stage2_root = os.path.join(STAGE2_RESULTS_ROOT, label)
    stage3_root = os.path.join(STAGE3_RESULTS_ROOT, label)
    log_root = os.path.join(SOURCE_ROOT, "logs")
    stage2_out = os.path.join(stage2_root, "out", "rqvae", "instruments")
    stage3_log_root = os.path.join(stage3_root, "logs")
    mechanism_name = f"iter30_{mapping_id}_seed{seed}"
    return {
        "label": label,
        "mapping_id": mapping_id,
        "curvature": tuple(curvature),
        "seed": seed,
        "mechanism_name": mechanism_name,
        "stage2_root": stage2_root,
        "rqvae_out_dir": stage2_out,
        "rqvae_ckpt_path": os.path.join(stage2_out, "rqvae_best.pth"),
        "raw_sids_npy": os.path.join(stage2_out, "sids_raw.npy"),
        "sids_npy": os.path.join(
            stage2_root, "dataset", "Instruments", "sids_for_hgrec.npy"
        ),
        "item_sids_json": os.path.join(stage2_root, "item_sids.json"),
        "stage2_outer_log": os.path.join(log_root, f"train_run_{label}.log"),
        "stage2_internal_log": os.path.join(log_root, f"train_migrated_{label}.log"),
        "stage2_grad_log": os.path.join(log_root, f"grad_check_{label}.log"),
        "stage2_identity": os.path.join(log_root, f"run_identity_{label}.json"),
        "stage2_entrypoint": os.path.join(SOURCE_ROOT, "scripts", f"run_stage2_{label}.py"),
        "grad_entrypoint": os.path.join(SOURCE_ROOT, "scripts", f"grad_check_{label}.py"),
        "stage3_root": stage3_root,
        "stage3_log_root": stage3_log_root,
        "stage3_ckpt_root": os.path.join(stage3_root, "ckpt"),
        "stage3_launcher_log": os.path.join(stage3_log_root, "_stage3_launcher.log"),
        "stage3_wrapper_log": os.path.join(stage3_log_root, "stage3_wrapper_stdout.log"),
        "stage3_identity": os.path.join(stage3_log_root, "run_identity.json"),
        "stage3_entrypoint": os.path.join(SOURCE_ROOT, "scripts", f"run_stage3_{label}.py"),
    }


PROFILE_ROUTES = {
    "iter26_mapping_seed43": _make_profile(
        "iter26_mapping_seed43", "iter26_mapping",
        (0.6145357379232853, 0.5333020920777128, 0.3814078098431606), 43,
    ),
    "iter29_mapping_seed43": _make_profile(
        "iter29_mapping_seed43", "iter29_mapping",
        (1.3660953164241916, 0.7347829661951981, 0.6439958072706683), 43,
    ),
    "iter26_mapping_seed44": _make_profile(
        "iter26_mapping_seed44", "iter26_mapping",
        (0.6145357379232853, 0.5333020920777128, 0.3814078098431606), 44,
    ),
    "iter29_mapping_seed44": _make_profile(
        "iter29_mapping_seed44", "iter29_mapping",
        (1.3660953164241916, 0.7347829661951981, 0.6439958072706683), 44,
    ),
    "iter26_mapping_seed45": _make_profile(
        "iter26_mapping_seed45", "iter26_mapping",
        (0.6145357379232853, 0.5333020920777128, 0.3814078098431606), 45,
    ),
    "iter29_mapping_seed45": _make_profile(
        "iter29_mapping_seed45", "iter29_mapping",
        (1.3660953164241916, 0.7347829661951981, 0.6439958072706683), 45,
    ),
}

DEFAULT_RUN_LABEL = "iter26_mapping_seed43"
DEFAULT_PROFILE = PROFILE_ROUTES[DEFAULT_RUN_LABEL]
DEFAULT_RUN_ROOT = DEFAULT_PROFILE["stage2_root"]

# Keep the exact literal path visible to the repository's pre-launch path check.
RQVAE_OUT_DIR = (
    "/home/wlia0047/ar57/wenyu/GeneRec/results/stage2_RQ-VAE/"
    "curvature_RQ-VAE_iter30/iter26_mapping_seed43/out/rqvae/instruments/"
)
RQVAE_CKPT_PATH = DEFAULT_PROFILE["rqvae_ckpt_path"]
RAW_SIDS_NPY = DEFAULT_PROFILE["raw_sids_npy"]
SIDS_NPY = DEFAULT_PROFILE["sids_npy"]
ITEM_SIDS_JSON = DEFAULT_PROFILE["item_sids_json"]
STAGE2_OUTER_LOG = DEFAULT_PROFILE["stage2_outer_log"]
STAGE2_INTERNAL_LOG = DEFAULT_PROFILE["stage2_internal_log"]
RUN_LABEL = "iter26_mapping_seed43"
MAPPING_ID = "iter26_mapping"
MECHANISM_NAME = "iter30_iter26_mapping_seed43"
SEED = 43

# Shared upstream Stage0/Stage1 inputs; all consumers use these absolute paths.
DATASET_ROOT = os.path.join(REPO_ROOT, "dataset")
STAGE0_DIR = os.path.join(REPO_ROOT, "results", "stage0_build_parquet")
STAGE1_OUT_DIR = os.path.join(REPO_ROOT, "stage1_GeneEmbedding", "output")
ITEM_JSON_PATH = os.path.join(
    DATASET_ROOT, "Amazon_2023_Instruments", "Instruments.item.json"
)
ITEM_EMB_NPY = os.path.join(STAGE1_OUT_DIR, "sentence_t5.npy")
ITEM_IDS_JSON = os.path.join(STAGE1_OUT_DIR, "item_ids.json")
TRAIN_PARQUET = os.path.join(STAGE0_DIR, "train.parquet")
VALID_PARQUET = os.path.join(STAGE0_DIR, "valid.parquet")
TEST_PARQUET = os.path.join(STAGE0_DIR, "test.parquet")
ITEMS_PARQUET = os.path.join(STAGE0_DIR, "items.parquet")
PARQUET_DIR = STAGE0_DIR

WARMSTART_PATH = (
    "/home/wlia0047/ar57/wenyu/GeneRec/results/stage2_RQ-VAE/"
    "curvature_RQ-VAE_iter8/out/rqvae/instruments/rqvae_best.pth"
)
WARMSTART_SHA256 = (
    "189e0affb156b9cdbd11f38a6fdb1127804e3f3095edd23c55f84825beb56c3b"
)

# Existing project-wide deterministic/runtime/cache settings, hardcoded identically.
os.environ["PYTHONHASHSEED"] = "42"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ["HF_HOME"] = "/home/wlia0047/hj82_scratch2/wenyu/.cache/huggingface"
os.environ["HUGGINGFACE_HUB_CACHE"] = (
    "/home/wlia0047/hj82_scratch2/wenyu/hf_models/hub"
)
os.environ["TRANSFORMERS_CACHE"] = (
    "/home/wlia0047/hj82_scratch2/wenyu/hf_models/hub"
)
CUDA_VISIBLE_DEVICES = "0,1,2,3"
os.environ["CUDA_VISIBLE_DEVICES"] = CUDA_VISIBLE_DEVICES
os.environ["USE_TF"] = "0"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
