"""Fixed, no-argument iter30 route table and serial Stage2 dispatcher."""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path("/home/wlia0047/ar57/wenyu/GeneRec")
SOURCE_ROOT = REPO_ROOT / "stage2_RQ-VAE" / "curvature_RQ-VAE_iter30"
STAGE2_RESULTS_ROOT = (
    REPO_ROOT / "results" / "stage2_RQ-VAE" / "curvature_RQ-VAE_iter30"
)
STAGE3_RESULTS_ROOT = (
    REPO_ROOT / "results" / "stage3_T5Train" / "curvature_RQ-VAE_iter30"
)
STAGE3_TRAINER = REPO_ROOT / "stage3_T5Train" / "train_HG-Rec.py"
STAGE3_TRAINER_SHA256 = "9861eb6b2a5b0bfac546524c50d402aba5827e5079a228e2ae5501909bc937cb"
WARMSTART_PATH = REPO_ROOT / "results" / "stage2_RQ-VAE" / "curvature_RQ-VAE_iter8" / "out" / "rqvae" / "instruments" / "rqvae_best.pth"
WARMSTART_SHA256 = "189e0affb156b9cdbd11f38a6fdb1127804e3f3095edd23c55f84825beb56c3b"

STAGE1_EMBEDDING = REPO_ROOT / "stage1_GeneEmbedding" / "output" / "sentence_t5.npy"
STAGE1_ITEM_IDS = REPO_ROOT / "stage1_GeneEmbedding" / "output" / "item_ids.json"
STAGE0_TRAIN = REPO_ROOT / "results" / "stage0_build_parquet" / "train.parquet"
STAGE0_TEST = REPO_ROOT / "results" / "stage0_build_parquet" / "test.parquet"
STAGE1_EMBEDDING_SHA256 = "6490c71753952b2d8788bb7e4e52cff6a5b0b59819a2cfbd737bee7bc07a77fb"
STAGE1_ITEM_IDS_SHA256 = "3df1f3ce6a468ae541c929a149f4dbd5e806f545b89d786b9c2d6801563b7c30"
STAGE0_TRAIN_SHA256 = "80597e38d081029e57e7434c92ff7be2c872f82aebc2e5f072ef973d73e88815"
STAGE0_TEST_SHA256 = "5290abf0073d798401b56bd0a9ec5f710ed2c01ae9fa4cb8bd3906c1a55f0bfc"

ITER26_MAPPING = "iter26_mapping"
ITER29_MAPPING = "iter29_mapping"
CONTROL_CURVATURE = (0.6145357379232853, 0.5333020920777128, 0.3814078098431606)
CANDIDATE_CURVATURE = (1.3660953164241916, 0.7347829661951981, 0.6439958072706683)

if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))
from curvature_config import (
    DEFAULT_RUN_LABEL,
    PROFILE_ORDER,
    PROFILE_ROUTES as CONFIG_PROFILE_ROUTES,
    RQVAE_OUT_DIR,
)

_PATH_FIELDS = frozenset(
    {
        "stage2_root",
        "rqvae_out_dir",
        "rqvae_ckpt_path",
        "raw_sids_npy",
        "sids_npy",
        "item_sids_json",
        "stage2_outer_log",
        "stage2_internal_log",
        "stage2_grad_log",
        "stage2_identity",
        "stage2_entrypoint",
        "grad_entrypoint",
        "stage3_root",
        "stage3_log_root",
        "stage3_ckpt_root",
        "stage3_launcher_log",
        "stage3_wrapper_log",
        "stage3_identity",
        "stage3_entrypoint",
    }
)
PROFILES = {}
for _label, _record in CONFIG_PROFILE_ROUTES.items():
    _profile = dict(_record)
    _profile["curvature"] = list(_record["curvature"])
    for _field in _PATH_FIELDS:
        _profile[_field] = Path(_record[_field])
    PROFILES[_label] = _profile


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_digest(path: Path, expected: str, label: str) -> str:
    if not path.is_absolute() or not path.is_file():
        raise FileNotFoundError(f"{label} is not a regular file at its absolute path: {path}")
    resolved = path.resolve(strict=True)
    digest = sha256_file(resolved)
    if digest != expected:
        raise RuntimeError(f"{label} SHA256 mismatch at {resolved}: {digest} != {expected}")
    return digest


def _path_is_below(path: Path, root: Path) -> bool:
    resolved_path = path.resolve(strict=False)
    resolved_root = root.resolve(strict=False)
    return resolved_path == resolved_root or resolved_root in resolved_path.parents



def _expected_stage2_source_files(profile: dict[str, Any]) -> tuple[Path, ...]:
    return (
        SOURCE_ROOT / "curvature_RQ-VAE.py",
        SOURCE_ROOT / "curvature_config.py",
        SOURCE_ROOT / "data" / "schemas.py",
        SOURCE_ROOT / "init" / "kmeans.py",
        SOURCE_ROOT / "modules" / "encoder.py",
        SOURCE_ROOT / "modules" / "hyperbolic.py",
        SOURCE_ROOT / "modules" / "loss.py",
        SOURCE_ROOT / "modules" / "normalize.py",
        SOURCE_ROOT / "modules" / "quantize.py",
        SOURCE_ROOT / "modules" / "rqvae.py",
        SOURCE_ROOT / "modules" / "sid_quality.py",
        SOURCE_ROOT / "modules" / "step_checks.py",
        SOURCE_ROOT / "modules" / "warm_start.py",
        SOURCE_ROOT / "scripts" / "compute_closed_form_curvature.py",
        SOURCE_ROOT / "scripts" / "computed_behavior_branching.json",
        SOURCE_ROOT / "logs" / "mechanism_contract_iter30.json",
        SOURCE_ROOT / "scripts" / "export_sids_for_stage3.py",
        SOURCE_ROOT / "scripts" / "grad_check.py",
        SOURCE_ROOT / "scripts" / "profile_routes.py",
        profile["stage2_entrypoint"],
        profile["grad_entrypoint"],
    )


def _validate_stage2_identity(profile: dict[str, Any]) -> dict[str, Any]:
    identity_path = profile["stage2_identity"]
    if not identity_path.is_file():
        raise RuntimeError(f"Stage2 route has no dispatcher identity: {identity_path}")
    identity = json.loads(identity_path.read_text(encoding="utf-8"))
    expected_fields = {
        "protocol_id": "FCCR-1_iter30_iter26_vs_iter29_matched_seed43-45",
        "profile": profile["label"],
        "mapping_id": profile["mapping_id"],
        "curvature": profile["curvature"],
        "seed_stage2": profile["seed"],
        "seed_stage3": profile["seed"],
        "mechanism_name": profile["mechanism_name"],
        "stage2_root": str(profile["stage2_root"]),
        "stage3_root": str(profile["stage3_root"]),
    }
    for key, expected in expected_fields.items():
        if identity.get(key) != expected:
            raise RuntimeError(
                f"Stage2 identity mismatch for {profile['label']}: {key}"
            )
    expected_input_paths = {
        "stage1_sentence_t5": str(STAGE1_EMBEDDING.resolve(strict=True)),
        "stage1_item_ids": str(STAGE1_ITEM_IDS.resolve(strict=True)),
        "stage0_train": str(STAGE0_TRAIN.resolve(strict=True)),
        "iter8_warmstart": str(WARMSTART_PATH.resolve(strict=True)),
    }
    expected_input_hashes = {
        "stage1_sentence_t5": STAGE1_EMBEDDING_SHA256,
        "stage1_item_ids": STAGE1_ITEM_IDS_SHA256,
        "stage0_train": STAGE0_TRAIN_SHA256,
        "iter8_warmstart": WARMSTART_SHA256,
    }
    if identity.get("input_paths") != expected_input_paths:
        raise RuntimeError(f"Stage2 input paths changed for {profile['label']}")
    if identity.get("input_sha256") != expected_input_hashes:
        raise RuntimeError(f"Stage2 input hashes changed for {profile['label']}")
    expected_consumers = [
        str(STAGE1_EMBEDDING),
        str(STAGE1_ITEM_IDS),
        str(STAGE0_TRAIN),
        str(WARMSTART_PATH),
    ]
    expected_non_consumers = [
        str(REPO_ROOT / "results" / "stage0_build_parquet" / "valid.parquet"),
        str(STAGE0_TEST),
        str(REPO_ROOT / "results" / "stage0_build_parquet" / "items.parquet"),
    ]
    if identity.get("stage2_actual_consumers") != expected_consumers:
        raise RuntimeError(f"Stage2 consumer set changed for {profile['label']}")
    if identity.get("stage2_non_consumers") != expected_non_consumers:
        raise RuntimeError(f"Stage2 non-consumer set changed for {profile['label']}")
    destinations = identity.get("prelaunch_destination_inventory")
    expected_root_keys = {
        str(profile["stage2_root"]),
        str(profile["stage3_root"]),
    }
    expected_file_keys = {
        str(profile["raw_sids_npy"]),
        str(profile["sids_npy"]),
        str(profile["item_sids_json"]),
        str(profile["stage2_outer_log"]),
        str(profile["stage2_internal_log"]),
        str(profile["stage2_grad_log"]),
        str(profile["stage2_identity"]),
        str(profile["stage3_launcher_log"]),
        str(profile["stage3_wrapper_log"]),
        str(profile["stage3_identity"]),
    }
    if not isinstance(destinations, dict) or set(destinations) != (
        expected_root_keys | expected_file_keys
    ):
        raise RuntimeError(f"Stage2 destination inventory is incomplete for {profile['label']}")
    for root_key in expected_root_keys:
        record = destinations[root_key]
        if (
            not isinstance(record, dict)
            or type(record.get("exists")) is not bool
            or record.get("entries") != []
        ):
            raise RuntimeError(f"Stage2 prelaunch root was not absent/empty: {root_key}")
    for file_key in expected_file_keys:
        if destinations[file_key] != {"exists": False}:
            raise RuntimeError(f"Stage2 prelaunch destination was not absent: {file_key}")

    expected_source_hashes = {
        str(path): sha256_file(path) for path in _expected_stage2_source_files(profile)
    }
    if identity.get("source_sha256") != expected_source_hashes:
        raise RuntimeError(f"Stage2 source identity changed for {profile['label']}")
    return identity

def validate_profiles() -> None:
    if DEFAULT_RUN_LABEL != PROFILE_ORDER[0]:
        raise RuntimeError(f"iter30 default profile changed: {DEFAULT_RUN_LABEL}")
    if Path(RQVAE_OUT_DIR) != PROFILES[DEFAULT_RUN_LABEL]["rqvae_out_dir"]:
        raise RuntimeError("static RQVAE_OUT_DIR is not the first registered Stage2 route")
    if tuple(PROFILES) != PROFILE_ORDER:
        raise RuntimeError(f"iter30 profile set/order changed: {tuple(PROFILES)}")
    seen_paths: set[Path] = set()
    for label in PROFILE_ORDER:
        profile = PROFILES[label]
        seed = int(label.rsplit("seed", 1)[1])
        arm = label.rsplit("_seed", 1)[0]
        if profile["label"] != label or profile["seed"] != seed:
            raise RuntimeError(f"iter30 profile identity mismatch: {label}")
        if profile["mapping_id"] != arm:
            raise RuntimeError(f"iter30 mapping identity mismatch: {label}")
        if profile["mechanism_name"] != f"iter30_{arm}_seed{seed}":
            raise RuntimeError(f"iter30 mechanism label mismatch: {label}")
        if profile["stage2_root"] != STAGE2_RESULTS_ROOT / label:
            raise RuntimeError(f"Stage2 route is not its exact short root: {label}")
        if profile["stage3_root"] != STAGE3_RESULTS_ROOT / label:
            raise RuntimeError(f"Stage3 route is not its exact short root: {label}")
        if profile["mapping_id"] == ITER26_MAPPING:
            if profile["curvature"] != list(CONTROL_CURVATURE):
                raise RuntimeError(f"iter26 registered vector mismatch: {label}")
        elif profile["mapping_id"] == ITER29_MAPPING:
            if profile["curvature"] != list(CANDIDATE_CURVATURE):
                raise RuntimeError(f"iter29 registered vector mismatch: {label}")
        else:
            raise RuntimeError(f"unknown iter30 map: {profile['mapping_id']}")
        for key in (
            "stage2_root", "rqvae_out_dir", "rqvae_ckpt_path", "raw_sids_npy",
            "sids_npy", "item_sids_json", "stage2_outer_log", "stage2_internal_log",
            "stage2_grad_log", "stage2_identity", "stage2_entrypoint", "grad_entrypoint",
            "stage3_root", "stage3_log_root", "stage3_ckpt_root", "stage3_launcher_log",
            "stage3_wrapper_log", "stage3_identity", "stage3_entrypoint",
        ):
            path = profile[key]
            if not isinstance(path, Path) or not path.is_absolute():
                raise RuntimeError(f"{label} {key} is not an absolute Path")
            if path in seen_paths:
                raise RuntimeError(f"iter30 route collision at {path}")
            seen_paths.add(path)
        if _path_is_below(profile["rqvae_out_dir"], SOURCE_ROOT):
            raise RuntimeError(f"Stage2 products point into the source subtree: {label}")
        if profile["rqvae_out_dir"] != profile["stage2_root"] / "out" / "rqvae" / "instruments":
            raise RuntimeError(f"Stage2 RQVAE_OUT_DIR convention mismatch: {label}")
        if profile["rqvae_ckpt_path"] != profile["rqvae_out_dir"] / "rqvae_best.pth":
            raise RuntimeError(f"Stage2 checkpoint path mismatch: {label}")
        if profile["raw_sids_npy"] != profile["rqvae_out_dir"] / "sids_raw.npy":
            raise RuntimeError(f"Stage2 raw SID path mismatch: {label}")
        if profile["sids_npy"] != profile["stage2_root"] / "dataset" / "Instruments" / "sids_for_hgrec.npy":
            raise RuntimeError(f"Stage2 exported SID path mismatch: {label}")
        if profile["item_sids_json"] != profile["stage2_root"] / "item_sids.json":
            raise RuntimeError(f"Stage2 JSON path mismatch: {label}")
        expected_routes = {
            "stage2_outer_log": SOURCE_ROOT / "logs" / f"train_run_{label}.log",
            "stage2_internal_log": SOURCE_ROOT / "logs" / f"train_migrated_{label}.log",
            "stage2_grad_log": SOURCE_ROOT / "logs" / f"grad_check_{label}.log",
            "stage2_identity": SOURCE_ROOT / "logs" / f"run_identity_{label}.json",
            "stage2_entrypoint": SOURCE_ROOT / "scripts" / f"run_stage2_{label}.py",
            "grad_entrypoint": SOURCE_ROOT / "scripts" / f"grad_check_{label}.py",
            "stage3_log_root": profile["stage3_root"] / "logs",
            "stage3_ckpt_root": profile["stage3_root"] / "ckpt",
            "stage3_launcher_log": profile["stage3_root"] / "logs" / "_stage3_launcher.log",
            "stage3_wrapper_log": profile["stage3_root"] / "logs" / "stage3_wrapper_stdout.log",
            "stage3_identity": profile["stage3_root"] / "logs" / "run_identity.json",
            "stage3_entrypoint": SOURCE_ROOT / "scripts" / f"run_stage3_{label}.py",
        }
        for key, expected in expected_routes.items():
            if profile[key] != expected:
                raise RuntimeError(f"iter30 {key} route mismatch: {label}")


def get_profile(label: str) -> dict[str, Any]:
    validate_profiles()
    if label not in PROFILES:
        raise ValueError(f"unregistered iter30 profile: {label}")
    return dict(PROFILES[label])


def _directory_inventory(path: Path) -> dict[str, Any]:
    if path.is_symlink():
        raise RuntimeError(f"Refusing symlink output root: {path}")
    if not path.exists():
        return {"exists": False, "entries": []}
    if not path.is_dir():
        raise RuntimeError(f"Expected output root is not a directory: {path}")
    entries = sorted(str(child.name) for child in path.iterdir())
    if entries:
        raise FileExistsError(f"Refusing nonempty output root {path}: {entries}")
    return {"exists": True, "entries": entries}


def _require_absent(path: Path) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"Refusing to overwrite existing route: {path}")


def prepare_stage2_run(profile: dict[str, Any]) -> dict[str, Any]:
    validate_profiles()
    label = profile["label"]
    if profile != PROFILES[label]:
        raise RuntimeError(f"profile record was altered before dispatch: {label}")
    _directory_inventory(profile["stage2_root"])
    _directory_inventory(profile["stage3_root"])
    for path in (
        profile["raw_sids_npy"], profile["sids_npy"], profile["item_sids_json"],
        profile["stage2_outer_log"], profile["stage2_internal_log"],
        profile["stage2_grad_log"], profile["stage2_identity"],
        profile["stage3_launcher_log"], profile["stage3_wrapper_log"],
        profile["stage3_identity"],
    ):
        _require_absent(path)

    input_hashes = {
        "stage1_sentence_t5": _require_digest(
            STAGE1_EMBEDDING, STAGE1_EMBEDDING_SHA256, "Stage1 embedding"
        ),
        "stage1_item_ids": _require_digest(
            STAGE1_ITEM_IDS, STAGE1_ITEM_IDS_SHA256, "Stage1 item-ID sidecar"
        ),
        "stage0_train": _require_digest(
            STAGE0_TRAIN, STAGE0_TRAIN_SHA256, "Stage0 Stage2 train parquet"
        ),
        "iter8_warmstart": _require_digest(
            WARMSTART_PATH, WARMSTART_SHA256, "iter8 warm-start"
        ),
    }
    input_paths = {
        "stage1_sentence_t5": str(STAGE1_EMBEDDING.resolve(strict=True)),
        "stage1_item_ids": str(STAGE1_ITEM_IDS.resolve(strict=True)),
        "stage0_train": str(STAGE0_TRAIN.resolve(strict=True)),
        "iter8_warmstart": str(WARMSTART_PATH.resolve(strict=True)),
    }
    source_hashes = {
        str(path): sha256_file(path)
        for path in _expected_stage2_source_files(profile)
    }
    destinations = {
        str(profile["stage2_root"]): _directory_inventory(profile["stage2_root"]),
        str(profile["stage3_root"]): _directory_inventory(profile["stage3_root"]),
        str(profile["raw_sids_npy"]): {"exists": profile["raw_sids_npy"].exists()},
        str(profile["sids_npy"]): {"exists": profile["sids_npy"].exists()},
        str(profile["item_sids_json"]): {"exists": profile["item_sids_json"].exists()},
        str(profile["stage2_outer_log"]): {"exists": profile["stage2_outer_log"].exists()},
        str(profile["stage2_internal_log"]): {"exists": profile["stage2_internal_log"].exists()},
        str(profile["stage2_grad_log"]): {"exists": profile["stage2_grad_log"].exists()},
        str(profile["stage2_identity"]): {"exists": profile["stage2_identity"].exists()},
        str(profile["stage3_launcher_log"]): {"exists": profile["stage3_launcher_log"].exists()},
        str(profile["stage3_wrapper_log"]): {"exists": profile["stage3_wrapper_log"].exists()},
        str(profile["stage3_identity"]): {"exists": profile["stage3_identity"].exists()},
    }
    identity = {
        "protocol_id": "FCCR-1_iter30_iter26_vs_iter29_matched_seed43-45",
        "profile": label,
        "mapping_id": profile["mapping_id"],
        "curvature": profile["curvature"],
        "seed_stage2": profile["seed"],
        "seed_stage3": profile["seed"],
        "mechanism_name": profile["mechanism_name"],
        "stage2_root": str(profile["stage2_root"]),
        "stage3_root": str(profile["stage3_root"]),
        "input_paths": input_paths,
        "input_sha256": input_hashes,
        "source_sha256": source_hashes,
        "prelaunch_destination_inventory": destinations,
        "stage2_actual_consumers": [
            str(STAGE1_EMBEDDING),
            str(STAGE1_ITEM_IDS),
            str(STAGE0_TRAIN),
            str(WARMSTART_PATH),
        ],
        "stage2_non_consumers": [
            str(REPO_ROOT / "results" / "stage0_build_parquet" / "valid.parquet"),
            str(STAGE0_TEST),
            str(REPO_ROOT / "results" / "stage0_build_parquet" / "items.parquet"),
        ],
    }
    profile["stage2_identity"].write_text(
        json.dumps(identity, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return identity


def _load_training_module():
    if str(SOURCE_ROOT) not in sys.path:
        sys.path.insert(0, str(SOURCE_ROOT))
    scripts_dir = str(SOURCE_ROOT / "scripts")
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    import importlib.util

    trainer_path = SOURCE_ROOT / "curvature_RQ-VAE.py"
    spec = importlib.util.spec_from_file_location(
        "iter30_shared_rqvae_trainer", trainer_path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load common Stage2 trainer: {trainer_path}")
    trainer = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = trainer
    spec.loader.exec_module(trainer)
    return trainer


def _validate_stage2_child_authorization(profile: dict[str, Any]) -> None:
    label = profile["label"]
    _validate_stage2_identity(profile)
    if not profile["stage2_outer_log"].is_file():
        raise RuntimeError(f"Stage2 dispatcher outer log is missing: {label}")
    outer_report = profile["stage2_outer_log"].read_text(encoding="utf-8")
    if f"STAGE2_PROFILE_START label={label} " not in outer_report:
        raise RuntimeError(f"Stage2 dispatcher did not authorize {label}")
    grad_log = profile["stage2_grad_log"]
    if not grad_log.is_file():
        raise RuntimeError(f"Stage2 per-profile gradient gate is missing: {label}")
    report = grad_log.read_text(encoding="utf-8")
    if "STAGE2_GRADIENT_GATE_PASS" not in report or f'"profile": "{label}"' not in report:
        raise RuntimeError(f"Stage2 per-profile gradient gate did not pass for {label}")
    _directory_inventory(profile["stage2_root"])
    _directory_inventory(profile["stage3_root"])
    for path in (
        profile["rqvae_ckpt_path"],
        profile["raw_sids_npy"],
        profile["sids_npy"],
        profile["item_sids_json"],
    ):
        _require_absent(path)


def run_stage2_profile(label: str, entrypoint: str) -> None:
    """Run one literal profile only as a root-dispatcher-launched DDP worker."""
    profile = get_profile(label)
    wrapper_path = Path(entrypoint).resolve(strict=True)
    if wrapper_path != profile["stage2_entrypoint"].resolve(strict=True):
        raise RuntimeError(f"Stage2 wrapper/profile mismatch: {wrapper_path} != {label}")
    if "RANK" not in os.environ:
        raise RuntimeError(
            "Stage2 route workers may only be started by curvature_RQ-VAE.py"
        )
    trainer = _load_training_module()
    trainer._apply_profile(profile)
    _validate_stage2_child_authorization(profile)
    try:
        trainer.main(profile)
    finally:
        trainer.cleanup_distributed()


def dispatch_profiles(trainer) -> None:
    """Run all six literal route gates and Stage2 jobs serially."""
    trainer_path = Path(trainer.__file__).resolve(strict=True)
    if trainer_path != (SOURCE_ROOT / "curvature_RQ-VAE.py").resolve(strict=True):
        raise RuntimeError(f"unexpected Stage2 dispatcher module: {trainer_path}")
    validate_profiles()
    for label in PROFILE_ORDER:
        profile = get_profile(label)
        trainer._apply_profile(profile)
        trainer._LAUNCHER["script"] = str(profile["stage2_entrypoint"])
        trainer._LAUNCHER["log"] = str(profile["stage2_internal_log"])
        trainer._LAUNCHER["nproc"] = 4
        trainer._LAUNCHER["master_port"] = 50200
        prepare_stage2_run(profile)
        with profile["stage2_outer_log"].open("xb") as outer_log:
            from contextlib import redirect_stderr, redirect_stdout

            with redirect_stdout(outer_log), redirect_stderr(outer_log):
                print(
                    f"STAGE2_PROFILE_START label={label} "
                    f"mapping={profile['mapping_id']} seed={profile['seed']} "
                    f"entrypoint={profile['stage2_entrypoint']}",
                    flush=True,
                )
                with profile["stage2_grad_log"].open("xb") as grad_log:
                    subprocess.run(
                        [sys.executable, str(profile["grad_entrypoint"])],
                        cwd=str(SOURCE_ROOT),
                        stdout=grad_log,
                        stderr=subprocess.STDOUT,
                        check=True,
                    )
                return_code = trainer._launch_via_torchrun()
                if return_code != 0:
                    raise subprocess.CalledProcessError(return_code, trainer._LAUNCHER["script"])
                print(f"STAGE2_PROFILE_COMPLETE label={label}", flush=True)
