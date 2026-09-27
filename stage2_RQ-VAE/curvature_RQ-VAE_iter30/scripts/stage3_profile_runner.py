"""Strict Stage3 runner for one literal iter30 profile."""
from __future__ import annotations

import contextlib
import importlib.util
import json
import os
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
SOURCE_ROOT = SCRIPT_DIR.parent
REPO_ROOT = SOURCE_ROOT.parent.parent
STAGE3_DIR = REPO_ROOT / "stage3_T5Train"
for import_root in (SOURCE_ROOT, SCRIPT_DIR, STAGE3_DIR):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from profile_routes import (
    STAGE0_TEST,
    STAGE0_TEST_SHA256,
    STAGE0_TRAIN,
    STAGE0_TRAIN_SHA256,
    STAGE2_RESULTS_ROOT,
    STAGE3_RESULTS_ROOT,
    STAGE3_TRAINER,
    STAGE3_TRAINER_SHA256,
    _directory_inventory,
    _path_is_below,
    _require_absent,
    _require_digest,
    _validate_stage2_identity,
    get_profile,
    sha256_file,
)


def _expected_stage3_source_files(profile):
    return (
        STAGE3_TRAINER,
        SCRIPT_DIR / "stage3_profile_runner.py",
        profile["stage3_entrypoint"],
        SOURCE_ROOT / "scripts" / "profile_routes.py",
        SOURCE_ROOT / "curvature_config.py",
    )


def _require_log_marker(path: Path, marker: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"required Stage2 log is missing: {path}")
    with path.open("r", encoding="utf-8", errors="replace") as stream:
        if not any(marker in line for line in stream):
            raise RuntimeError(f"required completion marker missing from {path}: {marker}")



def _validate_stage2_sid_payload(payload: object) -> dict:
    expected_keys = [str(item_id) for item_id in range(24_587)]
    if not isinstance(payload, dict) or list(payload) != expected_keys:
        raise RuntimeError("Stage2 SID JSON keys must be dense ordered item IDs")
    occurrence_by_triple = {}
    unique_rows = set()
    for item_id, key in enumerate(expected_keys):
        row = payload[key]
        if (
            not isinstance(row, list)
            or len(row) != 4
            or any(type(value) is not int for value in row)
        ):
            raise RuntimeError(f"Stage2 SID row {item_id} must contain four integer tokens")
        if any(value < 0 or value >= 256 for value in row[:3]):
            raise RuntimeError(f"Stage2 SID row {item_id} exceeds a 256-code layer")
        triple = tuple(row[:3])
        occurrence = occurrence_by_triple.get(triple, 0)
        if row[3] != 768 + occurrence:
            raise RuntimeError(
                f"Stage2 SID row {item_id} has an invalid collision extension"
            )
        occurrence_by_triple[triple] = occurrence + 1
        unique_rows.add(tuple(row))
    if len(unique_rows) != 24_587:
        raise RuntimeError("Stage2 four-token SID JSON contains collisions")
    return payload

def _require_stage2_exports(profile):
    for path in (
        profile["rqvae_ckpt_path"],
        profile["raw_sids_npy"],
        profile["sids_npy"],
        profile["item_sids_json"],
    ):
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(f"required Stage2 export missing/empty: {path}")
    resolved_stage2_root = profile["stage2_root"].resolve(strict=True)
    expected_json = resolved_stage2_root / "item_sids.json"
    if profile["item_sids_json"].resolve(strict=True) != expected_json:
        raise RuntimeError("Stage3 SID JSON is not the matching Stage2 profile export")
    if resolved_stage2_root.parent != STAGE2_RESULTS_ROOT.resolve(strict=True):
        raise RuntimeError("Stage3 SID JSON is outside the registered Stage2 short root")
    _validate_stage2_identity(profile)
    _require_log_marker(
        profile["stage2_outer_log"],
        f"STAGE2_PROFILE_COMPLETE label={profile['label']}",
    )
    _require_log_marker(
        profile["stage2_grad_log"],
        "STAGE2_GRADIENT_GATE_PASS",
    )
    payload = json.loads(profile["item_sids_json"].read_text(encoding="utf-8"))
    return _validate_stage2_sid_payload(payload)


def _load_trainer():
    spec = importlib.util.spec_from_file_location(
        "iter30_unchanged_stage3_trainer", STAGE3_TRAINER
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load Stage3 trainer: {STAGE3_TRAINER}")
    trainer = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = trainer
    spec.loader.exec_module(trainer)
    return trainer




def _prepare_stage3_run(profile):
    label = profile["label"]
    if profile["stage3_root"] != STAGE3_RESULTS_ROOT / label:
        raise RuntimeError(f"Stage3 output route is not its exact short root: {label}")
    if profile["stage3_root"].name != label:
        raise RuntimeError(f"Stage3 short-root mismatch: {label}")
    _directory_inventory(profile["stage3_root"])
    _require_absent(profile["stage3_wrapper_log"])
    _require_absent(profile["stage3_launcher_log"])
    _require_absent(profile["stage3_identity"])
    for path in (
        profile["stage2_root"],
        profile["rqvae_ckpt_path"],
        profile["raw_sids_npy"],
        profile["sids_npy"],
        profile["item_sids_json"],
    ):
        if not path.exists():
            raise FileNotFoundError(f"Stage2 output missing before Stage3: {path}")
    sid_payload = _require_stage2_exports(profile)
    if len(sid_payload) != 24_587:
        raise RuntimeError(
            f"Stage2 item_sids.json item count mismatch: {len(sid_payload)} != 24587"
        )
    input_hashes = {
        "stage0_train": _require_digest(
            STAGE0_TRAIN, STAGE0_TRAIN_SHA256, "Stage0 Stage3 train parquet"
        ),
        "stage0_test": _require_digest(
            STAGE0_TEST, STAGE0_TEST_SHA256, "Stage0 Stage3 test parquet"
        ),
        "stage3_trainer": _require_digest(
            STAGE3_TRAINER, STAGE3_TRAINER_SHA256, "unchanged Stage3 trainer"
        ),
        "stage2_item_sids_json": sha256_file(profile["item_sids_json"]),
        "stage2_best_checkpoint": sha256_file(profile["rqvae_ckpt_path"]),
        "stage2_raw_sids": sha256_file(profile["raw_sids_npy"]),
        "stage2_exported_sids": sha256_file(profile["sids_npy"]),
    }
    source_hashes = {
        str(path): sha256_file(path) for path in _expected_stage3_source_files(profile)
    }
    identity = {
        "protocol_id": "FCCR-1_iter30_iter26_vs_iter29_matched_seed43-45",
        "profile": label,
        "mapping_id": profile["mapping_id"],
        "mechanism_name": profile["mechanism_name"],
        "seed_stage2": profile["seed"],
        "seed_stage3": profile["seed"],
        "stage2_root": str(profile["stage2_root"]),
        "stage3_root": str(profile["stage3_root"]),
        "code_path": str(profile["item_sids_json"].resolve(strict=True)),
        "stage3_trainer": str(STAGE3_TRAINER.resolve(strict=True)),
        "stage2_identity_sha256": sha256_file(profile["stage2_identity"]),
        "stage3_source_sha256": source_hashes,
        "stage3_trainer_sha256": input_hashes["stage3_trainer"],
        "stage0_input_sha256": {
            "train.parquet": input_hashes["stage0_train"],
            "test.parquet": input_hashes["stage0_test"],
        },
        "stage2_output_sha256": {
            key: value
            for key, value in input_hashes.items()
            if key.startswith("stage2_")
        },
        "stage2_item_count": len(sid_payload),
        "stage3_protocol": {
            "seed": profile["seed"],
            "num_epochs": 150,
            "beam_size": 20,
            "topk_list": [5, 10],
            "no_eval": True,
            "skip_test": False,
            "nproc_per_node": 4,
            "master_port": 50201,
            "n_eval_expected": 57_439,
            "batch_size": 4096,
            "inference_batch_size": 1024,
        },
        "prelaunch_destination_inventory": {
            str(profile["stage3_root"]): _directory_inventory(profile["stage3_root"]),
            str(profile["stage3_launcher_log"]): {"exists": False},
            str(profile["stage3_wrapper_log"]): {"exists": False},
            str(profile["stage3_identity"]): {"exists": False},
        },
    }
    profile["stage3_log_root"].mkdir(parents=True, exist_ok=True)
    profile["stage3_ckpt_root"].mkdir(parents=True, exist_ok=True)
    profile["stage3_identity"].write_text(
        json.dumps(identity, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return identity


def _configure_trainer(trainer, profile):
    sid_path = profile["item_sids_json"].resolve(strict=True)
    stage2_root = profile["stage2_root"].resolve(strict=True)
    if sid_path.parent != stage2_root:
        raise RuntimeError("Stage3 CODE_PATH must be inside its paired Stage2 short root")
    identity_path = profile["stage3_identity"]
    if not identity_path.is_file():
        raise RuntimeError(f"Stage3 dispatcher identity is missing: {identity_path}")
    identity = json.loads(identity_path.read_text(encoding="utf-8"))
    for key, expected in (
        ("protocol_id", "FCCR-1_iter30_iter26_vs_iter29_matched_seed43-45"),
        ("profile", profile["label"]),
        ("mapping_id", profile["mapping_id"]),
        ("mechanism_name", profile["mechanism_name"]),
        ("seed_stage2", profile["seed"]),
        ("seed_stage3", profile["seed"]),
        ("stage2_root", str(stage2_root)),
        ("stage3_root", str(profile["stage3_root"])),
        ("code_path", str(sid_path)),
    ):
        if identity.get(key) != expected:
            raise RuntimeError(f"Stage3 identity mismatch for {profile['label']}: {key}")
    expected_sid_digest = identity["stage2_output_sha256"]["stage2_item_sids_json"]
    if sha256_file(sid_path) != expected_sid_digest:
        raise RuntimeError("Stage2 item_sids.json changed after Stage3 preflight")
    if sha256_file(profile["stage2_identity"]) != identity.get("stage2_identity_sha256"):
        raise RuntimeError("Stage2 dispatcher identity changed after Stage3 preflight")
    expected_sources = {
        str(path): sha256_file(path) for path in _expected_stage3_source_files(profile)
    }
    if identity.get("stage3_source_sha256") != expected_sources:
        raise RuntimeError("Stage3 source identity changed after Stage3 preflight")
    expected_protocol = {
        "seed": profile["seed"],
        "num_epochs": 150,
        "beam_size": 20,
        "topk_list": [5, 10],
        "no_eval": True,
        "skip_test": False,
        "nproc_per_node": 4,
        "master_port": 50201,
        "n_eval_expected": 57_439,
        "batch_size": 4096,
        "inference_batch_size": 1024,
    }
    if identity.get("stage3_protocol") != expected_protocol:
        raise RuntimeError("Stage3 identity does not match the locked protocol")
    if sha256_file(STAGE3_TRAINER) != STAGE3_TRAINER_SHA256:
        raise RuntimeError("Stage3 trainer changed after Stage3 preflight")

    expected_data_root = (REPO_ROOT / "results" / "stage0_build_parquet").resolve(
        strict=True
    )
    locked_settings = {
        "NUM_EPOCHS": 150,
        "BATCH_SIZE": 4096,
        "INFER_SIZE": 1024,
        "BEAM_SIZE": 20,
        "TOPK_LIST": [5, 10],
        "NO_EVAL": True,
        "SKIP_TEST": False,
    }
    for name, expected in locked_settings.items():
        if getattr(trainer, name) != expected:
            raise RuntimeError(
                f"unchanged Stage3 setting {name}={getattr(trainer, name)!r}, "
                f"expected {expected!r}"
            )
    if Path(trainer.DATASET_PATH).resolve(strict=True) != expected_data_root:
        raise RuntimeError("Stage3 data root changed from the locked Stage0 directory")
    if (
        trainer._LAUNCHER["nproc"] != 4
        or trainer._LAUNCHER["master_port"] != 50201
        or trainer._LAUNCHER["visible_dev"] != "0,1,2,3"
    ):
        raise RuntimeError("Stage3 DDP launcher no longer matches the locked protocol")

    trainer.CODE_PATH = str(sid_path)
    trainer.RQVAE_VARIANT = profile["mechanism_name"]
    trainer.SEED = int(profile["seed"])
    trainer.LOG_PATH = str(profile["stage3_log_root"]) + os.sep
    trainer.SAVE_PATH = str(profile["stage3_ckpt_root"]) + os.sep
    trainer._LAUNCHER["script"] = str(profile["stage3_entrypoint"])
    trainer._LAUNCHER["log"] = str(profile["stage3_launcher_log"])

    original_resolver = trainer._resolve_code_file

    def resolve_exact_code_file(dataset_dir, requested):
        if Path(dataset_dir).resolve(strict=True) != expected_data_root:
            raise RuntimeError("Stage3 resolver received a different Stage0 data root")
        requested_path = Path(requested)
        if not requested_path.is_absolute():
            raise RuntimeError("iter30 Stage3 CODE_PATH must be absolute")
        resolved_requested = requested_path.resolve(strict=True)
        if resolved_requested != sid_path:
            raise RuntimeError(
                f"Stage3 requested a different SID file: {resolved_requested} != {sid_path}"
            )
        resolved = original_resolver(dataset_dir, str(sid_path)).resolve(strict=True)
        if resolved != sid_path:
            raise RuntimeError(
                f"Stage3 SID resolver selected fallback {resolved}, expected {sid_path}"
            )
        if sha256_file(resolved) != expected_sid_digest:
            raise RuntimeError("Stage2 item_sids.json changed while Stage3 was resolving it")
        return resolved

    trainer._resolve_code_file = resolve_exact_code_file
    resolved = trainer._resolve_code_file(Path(trainer.DATASET_PATH), trainer.CODE_PATH)
    if resolved != sid_path:
        raise RuntimeError("Stage3 strict SID resolver failed its prelaunch identity check")


def run_stage3_profile(label: str, entrypoint: str) -> None:
    profile = get_profile(label)
    wrapper_path = Path(entrypoint).resolve(strict=True)
    if wrapper_path != profile["stage3_entrypoint"].resolve(strict=True):
        raise RuntimeError(f"Stage3 wrapper/profile mismatch: {wrapper_path} != {label}")
    if not _path_is_below(profile["stage3_root"], STAGE3_RESULTS_ROOT):
        raise RuntimeError(f"Stage3 profile escaped its registered result root: {label}")

    if "RANK" not in os.environ:
        identity = _prepare_stage3_run(profile)
        trainer = _load_trainer()
        _configure_trainer(trainer, profile)
        with profile["stage3_wrapper_log"].open("xb") as wrapper_log:
            with contextlib.redirect_stdout(wrapper_log), contextlib.redirect_stderr(wrapper_log):
                print(
                    f"STAGE3_PROFILE_START label={label} mapping={profile['mapping_id']} "
                    f"seed={profile['seed']} code_path={trainer.CODE_PATH}",
                    flush=True,
                )
                print(json.dumps(identity, indent=2, sort_keys=True), flush=True)
                trainer._launch_via_torchrun()
        return

    trainer = _load_trainer()
    _configure_trainer(trainer, profile)
    trainer.main()
