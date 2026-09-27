"""Shared fail-closed four-token SID exporter for iter30 profile routes."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
SOURCE_ROOT = SCRIPT_DIR.parent
REPO_ROOT = SOURCE_ROOT.parent.parent
STAGE2_RESULTS_ROOT = (
    REPO_ROOT / "results" / "stage2_RQ-VAE" / "curvature_RQ-VAE_iter30"
)
N_ITEMS = 24_587
CODEBOOK_SIZES = (256, 256, 256)


def _extend_collisions(tokens: np.ndarray, codebook_sizes: tuple[int, ...]) -> np.ndarray:
    if tokens.ndim != 2 or tokens.shape[1] != len(codebook_sizes):
        raise ValueError(
            f"raw SID shape {tokens.shape} is incompatible with {codebook_sizes}"
        )
    groups: dict[tuple[int, ...], list[int]] = {}
    for item_id, row in enumerate(tokens.tolist()):
        groups.setdefault(tuple(int(value) for value in row), []).append(item_id)
    offsets = np.cumsum([0, *codebook_sizes], dtype=np.int64)
    extended = np.empty((len(tokens), tokens.shape[1] + 1), dtype=np.int64)
    for row, item_ids in groups.items():
        for occurrence, item_id in enumerate(item_ids):
            extended[item_id, :-1] = np.asarray(row, dtype=np.int64)
            extended[item_id, -1] = int(offsets[-1] + occurrence)
    return extended


def export_sids_for_stage3(
    raw_sids_path: str | Path,
    sids_npy_path: str | Path,
    item_sids_json_path: str | Path,
) -> dict:
    expected_items = N_ITEMS
    codebook_sizes = CODEBOOK_SIZES
    raw_path = Path(raw_sids_path)
    npy_path = Path(sids_npy_path)
    json_path = Path(item_sids_json_path)
    if not all(path.is_absolute() for path in (raw_path, npy_path, json_path)):
        raise ValueError("SID source and output paths must be absolute")
    if len(set((raw_path, npy_path, json_path))) != 3:
        raise ValueError("raw SID source, Stage3 NPY, and JSON paths must be distinct")
    if not raw_path.is_file():
        raise FileNotFoundError(f"raw three-token SID source missing: {raw_path}")
    resolved_raw = raw_path.resolve(strict=True)
    profile_root = resolved_raw.parents[3]
    expected_raw = profile_root / "out" / "rqvae" / "instruments" / "sids_raw.npy"
    expected_npy = profile_root / "dataset" / "Instruments" / "sids_for_hgrec.npy"
    expected_json = profile_root / "item_sids.json"
    if profile_root.parent != STAGE2_RESULTS_ROOT:
        raise RuntimeError(f"raw SID is outside the registered iter30 result root: {raw_path}")
    if resolved_raw != expected_raw:
        raise RuntimeError(f"unexpected raw SID source path: {resolved_raw} != {expected_raw}")
    if npy_path.resolve(strict=False) != expected_npy:
        raise RuntimeError(f"unexpected Stage3 SID NPY route: {npy_path} != {expected_npy}")
    if json_path.resolve(strict=False) != expected_json:
        raise RuntimeError(f"unexpected Stage3 SID JSON route: {json_path} != {expected_json}")
    for path in (npy_path, json_path):
        if path.exists() or path.is_symlink():
            raise FileExistsError(f"refusing to overwrite Stage3 SID output: {path}")
    raw = np.load(resolved_raw, allow_pickle=False)
    expected_shape = (expected_items, len(codebook_sizes))
    if raw.shape != expected_shape:
        raise ValueError(f"raw SID shape {raw.shape} != {expected_shape}")
    if not np.issubdtype(raw.dtype, np.integer):
        raise ValueError(f"raw SID values must be integers, found {raw.dtype}")
    tokens = raw.astype(np.int64, copy=False)
    limits = np.asarray(codebook_sizes, dtype=np.int64)
    if len(codebook_sizes) != 3 or np.any(limits <= 0):
        raise ValueError(f"invalid registered codebook sizes: {codebook_sizes}")
    if np.any(tokens < 0) or np.any(tokens >= limits):
        raise ValueError("raw SID token is outside its layer codebook range")

    n_unique_three_token = int(np.unique(tokens, axis=0).shape[0])
    extended = _extend_collisions(tokens, codebook_sizes)
    if extended.shape != (expected_items, len(codebook_sizes) + 1):
        raise RuntimeError(f"four-token extension produced invalid shape {extended.shape}")
    if np.unique(extended, axis=0).shape[0] != expected_items:
        raise RuntimeError("four-token collision extension did not produce unique item codes")

    payload = {
        str(item_id): [int(value) for value in row]
        for item_id, row in enumerate(extended)
    }
    npy_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(npy_path, extended)
    json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    reloaded = np.load(npy_path, allow_pickle=False)
    if reloaded.shape != extended.shape or reloaded.dtype != np.int64:
        raise RuntimeError("Stage3 SID NPY read-back shape or dtype mismatch")
    if not np.array_equal(reloaded, extended):
        raise RuntimeError("Stage3 SID NPY read-back content mismatch")
    reloaded_json = json.loads(json_path.read_text(encoding="utf-8"))
    expected_keys = [str(item_id) for item_id in range(expected_items)]
    if list(reloaded_json) != expected_keys:
        raise RuntimeError("Stage3 SID JSON keys are not dense ordered item IDs")
    if any(
        not isinstance(reloaded_json[key], list)
        or len(reloaded_json[key]) != len(codebook_sizes) + 1
        for key in expected_keys
    ):
        raise RuntimeError("Stage3 SID JSON rows are not four-token lists")
    json_array = np.asarray([reloaded_json[key] for key in expected_keys], dtype=np.int64)
    if not np.array_equal(json_array, reloaded):
        raise RuntimeError("Stage3 SID JSON and NPY read-back contents differ")

    report = {
        "status": "SID_WIRING_PASS",
        "raw_sids": str(resolved_raw),
        "sids_npy": str(npy_path),
        "item_sids_json": str(json_path),
        "n_items": expected_items,
        "n_unique_three_token": n_unique_three_token,
        "collision_rows": expected_items - n_unique_three_token,
        "four_token_unique": expected_items,
        "extension_min": int(extended[:, -1].min()),
        "extension_max": int(extended[:, -1].max()),
    }
    print(json.dumps(report, sort_keys=True), flush=True)
    return report


def main() -> None:
    if str(SOURCE_ROOT) not in sys.path:
        sys.path.insert(0, str(SOURCE_ROOT))
    from curvature_config import ITEM_SIDS_JSON, RAW_SIDS_NPY, SIDS_NPY

    export_sids_for_stage3(RAW_SIDS_NPY, SIDS_NPY, ITEM_SIDS_JSON)


if __name__ == "__main__":
    main()
