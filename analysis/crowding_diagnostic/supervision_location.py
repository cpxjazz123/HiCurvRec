"""Compare the location of behaviour supervision across the five completed arms."""
from __future__ import annotations

import csv
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
import config as C
import distances
import residuals

LEVELS = C.LEVELS
PAIR_SETS = (
    ("primary_valid", C.PRIMARY_PAIRS),
    ("secondary_train", C.SECONDARY_PAIRS),
)
ARMS = {
    "TIGER": C.ARMS["TIGER"],
    "Exp1a": C.ARMS["Exp1a"],
    "l1only": {
        "checkpoint": C.ROOT / "results/stage2_RQ-VAE/TIGER_BEHAVIOR_SAMPLE_RQ-VAE/resample_l1only/rqvae_best.pth",
        "sid_json": C.ROOT / "results/stage2_RQ-VAE/TIGER_BEHAVIOR_SAMPLE_RQ-VAE/resample_l1only/item_sids.json",
        "sha256": "8f2f4070cf80fdcb30475ec493c7908b10c0b36e98cfdea5a32a9b5268a87ca7",
    },
    "l2only": {
        "checkpoint": C.ROOT / "results/stage2_RQ-VAE/TIGER_BEHAVIOR_SAMPLE_RQ-VAE/resample_l2only/rqvae_best.pth",
        "sid_json": C.ROOT / "results/stage2_RQ-VAE/TIGER_BEHAVIOR_SAMPLE_RQ-VAE/resample_l2only/item_sids.json",
        "sha256": "018ef7b19d471714da3055395cd49f1920cd12efa14c72dd9ac8f0e04f6f54fb",
    },
    "l3only": {
        "checkpoint": C.ROOT / "results/stage2_RQ-VAE/TIGER_BEHAVIOR_SAMPLE_RQ-VAE/resample_l3only/rqvae_best.pth",
        "sid_json": C.ROOT / "results/stage2_RQ-VAE/TIGER_BEHAVIOR_SAMPLE_RQ-VAE/resample_l3only/item_sids.json",
        "sha256": "018ef7b19d471714da3055395cd49f1920cd12efa14c72dd9ac8f0e04f6f54fb",
    },
}
CODEBOOK_SIZES = (256, 256, 256)
OUT_DIR = C.ROOT / "results/crowding_diagnostic/supervision_location"
TABLE = OUT_DIR / "comparison.csv"
TABLE_FIELDS = (
    "arm", "level", "pair_set", "auc", "auc_ci_low", "auc_ci_high", "ordering_error_rate",
    "residual_norm_mean", "occupied_codes", "codebook_size", "codebook_occupancy_fraction",
    "raw_sid_collision_count", "raw_sid_collision_rate",
)


def _prepare_extractor():
    """Route the shared extraction implementation to this analysis's arm/output set."""
    residuals.ARMS = ARMS
    residuals.OUT_DIR = OUT_DIR


def _summary_for_arm(label: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    residual_file = OUT_DIR / f"residuals_{label}.npz"
    if not residual_file.exists():
        residuals.extract(label)
    with np.load(residual_file) as data:
        return data["residuals"], data["codes"], data["sid"]


def _rows_for_arm(label: str, residual: np.ndarray, codes: np.ndarray, sid: np.ndarray) -> list[dict]:
    raw_unique = len(np.unique(codes, axis=0))
    collision_count = len(codes) - raw_unique
    rows = []
    pairs_by_name = [(name, distances.load_pairs(path)) for name, path in PAIR_SETS]
    for level, level_name in enumerate(LEVELS):
        vectors = np.asarray(residual[:, level, :], dtype=np.float64)
        residual_norm_mean = float(np.linalg.norm(vectors, axis=-1).mean())
        occupied = int(np.unique(codes[:, level]).size)
        for pair_set, pairs in pairs_by_name:
            pos, neg = distances._pair_distances(vectors, pairs)
            ci = distances._anchor_auc_ci(pos, neg, pairs["anchors"], C.SEED + level,
                                          C.BOOTSTRAP_REPLICATES)
            rows.append({
                "arm": label,
                "level": level_name,
                "pair_set": pair_set,
                "auc": distances.auc_neg_farther_than_pos(pos, neg),
                "auc_ci_low": None if ci is None else ci[0],
                "auc_ci_high": None if ci is None else ci[1],
                "ordering_error_rate": distances.ordering_error_rate(pos, neg),
                "residual_norm_mean": residual_norm_mean,
                "occupied_codes": occupied,
                "codebook_size": CODEBOOK_SIZES[level],
                "codebook_occupancy_fraction": occupied / CODEBOOK_SIZES[level],
                "raw_sid_collision_count": int(collision_count),
                "raw_sid_collision_rate": collision_count / len(codes),
            })
    return rows


def _write_table(rows: list[dict]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with TABLE.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=TABLE_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def _print_table(rows: list[dict]) -> None:
    print("arm,level,pair_set,AUC [anchor-bootstrap 95% CI],ordering_error_rate,residual_norm_mean,occupied_codes/codebook_size,codebook_occupancy_fraction,raw_SID_collision_count,raw_SID_collision_rate")
    for row in rows:
        ci = f"[{row['auc_ci_low']:.9f}, {row['auc_ci_high']:.9f}]"
        print(f"{row['arm']},{row['level']},{row['pair_set']},{row['auc']:.9f} {ci},"
              f"{row['ordering_error_rate']:.9f},{row['residual_norm_mean']:.9f},"
              f"{row['occupied_codes']}/{row['codebook_size']},"
              f"{row['codebook_occupancy_fraction']:.9f},{row['raw_sid_collision_count']},"
              f"{row['raw_sid_collision_rate']:.9f}")


def _load_table() -> list[dict] | None:
    if not TABLE.exists():
        return None
    if TABLE.stat().st_size == 0:
        return None
    with TABLE.open(newline="") as stream:
        return [
            {
                **row,
                "auc": float(row["auc"]),
                "auc_ci_low": float(row["auc_ci_low"]),
                "auc_ci_high": float(row["auc_ci_high"]),
                "ordering_error_rate": float(row["ordering_error_rate"]),
                "residual_norm_mean": float(row["residual_norm_mean"]),
                "occupied_codes": int(row["occupied_codes"]),
                "codebook_size": int(row["codebook_size"]),
                "codebook_occupancy_fraction": float(row["codebook_occupancy_fraction"]),
                "raw_sid_collision_count": int(row["raw_sid_collision_count"]),
                "raw_sid_collision_rate": float(row["raw_sid_collision_rate"]),
            }
            for row in csv.DictReader(stream)
        ]


def _print_conclusion() -> None:
    print("EXPECTED IDENTITY: l2only and l3only AUCs equal TIGER exactly at every level on both pair sets; their deep residual path has zero gradient by the straight-through residual construction, not sampling noise.")
    print("Informative comparison: l1only versus Exp1a (all-level supervision); l2only/l3only deep-level rows are not independent evidence.")
    print(f"TABLE={TABLE}")


def run() -> list[dict]:
    _prepare_extractor()
    cached = _load_table()
    if cached is not None:
        _print_table(cached)
        _print_conclusion()
        return cached
    rows = []
    for label in ARMS:
        residual, codes, sid = _summary_for_arm(label)
        if len(sid) != len(codes):
            raise ValueError(f"SID/code count mismatch for {label}: {len(sid)} != {len(codes)}")
        rows.extend(_rows_for_arm(label, residual, codes, sid))
    _write_table(rows)
    _print_table(rows)
    _print_conclusion()
    return rows


if __name__ == "__main__":
    run()
