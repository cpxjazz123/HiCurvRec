"""Sweep the ICML-2018 cone protocol over geometry, dimension and supervision.

The grid is embarrassingly parallel across independent runs, so this module
fans the cells out over every visible GPU from a single entry point: no CLI
arguments, no environment overrides (project rule 1). Each run writes a native
``metrics.json`` plus the per-positive test AUC vector used for paired
bootstrap; ``summary.json`` is rebuilt from those files, so a sweep can be
resumed or re-aggregated without retraining.
"""

from __future__ import annotations

import hashlib
import json
import multiprocessing as mp
import time
from pathlib import Path

import numpy as np
import torch

from bench_hier_behavior.behavior_metrics import paired_bootstrap_delta

from . import semantic_config as cfg
from .paper_cones import PaperConeData, load_dataset, load_structure, train_one

Cell = tuple[str, int, str, int]


def _run_key(geometry: str, dim: int, ratio: str, seed: int) -> str:
    return f"{geometry}_d{dim}_{ratio}_s{seed}"


def _run_dir(geometry: str, dim: int, ratio: str, seed: int) -> Path:
    return cfg.PAPER_RESULT_DIR / _run_key(geometry, dim, ratio, seed)


def _signature(geometry: str, dim: int, ratio: str, seed: int) -> str:
    """Everything that would change a run's numbers, hashed."""
    payload = {
        "dataset": cfg.PAPER_DATASET,
        "geometry": geometry,
        "dim": dim,
        "ratio": ratio,
        "seed": seed,
        "k": cfg.PAPER_K,
        "margin": cfg.PAPER_MARGIN,
        "epsilon": cfg.PAPER_EPSILON,
        "epochs": cfg.PAPER_EPOCHS,
        "batch": cfg.PAPER_BATCH,
        "negatives": cfg.PAPER_NEGATIVES,
        "update_cap": cfg.PAPER_UPDATE_CAP,
        "lr": cfg.PAPER_LR_BY_GEOMETRY[geometry],
        "init_low": cfg.PAPER_INIT_LOW,
        "init_high": cfg.PAPER_INIT_HIGH,
        "warm_start": cfg.PAPER_USE_WARM_START,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def _save_run(geometry: str, dim: int, ratio: str, seed: int, result: dict) -> None:
    run_dir = _run_dir(geometry, dim, ratio, seed)
    run_dir.mkdir(parents=True, exist_ok=True)
    metrics = dict(result["metrics"])
    metrics["run_signature"] = _signature(geometry, dim, ratio, seed)
    with (run_dir / "metrics.json").open("w") as handle:
        json.dump(metrics, handle, indent=2, sort_keys=True)
    np.savez_compressed(
        run_dir / "per_positive_auc.npz",
        test=np.asarray(result["test_per_positive_auc"], dtype=np.float32),
    )


def _load_run(geometry: str, dim: int, ratio: str, seed: int) -> dict | None:
    """Return a completed run, or ``None`` when it must be retrained.

    Compatibility is checked against the stored training block rather than
    assumed, so a run produced under different constants is never silently
    reused.
    """
    run_dir = _run_dir(geometry, dim, ratio, seed)
    metrics_path = run_dir / "metrics.json"
    per_positive_path = run_dir / "per_positive_auc.npz"
    if not metrics_path.is_file() or not per_positive_path.is_file():
        return None
    with metrics_path.open() as handle:
        metrics = json.load(handle)
    training = metrics.get("training", {})
    expected = {
        "epochs": cfg.PAPER_EPOCHS,
        "batch": cfg.PAPER_BATCH,
        "learning_rate": cfg.PAPER_LR_BY_GEOMETRY[geometry],
        "negatives": cfg.PAPER_NEGATIVES,
    }
    if any(training.get(key) != value for key, value in expected.items()):
        return None
    if (
        metrics.get("geometry") != geometry
        or metrics.get("dim") != dim
        or metrics.get("ratio") != ratio
        or metrics.get("seed") != seed
        or metrics.get("dataset") != cfg.PAPER_DATASET
        or metrics.get("k") != cfg.PAPER_K
        or metrics.get("margin") != cfg.PAPER_MARGIN
    ):
        return None
    stored = metrics.get("run_signature")
    if stored is not None and stored != _signature(geometry, dim, ratio, seed):
        return None
    with np.load(per_positive_path) as saved:
        per_positive = np.asarray(saved["test"], dtype=np.float64)
    return {"metrics": metrics, "test_per_positive_auc": per_positive}


def _cells() -> list[Cell]:
    return [
        (geometry, dim, ratio, seed)
        for ratio in cfg.PAPER_RATIOS
        for dim in cfg.PAPER_DIMS
        for geometry in cfg.PAPER_GEOMETRIES
        for seed in cfg.PAPER_SEEDS
    ]


def _shards(cells: list[Cell], shard_count: int) -> list[list[Cell]]:
    """Round-robin over cells ordered by cost so the shards finish together."""
    ratio_order = {ratio: index for index, ratio in enumerate(cfg.PAPER_RATIOS)}
    ordered = sorted(
        cells, key=lambda cell: (ratio_order[cell[2]], cell[1]), reverse=True
    )
    shards: list[list[Cell]] = [[] for _ in range(shard_count)]
    for position, cell in enumerate(ordered):
        shards[position % shard_count].append(cell)
    return shards


def _worker(device_index: int, cells: list[Cell]) -> None:
    torch.cuda.set_device(device_index)
    device = f"cuda:{device_index}"
    torch.set_num_threads(4)
    datasets: dict[str, PaperConeData] = {}
    for geometry, dim, ratio, seed in cells:
        if _load_run(geometry, dim, ratio, seed) is not None:
            print(f"[paper-cones] resume {_run_key(geometry, dim, ratio, seed)}", flush=True)
            continue
        if ratio not in datasets:
            datasets[ratio] = load_dataset(cfg.PAPER_DATA_DIR, cfg.PAPER_DATASET, ratio)
        print(
            f"[paper-cones] start {_run_key(geometry, dim, ratio, seed)} gpu{device_index}",
            flush=True,
        )
        result = train_one(geometry, dim, datasets[ratio], seed, device)
        _save_run(geometry, dim, ratio, seed, result)
        metrics = result["metrics"]
        print(
            f"[paper-cones] done {_run_key(geometry, dim, ratio, seed)} "
            f"test_f1={metrics['test']['f1']:.2f} "
            f"test_auc={metrics['test']['auc']:.4f} "
            f"valid_f1={metrics['valid']['f1']:.2f} "
            f"loss/pair={metrics['loss_curve'][-1]['train_loss']:.4f} "
            f"secs={metrics['training']['elapsed_seconds']:.1f}",
            flush=True,
        )


def _mean_std(values: list[float]) -> dict:
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
        "n_seeds": len(values),
    }


def _metrics_of(cell: Cell) -> dict:
    with (_run_dir(*cell) / "metrics.json").open() as handle:
        return json.load(handle)


def _per_positive_of(cell: Cell) -> np.ndarray:
    with np.load(_run_dir(*cell) / "per_positive_auc.npz") as saved:
        return np.asarray(saved["test"], dtype=np.float64)


def _paired_vectors(geometry: str, dim: int, ratio: str) -> np.ndarray:
    vectors = [
        _per_positive_of((geometry, dim, ratio, seed)) for seed in cfg.PAPER_SEEDS
    ]
    for vector in vectors[1:]:
        if vector.shape != vectors[0].shape:
            raise ValueError("Mismatched test pair count across seeds")
    return np.stack(vectors).mean(axis=0)


def _aggregate() -> dict:
    grid: dict[str, dict] = {}
    for geometry in cfg.PAPER_GEOMETRIES:
        grid[geometry] = {}
        for ratio in cfg.PAPER_RATIOS:
            grid[geometry][ratio] = {}
            for dim in cfg.PAPER_DIMS:
                rows = [
                    _metrics_of((geometry, dim, ratio, seed))
                    for seed in cfg.PAPER_SEEDS
                ]
                grid[geometry][ratio][f"d{dim}"] = {
                    "test_f1": _mean_std([row["test"]["f1"] for row in rows]),
                    "test_auc": _mean_std([row["test"]["auc"] for row in rows]),
                    "test_precision": _mean_std(
                        [row["test"]["precision"] for row in rows]
                    ),
                    "test_recall": _mean_std([row["test"]["recall"] for row in rows]),
                    "valid_f1": _mean_std([row["valid"]["f1"] for row in rows]),
                    "valid_auc": _mean_std([row["valid"]["auc"] for row in rows]),
                    "threshold": _mean_std([row["test"]["threshold"] for row in rows]),
                    "train_loss_per_pair": _mean_std(
                        [row["loss_curve"][-1]["train_loss"] for row in rows]
                    ),
                    "elapsed_seconds": _mean_std(
                        [row["training"]["elapsed_seconds"] for row in rows]
                    ),
                }

    comparisons: dict[str, dict] = {}
    for ratio in cfg.PAPER_RATIOS:
        for dim in cfg.PAPER_DIMS:
            left = _paired_vectors("poincare", dim, ratio)
            right = _paired_vectors("euclid", dim, ratio)
            entry = paired_bootstrap_delta(
                left,
                right,
                seed=cfg.BOOTSTRAP_SEED + dim + 100 * cfg.PAPER_RATIOS.index(ratio),
                samples=cfg.BOOTSTRAP_SAMPLES,
            )
            entry["seed_deltas"] = {
                str(seed): float(
                    _metrics_of(("poincare", dim, ratio, seed))["test"]["auc"]
                    - _metrics_of(("euclid", dim, ratio, seed))["test"]["auc"]
                )
                for seed in cfg.PAPER_SEEDS
            }
            entry["all_seeds_positive"] = bool(
                all(value > 0.0 for value in entry["seed_deltas"].values())
            )
            entry["f1_seed_deltas"] = {
                str(seed): float(
                    _metrics_of(("poincare", dim, ratio, seed))["test"]["f1"]
                    - _metrics_of(("euclid", dim, ratio, seed))["test"]["f1"]
                )
                for seed in cfg.PAPER_SEEDS
            }
            entry["f1_mean_delta"] = float(
                np.mean(list(entry["f1_seed_deltas"].values()))
            )
            entry["f1_all_seeds_positive"] = bool(
                all(value > 0.0 for value in entry["f1_seed_deltas"].values())
            )
            comparisons[f"{ratio}_d{dim}"] = entry
    return {"grid": grid, "comparisons": comparisons}


def _bucket_masks(
    values: np.ndarray, cuts: tuple[tuple[str, int, int], ...]
) -> dict[str, np.ndarray]:
    return {label: (values >= low) & (values <= high) for label, low, high in cuts}


def _stratify(summary: dict) -> dict:
    """Break the held-out AUC down by apex depth and apex branching."""
    depth, degree, cyclic = load_structure(cfg.PAPER_DATA_DIR, cfg.PAPER_DATASET)
    parent = load_dataset(
        cfg.PAPER_DATA_DIR, cfg.PAPER_DATASET, cfg.PAPER_RATIOS[0]
    ).test_u
    if depth[parent].min() < 1:
        raise ValueError("Test split contains nodes with unresolved depth")
    depth_masks = _bucket_masks(depth[parent], cfg.PAPER_DEPTH_CUTS)
    degree_masks = _bucket_masks(degree[parent], cfg.PAPER_DEGREE_CUTS)
    rows: dict[str, dict] = {}
    for geometry in cfg.PAPER_GEOMETRIES:
        rows[geometry] = {}
        for ratio in cfg.PAPER_RATIOS:
            rows[geometry][ratio] = {}
            for dim in cfg.PAPER_DIMS:
                vectors = [
                    _per_positive_of((geometry, dim, ratio, seed))
                    for seed in cfg.PAPER_SEEDS
                ]
                mean = np.stack(vectors).mean(axis=0)
                rows[geometry][ratio][f"d{dim}"] = {
                    "depth": {
                        label: float(mean[mask].mean()) if mask.any() else None
                        for label, mask in depth_masks.items()
                    },
                    "degree": {
                        label: float(mean[mask].mean()) if mask.any() else None
                        for label, mask in degree_masks.items()
                    },
                }
    deltas: dict[str, dict] = {}
    for ratio in cfg.PAPER_RATIOS:
        for dim in cfg.PAPER_DIMS:
            left = _paired_vectors("poincare", dim, ratio)
            right = _paired_vectors("euclid", dim, ratio)
            entry: dict[str, float | None] = {}
            for family, masks in (("depth", depth_masks), ("degree", degree_masks)):
                for label, mask in masks.items():
                    entry[f"{family}_{label}"] = (
                        float(left[mask].mean() - right[mask].mean())
                        if mask.any()
                        else None
                    )
            deltas[f"{ratio}_d{dim}"] = entry
    summary["stratification"] = {
        "depth_cuts": [list(cut) for cut in cfg.PAPER_DEPTH_CUTS],
        "degree_cuts": [list(cut) for cut in cfg.PAPER_DEGREE_CUTS],
        "depth_n": {label: int(mask.sum()) for label, mask in depth_masks.items()},
        "degree_n": {label: int(mask.sum()) for label, mask in degree_masks.items()},
        "mean_auc_by_geometry": rows,
        "poincare_minus_euclid": deltas,
        "cyclic_nodes_excluded": cyclic,
    }
    return summary


def _verdict(summary: dict) -> dict:
    rows = []
    for key, entry in summary["comparisons"].items():
        rows.append(
            {
                "cell": key,
                "auc_delta": entry["mean_delta"],
                "auc_ci_low": entry["ci95_low"],
                "auc_all_seeds_positive": entry["all_seeds_positive"],
                "f1_delta": entry["f1_mean_delta"],
                "f1_all_seeds_positive": entry["f1_all_seeds_positive"],
            }
        )
    supported = [
        row["cell"]
        for row in rows
        if row["auc_ci_low"] > 0.0 and row["auc_all_seeds_positive"]
    ]
    reversed_cells = [
        row["cell"]
        for row in rows
        if row["auc_ci_low"] < 0.0 and not row["auc_all_seeds_positive"]
    ]
    zero_ratio = [row for row in rows if row["cell"].startswith("0percent")]
    return {
        "cells": rows,
        "cells_with_poincare_auc_advantage": supported,
        "cells_with_euclid_auc_advantage": reversed_cells,
        "n_cells": len(rows),
        "n_advantage": len(supported),
        "n_euclid_advantage": len(reversed_cells),
        "zero_ratio_auc_deltas": {
            row["cell"]: row["auc_delta"] for row in zero_ratio
        },
        "zero_ratio_note": (
            "0% trains on the transitive reduction only; the reference reports "
            "the two geometries as near-tied there"
        ),
    }


def main() -> None:
    started = time.time()
    device_count = torch.cuda.device_count()
    if device_count == 0:
        raise RuntimeError("The paper cone sweep requires CUDA devices")
    cfg.PAPER_RESULT_DIR.mkdir(parents=True, exist_ok=True)
    cells = _cells()
    shards = _shards(cells, device_count)
    print(
        f"[paper-cones] {len(cells)} cells over {device_count} gpus "
        f"(shard sizes {[len(shard) for shard in shards]})",
        flush=True,
    )

    context = mp.get_context("spawn")
    processes = [
        context.Process(target=_worker, args=(index, shard))
        for index, shard in enumerate(shards)
    ]
    for process in processes:
        process.start()
    for process in processes:
        process.join()
    failed = [process.exitcode for process in processes if process.exitcode != 0]
    if failed:
        raise RuntimeError(f"Shard workers failed with exit codes {failed}")

    missing = [cell for cell in cells if _load_run(*cell) is None]
    if missing:
        raise RuntimeError(f"{len(missing)} cells have no usable artifact: {missing[:5]}")

    summary = _aggregate()
    summary = _stratify(summary)
    summary["verdict"] = _verdict(summary)
    summary["experiment"] = {
        "reference": "Ganea, Becigneul & Hofmann, ICML 2018",
        "data": f"{cfg.PAPER_DATASET}_closure.tsv splits from the authors' release",
        "data_dir": str(cfg.PAPER_DATA_DIR),
        "geometries": list(cfg.PAPER_GEOMETRIES),
        "dims": list(cfg.PAPER_DIMS),
        "ratios": list(cfg.PAPER_RATIOS),
        "seeds": list(cfg.PAPER_SEEDS),
        "k": float(cfg.PAPER_K),
        "margin": float(cfg.PAPER_MARGIN),
        "epochs": int(cfg.PAPER_EPOCHS),
        "batch": int(cfg.PAPER_BATCH),
        "paper_batch": 10,
        "learning_rate_by_geometry": dict(cfg.PAPER_LR_BY_GEOMETRY),
        "negatives": int(cfg.PAPER_NEGATIVES),
        "update_cap": float(cfg.PAPER_UPDATE_CAP),
        "init": (
            "cold start: identical random directions and radii in "
            f"[{cfg.PAPER_INIT_LOW}, {cfg.PAPER_INIT_HIGH}] for both arms; the "
            "optional Poincare NLL warm start is disabled"
        ),
        "gpus": device_count,
        "deviations": [
            f"gradients are summed over a {cfg.PAPER_BATCH}-pair chunk instead of "
            "a 10-pair batch, so the per-epoch update total matches but is "
            "applied in far coarser steps (~3000x fewer updates per run)",
            "a trust-region cap keeps coarse steps from reaching the Poincare "
            "boundary, where the metric diverges; identical for both arms",
            "the optional Poincare NLL warm start is disabled because the "
            "reimplementation drives every node onto the boundary",
        ],
        "elapsed_seconds": float(time.time() - started),
    }
    with (cfg.PAPER_RESULT_DIR / "summary.json").open("w") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)

    print("[paper-cones] test F1 / AUC by ratio and dimension", flush=True)
    for ratio in cfg.PAPER_RATIOS:
        for dim in cfg.PAPER_DIMS:
            eu = summary["grid"]["euclid"][ratio][f"d{dim}"]
            hy = summary["grid"]["poincare"][ratio][f"d{dim}"]
            delta = summary["comparisons"][f"{ratio}_d{dim}"]
            print(
                f"  {ratio:<10} d{dim:<3} "
                f"euclid f1={eu['test_f1']['mean']:.2f} auc={eu['test_auc']['mean']:.4f} | "
                f"poincare f1={hy['test_f1']['mean']:.2f} auc={hy['test_auc']['mean']:.4f} | "
                f"df1={delta['f1_mean_delta']:+.2f} "
                f"dauc={delta['mean_delta']:+.4f} "
                f"[{delta['ci95_low']:+.4f},{delta['ci95_high']:+.4f}]",
                flush=True,
            )
    print(
        f"[paper-cones] complete poincare_auc_wins="
        f"{summary['verdict']['n_advantage']}/{summary['verdict']['n_cells']} "
        f"euclid_auc_wins={summary['verdict']['n_euclid_advantage']} "
        f"elapsed_seconds={summary['experiment']['elapsed_seconds']:.1f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
