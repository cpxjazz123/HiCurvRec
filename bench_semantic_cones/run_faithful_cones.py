"""Faithful-optimisation round: does the geometry gap survive a smaller batch?

Two axes, both run through the same objective, data and metric protocol as the
main sweep:

* **granularity** - batch size swept from the wide-chunk default down to 64
  pairs on the decisive cell (dim 5, 50% supervision), to test whether the
  hyperbolic deficit is an artefact of coarse updates. The reference uses 10.
* **initialisation** - the Poincare NLL warm start against the shared cold
  start, to test whether the deficit comes from the initialisation that the
  reference's own Figure 3 shows collapsing onto the ball boundary.

Cells are sharded across every visible GPU from this single entry point and are
resumable: a completed run is reused only when its stored training block matches
the constants that would produce it.
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
from .paper_cones import (
    PaperConeData,
    load_dataset,
    load_structure,
    train_initializer,
    train_one,
)

Cell = tuple[str, int, str, int, str, int, str]  # geometry, dim, ratio, seed, init, batch, mode


def _run_key(cell: Cell) -> str:
    geometry, dim, ratio, seed, init, batch, mode = cell
    key = f"{geometry}_d{dim}_{ratio}_s{seed}"
    if init != "cold":
        key += f"_{init}"
    if batch != cfg.PAPER_BATCH:
        key += f"_b{batch}"
    if mode != "coarse":
        key += f"_{mode}"
    return key


def _learning_rate(cell: Cell) -> float:
    mode = cell[6]
    if mode == "coarse":
        return cfg.FAITHFUL_COARSE_LR[cell[0]]
    return cfg.FAITHFUL_LR[cell[0]]


def _update_cap(cell: Cell):
    return cfg.FAITHFUL_COARSE_CAP if cell[6] == "coarse" else cfg.FAITHFUL_UPDATE_CAP


def _run_dir(cell: Cell) -> Path:
    return cfg.FAITHFUL_RESULT_DIR / _run_key(cell)


def _signature(cell: Cell) -> str:
    geometry, dim, ratio, seed, init, batch, mode = cell
    payload = {
        "mode": mode,
        "dataset": cfg.PAPER_DATASET,
        "geometry": geometry,
        "dim": dim,
        "ratio": ratio,
        "seed": seed,
        "init": init,
        "batch": batch,
        "k": cfg.PAPER_K,
        "margin": cfg.PAPER_MARGIN,
        "epsilon": cfg.PAPER_EPSILON,
        "epochs": cfg.FAITHFUL_EPOCHS,
        "negatives": cfg.PAPER_NEGATIVES,
        "update_cap": _update_cap(cell),
        "lr": _learning_rate(cell),
        "init_lr": cfg.PAPER_INIT_LR,
        "init_epochs": cfg.PAPER_INIT_EPOCHS,
        "init_burn_in": cfg.PAPER_INIT_BURN_IN,
        "init_neg_power": cfg.PAPER_INIT_NEG_POWER,
        "resc_vecs": cfg.PAPER_RESC_VECS,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def _save_run(cell: Cell, result: dict) -> None:
    run_dir = _run_dir(cell)
    run_dir.mkdir(parents=True, exist_ok=True)
    metrics = dict(result["metrics"])
    metrics["run_signature"] = _signature(cell)
    metrics["init"] = cell[4]
    with (run_dir / "metrics.json").open("w") as handle:
        json.dump(metrics, handle, indent=2, sort_keys=True)
    np.savez_compressed(
        run_dir / "per_positive_auc.npz",
        test=np.asarray(result["test_per_positive_auc"], dtype=np.float32),
    )


def _load_run(cell: Cell) -> dict | None:
    run_dir = _run_dir(cell)
    metrics_path = run_dir / "metrics.json"
    per_positive_path = run_dir / "per_positive_auc.npz"
    if not metrics_path.is_file() or not per_positive_path.is_file():
        return None
    with metrics_path.open() as handle:
        metrics = json.load(handle)
    if metrics.get("run_signature") != _signature(cell):
        return None
    geometry, dim, ratio, seed, init, batch, mode = cell
    training = metrics.get("training", {})
    if (
        metrics.get("geometry") != geometry
        or metrics.get("dim") != dim
        or metrics.get("ratio") != ratio
        or metrics.get("seed") != seed
        or metrics.get("init") != init
        or training.get("batch") != batch
        or training.get("epochs") != cfg.FAITHFUL_EPOCHS
        or training.get("learning_rate") != _learning_rate(cell)
    ):
        return None
    with np.load(per_positive_path) as saved:
        per_positive = np.asarray(saved["test"], dtype=np.float64)
    return {"metrics": metrics, "test_per_positive_auc": per_positive}


def _cells() -> list[Cell]:
    cells: list[Cell] = []
    for geometry in cfg.PAPER_GEOMETRIES:
        for dim in cfg.FAITHFUL_DIMS:
            for ratio in cfg.FAITHFUL_RATIOS:
                for init in cfg.FAITHFUL_INITS:  # noqa: SIM118
                    cells.append(
                        (
                            geometry,
                            dim,
                            ratio,
                            cfg.FAITHFUL_SEED,
                            init,
                            cfg.FAITHFUL_BASE_BATCH,
                            "coarse",
                        )
                    )
    for geometry in cfg.PAPER_GEOMETRIES:
        for batch in cfg.FAITHFUL_BATCHES:
            # Every rung of the ladder runs at the reference learning rate, so
            # the batch size is the only thing that varies along it.
            cells.append(
                (
                    geometry,
                    cfg.FAITHFUL_GRID_DIM,
                    cfg.FAITHFUL_GRID_RATIO,
                    cfg.FAITHFUL_SEED,
                    "cold",
                    batch,
                    "paper",
                )
            )
    return cells


def _train_pairs(ratio: str) -> int:
    """Line count of the released train split, cached per ratio."""
    cached = _TRAIN_PAIRS_CACHE.get(ratio)
    if cached is None:
        path = Path(str(cfg.PAPER_DATA_DIR / f"{cfg.PAPER_DATASET}_closure.tsv")
                    + f".train_{ratio}")
        with path.open() as handle:
            cached = sum(1 for _ in handle)
        _TRAIN_PAIRS_CACHE[ratio] = cached
    return cached


_TRAIN_PAIRS_CACHE: dict[str, int] = {}


def _cost(cell: Cell) -> int:
    """Update count a cell will perform, warm start included."""
    _, _, ratio, _, init, batch, _ = cell
    pairs = _train_pairs(ratio)
    steps = -(-pairs // batch) * cfg.FAITHFUL_EPOCHS
    if init == "warm":
        steps += -(-pairs // batch) * cfg.PAPER_INIT_EPOCHS
    return steps


def _shards(cells: list[Cell], shard_count: int) -> list[list[Cell]]:
    """Longest-processing-time-first packing, so shards finish together.

    Cells differ by ~500x in update count, so round-robin leaves most cards
    idle while two cards chew through the small-batch cells.
    """
    ordered = sorted(cells, key=_cost, reverse=True)
    shards: list[list[Cell]] = [[] for _ in range(shard_count)]
    loads = [0] * shard_count
    for cell in ordered:
        target = loads.index(min(loads))
        shards[target].append(cell)
        loads[target] += _cost(cell)
    return shards


def _worker(device_index: int, cells: list[Cell]) -> None:
    torch.cuda.set_device(device_index)
    device = f"cuda:{device_index}"
    torch.set_num_threads(4)
    datasets: dict[str, PaperConeData] = {}
    for cell in cells:
        geometry, dim, ratio, seed, init, batch, _ = cell
        if _load_run(cell) is not None:
            print(f"[faithful] resume {_run_key(cell)}", flush=True)
            continue
        if ratio not in datasets:
            datasets[ratio] = load_dataset(cfg.PAPER_DATA_DIR, cfg.PAPER_DATASET, ratio)
        data = datasets[ratio]
        init_points = None
        if init == "warm":
            init_points = train_initializer(
                data,
                dim,
                seed,
                device,
                batch=batch,
                learning_rate=cfg.PAPER_INIT_LR,
                update_cap=_update_cap(cell),
            )
        print(f"[faithful] start {_run_key(cell)} gpu{device_index}", flush=True)
        result = train_one(
            geometry,
            dim,
            data,
            seed,
            device,
            init_points=init_points,
            batch=batch,
            learning_rate=_learning_rate(cell),
            update_cap=_update_cap(cell),
        )
        _save_run(cell, result)
        metrics = result["metrics"]
        print(
            f"[faithful] done {_run_key(cell)} "
            f"test_f1={metrics['test']['f1']:.2f} "
            f"test_auc={metrics['test']['auc']:.4f} "
            f"steps={metrics['training']['steps']} "
            f"loss/pair={metrics['loss_curve'][-1]['train_loss']:.4f} "
            f"secs={metrics['training']['elapsed_seconds']:.1f}",
            flush=True,
        )


def _metrics_of(cell: Cell) -> dict:
    with (_run_dir(cell) / "metrics.json").open() as handle:
        return json.load(handle)


def _per_positive_of(cell: Cell) -> np.ndarray:
    with np.load(_run_dir(cell) / "per_positive_auc.npz") as saved:
        return np.asarray(saved["test"], dtype=np.float64)


def _comparison(left: Cell, right: Cell) -> dict:
    entry = paired_bootstrap_delta(
        _per_positive_of(left),
        _per_positive_of(right),
        seed=cfg.BOOTSTRAP_SEED,
        samples=cfg.BOOTSTRAP_SAMPLES,
    )
    entry["f1_delta"] = float(
        _metrics_of(left)["test"]["f1"] - _metrics_of(right)["test"]["f1"]
    )
    entry["auc_delta_point"] = float(
        _metrics_of(left)["test"]["auc"] - _metrics_of(right)["test"]["auc"]
    )
    return entry


def main() -> None:
    started = time.time()
    device_count = torch.cuda.device_count()
    if device_count == 0:
        raise RuntimeError("The faithful cone round requires CUDA devices")
    cfg.FAITHFUL_RESULT_DIR.mkdir(parents=True, exist_ok=True)
    cells = _cells()
    shards = _shards(cells, device_count)
    print(
        f"[faithful] {len(cells)} cells over {device_count} gpus "
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
    missing = [cell for cell in cells if _load_run(cell) is None]
    if missing:
        raise RuntimeError(f"{len(missing)} cells have no usable artifact: {missing[:4]}")

    granularity: dict[str, dict] = {}
    for geometry in cfg.PAPER_GEOMETRIES:
        granularity[geometry] = {}
        for batch in cfg.FAITHFUL_BATCHES:
            cell = (
                geometry,
                cfg.FAITHFUL_GRID_DIM,
                cfg.FAITHFUL_GRID_RATIO,
                cfg.FAITHFUL_SEED,
                "cold",
                batch,
                "paper",
            )
            metrics = _metrics_of(cell)
            granularity[geometry][f"b{batch}"] = {
                "test_f1": metrics["test"]["f1"],
                "test_auc": metrics["test"]["auc"],
                "steps": metrics["training"]["steps"],
                "train_loss_per_pair": metrics["loss_curve"][-1]["train_loss"],
                "elapsed_seconds": metrics["training"]["elapsed_seconds"],
            }

    initialisation: dict[str, dict] = {}
    for dim in cfg.FAITHFUL_DIMS:
        for ratio in cfg.FAITHFUL_RATIOS:
            for init in cfg.FAITHFUL_INITS:
                for geometry in cfg.PAPER_GEOMETRIES:
                    cell = (
                        geometry,
                        dim,
                        ratio,
                        cfg.FAITHFUL_SEED,
                        init,
                        cfg.FAITHFUL_BASE_BATCH,
                        "coarse",
                    )
                    metrics = _metrics_of(cell)
                    initialisation[f"d{dim}_{ratio}_{init}_{geometry}"] = {
                        "test_f1": metrics["test"]["f1"],
                        "test_auc": metrics["test"]["auc"],
                        "train_loss_per_pair": metrics["loss_curve"][-1]["train_loss"],
                        "epochs": metrics["training"]["epochs"],
                    }

    comparisons: dict[str, dict] = {}
    for dim in cfg.FAITHFUL_DIMS:
        for ratio in cfg.FAITHFUL_RATIOS:
            for init in cfg.FAITHFUL_INITS:
                label = f"d{dim}_{ratio}_{init}"
                comparisons[label] = _comparison(
                    (
                        "poincare",
                        dim,
                        ratio,
                        cfg.FAITHFUL_SEED,
                        init,
                        cfg.FAITHFUL_BASE_BATCH,
                        "coarse",
                    ),
                    (
                        "euclid",
                        dim,
                        ratio,
                        cfg.FAITHFUL_SEED,
                        init,
                        cfg.FAITHFUL_BASE_BATCH,
                        "coarse",
                    ),
                )
    for batch in cfg.FAITHFUL_BATCHES:
        label = f"granularity_b{batch}"
        mode = "paper"
        comparisons[label] = _comparison(
            (
                "poincare",
                cfg.FAITHFUL_GRID_DIM,
                cfg.FAITHFUL_GRID_RATIO,
                cfg.FAITHFUL_SEED,
                "cold",
                batch,
                mode,
            ),
            (
                "euclid",
                cfg.FAITHFUL_GRID_DIM,
                cfg.FAITHFUL_GRID_RATIO,
                cfg.FAITHFUL_SEED,
                "cold",
                batch,
                mode,
            ),
        )

    depth, degree, cyclic = load_structure(cfg.PAPER_DATA_DIR, cfg.PAPER_DATASET)
    parent = load_dataset(
        cfg.PAPER_DATA_DIR, cfg.PAPER_DATASET, cfg.FAITHFUL_GRID_RATIO
    ).test_u
    shallow = depth[parent] <= 3
    stratification = {}
    for batch in cfg.FAITHFUL_BATCHES:
        mode = "paper"
        left = _per_positive_of(
            (
                "poincare",
                cfg.FAITHFUL_GRID_DIM,
                cfg.FAITHFUL_GRID_RATIO,
                cfg.FAITHFUL_SEED,
                "cold",
                batch,
                mode,
            )
        )
        right = _per_positive_of(
            (
                "euclid",
                cfg.FAITHFUL_GRID_DIM,
                cfg.FAITHFUL_GRID_RATIO,
                cfg.FAITHFUL_SEED,
                "cold",
                batch,
                mode,
            )
        )
        stratification[f"b{batch}"] = {
            "shallow_delta": float(left[shallow].mean() - right[shallow].mean()),
            "deep_delta": float(left[~shallow].mean() - right[~shallow].mean()),
        }

    summary = {
        "granularity": granularity,
        "initialisation": initialisation,
        "comparisons": comparisons,
        "shallow_vs_deep_delta": stratification,
        "experiment": {
            "reference": "Ganea, Becigneul & Hofmann, ICML 2018",
            "data": f"{cfg.PAPER_DATASET}_closure.tsv splits from the authors' release",
            "dims": list(cfg.FAITHFUL_DIMS),
            "ratios": list(cfg.FAITHFUL_RATIOS),
            "inits": list(cfg.FAITHFUL_INITS),
            "seed": int(cfg.FAITHFUL_SEED),
            "base_batch": int(cfg.FAITHFUL_BASE_BATCH),
            "batches": list(cfg.FAITHFUL_BATCHES),
            "reference_batch": 10,
            "epochs": int(cfg.FAITHFUL_EPOCHS),
            "learning_rate_by_geometry": dict(cfg.FAITHFUL_LR),
            "coarse_lr": dict(cfg.FAITHFUL_COARSE_LR),
            "coarse_update_cap": cfg.FAITHFUL_COARSE_CAP,
            "update_cap": cfg.FAITHFUL_UPDATE_CAP,
            "init_stage": {
                "epochs": int(cfg.PAPER_INIT_EPOCHS),
                "learning_rate": float(cfg.PAPER_INIT_LR),
                "burn_in": int(cfg.PAPER_INIT_BURN_IN),
                "negatives_power": float(cfg.PAPER_INIT_NEG_POWER),
                "rescale": float(cfg.PAPER_RESC_VECS),
            },
            "cyclic_nodes_excluded": cyclic,
            "elapsed_seconds": float(time.time() - started),
        },
    }
    with (cfg.FAITHFUL_RESULT_DIR / "summary.json").open("w") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)

    print("[faithful] granularity on dim 5 / 50% (cold start)", flush=True)
    for batch in cfg.FAITHFUL_BATCHES:
        eu = granularity["euclid"][f"b{batch}"]
        hy = granularity["poincare"][f"b{batch}"]
        print(
            f"  batch={batch:<6} steps={eu['steps']:<9} "
            f"euclid f1={eu['test_f1']:.2f} auc={eu['test_auc']:.4f} | "
            f"poincare f1={hy['test_f1']:.2f} auc={hy['test_auc']:.4f} | "
            f"df1={hy['test_f1']-eu['test_f1']:+.2f} "
            f"dauc={hy['test_auc']-eu['test_auc']:+.4f}",
            flush=True,
        )
    print("[faithful] initialisation axis at batch 256", flush=True)
    for dim in cfg.FAITHFUL_DIMS:
        for ratio in cfg.FAITHFUL_RATIOS:
            for init in cfg.FAITHFUL_INITS:
                eu = initialisation[f"d{dim}_{ratio}_{init}_euclid"]
                hy = initialisation[f"d{dim}_{ratio}_{init}_poincare"]
                print(
                    f"  d{dim:<3} {ratio:<10} {init:<5} "
                    f"euclid f1={eu['test_f1']:.2f} auc={eu['test_auc']:.4f} | "
                    f"poincare f1={hy['test_f1']:.2f} auc={hy['test_auc']:.4f} | "
                    f"df1={hy['test_f1']-eu['test_f1']:+.2f} "
                    f"dauc={hy['test_auc']-eu['test_auc']:+.4f}",
                    flush=True,
                )
    print(
        f"[faithful] complete elapsed_seconds="
        f"{summary['experiment']['elapsed_seconds']:.1f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
