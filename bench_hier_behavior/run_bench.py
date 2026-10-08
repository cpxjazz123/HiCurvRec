"""Driver: build the run grid, shard it over the local GPUs, print the tables.

No arguments. Re-running skips any cell whose ``metrics.json`` already exists,
so the grid is resumable.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import torch
import torch.multiprocessing as mp

from . import bench_config as cfg
from .train import run_key, run_one


def build_runs() -> list[dict]:
    runs: list[dict] = []
    seen: set[str] = set()

    def add(arm, dim, tree_shape, mix, data_arm):
        key = run_key(arm, dim, tree_shape, mix, data_arm)
        if key in seen:
            return
        seen.add(key)
        runs.append({"arm": arm, "dim": dim, "tree": tree_shape, "mix": mix,
                     "data_arm": data_arm, "key": key})

    for tree_shape in cfg.STAGE1_TREES:
        for arm in cfg.STAGE1_ARMS:
            for dim in cfg.STAGE1_DIMS:
                add(arm, dim, tree_shape, cfg.DEFAULT_MIX, "tree")
    for tree_shape in cfg.STAGE2_TREES:
        for arm in cfg.STAGE2_ARMS:
            for dim in cfg.STAGE2_DIMS:
                add(arm, dim, tree_shape, cfg.DEFAULT_MIX, "tree")
    for tree_shape in cfg.STAGE3_TREES:
        for mix in cfg.STAGE3_MIXES:
            for arm in cfg.STAGE3_ARMS:
                for dim in cfg.STAGE3_DIMS:
                    for data_arm in cfg.DATA_ARMS:
                        add(arm, dim, tree_shape, mix, data_arm)
    return runs


def _worker(gpu: int, runs: list[dict]) -> None:
    torch.cuda.set_device(gpu)
    for run in runs:
        out_dir = Path(cfg.RESULT_ROOT) / run["key"]
        if (out_dir / "metrics.json").is_file():
            continue
        try:
            started = time.time()
            run_one(
                arm=run["arm"], dim=run["dim"], tree_shape=run["tree"],
                mix_name=run["mix"], data_arm=run["data_arm"],
                device=f"cuda:{gpu}",
            )
            print(f"[gpu{gpu}] {run['key']} {time.time() - started:.1f}s",
                  flush=True)
        except Exception as error:  # keep the grid going, report the cell
            print(f"[gpu{gpu}] {run['key']} FAILED: {error!r}", flush=True)


def _load_results(runs: list[dict]) -> list[dict]:
    rows = []
    for run in runs:
        path = Path(cfg.RESULT_ROOT) / run["key"] / "metrics.json"
        if path.is_file():
            with path.open() as handle:
                rows.append(json.load(handle))
    return rows


def _table(title: str, rows: list[dict], columns: list[str]) -> None:
    if not rows:
        return
    print(f"\n== {title}")
    header = " ".join(f"{name:>12}" for name in columns)
    print(f"{'arm':>16} {'dim':>4} {'tree':>11} {'mix':>12} {'data':>20} {header}")
    for row in sorted(rows, key=lambda r: (r["tree"], r["mix"], r["data_arm"],
                                           r["dim"], r["arm"])):
        cells = " ".join(f"{row.get(name, float('nan')):>12.4f}"
                         for name in columns)
        print(f"{row['arm']:>16} {row['dim']:>4} {row['tree']:>11} "
              f"{row['mix']:>12} {row['data_arm']:>20} {cells}")


def main() -> int:
    runs = build_runs()
    pending = [
        run for run in runs
        if not (Path(cfg.RESULT_ROOT) / run["key"] / "metrics.json").is_file()
    ]
    print(f"grid={len(runs)} pending={len(pending)}", flush=True)
    if pending:
        world = min(torch.cuda.device_count() or 1, len(pending))
        started = time.time()
        context = mp.get_context("spawn")
        workers = [
            context.Process(target=_worker, args=(index, pending[index::world]))
            for index in range(world)
        ]
        for worker in workers:
            worker.start()
        for worker in workers:
            worker.join()
        print(f"sweep finished in {time.time() - started:.1f}s", flush=True)

    rows = _load_results(runs)
    Path(cfg.RESULT_ROOT).mkdir(parents=True, exist_ok=True)
    with (Path(cfg.RESULT_ROOT) / "summary.json").open("w") as handle:
        json.dump(rows, handle, indent=2, sort_keys=True)

    _table("Q1 tree fidelity (distance_spearman is primary; lower gromov delta is better)",
           [r for r in rows if r["arm"] in cfg.STAGE1_ARMS],
           ["distance_spearman", "gromov_delta", "prefix_purity_L1",
            "tree_purity_L1", "level_ami_L1", "collision_rate",
            "final_recon_mse"])
    _table("Q2 held-out cone edges (higher coverage/auc is better)",
           [r for r in rows if r["arm"] in cfg.STAGE2_ARMS],
           ["cone_test_coverage", "cone_edge_auc", "cone_train_coverage",
            "ball_test_coverage", "cone_fit_saturated", "cone_k",
            "level_ami_L1"])
    _table("Q3 held-out next-item ranking (random hit@10 over the catalogue is 4.1e-4)",
           [r for r in rows if r["arm"] in cfg.STAGE3_ARMS],
           ["full_hit@10", "full_hit@100", "full_mrr",
            "full_top10_same_interest", "sampled_hit@10", "collision_rate"])
    print(f"\nsummary -> {Path(cfg.RESULT_ROOT) / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
