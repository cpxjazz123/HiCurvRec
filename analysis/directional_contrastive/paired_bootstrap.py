"""User-level paired bootstrap for directional-contrastive Stage3 arms."""
import itertools
import json
from pathlib import Path

import math

import numpy as np

ROOT = Path("/home/wlia0047/ar57/wenyu/GeneRec")
STAGE3_ROOT = ROOT / "results/stage3_T5Train"
ARMS = (
    "resample",
    "resample_proj_sym",
    "resample_proj_dir",
    "crowd_couser_w3",
    "crowd_posattract_w3",
)
BOOTSTRAP_RESAMPLES = 1000
BOOTSTRAP_SEED = 42
METRICS = ("recall@10", "recall@5", "ndcg@10", "ndcg@5")


def global_row_index(rank, local_row_index, world_size):
    return rank + local_row_index * world_size


def newest_per_user_run(arm):
    variant_dir = STAGE3_ROOT / f"tiger_sample_{arm}" / "logs" / "Amazon_2023_Instruments"
    candidates = []
    if variant_dir.exists():
        for run_dir in variant_dir.iterdir():
            if not run_dir.is_dir():
                continue
            shard_paths = sorted((run_dir / "_test_shards").glob("rank*.json"))
            has_per_user = False
            for shard_path in shard_paths:
                try:
                    shard = json.loads(shard_path.read_text())
                except (OSError, json.JSONDecodeError):
                    continue
                if "per_user" in shard:
                    has_per_user = True
                    break
            if has_per_user:
                candidates.append((run_dir.stat().st_mtime_ns, run_dir, shard_paths))
    if not candidates:
        return None
    _, run_dir, shard_paths = max(candidates, key=lambda entry: entry[0])
    return run_dir, shard_paths


def load_and_verify(arm, run_dir, shard_paths):
    shards = []
    for path in shard_paths:
        with path.open(encoding="utf-8") as handle:
            shard = json.load(handle)
        if "per_user" not in shard:
            raise ValueError(f"{arm}: selected run has shard without per_user: {path}")
        shards.append(shard)
    shards.sort(key=lambda shard: int(shard["rank"]))
    ranks = [int(shard["rank"]) for shard in shards]
    if ranks != list(range(len(shards))):
        raise ValueError(f"{arm}: non-contiguous shard ranks: {ranks}")
    world_size = len(shards)
    users = {}
    reconstructed = {metric: 0.0 for metric in METRICS}
    total_n = 0
    for shard in shards:
        rank = int(shard["rank"])
        rows = sorted(shard["per_user"], key=lambda row: int(row["local_row_index"]))
        shard_n = int(shard["n_eval"])
        if len(rows) != shard_n:
            raise ValueError(f"{arm}: per_user length disagrees with n_eval in rank{rank}")
        rank_sums = {metric: 0.0 for metric in METRICS}
        for row in rows:
            local_index = int(row["local_row_index"])
            if local_index < 0 or local_index >= shard_n:
                raise ValueError(f"{arm}: invalid local row index {local_index} in rank{rank}")
            global_index = global_row_index(rank, local_index, world_size)
            target_rank = row["target_rank"]
            if target_rank is not None:
                target_rank = int(target_rank)
            values = {
                "recall@10": float(target_rank is not None and target_rank < 10),
                "recall@5": float(target_rank is not None and target_rank < 5),
                "ndcg@10": (1.0 / math.log2(target_rank + 2.0)) if target_rank is not None and target_rank < 10 else 0.0,
                "ndcg@5": (1.0 / math.log2(target_rank + 2.0)) if target_rank is not None and target_rank < 5 else 0.0,
            }
            if global_index in users:
                raise ValueError(f"{arm}: duplicate global row index {global_index}")
            users[global_index] = values
            for metric in METRICS:
                rank_sums[metric] += values[metric]
        # Preserve the original rank-ordered, within-rank accumulation semantics.
        for metric in METRICS:
            reconstructed[metric] += rank_sums[metric]
        total_n += shard_n

    test_path = run_dir / "test_final.json"
    if not test_path.is_file():
        raise ValueError(f"{arm}: missing test_final.json in {run_dir}")
    with test_path.open(encoding="utf-8") as handle:
        test_final = json.load(handle)
    if total_n != int(test_final["n_eval"]):
        raise ValueError(f"{arm}: reconstructed n_eval={total_n}, test_final n_eval={test_final['n_eval']}")
    for metric in METRICS:
        actual = float(test_final[f"test_{metric}"])
        recovered = reconstructed[metric] / total_n
        if recovered != actual:
            raise ValueError(
                f"{arm}: per-user reconstruction mismatch for {metric}: "
                f"reconstructed={recovered!r}, test_final={actual!r}"
            )
    return users


def paired_bootstrap(left, right, metric, rng):
    indices = sorted(set(left) & set(right))
    if len(indices) != len(left) or len(indices) != len(right):
        raise ValueError("Paired arms have different evaluated user-index sets")
    differences = np.fromiter(
        (left[index][metric] - right[index][metric] for index in indices),
        dtype=np.float64,
        count=len(indices),
    )
    estimates = np.empty(BOOTSTRAP_RESAMPLES, dtype=np.float64)
    for sample in range(BOOTSTRAP_RESAMPLES):
        draws = rng.integers(0, len(differences), size=len(differences))
        estimates[sample] = differences[draws].mean()
    low, high = np.percentile(estimates, [2.5, 97.5])
    return float(differences.mean()), float(low), float(high), len(indices)


def main():
    arm_users = {}
    for arm in ARMS:
        found = newest_per_user_run(arm)
        if found is None:
            print(f"{arm}: no per-user shard run found; skipped")
            continue
        run_dir, shard_paths = found
        print(f"{arm}: run_directory={run_dir}")
        arm_users[arm] = load_and_verify(arm, run_dir, shard_paths)
        print(f"{arm}: per-user reconstruction matches test_final.json exactly")

    available_pairs = [pair for pair in itertools.combinations(arm_users, 2)]
    if not available_pairs:
        print("No paired comparison is possible yet: fewer than two arms have per-user shards.")
        return
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    for left_arm, right_arm in available_pairs:
        print(f"Paired differences: {left_arm} - {right_arm} (1000 bootstrap resamples)")
        for metric in METRICS:
            mean, low, high, n_users = paired_bootstrap(
                arm_users[left_arm], arm_users[right_arm], metric, rng
            )
            print(
                f"  {metric}: mean={mean:.12g}, 95% percentile CI="
                f"[{low:.12g}, {high:.12g}], n_users={n_users}"
            )


if __name__ == "__main__":
    main()
