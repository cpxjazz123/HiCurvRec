"""Why the pair-level ER does not move the recommender (issue #107's immediate focus).

The campaign's repeated result is that a rising frozen Hard-L1 ER does not translate into
better Stage-3 quality. Before any further mechanism is designed, this module measures the
properties of the exported SIDs that a recommender actually consumes, on the archived arms
only: no training, no seed change, no control re-creation, and the frozen held-out
evaluation is used as a reported measurement and for nothing else.

Eight analyses, each cheap:

1. how much sequence information the SIDs carry (the next code predicted from the current
   one, against the marginal);
2. the drift between the code frequencies seen in training and the ones seen at test time;
3. the per-level token entropy and the Gini of the exported code distribution;
4. the pair-level positive-match rate split by the anchor's popularity, and the cold slice;
5. the collision rates per level;
6. the prefix distribution -- the L1 -> L2 and L2 -> L3 branch structure a beam search
   over the codes walks;
7. the Stage-3 outcome sliced by the SID properties of the target item;
8. the correlation between the ER delta and the Stage-3 delta across the arms.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
TRAIN_PARQUET = REPO_ROOT / "results" / "stage0_build_parquet" / "train.parquet"
TEST_PARQUET = REPO_ROOT / "results" / "stage0_build_parquet" / "test.parquet"
VALID_PAIRS = REPO_ROOT / "results" / "stage3_T5Train" / "monotonicity" / "behavior_pairs_valid_s42.npz"
STAGE3_ROOT = REPO_ROOT / "results" / "stage3_T5Train"

#: The archived control: the comparator of every arm.
CONTROL = "r8_control"


# ---------------------------------------------------------------------------------- helpers
def load_sids(path: Path) -> np.ndarray:
    path = Path(path)
    if path.suffix == ".npy":
        return np.load(path)
    raw = json.loads(path.read_text(encoding="utf-8"))
    matrix = np.zeros((max(int(key) for key in raw) + 1, 4), dtype=np.int64)
    for key, value in raw.items():
        matrix[int(key)] = value
    return matrix


def gini(values: np.ndarray) -> float:
    values = np.sort(np.asarray(values, dtype="float64"))
    size = values.size
    if size == 0 or values.sum() == 0:
        return 0.0
    index = np.arange(1, size + 1)
    return float((2.0 * (index * values).sum()) / (size * values.sum()) - (size + 1.0) / size)


def entropy(counts: np.ndarray) -> float:
    counts = np.asarray(counts, dtype="float64")
    total = counts.sum()
    if total <= 0:
        return 0.0
    probabilities = counts[counts > 0] / total
    return float(-(probabilities * np.log(probabilities)).sum())


def spearman(first: List[float], second: List[float]) -> Optional[float]:
    """The Spearman correlation, computed by hand (no scipy)."""
    if len(first) < 3:
        return None

    def ranks(values: List[float]) -> np.ndarray:
        order = np.argsort(values)
        ranks = np.empty(len(values), dtype="float64")
        ranks[order] = np.arange(len(values), dtype="float64")
        # average the ranks of ties
        counter = Counter(values)
        for value, count in counter.items():
            if count > 1:
                positions = [i for i, v in enumerate(values) if v == value]
                ranks[positions] = ranks[positions].mean()
        return ranks

    a = ranks(first)
    b = ranks(second)
    a = a - a.mean()
    b = b - b.mean()
    denominator = float(np.sqrt((a * a).sum() * (b * b).sum()))
    if denominator == 0:
        return None
    return float((a * b).sum() / denominator)


def _popularity() -> np.ndarray:
    """The number of interactions each item has in the training split."""
    import pandas as pd

    frame = pd.read_parquet(TRAIN_PARQUET, columns=["history", "target"])
    counts: Counter = Counter()
    for past, item in zip(frame["history"], frame["target"]):
        if past is not None:
            sequence = list(past) if isinstance(past, (list, tuple, np.ndarray)) else [past]
            for value in sequence:
                counts[int(value)] += 1
        if item is not None:
            counts[int(item)] += 1
    size = 0
    for value in counts:
        size = max(size, value)
    popularity = np.zeros(size + 1, dtype="int64")
    for item, count in counts.items():
        popularity[item] = count
    return popularity


def _train_pairs(limit: int = 400000) -> Tuple[np.ndarray, np.ndarray]:
    import pandas as pd

    frame = pd.read_parquet(TRAIN_PARQUET, columns=["history", "target"])
    sources: List[int] = []
    targets: List[int] = []
    for past, item in zip(frame["history"][:limit], frame["target"][:limit]):
        if past is None or item is None:
            continue
        sequence = list(past) if isinstance(past, (list, tuple, np.ndarray)) else [past]
        if not sequence:
            continue
        sources.append(int(sequence[-1]))
        targets.append(int(item))
    return np.asarray(sources, dtype=np.int64), np.asarray(targets, dtype=np.int64)


# ---------------------------------------------------------------------------------- analyses
def sequential_predictability(
    sids: np.ndarray, pairs: Tuple[np.ndarray, np.ndarray]
) -> Dict[str, Any]:
    """How much of the next code the current code predicts, against the marginal."""
    sources, targets = pairs
    out: Dict[str, Any] = {}
    for levels, name in ((1, "level_1"), (2, "level_1_2")):
        codes = sids[:, :levels]
        current = [tuple(row) for row in codes[sources].tolist()]
        following = [tuple(row) for row in codes[targets].tolist()]
        if not current:
            continue
        transitions: Counter = Counter(zip(current, following))
        marginal: Counter = Counter(following)
        best_follower: Dict[Any, Any] = {}
        best_count: Dict[Any, int] = {}
        for (before, after), count in transitions.items():
            if count > best_count.get(before, 0):
                best_count[before] = count
                best_follower[before] = after
        hits = sum(
            count
            for (before, after), count in transitions.items()
            if after == best_follower.get(before)
        )
        top1 = hits / len(current)
        baseline = max(marginal.values()) / len(current)
        out[name] = {
            "top1_accuracy": top1,
            "marginal_baseline": baseline,
            "excess_over_the_marginal": top1 - baseline,
            "distinct_codes_used": len(marginal),
            "pairs": len(current),
        }
    return out


def frequency_drift(
    sids: np.ndarray,
    train_pairs: Tuple[np.ndarray, np.ndarray],
    test_pairs: Tuple[np.ndarray, np.ndarray],
) -> Dict[str, Any]:
    """The per-level drift between the code frequencies of training and of test time."""
    out: Dict[str, Any] = {}
    for levels, name in ((1, "level_1"), (2, "level_1_2"), (3, "level_1_2_3")):
        codes = sids[:, :levels]
        train_counts = Counter(tuple(row) for row in codes[train_pairs[0]].tolist())
        test_counts = Counter(tuple(row) for row in codes[test_pairs[0]].tolist())
        keys = set(train_counts) | set(test_counts)
        total_train = sum(train_counts.values()) or 1
        total_test = sum(test_counts.values()) or 1
        drift = 0.5 * sum(
            abs(train_counts.get(key, 0) / total_train - test_counts.get(key, 0) / total_test)
            for key in keys
        )
        out[name] = {
            "total_variation_train_to_test": drift,
            "codes_in_train_not_in_test": len(set(train_counts) - set(test_counts)),
            "codes_in_test_not_in_train": len(set(test_counts) - set(train_counts)),
            "codes_in_train": len(train_counts),
            "codes_in_test": len(test_counts),
        }
    return out


def token_balance(sids: np.ndarray) -> Dict[str, Any]:
    """The entropy and the concentration of the exported code distribution, per level."""
    out: Dict[str, Any] = {}
    for levels, name in ((1, "level_1"), (2, "level_1_2"), (3, "level_1_2_3")):
        codes = sids[:, :levels]
        _, counts = np.unique(codes, axis=0, return_counts=True)
        out[name] = {
            "distinct_codes": int(len(counts)),
            "entropy": entropy(counts),
            "normalised_entropy": entropy(counts) / np.log(len(counts)) if len(counts) > 1 else 0.0,
            "gini": gini(counts),
            "largest_code_share": float(counts.max() / counts.sum()),
        }
    return out


def popularity_slices(
    sids: np.ndarray,
    popularity: np.ndarray,
    pairs: Tuple[np.ndarray, np.ndarray],
) -> Dict[str, Any]:
    """The pair-level positive-match rate by the anchor's popularity, and the cold slice."""
    sources, targets = pairs
    codes = sids[:, 0]
    if len(sources) == 0:
        return {}
    matched = codes[sources] == codes[targets]
    counts = np.bincount(sources, minlength=max(len(popularity), int(sources.max()) + 1))
    item_popularity = popularity[: len(counts)] if len(popularity) >= len(counts) else np.pad(
        popularity, (0, len(counts) - len(popularity))
    )
    order = np.argsort(item_popularity)
    quartiles = np.array_split(order, 4)
    out: Dict[str, Any] = {"quartiles": {}}
    for index, group in enumerate(quartiles, 1):
        if len(group) == 0:
            continue
        out["quartiles"][f"q{index}"] = {
            "items": int(len(group)),
            "mean_popularity": float(item_popularity[group].mean()),
            "positive_match_rate": float(matched[group].mean()),
        }
    cold = np.where(item_popularity == 0)[0]
    cold = np.intersect1d(cold, np.unique(sources))
    out["cold_slice"] = {
        "anchors_without_any_training_interaction": int(len(cold)),
        "positive_match_rate": float(matched[cold].mean()) if len(cold) else None,
    }
    return out


def collisions(sids: np.ndarray) -> Dict[str, Any]:
    """How many items share their code, per level and at the full three-token level."""
    out: Dict[str, Any] = {}
    for levels, name in ((1, "level_1"), (2, "level_1_2"), (3, "level_1_2_3")):
        codes = sids[:, :levels]
        _, counts = np.unique(codes, axis=0, return_counts=True)
        out[name] = {
            "items_sharing_their_code": int(counts[counts > 1].sum()),
            "collision_rate": float(1.0 - len(counts) / codes.shape[0]),
            "largest_cell": int(counts.max()),
        }
    return out


def prefix_structure(sids: np.ndarray) -> Dict[str, Any]:
    """The branch structure a beam search over the codes walks."""
    out: Dict[str, Any] = {}
    for parent_levels, child_levels, name in ((1, 2, "l1_to_l2"), (2, 3, "l2_to_l3")):
        parents = sids[:, :parent_levels]
        children = sids[:, :child_levels]
        branches: Dict[Any, set] = {}
        for parent, child in zip(parents.tolist(), children.tolist()):
            branches.setdefault(tuple(parent), set()).add(tuple(child))
        sizes = np.array([len(value) for value in branches.values()], dtype="float64")
        out[name] = {
            "distinct_parents": int(len(branches)),
            "mean_branches": float(sizes.mean()) if sizes.size else 0.0,
            "median_branches": float(np.median(sizes)) if sizes.size else 0.0,
            "tenth_percentile_branches": float(np.percentile(sizes, 10)) if sizes.size else 0.0,
            "parents_with_a_single_branch": int((sizes == 1).sum()),
        }
    return out


def stage3_slices(
    variant: str, sids: np.ndarray, popularity: np.ndarray
) -> Dict[str, Any]:
    """The Stage-3 outcome sliced by the SID properties of the target item.

    The Stage-3 shards carry the target's rank for every evaluated row; the rows are the
    test split in order, so the target item of a row is the test split's target column.
    """
    root = STAGE3_ROOT / variant / "logs" / "Amazon_2023_Instruments"
    if not root.is_dir():
        return {"available": False, "reason": f"no Stage-3 logs for {variant}"}
    runs = sorted(path for path in root.iterdir() if path.is_dir())
    if not runs:
        return {"available": False, "reason": "no Stage-3 run directory"}
    latest = runs[-1]
    shards = sorted((latest / "_test_shards").glob("*.json"))
    if not shards:
        return {"available": False, "reason": "no per-example shards"}
    import pandas as pd

    test = pd.read_parquet(TEST_PARQUET, columns=["target"])
    targets = test["target"].to_numpy()
    ranks: List[Optional[int]] = []
    for shard in shards:
        document = json.loads(shard.read_text(encoding="utf-8"))
        for record in document.get("per_user") or ():
            ranks.append(record.get("target_rank"))
    hits = np.array(
        [1.0 if (rank is not None and rank < 10) else 0.0 for rank in ranks], dtype="float64"
    )
    if len(hits) == 0 or len(hits) > len(targets):
        return {
            "available": False,
            "reason": f"the shards carry {len(hits)} rows but the test split has {len(targets)}",
        }
    items = targets[: len(hits)].astype("int64")
    code = sids[:, 0]
    cell_sizes = np.bincount(code)
    # the item's own level-0 code, and how many items share it
    item_cell = cell_sizes[code[items]]
    item_popularity = popularity[: max(len(popularity), int(items.max()) + 1)]
    item_popularity = np.pad(
        item_popularity, (0, max(0, len(items) - len(item_popularity)))
    )[: len(items)]
    out: Dict[str, Any] = {
        "available": True,
        "evaluated_rows": int(len(hits)),
        "recall@10": float(hits.mean()),
    }

    def slice_by(values: np.ndarray, edges: List[float], name: str) -> None:
        buckets: Dict[str, List[float]] = {}
        for index, value in enumerate(values):
            for low, high in zip(edges[:-1], edges[1:]):
                if value >= low and value < high:
                    buckets.setdefault(f"{name}[{low:g},{high:g})", []).append(hits[index])
                    break
        out[name] = {
            key: {"rows": len(values_), "recall@10": float(np.mean(values_))}
            for key, values_ in sorted(buckets.items())
        }

    slice_by(item_cell.astype("float64"), [0, 1, 2, 5, 20, 1e9], "l1_cell_size")
    slice_by(item_popularity.astype("float64"), [0, 1, 2, 5, 20, 1e9], "popularity")
    return out


# ---------------------------------------------------------------------------------- the report
def analyse(
    arms: Dict[str, Path],
    *,
    control: str = CONTROL,
    stage3_variants: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """Every analysis for every arm, with the control as the reference."""
    popularity = _popularity()
    train_pairs = _train_pairs()
    valid = np.load(VALID_PAIRS)
    test_pairs = (
        valid["anchors"].astype("int64"),
        valid["positives"].astype("int64"),
    )
    report: Dict[str, Any] = {"arms": {}, "control": control}
    for name, path in arms.items():
        sids = load_sids(Path(path) / "sids_for_hgrec.npy")
        entry: Dict[str, Any] = {
            "sid_path": str(path / "sids_for_hgrec.npy"),
            "sequential_predictability": sequential_predictability(sids, train_pairs),
            "frequency_drift": frequency_drift(sids, train_pairs, test_pairs),
            "token_balance": token_balance(sids),
            "popularity_slices": popularity_slices(sids, popularity, test_pairs),
            "collisions": collisions(sids),
            "prefix_structure": prefix_structure(sids),
        }
        if stage3_variants and name in stage3_variants:
            entry["stage3_slices"] = stage3_slices(stage3_variants[name], sids, popularity)
        report["arms"][name] = entry
    report["design_implications"] = design_implications(report, arms, control)
    return report


def design_implications(
    report: Dict[str, Any], arms: Dict[str, Path], control: str
) -> List[str]:
    """What the SIDs must satisfy for a downstream gain, stated as measurable requirements."""
    implications: List[str] = []
    reference = report["arms"].get(control)
    if reference is None:
        return implications
    for name, entry in report["arms"].items():
        if name == control:
            continue
        balance = entry["token_balance"]["level_1"]
        reference_balance = reference["token_balance"]["level_1"]
        # the TAIL of the level-0 usage tracks the downstream quality; the Gini, which
        # summarises the whole distribution, does not
        cell = entry["collisions"]["level_1"]["largest_cell"]
        reference_cell = reference["collisions"]["level_1"]["largest_cell"]
        if cell > reference_cell:
            implications.append(
                f"{name}: the largest level-0 cell holds {cell} items against the control's "
                f"{reference_cell}; the tail of the usage is what the recommender loses, and "
                "the whole-distribution Gini does not show it"
            )
        if balance["gini"] > reference_balance["gini"] + 0.02:
            implications.append(
                f"{name}: the level-0 usage is more concentrated than the control's "
                f"(Gini {balance['gini']:.3f} against {reference_balance['gini']:.3f}); a "
                "mechanism that concentrates the level-0 codes costs the recommender the "
                "candidates it can rank"
            )
        distinct = balance["distinct_codes"]
        reference_distinct = reference_balance["distinct_codes"]
        if distinct < reference_distinct:
            implications.append(
                f"{name}: fewer level-0 codes are in use than the control's "
                f"({distinct} against {reference_distinct}); a merge of two cells is how "
                "the codes concentrate, and the cells the recommender walks are the ones "
                "that disappeared"
            )
        drift = entry["frequency_drift"]["level_1"]["total_variation_train_to_test"]
        reference_drift = reference["frequency_drift"]["level_1"]["total_variation_train_to_test"]
        if drift > reference_drift + 0.05:
            implications.append(
                f"{name}: the level-0 code frequencies drift further from training to test "
                f"than the control's ({drift:.3f} against {reference_drift:.3f}), so the "
                "codes the recommender meets at test time are not the ones it was shaped by"
            )
        branches = entry["prefix_structure"]["l1_to_l2"]
        reference_branches = reference["prefix_structure"]["l1_to_l2"]
        if branches["mean_branches"] < reference_branches["mean_branches"] * 0.9:
            implications.append(
                f"{name}: the L1 -> L2 prefix tree is narrower than the control's "
                f"({branches['mean_branches']:.2f} against "
                f"{reference_branches['mean_branches']:.2f} branches per L1 code), which "
                "reduces the candidates a beam search can walk"
            )
        predictability = entry["sequential_predictability"]["level_1"]
        reference_predictability = reference["sequential_predictability"]["level_1"]
        if predictability["excess_over_the_marginal"] < reference_predictability[
            "excess_over_the_marginal"
        ]:
            implications.append(
                f"{name}: the codes carry less sequence information than the control's "
                f"(excess over the marginal {predictability['excess_over_the_marginal']:.3f} "
                f"against {reference_predictability['excess_over_the_marginal']:.3f}), so a "
                "recommender has less to predict from the previous item"
            )
    implications.append(
        "the requirement a mechanism has to meet, stated measurably: the largest level-0 cell "
        "must not hold more items than the control's largest cell, the number of level-0 "
        "codes in use must not fall below the control's, the L1 -> L2 prefix tree must keep at "
        "least as many branches per code, the train-to-test frequency drift must not grow, "
        "and the sequence information the codes carry must not fall -- while the pair-level ER "
        "rises. The tail of the usage and the width of the prefix tree are the two measures "
        "that track the downstream quality; the Gini, which summarises the whole "
        "distribution, does not"
    )
    return implications


def main(argv: Optional[List[str]] = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(prog="tools.harness2_genrec.stage3_transfer")
    parser.add_argument(
        "--arms",
        required=True,
        help="comma list of name=directory pairs, each holding sids_for_hgrec.npy",
    )
    parser.add_argument("--stage3", default="", help="comma list of name=variant pairs")
    parser.add_argument("--out", default="")
    args = parser.parse_args(argv)
    arms = {}
    for item in args.arms.split(","):
        if not item.strip():
            continue
        name, path = item.split("=", 1)
        arms[name.strip()] = Path(path.strip())
    variants = {}
    for item in args.stage3.split(","):
        if not item.strip():
            continue
        name, variant = item.split("=", 1)
        variants[name.strip()] = variant.strip()
    report = analyse(arms, stage3_variants=variants or None)
    text = json.dumps(report, indent=2, default=str)
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
