"""Does the community transition move a real next-item number?

The community-level test showed that the transition carries information the
identifiers do not have. The question that decides the whole reframe is one
level down: does that information help pick an ITEM?

This is a retrieval baseline, not the generator: the candidates are the most
popular items of the train split, each scored by

    log popularity + lambda * log P(community of the item | the prefix)

with lambda fitted on the TRAIN split and applied to the validation split. The
lambda = 0 arm is the popularity baseline, so the comparison answers exactly
one question: does the community prior add to what a pure popularity ordering
already knows?

The community prior is soft by construction. The transition's top-1 community
covers only a few percent of the next items, so a hard filter would throw away
the answer most of the time; a prior keeps the ranking's own evidence and only
tilts it.

Run:

    /home/wlia0047/ar57_scratch/wenyu/genrec_env/bin/python3.10 \
        tools/research/science/community_curvature/item_level_prior.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from tools.research.science.community_curvature import data  # noqa: E402

REPO = data.REPO
OUT = REPO / "results" / "research" / "issue121" / "community_curvature"
COMMUNITY = OUT / "pi_E_free_64_hyper_64_curvgcn.npy"
TRAIN = REPO / "results" / "stage0_build_parquet" / "train.parquet"
VALID = REPO / "results" / "stage0_build_parquet" / "valid.parquet"
CANDIDATES = 1000
EPS = 1e-9


def rows_of(frame) -> list[tuple[list[int], int]]:
    out = []
    for history, target in zip(frame["history"].tolist(), frame["target"].tolist()):
        if target is None:
            continue
        prefix = [int(item) for item in history]
        if prefix:
            out.append((prefix, int(target)))
    return out


def main() -> int:
    probabilities = np.load(COMMUNITY)
    communities = probabilities.argmax(axis=1)
    n_communities = int(probabilities.shape[1])

    train_rows = rows_of(pd.read_parquet(TRAIN, columns=["history", "target"]))
    valid_rows = rows_of(pd.read_parquet(VALID, columns=["history", "target"]))

    # the item popularity of the train split
    counts = np.zeros(probabilities.shape[0], dtype=np.float64)
    for prefix, target in train_rows:
        for item in prefix:
            counts[item] += 1.0
        counts[target] += 1.0
    popularity = np.log(np.maximum(counts, 1.0))
    candidates = np.argsort(-counts)[:CANDIDATES]
    candidate_communities = communities[candidates]
    candidate_popularity = popularity[candidates]

    # the community transition of the train split, and its product along the
    # prefix's community chain
    transitions = np.zeros((n_communities, n_communities), dtype=np.float64)
    for prefix, target in train_rows:
        transitions[int(communities[prefix[-1]]), int(communities[target])] += 1.0
    transitions /= np.maximum(transitions.sum(axis=1, keepdims=True), 1.0)
    log_transitions = np.log(np.maximum(transitions, EPS))

    def prior(prefix: list[int]) -> np.ndarray:
        """The next community's distribution along the prefix's chain."""
        accumulator = np.zeros(n_communities, dtype=np.float64)
        for position in range(len(prefix)):
            accumulator = accumulator + log_transitions[int(communities[prefix[position]])]
        accumulator -= accumulator.max()
        accumulator = np.exp(accumulator)
        total = accumulator.sum()
        if total <= 0:
            return np.full(n_communities, 1.0 / n_communities)
        return accumulator / total

    def evaluate(rows, lamdas, limit: int | None = None) -> dict:
        """The recall at 10 of the popularity ranking, tilted by the prior."""
        subset = rows if limit is None else rows[:limit]
        hits = {lam: 0 for lam in lamdas}
        ranks = {lam: [] for lam in lamdas}
        for prefix, target in subset:
            distribution = prior(prefix)
            tilt = np.log(np.maximum(distribution[candidate_communities], EPS))
            for lam in lamdas:
                score = candidate_popularity + lam * tilt
                order = np.argsort(-score)[:10]
                ranked = candidates[order]
                position = np.nonzero(ranked == target)[0]
                if position.size:
                    hits[lam] += 1
                    ranks[lam].append(int(position[0]) + 1)
        return {lam: {"recall@10": hits[lam] / len(subset),
                      "mrr@10": float(np.mean([1.0 / r for r in ranks[lam]])) if ranks[lam] else 0.0,
                      "n": len(subset)} for lam in lamdas}

    lamdas = [0.0, 0.25, 0.5, 1.0, 2.0, 4.0]
    train_metrics = evaluate(train_rows, lamdas, limit=40000)
    best = min(train_metrics, key=lambda lam: -train_metrics[lam]["recall@10"])
    valid_metrics = evaluate(valid_rows, sorted(set(lamdas + [best])))

    print("lambda fitted on the train split: %.2f" % best)
    print("%-8s %-12s %-12s %-10s" % ("lambda", "train R@10", "valid R@10", "valid MRR"))
    for lam in sorted(valid_metrics):
        print("%-8.2f %-12.4f %-12.4f %-10.4f" % (lam, train_metrics.get(lam, {}).get("recall@10", float("nan")),
                                                  valid_metrics[lam]["recall@10"], valid_metrics[lam]["mrr@10"]))
    baseline = valid_metrics[0.0]["recall@10"]
    tilted = valid_metrics[best]["recall@10"]
    print("popularity baseline %.4f -> with the community prior %.4f (%+.1f%%)"
          % (baseline, tilted, 100.0 * (tilted - baseline) / max(baseline, EPS)))

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "item_level_prior.json").write_text(json.dumps(
        {"lambda_fitted_on_train": best, "train": train_metrics, "valid": valid_metrics,
         "candidates": CANDIDATES, "communities": n_communities,
         "note": "a retrieval baseline, not the generator; the community enters as a soft prior"},
        indent=2, default=float) + "\n", encoding="utf-8")
    print("written:", OUT / "item_level_prior.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
