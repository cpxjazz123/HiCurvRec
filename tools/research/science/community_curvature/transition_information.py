"""Does the community transition carry information the Semantic ID does not?

The community experiments so far asked whether the next item is in the SAME
community as the current one, and measured that at the chance level (0.0158
against a chance level of 0.0158). That question is the wrong one for a
generative recommender: a user who buys a shampoo may next buy a conditioner,
which is a different community and a perfectly stable relation.

This script therefore asks the question the framework was rebuilt around: after
a prefix, which community comes next? Six predictors are compared on the
INDEPENDENT validation split (the held-out interactions), all of them scoring
the next item's COMMUNITY:

| predictor | what it uses |
|---|---|
| global community | the train split's community frequencies |
| same community | the hypothesis that the next community is the current one |
| community transition | the train split's transition matrix, from the current community |
| transition over the prefix | the same matrix, applied along the prefix's communities |
| SID alone | the community distribution of the items sharing the prefix's SID |
| SID + community transition | the two log-scores combined, the weight fitted on the train split |

and one control: the transition matrix with its targets shuffled, which
separates a real transition signal from community popularity.

The decision rule, fixed before the run:

* a transition signal that beats both the global baseline and the shuffled
  control means the community carries transition information;
* the combined predictor beating the SID alone means that information is NOT
  already in the identifiers -- which is what would justify a
  community-guided generator.

Run:

    /home/wlia0047/ar57_wlia0047/env/bin/python3.10 ... (see the README)
"""
from __future__ import annotations

import collections
import json
import math
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
SIDS = {
    "frozen baseline": REPO / "results" / "stage2_RQ-VAE" / "TIGER_RQ-VAE" / "item_sids.json",
    "resample": REPO / "results" / "stage2_RQ-VAE" / "TIGER_BEHAVIOR_SAMPLE_RQ-VAE" / "resample" / "item_sids.json",
}
TRAIN = data.TRAIN_PARQUET
VALID = REPO / "results" / "stage0_build_parquet" / "valid.parquet"
EPS = 1e-9


def sequences(frame) -> list[tuple[list[int], int]]:
    """Every row's prefix and its next item, as item ids."""
    out = []
    for history, target in zip(frame["history"].tolist(), frame["target"].tolist()):
        if target is None:
            continue
        prefix = [int(item) for item in history]
        if not prefix:
            continue
        out.append((prefix, int(target)))
    return out


def transition_matrix(rows, communities, n_communities: int, shuffle_seed: int | None = None):
    """The community transition counts of the train split's sequences."""
    counts = np.zeros((n_communities, n_communities), dtype=np.float64)
    for prefix, target in rows:
        source = int(communities[prefix[-1]])
        destination = int(communities[target])
        counts[source, destination] += 1.0
    if shuffle_seed is not None:
        # the control: the destinations are permuted, so the source's
        # popularity survives and the transition structure does not
        rng = np.random.default_rng(shuffle_seed)
        counts = counts[rng.permutation(n_communities)]
    totals = counts.sum(axis=1, keepdims=True)
    return counts / np.maximum(totals, 1.0)


def metrics(probabilities: np.ndarray, targets: np.ndarray) -> dict:
    """The negative log likelihood and the hit rates of a next-community predictor."""
    picked = probabilities[np.arange(targets.size), targets]
    nll = float(-np.log(np.maximum(picked, EPS)).mean())
    order = np.argsort(-probabilities, axis=1)
    hits = {1: 0, 5: 0, 10: 0}
    for rank in (1, 5, 10):
        hits[rank] = int((order[:, :rank] == targets[:, None]).any(axis=1).sum())
    return {"nll": nll, "hit@1": hits[1] / targets.size, "hit@5": hits[5] / targets.size,
            "hit@10": hits[10] / targets.size, "n": int(targets.size)}


def main() -> int:
    probabilities = np.load(COMMUNITY)
    communities = probabilities.argmax(axis=1)
    n_communities = int(probabilities.shape[1])

    train_rows = sequences(pd.read_parquet(TRAIN, columns=["history", "target"]))
    valid_rows = sequences(pd.read_parquet(VALID, columns=["history", "target"]))
    print("train rows %d | valid rows %d" % (len(train_rows), len(valid_rows)))

    global_counts = np.bincount(communities, minlength=n_communities).astype(np.float64)
    global_p = global_counts / global_counts.sum()

    transition = transition_matrix(train_rows, communities, n_communities)
    shuffled = transition_matrix(train_rows, communities, n_communities, shuffle_seed=17)

    # the prefixes have different lengths, so they stay a list
    prefix_communities = [[int(communities[i]) for i in prefix] for prefix, _ in valid_rows]
    last_community = np.asarray([row[-1] for row in prefix_communities], dtype=np.int64)
    targets = np.asarray([int(communities[target]) for _, target in valid_rows], dtype=np.int64)

    results = {}

    results["global community"] = metrics(
        np.repeat(global_p[None, :], targets.size, axis=0), targets)
    results["community transition (last community)"] = metrics(transition[last_community], targets)
    results["community transition (shuffled control)"] = metrics(shuffled[last_community], targets)

    # the hypothesis the campaign started from, as a rate rather than a model
    same = float((targets == last_community).mean())
    base = float(global_p.max())
    results["same community (the old hypothesis)"] = {
        "rate": same, "base_rate_of_the_modal_community": base,
        "n": int(targets.size)}

    # the transition along the whole prefix, as a product over the prefix's
    # community transitions
    log_product = np.zeros((targets.size, n_communities), dtype=np.float64)
    longest = max(len(row) for row in prefix_communities)
    for position in range(longest):
        active = [index for index, row in enumerate(prefix_communities) if len(row) > position]
        if not active:
            continue
        source = np.asarray([prefix_communities[index][position] for index in active])
        log_product[active] += np.log(np.maximum(transition[source], EPS))
    log_product -= log_product.max(axis=1, keepdims=True)
    product = np.exp(log_product)
    product /= np.maximum(product.sum(axis=1, keepdims=True), EPS)
    results["community transition over the prefix"] = metrics(product, targets)

    # the identifiers' own predictive content about the next community
    for label, path in SIDS.items():
        if not path.is_file():
            continue
        table = json.loads(path.read_text())
        by_prefix = collections.defaultdict(collections.Counter)
        for item, row in table.items():
            key = tuple(int(v) for v in row[:3])
            by_prefix[key][int(communities[int(item)])] += 1
        train_keys = []
        for prefix, _ in train_rows:
            row = table.get(str(prefix[-1]))
            train_keys.append(tuple(int(v) for v in row[:3]) if row else None)
        last_community_for_train = [int(communities[prefix[-1]]) for prefix, _ in train_rows]
        sid_keys = []
        for prefix, _ in valid_rows:
            row = table.get(str(prefix[-1]))
            sid_keys.append(tuple(int(v) for v in row[:3]) if row else None)
        estimates = np.zeros((targets.size, n_communities), dtype=np.float64)
        for position, key in enumerate(sid_keys):
            counts = by_prefix.get(key)
            if not counts:
                continue
            total = sum(counts.values())
            for community, n in counts.items():
                estimates[position, community] = n / total
        global_fallback = np.repeat(global_p[None, :], targets.size, axis=0)
        covered = estimates.sum(axis=1) > 0
        estimates[~covered] = global_fallback[~covered]
        estimates /= np.maximum(estimates.sum(axis=1, keepdims=True), EPS)
        results["SID alone (%s)" % label] = metrics(estimates, targets)

        # the decisive test: the transition table conditioned on the identifiers
        # as well as on the community. If the SID adds nothing, this table is
        # as good as the community's alone; if it adds, the table is better.
        # the table is keyed by (source community, the prefix's SID prefix)
        table_counts = collections.defaultdict(collections.Counter)
        for (prefix, target), key in zip(train_rows, train_keys):
            if key is None:
                continue
            source = int(communities[prefix[-1]])
            table_counts[(source, key)][int(communities[target])] += 1
        conditioned_rows = np.zeros((targets.size, n_communities), dtype=np.float64)
        hit = 0
        for position, (prefix, target) in enumerate(valid_rows):
            key = sid_keys[position]
            source = int(communities[prefix[-1]])
            counts = table_counts.get((source, key))
            if not counts:
                conditioned_rows[position] = transition[source]
                continue
            total = sum(counts.values())
            for community, n in counts.items():
                conditioned_rows[position, community] = n / total
            hit += int(int(communities[target]) == max(counts, key=counts.get))
        results["transition conditioned on the SID too"] = metrics(
            conditioned_rows, targets)

        # the combination: the two log-scores with one weight, fitted on the
        # TRAIN split's rows and applied to the validation rows
        train_estimates = np.zeros((len(train_rows), n_communities), dtype=np.float64)
        for position, key in enumerate(train_keys):
            counts = by_prefix.get(key)
            if not counts:
                continue
            total = sum(counts.values())
            for community, n in counts.items():
                train_estimates[position, community] = n / total
        fallback = np.repeat(global_p[None, :], len(train_rows), axis=0)
        missing = train_estimates.sum(axis=1) <= 0
        train_estimates[missing] = fallback[missing]
        train_estimates /= np.maximum(train_estimates.sum(axis=1, keepdims=True), EPS)
        train_targets = np.asarray([int(communities[t]) for _, t in train_rows], dtype=np.int64)
        train_last = np.asarray([int(communities[p[-1]]) for p, _ in train_rows], dtype=np.int64)
        train_transition = transition[train_last]
        best = None
        for weight in np.linspace(0.0, 1.0, 21):
            combined = (1.0 - weight) * np.log(np.maximum(train_estimates, EPS)) \
                + weight * np.log(np.maximum(train_transition, EPS))
            combined -= combined.max(axis=1, keepdims=True)
            combined = np.exp(combined)
            combined /= np.maximum(combined.sum(axis=1, keepdims=True), EPS)
            value = metrics(combined, train_targets)["nll"]
            if best is None or value < best[1]:
                best = (weight, value)
        weight = float(best[0])
        combined = (1.0 - weight) * np.log(np.maximum(estimates, EPS)) \
            + weight * np.log(np.maximum(transition[last_community], EPS))
        combined -= combined.max(axis=1, keepdims=True)
        combined = np.exp(combined)
        combined /= np.maximum(combined.sum(axis=1, keepdims=True), EPS)
        results["SID + community transition (%s, w=%.2f)" % (label, weight)] = metrics(combined, targets)

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "community_transition_information.json").write_text(
        json.dumps(results, indent=2) + "\n", encoding="utf-8")

    print("%-46s %10s %8s %8s %8s" % ("predictor", "NLL", "hit@1", "hit@5", "hit@10"))
    for label, value in results.items():
        if "nll" in value:
            print("%-46s %10.4f %8.4f %8.4f %8.4f" % (label, value["nll"], value["hit@1"],
                                                        value["hit@5"], value["hit@10"]))
        else:
            print("%-46s %10s %8s %8s %8s" % (label, "rate %.4f" % value["rate"], "-", "-", "-"))
    print("written:", OUT / "community_transition_information.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
