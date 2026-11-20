"""Issue #226: a leak-free graph Teacher, and whether it beats a matched embedding head.

Three variants, identical splits, negatives, heads, budget and initialisation:

    A  embedding   a trainable 768 -> 128 projection applied to each node, then the
                   same pair head (matched capacity without the graph),
    B  graph       in/out message passing over the TRAIN-observable graph, 128-d state,
                   then the same pair head,
    C  shuffled    B's architecture on a graph whose connections are degree-matched
                   shuffled, to separate "more parameters" from "the real topology".

Leakage rules (each has a regression test):

  * TRAIN pairs are split into train / valid / test by unordered pair; a valid or test
    pair is removed in **both** directions from the message graph and from every
    auxiliary label;
  * a second mask (15% of the train pairs, fixed seed) is removed from the message
    graph and used as the training positives, so the model never scores an edge it
    can read;
  * negatives are degree-matched and excluded against the **full** original graph
    (any direction), plus every masked positive; the blocked set is what the old
    sampler ignored;
  * direction labels come from the original graph and contain both classes (the node
    order is swapped at random, label = "the presented order is the true direction");
  * the orbit profile is recomputed on the observed graph only, and instances whose
    induced pattern no longer classifies are dropped with the coverage reported.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / "results" / "research" / "directed_graphlet_role"))

from relation_audit import auc  # noqa: E402

SEED = 226
TEST_FRACTION = 0.15
VALID_FRACTION = 0.15
TRAIN_MASK = 0.15
HIDDEN = 256
NODE_DIM = 128
ROLES = 376
EPOCHS = 40
BATCHES = 24
PAIRS_PER_BATCH = 4096
NO_GRAPH = "none"


def _pair_key(a: int, b: int):
    return (min(a, b), max(a, b))


def digest(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()


def split_pairs(edge_rows: np.ndarray, edge_cols: np.ndarray, seed: int = SEED) -> dict:
    """Pair-disjoint train / valid / test split of the TRAIN graph."""
    pairs = np.unique(np.stack([np.minimum(edge_rows, edge_cols),
                                np.maximum(edge_rows, edge_cols)], axis=1), axis=0)
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(pairs))
    n_test = int(len(pairs) * TEST_FRACTION)
    n_valid = int(len(pairs) * VALID_FRACTION)
    test = pairs[order[:n_test]]
    valid = pairs[order[n_test:n_test + n_valid]]
    train = pairs[order[n_test + n_valid:]]
    mask_rng = np.random.default_rng(seed + 1)
    masked_index = mask_rng.choice(len(train), int(len(train) * TRAIN_MASK), replace=False)
    masked = train[masked_index]
    observed_pairs = np.delete(train, masked_index, axis=0)
    return {"train": train, "valid": valid, "test": test,
            "masked": masked, "observed_pairs": observed_pairs,
            "split_sha": digest({"seed": seed,
                                 "train": train.tolist(), "valid": valid.tolist(),
                                 "test": test.tolist(), "masked": masked.tolist()})}


def materialise(payload: dict, edge_rows: np.ndarray, edge_cols: np.ndarray) -> dict:
    """Directed pieces derived from the split, with the leak invariants applied."""
    observed_set = set(map(tuple, payload["observed_pairs"].tolist()))
    directed = [(int(a), int(b)) for a, b in zip(edge_rows.tolist(), edge_cols.tolist())
                if (min(a, b), max(a, b)) in observed_set]
    full_directed = set((int(a), int(b)) for a, b in zip(edge_rows.tolist(), edge_cols.tolist()))
    masked_pairs = set(map(tuple, payload["masked"].tolist()))
    forbidden = set(full_directed) | {(b, a) for (a, b) in full_directed}
    forbidden |= masked_pairs | {(b, a) for (a, b) in masked_pairs}
    return {"observed_directed": directed, "observed_pairs": observed_set,
            "full_directed": full_directed, "masked_pairs": masked_pairs,
            "forbidden": forbidden}


def degree_matched_negative(anchor: int, positive: int, degrees: np.ndarray,
                            forbidden: set, rng) -> int | None:
    """A target whose degree matches `positive`'s, for the pair actually scored `(anchor, d)`.

    The old sampler checked `(positive, d)` while the head scored `(anchor, d)`, so a
    true `(anchor, d)` edge could be handed in as a negative.  Every rejection is on
    the scored pair, and the anchor and the positive are never candidates.
    """
    target = degrees[positive]
    candidates = np.flatnonzero(np.abs(degrees - target) <= max(1.0, 0.05 * target))
    candidates = candidates[(candidates != anchor) & (candidates != positive)]
    if len(candidates) == 0:
        return None
    order = rng.permutation(len(candidates))
    for index in order[:200]:
        candidate = int(candidates[index])
        if (anchor, candidate) in forbidden or (candidate, anchor) in forbidden:
            continue
        return candidate
    return None


def sample_negatives(pairs: np.ndarray, degrees: np.ndarray, forbidden: set, rng,
                     matcher=None, directed: set | None = None) -> tuple:
    """Aligned (positives, negative targets, failed row indices).

    A failed row is dropped from **both** arrays through one mask; the old code
    filtered only the negatives and then sliced the positives by count, which shifted
    every subsequent pair once a sample failed mid-list.
    """
    matcher = matcher or (lambda a, p, d, f, r: degree_matched_negative(a, p, d, f, r))
    positives, negatives, failures = [], [], []
    for index, (a, b) in enumerate(np.asarray(pairs).tolist()):
        # the split stores unordered pairs; orient each one along the direction that
        # actually exists in the TRAIN graph, otherwise half the "positives" are not
        # edges at all (this is what the first corrected run's assertion caught)
        a, b = int(a), int(b)
        if directed is not None and (a, b) not in directed and (b, a) in directed:
            a, b = b, a
        candidate = matcher(a, b, degrees, forbidden, rng)
        if candidate is None or int(candidate) in (int(a), int(b)):
            failures.append(index)
            continue
        positives.append([a, b])
        negatives.append(int(candidate))
    return (np.array(positives, dtype=np.int64),
            np.array(negatives, dtype=np.int64), failures)


def direction_metrics(confusion: np.ndarray) -> dict:
    """Balanced accuracy, macro F1 and both majority baselines.

    `majority_baseline` is the largest **true** class frequency (rows); the previous
    code used the largest prediction frequency (columns), which reported 0.95-1.00 for
    a nearly balanced test set.
    """
    confusion = np.asarray(confusion)
    total = confusion.sum()
    per_class = []
    for label in (0, 1):
        tp = confusion[label, label]
        fn = confusion[label].sum() - tp
        fp = confusion[:, label].sum() - tp
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        per_class.append(0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall))
    return {"balanced_accuracy": float(np.mean([confusion[0, 0] / max(confusion[0].sum(), 1),
                                                confusion[1, 1] / max(confusion[1].sum(), 1)])),
            "macro_f1": float(np.mean(per_class)),
            "majority_baseline": float(max(confusion.sum(1)) / max(total, 1)),
            "prediction_frequency_max": float(max(confusion.sum(0)) / max(total, 1)),
            "confusion": confusion.tolist(), "n": int(total)}


def direction_examples(masked_pairs: np.ndarray, full_directed: set, rng) -> dict:
    """Balanced direction task: node order is swapped at random, label = true direction."""
    rows, cols, labels = [], [], []
    for a, b in masked_pairs.tolist():
        forward, backward = (int(a), int(b)) in full_directed, (int(b), int(a)) in full_directed
        if forward == backward:
            continue  # reciprocal or absent: direction is not defined
        swap = bool(rng.integers(0, 2))
        if forward:
            rows.append(b if swap else a)
            cols.append(a if swap else b)
            labels.append(1 if swap else 0)
        else:
            rows.append(a if swap else b)
            cols.append(b if swap else a)
            # label 0 means "the presented order is the true direction", so the
            # backward branch uses the same rule as the forward one
            labels.append(1 if swap else 0)
    return {"rows": np.array(rows, dtype=np.int64), "cols": np.array(cols, dtype=np.int64),
            "labels": np.array(labels, dtype=np.int64)}


class GraphEncoder(nn.Module):
    def __init__(self, in_dim: int = 768, hidden: int = HIDDEN, out_dim: int = NODE_DIM):
        super().__init__()
        self.input = nn.Sequential(nn.Linear(in_dim, hidden), nn.GELU())
        self.self_proj = nn.Linear(hidden, hidden)
        self.in_proj = nn.Linear(hidden, hidden)
        self.out_proj = nn.Linear(hidden, hidden)
        self.output = nn.Linear(hidden, out_dim)

    def forward(self, features, adjacency):
        h = self.input(features)
        for _ in range(2):
            degree = torch.sparse.sum(adjacency, dim=1).to_dense().clamp(min=1.0).unsqueeze(-1)
            out_message = torch.sparse.mm(adjacency, h) / degree
            degree_in = torch.sparse.sum(adjacency, dim=0).to_dense().clamp(min=1.0).unsqueeze(-1)
            in_message = torch.sparse.mm(adjacency.transpose(0, 1), h) / degree_in
            h = F.gelu(self.self_proj(h) + self.out_proj(out_message) + self.in_proj(in_message))
        return self.output(h)


class PairHeads(nn.Module):
    """The same heads for every variant: only the node representation differs."""

    def __init__(self, dim: int = NODE_DIM, hidden: int = HIDDEN):
        super().__init__()
        self.edge = nn.Sequential(nn.Linear(3 * dim, hidden), nn.GELU(), nn.Linear(hidden, 1))
        self.direction = nn.Sequential(nn.Linear(3 * dim, hidden), nn.GELU(), nn.Linear(hidden, 2))
        self.orbit = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, ROLES))

    @staticmethod
    def pair(state, left, right):
        a, b = state[left], state[right]
        return torch.cat([a, b, a * b], dim=-1)


def shuffled_adjacency(adjacency: torch.Tensor, degrees: np.ndarray, seed: int,
                       forbidden: set | None = None) -> tuple:
    """Rewire targets to nodes with the same degree class: same size, different topology."""
    index = adjacency.indices().cpu().numpy()
    targets = []
    rng = np.random.default_rng(seed)
    order = np.argsort(degrees)
    groups = {}
    for position in order:
        groups.setdefault(int(degrees[position]), []).append(int(position))
    groups = {key: np.array(value) for key, value in groups.items()}
    new_cols = []
    for source in index[0]:
        bucket = groups[int(degrees[source])]
        new_cols.append(int(rng.choice(bucket[bucket != source])) if len(bucket) > 1 else int(source))
    new_index = np.stack([index[0], np.array(new_cols)])
    keep = new_index[0] != new_index[1]
    if forbidden is not None:
        blocked = np.array([(min(int(a), int(b)), max(int(a), int(b))) in forbidden
                            for a, b in new_index.T.tolist()])
        keep &= ~blocked
    skipped = int((~keep).sum())
    new_index = new_index[:, keep]
    tensor = torch.sparse_coo_tensor(torch.from_numpy(new_index),
                                     torch.ones(new_index.shape[1]),
                                     size=adjacency.shape).coalesce()
    return tensor, skipped


def bootstrap_ci(scores_a: np.ndarray, scores_b: np.ndarray,
                 n_boot: int = 1000, seed: int = SEED) -> dict:
    """Cluster bootstrap: one resample unit is a positive pair with its negative.

    `scores_*` are (n_pairs, 2) with column 0 the positive and column 1 the negative,
    so resampling rows keeps each pair's two observations together instead of treating
    them as independent points.
    """
    rng = np.random.default_rng(seed)
    differences = []
    n_pairs = scores_a.shape[0]
    flat_labels = np.array([1.0, 0.0])
    for _ in range(n_boot):
        pick = rng.integers(0, n_pairs, n_pairs)
        a = scores_a[pick].reshape(-1)
        b = scores_b[pick].reshape(-1)
        differences.append(auc(b, np.tile(flat_labels, n_pairs))
                           - auc(a, np.tile(flat_labels, n_pairs)))
    differences = np.array(differences)
    return {"mean": float(differences.mean()), "low": float(np.quantile(differences, 0.025)),
            "high": float(np.quantile(differences, 0.975)), "n": int(len(differences))}


def average_precision(scores: np.ndarray, labels: np.ndarray) -> float:
    order = np.argsort(-scores)
    sorted_labels = labels[order]
    hits = np.cumsum(sorted_labels)
    precision = hits / np.arange(1, len(sorted_labels) + 1)
    return float((precision * sorted_labels).sum() / max(sorted_labels.sum(), 1))


def observed_orbit_profile(instances_by_family, observed_set, n_items: int) -> tuple:
    """Orbit histogram recomputed on the observed graph only.

    Instances whose observed induced pattern leaves the five families are dropped and
    counted; nothing from a held-out pair can reach the labels because `observed_set`
    is the message graph itself.
    """
    from directed_graphlets import classify
    counts = np.zeros((n_items, ROLES), dtype=np.float64)
    kept, dropped = 0, 0
    for family, nodes, vocab in instances_by_family:
        for row in nodes:
            width = len(row)
            matrix = np.zeros((width, width), dtype=bool)
            for i in range(width):
                for j in range(width):
                    if i == j:
                        continue
                    matrix[i, j] = _pair_key(int(row[i]), int(row[j])) in observed_set and \
                        (int(row[i]), int(row[j])) in observed_set
            result = classify(matrix)
            if result is None:
                dropped += 1
                continue
            _skeleton, key, orbits = result
            subtype_id = vocab["subtype_of"].get(f"{family}:{key}")
            if subtype_id is None:
                dropped += 1
                continue
            orbit_ids = vocab["orbits_of"][subtype_id]
            for column, orbit in enumerate(orbits):
                counts[int(row[column]), orbit_ids[int(orbit)]] += 1.0
            kept += 1
    profile = counts / np.clip(counts.sum(1, keepdims=True), 1.0, None)
    coverage = float((counts.sum(1) > 0).mean())
    return profile, {"instances_kept": kept, "instances_dropped": dropped,
                     "item_coverage": coverage}


def train_variant(name: str, encode_fn, parameters, heads: PairHeads, features, adjacency,
                  positives: np.ndarray, negatives: np.ndarray, direction: dict,
                  role_target: torch.Tensor, device, seed: int = SEED) -> dict:
    torch.manual_seed(seed)
    optimizer = torch.optim.AdamW(parameters, lr=1e-3, weight_decay=1e-4)
    rng = np.random.default_rng(seed)
    trace = []
    for epoch in range(1, EPOCHS + 1):
        state = encode_fn(features, adjacency)
        total = 0.0
        for _ in range(BATCHES):
            pick = rng.integers(0, len(positives), PAIRS_PER_BATCH)
            positive = positives[pick]
            negative = negatives[pick]
            left = torch.from_numpy(np.concatenate([positive[:, 0], positive[:, 0]])).to(device)
            right = torch.from_numpy(np.concatenate([positive[:, 1], negative])).to(device)
            labels = torch.cat([torch.ones(len(positive)), torch.zeros(len(positive))]).to(device)
            edge_logits = heads.edge(heads.pair(state, left, right)).squeeze(-1)
            edge_loss = F.binary_cross_entropy_with_logits(edge_logits, labels)
            if len(direction["rows"]):
                chosen = rng.integers(0, len(direction["rows"]), PAIRS_PER_BATCH)
                d_logits = heads.direction(heads.pair(
                    state, torch.from_numpy(direction["rows"][chosen]).to(device),
                    torch.from_numpy(direction["cols"][chosen]).to(device)))
                d_loss = F.cross_entropy(d_logits,
                                         torch.from_numpy(direction["labels"][chosen]).to(device))
            else:
                d_loss = torch.zeros((), device=device)
            orbit_batch = torch.from_numpy(rng.integers(0, len(features), 2048)).to(device)
            orbit_logits = heads.orbit(state[orbit_batch])
            orbit_loss = -(role_target[orbit_batch] * F.log_softmax(orbit_logits, -1)).sum(-1).mean()
            total = total + edge_loss + 0.5 * d_loss + 0.2 * orbit_loss
        optimizer.zero_grad(set_to_none=True)
        (total / BATCHES).backward()
        torch.nn.utils.clip_grad_norm_(parameters, 5.0)
        optimizer.step()
        trace.append(float((total / BATCHES).detach()))
    return {"loss_trace": trace, "final_loss": trace[-1]}


def evaluate(state, heads, test_positive: np.ndarray, test_negative: np.ndarray,
             direction_test: dict, role_target: torch.Tensor, device) -> dict:
    with torch.no_grad():
        left = torch.from_numpy(np.concatenate([test_positive[:, 0], test_positive[:, 0]])).to(device)
        right = torch.from_numpy(np.concatenate([test_positive[:, 1], test_negative])).to(device)
        scores = heads.edge(heads.pair(state, left, right)).squeeze(-1).cpu().numpy()
        labels = np.concatenate([np.ones(len(test_positive)), np.zeros(len(test_positive))])
        metrics = {"auc": auc(scores, labels), "ap": average_precision(scores, labels),
                   "positive_rate": float(labels.mean()), "pairs": int(len(labels))}
        if len(direction_test["rows"]):
            logits = heads.direction(heads.pair(
                state, torch.from_numpy(direction_test["rows"]).to(device),
                torch.from_numpy(direction_test["cols"]).to(device)))
            prediction = logits.argmax(-1).cpu().numpy()
            target = direction_test["labels"]
            confusion = np.zeros((2, 2), dtype=np.int64)
            for truth, predicted in zip(target, prediction):
                confusion[truth, predicted] += 1
            metrics["direction"] = direction_metrics(confusion)
        orbit_logits = heads.orbit(state)
        ce = float(-(role_target * F.log_softmax(orbit_logits, -1)).sum(-1).mean())
        entropy = float(-(role_target * torch.log(role_target.clamp_min(1e-12))).sum(-1).mean())
        metrics["orbit"] = {"ce": ce, "kl": ce - entropy, "target_entropy": entropy}
        return metrics


def main() -> None:
    from directed_graphlets import load_adjacency
    from directed_role import FAMILIES, load_pool, load_vocabulary
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(SEED)
    graph = load_adjacency()
    n_items = graph.shape[0]
    rows, cols = graph.nonzero()
    payload = split_pairs(rows, cols)
    materialised = materialise(payload, rows, cols)
    observed = materialised["observed_directed"]
    index = np.array(observed, dtype=np.int64)
    adjacency = torch.sparse_coo_tensor(torch.from_numpy(index.T).to(device),
                                        torch.ones(len(index), dtype=torch.float32, device=device),
                                        size=(n_items, n_items)).coalesce()
    features = torch.from_numpy(
        np.load(REPO / "stage1_GeneEmbedding" / "output" / "sentence_t5.npy").astype(np.float32)
    ).to(device)
    degrees = (np.bincount(index[:, 0], minlength=n_items)
               + np.bincount(index[:, 1], minlength=n_items)).astype(np.float64)
    full = materialised["forbidden"]

    rng = np.random.default_rng(SEED)

    def validate(positives, negatives):
        """Every row must be a true positive and a genuinely absent negative pair."""
        violations = {"positive_not_edge": 0, "negative_is_edge": 0, "degenerate": 0,
                      "misaligned": int(len(negatives) != len(positives))}
        for (a, b), d in zip(positives.tolist(), negatives.tolist()):
            if (int(a), int(b)) not in materialised["full_directed"]:
                violations["positive_not_edge"] += 1
            if (int(a), int(d)) in materialised["full_directed"] or \
                    (int(d), int(a)) in materialised["full_directed"]:
                violations["negative_is_edge"] += 1
            if int(d) in (int(a), int(b)):
                violations["degenerate"] += 1
        return violations

    train_positive, train_negative, train_failures = sample_negatives(
        payload["masked"], degrees, full, rng, directed=materialised["full_directed"])
    test_positive, test_negative, test_failures = sample_negatives(
        payload["test"], degrees, full, rng, directed=materialised["full_directed"])
    train_check = validate(train_positive, train_negative)
    test_check = validate(test_positive, test_negative)
    assert not any(train_check[key] or test_check[key] for key in
                   ("positive_not_edge", "negative_is_edge", "degenerate", "misaligned")), \
        f"pair validation failed: train={train_check} test={test_check}"
    direction_train = direction_examples(payload["masked"], materialised["full_directed"], rng)
    direction_test = direction_examples(payload["test"], materialised["full_directed"], rng)

    vocab = load_vocabulary("directed")
    instances = [(family, load_pool("train", vocab, max_instances=8000)[family][0], vocab)
                 for family in FAMILIES]
    profile, coverage = observed_orbit_profile(instances, materialised["observed_pairs"], n_items)
    role_target = torch.from_numpy(profile.astype(np.float32)).to(device)
    print("orbit coverage", coverage, "kept", len(train_positive), len(test_positive), flush=True)

    forbidden_pairs = {(min(int(a), int(b)), max(int(a), int(b)))
                       for a, b in payload["test"].tolist() + payload["valid"].tolist()}
    shuffled_raw, shuffled_skipped = shuffled_adjacency(adjacency, degrees, SEED + 2,
                                                        forbidden=forbidden_pairs)
    shuffled = shuffled_raw.to(device)
    results = {"config": {"seed": SEED, "test_fraction": TEST_FRACTION, "valid_fraction": VALID_FRACTION,
                          "train_mask": TRAIN_MASK, "epochs": EPOCHS, "batches": BATCHES,
                          "pairs_per_batch": PAIRS_PER_BATCH, "split_sha": payload["split_sha"],
                          "orbit_coverage": coverage, "observed_edges": int(len(index)),
                          "masked_pairs": int(len(payload["masked"])),
                          "test_pairs": int(len(test_positive)),
                          "negative_failures": {"train": len(train_failures),
                                                "test": len(test_failures)},
                          "negative_failure_indices": {"train": train_failures[:50],
                                                       "test": test_failures[:50]},
                          "pair_validation": {"train": train_check, "test": test_check},
                          "shuffled_skipped_edges": shuffled_skipped,
                          "direction_train": {"n": int(len(direction_train["rows"])),
                                              "class_counts": np.bincount(
                                                  direction_train["labels"], minlength=2).tolist()},
                          "direction_test": {"n": int(len(direction_test["rows"])),
                                             "class_counts": np.bincount(
                                                 direction_test["labels"], minlength=2).tolist()}},
               "variants": {}}
    trained = {}
    for name in ("A_embedding", "B_graph", "C_shuffled"):
        torch.manual_seed(SEED)
        if name == "A_embedding":
            projection = nn.Linear(768, NODE_DIM).to(device)
            encode_fn = (lambda features, adjacency, projection=projection: projection(features))
            used_adjacency = None
            encoder_state = projection
        else:
            model = GraphEncoder().to(device)
            used_adjacency = adjacency if name == "B_graph" else shuffled
            encode_fn = (lambda features, adjacency, model=model, used=used_adjacency:
                         model(features, used))
            encoder_state = model
        heads = PairHeads().to(device)
        parameters = list(encoder_state.parameters()) + list(heads.parameters())
        training = train_variant(name, encode_fn, parameters, heads, features, used_adjacency,
                                 train_positive, train_negative, direction_train,
                                 role_target, device)
        with torch.no_grad():
            state = encode_fn(features, used_adjacency)
        metrics = evaluate(state, heads, test_positive, test_negative, direction_test,
                           role_target, device)
        checkpoint = HERE / f"graph_teacher2_{name}.pth"
        torch.save({"encoder": encoder_state.state_dict(), "heads": heads.state_dict(),
                    "config": {"variant": name, "node_dim": NODE_DIM, "hidden": HIDDEN}},
                   checkpoint)
        trained[name] = (encode_fn, heads, used_adjacency, state)
        results["variants"][name] = {"training": training, "metrics": metrics,
                                     "parameters": int(sum(p.numel() for p in parameters)),
                                     "checkpoint": checkpoint.name}
        print(name, json.dumps(metrics), flush=True)

    scores = {}
    for name, (encode_fn, heads, used_adjacency, _state) in trained.items():
        with torch.no_grad():
            state = encode_fn(features, used_adjacency)
            positive_scores = heads.edge(heads.pair(
                state, torch.from_numpy(test_positive[:, 0]).to(device),
                torch.from_numpy(test_positive[:, 1]).to(device))).squeeze(-1).cpu().numpy()
            negative_scores = heads.edge(heads.pair(
                state, torch.from_numpy(test_positive[:, 0]).to(device),
                torch.from_numpy(test_negative).to(device))).squeeze(-1).cpu().numpy()
            scores[name] = np.stack([positive_scores, negative_scores], axis=1)
    results["comparisons"] = {
        "B_minus_A": bootstrap_ci(scores["A_embedding"], scores["B_graph"]),
        "B_minus_C": bootstrap_ci(scores["C_shuffled"], scores["B_graph"])}
    (HERE / "graph_teacher2_corrected_report.json").write_text(json.dumps(results, indent=1))
    print(json.dumps(results["comparisons"], indent=1))


if __name__ == "__main__":
    main()
