"""Issue #225 step 1: is the Teacher's relation score informative about the behaviour graph?

Everything downstream assumes the Teacher's node states encode something worth
preserving.  The check that has not been run yet is the simplest one: inside an
instance, does `s_T(i,j)` rank the pairs that actually have a behavioural edge
above the pairs that do not -- and does it beat plain semantic cosine on the same
pairs?  If the answers are "no" and "no", there is nothing to distil and the
composition of the loss is irrelevant.

Scores compared on identical pairs, identical labels, identical instances:

  semantic   cosine of the raw 768-d Stage1 embeddings
  teacher    cosine of the frozen Teacher's node states (the target of #224)
  sid        cosine of the frozen baseline RQ-VAE's hard q (what the SID carries today)

AUC is the Mann-Whitney statistic; the degree control repeats it on pairs matched
by the sum of the two nodes' degrees, because popular items are both more likely
to be adjacent and more likely to look similar to everything.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / "results" / "research" / "directed_graphlet_role"))

from directed_graphlets import load_adjacency  # noqa: E402
from directed_role import FAMILIES, load_pool, load_vocabulary  # noqa: E402
from frozen_sid import FrozenSID  # noqa: E402
from teacher_model import load_frozen_teacher  # noqa: E402

CAP = 4000
CHUNK = 2048


def auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Mann-Whitney AUC with ties counted as half."""
    positive = scores[labels > 0]
    negative = scores[labels <= 0]
    if len(positive) == 0 or len(negative) == 0:
        return float("nan")
    merged = np.concatenate([positive, negative])
    # average ranks for ties
    sorted_values, inverse, counts = np.unique(merged, return_inverse=True, return_counts=True)
    cumulative = np.cumsum(counts)
    average = cumulative - (counts - 1) / 2.0
    ranks = average[inverse]
    rank_positive = ranks[:len(positive)].sum()
    return float((rank_positive - len(positive) * (len(positive) + 1) / 2.0)
                 / (len(positive) * len(negative)))


def top1_precision(scores: np.ndarray, labels: np.ndarray, instances: np.ndarray) -> float:
    """Per instance, is the highest-scoring other node an actual neighbour?"""
    hits, total = 0, 0
    for instance in np.unique(instances):
        row = scores[instances == instance]
        label = labels[instances == instance]
        if len(row) < 1:
            continue
        best = int(np.argmax(row))
        hits += int(label[best] > 0)
        total += 1
    return float(hits / total) if total else float("nan")


def main() -> None:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    teacher, record = load_frozen_teacher(HERE / "teacher" / "teacher_directed_v1.pth")
    teacher.to(device).eval()
    semantic = np.load(REPO / "stage1_GeneEmbedding" / "output" / "sentence_t5.npy", mmap_mode="r")
    sid = FrozenSID()
    graph = load_adjacency()
    n_items = graph.shape[0]
    rows, cols = graph.nonzero()
    directed = set((rows.astype(np.int64) * n_items + cols).tolist())
    skeleton = set()
    for (source, target) in zip(rows.tolist(), cols.tolist()):
        skeleton.add((min(source, target), max(source, target)))

    vocab = load_vocabulary("directed")
    payload = {"teacher": record.get("extra", {}), "families": {}, "pooled": {}}
    pooled = {"semantic": [], "teacher": [], "sid": []}
    pooled_labels, pooled_instances, pooled_degree = [], [], []
    offset = 0

    for split in ("train", "heldout"):
        families = load_pool(split, vocab, max_instances=CAP)
        payload["families"][split] = {}
        for family in FAMILIES:
            nodes = families[family][0]
            width = nodes.shape[1]
            adjacency = np.zeros((len(nodes), width, width), dtype=np.float32)
            for i in range(width):
                for j in range(width):
                    if i == j:
                        continue
                    keys = nodes[:, i].astype(np.int64) * n_items + nodes[:, j]
                    adjacency[:, i, j] = [int(key) in directed for key in keys]

            teacher_scores = np.empty((len(nodes), width, width), dtype=np.float32)
            for start in range(0, len(nodes), CHUNK):
                block = nodes[start:start + CHUNK]
                items = torch.from_numpy(
                    np.asarray(semantic[block.reshape(-1)], dtype=np.float32)).to(device)
                with torch.no_grad():
                    _, _, hidden = teacher(items.view(len(block), width, -1),
                                           torch.from_numpy(adjacency[start:start + CHUNK]).to(device))
                normalised = torch.nn.functional.normalize(hidden, dim=-1)
                teacher_scores[start:start + CHUNK] = (normalised @ normalised.transpose(-1, -2)).cpu().numpy()

            semantic_vectors = np.asarray(semantic[nodes.reshape(-1)], dtype=np.float32).reshape(
                len(nodes), width, -1)
            semantic_vectors = semantic_vectors / np.linalg.norm(semantic_vectors, axis=-1, keepdims=True)
            semantic_scores = np.einsum("bik,bjk->bij", semantic_vectors, semantic_vectors)

            q = sid.q(nodes.reshape(-1)).numpy().reshape(len(nodes), width, -1)
            q = q / (np.linalg.norm(q, axis=-1, keepdims=True) + 1e-12)
            sid_scores = np.einsum("bik,bjk->bij", q, q)

            pairs_i, pairs_j = [], []
            for i in range(width):
                for j in range(width):
                    if i != j:
                        pairs_i.append(i)
                        pairs_j.append(j)
            pairs_i = np.array(pairs_i)
            pairs_j = np.array(pairs_j)

            def take(scores):
                return scores[:, pairs_i, pairs_j].reshape(-1)

            left = np.stack([nodes[:, i] for i in pairs_i], axis=1).reshape(-1)
            right = np.stack([nodes[:, j] for j in pairs_j], axis=1).reshape(-1)
            labels = np.array([1.0 if (min(int(a), int(b)), max(int(a), int(b))) in skeleton else 0.0
                               for a, b in zip(left.tolist(), right.tolist())])

            degree = np.zeros(n_items, dtype=np.float64)
            for (source, target) in zip(rows.tolist(), cols.tolist()):
                degree[source] += 1.0
                degree[target] += 1.0
            pair_degree = np.array([degree[int(a)] + degree[int(b)] for a, b in zip(left.tolist(), right.tolist())])

            instance_ids = np.repeat(np.arange(offset, offset + len(nodes)), len(pairs_i))
            rows_out = {}
            for name, scores in (("semantic", take(semantic_scores)),
                                 ("teacher", take(teacher_scores)),
                                 ("sid", take(sid_scores))):
                rows_out[name] = {"auc": auc(scores, labels),
                                  "top1": top1_precision(scores, labels, instance_ids)}
                pooled[name].append(scores)
            pooled_labels.append(labels)
            pooled_instances.append(instance_ids)
            pooled_degree.append(pair_degree)
            offset += len(nodes)
            payload["families"][split][family] = {
                "instances": int(len(nodes)), "pairs": int(len(labels)),
                "positive_rate": float(labels.mean()), **rows_out}

    labels = np.concatenate(pooled_labels)
    instances = np.concatenate(pooled_instances)
    degree = np.concatenate(pooled_degree)
    for name in pooled:
        scores = np.concatenate(pooled[name])
        payload["pooled"][name] = {"auc": auc(scores, labels),
                                   "top1": top1_precision(scores, labels, instances)}
        # degree-matched: keep pairs whose degree sum sits in the middle band
        band = np.quantile(degree, [0.25, 0.75])
        keep = (degree >= band[0]) & (degree <= band[1])
        payload["pooled"][name]["auc_degree_band"] = auc(scores[keep], labels[keep])
        payload["pooled"][name]["pairs_in_band"] = int(keep.sum())
    payload["pooled"]["positive_rate"] = float(labels.mean())
    payload["pooled"]["pairs"] = int(len(labels))
    (HERE / "relation_audit.json").write_text(json.dumps(payload, indent=1))
    print(json.dumps(payload["pooled"], indent=1))
    for split, families in payload["families"].items():
        print(split, {family: {name: round(values[name]["auc"], 3)
                               for name in ("semantic", "teacher", "sid")}
                      for family, values in families.items()})


if __name__ == "__main__":
    main()
