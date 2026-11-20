"""Issue #226 step 1: which part of the Teacher's state is stably attributable to an item?

The user's question is precise: an item appears in many graphlets and the Teacher
returns a different state each time; before distilling anything, ask which of that
variance is the item's and which is the graphlet's.

Three numbers decide it:

  ICC          between-item variance of the per-item mean / (between + within/M).
               Per dimension; 0 means the mean carries nothing item-specific.
  mean cosine  pairwise cosine between per-item means. Near 1 means every item's
               mean state points the same way -- a constant target.
  role TV      for items with >=2 instances, split their instances in half and
               compare the two role histograms.  Compared against the same
               statistic between two *different* items: if they match, the role
               histogram is not the item's, it is the pool's.

The role vocabulary is the directed-orbit label the pool already carries, so the
histogram is over the same classes the Teacher was trained to predict.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / "results" / "research" / "directed_graphlet_role"))

from directed_graphlets import load_adjacency  # noqa: E402
from directed_role import FAMILIES, instance_targets, load_pool, load_vocabulary  # noqa: E402
from teacher_model import load_frozen_teacher  # noqa: E402

CAP = 20000
CHUNK = 2048
MIN_OCCURRENCES = 3
ROLES = 376


def main() -> None:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    teacher, record = load_frozen_teacher(HERE / "teacher" / "teacher_directed_v1.pth")
    teacher.to(device).eval()
    semantic = np.load(REPO / "stage1_GeneEmbedding" / "output" / "sentence_t5.npy", mmap_mode="r")
    graph = load_adjacency()
    n_items = graph.shape[0]
    rows, cols = graph.nonzero()
    directed = set((rows.astype(np.int64) * n_items + cols).tolist())
    vocab = load_vocabulary("directed")

    states: dict[int, list[np.ndarray]] = defaultdict(list)
    roles: dict[int, list[int]] = defaultdict(list)
    for family in FAMILIES:
        nodes, orbits, subtypes = load_pool("train", vocab, max_instances=CAP)[family]
        targets = instance_targets(family, subtypes, orbits, vocab)
        valid = targets["subtype"] >= 0
        width = nodes.shape[1]
        adjacency = np.zeros((len(nodes), width, width), dtype=np.float32)
        for i in range(width):
            for j in range(width):
                if i == j:
                    continue
                keys = nodes[:, i].astype(np.int64) * n_items + nodes[:, j]
                adjacency[:, i, j] = [int(key) in directed for key in keys]
        for start in range(0, len(nodes), CHUNK):
            block = nodes[start:start + CHUNK]
            items = torch.from_numpy(
                np.asarray(semantic[block.reshape(-1)], dtype=np.float32)).to(device)
            with torch.no_grad():
                _, _, hidden = teacher(items.view(len(block), width, -1),
                                       torch.from_numpy(adjacency[start:start + CHUNK]).to(device))
            hidden = hidden.cpu().numpy()
            orbit_block = targets["orbit"][start:start + CHUNK]
            valid_block = valid[start:start + CHUNK]
            for row in range(len(block)):
                if not valid_block[row]:
                    continue
                for column in range(width):
                    item = int(block[row, column])
                    states[item].append(hidden[row, column])
                    role = int(orbit_block[row, column])
                    if 0 <= role < ROLES:
                        roles[item].append(role)
        print(family, "done", flush=True)

    repeated = {item: np.stack(values) for item, values in states.items()
                if len(values) >= MIN_OCCURRENCES}
    items = sorted(repeated)
    means = np.stack([repeated[item].mean(0) for item in items])
    within_var = np.stack([repeated[item].var(0, ddof=1) for item in items])
    counts = np.array([len(repeated[item]) for item in items], dtype=np.float64)
    between_var = means.var(0, ddof=1)
    icc = between_var / (between_var + within_var.mean(0) / counts.mean())
    normalised = means / (np.linalg.norm(means, axis=1, keepdims=True) + 1e-12)
    cosine = normalised @ normalised.T
    upper = cosine[np.triu_indices(len(items), 1)]

    def histogram(item):
        counts_ = np.bincount(roles[item], minlength=ROLES).astype(np.float64)
        return counts_ / max(counts_.sum(), 1)

    rng = np.random.default_rng(226)
    within_tv, between_tv = [], []
    for item in items[:4000]:
        occurrences = roles[item]
        if len(occurrences) < 2:
            continue
        index = rng.permutation(len(occurrences))
        half = len(index) // 2
        if half == 0:
            continue
        left = np.bincount(np.array(occurrences)[index[:half]], minlength=ROLES).astype(np.float64)
        right = np.bincount(np.array(occurrences)[index[half:]], minlength=ROLES).astype(np.float64)
        within_tv.append(0.5 * np.abs(left / max(left.sum(), 1) - right / max(right.sum(), 1)).sum())
    for _ in range(4000):
        left, right = rng.choice(items, 2, replace=False)
        within_tv_item = histogram(left)
        within_tv_other = histogram(right)
        between_tv.append(0.5 * np.abs(within_tv_item - within_tv_other).sum())

    payload = {
        "teacher": record.get("extra", {}),
        "items": len(items),
        "occurrences": int(counts.sum()),
        "min_occurrences": MIN_OCCURRENCES,
        "hidden": {
            "icc_median": float(np.median(icc)),
            "icc_p90": float(np.quantile(icc, 0.90)),
            "dims_icc_above_half": int((icc > 0.5).sum()),
            "dims": int(icc.size),
            "mean_cosine_between_items": {"mean": float(upper.mean()),
                                          "median": float(np.median(upper)),
                                          "p10": float(np.quantile(upper, 0.10)),
                                          "p90": float(np.quantile(upper, 0.90))},
            "verdict": "usable" if float(np.median(icc)) > 0.5 else "unusable",
        },
        "role_distribution": {
            "within_item_tv": {"mean": float(np.mean(within_tv)),
                               "median": float(np.median(within_tv)), "n": len(within_tv)},
            "between_item_tv": {"mean": float(np.mean(between_tv)),
                                "median": float(np.median(between_tv)), "n": len(between_tv)},
            "verdict": ("usable" if float(np.median(within_tv)) + 0.05
                        < float(np.median(between_tv)) else "unusable"),
        },
    }
    (HERE / "item_signal_audit.json").write_text(json.dumps(payload, indent=1))
    print(json.dumps(payload, indent=1))


if __name__ == "__main__":
    main()
