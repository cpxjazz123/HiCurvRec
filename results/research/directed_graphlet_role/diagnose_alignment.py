"""Issue #221: is the directed subtype task aligned with what the head can see?

The joint diagnostic left one anomaly open: shuffled labels reach 100 % train
accuracy while the real labels stop at 61.5 %.  A label set that is *harder*
than random labels usually means the readout and the label disagree about what
the input is -- not that the input lacks the information.

Three checks, all on the frozen TIGER encoder so that nothing but the readout
moves:

  1. label consistency: how many instances share the same *unordered* node set
     (hence the same pooled feature) but carry different labels;
  2. readout alignment: the shipped permutation-invariant (mean+max) readout vs
     a position-aware (per-slot) readout, identical initialisation, identical
     optimiser, real labels and a shuffled-label control, growing sample sizes;
  3. what the pooled readout actually learns: its confusion pattern.

A readout that cannot fit the training set while a position-aware one can is an
architecture/label-alignment defect, not evidence about the graph.
"""
from __future__ import annotations

import json
import sys
from itertools import permutations
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
TEMPLATE = HERE / "template"
sys.path.insert(0, str(TEMPLATE))

import drq  # noqa: E402
from directed_role import load_pool, load_vocabulary  # noqa: E402
from model import RQVAE  # noqa: E402
from train_config import load_train_config  # noqa: E402

REPO = HERE.parents[2]
FAMILY = "P3"
SIZES = (256, 1024, 4096)
STEPS = 400


class PooledHead(nn.Module):
    """The shipped readout: mean+max pooling, so node order is discarded."""

    def __init__(self, dim: int, hidden: int = 96):
        super().__init__()
        self.phi = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, hidden), nn.GELU())
        self.head = nn.Sequential(nn.Linear(2 * hidden, hidden), nn.GELU(), nn.Linear(hidden, 2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.phi(x)
        return self.head(torch.cat([h.mean(1), h.amax(1)], -1))


class SlotHead(nn.Module):
    """Same capacity budget per node, but the three node slots stay separate."""

    def __init__(self, dim: int, hidden: int = 96):
        super().__init__()
        self.phi = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, hidden), nn.GELU())
        self.head = nn.Sequential(nn.Linear(3 * hidden, hidden), nn.GELU(), nn.Linear(hidden, 2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.phi(x)
        return self.head(h.reshape(len(h), -1))


def label_conflicts(nodes: np.ndarray, labels: np.ndarray) -> dict:
    """Instances whose *unordered* node set repeats with a different label."""
    canonical = np.sort(nodes, axis=1)
    order = np.lexsort(canonical.T[::-1])
    sorted_sets = canonical[order]
    sorted_labels = labels[order]
    start = np.r_[True, np.any(sorted_sets[1:] != sorted_sets[:-1], axis=1)]
    group = np.cumsum(start) - 1
    conflicts, duplicate_pairs = 0, 0
    for g in np.unique(group):
        rows = np.flatnonzero(group == g)
        if len(rows) > 1:
            duplicate_pairs += len(rows) * (len(rows) - 1) // 2
            if len(np.unique(sorted_labels[rows])) > 1:
                conflicts += len(rows)
    return {"duplicate_unordered_sets": int(duplicate_pairs),
            "instances_in_conflicting_sets": int(conflicts),
            "instances": int(len(nodes))}


def train_head(features: np.ndarray, labels: np.ndarray, kind: str,
               seed: int, steps: int = STEPS, lr: float = 1e-2) -> dict:
    torch.manual_seed(seed)
    x = torch.from_numpy(features)
    y = torch.from_numpy(labels).long()
    head = (PooledHead if kind == "pooled" else SlotHead)(x.shape[-1])
    optimizer = torch.optim.AdamW(head.parameters(), lr=lr)
    for _ in range(steps):
        loss = F.cross_entropy(head(x), y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        logits = head(x)
        predicted = logits.argmax(-1)
        accuracy = float((predicted == y).float().mean())
        minority = int(np.bincount(labels, minlength=2).min())
        # can the readout separate every training instance at all?
        per_class = [float((predicted[labels == c] == c).float().mean()) for c in (0, 1)
                     if (labels == c).any()]
    return {"kind": kind, "loss": float(loss.detach()), "train_accuracy": accuracy,
            "per_class_accuracy": per_class, "minority": minority, "steps": steps}


def main() -> None:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    config = load_train_config(TEMPLATE / "config" / "train_config.json")
    model_config = type("C", (), dict(hidden_sizes=tuple(config.HIDDEN_SIZES),
                                      codebook_num=config.CODEBOOK_NUM,
                                      codebook_size=tuple(config.CODEBOOK_SIZE),
                                      codebook_dim=config.CODEBOOK_DIM, dropout=0.0,
                                      beta=config.BETA, vq_type=config.VQ_TYPE,
                                      ema_decay=config.EMA_DECAY, fix_code_embs=False,
                                      sk_epsilon=config.SK_EPSILON, sk_iters=config.SK_ITERS))()
    embeddings = torch.from_numpy(np.load(REPO / "stage1_GeneEmbedding" / "output" /
                                          "sentence_t5.npy").astype("float32"))
    state = torch.load(REPO / "results" / "stage2_RQ-VAE" / "TIGER_RQ-VAE" /
                       "rqvae_best.pth", map_location="cpu")["state_dict"]
    torch.manual_seed(42)
    encoder = RQVAE(model_config, in_dim=embeddings.shape[1]).to(device)
    encoder.load_state_dict(state)
    encoder.eval()
    for parameter in encoder.parameters():
        parameter.requires_grad_(False)

    vocab = load_vocabulary("directed")
    train_families = load_pool("train", vocab, max_instances=60000)
    held_families = load_pool("heldout", vocab, max_instances=20000)
    train_nodes, _o, train_sub = train_families[FAMILY]
    held_nodes, _oh, held_sub = held_families[FAMILY]
    counts = np.bincount(train_sub)
    best_two = np.argsort(counts)[::-1][:2]
    mapping = {int(old): new for new, old in enumerate(best_two)}
    rng = np.random.default_rng(7)

    def balanced(nodes: np.ndarray, subtypes: np.ndarray, total: int):
        picks = []
        for label in best_two:
            index = np.flatnonzero(subtypes == label)
            picks.append(rng.choice(index, min(total // 2, len(index)), replace=False))
        chosen = np.concatenate(picks)
        rng.shuffle(chosen)
        return chosen, nodes[chosen], np.array([mapping[int(s)] for s in subtypes[chosen]])

    def encode(nodes: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            index = torch.from_numpy(nodes.reshape(-1)).long()
            latent = encoder.encoder(drq.gather(embeddings, index).to(device))
        return latent.view(len(nodes), nodes.shape[1], -1).cpu().numpy()

    results = {"family": FAMILY,
               "subtypes": [int(v) for v in best_two],
               "subtype_keys": [k for k, v in vocab["raw"][FAMILY].items()
                                if v in (int(best_two[0]), int(best_two[1]))],
               "support": [int(counts[v]) for v in best_two],
               "consistency": {}}

    # 1. label consistency on the full family
    pick_all, nodes_all, labels_all = balanced(train_nodes, train_sub, 40000)
    results["consistency"]["train_full"] = label_conflicts(nodes_all, labels_all)
    results["consistency"]["label_counts"] = np.bincount(labels_all).tolist()

    # 2. readout alignment, growing samples, identical init per condition
    curves = []
    for total in SIZES:
        pick, nodes, labels = balanced(train_nodes, train_sub, total)
        features = encode(nodes)
        shuffled = np.random.default_rng(2026).permutation(labels)
        row = {"n": int(len(labels)), "features_shape": list(features.shape),
               "conditions": {}}
        for kind in ("pooled", "slot"):
            for name, target in (("real", labels), ("shuffled", shuffled)):
                row["conditions"][f"{kind}_{name}"] = train_head(features, target, kind, seed=11)
        curves.append(row)
        print(json.dumps(row, indent=None)[:400], flush=True)
    results["curves"] = curves

    # 3. heldout check for the two readouts at the largest sample
    pick, held_nodes_bal, held_labels = balanced(held_nodes, held_sub, 2000)
    held_features = encode(held_nodes_bal)
    results["heldout"] = {}
    for kind in ("pooled", "slot"):
        pick, nodes, labels = balanced(train_nodes, train_sub, SIZES[-1])
        features = encode(nodes)
        torch.manual_seed(11)
        head = (PooledHead if kind == "pooled" else SlotHead)(features.shape[-1])
        optimizer = torch.optim.AdamW(head.parameters(), lr=1e-2)
        x, y = torch.from_numpy(features), torch.from_numpy(labels).long()
        for _ in range(STEPS):
            loss = F.cross_entropy(head(x), y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            pred = head(torch.from_numpy(held_features)).argmax(-1).numpy()
        results["heldout"][kind] = {"accuracy": float((pred == held_labels).mean()),
                                    "chance": 0.5}
    (HERE / "diagnose_alignment.json").write_text(json.dumps(results, indent=1))
    print(json.dumps({k: results[k] for k in ("consistency", "heldout")}, indent=1))


if __name__ == "__main__":
    main()
