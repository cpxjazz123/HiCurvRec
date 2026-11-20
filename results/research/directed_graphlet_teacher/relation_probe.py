"""Issue #225 step 1b: does the Teacher's hidden state predict edges better than semantics?

`relation_audit.py` shows the raw cosine of the Teacher's node states is at chance
for behavioural adjacency (AUC 0.495) while the semantic cosine is 0.580.  That
kills the *unsupervised* relation score, but not the Teacher: the hidden state was
never trained to make cosine a behavioural affinity.

So: fit the same supervised probe on two representations and compare on held-out
instances.  Features per pair are [a, b, a*b, |a-b|]; the probe is a two-layer MLP
trained on train pairs, scored by AUC on held-out pairs.

    semantic : 768-d Stage1 embedding
    teacher  : 128-d frozen Teacher node state

If the teacher probe beats the semantic probe, the Teacher carries structure that
the embeddings do not, and a learned relation score is the right target.  If it
only matches, the Teacher adds nothing to a design that already has the
embeddings -- which is the whole point of using it.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / "results" / "research" / "directed_graphlet_role"))

from directed_graphlets import load_adjacency  # noqa: E402
from directed_role import load_pool, load_vocabulary  # noqa: E402
from relation_audit import auc  # noqa: E402
from teacher_model import load_frozen_teacher  # noqa: E402

CAP = 4000
CHUNK = 2048
FAMILY = "K1_3"
STEPS = 1500
HIDDEN = 256


def _adjacency(nodes, directed, n_items):
    width = nodes.shape[1]
    out = np.zeros((len(nodes), width, width), dtype=np.float32)
    for i in range(width):
        for j in range(width):
            if i == j:
                continue
            keys = nodes[:, i].astype(np.int64) * n_items + nodes[:, j]
            out[:, i, j] = [int(key) in directed for key in keys]
    return out


def _pairs(width):
    return [(i, j) for i in range(width) for j in range(i + 1, width)]


def _features(representations, pairs):
    left = np.stack([[row[i] for row in representations] for i, _ in pairs], axis=1)
    right = np.stack([[row[j] for row in representations] for _, j in pairs], axis=1)
    return np.concatenate([left, right, left * right, np.abs(left - right)], axis=-1)


def main() -> None:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    teacher, _ = load_frozen_teacher(HERE / "teacher" / "teacher_directed_v1.pth")
    teacher.to(device).eval()
    semantic = np.load(REPO / "stage1_GeneEmbedding" / "output" / "sentence_t5.npy", mmap_mode="r")
    graph = load_adjacency()
    n_items = graph.shape[0]
    rows, cols = graph.nonzero()
    directed = set((rows.astype(np.int64) * n_items + cols).tolist())
    skeleton = {(min(int(a), int(b)), max(int(a), int(b))) for a, b in zip(rows.tolist(), cols.tolist())}

    vocab = load_vocabulary("directed")
    payload = {"family": FAMILY, "steps": STEPS, "hidden": HIDDEN, "representations": {}}
    store = {}
    for split in ("train", "heldout"):
        nodes = load_pool(split, vocab, max_instances=CAP)[FAMILY][0]
        width = nodes.shape[1]
        adjacency = _adjacency(nodes, directed, n_items)
        hidden = np.empty((len(nodes), width, 128), dtype=np.float32)
        for start in range(0, len(nodes), CHUNK):
            block = nodes[start:start + CHUNK]
            items = torch.from_numpy(
                np.asarray(semantic[block.reshape(-1)], dtype=np.float32)).to(device)
            with torch.no_grad():
                _, _, state = teacher(items.view(len(block), width, -1),
                                      torch.from_numpy(adjacency[start:start + CHUNK]).to(device))
            hidden[start:start + CHUNK] = state.cpu().numpy()
        store[split] = {
            "semantic": np.asarray(semantic[nodes.reshape(-1)], dtype=np.float32).reshape(
                len(nodes), width, -1),
            "teacher": hidden,
            "nodes": nodes,
        }
    pairs = _pairs(store["train"]["nodes"].shape[1])
    labels = {}
    for split in ("train", "heldout"):
        nodes = store[split]["nodes"]
        labels[split] = np.array([
            1.0 if (min(int(row[i]), int(row[j])), max(int(row[i]), int(row[j]))) in skeleton else 0.0
            for row in nodes for i, j in pairs])

    for name in ("semantic", "teacher"):
        width = store["train"]["nodes"].shape[1]
        train_x = torch.from_numpy(
            _features(store["train"][name], pairs).reshape(-1, 4 * store["train"][name].shape[-1])
        ).to(device)
        test_x = torch.from_numpy(
            _features(store["heldout"][name], pairs).reshape(-1, 4 * store["heldout"][name].shape[-1])
        ).to(device)
        train_y = torch.from_numpy(labels["train"]).to(device)
        test_y = labels["heldout"]
        torch.manual_seed(225)
        probe = torch.nn.Sequential(
            torch.nn.Linear(train_x.shape[-1], HIDDEN), torch.nn.GELU(),
            torch.nn.Linear(HIDDEN, HIDDEN), torch.nn.GELU(),
            torch.nn.Linear(HIDDEN, 1)).to(device)
        optimizer = torch.optim.AdamW(probe.parameters(), lr=1e-3, weight_decay=1e-2)
        gate = None
        with torch.no_grad():
            heldout_logits = probe(test_x).squeeze(-1).cpu().numpy()
        before = auc(heldout_logits, test_y)
        for step in range(STEPS):
            logits = probe(train_x).squeeze(-1)
            loss = F.binary_cross_entropy_with_logits(logits, train_y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            if step == 400:
                gate = float(loss.detach())
        with torch.no_grad():
            logits = probe(test_x).squeeze(-1).cpu().numpy()
        payload["representations"][name] = {
            "dim": int(train_x.shape[-1]),
            "train_pairs": int(len(train_y)), "heldout_pairs": int(len(test_y)),
            "positive_rate": float(train_y.mean()),
            "auc_before_training": before,
            "auc_heldout": auc(logits, test_y),
            "train_loss_at_step_400": gate,
        }
        print(name, json.dumps(payload["representations"][name]), flush=True)
    (HERE / "relation_probe.json").write_text(json.dumps(payload, indent=1))


if __name__ == "__main__":
    main()
