"""Issue #224 ceiling: can *anything* that reads only item embeddings predict P_T?

The Student only ever sees the k items' embeddings (through the encoder).  Before
reading a null from `relation_gate.py` as "the loss is too weak", measure the best
a readout of those embeddings can do: an MLP on the concatenated 768-d vectors ->
k x k relation logits, trained to the Teacher's relation on TRAIN instances and
scored on the held-out instances.

If that ceiling is no better than the RQ-VAE's own relation error, the target is
not a function of what the Student is allowed to see, and no amount of lambda
will move it.  If it is much better, the information exists and the quantiser is
the thing losing it.
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

import relational_distill as rd  # noqa: E402
from directed_graphlets import load_adjacency  # noqa: E402
from directed_role import load_pool, load_vocabulary  # noqa: E402
from teacher_model import load_frozen_teacher  # noqa: E402

FAMILY = "K3"
INSTANCES = 512
STEPS = 3000
HIDDEN = 256
TEMPERATURE = 0.1
SEED = 224


def _adjacency(nodes, edges, n_items):
    width = nodes.shape[1]
    out = np.zeros((len(nodes), width, width), dtype=np.float32)
    for i in range(width):
        for j in range(width):
            if i == j:
                continue
            keys = nodes[:, i].astype(np.int64) * n_items + nodes[:, j]
            out[:, i, j] = [int(key) in edges for key in keys]
    return out


def main() -> None:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(SEED)
    embeddings = torch.from_numpy(np.load(REPO / "stage1_GeneEmbedding" / "output" /
                                          "sentence_t5.npy").astype("float32"))
    teacher, _ = load_frozen_teacher(HERE / "teacher" / "teacher_directed_v1.pth")
    teacher.to(device).eval()
    vocab = load_vocabulary("directed")
    train = load_pool("train", vocab, max_instances=INSTANCES)[FAMILY][0]
    heldout = load_pool("heldout", vocab, max_instances=INSTANCES)[FAMILY][0]
    graph = load_adjacency()
    n_items = graph.shape[0]
    rows, cols = graph.nonzero()
    edges = set((rows.astype(np.int64) * n_items + cols).tolist())

    def targets(nodes):
        adjacency = torch.from_numpy(_adjacency(nodes, edges, n_items)).to(device)
        items = embeddings[torch.from_numpy(nodes.reshape(-1))].to(device)
        return rd.teacher_relation(teacher, items.view(len(nodes), nodes.shape[1], -1),
                                   adjacency, TEMPERATURE), items

    train_target, train_items = targets(train)
    heldout_target, heldout_items = targets(heldout)
    flat_train = train_items.reshape(len(train), -1)
    flat_heldout = heldout_items.reshape(len(heldout), -1)

    # A linear readout, so "it did not fit" cannot be an architecture artefact: the
    # trivial solution (a per-instance constant row) is reachable in one step.
    model = torch.nn.Linear(flat_train.shape[1], train.shape[1] ** 2).to(device)
    torch.nn.init.zeros_(model.weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)
    for step in range(STEPS):
        logits = model(flat_train).view(-1, train.shape[1], train.shape[1])
        eye = torch.eye(train.shape[1], dtype=torch.bool, device=device)
        # A finite mask: 0 * log_softmax(-inf) is NaN and would poison the probe.
        logits = logits.masked_fill(eye, -1e9)
        probabilities = F.log_softmax(logits, dim=-1)
        loss = -(train_target * probabilities).sum(-1).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        logits = model(flat_heldout).view(-1, heldout.shape[1], heldout.shape[1])
        eye = torch.eye(heldout.shape[1], dtype=torch.bool, device=device)
        probabilities = F.softmax(logits.masked_fill(eye, -1e9), dim=-1)
        heldout_ce = float(-(heldout_target * torch.log(probabilities.clamp_min(1e-12))).sum(-1).mean())
        top1 = float((probabilities.argmax(-1) == heldout_target.argmax(-1)).float().mean())
        train_kl = float(loss.detach())

    def _entropy(target):
        return float(-(target * torch.log(target.clamp_min(1e-12))).sum(-1).mean())

    def _uniform_kl(target):
        nodes = target.shape[-1]
        eye = torch.eye(nodes, dtype=torch.bool, device=target.device)
        uniform = torch.full_like(target, 1.0 / (nodes - 1)).masked_fill(eye, 0.0)
        return float((target * (torch.log(target.clamp_min(1e-12))
                                - torch.log(uniform.clamp_min(1e-12)))).sum(-1).mean())

    entropy_train = _entropy(train_target)
    entropy_heldout = _entropy(heldout_target)
    payload = {
        "family": FAMILY, "instances": INSTANCES, "steps": STEPS, "readout": "linear",
        # Cross-entropy minus the target entropy is the KL the probe actually reaches.
        "train": {"ce": train_kl, "kl": train_kl - entropy_train,
                  "uniform_kl": _uniform_kl(train_target)},
        "heldout": {"ce": heldout_ce, "kl": heldout_ce - entropy_heldout,
                    "uniform_kl": _uniform_kl(heldout_target),
                    "top1": top1},
        "target_entropy": {"train": entropy_train, "heldout": entropy_heldout},
        "reference": {"note": "relation_gate.json, held-out KL at step 200 on the same "
                              "instances; uniform is the no-information score",
                      "uniform_kl": _uniform_kl(heldout_target),
                      "none_kl": 0.2858, "teacher_rel_kl": 0.2495,
                      "shuffled_rel_kl": 0.2440,
                      "init_kl": 0.5065, "shuffled_value_kl": 0.4865},
    }
    (HERE / "relation_ceiling.json").write_text(json.dumps(payload, indent=1))
    print(json.dumps(payload, indent=1))


if __name__ == "__main__":
    main()
