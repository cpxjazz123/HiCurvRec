"""Issue #224 reference points for the relational KL: what does "no information" score?

KL(P_T || P_Q) is not zero for a P_Q that knows nothing: a uniform row scores
`log(k-1) - H(P_T)`.  Without that number the gate's 0.24 cannot be read as
"some relation was learned".  This script records, on the same held-out K3
instances and the same temperature:

  uniform       one row for every instance, uniform over the other nodes
  init          the RQ-VAE right after k-means init (step 0)
  shuffled_val  the init model's q vectors, with instances' q permuted (a P_Q that
                has the right marginal but no instance-specific structure)
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
sys.path.insert(0, str(REPO / "stage2_RQ-VAE" / "TIGER_drq_D1"))
sys.path.insert(0, str(REPO / "results" / "research" / "directed_graphlet_role"))

import train_rqvae as trainer  # noqa: E402
import relational_distill as rd  # noqa: E402
import trainable_student as ts  # noqa: E402
from directed_graphlets import load_adjacency  # noqa: E402
from directed_role import load_pool, load_vocabulary  # noqa: E402
from teacher_model import load_frozen_teacher  # noqa: E402

FAMILY = "K3"
INSTANCES = 512
TEMPERATURE = 0.1
SEED = 224


def main() -> None:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(SEED)
    embeddings = torch.from_numpy(np.load(REPO / "stage1_GeneEmbedding" / "output" /
                                          "sentence_t5.npy").astype("float32"))
    teacher, _ = load_frozen_teacher(HERE / "teacher" / "teacher_directed_v1.pth")
    teacher.to(device).eval()
    vocab = load_vocabulary("directed")
    heldout = load_pool("heldout", vocab, max_instances=INSTANCES)[FAMILY][0]
    graph = load_adjacency()
    n_items = graph.shape[0]
    rows, cols = graph.nonzero()
    edges = set((rows.astype(np.int64) * n_items + cols).tolist())
    width = heldout.shape[1]
    adjacency = np.zeros((len(heldout), width, width), dtype=np.float32)
    for i in range(width):
        for j in range(width):
            if i == j:
                continue
            keys = heldout[:, i].astype(np.int64) * n_items + heldout[:, j]
            adjacency[:, i, j] = [int(key) in edges for key in keys]
    items = embeddings[torch.from_numpy(heldout.reshape(-1))].to(device)
    target = rd.teacher_relation(teacher, items.view(len(heldout), width, -1),
                                 torch.from_numpy(adjacency).to(device), TEMPERATURE)

    uniform = torch.full((len(heldout), width, width), 1.0 / (width - 1)).to(device)
    eye = torch.eye(width, dtype=torch.bool, device=device)
    uniform = uniform.masked_fill(eye, 0.0)

    config = trainer._tokenizer_config()
    model = trainer.RQVAE(config, in_dim=embeddings.shape[1], distance=trainer.DISTANCE).to(device)
    trainer.initialize_tiger_weights(model)
    model.init_codebook(embeddings.to(device))
    path = ts.quantised_student_input(model, items, instances=len(heldout), nodes=width)
    init = rd.relation_fidelity(path.q, target, TEMPERATURE)
    permutation = torch.randperm(len(heldout), generator=torch.Generator().manual_seed(SEED))
    shuffled_val = rd.relation_fidelity(path.q[permutation], target, TEMPERATURE)

    payload = {
        "family": FAMILY, "instances": len(heldout), "temperature": TEMPERATURE,
        "uniform": rd.relation_fidelity(torch.zeros_like(path.q) + 0.0, target, TEMPERATURE)
        if False else {"kl": float((target * (torch.log(target.clamp_min(1e-12))
                                             - torch.log(uniform.clamp_min(1e-12)))).sum(-1).mean())},
        "init": {key: float(value) for key, value in init.items()},
        "shuffled_val": {key: float(value) for key, value in shuffled_val.items()},
        "target_entropy": float(-(target * torch.log(target.clamp_min(1e-12))).sum(-1).mean()),
    }
    (HERE / "relation_reference.json").write_text(json.dumps(payload, indent=1))
    print(json.dumps(payload, indent=1))


if __name__ == "__main__":
    main()
