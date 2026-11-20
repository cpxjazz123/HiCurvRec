"""Issue #225: what can a fixed SID actually absorb from the behaviour graph?

The audit (`relation_audit.json`, `relation_probe.json`) leaves one well-posed
target.  The Teacher's hidden state predicts its own input adjacency perfectly
(AUC 1.000) and its raw cosine is at chance (0.495); a *learned* pairwise readout
of the **item embeddings** reaches AUC 0.597, while the frozen SID's own q carries
0.528.  So the only supervision a fixed SID can absorb is behavioural affinity
that the items already carry -- and the gap to close is 0.528 -> ~0.597.

This gate trains that, keeping the TIGER architecture and SID protocol untouched:

    s_Q(i,j) = cos(q_i, q_j) / tau
    L_aff    = BCE( s_Q(i,j), edge(i,j) )   over pairs of TRAIN instances only
    L_total  = L_recon + L_VQ + lambda * L_aff

Arms share one k-means initialisation and one weight:
    none          lambda = 0
    affinity      labels from the real TRAIN skeleton
    shuffled      the same labels, permuted across pairs (control)


Metric on held-out instances: AUC of s_Q against the real skeleton, plus SID health.
"""
from __future__ import annotations

import copy
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / "stage2_RQ-VAE" / "TIGER_drq_D1"))

import train_rqvae as trainer  # noqa: E402
import trainable_student as ts  # noqa: E402
from relation_audit import auc  # noqa: E402

FAMILY = "K1_3"
STEPS = 300
RECON_BATCH = 512
INSTANCES = 512
SEED = 225
TEMPERATURE = 0.1
AUX_SHARE = 0.10
EVAL_EVERY = 150


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


def _pair_index(width):
    return torch.tensor([[i, j] for i in range(width) for j in range(i + 1, width)])


def _sha(state_dict):
    import hashlib
    digest = hashlib.sha256()
    for key in sorted(state_dict):
        digest.update(key.encode())
        digest.update(state_dict[key].detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def affinity_loss(q, labels, pairs, temperature=TEMPERATURE):
    normalised = F.normalize(q, dim=-1)
    cosine = (normalised[:, pairs[:, 0]] * normalised[:, pairs[:, 1]]).sum(-1) / temperature
    return F.binary_cross_entropy_with_logits(cosine, labels)


def main() -> None:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(SEED)
    embeddings = torch.from_numpy(np.load(REPO / "stage1_GeneEmbedding" / "output" /
                                          "sentence_t5.npy").astype("float32"))
    sys.path.insert(0, str(REPO / "results" / "research" / "directed_graphlet_role"))
    from directed_graphlets import load_adjacency
    from directed_role import load_pool, load_vocabulary
    vocab = load_vocabulary("directed")
    train_nodes = load_pool("train", vocab, max_instances=INSTANCES)[FAMILY][0]
    heldout_nodes = load_pool("heldout", vocab, max_instances=INSTANCES)[FAMILY][0]
    graph = load_adjacency()
    n_items = graph.shape[0]
    rows, cols = graph.nonzero()
    directed = set((rows.astype(np.int64) * n_items + cols).tolist())
    skeleton = {(min(int(a), int(b)), max(int(a), int(b))) for a, b in zip(rows.tolist(), cols.tolist())}
    width = train_nodes.shape[1]
    pairs = _pair_index(width)

    def labels_for(nodes):
        out = np.zeros((len(nodes), len(pairs)), dtype=np.float32)
        for row_index, row in enumerate(nodes):
            for pair_index, (i, j) in enumerate(pairs.tolist()):
                key = (min(int(row[i]), int(row[j])), max(int(row[i]), int(row[j])))
                out[row_index, pair_index] = 1.0 if key in skeleton else 0.0
        return torch.from_numpy(out)

    train_labels = labels_for(train_nodes)
    heldout_labels = labels_for(heldout_nodes)
    train_items = embeddings[torch.from_numpy(train_nodes.reshape(-1))].to(device)
    heldout_items = embeddings[torch.from_numpy(heldout_nodes.reshape(-1))].to(device)

    config = trainer._tokenizer_config()
    torch.manual_seed(SEED)
    base = trainer.RQVAE(config, in_dim=embeddings.shape[1], distance=trainer.DISTANCE).to(device)
    trainer.initialize_tiger_weights(base)
    base.init_codebook(embeddings.to(device))
    base_state = {key: value.detach().clone() for key, value in base.state_dict().items()}

    # weight from the main-loss / affinity gradient ratio on the first batch
    probe = copy.deepcopy(base).to(device)
    probe_q = ts.quantised_student_input(probe, train_items[:64 * width],
                                         instances=64, nodes=width).q
    aux = affinity_loss(probe_q, train_labels[:64].to(device), pairs.to(device))
    aux_encoder = float(torch.autograd.grad(aux, [p for p in probe.encoder.parameters()
                                                  if p.requires_grad],
                                            allow_unused=True)[0].detach().norm())
    optimizer = torch.optim.AdamW(probe.parameters(), lr=trainer.LR)
    reconstructed, parts, _, _ = probe(embeddings[:RECON_BATCH].to(device))
    main = trainer.CRITERION(reconstructed, embeddings[:RECON_BATCH].to(device), parts)[0]
    main_encoder = float(torch.autograd.grad(main, [p for p in probe.encoder.parameters()
                                                    if p.requires_grad],
                                             allow_unused=True)[0].detach().norm())
    weight = float(AUX_SHARE * main_encoder / max(aux_encoder, 1e-12))
    print("weight", weight, "aux_encoder", aux_encoder, "main_encoder", main_encoder, flush=True)

    results = {"config": {"family": FAMILY, "steps": STEPS, "instances": len(train_nodes),
                          "heldout": len(heldout_nodes), "weight": weight,
                          "aux_share": AUX_SHARE, "seed": SEED,
                          "positive_rate": float(train_labels.mean())},
               "runs": {}}

    for arm in ("none", "affinity", "shuffled"):
        model = copy.deepcopy(base).to(device)
        model.load_state_dict(base_state, strict=True)
        optimizer = torch.optim.AdamW(model.parameters(), lr=trainer.LR)
        init_sha = _sha(model.state_dict())
        row = {"arm": arm, "init_sha": init_sha, "trace": []}
        generator = torch.Generator().manual_seed(SEED)
        order = torch.randperm(len(embeddings), generator=generator)
        labels = train_labels
        if arm == "shuffled":
            perm = torch.randperm(train_labels.numel(), generator=torch.Generator().manual_seed(SEED))
            labels = train_labels.view(-1)[perm].view(train_labels.shape)
        for step in range(1, STEPS + 1):
            if (step * RECON_BATCH) + RECON_BATCH > len(order):
                order = torch.randperm(len(embeddings), generator=generator)
            index = order[(step * RECON_BATCH) % max(len(order) - RECON_BATCH, 1):
                          (step * RECON_BATCH) % max(len(order) - RECON_BATCH, 1) + RECON_BATCH]
            batch = embeddings[index].to(device)
            optimizer.zero_grad(set_to_none=True)
            reconstructed, parts, _, _ = model(batch)
            main = trainer.CRITERION(reconstructed, batch, parts)[0]
            total = main
            aux_value = 0.0
            if arm != "none":
                start = (step * 64) % max(len(train_nodes) - 64, 1)
                stop = start + 64
                q = ts.quantised_student_input(model, train_items[start * width:stop * width],
                                               instances=stop - start, nodes=width).q
                aux = affinity_loss(q, labels[start:stop].to(device), pairs.to(device))
                total = main + weight * aux
                aux_value = float(aux.detach())
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), trainer.GRADIENT_CLIP_NORM)
            optimizer.step()
            if step % EVAL_EVERY == 0 or step == STEPS:
                with torch.no_grad():
                    q = ts.quantised_student_input(model, heldout_items,
                                                   instances=len(heldout_nodes), nodes=width).q
                    normalised = F.normalize(q, dim=-1)
                    scores = (normalised[:, pairs[:, 0]] * normalised[:, pairs[:, 1]]).sum(-1)
                    flat_scores = scores.reshape(-1).cpu().numpy()
                    flat_labels = heldout_labels.reshape(-1).numpy()
                    tokens = model.get_indices(heldout_items).cpu().numpy()
                row["trace"].append({
                    "step": step, "main": float(main.detach()), "aux": aux_value,
                    "heldout_auc": auc(flat_scores, flat_labels),
                    "l1_used": int(len(np.unique(tokens[:, 0]))),
                    "l3_used": int(len(np.unique(tokens[:, :3], axis=0))),
                    "collision": float(1.0 - len(np.unique(tokens[:, :3], axis=0)) / len(tokens)),
                    "spacing": float(trainer.codeword_spacing(model.rq.vq_layers[0].get_code_embs())),
                })
        row["final_sha"] = _sha(model.state_dict())
        results["runs"][arm] = row
        print(arm, json.dumps(row["trace"][-1]), flush=True)

    (HERE / "affinity_gate.json").write_text(json.dumps(results, indent=1))
    print(json.dumps({arm: {key: value for key, value in run["trace"][-1].items()
                            if key in ("heldout_auc", "collision", "l3_used")}
                      for arm, run in results["runs"].items()}, indent=1))


if __name__ == "__main__":
    main()
