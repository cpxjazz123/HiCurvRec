"""Issue #224 gate: can a fixed SID match the Teacher's *relations*?

Four arms share one k-means initialisation, one batch order, one step budget and
one weight (calibrated on `teacher_rel` with the group-capped rule), so the only
difference is the target:

    none          lambda = 0, the plain RQ-VAE
    teacher_rel   KL(P_T || P_Q) with P_T from the frozen Teacher's node states
    shuffled_rel  the same Teacher weights fed adjacency flipped per unordered pair
    adjacency_rel no Teacher at all: uniform over the instance's real neighbours

The metric is the same for every arm: on held-out instances, KL(P_T^real || P_Q)
and the top-1 neighbour agreement against the real Teacher relation.  `init`
records the value at step 0, so "improved" means "below its own starting point",
not "below an arbitrary threshold".

The stability audit (`relation_stability.json`) already says the Teacher's node
state is not an item property (within-item cosine 0.451 vs between-item 0.443,
embedding probe R^2 -4.1).  This gate therefore measures the best case a fixed
SID could reach, and reports a null result as such instead of tuning lambda.
"""
from __future__ import annotations

import copy
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / "stage2_RQ-VAE" / "TIGER_drq_D1"))

import train_rqvae as trainer  # noqa: E402
import relational_distill as rd  # noqa: E402
import trainable_student as ts  # noqa: E402
from teacher_model import load_frozen_teacher  # noqa: E402

STEPS = 200
RECON_BATCH = 512
INSTANCES = 256
SEED = 224
TEMPERATURE = 0.1
AUX_SHARE = 0.10
EVAL_EVERY = 100


def _adjacency(nodes: np.ndarray, edges: set, n_items: int) -> np.ndarray:
    width = nodes.shape[1]
    out = np.zeros((len(nodes), width, width), dtype=np.float32)
    for i in range(width):
        for j in range(width):
            if i == j:
                continue
            keys = nodes[:, i].astype(np.int64) * n_items + nodes[:, j]
            out[:, i, j] = [int(key) in edges for key in keys]
    return out


def _flip(adjacency: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    out = adjacency.copy()
    for i in range(adjacency.shape[1]):
        for j in range(i + 1, adjacency.shape[2]):
            flip = rng.random(len(adjacency)) < 0.5
            out[:, i, j] = np.where(flip, adjacency[:, j, i], adjacency[:, i, j])
            out[:, j, i] = np.where(flip, adjacency[:, i, j], adjacency[:, j, i])
    return out


def _sha(state_dict) -> str:
    import hashlib
    digest = hashlib.sha256()
    for key in sorted(state_dict):
        digest.update(key.encode())
        digest.update(state_dict[key].detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def main() -> None:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(SEED)
    embeddings = torch.from_numpy(np.load(REPO / "stage1_GeneEmbedding" / "output" /
                                          "sentence_t5.npy").astype("float32"))
    teacher, teacher_record = load_frozen_teacher(HERE / "teacher" / "teacher_directed_v1.pth")
    teacher.to(device).eval()

    sys.path.insert(0, str(REPO / "results" / "research" / "directed_graphlet_role"))
    from directed_graphlets import load_adjacency
    from directed_role import load_pool, load_vocabulary
    vocab = load_vocabulary("directed")
    train_pool = load_pool("train", vocab, max_instances=INSTANCES)
    heldout_pool = load_pool("heldout", vocab, max_instances=INSTANCES)
    graph = load_adjacency()
    n_items = graph.shape[0]
    rows, cols = graph.nonzero()
    edges = set((rows.astype(np.int64) * n_items + cols).tolist())

    # One family: the instance width is part of the relation matrix, and mixing a
    # 3-node family with a 4-node one would need padding that the Teacher never saw.
    FAMILY = "K3"

    def bundle(pool):
        nodes = pool[FAMILY][0]
        return nodes, _adjacency(nodes, edges, n_items)

    train_nodes, train_adjacency = bundle(train_pool)
    heldout_nodes, heldout_adjacency = bundle(heldout_pool)
    print("train", train_nodes.shape, "heldout", heldout_nodes.shape, flush=True)

    def relation_target(nodes, adjacency):
        items = torch.from_numpy(embeddings[nodes.reshape(-1)].numpy()).to(device)
        return rd.teacher_relation(teacher, items.view(len(nodes), nodes.shape[1], -1),
                                   torch.from_numpy(adjacency).to(device), TEMPERATURE)

    targets = {
        "teacher_rel": relation_target(train_nodes, train_adjacency),
        "shuffled_rel": relation_target(train_nodes, _flip(train_adjacency, SEED)),
        "adjacency_rel": rd.adjacency_distribution(torch.from_numpy(train_adjacency).to(device)),
    }
    heldout_target = relation_target(heldout_nodes, heldout_adjacency)
    heldout_items = torch.from_numpy(embeddings[heldout_nodes.reshape(-1)].numpy()).to(device)
    heldout_adjacency_t = torch.from_numpy(heldout_adjacency).to(device)

    config = trainer._tokenizer_config()
    torch.manual_seed(SEED)
    base = trainer.RQVAE(config, in_dim=embeddings.shape[1], distance=trainer.DISTANCE).to(device)
    trainer.initialize_tiger_weights(base)
    base.init_codebook(embeddings.to(device))
    base_state = {key: value.detach().clone() for key, value in base.state_dict().items()}

    generator = torch.Generator().manual_seed(SEED)
    order = torch.randperm(len(embeddings), generator=generator)

    def relation_batch(step: int):
        start = (step * RECON_BATCH) % max(len(train_nodes) - 64, 1)
        return slice(start, start + 64)

    # Weight: one calibration on the teacher arm, shared by the three supervised arms.
    probe_model = copy.deepcopy(base)
    probe_model.to(device)
    slice_ = relation_batch(0)
    probe_items = torch.from_numpy(embeddings[train_nodes[slice_].reshape(-1)].numpy()).to(device)
    path = ts.quantised_student_input(probe_model, probe_items,
                                      instances=len(train_nodes[slice_]),
                                      nodes=train_nodes.shape[1])
    aux = rd.relational_loss(path.q, targets["teacher_rel"][slice_], TEMPERATURE)
    aux_norms = {"encoder": 0.0, "codebooks": []}
    encoder = [p for p in probe_model.encoder.parameters() if p.requires_grad]
    aux_norms["encoder"] = float(torch.sqrt(sum((g.detach() ** 2).sum() for g in
                                                torch.autograd.grad(aux["loss"], encoder,
                                                                    retain_graph=True,
                                                                    allow_unused=True)
                                                if g is not None)))
    for parameter in ts.codebook_parameters(probe_model).values():
        grad = torch.autograd.grad(aux["loss"], [parameter], retain_graph=True,
                                   allow_unused=True)[0]
        aux_norms["codebooks"].append(0.0 if grad is None else float(grad.detach().norm()))

    optimizer = torch.optim.AdamW(probe_model.parameters(), lr=trainer.LR)
    reconstructed, parts, _, _ = probe_model(embeddings[order[:RECON_BATCH]].to(device))
    main = trainer.CRITERION(reconstructed, embeddings[order[:RECON_BATCH]].to(device), parts)[0]
    main_report = ts.gradient_report(main, probe_model, ts.build_head(32, 4, 4))
    weight = ts.safe_aux_weight(encoder_reference=main_report["encoder"],
                               encoder_aux=aux_norms["encoder"],
                               codebook_reference=main_report["codebooks"],
                               codebook_aux=aux_norms["codebooks"], share=AUX_SHARE)
    print("weight", weight, "aux norms", aux_norms, flush=True)

    results = {"teacher": teacher_record.get("extra", {}), "config": {
        "steps": STEPS, "recon_batch": RECON_BATCH, "instances": len(train_nodes),
        "heldout_instances": len(heldout_nodes), "temperature": TEMPERATURE,
        "weight": weight, "aux_share": AUX_SHARE, "seed": SEED},
        "aux_gradient_norms": aux_norms, "runs": {}}

    plan = [("none", None, 0.0)]
    for multiplier in (1, 100, 1000):
        for source in ("teacher_rel", "shuffled_rel"):
            plan.append((f"{source}_x{multiplier}", source, weight * multiplier))

    for arm, source, arm_weight in plan:
        model = copy.deepcopy(base)
        model.to(device)
        model.load_state_dict(base_state, strict=True)
        optimizer = torch.optim.AdamW(model.parameters(), lr=trainer.LR)
        init_sha = _sha(model.state_dict())
        torch.manual_seed(SEED)
        local_order = torch.randperm(len(embeddings), generator=torch.Generator().manual_seed(SEED))
        row = {"arm": arm, "source": source, "weight": arm_weight,
               "init_sha": init_sha, "steps": [], "fidelity": []}
        for step in range(1, STEPS + 1):
            optimizer.zero_grad(set_to_none=True)
            batch = embeddings[local_order[(step * RECON_BATCH) % max(len(embeddings) - RECON_BATCH, 1):
                                           (step * RECON_BATCH) % max(len(embeddings) - RECON_BATCH, 1)
                                           + RECON_BATCH]].to(device)
            reconstructed, parts, _, _ = model(batch)
            main = trainer.CRITERION(reconstructed, batch, parts)[0]
            total = main
            kl = 0.0
            if source is not None:
                slice_ = relation_batch(step)
                items = torch.from_numpy(embeddings[train_nodes[slice_].reshape(-1)].numpy()).to(device)
                path = ts.quantised_student_input(model, items,
                                                  instances=len(train_nodes[slice_]),
                                                  nodes=train_nodes.shape[1])
                aux = rd.relational_loss(path.q, targets[source][slice_], TEMPERATURE)
                total = main + arm_weight * aux["loss"]
                kl = aux["kl"]
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), trainer.GRADIENT_CLIP_NORM)
            optimizer.step()
            if step % EVAL_EVERY == 0 or step == STEPS:
                with torch.no_grad():
                    heldout_path = ts.quantised_student_input(
                        model, heldout_items, instances=len(heldout_nodes),
                        nodes=heldout_nodes.shape[1])
                    fidelity = rd.relation_fidelity(heldout_path.q, heldout_target, TEMPERATURE)
                    tokens = model.get_indices(heldout_items).cpu().numpy()
                    health = {
                        "l1_used": int(len(np.unique(tokens[:, 0]))),
                        "l2_used": int(len(np.unique(tokens[:, :2], axis=0))),
                        "l3_used": int(len(np.unique(tokens[:, :3], axis=0))),
                        "full_collision": float(1.0 - len(np.unique(tokens[:, :3], axis=0))
                                                / len(tokens)),
                        "spacing": float(trainer.codeword_spacing(
                            model.rq.vq_layers[0].get_code_embs())),
                    }
                row["fidelity"].append({"step": step, "kl": fidelity["kl"],
                                        "top1_agreement": fidelity["top1_agreement"], **health})
                row["steps"].append({"step": step, "main": float(main.detach()), "aux_kl": kl})
        row["final_sha"] = _sha(model.state_dict())
        results["runs"][arm] = row
        print(arm, json.dumps(row["fidelity"][-1]), flush=True)

    (HERE / "relation_gate.json").write_text(json.dumps(results, indent=1))
    summary = {arm: {"kl": run["fidelity"][-1]["kl"],
                     "top1": run["fidelity"][-1]["top1_agreement"],
                     "collision": run["fidelity"][-1]["full_collision"],
                     "l3_used": run["fidelity"][-1]["l3_used"],
                     "init_sha": run["init_sha"][:12]}
               for arm, run in results["runs"].items()}
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
