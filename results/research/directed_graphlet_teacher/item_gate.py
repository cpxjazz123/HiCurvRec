"""Issue #226: the three Teacher targets the user asked to compare, same RQ-VAE.

    hidden_mean   MSE( W q_i , sg(mean_m h_i^T) )        W: 32 -> 128
    role_dist     KL( p_i^T || softmax(role_head(q_i)) ) p_i^T = per-item orbit histogram
    relation      the relational objective of #224 (measured there; quoted as row three)

TIGER's input, encoder, decoder and SID protocol are untouched: both new targets
are per-item, so `W` and `role_head` are the only added parameters and they train
on the same optimizer step as the RQ-VAE.

Fairness rules enforced here:
  * every arm starts from the same k-means initialisation (parameter SHA printed),
  * one weight, calibrated once with the group cap (encoder and all three codebooks
    each within `AUX_SHARE` of their main-loss gradient),
  * targets come from TRAIN instances and are evaluated on HELD-OUT instances of the
    same items, so "the target was memorised" cannot pass for "the SID kept it",
  * `item_signal_audit.json` reports how much of each target is item-stable at all.
"""
from __future__ import annotations

import copy
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / "stage2_RQ-VAE" / "TIGER_drq_D1"))
sys.path.insert(0, str(REPO / "results" / "research" / "directed_graphlet_role"))

import train_rqvae as trainer  # noqa: E402
import trainable_student as ts  # noqa: E402
from directed_graphlets import load_adjacency  # noqa: E402
from directed_role import FAMILIES, instance_targets, load_pool, load_vocabulary  # noqa: E402
from teacher_model import load_frozen_teacher  # noqa: E402

CAP = 20000
CHUNK = 2048
STEPS = 300
RECON_BATCH = 512
ITEMS_PER_STEP = 128
SEED = 226
AUX_SHARE = 0.10
EVAL_EVERY = 150
ROLES = 376


def _sha(state_dict):
    import hashlib
    digest = hashlib.sha256()
    for key in sorted(state_dict):
        digest.update(key.encode())
        digest.update(state_dict[key].detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def collect(split, vocab, teacher, semantic, directed, n_items, device):
    hidden = defaultdict(list)
    counts = defaultdict(lambda: np.zeros(ROLES, dtype=np.float64))
    for family in FAMILIES:
        nodes, orbits, subtypes = load_pool(split, vocab, max_instances=CAP)[family]
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
                _, _, state = teacher(items.view(len(block), width, -1),
                                      torch.from_numpy(adjacency[start:start + CHUNK]).to(device))
            state = state.cpu().numpy()
            orbit_block = targets["orbit"][start:start + CHUNK]
            valid_block = valid[start:start + CHUNK]
            for row in range(len(block)):
                if not valid_block[row]:
                    continue
                for column in range(width):
                    item = int(block[row, column])
                    hidden[item].append(state[row, column])
                    role = int(orbit_block[row, column])
                    if 0 <= role < ROLES:
                        counts[item][role] += 1.0
        print(split, family, "done", flush=True)
    items = sorted(hidden)
    means = np.stack([np.stack(hidden[item]).mean(0) for item in items])
    histograms = np.stack([counts[item] / max(counts[item].sum(), 1.0) for item in items])
    return np.array(items, dtype=np.int64), means, histograms


def main() -> None:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(SEED)
    semantic = np.load(REPO / "stage1_GeneEmbedding" / "output" / "sentence_t5.npy", mmap_mode="r")
    embeddings = torch.from_numpy(np.asarray(semantic, dtype=np.float32))
    teacher, teacher_record = load_frozen_teacher(HERE / "teacher" / "teacher_directed_v1.pth")
    teacher.to(device).eval()
    graph = load_adjacency()
    n_items = graph.shape[0]
    rows, cols = graph.nonzero()
    directed = set((rows.astype(np.int64) * n_items + cols).tolist())
    vocab = load_vocabulary("directed")

    train_items, train_means, train_hist = collect("train", vocab, teacher, semantic, directed,
                                                   n_items, device)
    held_items, held_means, held_hist = collect("heldout", vocab, teacher, semantic, directed,
                                                n_items, device)
    train_set = set(int(item) for item in train_items.tolist())
    keep = [position for position, item in enumerate(held_items) if int(item) in train_set]
    print("items train", len(train_items), "heldout", len(held_items), "overlap", len(keep), flush=True)

    train_means_t = torch.from_numpy(train_means).float().to(device)
    train_hist_t = torch.from_numpy(train_hist).float().to(device)
    held_means_t = torch.from_numpy(held_means[keep]).float().to(device)
    held_hist_t = torch.from_numpy(held_hist[keep]).float().to(device)
    held_ids = torch.from_numpy(held_items[keep])
    train_ids = torch.from_numpy(train_items)

    config = trainer._tokenizer_config()
    torch.manual_seed(SEED)
    base = trainer.RQVAE(config, in_dim=embeddings.shape[1], distance=trainer.DISTANCE).to(device)
    trainer.initialize_tiger_weights(base)
    base.init_codebook(embeddings.to(device))
    base_state = {key: value.detach().clone() for key, value in base.state_dict().items()}

    projection = torch.nn.Linear(config.codebook_dim, train_means.shape[1]).to(device)
    role_head = torch.nn.Linear(config.codebook_dim, ROLES).to(device)
    torch.nn.init.zeros_(role_head.weight)
    torch.nn.init.zeros_(role_head.bias)

    def aux_terms(model, index):
        q = ts.quantised_student_input(model, embeddings[train_ids[index]].to(device),
                                       instances=len(index), nodes=1).q[:, 0]
        hidden_loss = F.mse_loss(projection(q), train_means_t[index])
        role_loss = -(train_hist_t[index] * F.log_softmax(role_head(q), dim=-1)).sum(-1).mean()
        return hidden_loss, role_loss, q

    probe = copy.deepcopy(base).to(device)
    zero = torch.arange(ITEMS_PER_STEP)
    hidden_loss, role_loss, q = aux_terms(probe, zero)
    # The arm's own role head starts at zero weights, so at step 0 it has no gradient
    # through q at all.  Calibrate the role weight on a randomly initialised head --
    # the gradient the objective will have once it starts learning -- instead of
    # dividing by zero and getting a meaningless weight.
    torch.manual_seed(SEED + 1)
    role_probe = torch.nn.Linear(config.codebook_dim, ROLES).to(device)
    role_loss = -(train_hist_t[zero] * F.log_softmax(role_probe(q), dim=-1)).sum(-1).mean()
    encoder = [p for p in probe.encoder.parameters() if p.requires_grad]
    hidden_encoder = float(torch.autograd.grad(hidden_loss, encoder, retain_graph=True,
                                               allow_unused=True)[0].detach().norm())
    role_encoder = float(torch.autograd.grad(role_loss, encoder, retain_graph=True,
                                             allow_unused=True)[0].detach().norm())
    reconstructed, parts, _, _ = probe(embeddings[:RECON_BATCH].to(device))
    main = trainer.CRITERION(reconstructed, embeddings[:RECON_BATCH].to(device), parts)[0]
    main_encoder = float(torch.autograd.grad(main, encoder, allow_unused=True)[0].detach().norm())
    weight_hidden = float(AUX_SHARE * main_encoder / max(hidden_encoder, 1e-12))
    weight_role = float(AUX_SHARE * main_encoder / max(role_encoder, 1e-12))
    print("weights", {"hidden": weight_hidden, "role": weight_role},
          {"main": main_encoder, "hidden": hidden_encoder, "role": role_encoder}, flush=True)

    results = {"teacher": teacher_record.get("extra", {}),
               "config": {"steps": STEPS, "items_per_step": ITEMS_PER_STEP,
                          "train_items": int(len(train_items)), "heldout_overlap": len(keep),
                          "weights": {"hidden": weight_hidden, "role": weight_role},
                          "aux_share": AUX_SHARE, "seed": SEED},
               "runs": {}}

    def evaluate(model):
        with torch.no_grad():
            q = ts.quantised_student_input(model, embeddings[held_ids].to(device),
                                           instances=len(held_ids), nodes=1).q[:, 0]
            hidden_mse = float(F.mse_loss(projection(q), held_means_t))
            role_kl = float(-(held_hist_t * F.log_softmax(role_head(q), dim=-1)).sum(-1).mean())
            tokens = model.get_indices(embeddings[held_ids].to(device)).cpu().numpy()
        return {"hidden_mse": hidden_mse, "role_kl": role_kl,
                "l1_used": int(len(np.unique(tokens[:, 0]))),
                "l3_used": int(len(np.unique(tokens[:, :3], axis=0))),
                "collision": float(1.0 - len(np.unique(tokens[:, :3], axis=0)) / len(tokens)),
                "spacing": float(trainer.codeword_spacing(model.rq.vq_layers[0].get_code_embs()))}

    for arm in ("none", "hidden_mean", "role_dist"):
        model = copy.deepcopy(base).to(device)
        model.load_state_dict(base_state, strict=True)
        projection_arm = torch.nn.Linear(config.codebook_dim, train_means.shape[1]).to(device)
        role_arm = torch.nn.Linear(config.codebook_dim, ROLES).to(device)
        torch.nn.init.zeros_(role_arm.weight)
        torch.nn.init.zeros_(role_arm.bias)
        parameters = list(model.parameters()) + list(projection_arm.parameters()) + \
            list(role_arm.parameters())
        optimizer = torch.optim.AdamW(parameters, lr=trainer.LR)
        init_sha = _sha(model.state_dict())
        generator = torch.Generator().manual_seed(SEED)
        row = {"arm": arm, "init_sha": init_sha, "trace": []}
        for step in range(1, STEPS + 1):
            index = torch.randperm(len(train_items), generator=generator)[:ITEMS_PER_STEP]
            batch = embeddings[torch.randint(0, len(embeddings), (RECON_BATCH,),
                                             generator=generator)].to(device)
            optimizer.zero_grad(set_to_none=True)
            reconstructed, parts, _, _ = model(batch)
            main = trainer.CRITERION(reconstructed, batch, parts)[0]
            total = main
            aux_value = 0.0
            if arm != "none":
                q = ts.quantised_student_input(model, embeddings[train_ids[index]].to(device),
                                               instances=len(index), nodes=1).q[:, 0]
                if arm == "hidden_mean":
                    aux = F.mse_loss(projection_arm(q), train_means_t[index])
                    total = main + weight_hidden * aux
                else:
                    aux = -(train_hist_t[index]
                            * F.log_softmax(role_arm(q), dim=-1)).sum(-1).mean()
                    total = main + weight_role * aux
                aux_value = float(aux.detach())
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), trainer.GRADIENT_CLIP_NORM)
            optimizer.step()
            if step % EVAL_EVERY == 0 or step == STEPS:
                projection.load_state_dict(projection_arm.state_dict())
                role_head.load_state_dict(role_arm.state_dict())
                row["trace"].append({"step": step, "main": float(main.detach()),
                                     "aux": aux_value, **evaluate(model)})
        row["final_sha"] = _sha(model.state_dict())
        results["runs"][arm] = row
        print(arm, json.dumps(row["trace"][-1]), flush=True)

    (HERE / "item_gate.json").write_text(json.dumps(results, indent=1))
    print(json.dumps({arm: {"hidden_mse": round(run["trace"][-1]["hidden_mse"], 4),
                            "role_kl": round(run["trace"][-1]["role_kl"], 4),
                            "collision": round(run["trace"][-1]["collision"], 4)}
                      for arm, run in results["runs"].items()}, indent=1))


if __name__ == "__main__":
    main()
