"""Issue #225: safe structure supervision -- encoder-only vs controlled codebook.

Five arms, one k-means initialisation, one batch order, one step budget:

    A none                     no structure loss
    B role_teacher_enc         role target, gradient reaches the encoder only
    C role_teacher_enc_cb      role target, encoder + codebooks with a per-group cap
    D role_teacher_shuffled    B's target, permuted across items (regulariser control)
    E role_labels_enc          B's wiring, target from the real orbit labels instead of
                               the Teacher's predicted distribution

Three things the previous round lacked, all enforced here:

  * SID is read the way the export reads it: `model.eval()`, `get_indices`, and a
    re-encode of the same checkpoint must reproduce the same tokens item by item;
  * every auxiliary gradient is reported per group (encoder, L1, L2, L3) as a share
    of the main-loss gradient, and arm C's weight is the minimum over all four;
  * an arm stops as soon as its exported collision exceeds the control's at the
    same step by `COLLAPSE_MARGIN`, writing `collapse_stop.json`.
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
from relation_audit import auc  # noqa: E402
from teacher_model import load_frozen_teacher  # noqa: E402

CAP = 8000
CHUNK = 2048
STEPS = 300
RECON_BATCH = 512
ITEMS_PER_STEP = 128
SEED = 227
AUX_SHARE = 0.10
EVAL_EVERY = 50
COLLAPSE_MARGIN = 0.10
ROLES = 376
FAMILY = "K1_3"


def _sha(state_dict):
    import hashlib
    digest = hashlib.sha256()
    for key in sorted(state_dict):
        digest.update(key.encode())
        digest.update(state_dict[key].detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def collect(split, vocab, teacher, semantic, directed, n_items, device):
    """Per-item Teacher role distribution and real-orbit histogram."""
    probs = defaultdict(lambda: np.zeros(ROLES, dtype=np.float64))
    labels = defaultdict(lambda: np.zeros(ROLES, dtype=np.float64))
    occurrences = defaultdict(int)
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
                _, orbit_logits, _ = teacher(items.view(len(block), width, -1),
                                             torch.from_numpy(adjacency[start:start + CHUNK]).to(device))
                posterior = F.softmax(orbit_logits, dim=-1).cpu().numpy()
            orbit_block = targets["orbit"][start:start + CHUNK]
            valid_block = valid[start:start + CHUNK]
            for row in range(len(block)):
                if not valid_block[row]:
                    continue
                for column in range(width):
                    item = int(block[row, column])
                    probs[item] += posterior[row, column]
                    role = int(orbit_block[row, column])
                    if 0 <= role < ROLES:
                        labels[item][role] += 1.0
                    occurrences[item] += 1
        print(split, family, "done", flush=True)
    items = sorted(probs)
    prob_matrix = np.stack([probs[item] / max(occurrences[item], 1) for item in items])
    label_matrix = np.stack([labels[item] / max(labels[item].sum(), 1.0) for item in items])
    return np.array(items, dtype=np.int64), prob_matrix, label_matrix


def aux_quantised(model, items, *, encoder_only: bool):
    """q from the real quantiser; with `encoder_only` the tables get no gradient."""
    tables = [layer.get_code_embs() for layer in model.rq.vq_layers]
    if encoder_only:
        for table in tables:
            table.requires_grad_(False)
    try:
        return ts.quantised_student_input(model, items, instances=len(items), nodes=1).q[:, 0]
    finally:
        if encoder_only:
            for table in tables:
                table.requires_grad_(True)


def group_gradients(loss, model):
    encoder = [p for p in model.encoder.parameters() if p.requires_grad]
    encoder_norm = float(torch.sqrt(sum((g.detach() ** 2).sum() for g in
                                        torch.autograd.grad(loss, encoder, retain_graph=True,
                                                            allow_unused=True) if g is not None)))
    books = []
    for parameter in ts.codebook_parameters(model).values():
        grad = torch.autograd.grad(loss, [parameter], retain_graph=True, allow_unused=True)[0]
        books.append(0.0 if grad is None else float(grad.detach().norm()))
    return encoder_norm, books


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
    skeleton = {(min(int(a), int(b)), max(int(a), int(b))) for a, b in zip(rows.tolist(), cols.tolist())}
    vocab = load_vocabulary("directed")

    train_items, train_probs, train_labels = collect("train", vocab, teacher, semantic, directed,
                                                     n_items, device)
    held_items, held_probs, held_labels = collect("heldout", vocab, teacher, semantic, directed,
                                                  n_items, device)
    print("train items", len(train_items), "heldout", len(held_items), flush=True)

    # behaviour pairs from K1_3 held-out instances, for the fidelity metric
    held_instances = load_pool("heldout", vocab, max_instances=CAP)[FAMILY][0]
    width = held_instances.shape[1]
    pair_index = torch.tensor([[i, j] for i in range(width) for j in range(i + 1, width)])
    pair_labels = np.array([
        1.0 if (min(int(row[i]), int(row[j])), max(int(row[i]), int(row[j]))) in skeleton else 0.0
        for row in held_instances for i, j in pair_index.tolist()])

    # map raw item ids to positions in the held-out item list, so the export tokens
    # and the instance pairs address the same rows
    held_position = {int(item): index for index, item in enumerate(held_items.tolist())}
    held_mapped = np.array([[held_position.get(int(value), -1) for value in row]
                            for row in held_instances])
    kept_instances = (held_mapped >= 0).all(axis=1)
    held_mapped = held_mapped[kept_instances]
    pair_matrix = pair_labels.reshape(len(held_instances), -1)[kept_instances]
    pair_labels = pair_matrix.reshape(-1)
    held_instances_kept = held_instances[kept_instances]
    print("instances kept for fidelity", int(kept_instances.sum()), "of", len(held_instances), flush=True)

    train_probs_t = torch.from_numpy(train_probs).float().to(device)
    train_labels_t = torch.from_numpy(train_labels).float().to(device)
    held_probs_t = torch.from_numpy(held_probs).float().to(device)
    held_labels_t = torch.from_numpy(held_labels).float().to(device)
    train_ids = torch.from_numpy(train_items)
    held_ids = torch.from_numpy(held_items)
    held_instance_items = torch.from_numpy(held_instances.reshape(-1))

    config = trainer._tokenizer_config()
    torch.manual_seed(SEED)
    base = trainer.RQVAE(config, in_dim=embeddings.shape[1], distance=trainer.DISTANCE).to(device)
    trainer.initialize_tiger_weights(base)
    base.init_codebook(embeddings.to(device))
    base_state = {key: value.detach().clone() for key, value in base.state_dict().items()}

    def role_loss(model, head, ids, targets):
        q = aux_quantised(model, embeddings[ids].to(device), encoder_only=False)
        return -(targets * F.log_softmax(head(q), dim=-1)).sum(-1).mean()

    # weight calibration: one randomly initialised head, per-group norms
    zero = torch.arange(ITEMS_PER_STEP)
    probe = copy.deepcopy(base).to(device)
    torch.manual_seed(SEED + 1)
    probe_head = torch.nn.Linear(config.codebook_dim, ROLES).to(device)
    aux = role_loss(probe, probe_head, train_ids[zero], train_probs_t[zero])
    aux_encoder, aux_books = group_gradients(aux, probe)
    reconstructed, parts, _, _ = probe(embeddings[:RECON_BATCH].to(device))
    main = trainer.CRITERION(reconstructed, embeddings[:RECON_BATCH].to(device), parts)[0]
    main_encoder, main_books = group_gradients(main, probe)
    capped = [AUX_SHARE * reference / max(value, 1e-12)
              for reference, value in zip(main_books, aux_books)]
    weight_encoder_only = float(AUX_SHARE * main_encoder / max(aux_encoder, 1e-12))
    weight_capped = float(min([weight_encoder_only] + capped))
    print("weights", {"encoder_only": weight_encoder_only, "capped": weight_capped},
          {"main": [main_encoder] + main_books, "aux": [aux_encoder] + aux_books}, flush=True)

    results = {"teacher": teacher_record.get("extra", {}),
               "config": {"steps": STEPS, "items_per_step": ITEMS_PER_STEP,
                          "train_items": int(len(train_items)), "heldout_items": int(len(held_ids)),
                          "weights": {"encoder_only": weight_encoder_only, "capped": weight_capped},
                          "aux_share": AUX_SHARE, "collapse_margin": COLLAPSE_MARGIN,
                          "seed": SEED, "eval_every": EVAL_EVERY},
               "gradient_calibration": {"main": [main_encoder] + main_books,
                                        "aux": [aux_encoder] + aux_books},
               "runs": {}}

    def exported(model):
        model.eval()
        with torch.no_grad():
            tokens = model.get_indices(embeddings[held_ids].to(device)).cpu().numpy()
            tokens_again = model.get_indices(embeddings[held_ids].to(device)).cpu().numpy()
            q = ts.quantised_student_input(model, embeddings[held_instances_kept.reshape(-1)].to(device),
                                           instances=len(held_instances_kept), nodes=width).q
        model.train()
        consistent = bool(np.array_equal(tokens, tokens_again))
        collision = float(1.0 - len(np.unique(tokens[:, :3], axis=0)) / len(tokens))
        normalised = F.normalize(q, dim=-1)
        scores = (normalised[:, pair_index[:, 0]] * normalised[:, pair_index[:, 1]]).sum(-1)
        edge_auc = float(auc(scores.reshape(-1).cpu().numpy(), pair_labels))
        l1 = tokens[:, 0]
        instance_l1 = l1[held_mapped]
        same = (instance_l1[:, pair_index[:, 0]] == instance_l1[:, pair_index[:, 1]]).reshape(-1)
        edge = pair_labels > 0.5
        lift = float(same[edge].mean() - same[~edge].mean()) if (~edge).any() else float("nan")
        return {"consistent": consistent, "collision": collision,
                "l1_used": int(len(np.unique(tokens[:, 0]))),
                "l2_used": int(len(np.unique(tokens[:, :2], axis=0))),
                "l3_used": int(len(np.unique(tokens[:, :3], axis=0))),
                "edge_auc": edge_auc, "l1_lift": lift,
                "spacing": float(trainer.codeword_spacing(model.rq.vq_layers[0].get_code_embs()))}

    plan = [("A_none", None, 0.0, "teacher"),
            ("B_role_teacher_enc", "teacher", weight_encoder_only, "encoder"),
            ("C_role_teacher_enc_cb", "teacher", weight_capped, "both"),
            ("D_role_teacher_shuffled", "shuffled", weight_encoder_only, "encoder"),
            ("E_role_labels_enc", "labels", weight_encoder_only, "encoder")]

    control_collision = {}
    stops = {}
    for name, source, weight, pathway in plan:
        model = copy.deepcopy(base).to(device)
        model.load_state_dict(base_state, strict=True)
        torch.manual_seed(SEED + 1)
        head = torch.nn.Linear(config.codebook_dim, ROLES).to(device)
        optimizer = torch.optim.AdamW(list(model.parameters()) + list(head.parameters()), lr=trainer.LR)
        init_sha = _sha(model.state_dict())
        generator = torch.Generator().manual_seed(SEED)
        row = {"arm": name, "source": source, "pathway": pathway, "weight": weight,
               "init_sha": init_sha, "trace": [], "gradients": []}
        if source == "shuffled":
            permutation = torch.randperm(train_probs.shape[0], generator=torch.Generator().manual_seed(SEED))
        for step in range(1, STEPS + 1):
            index = torch.randperm(len(train_items), generator=generator)[:ITEMS_PER_STEP]
            batch = embeddings[torch.randint(0, len(embeddings), (RECON_BATCH,), generator=generator)].to(device)
            optimizer.zero_grad(set_to_none=True)
            reconstructed, parts, _, _ = model(batch)
            main = trainer.CRITERION(reconstructed, batch, parts)[0]
            total = main
            aux_value = 0.0
            if source is not None:
                if source == "shuffled":
                    targets = train_probs_t[permutation][index]
                elif source == "teacher":
                    targets = train_probs_t[index]
                else:
                    targets = train_labels_t[index]
                encoder_only = pathway == "encoder"
                q = aux_quantised(model, embeddings[train_ids[index]].to(device), encoder_only=encoder_only)
                aux = -(targets * F.log_softmax(head(q), dim=-1)).sum(-1).mean()
                total = main + weight * aux
                aux_value = float(aux.detach())
                if step <= 3 or step % 50 == 0:
                    encoder_norm, book_norms = group_gradients(aux, model)
                    row["gradients"].append({"step": step, "encoder": encoder_norm,
                                             "codebooks": book_norms})
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), trainer.GRADIENT_CLIP_NORM)
            optimizer.step()
            if step % EVAL_EVERY == 0 or step == STEPS:
                metrics = exported(model)
                metrics["step"] = step
                metrics["aux"] = aux_value
                metrics["main"] = float(main.detach())
                with torch.no_grad():
                    q = ts.quantised_student_input(model, embeddings[held_ids].to(device),
                                                   instances=len(held_ids), nodes=1).q[:, 0]
                    metrics["role_kl_teacher"] = float(
                        -(held_probs_t * F.log_softmax(head(q), dim=-1)).sum(-1).mean())
                    metrics["role_kl_labels"] = float(
                        -(held_labels_t * F.log_softmax(head(q), dim=-1)).sum(-1).mean())
                row["trace"].append(metrics)
                if name.startswith("A_"):
                    control_collision[step] = metrics["collision"]
                elif not metrics["consistent"]:
                    stops[name] = {"step": step, "reason": "export re-encode mismatch"}
                    break
                elif step in control_collision and \
                        metrics["collision"] > control_collision[step] + COLLAPSE_MARGIN:
                    stops[name] = {"step": step, "reason": "collision above control margin",
                                   "collision": metrics["collision"],
                                   "control": control_collision[step]}
                    break
        row["final_sha"] = _sha(model.state_dict())
        results["runs"][name] = row
        print(name, json.dumps({key: row["trace"][-1][key] for key in
                                ("step", "collision", "l1_used", "l3_used", "edge_auc",
                                 "l1_lift", "role_kl_teacher", "consistent")}), flush=True)

    (HERE / "collapse_gate.json").write_text(json.dumps(results, indent=1))
    if stops:
        (HERE / "collapse_stop.json").write_text(json.dumps(stops, indent=1))
    summary = {name: {"steps": run["trace"][-1]["step"],
                      "collision": round(run["trace"][-1]["collision"], 4),
                      "l1_used": run["trace"][-1]["l1_used"],
                      "l3_used": run["trace"][-1]["l3_used"],
                      "edge_auc": round(run["trace"][-1]["edge_auc"], 4),
                      "l1_lift": round(run["trace"][-1]["l1_lift"], 4),
                      "role_kl": round(run["trace"][-1]["role_kl_teacher"], 4),
                      "consistent": run["trace"][-1]["consistent"]}
               for name, run in results["runs"].items()}
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
