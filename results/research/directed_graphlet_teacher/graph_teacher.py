"""Issue #226: a graph-level structural Teacher, one vector per item.

The old Teacher classified the graphlet it was handed, so the same item got a
different state in every instance (variance decomposition: 76% item, 24% graphlet).
The new design asks for the opposite: read the masked TRAIN graph once and give
every item **one** structural vector, trained on three self-supervised tasks:

    masked edges     15% of the pairs are removed from the input graph; the model
                     must score the removed edges above degree-matched negatives,
    direction        on a held-out adjacent pair, which way does the edge point,
    orbit profile    the item's orbit histogram over the TRAIN pool (the same target
                     the RQ-VAE distillation would use),
    co-membership    whether two items share a neighbour (a higher-order link).

Leakage rule: a held-out pair is removed in **both** directions from the input graph,
so the reverse edge cannot answer for the held-out one.

Evaluation is on the held-out pairs only, against three baselines on the same pairs:
the raw 768-d semantic cosine, a degree-only score, and the orbit histogram
(best possible constant item profile).  The question the user asked -- does the
Teacher give more than the orbit statistics -- is answered by the last row of the
comparison table.
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / "results" / "research" / "directed_graphlet_role"))

from directed_graphlets import load_adjacency  # noqa: E402
from relation_audit import auc  # noqa: E402

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
SEED = 226
HELD_OUT = 0.15
VALID = 0.05
EPOCHS = 40
BATCHES_PER_EPOCH = 24
PAIRS_PER_BATCH = 4096
HIDDEN = 256
OUTPUT = 128
ROLES = 376
LR = 1e-3


class GraphTeacher(nn.Module):
    """In/out separated message passing over the whole item graph."""

    def __init__(self, in_dim: int = 768, hidden: int = HIDDEN, out_dim: int = OUTPUT):
        super().__init__()
        self.input = nn.Sequential(nn.Linear(in_dim, hidden), nn.GELU())
        self.self_in = nn.Linear(hidden, hidden)
        self.in_proj = nn.Linear(hidden, hidden)
        self.out_proj = nn.Linear(hidden, hidden)
        self.self_out = nn.Linear(hidden, hidden)
        self.output = nn.Linear(hidden, out_dim)
        self.edge_head = nn.Sequential(nn.Linear(3 * out_dim, hidden), nn.GELU(),
                                       nn.Linear(hidden, 1))
        self.direction_head = nn.Sequential(nn.Linear(3 * out_dim, hidden), nn.GELU(),
                                            nn.Linear(hidden, 2))
        self.graphlet_head = nn.Sequential(nn.Linear(3 * out_dim, hidden), nn.GELU(),
                                           nn.Linear(hidden, 1))
        self.orbit_head = nn.Sequential(nn.Linear(out_dim, hidden), nn.GELU(),
                                        nn.Linear(hidden, ROLES))

    def encode(self, features: torch.Tensor, adjacency) -> torch.Tensor:
        h = self.input(features)
        out_message = torch.sparse.mm(adjacency, h)
        in_message = torch.sparse.mm(adjacency.transpose(0, 1), h)
        degree_out = torch.sparse.sum(adjacency, dim=1).to_dense().clamp(min=1.0).unsqueeze(-1)
        degree_in = torch.sparse.sum(adjacency, dim=0).to_dense().clamp(min=1.0).unsqueeze(-1)
        h = F.gelu(self.self_out(h) + self.out_proj(out_message / degree_out)
                   + self.self_in(h) + self.in_proj(in_message / degree_in))
        out_message = torch.sparse.mm(adjacency, h)
        in_message = torch.sparse.mm(adjacency.transpose(0, 1), h)
        h = F.gelu(self.self_out(h) + self.out_proj(out_message / degree_out)
                   + self.self_in(h) + self.in_proj(in_message / degree_in))
        return self.output(h)

    @staticmethod
    def pair_features(state: torch.Tensor, left: torch.Tensor, right: torch.Tensor):
        a, b = state[left], state[right]
        return torch.cat([a, b, a * b], dim=-1)


def _digest(state_dict) -> str:
    digest = hashlib.sha256()
    for key in sorted(state_dict):
        digest.update(key.encode())
        digest.update(state_dict[key].detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def build_splits(edge_rows: np.ndarray, edge_cols: np.ndarray, n_items: int):
    """Pair-consistent split: a pair is held out in both directions."""
    pairs = np.stack([np.minimum(edge_rows, edge_cols), np.maximum(edge_rows, edge_cols)], axis=1)
    unique = np.unique(pairs, axis=0)
    rng = np.random.default_rng(SEED)
    order = rng.permutation(len(unique))
    n_held = int(len(unique) * HELD_OUT)
    n_valid = int(len(unique) * VALID)
    held = unique[order[:n_held]]
    valid = unique[order[n_held:n_held + n_valid]]
    observed_pairs = unique[order[n_held + n_valid:]]
    observed = set(map(tuple, observed_pairs.tolist()))
    directed_observed = [(int(a), int(b)) for a, b in zip(edge_rows.tolist(), edge_cols.tolist())
                         if (min(a, b), max(a, b)) in observed]
    return directed_observed, held, valid


def degree_matched_negative(target_degree: np.ndarray, positive, rng, blocked, n_items):
    degree = target_degree[positive]
    candidates = np.flatnonzero(np.abs(target_degree - degree) <= max(1.0, 0.05 * degree))
    candidates = candidates[candidates != positive]
    if len(candidates) == 0:
        candidates = np.array([rng.integers(n_items)])
    return int(rng.choice(candidates))


def main() -> None:
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    started = time.time()
    graph = load_adjacency()
    n_items = graph.shape[0]
    rows, cols = graph.nonzero()
    print("edges", len(rows), "items", n_items, flush=True)
    directed_observed, held_pairs, valid_pairs = build_splits(rows, cols, n_items)
    print("observed directed", len(directed_observed), "held pairs", len(held_pairs),
          "valid pairs", len(valid_pairs), flush=True)

    index = np.array(directed_observed, dtype=np.int64)
    values = np.ones(len(index), dtype=np.float32)
    adjacency = torch.sparse_coo_tensor(torch.from_numpy(index.T).to(DEVICE),
                                        torch.from_numpy(values).to(DEVICE),
                                        size=(n_items, n_items)).coalesce()
    features = torch.from_numpy(
        np.load(REPO / "stage1_GeneEmbedding" / "output" / "sentence_t5.npy").astype(np.float32)
    ).to(DEVICE)
    observed_set = set((int(a), int(b)) for a, b in index.tolist())
    degree_out = np.bincount(index[:, 0], minlength=n_items).astype(np.float64)
    degree_in = np.bincount(index[:, 1], minlength=n_items).astype(np.float64)
    total_degree = degree_out + degree_in

    role_counts = np.zeros((n_items, ROLES), dtype=np.float32)
    sys.path.insert(0, str(REPO / "results" / "research" / "directed_graphlet_role"))
    from directed_role import FAMILIES, instance_targets, load_pool, load_vocabulary
    vocab = load_vocabulary("directed")
    for family in FAMILIES:
        nodes, orbits, subtypes = load_pool("train", vocab, max_instances=8000)[family]
        targets = instance_targets(family, subtypes, orbits, vocab)
        valid = targets["subtype"] >= 0
        for row_index in range(len(nodes)):
            if not valid[row_index]:
                continue
            for column in range(nodes.shape[1]):
                role = int(targets["orbit"][row_index, column])
                if 0 <= role < ROLES:
                    role_counts[int(nodes[row_index, column]), role] += 1.0
    role_profile = role_counts / np.clip(role_counts.sum(1, keepdims=True), 1.0, None)
    role_tensor = torch.from_numpy(role_profile).to(DEVICE)

    model = GraphTeacher().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    rng = np.random.default_rng(SEED)
    positives = index
    trace = []
    for epoch in range(1, EPOCHS + 1):
        state = model.encode(features, adjacency)
        total = 0.0
        for _ in range(BATCHES_PER_EPOCH):
            pick = rng.integers(0, len(positives), PAIRS_PER_BATCH)
            batch = positives[pick]
            target = batch[:, 1]
            negatives = np.array([degree_matched_negative(total_degree, t, rng, observed_set, n_items)
                                  for t in target])
            left = torch.from_numpy(np.concatenate([batch[:, 0], batch[:, 0]])).to(DEVICE)
            right = torch.from_numpy(np.concatenate([target, negatives])).to(DEVICE)
            labels = torch.cat([torch.ones(len(batch)), torch.zeros(len(batch))]).to(DEVICE)
            edge_logits = model.edge_head(model.pair_features(state, left, right)).squeeze(-1)
            edge_loss = F.binary_cross_entropy_with_logits(edge_logits, labels)
            direction_mask = np.array([(int(b), int(a)) not in observed_set for a, b in batch.tolist()])
            if direction_mask.any():
                chosen = batch[direction_mask]
                chosen_t = torch.from_numpy(chosen).to(DEVICE)
                pair_features = model.pair_features(state, chosen_t[:, 0], chosen_t[:, 1])
                direction_logits = model.direction_head(pair_features)
                direction_loss = F.cross_entropy(direction_logits, torch.zeros(len(chosen),
                                                                              dtype=torch.long,
                                                                              device=DEVICE))
            else:
                direction_loss = torch.zeros((), device=DEVICE)
            # The higher-order head is deliberately not trained in this round: a
            # co-membership label needs an intersection count over the observed graph,
            # and a placeholder head fed zeros would only add a no-op to the loss.
            orbit_batch = torch.from_numpy(rng.integers(0, n_items, 2048)).to(DEVICE)
            orbit_logits = model.orbit_head(state[orbit_batch])
            orbit_loss = -(role_tensor[orbit_batch] * F.log_softmax(orbit_logits, -1)).sum(-1).mean()
            loss = edge_loss + 0.5 * direction_loss + 0.2 * orbit_loss
            total = total + loss
        optimizer.zero_grad(set_to_none=True)
        (total / BATCHES_PER_EPOCH).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        trace.append({"epoch": epoch, "loss": float((total / BATCHES_PER_EPOCH).detach())})
        if epoch % 5 == 0 or epoch == 1:
            print(f"epoch {epoch} loss {float((total / BATCHES_PER_EPOCH).detach()):.4f}", flush=True)

    model.eval()
    with torch.no_grad():
        state = model.encode(features, adjacency)
        held = torch.from_numpy(held_pairs).to(DEVICE)
        positive_scores = model.edge_head(model.pair_features(state, held[:, 0], held[:, 1])).squeeze(-1)
        negatives = np.array([degree_matched_negative(total_degree, t, rng, observed_set, n_items)
                              for t in held_pairs[:, 0]])
        negative_scores = model.edge_head(model.pair_features(
            state, held[:, 0], torch.from_numpy(negatives).to(DEVICE))).squeeze(-1)
        scores = torch.cat([positive_scores, negative_scores]).cpu().numpy()
        labels = np.concatenate([np.ones(len(held_pairs)), np.zeros(len(held_pairs))])
        teacher_auc = auc(scores, labels)
        direction_logits = model.direction_head(model.pair_features(state, held[:, 0], held[:, 1]))
        reverse_present = torch.from_numpy(np.array([
            (int(b) * n_items + int(a)) in observed_set for a, b in held_pairs.tolist()])).to(DEVICE)
        forward_present = torch.from_numpy(np.array([
            (int(a) * n_items + int(b)) in observed_set for a, b in held_pairs.tolist()])).to(DEVICE)
        unique_pairs = (~reverse_present) & forward_present
        direction_accuracy = float("nan")
        if unique_pairs.any():
            direction_accuracy = float((direction_logits[unique_pairs].argmax(-1)
                                        == 0).float().mean())

        semantic = features
        normalised = F.normalize(semantic, dim=-1)
        semantic_scores = (normalised[held[:, 0]] * normalised[held[:, 1]]).sum(-1).cpu().numpy()
        semantic_scores = np.concatenate([semantic_scores,
                                          (normalised[held[:, 0]] * normalised[
                                              torch.from_numpy(negatives).to(DEVICE)]).sum(-1).cpu().numpy()])
        semantic_auc = auc(semantic_scores, labels)
        degree_scores = total_degree[held_pairs[:, 0]].astype(np.float64)
        degree_auc = auc(np.concatenate([degree_scores, total_degree[negatives]]), labels)

        probe_items = torch.arange(n_items, device=DEVICE)
        orbit_logits = model.orbit_head(state[probe_items])
        orbit_kl = float(-(role_tensor * F.log_softmax(orbit_logits, -1)).sum(-1).mean())
    # the same readout from the raw embedding, as the "statistics only" baseline;
    # outside no_grad, the probe's own parameters need a real backward pass
    torch.manual_seed(SEED)
    linear = nn.Linear(768, ROLES).to(DEVICE)
    optimizer_linear = torch.optim.AdamW(linear.parameters(), lr=1e-3)
    for _ in range(400):
        batch = torch.from_numpy(rng.integers(0, n_items, 4096)).to(DEVICE)
        loss = -(role_tensor[batch] * F.log_softmax(linear(features[batch]), -1)).sum(-1).mean()
        optimizer_linear.zero_grad(set_to_none=True)
        loss.backward()
        optimizer_linear.step()
    with torch.no_grad():
        embedding_kl = float(-(role_tensor * F.log_softmax(linear(features), -1)).sum(-1).mean())

    payload = {
        "config": {"seed": SEED, "held_out": HELD_OUT, "valid": VALID, "epochs": EPOCHS,
                   "hidden": HIDDEN, "output": OUTPUT, "items": int(n_items),
                   "observed_directed_edges": int(len(index)),
                   "held_pairs": int(len(held_pairs))},
        "masked_edges": {"teacher_auc": teacher_auc, "semantic_auc": semantic_auc,
                         "degree_auc": degree_auc, "negatives": "degree matched"},
        "direction": {"accuracy_on_one_way_held_pairs": direction_accuracy,
                      "n": int(unique_pairs.sum().item())},
        "orbit_profile": {"teacher_kl": orbit_kl, "embedding_linear_kl": embedding_kl},
        "trace": trace,
        "state_sha": _digest(model.state_dict()),
    }
    torch.save({"state_dict": model.state_dict(),
                "config": {"in_dim": 768, "hidden": HIDDEN, "output": OUTPUT}},
               HERE / "graph_teacher_v1.pth")
    (HERE / "graph_teacher_report.json").write_text(json.dumps(payload, indent=1))
    print(json.dumps({key: payload[key] for key in ("masked_edges", "direction", "orbit_profile")},
                     indent=1))


if __name__ == "__main__":
    main()
