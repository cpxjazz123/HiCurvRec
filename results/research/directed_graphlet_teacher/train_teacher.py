"""Issue #229: one jointly trained teacher, and whether joint training fixes roles.

Earlier the orbit head was probed on *frozen* node states and failed to beat the
constant profile.  This script trains the encoder with edge + direction + role
losses in one pass, then freezes a single checkpoint with every head.

Held-out design:

  * edge / direction use the #228 splits (masked train pairs, valid, test);
  * the role heads are evaluated on 20% of the covered items that are **excluded
    from the role loss** during training, so the number is about unseen items;
  * the role targets are the observed orbit/subtype distributions, which are
    functions of the observed graph the encoder is allowed to read.

The question is narrow: does joint training make the model recognise roles
better than the constant profile, where the frozen-state probe could not?
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
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "directed_graphlet_role"))

import graph_teacher2 as gt2  # noqa: E402
import graph_teacher3 as gt3  # noqa: E402
import structure_teacher as st  # noqa: E402

STEPS = 1200
ACCUM = 3
MICRO = 4096
HIDDEN = 256
DIM = 128
ROLES = 376


class JointTeacher(nn.Module):
    def __init__(self, subtype_count: int = 47, roles: int = ROLES):
        super().__init__()
        self.encoder = gt3.ResidualGraphEncoder(768, HIDDEN, DIM)
        self.edge = gt3.EdgeHead(DIM, HIDDEN)
        self.direction = gt3.AsymmetricDirectionHead(DIM, HIDDEN)
        self.subtype = nn.Sequential(nn.Linear(DIM, HIDDEN), nn.GELU(),
                                     nn.Linear(HIDDEN, subtype_count))
        self.orbit = nn.Sequential(nn.Linear(DIM, HIDDEN), nn.GELU(),
                                   nn.Linear(HIDDEN, roles))

    def encode(self, features, adjacency):
        return self.encoder(features, adjacency)

    @staticmethod
    def pair(state, left, right):
        a, b = state[left], state[right]
        return torch.cat([a, b, a * b], dim=-1)


def distribution_report(logits, target, mean_profile):
    ce = float(-(target * F.log_softmax(logits, -1)).sum(-1).mean())
    entropy = float(-(target * torch.log(target.clamp_min(1e-12))).sum(-1).mean())
    constant = float(-(target * torch.log(mean_profile.clamp_min(1e-12))).sum(-1).mean())
    probabilities = F.softmax(logits, -1)
    p = probabilities - probabilities.mean(1, keepdims=True)
    t = target - target.mean(1, keepdims=True)
    denominator = torch.sqrt((p * p).sum(1) * (t * t).sum(1)).clamp_min(1e-12)
    prediction = probabilities.argmax(-1).cpu().numpy()
    label = target.argmax(-1).cpu().numpy()
    return {"ce": ce, "constant_ce": constant,
            "kl": ce - entropy, "target_entropy": entropy,
            "relative_improvement": (constant - ce) / max(constant, 1e-9),
            "carried": bool(ce < constant * 0.98),
            "role_rank_correlation": float(((p * t).sum(1) / denominator).mean()),
            "macro_f1": st._macro_f1(prediction, label, target.shape[1]),
            "majority_f1": st._macro_f1(np.full_like(label, np.bincount(label).argmax()),
                                        label, target.shape[1])}


def main() -> None:
    started = time.time()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    data = gt3.prepare_data(device)
    n = data["n_items"]
    orbit, orbit_coverage = gt3.cached_orbit_profile(data)
    subtype, family, subtype_count = st._subtype_family_profiles(data, n)
    orbit_t = torch.from_numpy(orbit.astype(np.float32)).to(device)
    subtype_t = torch.from_numpy(subtype.astype(np.float32)).to(device)

    covered = np.flatnonzero(orbit.sum(1) > 0)
    rng = np.random.default_rng(229)
    order = rng.permutation(len(covered))
    cut = int(0.8 * len(order))
    role_train, role_eval = covered[order[:cut]], covered[order[cut:]]
    role_train_t = torch.from_numpy(role_train).to(device)
    role_eval_t = torch.from_numpy(role_eval).to(device)

    torch.manual_seed(226)
    model = JointTeacher(subtype_count).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    stream = gt3.PairStream(len(data["train_positive"]), 3226, MICRO)
    direction = {key: torch.from_numpy(data["train_direction"][key]).to(device)
                 for key in ("rows", "cols", "labels")}
    direction_stream = gt3.PairStream(len(data["train_direction"]["rows"]), 3227, MICRO)
    base = torch.from_numpy(data["train_positive"]).to(device)
    negative = torch.from_numpy(data["train_negative"]).to(device)
    trace = []
    for step in range(1, STEPS + 1):
        optimizer.zero_grad(set_to_none=True)
        state = model.encode(data["features"], data["adjacency"])
        orbit_loss = -(orbit_t[role_train_t]
                       * F.log_softmax(model.orbit(state[role_train_t]), -1)
                       ).sum(-1).mean()
        subtype_loss = -(subtype_t[role_train_t]
                         * F.log_softmax(model.subtype(state[role_train_t]), -1)
                         ).sum(-1).mean()
        total = None
        for _ in range(ACCUM):
            pick = torch.from_numpy(stream.next()).to(device)
            positive = base[pick]
            left = torch.cat([positive[:, 0], positive[:, 0]])
            right = torch.cat([positive[:, 1], negative[pick]])
            labels = torch.cat([torch.ones(len(pick)), torch.zeros(len(pick))]).to(device)
            edge_loss = F.binary_cross_entropy_with_logits(
                model.edge(model.pair(state, left, right)), labels)
            d_pick = torch.from_numpy(direction_stream.next()).to(device)
            d_loss = F.cross_entropy(model.direction(state, direction["rows"][d_pick],
                                                     direction["cols"][d_pick]),
                                     direction["labels"][d_pick])
            loss = edge_loss + 0.5 * d_loss
            total = loss if total is None else total + loss
        (total / ACCUM + 0.2 * orbit_loss + 0.1 * subtype_loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        trace.append(float((total / ACCUM + 0.2 * orbit_loss
                            + 0.1 * subtype_loss).detach()))
        if step % 200 == 0:
            print(f"step {step} loss {trace[-1]:.4f}", flush=True)
    model.eval()

    with torch.no_grad():
        state = model.encode(data["features"], data["adjacency"])
        # edge and direction on the held-out splits
        def pair_scores(positive, negatives):
            left = torch.from_numpy(positive[:, 0]).to(device)
            right = torch.from_numpy(positive[:, 1]).to(device)
            target = torch.from_numpy(negatives).to(device)
            positive_score = model.edge(model.pair(state, left, right)).cpu().numpy()
            negative_score = model.edge(model.pair(state, left, target)).cpu().numpy()
            return np.stack([positive_score, negative_score], 1)

        scores = pair_scores(data["test_positive"], data["test_negative"])
        edge_auc = gt3.auc_of(scores)
        direction_logits = model.direction(state,
                                           torch.from_numpy(data["test_direction"]["rows"]).to(device),
                                           torch.from_numpy(data["test_direction"]["cols"]).to(device))
        prediction = direction_logits.argmax(-1).cpu().numpy()
        target = data["test_direction"]["labels"]
        confusion = np.zeros((2, 2), dtype=np.int64)
        for truth, predicted in zip(target, prediction):
            confusion[truth, predicted] += 1
        direction_metrics = gt2.direction_metrics(confusion)
        # roles on the held-out items
        orbit_eval = distribution_report(model.orbit(state[role_eval_t]),
                                         orbit_t[role_eval_t],
                                         orbit_t[role_train_t].mean(0))
        subtype_eval = distribution_report(model.subtype(state[role_eval_t]),
                                           subtype_t[role_eval_t],
                                           subtype_t[role_train_t].mean(0))

    checkpoint = HERE / "teacher" / "structure" / "joint" / "teacher_joint_v1.pth"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    payload = {"state": model.state_dict(), "subtype_count": subtype_count,
               "hidden": HIDDEN, "dim": DIM, "roles": ROLES, "steps": STEPS,
               "graph_sha": data["meta"]["graph_sha"], "split_sha": data["split_sha"],
               "meta": {"role_train": int(len(role_train)),
                        "role_eval": int(len(role_eval))}}
    torch.save(payload, checkpoint)
    result = {"steps": STEPS, "accum": ACCUM, "micro": MICRO, "lr": 3e-4,
              "final_loss": trace[-1], "first_loss": trace[0],
              "edge": {"test_auc": edge_auc},
              "direction": {"test_balanced_accuracy":
                                direction_metrics["balanced_accuracy"],
                            "majority": direction_metrics["majority_baseline"],
                            "prediction_frequency_max":
                                direction_metrics["prediction_frequency_max"]},
              "orbit": orbit_eval, "subtype": subtype_eval,
              "frozen_probe_reference": {"orbit_ce": 3.5488, "orbit_constant_ce": 3.5163,
                                         "source": "teacher/opt graph_teacher3 orbit probe"},
              "checkpoint": checkpoint.name,
              "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
              "seconds": time.time() - started}
    out = HERE / "teacher" / "opt" / "joint_teacher_report.json"
    out.write_text(json.dumps(result, indent=1))
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
