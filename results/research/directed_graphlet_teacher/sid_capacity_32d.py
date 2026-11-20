"""Can a 32-d student (the SID-sized representation) carry the structure target?

#229 asks whether the structure target can in principle be carried by the fixed
item representation.  The bundle's carrier is a 1024-wide probe; the RQ-VAE's
encoder emits 32-d vectors, so this script trains exactly that capacity:

  student:  768 -> 256 -> 32 (GELU)        the SID-side representation
  aux head: 32 -> 376                       only used while training the student
  losses:   1.0 ranking + 0.1 distill (relation) + 0.5 orbit (neighbourhood)

Controls: the same architecture and losses on item-shuffled targets, which keeps
the marginals but destroys the item alignment.  Measured on held-out items/pairs:

  relation_cosine   student 32-d vs the teacher relation on held-out items
  link_auc          student cosine on held-out test pairs vs degree-matched negatives
  orbit_rank        per-item correlation between the student's role head and the
                    held-out neighbour-orbit target
"""
from __future__ import annotations

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

import graph_teacher3 as gt3  # noqa: E402
import structure_teacher as st  # noqa: E402
import structure_teacher_bundle as stb  # noqa: E402


def auc_of(scores_pos, scores_neg):
    labels = np.concatenate([np.ones(len(scores_pos)), np.zeros(len(scores_neg))])
    return float(gt3.auc(np.concatenate([scores_pos, scores_neg]), labels))


def train_student(features, targets, coverage, train_items, device, *, shuffle_seed=None,
                  steps=1500, dim=32):
    relation = torch.from_numpy(targets["relation"]).float().to(device)
    orbit = torch.from_numpy(targets["neighbour_orbit"]).float().to(device)
    pairs = torch.from_numpy(targets["positive_pairs"]).long().to(device)
    negatives = torch.from_numpy(targets["negative_pairs"]).long().to(device)
    if shuffle_seed is not None:
        # item-alignment control: permute the *target rows* over items
        rng = np.random.default_rng(shuffle_seed)
        permutation = torch.from_numpy(rng.permutation(len(relation))).to(device)
        relation = relation[permutation]
        orbit = orbit[permutation]
    encoder = nn.Sequential(nn.Linear(features.shape[1], 256), nn.GELU(),
                            nn.Dropout(0.1), nn.Linear(256, dim)).to(device)
    head = nn.Linear(dim, orbit.shape[1]).to(device)
    optimizer = torch.optim.AdamW(list(encoder.parameters()) + list(head.parameters()),
                                  lr=1e-3, weight_decay=1e-4)
    train = torch.from_numpy(train_items).to(device)
    coverage_t = torch.from_numpy(coverage).float().to(device)
    trace = []
    unit_teacher = F.normalize(relation, dim=1)
    for _ in range(steps):
        optimizer.zero_grad()
        student = encoder(features)
        unit_student = F.normalize(student, dim=1)
        positive = (unit_student[pairs[:, 0]] * unit_student[pairs[:, 1]]).sum(-1)
        negative = (unit_student[pairs[:, 0]] * unit_student[negatives]).sum(-1)
        rank = F.relu(0.1 - positive + negative).mean()
        target_similarity = (unit_teacher[pairs[:, 0]] * unit_teacher[pairs[:, 1]]).sum(-1)
        distill = F.mse_loss(positive, target_similarity)
        mask = coverage_t[train] > 0
        orbit_loss = -(orbit[train][mask]
                       * F.log_softmax(head(student[train][mask]), -1)).sum(-1).mean()
        loss = 1.0 * rank + 0.1 * distill + 0.5 * orbit_loss
        loss.backward()
        optimizer.step()
        trace.append(float(loss.detach()))
    encoder.eval()
    head.eval()
    return encoder, head, trace


def decode_cosine(student_vectors, teacher_relation, train_items, eval_items, device,
                  steps: int = 600) -> float:
    """A decoder trained on held-out-split items tests whether the 32-d code is a
    consistent representation of the 128-d teacher relation (not item memorisation)."""
    teacher = F.normalize(teacher_relation.float().to(device), dim=1)
    decoder = nn.Linear(student_vectors.shape[1], teacher.shape[1]).to(device)
    optimizer = torch.optim.Adam(decoder.parameters(), lr=1e-2, weight_decay=1e-4)
    train = torch.from_numpy(train_items).to(device)
    for _ in range(steps):
        optimizer.zero_grad()
        predicted = F.normalize(decoder(student_vectors[train]), dim=1)
        loss = F.mse_loss(predicted, teacher[train])
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        evaluate = torch.from_numpy(eval_items).to(device)
        predicted = F.normalize(decoder(student_vectors[evaluate]), dim=1)
        return float((predicted * teacher[evaluate]).sum(-1).mean())


def evaluate(encoder, head, features, targets, data, eval_items, train_items, device):
    encoder.eval()
    head.eval()
    with torch.no_grad():
        student = encoder(features)
        teacher = torch.from_numpy(targets["relation"]).float().to(device)
        unit_student = F.normalize(student, dim=1)
        unit_teacher = F.normalize(teacher, dim=1)
        evaluate = torch.from_numpy(eval_items).to(device)
        relation_cosine = float("nan")
        orbit = torch.from_numpy(targets["neighbour_orbit"]).float().to(device)
        logits = head(student[evaluate])
        probabilities = F.softmax(logits, -1)
        truth = orbit[evaluate]
        p = probabilities - probabilities.mean(1, keepdims=True)
        t = truth - truth.mean(1, keepdims=True)
        denominator = torch.sqrt((p * p).sum(1) * (t * t).sum(1)).clamp_min(1e-12)
        orbit_rank = float(((p * t).sum(1) / denominator).mean())
        positive = data["test_positive"]
        negative = data["test_negative"]
        pos = (unit_student[torch.from_numpy(positive[:, 0]).to(device)]
               * unit_student[torch.from_numpy(positive[:, 1]).to(device)]).sum(-1)
        neg = (unit_student[torch.from_numpy(positive[:, 0]).to(device)]
               * unit_student[torch.from_numpy(negative).to(device)]).sum(-1)
        link_auc = auc_of(pos.cpu().numpy(), neg.cpu().numpy())
    return {"relation_decode_cosine": None, "orbit_rank": orbit_rank,
            "link_auc": link_auc}


def main() -> None:
    started = time.time()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    data = gt3.prepare_data(device)
    bundle = stb.load_structure_teacher(stb.BUNDLE_DIR)
    targets = bundle.targets
    coverage = targets["coverage_neighbour"]
    covered = np.flatnonzero(coverage > 0)
    rng = np.random.default_rng(229)
    order = rng.permutation(len(covered))
    cut = int(0.8 * len(order))
    train_items, eval_items = covered[order[:cut]], covered[order[cut:]]
    features = data["features"]

    result = {"train_items": int(len(train_items)), "eval_items": int(len(eval_items)),
              "test_pairs": int(len(data["test_positive"]))}
    encoder, head, trace = train_student(features, targets, coverage, train_items, device)
    with torch.no_grad():
        student_vectors = encoder(features)
    relation_cosine = decode_cosine(student_vectors, torch.from_numpy(targets["relation"]),
                                   train_items, eval_items, device)
    result["bundle_targets"] = {**evaluate(encoder, head, features, targets, data,
                                           eval_items, train_items, device),
                                "relation_decode_cosine": relation_cosine,
                                "final_loss": trace[-1], "first_loss": trace[0]}
    torch.save({"encoder": encoder.state_dict(), "head": head.state_dict(),
                "dim": 32, "targets": "structure_targets_v3.npz"},
               HERE / "teacher" / "opt" / "sid_capacity_student_32d.pth")

    encoder_s, head_s, trace_s = train_student(features, targets, coverage, train_items,
                                               device, shuffle_seed=229)
    with torch.no_grad():
        shuffled_vectors = encoder_s(features)
    shuffled_cosine = decode_cosine(shuffled_vectors, torch.from_numpy(targets["relation"]),
                                    train_items, eval_items, device)
    result["shuffled_targets"] = {**evaluate(encoder_s, head_s, features,
                                             {"relation": targets["relation"],
                                              "neighbour_orbit": targets["neighbour_orbit"]},
                                             data, eval_items, train_items, device),
                                  "relation_decode_cosine": shuffled_cosine,
                                  "final_loss": trace_s[-1]}

    # raw 768-d semantic cosine on the same pairs, for reference
    unit = F.normalize(features, dim=1)
    positive = data["test_positive"]
    negative = data["test_negative"]
    pos = (unit[torch.from_numpy(positive[:, 0]).to(device)]
           * unit[torch.from_numpy(positive[:, 1]).to(device)]).sum(-1)
    neg = (unit[torch.from_numpy(positive[:, 0]).to(device)]
           * unit[torch.from_numpy(negative).to(device)]).sum(-1)
    result["raw_semantic_768"] = {"link_auc": auc_of(pos.cpu().numpy(), neg.cpu().numpy())}
    # bottleneck test: is the carried relation topology-specific at 32-d?
    try:
        shuffled = np.load(gt3.OUT / "shuffled_edges.npz")["index"]
        shuffled_tensor = torch.sparse_coo_tensor(
            torch.from_numpy(shuffled.T.astype(np.int64)),
            torch.ones(shuffled.shape[0], dtype=torch.float32),
            size=(data["n_items"], data["n_items"])).coalesce().to(device)
        other = gt3.build_variant("D_shuffled")
        gt3.load_checkpoint(gt3.OUT / "D_shuffled_edge_main_seed226.pth", other)
        other.to(device).eval()
        with torch.no_grad():
            relation_shuffled = F.normalize(
                other.encode(data["features"], shuffled_tensor).float(), dim=1).cpu()
        bottleneck = {}
        for name, target in (("real_topology", torch.from_numpy(targets["relation"])),
                             ("shuffled_topology", relation_shuffled)):
            torch.manual_seed(0)
            encoder_b = nn.Sequential(nn.Linear(768, 32), nn.GELU(),
                                      nn.Linear(32, 128)).to(device)
            optimizer = torch.optim.AdamW(encoder_b.parameters(), lr=1e-3,
                                          weight_decay=1e-4)
            train_tensor = torch.from_numpy(train_items).to(device)
            truth = F.normalize(target.float().to(device), dim=1)
            for _ in range(1200):
                optimizer.zero_grad()
                predicted = F.normalize(encoder_b(features[train_tensor]), dim=1)
                loss = F.mse_loss(predicted, truth[train_tensor])
                loss.backward()
                optimizer.step()
            with torch.no_grad():
                eval_tensor = torch.from_numpy(eval_items).to(device)
                predicted = F.normalize(encoder_b(features[eval_tensor]), dim=1)
                bottleneck[name] = {
                    "decode_cosine_vs_real": float(
                        (predicted * F.normalize(torch.from_numpy(targets["relation"]).float()
                                                 .to(device)[eval_tensor], dim=1)).sum(-1).mean())}
        result["bottleneck_32d"] = bottleneck
    except FileNotFoundError as error:
        result["bottleneck_32d"] = {"status": "MISSING", "detail": str(error)}

    result["seconds"] = time.time() - started
    out = HERE / "teacher" / "opt" / "sid_capacity_32d.json"
    out.write_text(json.dumps(result, indent=1))
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
