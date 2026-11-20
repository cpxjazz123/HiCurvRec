"""Can a *neighbourhood* role target be carried by the fixed 768-d embedding?

The v2 probe showed the item's own orbit histogram is not predictable from its
embedding.  The structural context (the role mix of its observed neighbours) is
smoother and is the natural multi-role target that still depends on the graph.
This probe measures it before anything is frozen:

  neighbour_orbit  = mean orbit profile of the observed neighbours (376)
  neighbour_family = mean family profile of the observed neighbours (5)
  neighbour_relation = mean relation vector of the observed neighbours (128)

Each is fitted 768 -> target on 80% of the covered items and evaluated on the
held-out 20%, against the constant (train mean) profile.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "directed_graphlet_role"))

import graph_teacher3 as gt3  # noqa: E402
import structure_teacher as st  # noqa: E402


def probe(features, target, coverage, device, steps=1200, hidden=1024):
    covered = np.flatnonzero(coverage > 0)
    rng = np.random.default_rng(229)
    order = rng.permutation(len(covered))
    cut = int(0.8 * len(order))
    train, evaluate = covered[order[:cut]], covered[order[cut:]]
    model = nn.Sequential(nn.Linear(features.shape[1], hidden), nn.GELU(), nn.Dropout(0.1),
                          nn.Linear(hidden, target.shape[1])).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    index = torch.from_numpy(train).to(device)
    for _ in range(steps):
        optimizer.zero_grad()
        logits = model(features[index])
        loss = -(target[index] * F.log_softmax(logits, -1)).sum(-1).mean()
        loss.backward()
        optimizer.step()
    model.eval()
    with torch.no_grad():
        evaluation = torch.from_numpy(evaluate).to(device)
        logits = model(features[evaluation])
        ce = float(-(target[evaluation] * F.log_softmax(logits, -1)).sum(-1).mean())
        entropy = float(-(target[evaluation] * torch.log(target[evaluation].clamp_min(1e-12)))
                        .sum(-1).mean())
        mean_profile = target[index].mean(0)
        constant = float(-(target[evaluation] *
                           torch.log(mean_profile.clamp_min(1e-12))).sum(-1).mean())
        probabilities = F.softmax(logits, -1).cpu().numpy()
    label = target[evaluate].argmax(-1).cpu().numpy()
    majority = np.bincount(label).argmax()
    minority = label != majority
    prediction = probabilities.argmax(-1)
    per_item = probability_target_alignment(probabilities, target[evaluate].cpu().numpy())
    return {"items": int(len(covered)), "train": int(len(train)), "eval": int(len(evaluate)),
            "ce": ce, "kl": ce - entropy, "target_entropy": entropy,
            "constant_ce": constant, "constant_kl": constant - entropy,
            "improvement": (constant - ce) / max(constant, 1e-9),
            "carried": bool(ce < constant * 0.98),
            "macro_f1": st._macro_f1(prediction, label, target.shape[1]),
            "majority_f1": st._macro_f1(np.full_like(label, majority), label,
                                        target.shape[1]),
            "top1_accuracy": float((prediction == label).mean()),
            "minority_items": int(minority.sum()),
            "minority_top1_accuracy": float((prediction[minority] == label[minority]).mean())
            if minority.any() else None,
            "role_rank_correlation": per_item}


def probability_target_alignment(probabilities, target):
    """Per-item Pearson between the predicted distribution and the target roles."""
    p = probabilities - probabilities.mean(1, keepdims=True)
    t = target - target.mean(1, keepdims=True)
    numerator = (p * t).sum(1)
    denominator = np.sqrt((p * p).sum(1) * (t * t).sum(1))
    return float(np.mean(numerator / np.clip(denominator, 1e-12, None)))


def main() -> None:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    data = gt3.prepare_data(device)
    n = data["n_items"]
    orbit, coverage_info = gt3.cached_orbit_profile(data)
    subtype, family, _ = st._subtype_family_profiles(data, n)
    variant = gt3.build_variant("C_residual")
    gt3.load_checkpoint(gt3.OUT / "C_residual_edge_main_seed226.pth", variant)
    variant.to(device).eval()
    with torch.no_grad():
        relation = F.normalize(variant.encode(data["features"], data["adjacency"]),
                               dim=1).cpu().numpy()
    edges = data["message_edges"]
    adjacency = sp.csr_matrix((np.ones(2 * len(edges)),
                               (np.concatenate([edges[:, 0], edges[:, 1]]),
                                np.concatenate([edges[:, 1], edges[:, 0]]))),
                              shape=(n, n))
    adjacency.data[:] = 1.0
    adjacency.setdiag(0)
    adjacency.eliminate_zeros()
    degree = np.asarray(adjacency.sum(1)).ravel()
    positive = degree > 0
    neighbour_orbit = (adjacency @ orbit) / np.clip(degree, 1, None)[:, None]
    neighbour_family = (adjacency @ family) / np.clip(degree, 1, None)[:, None]
    neighbour_relation = (adjacency @ relation) / np.clip(degree, 1, None)[:, None]
    neighbour_relation /= np.clip(np.linalg.norm(neighbour_relation, axis=1, keepdims=True),
                                  1e-9, None)
    coverage_orbit = (orbit.sum(1) > 0) & positive
    coverage_family = (family.sum(1) > 0) & positive

    result = {"coverage": coverage_info,
              "degree": {"zero": int((degree == 0).sum()),
                         "median": float(np.median(degree[positive]))},
              "own_orbit": probe(data["features"], torch.from_numpy(orbit.astype(np.float32)).to(device),
                                 coverage_orbit.astype(np.float32), device),
              "neighbour_orbit": probe(data["features"],
                                       torch.from_numpy(neighbour_orbit.astype(np.float32)).to(device),
                                       coverage_orbit.astype(np.float32), device),
              "neighbour_family": probe(data["features"],
                                        torch.from_numpy(neighbour_family.astype(np.float32)).to(device),
                                        coverage_family.astype(np.float32), device)}
    # relation regression with the same split
    covered = np.flatnonzero((orbit.sum(1) > 0) & positive)
    rng = np.random.default_rng(229)
    order = rng.permutation(len(covered))
    cut = int(0.8 * len(order))
    train, evaluate = covered[order[:cut]], covered[order[cut:]]
    model = nn.Sequential(nn.Linear(768, 1024), nn.GELU(), nn.Dropout(0.1),
                          nn.Linear(1024, 128)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    index = torch.from_numpy(train).to(device)
    target = torch.from_numpy(relation.astype(np.float32)).to(device)
    for _ in range(1200):
        optimizer.zero_grad()
        predicted = F.normalize(model(data["features"][index]), dim=1)
        loss = F.mse_loss(predicted, target[index])
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        evaluation = torch.from_numpy(evaluate).to(device)
        predicted = F.normalize(model(data["features"][evaluation]), dim=1)
        cosine = float((predicted * target[evaluation]).sum(-1).mean())
        constant = float((target[index].mean(0) * target[evaluation]).sum(-1).mean())
    result["neighbour_relation"] = {"count": int(regular := len(covered)),
                                    "cosine": cosine, "constant_cosine": constant,
                                    "carried": bool(cosine > 0.8)}
    out = HERE / "teacher" / "opt" / "neighbour_role_probe.json"
    out.write_text(json.dumps(result, indent=1))
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
