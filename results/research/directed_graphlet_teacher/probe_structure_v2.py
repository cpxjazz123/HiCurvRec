"""Issue #229 follow-up: can the fixed 768-d item embedding carry the structure target,
and can direction be learned beyond the degree baseline?

Three measurements, each on held-out items/pairs:

  1. recognition from the embedding: 768 -> subtype (46) and 768 -> orbit (376)
     classifiers, CE/KL vs the constant profile and macro-F1 on the argmax label;
  2. carrier: 768 -> (relation, subtype, orbit) trained on the covered items, CE/KL
     and relation cosine on held-out items;
  3. direction: an asymmetric head trained alone on one-way pairs, early-stopped on
     valid balanced accuracy, against the in-degree rule and a degree logistic.

Nothing is written to the v1 artifact; the numbers decide whether a v2 target is
worth freezing.
"""
from __future__ import annotations

import json
import sys
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


def subtype_profile(data, n, roles=46):
    from directed_role import FAMILIES, load_pool, load_vocabulary
    vocab = load_vocabulary("directed")
    pools = load_pool("train", vocab, max_instances=8000)
    counts = np.zeros((n, roles), dtype=np.float64)
    for family in FAMILIES:
        nodes, _orbits, subtypes = pools[family]
        for row, subtype in zip(nodes, subtypes):
            counts[row, int(subtype)] += 1.0
    profile = counts / np.clip(counts.sum(1, keepdims=True), 1.0, None)
    return profile


def split_items(coverage, seed):
    covered = np.flatnonzero(coverage > 0)
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(covered))
    cut = int(0.8 * len(order))
    return covered[order[:cut]], covered[order[cut:]]


def ce_kl(logits, target):
    ce = float(-(target * F.log_softmax(logits, -1)).sum(-1).mean())
    entropy = float(-(target * torch.log(target.clamp_min(1e-12))).sum(-1).mean())
    return ce, ce - entropy, entropy


def constant_ce(target, mean_profile):
    ce = float(-(target * torch.log(mean_profile.clamp_min(1e-12))).sum(-1).mean())
    entropy = float(-(target * torch.log(target.clamp_min(1e-12))).sum(-1).mean())
    return ce, ce - entropy


def macro_f1(prediction, label, classes):
    f1s = []
    for cls in range(classes):
        tp = int(((prediction == cls) & (label == cls)).sum())
        fp = int(((prediction == cls) & (label != cls)).sum())
        fn = int(((prediction != cls) & (label == cls)).sum())
        if tp + fp + fn == 0:
            continue
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1s.append(0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall))
    return float(np.mean(f1s)) if f1s else 0.0


def train_classifier(features, target, train_items, eval_items, device, steps=800, lr=1e-3):
    model = nn.Sequential(nn.Linear(features.shape[1], 512), nn.GELU(), nn.Dropout(0.1),
                          nn.Linear(512, target.shape[1])).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    train = torch.from_numpy(train_items).to(device)
    for step in range(steps):
        optimizer.zero_grad()
        logits = model(features[train])
        loss = -(target[train] * F.log_softmax(logits, -1)).sum(-1).mean()
        loss.backward()
        optimizer.step()
    model.eval()
    with torch.no_grad():
        logits = model(features[torch.from_numpy(eval_items).to(device)])
        ce, kl, entropy = ce_kl(logits, target[torch.from_numpy(eval_items).to(device)])
        prediction = logits.argmax(-1).cpu().numpy()
    label = target[torch.from_numpy(eval_items).to(device)].argmax(-1).cpu().numpy()
    return model, {"ce": ce, "kl": kl, "entropy": entropy,
                   "macro_f1": macro_f1(prediction, label, target.shape[1])}, logits


def direction_gate(data, device):
    variant = gt3.build_variant("C_residual")
    path = gt3.OUT / f"C_residual_edge_main_seed{gt3.SEEDS[0]}.pth"
    gt3.load_checkpoint(path, variant)
    variant.to(device).eval()
    for parameter in variant.parameters():
        parameter.requires_grad_(False)
    with torch.no_grad():
        state = variant.encode(data["features"], data["adjacency"]).detach()
    head = gt3.AsymmetricDirectionHead(state.shape[1], 256).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-4)
    rows = torch.from_numpy(data["train_direction"]["rows"]).to(device)
    cols = torch.from_numpy(data["train_direction"]["cols"]).to(device)
    labels = torch.from_numpy(data["train_direction"]["labels"]).to(device)
    valid = {key: torch.from_numpy(data["valid_direction"][key]).to(device)
             for key in ("rows", "cols", "labels")}
    test = {key: torch.from_numpy(data["test_direction"][key]).to(device)
            for key in ("rows", "cols", "labels")}
    rng = np.random.default_rng(7)

    def balanced_accuracy(logits, target):
        prediction = logits.argmax(-1)
        scores = []
        for cls in (0, 1):
            mask = target == cls
            if mask.any():
                scores.append(float((prediction[mask] == cls).float().mean()))
        return float(np.mean(scores)), prediction

    best = {"acc": -1.0, "step": 0, "state": None, "prediction": None}
    for step in range(1, 1201):
        pick = torch.from_numpy(rng.integers(0, len(rows), 4096)).to(device)
        optimizer.zero_grad()
        logits = head(state, rows[pick], cols[pick])
        loss = F.cross_entropy(logits, labels[pick])
        loss.backward()
        optimizer.step()
        if step % 50 == 0:
            with torch.no_grad():
                acc, prediction = balanced_accuracy(head(state, valid["rows"], valid["cols"]),
                                                    valid["labels"])
            if acc > best["acc"]:
                best = {"acc": acc, "step": step, "prediction": prediction.clone(),
                        "state": {k: v.detach().clone() for k, v in head.state_dict().items()}}
    head.load_state_dict(best["state"])
    with torch.no_grad():
        test_logits = head(state, test["rows"], test["cols"])
        test_acc, test_pred = balanced_accuracy(test_logits, test["labels"])
    # degree baselines on the same test pairs
    degrees = np.bincount(data["message_edges"].ravel(), minlength=data["n_items"]).astype(float)
    test_rows = data["test_direction"]["rows"]
    test_cols = data["test_direction"]["cols"]
    test_labels = data["test_direction"]["labels"]
    def rule_accuracy(rows, cols, labels, flip=False):
        rule = (degrees[rows] < degrees[cols]).astype(np.int64)
        if flip:
            rule = 1 - rule
        return float(np.mean([(rule[labels == 0] == 0).mean(),
                              (rule[labels == 1] == 1).mean()]))
    valid_rows = data["valid_direction"]["rows"]
    valid_cols = data["valid_direction"]["cols"]
    valid_labels = data["valid_direction"]["labels"]
    flip = rule_accuracy(valid_rows, valid_cols, valid_labels, False) < \
        rule_accuracy(valid_rows, valid_cols, valid_labels, True)
    rule_acc = rule_accuracy(test_rows, test_cols, test_labels, flip)
    x = np.stack([np.log1p(degrees[test_rows]), np.log1p(degrees[test_cols]),
                  np.log1p(degrees[test_rows]) - np.log1p(degrees[test_cols])], 1)
    train_x = np.stack([np.log1p(degrees[data["train_direction"]["rows"]]),
                        np.log1p(degrees[data["train_direction"]["cols"]]),
                        np.log1p(degrees[data["train_direction"]["rows"]])
                        - np.log1p(degrees[data["train_direction"]["cols"]])], 1)
    margin = torch.from_numpy(gt3._logistic_fit_predict(
        train_x, data["train_direction"]["labels"], x, epochs=600)).float()
    logits = torch.stack([-margin, margin], -1)
    degree_acc, _ = balanced_accuracy(logits, torch.from_numpy(test_labels))
    counts = np.bincount(test_pred.cpu().numpy(), minlength=2)
    return {"head": {"test_balanced_accuracy": test_acc, "best_valid": best["acc"],
                     "best_step": best["step"],
                     "prediction_frequency": (counts / counts.sum()).tolist()},
            "baselines": {"in_degree_rule": rule_acc, "degree_logistic": degree_acc,
                          "majority": float(max(np.bincount(test_labels)) / len(test_labels))},
            "test_pairs": int(len(test_labels))}


def main() -> None:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    data = gt3.prepare_data(device)
    n = data["n_items"]
    orbit, coverage_info = gt3.cached_orbit_profile(data)
    subtype = subtype_profile(data, n, 46)
    coverage = (orbit.sum(1) > 0).astype(np.float32)
    train_items, eval_items = split_items(coverage, seed=229)
    features = data["features"]
    orbit_t = torch.from_numpy(orbit.astype(np.float32)).to(device)
    subtype_t = torch.from_numpy(subtype.astype(np.float32)).to(device)
    result = {"coverage": coverage_info, "items": int(coverage.sum()),
              "train_items": int(len(train_items)), "eval_items": int(len(eval_items)),
              "target_entropy": {"orbit": float(-(orbit_t[torch.from_numpy(eval_items).to(device)]
                                                  * torch.log(orbit_t[torch.from_numpy(eval_items).to(device)]
                                                              .clamp_min(1e-12))).sum(-1).mean()),
                                 "subtype": float(-(subtype_t[torch.from_numpy(eval_items).to(device)]
                                                    * torch.log(subtype_t[torch.from_numpy(eval_items).to(device)]
                                                                .clamp_min(1e-12))).sum(-1).mean())}}
    for name, target in (("subtype", subtype_t), ("orbit", orbit_t)):
        model, metrics, logits = train_classifier(features, target, train_items, eval_items, device)
        mean_profile = target[torch.from_numpy(train_items).to(device)].mean(0)
        const_ce, const_kl = constant_ce(target[torch.from_numpy(eval_items).to(device)],
                                         mean_profile)
        metrics["constant_ce"] = const_ce
        metrics["constant_kl"] = const_kl
        result[f"recognition_{name}"] = metrics
        torch.save(model.state_dict(), HERE / f"teacher" / "opt" / f"probe_{name}_768.pth")

    carrier = st.FrozenTeacher(in_dim=768, relation_dim=128, roles=46).to(device)
    carrier.trunk = nn.Sequential(nn.Linear(768, 1024), nn.GELU(), nn.Dropout(0.1),
                                  nn.Linear(1024, 128 + 46 + 376)).to(device)
    optimizer = torch.optim.AdamW(carrier.parameters(), lr=1e-3, weight_decay=1e-4)
    relation_target = F.normalize(torch.from_numpy(
        np.load(HERE / "teacher" / "structure" / "structure_targets_v1.npz")["relation"]
    ).to(device), dim=1)
    train = torch.from_numpy(train_items).to(device)
    subset = torch.from_numpy(eval_items).to(device)
    for step in range(1200):
        optimizer.zero_grad()
        output = carrier(features[train])
        relation = F.normalize(output[:, :128], dim=1)
        subtype_logits = output[:, 128:128 + 46]
        orbit_logits = output[:, 128 + 46:]
        loss = F.mse_loss(relation, relation_target[train])
        loss = loss - (subtype_t[train] * F.log_softmax(subtype_logits, -1)).sum(-1).mean()
        loss = loss - (orbit_t[train] * F.log_softmax(orbit_logits, -1)).sum(-1).mean()
        loss.backward()
        optimizer.step()
    carrier.eval()
    with torch.no_grad():
        output = carrier(features[subset])
        relation = F.normalize(output[:, :128], dim=1)
        subtype_logits = output[:, 128:128 + 46]
        orbit_logits = output[:, 128 + 46:]
        st_ce, st_kl, st_ent = ce_kl(subtype_logits, subtype_t[subset])
        orb_ce, orb_kl, orb_ent = ce_kl(orbit_logits, orbit_t[subset])
        relation_cosine = float((relation * relation_target[subset]).sum(-1).mean())
    mean_sub = subtype_t[train].mean(0)
    mean_orbit = orbit_t[train].mean(0)
    result["carrier"] = {
        "subtype": {"ce": st_ce, "kl": st_kl,
                    "constant_ce": constant_ce(subtype_t[subset], mean_sub)[0],
                    "constant_kl": constant_ce(subtype_t[subset], mean_sub)[1]},
        "orbit": {"ce": orb_ce, "kl": orb_kl,
                  "constant_ce": constant_ce(orbit_t[subset], mean_orbit)[0],
                  "constant_kl": constant_ce(orbit_t[subset], mean_orbit)[1]},
        "relation_cosine": relation_cosine}
    result["direction"] = direction_gate(data, device)
    out = HERE / "teacher" / "opt" / "structure_probe_v2.json"
    out.write_text(json.dumps(result, indent=1))
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
