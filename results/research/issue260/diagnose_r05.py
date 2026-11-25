#!/usr/bin/env python3.9
"""R05 root-cause probe: read-only checkpoint interventions, no training.

Measures the chain the issue asks about:
  topology tokens -> gate -> curvature cost -> hard code -> prefix behaviour.
Every condition reuses the same encoder and codebooks; only the declared switch
changes. Output: results/research/issue260/diagnosis/r05_probe.json
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
ARM = REPO / "stage2_RQ-VAE" / "TIGER_role_conditioned_l2_rq"
RUN = REPO / "results/stage2_RQ-VAE/_runs/issue256_r05_formal/TIGER_role_conditioned_l2_rq/result"
OUT = REPO / "results/research/issue260/diagnosis"
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(ARM))

import torch
from types import SimpleNamespace
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from train_config import load_train_config
from distances import build_distance
from model import RQVAE


def entropy(counts):
    total = sum(counts)
    return -sum((c / total) * math.log2(c / total) for c in counts if c)


def argmax_share(values):
    counts = np.bincount(values, minlength=3)
    return counts / counts.sum()


def behaviour_rates(codes):
    pairs = np.load(REPO / "results/stage3_T5Train/monotonicity/behavior_pairs_valid_s42.npz")
    anchors, positives, negatives = pairs["anchors"], pairs["positives"], pairs["negatives"]
    out = {}
    for level in (1, 2, 3):
        pos = (codes[anchors, :level] == codes[positives, :level]).all(axis=1).mean()
        neg = (codes[anchors, None, :level] == codes[negatives, :level]).all(axis=2).mean()
        out["L%d" % level] = {"positive_rate": float(pos), "negative_rate": float(neg),
                              "lift": float(pos / neg) if neg else None}
    return out


def prefix_stats(codes, roles):
    out = {}
    for level in (1, 2, 3):
        seen = {}
        for row, role in zip(codes, roles):
            key = tuple(int(v) for v in row[:level])
            seen.setdefault(key, []).append(int(role))
        out["L%d" % level] = {
            "prefixes": len(seen),
            "items_per_prefix_mean": len(codes) / len(seen),
            "items_per_prefix_median": float(np.median([len(v) for v in seen.values()])),
            "role_purity_weighted": float(sum(max(__import__("collections").Counter(v).values())
                                              for v in seen.values()) / len(codes)),
            "collision": float(1 - len(seen) / len(codes)),
        }
    out["triples"] = len({tuple(int(v) for v in row[:3]) for row in codes})
    return out


def role_mixture_table(model):
    weights = model.role_mix.weights().detach().cpu().numpy()
    rows = {}
    for name, row in [("NO_ROLE", 8), ("GLOBAL", 9)] + [("role%d" % i, i) for i in range(8)]:
        rows[name] = {"weights": [round(float(v), 6) for v in weights[row]],
                      "entropy": round(float(entropy(weights[row])), 6)}
    return rows


def gate_metrics(weights, roles, degree):
    w = weights.detach().cpu().numpy()  # [N,2,3]
    out = {}
    for level in range(2):
        rows = w[:, level, :]
        ent = np.array([entropy(row) for row in rows])
        out["level%d" % (level + 1)] = {
            "mean_weights": [float(v) for v in rows.mean(axis=0)],
            "mean_entropy": float(ent.mean()),
            "argmax_share": [float(v) for v in argmax_share(rows.argmax(axis=1))],
            "share_entropy_below_0_5": float((ent < 0.5).mean()),
            "role_eta2": float(role_eta2(rows, roles)),
            "degree_pearson": [float(np.corrcoef(rows[:, e], degree)[0, 1])
                               for e in range(3)],
        }
    return out


def role_eta2(rows, roles):
    valid = roles >= 0
    if not valid.any():
        return 0.0
    total = rows[valid].var(axis=0).sum()
    between = 0.0
    for role in range(8):
        mask = valid & (roles == role)
        if mask.any():
            between += mask.sum() * ((rows[mask].mean(axis=0) - rows[valid].mean(axis=0)) ** 2).sum()
    return float(between / (len(rows[valid]) * total)) if total > 0 else 0.0


def token_probe(tokens, mask, roles, degree, item_ids, train_ids):
    valid_items = [i for i in train_ids if roles[i] >= 0]
    rng = np.random.default_rng(42)
    if len(valid_items) > 6000:
        valid_items = list(rng.choice(valid_items, 6000, replace=False))
    valid_items = np.asarray(valid_items, dtype=np.int64)
    token_mean = []
    for item in valid_items:
        token_mean.append(tokens[item][mask[item][:-1]].mean(axis=0))
    token_mean = np.asarray(token_mean)
    labels = roles[valid_items]
    degree_only = degree[valid_items][:, None]
    split = int(len(valid_items) * 0.8)
    out = {}
    for name, features in (("tokens", token_mean),
                           ("degree", degree_only),
                           ("tokens_plus_degree", np.column_stack([token_mean, degree_only]))):
        scaler = StandardScaler().fit(features[:split])
        clf = LogisticRegression(max_iter=1000, n_jobs=1).fit(
            scaler.transform(features[:split]), labels[:split])
        out[name] = {
            "accuracy": float(clf.score(scaler.transform(features[split:]), labels[split:])),
            "majority_baseline": float(np.bincount(labels[split:]).max() / len(labels[split:])),
            "train_items": int(split), "test_items": int(len(valid_items) - split),
        }
    return out


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    config = load_train_config(ARM / "config/train_config.json")
    sizes = list(config.CODEBOOK_SIZE)
    if len(sizes) == 1:
        sizes = sizes * config.CODEBOOK_NUM
    tokenizer = SimpleNamespace(hidden_sizes=config.HIDDEN_SIZES,
                                codebook_num=config.CODEBOOK_NUM,
                                codebook_size=tuple(sizes),
                                codebook_dim=config.CODEBOOK_DIM,
                                dropout=0.0, beta=config.BETA, vq_type=config.VQ_TYPE,
                                ema_decay=config.EMA_DECAY, fix_code_embs=False,
                                sk_epsilon=config.SK_EPSILON, sk_iters=config.SK_ITERS)
    model = RQVAE(tokenizer, in_dim=768, distance=build_distance(config)).to(device)
    checkpoint = torch.load(RUN / "rqvae_best.pth", map_location="cpu")
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()

    embeddings = torch.from_numpy(np.load(REPO / "stage1_GeneEmbedding/output/sentence_t5.npy")).float().to(device)
    ids = np.arange(len(embeddings), dtype=np.int64)
    roles = np.asarray(json.loads((ARM / "role_table.json").read_text())["table"], dtype=np.int64)
    degree = np.diff(model.store.undirected.indptr).astype(np.float64)

    with torch.no_grad():
        encoded = model.encoder(embeddings)
        tokens, weights = model.structure.gates_for_items(model.store, ids, device, chunk=1024,
                                                          behavior_on=True)
        model.last_weights = weights
        # Base assignment exactly as the exporter: role mixture, curve on, hard nearest.
        codes = {}
        costs = {}
        for name, role_on, curve_on in (("base", True, True),
                                        ("behavior_off", False, True),
                                        ("curve_off", True, False),
                                        ("both_off", False, False)):
            model.role_mixture_on = role_on
            model.curvature_on = curve_on
            _, _, _, token = model.quantize(encoded, weights, balanced=False,
                                            role_rows=model.role_mix.rows_for(ids, device))
            codes[name] = token.detach().cpu().numpy()
            costs[name] = [cost.detach().cpu().numpy() for cost in model.last_costs]
        model.role_mixture_on = True
        model.curvature_on = True
        # Gate intervention: same encoder and cost map, one uniform mixture per level.
        uniform = torch.full_like(weights, 1.0 / 3.0)
        _, _, _, token = model.quantize(encoded, uniform, balanced=False,
                                        role_rows=model.role_mix.rows_for(ids, device))
        codes["gate_uniform"] = token.detach().cpu().numpy()
        costs["gate_uniform"] = [cost.detach().cpu().numpy() for cost in model.last_costs]

        # Per-level hard quantisation error from the actual selected hard codes.
        hard = torch.zeros_like(encoded)
        residual = encoded
        quant = {}
        for level, layer in enumerate(model.rq.vq_layers):
            selected = layer.embed.weight[torch.as_tensor(codes["base"][:, level], device=device)]
            hard = hard + selected
            residual = encoded - hard
            quant["L%d" % (level + 1)] = float(residual.square().mean().item())
        base_codes = codes["base"]
        # Exported SID identity check.
        exported = json.loads((RUN / "item_sids.json").read_text())
        exported_codes = np.array([exported[str(i)][:3] for i in range(len(embeddings))], dtype=np.int64)

        # Margins from the base cost matrices.
        margins = {}
        for level in range(3):
            cost = costs["base"][level]
            sorted_cost = np.sort(cost, axis=1)
            gap = sorted_cost[:, 1] - sorted_cost[:, 0]
            scale = np.abs(sorted_cost[:, 0]) + 1e-9
            margins["L%d" % (level + 1)] = {
                "mean": float(gap.mean()), "median": float(np.median(gap)),
                "p10": float(np.percentile(gap, 10)),
                "relative_median": float(np.median(gap / scale)),
                "fraction_gap_lt_1pct_best": float((gap < 0.01 * scale).mean()),
            }

        # Code changes under each switch.
        changes = {}
        for name in ("behavior_off", "curve_off", "both_off", "gate_uniform"):
            changes[name] = {
                "L%d" % (level + 1): float((codes[name][:, level] != base_codes[:, level]).mean())
                for level in range(3)
            }
            changes[name]["prefix2_change"] = float((codes[name][:, :2] != base_codes[:, :2]).any(axis=1).mean())
            changes[name]["prefix3_change"] = float((codes[name][:, :3] != base_codes[:, :3]).any(axis=1).mean())

        train_ids = np.unique(__import__("pandas").read_parquet(
            REPO / "results/stage0_build_parquet/train.parquet", columns=["target"])["target"].to_numpy(np.int64))
        baseline_map = json.loads((REPO / "results/stage2_RQ-VAE/TIGER_RQ-VAE/item_sids.json").read_text())
        baseline_codes = np.array(
            [[int(v) for v in baseline_map[str(i)][:3]] for i in range(len(embeddings))], dtype=np.int64)
        np.savez_compressed(OUT / "r05_codes.npz",
                            **{name: codes[name] for name in codes},
                            base_exported=exported_codes, baseline=baseline_codes)
        print("stage: codes saved, building report", flush=True)
        report = {
            "schema": "genrec.issue260.r05_probe/1",
            "run_id": checkpoint.get("run_id"),
            "checkpoint": {"path": str(RUN / "rqvae_best.pth"), "epoch": checkpoint.get("epoch"),
                           "collision": checkpoint.get("collision")},
            "items": int(len(embeddings)),
            "device": str(device),
            "export_match": bool(np.array_equal(base_codes, exported_codes)),
            "role_distribution": {str(int(k)): int(v) for k, v in zip(*np.unique(roles, return_counts=True))},
            "role_mixture": role_mixture_table(model),
            "gate": gate_metrics(weights, roles, degree),
            "token_probe": token_probe(tokens.detach().cpu().numpy(),
                                       model.structure.last_token_mask.detach().cpu().numpy(),
                                       roles, degree, ids, train_ids),
            "quantisation_error": quant,
            "margins": margins,
            "code_changes": changes,
            "prefixes": {name: prefix_stats(codes[name], roles) for name in codes},
            "behaviour": {name: behaviour_rates(codes[name]) for name in codes},
            "behaviour_reference": {
                "R05_item_sids": behaviour_rates(exported_codes),
                "baseline_item_sids": behaviour_rates(baseline_codes),
            },
        }
    (OUT / "r05_probe.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: report[k] for k in ("export_match", "role_mixture", "gate", "token_probe",
                                             "quantisation_error", "margins", "code_changes",
                                             "behaviour_reference")}, indent=1))


if __name__ == "__main__":
    main()
