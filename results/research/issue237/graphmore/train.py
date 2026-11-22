"""Train the GraphMoRE representation on the frozen item graph.

Objective (official ``exp.py::cal_lp_loss`` + ``compute_distortion``):

    score(u,v) = sigmoid((r - d_w(u,v)) / t)         Fermi-Dirac decoder
    d_w(u,v)   = sum_e softmax(w_u ⊙ w_v)[e] · ||z_u^e - z_v^e||²
    L_lp       = BCE(score(pos), 1) + BCE(score(neg), 0)     clipped to [0.01, 0.99]
    L_dis      = mean | d_w(m,n) / d_graph(m,n) - 1 |
    L          = L_lp + coef_dis · L_dis

``d_graph`` and the propagation graph both come from the **training** edge split only
(issue #237 C13).  Model selection is on validation AP; the test split is touched once,
after the best epoch is frozen.
"""
from __future__ import annotations

import json
import resource
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from . import data as data_module
from .model import CURVATURES, FermiDirac, GraphMoRE, sparse_adjacency

CONFIG = {
    "hidden_dim": 256,
    "expert_dim": 128,
    "gating_hidden": 64,
    "gating_out": 32,
    "out_dim": 768,
    "learning_rate": 1e-3,
    "weight_decay": 1e-6,
    "coef_dis": 0.1,
    # Project adaptation (issue #237 C17): cosines between the exported vector and the
    # frozen TIGER embedding. 0 disables it, which is the faithful GraphMoRE recipe.
    "coef_sem": 0.0,
    "epochs": 400,
    "eval_every": 5,
    "patience": 40,
    "decoder_r": 2.0,
    "decoder_t": 1.0,
    "seed": 42,
}


def _tensor(array, device, dtype=torch.float32):
    return torch.as_tensor(array, dtype=dtype, device=device)


def _edges_tensor(edges, device):
    return torch.as_tensor(edges.T.copy(), dtype=torch.long, device=device)


def auroc_ap(scores: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    """AUC and average precision, computed without sklearn."""
    order = np.argsort(-scores)
    ranked = labels[order]
    positives = int(ranked.sum())
    negatives = int(ranked.size - positives)
    if positives == 0 or negatives == 0:
        return float("nan"), float("nan")
    tp = np.cumsum(ranked)
    fp = np.cumsum(1 - ranked)
    tpr = tp / positives
    fpr = fp / negatives
    auc = float(np.trapezoid(tpr, fpr)) if hasattr(np, "trapezoid") else float(np.trapz(tpr, fpr))
    precision = tp / np.maximum(tp + fp, 1)
    ap = float((precision * ranked).sum() / positives)
    return auc, ap


def evaluate_split(model, embedding, weights, decoder, edges, negatives, device) -> dict:
    with torch.no_grad():
        pos = model.align(embedding, weights, _tensor(edges[:, 0], device, torch.long),
                          _tensor(edges[:, 1], device, torch.long))
        neg = model.align(embedding, weights, _tensor(negatives[:, 0], device, torch.long),
                          _tensor(negatives[:, 1], device, torch.long))
        pos_scores = decoder(pos).cpu().numpy()
        neg_scores = decoder(neg).cpu().numpy()
    scores = np.concatenate([pos_scores, neg_scores])
    labels = np.concatenate([np.ones_like(pos_scores), np.zeros_like(neg_scores)])
    auc, ap = auroc_ap(scores, labels)
    return {"auc": auc, "ap": ap, "pos_mean": float(pos_scores.mean()),
            "neg_mean": float(neg_scores.mean())}


def train(device_name: str = "cuda", config: dict | None = None,
          output_dir: Path | None = None) -> dict:
    config = {**CONFIG, **(config or {})}
    torch.manual_seed(config["seed"])
    np.random.seed(config["seed"])
    device = torch.device(device_name)
    output_dir = output_dir or Path(__file__).resolve().parents[1]
    output_dir.mkdir(parents=True, exist_ok=True)

    bundle = data_module.build(config["seed"])
    n = bundle["n"]
    adjacency = sparse_adjacency(
        _edges_tensor(bundle["split"]["train"], device), n, device)
    # ego indicators stay sparse: a dense 24587 x 24587 float matrix is 2.4 GB each
    egos = [torch.sparse_coo_tensor(
                torch.as_tensor(np.stack(matrix.nonzero()), dtype=torch.long, device=device),
                torch.as_tensor(matrix.data, dtype=torch.float32, device=device),
                matrix.shape).coalesce()
            for matrix in (bundle["ego"][2], bundle["ego"][3])]
    x = _tensor(bundle["embeddings"], device)

    model = GraphMoRE(in_dim=x.shape[1], hidden_dim=config["hidden_dim"],
                      expert_dim=config["expert_dim"],
                      gating_hidden=config["gating_hidden"],
                      gating_out=config["gating_out"], out_dim=config["out_dim"],
                      curvatures=CURVATURES, num_scales=len(egos)).to(device)
    decoder = FermiDirac(config["decoder_r"], config["decoder_t"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"],
                                 weight_decay=config["weight_decay"])

    train_edges = _edges_tensor(bundle["split"]["train"], device)
    train_negatives = _edges_tensor(bundle["negatives"]["train"], device)
    distortion_pairs = _edges_tensor(bundle["distortion_pairs"], device)
    distortion_target = _tensor(bundle["distortion_target"], device)
    distortion_mask = torch.as_tensor(bundle["distortion_mask"], device=device)

    log_path = output_dir / "train_log.jsonl"
    log_path.write_text("", encoding="utf-8")
    best = {"ap": -1.0, "epoch": -1, "state": None}
    start = time.time()
    history = []
    for epoch in range(1, config["epochs"] + 1):
        model.train()
        optimizer.zero_grad()
        embedding, weights = model(x, adjacency, egos)

        pos = model.align(embedding, weights, train_edges[0], train_edges[1])
        neg = model.align(embedding, weights, train_negatives[0], train_negatives[1])
        pos_score = decoder(pos).clamp(0.01, 0.99)
        neg_score = decoder(neg).clamp(0.01, 0.99)
        loss_lp = (F.binary_cross_entropy(pos_score, torch.ones_like(pos_score))
                   + F.binary_cross_entropy(neg_score, torch.zeros_like(neg_score)))

        dis = model.align(embedding, weights, distortion_pairs[0], distortion_pairs[1])
        ratio = dis[distortion_mask] / distortion_target[distortion_mask]
        loss_dis = (ratio - 1.0).abs().mean()
        loss = loss_lp + config["coef_dis"] * loss_dis
        loss_sem = torch.zeros((), device=device)
        if config.get("coef_sem", 0.0) > 0.0:
            projected = model.head(embedding)
            loss_sem = (1.0 - F.cosine_similarity(projected, x, dim=-1)).mean()
            loss = loss + config["coef_sem"] * loss_sem
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()

        row = {"epoch": epoch, "loss": float(loss), "loss_lp": float(loss_lp),
               "loss_dis": float(loss_dis), "loss_sem": float(loss_sem),
               "elapsed_s": round(time.time() - start, 1)}
        if epoch % config["eval_every"] == 0 or epoch == 1:
            model.eval()
            with torch.no_grad():
                embedding_eval, weights_eval = model(x, adjacency, egos)
            val = evaluate_split(model, embedding_eval, weights_eval, decoder,
                                 bundle["split"]["val"], bundle["negatives"]["val"], device)
            row["val_auc"] = val["auc"]
            row["val_ap"] = val["ap"]
            if val["ap"] > best["ap"]:
                best = {"ap": val["ap"], "epoch": epoch,
                        "state": {key: value.detach().cpu().clone()
                                  for key, value in model.state_dict().items()}}
                torch.save({"model": best["state"], "config": config, "epoch": epoch,
                            "curvatures": model.experts.curvature_values()},
                           output_dir / "best_checkpoint.pt")
        history.append(row)
        with log_path.open("a", encoding="utf-8") as sink:
            sink.write(json.dumps(row) + "\n")
        if epoch - best["epoch"] > config["patience"] and best["epoch"] > 0:
            row["early_stop"] = True
            with log_path.open("a", encoding="utf-8") as sink:
                sink.write(json.dumps(row) + "\n")
            break

    # test split: touched once, after the best epoch is frozen
    model.load_state_dict(best["state"])
    model.eval()
    with torch.no_grad():
        embedding, weights = model(x, adjacency, egos)
    test = evaluate_split(model, embedding, weights, decoder,
                          bundle["split"]["test"], bundle["negatives"]["test"], device)

    summary = {
        "best_epoch": best["epoch"],
        "best_val_ap": best["ap"],
        "test": test,
        "curvatures_final": model.experts.curvature_values(),
        "config": config,
        "split_sizes": bundle["split_sizes"],
        "graph_sha256": bundle["graph_sha256"],
        "epochs_run": len(history),
        "train_s": round(time.time() - start, 1),
        "peak_rss_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1),
        "device": str(device),
    }
    (output_dir / "train_summary.json").write_text(json.dumps(summary, indent=1),
                                                   encoding="utf-8")
    return summary
