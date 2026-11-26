"""Issue #263: independent pretraining of the parent directed ego GNN.

This is a candidate-local pretraining entry, not a new public framework. It
reads the frozen TRAIN graph/egos, the existing directed graphlet-orbit tables
and the frozen curvature targets; it never reads validation/test behaviour.

The GNN is the parent four-token `SubgraphTokens` architecture. A lightweight
per-token distribution head is trained only during pretraining, then removed.
One final checkpoint and the full 24,587-item token/mask cache are exported.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[3]
ISSUE = ROOT / "results/research/issue263"
ARM = "TIGER_frozen_topology_gnn_rq"
SOURCE = ROOT / "stage2_RQ-VAE" / ARM
RESULT = ROOT / "results" / "stage2_RQ-VAE" / ARM
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SOURCE))

from model.structure import TOKEN_TYPES, SubgraphTokens  # noqa: E402
from tools.graph_infra.ego_gnn import EgoStore  # noqa: E402

N_ITEMS = 24587
TOKEN_DIM = 32
SEED = 42
STEPS = 1500
BATCH = 256
TOKEN_ATOL = 1e-4
LR = 1e-3
OTHER_COVERAGE = ROOT / "results/research/directed_graphlet_teacher/teacher/structure/v3/structure_targets_v3.npz"
DGDV = ROOT / "results/research/directed_graphlet_role/dgdv.npz"
TARGETS = ROOT / "assets/v8_runtime_layout/targets.npy"
OFFSETS = ROOT / "assets/v8_runtime_layout/target_offsets.npy"
GRAPH = ROOT / "results/research/behavior_graph_directed/graph.npz"
NODE_MOMENTS = ROOT / "assets/node_moments.npz"
INDEX = ROOT / "assets/index.npz"


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_targets() -> dict:
    """Own orbit / DGDV / curvature distributions, all from TRAIN assets."""
    own = np.load(OTHER_COVERAGE, allow_pickle=False)
    own_orbit = own["own_orbit"].astype(np.float32)
    own_coverage = own["coverage"].astype(np.float32)
    dgdv = np.load(DGDV, allow_pickle=False)["directed_counts"].astype(np.float32)
    graph = np.load(GRAPH, allow_pickle=False)
    rows = graph["rows"].astype(np.int64)
    cols = graph["cols"].astype(np.int64)
    degree = np.bincount(rows, minlength=N_ITEMS) + np.bincount(cols, minlength=N_ITEMS)

    targets = np.load(TARGETS, mmap_mode="r", allow_pickle=False)
    offsets = np.load(OFFSETS, allow_pickle=False)
    if offsets.shape != (N_ITEMS + 1,) or offsets[0] != 0 or offsets[-1] != len(targets):
        raise ValueError("invalid frozen curvature offsets")
    # Fixed signed bins, frozen before any pretraining. Zero is its own bin.
    edges = np.asarray([-np.inf, -4., -3., -2., -1., 0., 1., 2., 3., 4., 8., np.inf])
    hist = np.zeros((N_ITEMS, len(edges) - 1), dtype=np.float32)
    curve_mask = np.zeros(N_ITEMS, dtype=bool)
    for item in range(N_ITEMS):
        start, stop = int(offsets[item]), int(offsets[item + 1])
        if stop <= start:
            continue
        values = np.asarray(targets[start:stop]["target"], dtype=np.float64)
        values = values[np.isfinite(values)]
        if not len(values):
            continue
        counts, _ = np.histogram(values, bins=edges)
        hist[item] = counts.astype(np.float32) / max(float(counts.sum()), 1.0)
        curve_mask[item] = True
    # Normalise only where there is evidence; missing rows stay explicit masks.
    own_norm = own_orbit / np.clip(own_orbit.sum(1, keepdims=True), 1.0, None)
    dgdv_norm = dgdv / np.clip(dgdv.sum(1, keepdims=True), 1.0, None)
    own_mask = own_coverage > 0
    dgdv_mask = dgdv.sum(1) > 0
    return {
        "own": own_norm.astype(np.float32), "own_mask": own_mask,
        "dgdv": dgdv_norm.astype(np.float32), "dgdv_mask": dgdv_mask,
        "curvature": hist, "curve_mask": curve_mask, "degree": degree,
        "curve_edges": edges,
    }


class TokenHead(nn.Module):
    """Per-token distribution readout; mask-weighted pooling over four tokens."""

    def __init__(self, classes: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(TOKEN_DIM, hidden), nn.GELU(),
                                 nn.Linear(hidden, classes))

    def forward(self, tokens: torch.Tensor, mask: torch.Tensor):
        logits = self.net(tokens)
        weights = mask.to(logits.dtype)
        pooled = (logits * weights[..., None]).sum(1) / weights.sum(1).clamp_min(1.0)[:, None]
        return pooled


def kl(logits, target, mask):
    valid = mask & (target.sum(1) > 0)
    if not bool(valid.any()):
        return logits.sum() * 0.0, {"n": 0}
    loss = F.kl_div(F.log_softmax(logits[valid], -1), target[valid],
                    reduction="batchmean")
    return loss, {"n": int(valid.sum())}


def token_effective_rank(tokens):
    # tokens: [N,4,32]; one number per token type, plus the mean.
    values = tokens.detach().double()
    ranks = []
    for index in range(tokens.shape[1]):
        x = values[:, index]
        x = x - x.mean(0)
        eigenvalues = torch.linalg.eigvalsh(x.T @ x / len(x)).clamp_min(1e-12)
        ranks.append(float((eigenvalues.sum() ** 2 / (eigenvalues ** 2).sum()).cpu()))
    return {"per_token": ranks, "mean": float(np.mean(ranks))}


def calibrate_lambda(gnn, heads, tokens, mask, targets, ids):
    gnn_params = tuple(p for p in gnn.parameters() if p.requires_grad)
    role_logits = heads["own"](tokens, mask)
    dgdv_logits = heads["dgdv"](tokens, mask)
    curve_logits = heads["curve"](tokens, mask)
    own_loss, _ = kl(role_logits, torch.from_numpy(targets["own"][ids]).to(tokens.device),
                     torch.from_numpy(targets["own_mask"][ids]).to(tokens.device))
    dgdv_loss, _ = kl(dgdv_logits, torch.from_numpy(targets["dgdv"][ids]).to(tokens.device),
                      torch.from_numpy(targets["dgdv_mask"][ids]).to(tokens.device))
    curve_loss, _ = kl(curve_logits, torch.from_numpy(targets["curvature"][ids]).to(tokens.device),
                       torch.from_numpy(targets["curve_mask"][ids]).to(tokens.device))
    role = 0.5 * (own_loss + dgdv_loss)
    gr = torch.autograd.grad(role, gnn_params, retain_graph=True, allow_unused=True)
    gc = torch.autograd.grad(curve_loss, gnn_params, retain_graph=True, allow_unused=True)
    nr = math.sqrt(sum(float((g.double() ** 2).sum()) for g in gr if g is not None))
    nc = math.sqrt(sum(float((g.double() ** 2).sum()) for g in gc if g is not None))
    value = nr / nc if nc > 0 and math.isfinite(nr / nc) else 1.0
    return float(min(max(value, 1e-3), 1e3)), {"role_grad": nr, "curve_grad": nc,
                                               "raw_ratio": value}


def evaluate(gnn, heads, targets, eval_ids, device, store):
    gnn.eval()
    with torch.inference_mode():
        tokens, mask = gnn.tokens_for_items(store, eval_ids, device, chunk=512)
        own_logits = heads["own"](tokens, mask)
        dgdv_logits = heads["dgdv"](tokens, mask)
        curve_logits = heads["curve"](tokens, mask)
    def metrics(logits, target, valid):
        valid = np.asarray(valid, dtype=bool) & (np.asarray(target).sum(1) > 0)
        idx = np.flatnonzero(valid)
        if not len(idx):
            return {"n": 0}
        logits = logits[idx]
        target_np = np.asarray(target, dtype=np.float64)[idx]
        target = torch.as_tensor(target_np, dtype=torch.float32, device=logits.device)
        ce = float(-(target * F.log_softmax(logits, -1)).sum(-1).mean().cpu())
        entropy = float(-(target * torch.log(torch.clamp(target, 1e-12, None))).sum(-1).mean().cpu())
        pred = logits.argmax(-1).cpu().numpy()
        label = target.argmax(-1).cpu().numpy()
        majority = int(np.bincount(label).argmax())
        minority = label != majority
        # per-item Pearson between predicted probabilities and target distribution
        prob = F.softmax(logits, -1).cpu().numpy()
        p = prob - prob.mean(1, keepdims=True)
        t = target_np - target_np.mean(1, keepdims=True)
        corr = np.mean((p * t).sum(1) / np.clip(np.sqrt((p * p).sum(1) * (t * t).sum(1)),
                                                1e-12, None))
        return {"n": int(len(idx)), "ce": ce, "kl": ce - entropy,
                "target_entropy": entropy, "top1": float((pred == label).mean()),
                "majority_top1": float((label == majority).mean()),
                "minority_items": int(minority.sum()),
                "minority_top1": float((pred[minority] == label[minority]).mean())
                if minority.any() else None, "rank_corr": float(corr)}
    eval_degree = targets["degree"][eval_ids]
    degree_bins = np.unique(np.quantile(targets["degree"], np.linspace(0, 1, 11)))
    degree_key = np.clip(np.digitize(targets["degree"], degree_bins[1:-1], right=True),
                         0, len(degree_bins) - 2)
    fit_ids = np.setdiff1d(np.arange(N_ITEMS), eval_ids)
    fit_keys = degree_key[fit_ids]
    def baseline(target, mask):
        # Full-length array so the same target/mask indices used by base_ce apply.
        out = np.zeros_like(target, dtype=np.float64)
        for key in np.unique(degree_key[eval_ids]):
            fit = fit_ids[fit_keys == key]
            if len(fit):
                selected = eval_ids[degree_key[eval_ids] == key]
                out[selected] = target[fit].mean(0)
        return out
    own_base = baseline(targets["own"], targets["own_mask"])
    dgdv_base = baseline(targets["dgdv"], targets["dgdv_mask"])
    curve_base = baseline(targets["curvature"], targets["curve_mask"])
    def base_ce(target, mask, base):
        idx = np.flatnonzero(mask & (target.sum(1) > 0))
        if not len(idx): return None
        return float(-(target[idx] * np.log(np.clip(base[idx], 1e-12, None))).sum(-1).mean())
    return {
        "own": metrics(own_logits, targets["own"][eval_ids], targets["own_mask"][eval_ids]),
        "dgdv": metrics(dgdv_logits, targets["dgdv"][eval_ids], targets["dgdv_mask"][eval_ids]),
        "curve": metrics(curve_logits, targets["curvature"][eval_ids],
                         targets["curve_mask"][eval_ids]),
        "degree_baseline": {
            "own_ce": base_ce(targets["own"][eval_ids], targets["own_mask"][eval_ids],
                              own_base[eval_ids]),
            "dgdv_ce": base_ce(targets["dgdv"][eval_ids], targets["dgdv_mask"][eval_ids],
                               dgdv_base[eval_ids]),
            "curve_ce": base_ce(targets["curvature"][eval_ids], targets["curve_mask"][eval_ids],
                                curve_base[eval_ids]),
        },
        "eval_items": int(len(eval_ids)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=STEPS)
    parser.add_argument("--out", type=Path, default=ISSUE / "PRETRAIN.json")
    args = parser.parse_args()
    if args.steps <= 0 or args.steps > 5000:
        raise SystemExit("invalid pretraining budget")
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    graph = np.load(GRAPH, allow_pickle=False)
    graph_sha = sha(GRAPH)
    targets = load_targets()
    own_sha = sha(OTHER_COVERAGE)
    dgdv_sha = sha(DGDV)
    target_sha = sha(TARGETS)
    store = EgoStore(ROOT, ROOT / "assets")
    gnn = SubgraphTokens().to(device)
    heads = nn.ModuleDict({
        "own": TokenHead(targets["own"].shape[1]).to(device),
        "dgdv": TokenHead(targets["dgdv"].shape[1]).to(device),
        "curve": TokenHead(targets["curvature"].shape[1]).to(device),
    })
    fit_rng = np.random.default_rng(SEED)
    order = fit_rng.permutation(N_ITEMS)
    cut = int(0.8 * len(order))
    fit_ids = np.sort(order[:cut])
    eval_ids = np.sort(order[cut:])
    optimizer = torch.optim.AdamW(list(gnn.parameters()) + list(heads.parameters()),
                                  lr=LR, weight_decay=0.0)
    calib_ids = fit_rng.choice(fit_ids, size=min(BATCH, len(fit_ids)), replace=False)
    # Do not wrap this call in no_grad: lambda is defined from the actual
    # GNN gradient ratio, so the calibration tokens must retain their graph.
    calib_tokens, calib_mask = gnn.tokens_for_items(store, calib_ids, device, chunk=256)
    lam, grad_diag = calibrate_lambda(gnn, heads, calib_tokens, calib_mask, targets, calib_ids)
    if grad_diag["role_grad"] <= 0 or grad_diag["curve_grad"] <= 0:
        raise RuntimeError("pretraining calibration received a zero GNN gradient")
    print(f"calibrated lambda={lam:.6g} role_grad={grad_diag['role_grad']:.6g} "
          f"curve_grad={grad_diag['curve_grad']:.6g}", flush=True)
    history = []
    torch.manual_seed(SEED)
    for step in range(1, args.steps + 1):
        ids = fit_rng.choice(fit_ids, size=BATCH, replace=False)
        tokens, mask = gnn.tokens_for_items(store, ids, device, chunk=256)
        own_t = torch.from_numpy(targets["own"][ids]).to(device)
        dgdv_t = torch.from_numpy(targets["dgdv"][ids]).to(device)
        curve_t = torch.from_numpy(targets["curvature"][ids]).to(device)
        own_m = torch.from_numpy(targets["own_mask"][ids]).to(device)
        dgdv_m = torch.from_numpy(targets["dgdv_mask"][ids]).to(device)
        curve_m = torch.from_numpy(targets["curve_mask"][ids]).to(device)
        own_loss, own_info = kl(heads["own"](tokens, mask), own_t, own_m)
        dgdv_loss, dgdv_info = kl(heads["dgdv"](tokens, mask), dgdv_t, dgdv_m)
        curve_loss, curve_info = kl(heads["curve"](tokens, mask), curve_t, curve_m)
        role_loss = 0.5 * (own_loss + dgdv_loss)
        loss = role_loss + lam * curve_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(gnn.parameters()) + list(heads.parameters()), 1.0)
        optimizer.step()
        if step == 1 or step % 100 == 0 or step == args.steps:
            entry = {"step": step, "loss": float(loss.detach()), "own": float(own_loss.detach()),
                     "dgdv": float(dgdv_loss.detach()), "curve": float(curve_loss.detach()),
                     "own_n": own_info["n"], "dgdv_n": dgdv_info["n"],
                     "curve_n": curve_info["n"]}
            history.append(entry)
            print(json.dumps(entry), flush=True)
    gnn.eval()
    for p in gnn.parameters():
        p.requires_grad_(False)
    # Export all four token types for every item once, then independently
    # reload the saved GNN and repeat the same full inference.
    with torch.no_grad():
        tokens, mask = gnn.tokens_for_items(store, np.arange(N_ITEMS), device, chunk=512)
        tokens = tokens.detach().cpu().numpy().astype(np.float32)
        mask = mask.detach().cpu().numpy().astype(bool)
        ranks = token_effective_rank(torch.from_numpy(tokens))
    if tokens.shape != (N_ITEMS, 4, TOKEN_DIM) or mask.shape != (N_ITEMS, 4):
        raise ValueError("invalid token export shape")
    if not np.isfinite(tokens).all() or not mask.any():
        raise ValueError("invalid token values/mask")
    RESULT.mkdir(parents=True, exist_ok=True)
    cache_path = RESULT / "token_cache.npz"
    np.savez_compressed(cache_path, tokens=tokens, mask=mask,
                        item_ids=np.arange(N_ITEMS, dtype=np.int64))
    checkpoint = RESULT / "pretrained_gnn.pth"
    torch.save({"state_dict": gnn.state_dict(), "seed": SEED, "steps": args.steps,
                "lambda": lam, "token_types": list(TOKEN_TYPES),
                "role_classes": int(targets["own"].shape[1]),
                "dgdv_classes": int(targets["dgdv"].shape[1]),
                "curve_bins": int(targets["curvature"].shape[1]),
                "curve_edges": [float(x) for x in targets["curve_edges"]],
                "graph_sha256": graph_sha, "own_target_sha256": own_sha,
                "dgdv_target_sha256": dgdv_sha, "curvature_target_sha256": target_sha},
               checkpoint)
    # Independent direct inference from the saved checkpoint, not the live object.
    fresh = SubgraphTokens().to(device)
    fresh.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=False)["state_dict"])
    fresh.eval()
    for p in fresh.parameters():
        p.requires_grad_(False)
    with torch.no_grad():
        direct, direct_mask = fresh.tokens_for_items(store, np.arange(N_ITEMS), device, chunk=512)
        direct = direct.cpu().numpy().astype(np.float32)
        direct_mask = direct_mask.cpu().numpy().astype(bool)
        repeat, repeat_mask = gnn.tokens_for_items(store, np.arange(N_ITEMS), device, chunk=512)
        repeat = repeat.cpu().numpy().astype(np.float32)
        repeat_mask = repeat_mask.cpu().numpy().astype(bool)
    max_abs_diff = float(np.abs(tokens - direct).max()) if tokens.size else 0.0
    same_mask = bool(np.array_equal(mask, direct_mask))
    reproducible = bool(same_mask and max_abs_diff <= TOKEN_ATOL)
    deterministic = bool(np.array_equal(mask, repeat_mask)
                         and float(np.abs(tokens - repeat).max()) <= TOKEN_ATOL)
    if not reproducible or not deterministic:
        raise ValueError("token cache is not reproducible from the frozen checkpoint")
    metrics = evaluate(gnn, heads, targets, eval_ids, device, store)
    result = {
        "schema": "genrec.issue263.pretrain/1",
        "arm": ARM, "seed": SEED, "steps": args.steps, "batch": BATCH,
        "device": str(device), "fit_items": int(len(fit_ids)), "eval_items": int(len(eval_ids)),
        "lambda": lam, "lambda_gradient_diagnostic": grad_diag,
        "history": history, "metrics": metrics,
        "token_effective_rank": ranks,
        "cache": {"path": str(cache_path), "sha256": sha(cache_path),
                  "tokens_shape": list(tokens.shape), "mask_shape": list(mask.shape),
                  "valid_tokens": int(mask.sum()), "empty_items": int((~mask.any(1)).sum())},
        "checkpoint": {"path": str(checkpoint), "sha256": sha(checkpoint)},
        "identity": {"graph": str(GRAPH), "graph_sha256": graph_sha,
                     "own": str(OTHER_COVERAGE), "own_sha256": own_sha,
                     "dgdv": str(DGDV), "dgdv_sha256": dgdv_sha,
                     "curvature": str(TARGETS), "curvature_sha256": target_sha,
                     "source": str(SOURCE), "source_files": len(list(SOURCE.rglob("*.py")))},
        "cache_reproducibility": {"exact_reload": reproducible, "same_mask": same_mask,
                                   "max_abs_diff": max_abs_diff, "tolerance": TOKEN_ATOL,
                                   "deterministic_reexport": deterministic},
        "scope": "pretraining on TRAIN graph and TRAIN labels only; no validation/test data",
    }
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    manifest = {
        "schema": "genrec.issue263.token_manifest/1", "arm": ARM,
        "cache": result["cache"], "checkpoint": result["checkpoint"],
        "item_order": "0..24586", "token_types": list(TOKEN_TYPES),
        "dtype": "float32", "mask_dtype": "bool", "seed": SEED, "steps": args.steps,
        "identity": result["identity"],
        "pretrain_report": str(args.out),
        "cache_reproducibility": result["cache_reproducibility"],
    }
    (RESULT / "token_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"cache": result["cache"], "checkpoint": result["checkpoint"],
                      "metrics": metrics, "lambda": lam}, indent=2), flush=True)


if __name__ == "__main__":
    main()
