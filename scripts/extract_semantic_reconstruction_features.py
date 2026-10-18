"""Extract semantic reconstruction quality and prefix neighborhood metrics for 38 models.

Issue #50 implementation:
1. Reference semantic neighbors from frozen Sentence-T5 (24,587 items x 768 dims).
2. Tie-aware prefix neighborhood preservation N_sem (expected Top-20 overlap under uniform tie-breaking).
3. Semantic-pair discrimination AUC between Top-20 semantic positives and matched negatives.
4. True decoder reconstruction R_cos, R_mse, R_norm from trained RQ-VAE checkpoints on all eligible arms.
5. Catalogue-uniform (primary) vs demand-weighted averages.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

REPO_ROOT = Path("/home/wlia0047/ar57/wenyu/GeneRec")
STAGE1_EMB = REPO_ROOT / "stage1_GeneEmbedding/output/sentence_t5.npy"
INVENTORY_PATH = REPO_ROOT / "results/stage3_T5Train/monotonicity/model_inventory.json"
VALID_PAIRS_PATH = REPO_ROOT / "results/stage3_T5Train/monotonicity/behavior_pairs_valid_s42.npz"
TRAIN_PARQUET = REPO_ROOT / "results/stage0_build_parquet/train.parquet"
OUTPUT_DIR = REPO_ROOT / "results/stage3_T5Train/monotonicity/semantic_reconstruction_er_predictors"
LFS_CACHE_DIR = REPO_ROOT / ".git/lfs/objects"


def get_historical_checkpoints() -> dict[str, tuple[str, str, int]]:
    """Map historical git paths to (rev, lfs_oid, size)."""
    mapping: dict[str, tuple[str, str, int]] = {}
    for rev in ["a5f8d9e^", "19d0d69^"]:
        res = subprocess.run(
            ["git", "ls-tree", "-r", "--name-only", rev],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
        )
        for p in res.stdout.splitlines():
            if p.endswith("rqvae_best.pth") and p not in mapping:
                content = subprocess.run(
                    ["git", "show", f"{rev}:{p}"],
                    cwd=str(REPO_ROOT),
                    capture_output=True,
                    text=True,
                ).stdout
                oid = None
                size = 0
                for line in content.splitlines():
                    if line.startswith("oid sha256:"):
                        oid = line.split(":")[1].strip()
                    elif line.startswith("size "):
                        size = int(line.split()[1].strip())
                if oid:
                    mapping[p] = (rev, oid, size)
    return mapping


def resolve_checkpoint(
    sid_path: str, historical: dict[str, tuple[str, str, int]]
) -> tuple[Path | None, str, str]:
    """Find checkpoint file, returning (path, source_desc, lfs_oid)."""
    direct = REPO_ROOT / Path(sid_path).parent / "rqvae_best.pth"
    if direct.is_file() and direct.stat().st_size > 1000:
        return direct, "working_tree", ""

    sid_dir = str(Path(sid_path).parent)
    arm_name = Path(sid_path).parent.name
    for hp, (rev, oid, size) in historical.items():
        if hp.startswith(sid_dir) or sid_dir in hp or f"/{arm_name}/" in hp:
            if oid:
                cached = LFS_CACHE_DIR / oid[:2] / oid[2:4] / oid
                if cached.is_file():
                    return cached, f"git_lfs_cache({rev})", oid
            break
    return None, "missing", ""


def load_target_weights(n_items: int) -> tuple[np.ndarray, np.ndarray]:
    """Return normalized target weights for train and valid sets."""
    # Train targets
    train_df = pd.read_parquet(TRAIN_PARQUET, columns=["target"])
    train_counts = np.bincount(train_df["target"].to_numpy(dtype=np.int64), minlength=n_items)[:n_items]
    w_train = train_counts.astype(np.float64)
    w_train /= w_train.sum()

    # Valid targets
    valid_pos = np.load(VALID_PAIRS_PATH)["positives"].astype(np.int64)
    valid_counts = np.bincount(valid_pos, minlength=n_items)[:n_items]
    w_valid = valid_counts.astype(np.float64)
    w_valid /= w_valid.sum()

    return w_train, w_valid


class MLPDecoder(nn.Module):
    """Reconstruct original 768-d embeddings from 32-d quantized latent."""

    def __init__(self, in_dim: int = 32, hidden_sizes: tuple[int, ...] = (128, 256, 512), out_dim: int = 768):
        super().__init__()
        layers = [
            nn.Dropout(p=0.0),
            nn.Linear(in_dim, hidden_sizes[0]),
            nn.ReLU(),
            nn.Dropout(p=0.0),
            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            nn.ReLU(),
            nn.Dropout(p=0.0),
            nn.Linear(hidden_sizes[1], hidden_sizes[2]),
            nn.ReLU(),
            nn.Dropout(p=0.0),
            nn.Linear(hidden_sizes[2], out_dim),
        ]
        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


def compute_neighborhood_metrics(
    sid_3tokens: np.ndarray,
    top20_neighbors: np.ndarray,
    neg_indices: np.ndarray,
) -> tuple[float, float, float]:
    """Compute tie-aware Top-20 prefix overlap (N_sem) and semantic AUC."""
    N = len(sid_3tokens)
    sid = sid_3tokens

    # Prefix counts
    u1, c1 = np.unique(sid[:, 0], return_counts=True)
    count_l1 = dict(zip(u1, c1))
    p2 = sid[:, 0] * 1000 + sid[:, 1]
    u2, c2 = np.unique(p2, return_counts=True)
    count_l2 = dict(zip(u2, c2))
    p3 = sid[:, 0] * 1000000 + sid[:, 1] * 1000 + sid[:, 2]
    u3, c3 = np.unique(p3, return_counts=True)
    count_l3 = dict(zip(u3, c3))

    n3 = np.array([count_l3[p3[i]] - 1 for i in range(N)])
    n2 = np.array([count_l2[p2[i]] - count_l3[p3[i]] for i in range(N)])
    n1 = np.array([count_l1[sid[i, 0]] - count_l2[p2[i]] for i in range(N)])
    n0 = (N - 1) - (n3 + n2 + n1)

    # Positive neighbor depths
    sid_self = sid[:, None, :]
    sid_neigh = sid[top20_neighbors, :]
    m_l1 = (sid_self[:, :, 0] == sid_neigh[:, :, 0])
    m_l2 = m_l1 & (sid_self[:, :, 1] == sid_neigh[:, :, 1])
    m_l3 = m_l2 & (sid_self[:, :, 2] == sid_neigh[:, :, 2])
    depths_pos = m_l1.astype(int) + m_l2.astype(int) + m_l3.astype(int)

    m3 = (depths_pos == 3).sum(axis=1)
    m2 = (depths_pos == 2).sum(axis=1)
    m1 = (depths_pos == 1).sum(axis=1)
    m0 = (depths_pos == 0).sum(axis=1)

    # Tie-aware analytic overlap
    take3 = np.minimum(20, n3)
    frac3 = np.divide(take3, n3, out=np.zeros_like(take3, dtype=float), where=n3 > 0)
    exp3 = np.where(n3 <= 20, m3, m3 * frac3)
    rem = 20 - take3

    take2 = np.minimum(rem, n2)
    frac2 = np.divide(take2, n2, out=np.zeros_like(take2, dtype=float), where=n2 > 0)
    exp2 = np.where(n2 <= rem, m2, m2 * frac2)
    rem = rem - take2

    take1 = np.minimum(rem, n1)
    frac1 = np.divide(take1, n1, out=np.zeros_like(take1, dtype=float), where=n1 > 0)
    exp1 = np.where(n1 <= rem, m1, m1 * frac1)
    rem = rem - take1

    take0 = np.minimum(rem, n0)
    frac0 = np.divide(take0, n0, out=np.zeros_like(take0, dtype=float), where=n0 > 0)
    exp0 = np.where(n0 <= rem, m0, m0 * frac0)

    expected_overlap = exp3 + exp2 + exp1 + exp0
    n_sem = float((expected_overlap / 20.0).mean())
    overlap_count = float(expected_overlap.mean())

    # Semantic AUC
    sid_neg = sid[neg_indices, :]
    n_l1 = (sid_self[:, :, 0] == sid_neg[:, :, 0])
    n_l2 = n_l1 & (sid_self[:, :, 1] == sid_neg[:, :, 1])
    n_l3 = n_l2 & (sid_self[:, :, 2] == sid_neg[:, :, 2])
    depths_neg = n_l1.astype(int) + n_l2.astype(int) + n_l3.astype(int)

    # Compare depths: (N, 20, 1) vs (N, 1, 20)
    dp = depths_pos[:, :, None]
    dn = depths_neg[:, None, :]
    auc = float(((dp > dn).astype(float) + 0.5 * (dp == dn).astype(float)).mean())

    return n_sem, overlap_count, auc


def compute_decoder_reconstruction(
    ckpt_path: Path,
    sid_3tokens: np.ndarray,
    orig_emb: np.ndarray,
    w_train: np.ndarray,
    w_valid: np.ndarray,
) -> dict[str, float]:
    """Run faithful decoder reconstruction from frozen 3-token SIDs."""
    state = torch.load(ckpt_path, map_location="cpu")
    sd = state.get("state_dict", state)

    # Load codebook embeddings
    cb0 = sd["rq.vq_layers.0.embed.weight"]
    cb1 = sd["rq.vq_layers.1.embed.weight"]
    cb2 = sd["rq.vq_layers.2.embed.weight"]

    # Build decoder
    decoder = MLPDecoder()
    dec_state = {
        "mlp.1.weight": sd["decoder.mlp.1.weight"],
        "mlp.1.bias": sd["decoder.mlp.1.bias"],
        "mlp.4.weight": sd["decoder.mlp.4.weight"],
        "mlp.4.bias": sd["decoder.mlp.4.bias"],
        "mlp.7.weight": sd["decoder.mlp.7.weight"],
        "mlp.7.bias": sd["decoder.mlp.7.bias"],
        "mlp.10.weight": sd["decoder.mlp.10.weight"],
        "mlp.10.bias": sd["decoder.mlp.10.bias"],
    }
    decoder.load_state_dict(dec_state)
    decoder.eval()

    # Reconstruct quantized representations
    toks = torch.from_numpy(sid_3tokens)
    c0 = cb0[toks[:, 0]]
    c1 = cb1[toks[:, 1]]
    c2 = cb2[toks[:, 2]]
    q = c0 + c1 + c2

    with torch.no_grad():
        recon = decoder(q)

    # Compute metrics against orig_emb
    e = torch.from_numpy(orig_emb)
    cos = F.cosine_similarity(e, recon, dim=1).numpy()
    diff_sq = ((e - recon) ** 2).sum(dim=1).numpy()
    norm_sq = (e ** 2).sum(dim=1).numpy() + 1e-12

    mse_per_item = diff_sq / 768.0
    rel_norm_per_item = diff_sq / norm_sq

    return {
        "R_cos_uniform": float(cos.mean()),
        "R_cos_train_weighted": float((cos * w_train).sum()),
        "R_cos_valid_weighted": float((cos * w_valid).sum()),
        "R_mse_uniform": float(mse_per_item.mean()),
        "R_mse_train_weighted": float((mse_per_item * w_train).sum()),
        "R_mse_valid_weighted": float((mse_per_item * w_valid).sum()),
        "R_norm_uniform": float(rel_norm_per_item.mean()),
        "R_norm_train_weighted": float((rel_norm_per_item * w_train).sum()),
        "R_norm_valid_weighted": float((rel_norm_per_item * w_valid).sum()),
    }


def main() -> None:
    t0 = time.time()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading Sentence-T5 original embeddings...")
    orig_emb = np.load(STAGE1_EMB)
    N, D = orig_emb.shape
    print(f"Loaded {N} items x {D} dims")

    # Compute Top-20 cosine neighbors on GPU
    print("Computing Top-20 semantic neighbors on GPU...")
    emb_t = torch.from_numpy(orig_emb).cuda()
    emb_norm = F.normalize(emb_t, p=2, dim=1)
    sim = torch.mm(emb_norm, emb_norm.t())
    sim.fill_diagonal_(-float("inf"))
    _, top20_indices = torch.topk(sim, k=20, dim=1)
    top20 = top20_indices.cpu().numpy()

    # Pre-generate 20 popularity-matched random negatives per item
    print("Generating deterministic negatives (seed=42)...")
    rng = np.random.default_rng(42)
    neg_indices = np.zeros((N, 20), dtype=np.int64)
    for i in range(N):
        forbidden = set(top20[i]).union({i})
        # Random draw from catalogue excluding self and top20
        candidates = rng.choice(N, size=40, replace=False)
        chosen = [c for c in candidates if c not in forbidden][:20]
        while len(chosen) < 20:
            extra = int(rng.integers(0, N))
            if extra not in forbidden and extra not in chosen:
                chosen.append(extra)
        neg_indices[i] = chosen

    # Load target weights
    print("Loading train and valid target demand weights...")
    w_train, w_valid = load_target_weights(N)

    # Historical checkpoints mapping
    print("Resolving historical checkpoints...")
    historical = get_historical_checkpoints()

    # Model inventory
    inv = json.load(open(INVENTORY_PATH))
    models = inv["models"]
    print(f"Auditing and extracting features for {len(models)} models...")

    rows = []
    for idx, m in enumerate(models):
        mid = m["model"]
        fam = m["family"]
        sid_p = m["sid_path"]
        rec10 = m["recall@10"]
        ndcg10 = m["ndcg@10"]

        print(f"[{idx+1}/{len(models)}] {mid} ({fam})...")

        # Load SID
        sid_full_path = REPO_ROOT / sid_p
        if not sid_full_path.is_file():
            # Check /tmp/r3_recover
            sid_full_path = Path("/tmp/r3_recover") / sid_p
        sids_raw = json.load(open(sid_full_path))
        sid_matrix = np.full((N, 4), -1, dtype=np.int64)
        for iid_str, code in sids_raw.items():
            sid_matrix[int(iid_str)] = code
        sid_3 = sid_matrix[:, :3]

        # 1. Neighborhood features (always computed)
        n_sem, exp_overlap, auc = compute_neighborhood_metrics(sid_3, top20, neg_indices)

        # 2. Checkpoint resolution and decoder reconstruction
        ckpt_path, ckpt_source, lfs_oid = resolve_checkpoint(sid_p, historical)
        recon_metrics: dict[str, Any] = {}
        if ckpt_path and ckpt_path.is_file():
            try:
                recon_metrics = compute_decoder_reconstruction(
                    ckpt_path, sid_3, orig_emb, w_train, w_valid
                )
                recon_available = True
            except Exception as e:
                print(f"  Warning: failed to compute reconstruction: {e}")
                recon_available = False
        else:
            recon_available = False

        row = {
            "model": mid,
            "family": fam,
            "recall@10": rec10,
            "ndcg@10": ndcg10,
            "sid_path": sid_p,
            "recon_available": recon_available,
            "ckpt_source": ckpt_source,
            "N_sem": n_sem,
            "N_sem_expected_overlap_count": exp_overlap,
            "semantic_AUC": auc,
            "R_cos_uniform": recon_metrics.get("R_cos_uniform", np.nan),
            "R_cos_train_weighted": recon_metrics.get("R_cos_train_weighted", np.nan),
            "R_cos_valid_weighted": recon_metrics.get("R_cos_valid_weighted", np.nan),
            "R_mse_uniform": recon_metrics.get("R_mse_uniform", np.nan),
            "R_mse_train_weighted": recon_metrics.get("R_mse_train_weighted", np.nan),
            "R_mse_valid_weighted": recon_metrics.get("R_mse_valid_weighted", np.nan),
            "R_norm_uniform": recon_metrics.get("R_norm_uniform", np.nan),
            "R_norm_train_weighted": recon_metrics.get("R_norm_train_weighted", np.nan),
            "R_norm_valid_weighted": recon_metrics.get("R_norm_valid_weighted", np.nan),
        }
        rows.append(row)

    out_csv = OUTPUT_DIR / "per_model_semantic_metrics.csv"
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nWrote {len(rows)} rows to {out_csv} in {time.time()-t0:.2f}s")
    n_recon = sum(1 for r in rows if r["recon_available"])
    print(f"Complete cases: N_sem = {len(rows)}/38, R_cos = {n_recon}/38")


if __name__ == "__main__":
    main()
