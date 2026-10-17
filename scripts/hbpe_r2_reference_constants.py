"""Frozen-reference constants for the HBPE R2 rounds (issues #34, #35, #36).

Everything here is measured from the frozen canonical Exp1a Resample export and
from TRAIN data only. Resample is a threshold measurer, never a warm start.

  H_REF_L1
      Entropy of the hard L1 occupancy histogram over the training item catalogue:
      the ``h = argmin_c ||z - c||`` assignment each training item receives under the
      frozen ``rqvae_best.pth``. This is what the hard-forward/soft-backward
      occupancy barrier regresses toward. It is an entropy rather than a per-code
      target because code identities permute freely between runs, so only the
      entropy is comparable.

  teacher_soft_negative_sharing_endpoints
      B1_neg(tau) and B12_neg(tau) under the frozen Resample teacher, evaluated
      with the same soft formulas the runs use and each model's own
      reference-distance-gap scale. The #36 runs evaluate this online under
      ``no_grad`` on the same batch, which is what the issue specifies; the values
      here are a scale sanity check and are read from the same draws the audit
      already produced.

  negative_sampler_audit
      Issue #34 step 5 asks for an audit of the sampler the A1 family already uses
      -- negatives drawn from the ANCHOR's popularity bucket -- without changing it,
      so the weight ablation stays single-factor. Reports the bucket-match rate
      against both the anchor and the positive, plus the false-negative and
      duplicate checks.

Run:  /home/wlia0047/ar57_scratch/wenyu/genrec_env_v2/bin/python3.9 \
          scripts/hbpe_r2_reference_constants.py
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
A1_TREE = REPO_ROOT / "stage2_RQ-VAE/TIGER_HBPE_A1"
RESAMPLE_EXPORT = REPO_ROOT / "results/stage2_RQ-VAE/TIGER_BEHAVIOR_SAMPLE_RQ-VAE/resample"
TRAIN_FILE = REPO_ROOT / "results/stage0_build_parquet/train.parquet"
OUT_PATH = REPO_ROOT / "results/stage2_RQ-VAE/hbpe_r2_reference_constants.json"

EPS = 1e-8
SEED = 42
N_ITEMS = 24587
K_NEG = 10


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def squared_distances(z: torch.Tensor, codebook: torch.Tensor) -> torch.Tensor:
    z_sq = (z ** 2).sum(-1, keepdim=True)
    c_sq = (codebook ** 2).sum(-1)
    return (z_sq + c_sq - 2.0 * torch.matmul(z, codebook.T)).clamp_min(0.0)


def load_train_pairs() -> np.ndarray:
    frame = pd.read_parquet(TRAIN_FILE, columns=["seen_history", "target"])
    pairs = []
    for history, target in zip(
        frame["seen_history"].to_numpy(), frame["target"].to_numpy(dtype=np.int64)
    ):
        if history is None or len(history) == 0:
            continue
        pairs.append((int(history[-1]), int(target)))
    return np.array(pairs, dtype=np.int64)


def popularity_buckets(pairs: np.ndarray, n_items: int) -> np.ndarray:
    pop = np.bincount(pairs[:, 1], minlength=n_items)
    quantiles = np.quantile(pop[pop > 0], np.linspace(0, 1, 11))
    quantiles[-1] += 1
    return np.clip(np.digitize(pop, quantiles) - 1, 0, 9).astype(np.int64)


def audit_negative_sampler(
    pairs: np.ndarray, buckets: np.ndarray, negatives: np.ndarray
) -> dict:
    """Measure what the family-shared sampler actually does, without changing it."""
    anchors = pairs[:, 0]
    positives = pairs[:, 1]

    neg_buckets = buckets[negatives]
    anchor_buckets = buckets[anchors][:, None]
    positive_buckets = buckets[positives][:, None]

    successor_pairs = set(zip(pairs[:, 0].tolist(), pairs[:, 1].tolist()))
    flat_anchor = np.repeat(anchors, negatives.shape[1])
    false_negatives = sum(
        (int(a), int(n)) in successor_pairs
        for a, n in zip(flat_anchor.tolist(), negatives.reshape(-1).tolist())
    )
    ordered = np.sort(negatives, axis=1)

    return {
        "sampler": "anchor popularity decile, excluding anchor/target/observed successors",
        "rows": int(len(pairs)),
        "k_negatives": int(negatives.shape[1]),
        "bucket_match_rate_vs_anchor": float((neg_buckets == anchor_buckets).mean()),
        "bucket_match_rate_vs_positive": float((neg_buckets == positive_buckets).mean()),
        "bucket_histogram_of_negatives": np.bincount(
            neg_buckets.reshape(-1), minlength=10
        ).tolist(),
        "bucket_histogram_of_anchors": np.bincount(
            buckets[anchors], minlength=10
        ).tolist(),
        "duplicate_draws_within_row": int(
            (ordered[:, 1:] == ordered[:, :-1]).sum()
        ),
        "false_negatives_drawn": int(false_negatives),
        "negative_equals_anchor": int((negatives == anchors[:, None]).sum()),
        "negative_equals_positive": int((negatives == positives[:, None]).sum()),
        "distinct_negative_items": int(len(np.unique(negatives))),
    }


def main() -> None:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    sys.path.insert(0, str(A1_TREE))
    import train_rqvae as trainer  # noqa: E402
    from model import RQVAE  # noqa: E402
    from hbpe_a1 import (  # noqa: E402
        _compute_reference_gap,
        _precompute_matched_negatives,
        _soft_prefix_distribution,
    )

    pairs = load_train_pairs()
    pairs_sha = hashlib.sha256(pairs.tobytes()).hexdigest()
    buckets = popularity_buckets(pairs, N_ITEMS)
    negatives = _precompute_matched_negatives(pairs, N_ITEMS, seed=SEED, k_neg=K_NEG)
    audit = audit_negative_sampler(pairs, buckets, negatives)

    embeddings = trainer.load_embeddings(trainer.EMBEDDING_FILE)
    all_embeddings = torch.from_numpy(embeddings)
    train_frame = pd.read_parquet(TRAIN_FILE)
    train_ids = np.unique(train_frame["target"].to_numpy(dtype=np.int64))
    train_embeddings = all_embeddings[torch.from_numpy(train_ids)]

    checkpoint = RESAMPLE_EXPORT / "rqvae_best.pth"
    payload = torch.load(checkpoint, map_location=device)
    model = RQVAE(trainer._tokenizer_config(), in_dim=embeddings.shape[1]).to(device)
    model.load_state_dict(payload.get("state_dict", payload), strict=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    codebook = model.rq.vq_layers[0].get_code_embs().detach()
    codebook2 = model.rq.vq_layers[1].get_code_embs().detach()

    # --- H_ref: entropy of the hard L1 occupancy over the training catalogue ---
    histogram = np.zeros(int(codebook.shape[0]), dtype=np.float64)
    with torch.no_grad():
        for start in range(0, len(train_embeddings), 4096):
            z = model.encoder(train_embeddings[start : start + 4096].to(device))
            hard = squared_distances(z, codebook).argmin(dim=-1).cpu().numpy()
            histogram += np.bincount(hard, minlength=histogram.size)
    histogram /= histogram.sum()
    nonzero = histogram[histogram > 0]
    h_ref = float(-(nonzero * np.log(nonzero)).sum())

    # --- Teacher reference distance gaps and soft negative sharing ---
    with torch.no_grad():
        z_ref = model.encoder(train_embeddings[:8192].to(device))
        d_ref1 = _compute_reference_gap(z_ref, codebook)
        d_ref2 = _compute_reference_gap(z_ref - codebook[
            squared_distances(z_ref, codebook).argmin(dim=-1)
        ], codebook2)

        probe = pairs[:4096]
        negative_ids = negatives[: len(probe)]
        z_a = model.encoder(all_embeddings[torch.from_numpy(probe[:, 0])].to(device))
        z_n_flat = model.encoder(
            all_embeddings[torch.from_numpy(negative_ids.reshape(-1))].to(device)
        )
        z_n = z_n_flat.view(len(probe), negative_ids.shape[1], z_n_flat.shape[-1])

        def hard_l1(z: torch.Tensor) -> torch.Tensor:
            return squared_distances(z, codebook).argmin(dim=-1)

        teacher_neg = {}
        for tau in (1.0, 0.2):
            t1 = tau * max(d_ref1, EPS)
            t2 = tau * max(d_ref2, EPS)
            p1_a = _soft_prefix_distribution(z_a, codebook, t1)
            p1_n = _soft_prefix_distribution(z_n, codebook, t1)
            s1_neg = (p1_a.unsqueeze(1) * p1_n).sum(-1).mean()

            p2_a = _soft_prefix_distribution(z_a - codebook[hard_l1(z_a)], codebook2, t2)
            p2_n = _soft_prefix_distribution(z_n - codebook[hard_l1(z_n)], codebook2, t2)
            s2_neg = (p2_a.unsqueeze(1) * p2_n).sum(-1).mean()
            teacher_neg[f"tau_{tau:.1f}"] = {
                "T1": t1,
                "T2": t2,
                "B1_neg": float(s1_neg),
                "B2_neg": float(s2_neg),
                "B12_neg": float(s1_neg.item() * s2_neg.item()),
            }

    payload_out = {
        "purpose": (
            "frozen-reference constants for the HBPE R2 rounds; the canonical "
            "Resample export is a threshold measurer only, never a warm start"
        ),
        "teacher_checkpoint": str(checkpoint.relative_to(REPO_ROOT)),
        "teacher_checkpoint_sha256": sha256_file(checkpoint),
        "train_pairs_sha256": pairs_sha,
        "train_items": int(len(train_ids)),
        "seed": SEED,
        "eps": EPS,
        "H_REF_L1": h_ref,
        "H_REF_L1_distinct_codes": int((histogram > 0).sum()),
        "H_REF_L1_effective_codes_exp_H": float(np.exp(h_ref)),
        "H_REF_L1_max_possible": float(np.log(int(codebook.shape[0]))),
        "teacher_reference_distance_gaps": {"D_ref1": d_ref1, "D_ref2": d_ref2},
        "teacher_soft_negative_sharing_endpoints": teacher_neg,
        "negative_sampler_audit": audit,
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(payload_out, indent=2), encoding="utf-8")
    print(json.dumps(payload_out, indent=2))
    print(f"\nwritten: {OUT_PATH}")


if __name__ == "__main__":
    main()
