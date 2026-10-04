"""Symmetric ranking-curvature sweep: c_pos = c_neg = c_b.

The asymmetric run (c_pos=0.9, c_neg=1.0) weakened the ranking hinge and
dropped test_R@10 to 0.0359, 24 s.e. below the parent. That raises a question
the asymmetric form cannot answer cleanly: is the parent's ranking term too
strong or too weak, independent of any asymmetry?

Scaling both endpoints together is the clean form of that question. Writing
d(c) = f(c) * d(1) for the monotone curvature factor f gives

    hinge = relu( d_pos + margin - d_neg ) = relu( margin - f(c) * delta )

with delta = d_neg - d_pos. On the parent checkpoint delta is positive on
average (3.095 against 2.598), so raising c_b raises f, raises delta and drives
the hinge toward zero -- the term gets weaker. Lowering c_b does the opposite.
The asymmetric run moved along that same axis in the weakening direction, which
is why it failed, so the direction of the effect is the thing to establish
before spending a Stage2 budget.

Everything is measured on the parent's trained checkpoint, and the latent norm
is asserted first: an earlier check ran against a randomly initialised encoder
and inverted every conclusion it drew.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

SOURCE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOURCE_DIR))

import curvature_config as experiment
import train_rqvae as training
from model import RQVAE
from model.layers import _poincare_distance_tangent_pairs

PARENT_CKPT = Path(
    "/home/wlia0047/ar57/wenyu/GeneRec/results/stage2_RQ-VAE/"
    "curvature_RQ-VAE/out/rqvae/instruments/rqvae_best.pth"
)
C_B_SWEEP = (0.1, 0.2, 0.3, 0.5, 0.7, 0.85, 1.0, 1.25, 1.5, 2.0, 4.0)
TAKE = 8192


def main() -> None:
    training.configure_run(__file__)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(training.SEED)
    margin = training.BEHAVIOUR_MARGIN

    embeddings = np.asarray(np.load(experiment.EMBEDDING_FILE), dtype=np.float32)
    model = RQVAE(
        training._tokenizer_config(), in_dim=int(embeddings.shape[1])
    ).to(device)
    if not PARENT_CKPT.is_file():
        raise SystemExit(f"missing parent checkpoint {PARENT_CKPT}")
    payload = torch.load(PARENT_CKPT, map_location=device, weights_only=False)
    model.load_state_dict(payload["state_dict"], strict=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    print(f"parent @ step {payload['global_step']}, margin {margin}")

    frame = pd.read_parquet(experiment.TRAIN_FILE)
    source_ids, successor_ids = training._transition_pairs(frame)
    take = min(TAKE, len(source_ids))
    with torch.no_grad():
        sources = torch.from_numpy(
            embeddings[torch.from_numpy(source_ids[:take])]
        ).to(device)
        successors = torch.from_numpy(
            embeddings[torch.from_numpy(successor_ids[:take])]
        ).to(device)
        z_source = model.encoder(sources)
        z_successor = model.encoder(successors)
        negatives = z_source[
            torch.randperm(z_source.shape[0], device=device)
        ]
        norm_p50 = float(torch.linalg.vector_norm(z_source, dim=-1).median())
    print(
        f"trained latent norm p50 {norm_p50:.5f} (a random encoder would be "
        "~0.01 and every conclusion below would invert)"
    )
    if norm_p50 < 0.5:
        raise SystemExit(
            f"latent norm {norm_p50:.5f} is far below the trained parent; the "
            "checkpoint weights were probably not applied"
        )

    print()
    header = (
        f"{'c_b':>6} {'d_pos':>10} {'d_neg':>10} {'delta':>10} "
        f"{'hinge_mean':>11} {'active%':>8} {'vs parent':>10}"
    )
    print(header)
    print("-" * len(header))
    rows = []
    for c_b in C_B_SWEEP:
        with torch.no_grad():
            d_pos = _poincare_distance_tangent_pairs(z_source, z_successor, c_b)
            d_neg = _poincare_distance_tangent_pairs(z_source, negatives, c_b)
            slack = d_pos + margin - d_neg
            hinge = torch.relu(slack)
            active = 100.0 * float((slack > 0).float().mean())
        rows.append(
            {
                "c_b": c_b,
                "d_pos": float(d_pos.mean()),
                "d_neg": float(d_neg.mean()),
                "delta": float((d_neg - d_pos).mean()),
                "hinge": float(hinge.mean()),
                "active": active,
            }
        )
    parent = next(r for r in rows if r["c_b"] == 1.0)
    for r in rows:
        print(
            f"{r['c_b']:>6.2f} {r['d_pos']:>10.4f} {r['d_neg']:>10.4f} "
            f"{r['delta']:>10.4f} {r['hinge']:>11.6f} {r['active']:>8.2f} "
            f"{100 * (r['hinge'] / parent['hinge'] - 1):>+9.1f}%"
        )

    print()
    print(
        f"parent reference: d_pos {parent['d_pos']:.4f}, d_neg "
        f"{parent['d_neg']:.4f}, delta {parent['delta']:.4f}, hinge "
        f"{parent['hinge']:.6f}, active {parent['active']:.2f}%"
    )
    stronger = [r for r in rows if r["hinge"] > parent["hinge"]]
    weaker = [r for r in rows if r["hinge"] < parent["hinge"]]
    print(
        "stronger than parent: "
        + (", ".join(f"c_b={r['c_b']}" for r in stronger) or "none")
    )
    print(
        "weaker than parent:   "
        + (", ".join(f"c_b={r['c_b']}" for r in weaker) or "none")
    )

    alive = [r for r in rows if r["active"] >= 20.0]
    if not alive:
        print(
            "\nNo c_b keeps the hinge above 20% active, so the ranking term "
            "cannot be retuned in either direction without killing it."
        )
        return
    strongest = max(alive, key=lambda r: r["hinge"])
    print(
        f"\nstrongest alive option: c_b={strongest['c_b']} with hinge "
        f"{strongest['hinge']:.6f} "
        f"({100 * (strongest['hinge'] / parent['hinge'] - 1):+.1f}% vs parent) "
        f"at {strongest['active']:.2f}% active"
    )
    band = [
        r for r in alive
        if r["hinge"] <= 3.0 * parent["hinge"] and r["c_b"] != 1.0
    ]
    if band:
        pick = max(band, key=lambda r: r["hinge"])
        print(
            f"recommended c_b (stronger than parent, hinge within 3x, hinge "
            f"alive): {pick['c_b']} -> hinge {pick['hinge']:.6f}, "
            f"{pick['active']:.2f}% active"
        )
    else:
        print(
            "no c_b both strengthens the hinge and keeps it within 3x of the "
            "parent; the ranking term is already near its usable ceiling at "
            "c_b=1.0"
        )


if __name__ == "__main__":
    main()
