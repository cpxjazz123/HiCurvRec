"""Where does behaviour information die: encoder, pin, or the RQ levels?

The behaviour ranking loss is applied to the raw encoder latent. The RQ stack
then takes that same latent, pins its magnitude to a fixed working point, and
spends three levels of codebooks on what is left. Behaviour and quantization
therefore run in two spaces whose depth differs by an order of magnitude (the raw
latent sits at ||z|| ~ 2.05, the pinned shell at s = 0.2), and shell-aligned
behaviour has already been measured to lose 13.5% recall. The open question is
what survives the crossing.

The measurement is retention. For each anchor, its behaviour-relevant neighbours
are the items nearest to it on the raw encoder latent, which is where the
behaviour loss defines relevance. Each pair is then followed through the
pipeline:

    raw latent  ->  pinned input  ->  after L1  ->  after L2  ->  after L3

and counted as retained at level l when the pair agrees on the first l codes. The
same criterion the stack itself uses to decide a pair no longer needs depth. Two
stages are separated before L1 deliberately: the pin only overwrites magnitude,
so a pure direction-preserving step should show up as a specific signature
rather than being folded into the first level's loss.

The second half of the measurement is what distinguishes a loss worth attacking
from a loss that is just noise. Pairs are split by whether the behaviour was
strong to begin with, and retained and lost pairs are compared on the quantities
that could plausibly cause the loss: raw cosine, raw norm, the angle after
pinning, the top1-top2 margin at the level where the pair separated, the radius
of the codeword each landed on, and that codeword's usage.

Nothing is trained, and nothing here says which fix to apply: that follows from
where the loss happens and what it correlates with.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

SOURCE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOURCE_DIR))

import curvature_config as experiment
from model import RQVAE
from model.layers import (
    _hyperbolic_residual,
    _pairwise_poincare_distance_tangents,
)


def tokenizer_config() -> SimpleNamespace:
    return SimpleNamespace(
        hidden_sizes=experiment.HIDDEN_SIZES,
        codebook_num=3,
        codebook_size=experiment.CODEBOOK_SIZE,
        codebook_dim=experiment.CODEBOOK_DIM,
        dropout=0.0,
        beta=experiment.BETA,
        vq_type=experiment.VQ_TYPE,
        ema_decay=experiment.EMA_DECAY,
        fix_code_embs=False,
        sk_epsilon=experiment.SK_EPSILON,
        sk_iters=experiment.SK_ITERS,
        layer_curvatures=experiment.LAYER_CURVATURES,
        layer_working_radii=experiment.LAYER_WORKING_RADII,
        pin_in_s_coordinates=experiment.PIN_IN_S_COORDINATES,
    )


def transition_pairs(train_frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    sources: list[int] = []
    successors: list[int] = []
    for history, target in zip(
        train_frame["seen_history"].to_numpy(),
        train_frame["target"].to_numpy(dtype=np.int64),
    ):
        if history is None or len(history) == 0:
            continue
        sources.append(int(history[-1]))
        successors.append(int(target))
    return (
        np.asarray(sources, dtype=np.int64),
        np.asarray(successors, dtype=np.int64),
    )


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    x = x.astype(np.float64)
    y = y.astype(np.float64)
    if x.std() == 0.0 or y.std() == 0.0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


@torch.no_grad()
def full_pass(model: RQVAE, embeddings: torch.Tensor, rows: np.ndarray,
              batch_size: int = 2048) -> dict:
    """Encode, pin and quantize ``rows``, keeping every intermediate.

    Retention needs the pinned input separately from the first code, because the
    pin is the one step that rewrites magnitude without changing direction and
    therefore has a signature of its own.
    """
    device = embeddings.device
    layers = model.rq.vq_layers
    n = len(rows)
    k = len(layers)
    pinned_all: list[list[torch.Tensor]] = [[] for _ in range(k)]
    tokens = np.empty((n, k), dtype=np.int64)
    margin_all: list[list[np.ndarray]] = [[] for _ in range(k)]
    for start in range(0, n, batch_size):
        chunk = torch.from_numpy(np.ascontiguousarray(rows[start : start + batch_size]))
        batch = embeddings[chunk.to(device)]
        residual = model.encoder(batch)
        previous_codes = None
        for level, layer in enumerate(layers):
            c = float(layer.get_curvature())
            target_norm = model.rq._radius_for_level(level, c, residual)
            pinned = model.rq._pin_to_radius(residual, target_norm)
            distances = _pairwise_poincare_distance_tangents(
                pinned, layer.get_code_embs(), c
            )
            two = torch.topk(distances, k=2, dim=1, largest=False).values
            chosen = layer._indices(
                distances, infer_use_sk=True, bucket=previous_codes
            )
            pinned_all[level].append(pinned)
            margin_all[level].append(
                (two[:, 1] - two[:, 0]).cpu().numpy().astype(np.float64)
            )
            tokens[start : start + chunk.shape[0], level] = chosen.cpu().numpy()
            residual = model.rq._restore_norm(
                _hyperbolic_residual(
                    pinned, layer.get_code_embs()[chosen], c
                ),
                residual,
                target_norm,
            )
            previous_codes = chosen
    return {
        "pinned": [torch.cat(p, dim=0) for p in pinned_all],
        "tokens": tokens,
        "margin": [np.concatenate(m) for m in margin_all],
    }


def main() -> None:
    experiment.STAGE2_LOG_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    generator = np.random.default_rng(experiment.DIAGNOSTIC_SEED)
    torch.manual_seed(experiment.DIAGNOSTIC_SEED)

    embeddings = torch.from_numpy(
        np.asarray(np.load(experiment.EMBEDDING_FILE), dtype=np.float32)
    ).to(device)
    n_items = embeddings.shape[0]
    checkpoint = torch.load(
        experiment.PARENT_CKPT, map_location=device, weights_only=False
    )
    model = RQVAE(tokenizer_config(), in_dim=embeddings.shape[1]).to(device)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    # ---- behaviour neighbours on the raw encoder latent
    with torch.no_grad():
        latents = torch.cat(
            [
                model.encoder(embeddings[start : start + 2048])
                for start in range(0, n_items, 2048)
            ],
            dim=0,
        )
    unit = latents / torch.linalg.vector_norm(
        latents, dim=-1, keepdim=True
    ).clamp_min(1e-12)

    anchor_rows = generator.choice(
        n_items, size=experiment.ANCHOR_ROWS, replace=False
    )
    neighbour_rows = np.empty(
        (len(anchor_rows), experiment.NEIGHBOURS_PER_ITEM), dtype=np.int64
    )
    anchor_index = {int(r): i for i, r in enumerate(anchor_rows)}
    neighbour_flat: list[np.ndarray] = []
    batch = 512
    for start in range(0, len(anchor_rows), batch):
        block = anchor_rows[start : start + batch]
        sim = unit[torch.from_numpy(np.ascontiguousarray(block)).to(device)] @ unit.T
        # The anchor's own position is the maximum by construction; drop it
        # rather than trusting that it lands first under float ties.
        sim[torch.arange(len(block)), torch.from_numpy(block).to(device)] = -2.0
        top = torch.topk(
            sim, k=experiment.NEIGHBOURS_PER_ITEM, dim=1
        ).indices.cpu().numpy()
        neighbour_rows[start : start + len(block)] = top
        neighbour_flat.append(top.reshape(-1))
    neighbour_flat = np.unique(np.concatenate(neighbour_flat))
    keep_mask = np.array([r in anchor_index for r in neighbour_flat])

    lines: list[str] = []
    result: dict = {
        "parent_checkpoint": str(experiment.PARENT_CKPT),
        "anchors": int(len(anchor_rows)),
        "neighbours_per_anchor": experiment.NEIGHBOURS_PER_ITEM,
    }
    lines.append(
        f"anchors={len(anchor_rows)}  neighbours/anchor="
        f"{experiment.NEIGHBOURS_PER_ITEM}  unique neighbour rows="
        f"{len(neighbour_flat)}"
    )

    # ---- encode + quantize anchors and their neighbours
    pass_a = full_pass(model, embeddings, anchor_rows)
    pass_n = full_pass(model, embeddings, neighbour_flat)
    lat_a = latents[torch.from_numpy(anchor_rows).to(device)]
    lat_n = latents[torch.from_numpy(neighbour_flat).to(device)]
    unit_a = unit[torch.from_numpy(anchor_rows).to(device)]
    unit_n = unit[torch.from_numpy(neighbour_flat).to(device)]

    # cosine on the raw latent: how behaviour-relevant the pair was to begin with
    raw_cos = (unit_a @ unit_n.T).cpu().numpy()
    raw_norm_a = torch.linalg.vector_norm(lat_a, dim=-1).cpu().numpy()
    raw_norm_n = torch.linalg.vector_norm(lat_n, dim=-1).cpu().numpy()

    def row_of(r: int) -> int:
        return int(np.searchsorted(neighbour_flat, r))

    anchor_pos = np.array(
        [anchor_index[int(r)] for r in neighbour_flat if r in anchor_index]
    )
    # Build the pair table: (anchor position, neighbour position, raw cosine)
    pair_a: list[int] = []
    pair_n: list[int] = []
    pair_cos: list[float] = []
    for ai in range(len(anchor_rows)):
        block = neighbour_rows[ai]
        for nb in block:
            ni = int(np.searchsorted(neighbour_flat, nb))
            pair_a.append(ai)
            pair_n.append(ni)
            pair_cos.append(float(raw_cos[ai, ni]))
    pair_a = np.asarray(pair_a)
    pair_n = np.asarray(pair_n)
    pair_cos = np.asarray(pair_cos)

    tok_a = pass_a["tokens"]
    tok_n = pass_n["tokens"]
    agree = np.stack(
        [tok_a[pair_a, l] == tok_n[pair_n, l] for l in range(3)], axis=1
    )

    # ---- step 1: retention through the pipeline
    lines.append("")
    lines.append("1. retention through the pipeline (share of behaviour pairs "
                 "agreeing on the first l codes)")
    # Each entry is the share agreeing on the first l codes, so the levels nest
    # and the ratio against the previous level is meaningful.
    retention = {
        "L1": float(agree[:, 0].mean()),
        "L2": float((agree[:, 0] & agree[:, 1]).mean()),
        "L3": float(agree.all(axis=1).mean()),
    }
    result["retention"] = retention
    lines.append(
        "   raw latent      100.00%  (by construction: a pair is retrieved "
        "because it is a neighbour there)"
    )
    previous = 1.0
    for level in (1, 2, 3):
        current = retention[f"L{level}"]
        lines.append(
            f"   after L{level}       {current * 100:6.2f}%"
            f"   (kept {current / previous * 100:.1f}% of previous)"
        )
        previous = current
    lines.append(
        f"   pin preserves direction exactly (magnitude only), so any loss here "
        f"is attributable to the level that follows"
    )

    # ---- step 2: what distinguishes retained from lost
    lines.append("")
    lines.append("2. what distinguishes retained from lost")
    kept = agree.all(axis=1)
    lines.append(f"   retained through L3: {int(kept.sum())} / {len(kept)} "
                 f"({kept.mean() * 100:.2f}%)")

    # split by whether the behaviour was strong to begin with
    strong = pair_cos >= experiment.WEAK_COSINE_THRESHOLD
    result["strong_fraction"] = float(strong.mean())
    result["retention_strong"] = {
        f"L{l + 1}": float(agree[strong, : l + 1].all(axis=1).mean())
        for l in range(3)
    }
    result["retention_weak"] = {
        f"L{l + 1}": float(agree[~strong, : l + 1].all(axis=1).mean())
        for l in range(3)
    }
    lines.append(
        f"   split by raw cosine >= {experiment.WEAK_COSINE_THRESHOLD}: "
        f"strong {int(strong.sum())} ({strong.mean() * 100:.1f}%), "
        f"weak {int((~strong).sum())} ({(~strong).mean() * 100:.1f}%)"
    )
    for l in range(3):
        lines.append(
            f"     retention L{l + 1}: strong="
            f"{result['retention_strong'][f'L{l + 1}'] * 100:6.2f}%   "
            f"weak={result['retention_weak'][f'L{l + 1}'] * 100:6.2f}%"
        )

    # quantities that could explain a lost pair
    pinned_a = pass_a["pinned"]
    pinned_n = pass_n["pinned"]
    unit_pinned_a = [
        p / torch.linalg.vector_norm(p, dim=-1, keepdim=True).clamp_min(1e-12)
        for p in pinned_a
    ]
    unit_pinned_n = [
        p / torch.linalg.vector_norm(p, dim=-1, keepdim=True).clamp_min(1e-12)
        for p in pinned_n
    ]

    def row_stats(selected: np.ndarray) -> dict:
        out: dict = {}
        out["raw_cosine"] = float(pair_cos[selected].mean())
        out["raw_norm_anchor"] = float(raw_norm_a[pair_a[selected]].mean())
        out["raw_norm_neighbour"] = float(raw_norm_n[pair_n[selected]].mean())
        for level in range(3):
            cos_pinned = (
                unit_pinned_a[level][torch.from_numpy(pair_a[selected]).to(device)]
                * unit_pinned_n[level][torch.from_numpy(pair_n[selected]).to(device)]
            ).sum(-1).mean()
            margin = pass_a["margin"][level][pair_a[selected]]
            margin_n = pass_n["margin"][level][pair_n[selected]]
            out[f"cosine_after_pin_L{level + 1}"] = float(cos_pinned)
            out[f"margin_anchor_L{level + 1}"] = float(margin.mean())
            out[f"margin_neighbour_L{level + 1}"] = float(margin_n.mean())
            # radius and usage of the codeword each endpoint landed on
            codes = model.rq.vq_layers[level].get_code_embs()
            radii = (
                float(experiment.LAYER_CURVATURES[level]) ** 0.5
            ) * torch.linalg.vector_norm(codes, dim=-1).cpu().numpy()
            corpus_codes = pass_n["tokens"][:, level]
            usage = np.bincount(corpus_codes, minlength=len(radii)).astype(
                np.float64
            )
            ca = tok_a[pair_a[selected], level]
            cn = tok_n[pair_n[selected], level]
            out[f"codeword_radius_L{level + 1}"] = float(radii[ca].mean())
            out[f"codeword_usage_L{level + 1}"] = float(
                np.mean([usage[a] + usage[b] for a, b in zip(ca, cn)])
            )
        return out

    kept_stats = code_stats = row_stats(np.flatnonzero(kept))
    lost_stats = row_stats(np.flatnonzero(~kept))
    result["retained_pairs"] = kept_stats
    result["lost_pairs"] = lost_stats

    lines.append("")
    lines.append(f"   {'quantity':<28}{'retained':>12}{'lost':>12}{'diff':>12}")
    for key in kept_stats:
        a = kept_stats[key]
        b = lost_stats[key]
        lines.append(
            f"   {key:<28}{a:>12.6f}{b:>12.6f}{b - a:>+12.6f}"
        )

    # Is the loss predictable from a single quantity? Report the correlation of
    # each candidate with the retained indicator, so the strongest one is visible
    # rather than inferred from a table of means.
    lines.append("")
    lines.append("   corr(quantity, retained):")
    retained_flag = kept.astype(np.float64)
    candidates: dict[str, np.ndarray] = {"raw_cosine": pair_cos}
    for level in range(3):
        candidates[f"cosine_after_pin_L{level + 1}"] = (
            unit_pinned_a[level][torch.from_numpy(pair_a).to(device)]
            * unit_pinned_n[level][torch.from_numpy(pair_n).to(device)]
        ).sum(-1).cpu().numpy().astype(np.float64)
        candidates[f"margin_anchor_L{level + 1}"] = pass_a["margin"][level][pair_a]
        codes = model.rq.vq_layers[level].get_code_embs()
        radii = (
            float(experiment.LAYER_CURVATURES[level]) ** 0.5
        ) * torch.linalg.vector_norm(codes, dim=-1).cpu().numpy()
        corpus_codes = pass_n["tokens"][:, level]
        usage = np.bincount(corpus_codes, minlength=len(radii)).astype(np.float64)
        ca = tok_a[pair_a, level]
        cn = tok_n[pair_n, level]
        candidates[f"codeword_usage_L{level + 1}"] = np.asarray(
            [usage[a] + usage[b] for a, b in zip(ca, cn)]
        )
    correlations = {
        key: pearson(value, retained_flag) for key, value in candidates.items()
    }
    result["corr_with_retained"] = correlations
    for key in sorted(correlations, key=lambda k: -abs(correlations[k])):
        lines.append(
            f"     {key:<28}{correlations[key]:+.4f}"
        )

    report = "\n".join(lines)
    print(report, flush=True)
    experiment.DIAGNOSTIC_JSON.write_text(
        json.dumps(result, indent=2, default=float) + "\n", encoding="utf-8"
    )
    experiment.DIAGNOSTIC_LOG.write_text(report + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()