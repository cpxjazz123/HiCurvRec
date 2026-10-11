"""Why does the hyperbolic behaviour aggregation not beat the Euclidean one?

Diagnostic over the two already-trained checkpoints. No training, no gradient,
no Stage3: this only reads the frozen encoders and asks what geometry the
hyperbolic arm actually ended up with.

Three checks, as proposed:

1. Radius distribution of the ball image. Mean/std and P10/P50/P90/P99 of
   ||z||/R, where R = 1/sqrt(c). If the spread collapses, the radial coordinate
   carries nothing and the exponential volume of the ball is unusable.
2. Spearman between the Euclidean distance on the raw latent and the Poincare
   distance on the same encoder's ball image. Near 1 means the two metrics induce
   almost the same ordering.
3. Top-10 neighbour overlap between the two geometries on the SAME encoder, plus
   the overlap between the two ARMS' own neighbourhoods.

The per-arm comparison is the one that answers the question: it holds the
representation fixed and varies only the metric, so it cannot be confounded by
the two arms having trained apart. The cross-arm overlap is reported separately
because it measures something different.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path("/home/wlia0047/ar57/wenyu/GeneRec")
TREE = ROOT / "stage2_RQ-VAE/TIGER_FORK_RQ-VAE"
sys.path.insert(0, str(TREE))

import train_rqvae as T  # noqa: E402
from hyperbolic import ball_radius, encode_ball, pair_poincare_distance  # noqa: E402
from model import RQVAE  # noqa: E402

ARMS = {
    "nce_euclid_group": ROOT / "results/stage2_RQ-VAE/TIGER_FORK_RQ-VAE/nce_euclid_group/rqvae_best.pth",
    "nce_hyp_group": ROOT / "results/stage2_RQ-VAE/TIGER_FORK_RQ-VAE/nce_hyp_group/rqvae_best.pth",
}
CURVATURE = 1.0
CHUNK = 256
TOP_K = 10
PAIR_SAMPLE_PER_CHUNK = 2000
SEED = 42


def load_encoder(ckpt_path: Path, train_emb: torch.Tensor, device: torch.device):
    T.set_seed(T.SEED)
    model = RQVAE(T._tokenizer_config(), in_dim=train_emb.shape[1]).to(device)
    payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(payload["state_dict"], strict=True)
    model.eval()
    for layer in model.rq.vq_layers:
        layer._skip_ddp_reduce = True
    with torch.no_grad():
        latent = model.encoder(train_emb.to(device))
        normalizer = torch.as_tensor(
            T.resolve_normalizer(model, train_emb.to(device)),
            dtype=torch.float32,
            device=device,
        )
    return model, latent, normalizer


def spearman_from_pairs(a: np.ndarray, b: np.ndarray, max_pairs: int = 4_000_000) -> float:
    """Spearman via average ranks, exact enough and avoids a scipy import."""
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    ra = _average_rank(a)
    rb = _average_rank(b)
    ra = ra - ra.mean()
    rb = rb - rb.mean()
    denom = float(np.sqrt((ra * ra).sum() * (rb * rb).sum()))
    return float((ra * rb).sum() / denom) if denom > 0 else float("nan")


def _average_rank(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values), dtype=np.float64)
    ranks[order] = np.arange(len(values), dtype=np.float64)
    # average ties so that exact duplicates do not inflate the correlation
    sorted_values = values[order]
    start = 0
    for i in range(1, len(values) + 1):
        if i == len(values) or sorted_values[i] != sorted_values[start]:
            if i - start > 1:
                ranks[order[start:i]] = ranks[order[start:i]].mean()
            start = i
    return ranks


def analyse(name: str, ckpt: Path, embeddings: np.ndarray, train_ids: np.ndarray, device):
    all_emb = torch.from_numpy(embeddings)
    train_emb = all_emb[torch.from_numpy(train_ids)]
    model, latent, normalizer = load_encoder(ckpt, train_emb, device)
    n = latent.shape[0]
    radius = ball_radius(CURVATURE)
    with torch.no_grad():
        z = encode_ball(latent, CURVATURE, normalizer)
        z_norm = torch.linalg.vector_norm(z, dim=-1)
        radius_fraction = (z_norm / radius).cpu().numpy()
        raw_norm = torch.linalg.vector_norm(latent, dim=-1).cpu().numpy()
    quantiles = np.percentile(radius_fraction, [10, 50, 90, 99])
    report = {
        "n_items": int(n),
        "radius_fraction_mean": float(radius_fraction.mean()),
        "radius_fraction_std": float(radius_fraction.std()),
        "radius_fraction_percentiles": {
            "p10": float(quantiles[0]),
            "p50": float(quantiles[1]),
            "p90": float(quantiles[2]),
            "p99": float(quantiles[3]),
        },
        "saturated_fraction_above_0p99": float((radius_fraction > 0.99).mean()),
        "raw_latent_norm_mean": float(raw_norm.mean()),
        "raw_latent_norm_std": float(raw_norm.std()),
    }

    # ---- 2 and 3 on the SAME encoder: only the metric varies --------------
    rng = np.random.default_rng(SEED)
    pairs_e: list[np.ndarray] = []
    pairs_h: list[np.ndarray] = []
    overlap_sum = 0.0
    anchors_done = 0
    with torch.no_grad():
        for start in range(0, n, CHUNK):
            stop = min(start + CHUNK, n)
            a = latent[start:stop]
            de = torch.cdist(a, latent, p=2)
            dh = pair_poincare_distance(
                encode_ball(a, CURVATURE, normalizer), z, CURVATURE
            )
            # exclude self
            rows = torch.arange(stop - start, device=latent.device)
            de[rows, torch.arange(start, stop, device=latent.device)] = float("inf")
            dh[rows, torch.arange(start, stop, device=latent.device)] = float("inf")
            top_e = de.topk(TOP_K, dim=1, largest=False).indices
            top_h = dh.topk(TOP_K, dim=1, largest=False).indices
            hit = (top_e.unsqueeze(2) == top_h.unsqueeze(1)).any(dim=2).float().sum().item()
            overlap_sum += hit
            anchors_done += stop - start
            take = rng.integers(0, n, size=PAIR_SAMPLE_PER_CHUNK)
            j = rng.integers(0, stop - start, size=PAIR_SAMPLE_PER_CHUNK)
            pairs_e.append(de[j, take].cpu().numpy())
            pairs_h.append(dh[j, take].cpu().numpy())

    report["top10_overlap_same_encoder"] = overlap_sum / (anchors_done * TOP_K)
    report["spearman_euclid_vs_hyperbolic_same_encoder"] = spearman_from_pairs(
        np.concatenate(pairs_e), np.concatenate(pairs_h)
    )
    return report, latent.cpu().numpy()


def main() -> None:
    torch.use_deterministic_algorithms(True, warn_only=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    embeddings = np.asarray(
        np.load(ROOT / "stage1_GeneEmbedding/output/sentence_t5.npy"), dtype=np.float32
    )
    frame = pd.read_parquet(
        ROOT / "results/stage0_build_parquet/train.parquet", columns=["target"]
    )
    train_ids = np.unique(frame["target"].to_numpy(dtype=np.int64))

    out: dict = {"curvature": CURVATURE, "top_k": TOP_K, "arms": {}}
    latents: dict[str, np.ndarray] = {}
    for name, ckpt in ARMS.items():
        if not Path(ckpt).is_file():
            out["arms"][name] = {"missing": str(ckpt)}
            continue
        report, latent = analyse(name, Path(ckpt), embeddings, train_ids, device)
        out["arms"][name] = report
        latents[name] = latent
        print(f"[done] {name}", file=sys.stderr, flush=True)

    # cross-arm: did the two mechanisms learn the same item neighbourhoods?
    if len(latents) == 2:
        (ka, la), (kb, lb) = list(latents.items())
        a = torch.from_numpy(la).to(device)
        b = torch.from_numpy(lb).to(device)
        overlap = 0.0
        count = 0
        with torch.no_grad():
            for start in range(0, len(la), CHUNK):
                stop = min(start + CHUNK, len(la))
                da = torch.cdist(a[start:stop], a, p=2)
                db = torch.cdist(b[start:stop], b, p=2)
                rows = torch.arange(stop - start, device=a.device)
                da[rows, torch.arange(start, stop, device=a.device)] = float("inf")
                db[rows, torch.arange(start, stop, device=a.device)] = float("inf")
                ta = da.topk(TOP_K, dim=1, largest=False).indices
                tb = db.topk(TOP_K, dim=1, largest=False).indices
                overlap += (ta.unsqueeze(2) == tb.unsqueeze(1)).any(dim=2).float().sum().item()
                count += stop - start
        out["cross_arm_top10_overlap"] = overlap / (count * TOP_K)
        out["cross_arm_labels"] = [ka, kb]

    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()