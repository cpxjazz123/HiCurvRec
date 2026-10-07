"""Is the encoder's angular range too narrow, or is the metric failing to use it?

iter121 found that a quarter of the behaviour pairs stay active at every
checkpoint, with d_pos = 2.93 against d_neg = 3.03, a 4% separation, and that
neither distance is an outlier and the negatives are clean. The open question is
which of two explanations holds:

  A. the encoder learned item directions so close together that no metric can
     separate them, or
  B. the encoder does carry usable angular structure and the hyperbolic distance
     is not expressing it.

A cosine near 1 does not settle this on its own. A stable 0.994 against 0.988 is
a small gap that can still carry a lot of ranking information, so the question is
whether the angular distributions separate at all, not whether their absolute
value is high. Four measurements answer it:

  1. The cosine distribution for random, positive and negative pairs side by
     side. Separation between the three is the direct test.
  2. Rank correlation between the hyperbolic distance and the negated cosine. Near
     unity means the metric is re-expressing the angle, so the encoder's angular
     range is what limits the loss. Low means the distance is dominated by
     something else and the metric is the suspect.
  3. Rank correlation between the distance and the norm difference, which
     attributes the part of the distance that is not angular.
  4. The angular margin, cos(A,B+) - cos(A,X-), split by the persistent,
     rotating and never-active populations from iter121. If it collapses to zero
     exactly where the hyperbolic margin stays small, the encoder is the limit.

Nothing is trained.
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
from model.layers import _poincare_distance_tangent_pairs


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


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    """Rank correlation without a scipy dependency."""
    def ranks(v: np.ndarray) -> np.ndarray:
        order = np.argsort(v, kind="stable")
        r = np.empty(len(v), dtype=np.float64)
        r[order] = np.arange(len(v), dtype=np.float64)
        # average ties so a constant column does not fake a perfect correlation
        _, inverse, counts = np.unique(v, return_inverse=True, return_counts=True)
        if len(counts) > 1 and counts.max() > 1:
            sums = np.zeros(len(counts), dtype=np.float64)
            np.add.at(sums, inverse, r)
            r = (sums / counts)[inverse]
        return r

    rx, ry = ranks(x), ranks(y)
    if rx.std() == 0.0 or ry.std() == 0.0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def percentile_line(values: np.ndarray, label: str) -> str:
    q = np.percentile(values, [1, 5, 50, 95, 99])
    return (
        f"   {label:<22}p1={q[0]:.6f}  p5={q[1]:.6f}  p50={q[2]:.6f}  "
        f"p95={q[3]:.6f}  p99={q[4]:.6f}  mean={values.mean():.6f}"
    )


@torch.no_grad()
def encode_all(model: RQVAE, embeddings: torch.Tensor, chunk: int = 4096):
    parts = []
    for start in range(0, embeddings.shape[0], chunk):
        parts.append(model.encoder(embeddings[start : start + chunk]))
    return torch.cat(parts, dim=0)


def cosine_between(unit: torch.Tensor, left: torch.Tensor, right: torch.Tensor) -> np.ndarray:
    return (unit[left] * unit[right]).sum(-1).cpu().numpy().astype(np.float64)


@torch.no_grad()
def hyperbolic(left: torch.Tensor, right: torch.Tensor, curvature: float) -> np.ndarray:
    return (
        _poincare_distance_tangent_pairs(left, right, curvature)
        .cpu()
        .numpy()
        .astype(np.float64)
    )


def main() -> None:
    experiment.STAGE2_LOG_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    generator = np.random.default_rng(experiment.DECOMP_SEED)
    torch.manual_seed(experiment.DECOMP_SEED)

    embeddings = torch.from_numpy(
        np.asarray(np.load(experiment.EMBEDDING_FILE), dtype=np.float32)
    ).to(device)
    n_items = embeddings.shape[0]
    model = RQVAE(tokenizer_config(), in_dim=embeddings.shape[1]).to(device)
    model.load_state_dict(
        torch.load(
            experiment.PARENT_CKPT, map_location=device, weights_only=False
        )["state_dict"],
        strict=True,
    )
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    latent = encode_all(model, embeddings)
    norms = torch.linalg.vector_norm(latent, dim=-1).cpu().numpy().astype(
        np.float64
    )
    unit = latent / torch.linalg.vector_norm(
        latent, dim=-1, keepdim=True
    ).clamp_min(1e-12)
    curvature = float(experiment.LAYER_CURVATURES[0])

    lines: list[str] = []
    result: dict = {
        "parent_checkpoint": str(experiment.PARENT_CKPT),
        "curvature": curvature,
        "latent_norm_mean": float(norms.mean()),
        "latent_norm_std": float(norms.std()),
    }
    lines.append(
        f"raw latent: ||z|| mean={norms.mean():.6f} std={norms.std():.6f}  "
        f"min={norms.min():.6f} max={norms.max():.6f}"
    )

    # ---- 1. cosine distributions, with random pairs as the reference
    size = experiment.DECOMP_PAIRS
    train_frame = pd.read_parquet(experiment.TRAIN_FILE)
    source_ids, successor_ids = transition_pairs(train_frame)
    pick = generator.choice(len(source_ids), size=size, replace=False)
    source_ids = np.ascontiguousarray(source_ids[pick])
    successor_ids = np.ascontiguousarray(successor_ids[pick])
    perm = generator.permutation(size)
    negative_ids = source_ids[perm]
    random_ids = generator.integers(0, n_items, size=size)

    idx_s = torch.from_numpy(source_ids).to(device)
    idx_p = torch.from_numpy(successor_ids).to(device)
    idx_n = torch.from_numpy(negative_ids).to(device)
    idx_r = torch.from_numpy(random_ids).to(device)

    cos_pos = cosine_between(unit, idx_s, idx_p)
    cos_neg = cosine_between(unit, idx_s, idx_n)
    cos_rand = cosine_between(unit, idx_s, idx_r)

    lines.append("")
    lines.append("1. angular distributions on the raw latent")
    lines.append(percentile_line(cos_rand, "random pairs"))
    lines.append(percentile_line(cos_pos, "positive pairs"))
    lines.append(percentile_line(cos_neg, "negative pairs (batch shuffle)"))
    lines.append(
        f"   separation: p50 positive - p50 random = "
        f"{np.median(cos_pos) - np.median(cos_rand):+.6f}   "
        f"p50 positive - p50 negative = "
        f"{np.median(cos_pos) - np.median(cos_neg):+.6f}"
    )
    result["cosine"] = {
        "random": np.percentile(cos_rand, [1, 5, 50, 95, 99]).tolist(),
        "positive": np.percentile(cos_pos, [1, 5, 50, 95, 99]).tolist(),
        "negative": np.percentile(cos_neg, [1, 5, 50, 95, 99]).tolist(),
        "positive_minus_random_p50": float(
            np.median(cos_pos) - np.median(cos_rand)
        ),
        "positive_minus_negative_p50": float(
            np.median(cos_pos) - np.median(cos_neg)
        ),
    }

    # ---- 2 and 3. what the hyperbolic distance is actually made of
    d_pos = hyperbolic(latent[idx_s], latent[idx_p], curvature)
    d_neg = hyperbolic(latent[idx_s], latent[idx_n], curvature)
    d_rand = hyperbolic(latent[idx_s], latent[idx_r], curvature)
    norm_gap_pos = np.abs(norms[source_ids] - norms[successor_ids])
    norm_gap_neg = np.abs(norms[source_ids] - norms[negative_ids])
    norm_gap_rand = np.abs(norms[source_ids] - norms[random_ids])

    def attribute(label: str, d: np.ndarray, cos: np.ndarray, gap: np.ndarray):
        return {
            "spearman_distance_vs_neg_cosine": spearman(d, -cos),
            "spearman_distance_vs_norm_gap": spearman(d, gap),
        }

    result["attribution"] = {
        "positive": attribute("positive", d_pos, cos_pos, norm_gap_pos),
        "negative": attribute("negative", d_neg, cos_neg, norm_gap_neg),
        "random": attribute("random", d_rand, cos_rand, norm_gap_rand),
    }
    lines.append("")
    lines.append("2/3. what the distance is made of (Spearman)")
    lines.append(
        f"   {'pair type':<12}{'rho(d_H, -cos)':>18}{'rho(d_H, |d||z||)':>22}"
    )
    for label in ("random", "positive", "negative"):
        entry = result["attribution"][label]
        lines.append(
            f"   {label:<12}"
            f"{entry['spearman_distance_vs_neg_cosine']:>18.4f}"
            f"{entry['spearman_distance_vs_norm_gap']:>22.4f}"
        )

    # ---- 4. angular margin by population
    trajectory_dir = experiment.REPO_ROOT / "results/stage2_RQ-VAE/curvature_RQ-VAE_iter120/trajectory"
    persistent, rotating, never, n_checkpoints = populations(
        source_ids, successor_ids, negative_ids, embeddings, trajectory_dir,
        experiment.PARENT_CKPT, device,
    )
    result["checkpoints_used"] = n_checkpoints
    cos_margin = cos_pos - cos_neg
    hyp_margin = d_neg - d_pos
    result["by_population"] = {}
    lines.append("")
    lines.append("4. angular vs hyperbolic margin, by population")
    lines.append(
        f"   {'population':<12}{'n':>7}{'cos margin p50':>17}"
        f"{'hyp margin p50':>17}{'cos pos p50':>13}"
    )
    for label, mask in (
        ("persistent", persistent),
        ("rotating", rotating),
        ("never", never),
    ):
        if mask.sum() == 0:
            lines.append(f"   {label:<12}{int(mask.sum()):>7}  (empty)")
            continue
        entry = {
            "pairs": int(mask.sum()),
            "cosine_margin_p50": float(np.median(cos_margin[mask])),
            "hyperbolic_margin_p50": float(np.median(hyp_margin[mask])),
            "cosine_positive_p50": float(np.median(cos_pos[mask])),
            "cosine_negative_p50": float(np.median(cos_neg[mask])),
        }
        result["by_population"][label] = entry
        lines.append(
            f"   {label:<12}{entry['pairs']:>7}"
            f"{entry['cosine_margin_p50']:>17.6f}"
            f"{entry['hyperbolic_margin_p50']:>17.6f}"
            f"{entry['cosine_positive_p50']:>13.6f}"
        )

    # Does the angular margin order the populations the same way the hyperbolic
    # margin does? If it does, the metric is not the limit.
    lines.append("")
    lines.append("verdict")
    rng = result["attribution"]["random"]["spearman_distance_vs_neg_cosine"]
    pos_gap = result["cosine"]["positive_minus_negative_p50"]
    ang_orders = (
        result["by_population"]["persistent"]["cosine_margin_p50"]
        < result["by_population"]["rotating"]["cosine_margin_p50"]
        < result["by_population"]["never"]["cosine_margin_p50"]
    )
    metric_follows_angle = abs(rng) > 0.8
    if pos_gap > 0.0 and ang_orders and metric_follows_angle:
        verdict = (
            "the encoder does carry angular structure, the metric tracks it, and the "
            "angular margin collapses exactly where the hyperbolic one does; the "
            "limit is how much angular range the encoder learned"
        )
    elif pos_gap > 0.0 and not metric_follows_angle:
        verdict = (
            "angular structure exists but the hyperbolic distance does not track it; "
            "the metric definition is the suspect, not the encoder"
        )
    else:
        verdict = (
            "the angular distributions do not separate; the encoder learned "
            "directions too close to rank on"
        )
    result["verdict"] = verdict
    lines.append(
        f"   positive/negative cosine separation p50 = {pos_gap:+.6f}"
    )
    lines.append(f"   rho(d_H, -cos) on random pairs = {rng:+.4f}")
    lines.append(f"   angular margin orders the populations: {ang_orders}")
    lines.append(f"   {verdict}")

    report = "\n".join(lines)
    print(report, flush=True)
    experiment.DECOMP_JSON.write_text(
        json.dumps(result, indent=2, default=float) + "\n", encoding="utf-8"
    )
    experiment.DECOMP_LOG.write_text(report + "\n", encoding="utf-8")


def populations(
    source_ids: np.ndarray,
    successor_ids: np.ndarray,
    negative_ids: np.ndarray,
    embeddings: torch.Tensor,
    trajectory_dir: Path,
    parent_ckpt: Path,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Persistent / rotating / never, defined exactly as iter121 did.

    The split is "active at every checkpoint" rather than "active now", so it has
    to be recomputed from the checkpoints. Falling back to a single-point margin
    would silently change what persistent means, since a quarter of pairs are
    active at 72k without having been active throughout.
    """
    idx_s = torch.from_numpy(source_ids).to(device)
    idx_p = torch.from_numpy(successor_ids).to(device)
    idx_n = torch.from_numpy(negative_ids).to(device)
    paths = sorted(
        trajectory_dir.glob("step_*.pth"),
        key=lambda q: int(q.stem.split("_")[1]),
    ) + [parent_ckpt]
    columns = []
    for path in paths:
        other = RQVAE(tokenizer_config(), in_dim=embeddings.shape[1]).to(device)
        other.load_state_dict(
            torch.load(path, map_location=device, weights_only=False)["state_dict"],
            strict=True,
        )
        other.eval()
        for parameter in other.parameters():
            parameter.requires_grad_(False)
        with torch.no_grad():
            a = encode_all(other, embeddings)[idx_s]
            b = encode_all(other, embeddings)[idx_p]
            c = encode_all(other, embeddings)[idx_n]
            curvature_value = float(experiment.LAYER_CURVATURES[0])
            positive = hyperbolic(a, b, curvature_value)
            negative = hyperbolic(a, c, curvature_value)
        columns.append((negative - positive) < experiment.BEHAVIOUR_MARGIN)
        del other
        torch.cuda.empty_cache()
    stack = np.stack(columns, axis=1)
    persistent = stack.all(axis=1)
    never = (~stack).all(axis=1)
    rotating = ~persistent & ~never
    return persistent, rotating, never, len(paths)


if __name__ == "__main__":
    main()