"""Why does L2 separate behaviour pairs more than random ones?

iter114 found the sharpest asymmetry in the stack: among pairs that already share
their L1 code, the ones that are a real transition go on to share their L2 code
2.50% of the time while random pairs sharing the same L1 code do so 3.05%. A level
that actively pushes behaviour pairs apart is the opposite of what a residual
quantizer is for, and L1 shows no such effect (15.2x enrichment over random).

The suspect is the balancing rule. L2 is assigned inside each L1 bucket, and
Sinkhorn forces an even split of codes across whatever pool it is handed. If the
pool is the whole bucket rather than behaviour-related members of it, then
enforcing even usage inside a bucket will actively push apart items that were
close, which is exactly the observed sign. That is a hypothesis about the
assignment, so it is checked against the assignment rather than assumed.

Three things are measured:

  1. Behaviour density inside an L1 bucket. For each real transition whose two
     endpoints share an L1 code, how many of the bucket's other members are
     also behaviour-related to the same anchor? If behaviour pairs concentrate,
     even balancing fights them; if they are spread evenly, it cannot.
  2. Whether L2's usage inside a bucket is even, and how that evenness compares
     with the behaviour density in the same bucket. This is the direct test of
     "the rule fights behaviour": high usage evenness with non-uniform behaviour
     density is the mechanism.
  3. Whether the pairs L2 separates share a geometry. Behaviour pairs it splits
     are compared with behaviour pairs it keeps on the L2 margin, the angle
     between their L2 residuals, and the codeword radius they land on. If the
     split pairs sit near a codeword boundary, L2 is resolving a real ambiguity;
     if they do not, the split is an artefact of the balancing constraint.

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


@torch.no_grad()
def stack_pass(model: RQVAE, embeddings: torch.Tensor, batch_size: int = 4096) -> dict:
    """Codes and L2-level geometry for the whole catalogue."""
    device = embeddings.device
    layers = model.rq.vq_layers
    n = embeddings.shape[0]
    codes = np.empty((n, len(layers)), dtype=np.int64)
    l2_margin = np.empty(n, dtype=np.float64)
    l2_residual = np.empty((n, experiment.CODEBOOK_DIM), dtype=np.float32)
    l2_codeword = np.empty((n, experiment.CODEBOOK_DIM), dtype=np.float32)
    for start in range(0, n, batch_size):
        batch = embeddings[start : start + batch_size]
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
            codes[start : start + batch.shape[0], level] = chosen.cpu().numpy()
            if level == 1:
                l2_margin[start : start + batch.shape[0]] = (
                    two[:, 1] - two[:, 0]
                ).cpu().numpy()
                l2_residual[start : start + batch.shape[0]] = pinned.cpu().numpy()
                l2_codeword[start : start + batch.shape[0]] = (
                    layer.get_code_embs()[chosen].cpu().numpy()
                )
            residual = model.rq._restore_norm(
                _hyperbolic_residual(
                    pinned, layer.get_code_embs()[chosen], c
                ),
                residual,
                target_norm,
            )
            previous_codes = chosen
    return {
        "codes": codes,
        "l2_margin": l2_margin,
        "l2_residual": l2_residual,
        "l2_codeword": l2_codeword,
    }


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    x = np.ascontiguousarray(x, dtype=np.float64)
    y = np.ascontiguousarray(y, dtype=np.float64)
    if x.std() == 0.0 or y.std() == 0.0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


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

    run = stack_pass(model, embeddings)
    codes = run["codes"]

    train_frame = pd.read_parquet(experiment.TRAIN_FILE)
    source_ids, successor_ids = transition_pairs(train_frame)
    pick = generator.choice(
        len(source_ids), size=experiment.BUCKET_PAIR_ROWS, replace=False
    )
    source_ids = np.ascontiguousarray(source_ids[pick])
    successor_ids = np.ascontiguousarray(successor_ids[pick])

    lines: list[str] = []
    result: dict = {
        "parent_checkpoint": str(experiment.PARENT_CKPT),
        "pairs": int(len(source_ids)),
    }
    lines.append(f"catalogue={n_items}  behaviour pairs sampled={len(source_ids)}")

    # ---- bucket structure at L1
    bucket_of_item = codes[:, 0]
    bucket_members: dict[int, np.ndarray] = {}
    for bucket in np.unique(bucket_of_item):
        bucket_members[int(bucket)] = np.flatnonzero(bucket_of_item == bucket)
    sizes = np.array([len(m) for m in bucket_members.values()])
    lines.append(
        f"L1 buckets: {len(bucket_members)}  "
        f"size p5/p50/p95 = {np.percentile(sizes, 5):.0f}/"
        f"{np.percentile(sizes, 50):.0f}/{np.percentile(sizes, 95):.0f}  "
        f"min={sizes.min()} max={sizes.max()}"
    )
    result["bucket_count"] = int(len(bucket_members))
    result["bucket_size"] = {
        "p5": float(np.percentile(sizes, 5)),
        "p50": float(np.percentile(sizes, 50)),
        "p95": float(np.percentile(sizes, 95)),
    }

    # ---- (1) behaviour density inside a bucket
    # For each real transition whose endpoints share an L1 code, count how many
    # behaviour successors the anchor has inside the same bucket. Comparing that
    # count against the bucket size says whether the bucket is a behaviour
    # community or a mixed bag.
    same_l1 = bucket_of_item[source_ids] == bucket_of_item[successor_ids]
    lines.append("")
    lines.append("1. is an L1 bucket a behaviour community?")
    lines.append(
        f"   behaviour pairs already sharing L1: {int(same_l1.sum())} "
        f"({same_l1.mean() * 100:.3f}%)"
    )
    result["pairs_sharing_L1"] = float(same_l1.mean())

    # successors per bucket, and per anchor how many successors share its bucket
    succ_per_anchor = np.bincount(
        source_ids, minlength=n_items
    ).astype(np.float64)
    lines.append(
        f"   behaviour out-degree: items with >=1 successor = "
        f"{int((succ_per_anchor > 0).sum())}/{n_items}; "
        f"mean out-degree = {succ_per_anchor.mean():.3f}"
    )
    result["mean_out_degree"] = float(succ_per_anchor.mean())

    # Expected number of a bucket's members that are behaviour successors of the
    # anchor, if behaviour partners were placed uniformly at random.
    uniform_share = np.array(
        [len(m) / n_items for m in bucket_members.values()]
    )
    lines.append(
        f"   if partners were uniform, a bucket of median size would hold "
        f"{np.median(sizes) / n_items * succ_per_anchor.mean():.3f} partners "
        f"per item on average"
    )

    # ---- (2) usage evenness inside a bucket vs behaviour density
    lines.append("")
    lines.append("2. how even is L2 usage inside a bucket?")
    evenness: list[float] = []
    for bucket, members in list(bucket_members.items()):
        if len(members) < 8:
            continue
        counts = np.bincount(codes[members, 1], minlength=256).astype(np.float64)
        used = counts[counts > 0]
        if len(used) < 2:
            continue
        # 1 - CV, so 1 is perfectly even usage
        evenness.append(float(1.0 - used.std() / used.mean()))
    evenness_arr = np.array(evenness)
    lines.append(
        f"   buckets with >=8 members: {len(evenness)}  "
        f"L2 usage evenness (1-CV) p5/p50/p95 = "
        f"{np.percentile(evenness_arr, 5):.3f}/"
        f"{np.percentile(evenness_arr, 50):.3f}/"
        f"{np.percentile(evenness_arr, 95):.3f}"
    )
    result["l2_usage_evenness"] = {
        "buckets": int(len(evenness)),
        "p5": float(np.percentile(evenness_arr, 5)),
        "p50": float(np.percentile(evenness_arr, 50)),
        "p95": float(np.percentile(evenness_arr, 95)),
    }

    # how many L2 codes does a bucket actually reach
    reach = [
        len(np.unique(codes[m, 1])) for m in bucket_members.values()
        if len(m) >= 8
    ]
    lines.append(
        f"   L2 codes reached per bucket: p5/p50/p95 = "
        f"{np.percentile(reach, 5):.0f}/{np.percentile(reach, 50):.0f}/"
        f"{np.percentile(reach, 95):.0f} of 256"
    )
    result["l2_codes_per_bucket"] = {
        "p5": float(np.percentile(reach, 5)),
        "p50": float(np.percentile(reach, 50)),
        "p95": float(np.percentile(reach, 95)),
    }

    # ---- (3) geometry of the pairs L2 separates
    lines.append("")
    lines.append("3. what does the split pair look like?")
    kept = same_l1 & (codes[source_ids, 1] == codes[successor_ids, 1])
    split = same_l1 & (codes[source_ids, 1] != codes[successor_ids, 1])
    lines.append(
        f"   among behaviour pairs sharing L1: kept L2 {int(kept.sum())}, "
        f"split L2 {int(split.sum())} "
        f"(split rate {split.sum() / max((same_l1).sum(), 1) * 100:.2f}%)"
    )
    result["kept_L2"] = int(kept.sum())
    result["split_L2"] = int(split.sum())

    def describe(mask: np.ndarray, label: str) -> dict:
        idx = np.flatnonzero(mask)
        if len(idx) == 0:
            return {}
        margin = run["l2_margin"][source_ids[idx]]
        residual = run["l2_residual"][source_ids[idx]]
        codeword = run["l2_codeword"][source_ids[idx]]
        cosine = (
            residual * codeword
        ).sum(-1) / (
            np.linalg.norm(residual, axis=-1) * np.linalg.norm(codeword, axis=-1)
        ).clip(1e-12)
        radii = np.linalg.norm(codeword, axis=-1)
        out = {
            "pairs": int(len(idx)),
            "l2_margin_p50": float(np.median(margin)),
            "l2_margin_mean": float(margin.mean()),
            "cosine_to_codeword_p50": float(np.median(cosine)),
            "codeword_radius_p50": float(np.median(radii)),
        }
        lines.append(
            f"   {label:<8} n={out['pairs']:<7} margin p50={out['l2_margin_p50']:.6f} "
            f"cos-to-code p50={out['cosine_to_codeword_p50']:.6f} "
            f"codeword radius p50={out['codeword_radius_p50']:.6f}"
        )
        return out

    result["kept_geometry"] = describe(kept, "kept")
    result["split_geometry"] = describe(split, "split")

    # Where do the split pairs sit relative to a codeword boundary? The gap to the
    # runner-up is the margin; a split pair with a large margin was not ambiguous
    # at all, which would mean the split came from the balancing rule rather than
    # from the geometry.
    lines.append("")
    lines.append("   is the split explained by ambiguity?")
    kept_margin = (
        run["l2_margin"][source_ids[np.flatnonzero(kept)]]
        if kept.any()
        else np.array([0.0])
    )
    split_margin = (
        run["l2_margin"][source_ids[np.flatnonzero(split)]]
        if split.any()
        else np.array([0.0])
    )
    ratio = (
        float(np.median(split_margin) / np.median(kept_margin))
        if kept.any() and np.median(kept_margin) > 0
        else float("nan")
    )
    lines.append(
        f"   split-pair margin is {ratio:.3f}x the kept-pair margin "
        f"(1.0 would mean splits and keeps are equally ambiguous)"
    )
    result["split_to_kept_margin_ratio"] = ratio

    report = "\n".join(lines)
    print(report, flush=True)
    experiment.SEPARATION_JSON.write_text(
        json.dumps(result, indent=2, default=float) + "\n", encoding="utf-8"
    )
    experiment.SEPARATION_LOG.write_text(report + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()