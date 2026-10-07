"""Frozen test: can curvature amplify a behaviour separation that already exists?

iter112 pointed curvature at the wrong ambiguity. It made the hard rows, the
ones whose top-1 and top-2 codewords are nearly tied, and it worked: those gaps
grew 7.6-9.5x more than the easy rows' gaps. But 97% of the resulting flips were
behaviour-neutral, because the two codewords a hard row cannot separate carry no
behaviour signal to recover. Curvature solved a geometric ambiguity that the
recommendation task does not have an opinion about.

This points it at the ambiguity that does matter. For a real transition A -> B+
and a random A -> X-, define the margin on the raw encoder latent, where the
behaviour ranking loss already lives and where no s-pin cancels the metric:

    M_beh = cos(A, B+) - cos(A, X-)

A behaviour-hard case is a pair with M_beh > 0 but small: the model already puts
the true successor in the right direction and the gap is merely weak. Curvature
cannot manufacture the sign, because both endpoints of a pair are measured at
the same curvature; all it can do is change the size of a margin that is already
positive. If a stronger metric cannot widen a real behavioural gap, then curvature
has no task-aligned degree of freedom left.

Passing needs the hard band's mean margin to grow by at least 10% while the easy
band stays within 10% of the parent. Nothing is trained.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

SOURCE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOURCE_DIR))

import curvature_config as experiment
from model.layers import _poincare_distance_tangent_pairs


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


def load_encoder(device: torch.device):
    """The parent encoder only; the behaviour loss never touches a codebook."""
    from types import SimpleNamespace

    from model import RQVAE

    config = SimpleNamespace(
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
    embeddings = torch.from_numpy(
        np.asarray(np.load(experiment.EMBEDDING_FILE), dtype=np.float32)
    ).to(device)
    model = RQVAE(config, in_dim=embeddings.shape[1]).to(device)
    checkpoint = torch.load(
        experiment.PARENT_CKPT, map_location=device, weights_only=False
    )
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, embeddings


def margins_at(curvature: float, anchors, positives, negatives, chunk: int = 4096):
    """Per-pair behaviour margin at one curvature, both endpoints at that value.

    The hyperbolic distance is used rather than the cosine, because curvature is
    the metric's parameter and a cosine would be invariant to it by
    construction. The pair is symmetric in curvature by construction too: there
    is no separate positive or negative curvature.
    """
    n = anchors.shape[0]
    out = np.empty(n, dtype=np.float64)
    for start in range(0, n, chunk):
        stop = min(start + chunk, n)
        a = anchors[start:stop]
        with torch.no_grad():
            positive = _poincare_distance_tangent_pairs(
                a, positives[start:stop], curvature
            )
            negative = _poincare_distance_tangent_pairs(
                a, negatives[start:stop], curvature
            )
        out[start:stop] = (
            negative - positive
        ).cpu().numpy().astype(np.float64)
    return out


def main() -> None:
    experiment.STAGE2_LOG_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    generator = np.random.default_rng(experiment.DIAGNOSTIC_SEED)
    torch.manual_seed(experiment.DIAGNOSTIC_SEED)

    model, embeddings = load_encoder(device)
    n_items = embeddings.shape[0]
    train_frame = pd.read_parquet(experiment.TRAIN_FILE)
    source_ids, successor_ids = transition_pairs(train_frame)
    pick = generator.choice(
        len(source_ids), size=experiment.BEHAVIOUR_ROWS, replace=False
    )
    source_ids = np.ascontiguousarray(source_ids[pick])
    successor_ids = np.ascontiguousarray(successor_ids[pick])
    negative_ids = generator.integers(0, n_items, size=len(source_ids))
    # A negative that is the pair's own successor measures nothing.
    negative_ids[negative_ids == successor_ids] = (
        negative_ids[negative_ids == successor_ids] + 1
    ) % n_items

    with torch.no_grad():
        anchors = model.encoder(embeddings[torch.from_numpy(source_ids).to(device)])
        positives = model.encoder(
            embeddings[torch.from_numpy(successor_ids).to(device)]
        )
        negatives = anchors[torch.from_numpy(negative_ids).to(device)]

    lines: list[str] = []
    result: dict = {
        "parent_checkpoint": str(experiment.PARENT_CKPT),
        "pairs": int(len(source_ids)),
        "probe_curvatures": list(experiment.PROBE_CURVATURES),
    }
    lines.append(
        f"behaviour pairs: {len(source_ids)}  "
        f"raw latent ||z||_p50={float(anchors.norm(dim=-1).median()):.4f} (unpinned)"
    )

    parent = margins_at(1.0, anchors, positives, negatives)
    positive_margin = float((parent > experiment.BEHAVIOUR_MARGIN_MIN).mean())
    lines.append(
        f"parent: margin = d_1(A,X-) - d_1(A,B+); "
        f"rows with margin > 0 = {positive_margin * 100:.2f}%"
    )
    result["parent"] = {
        "positive_fraction": positive_margin,
        "mean": float(parent.mean()),
        "median": float(np.median(parent)),
    }

    # Behaviour-hard: positive but in the lowest quartile of the positive ones.
    eligible = parent > experiment.BEHAVIOUR_MARGIN_MIN
    positive_margins = parent[eligible]
    floor = float(
        np.quantile(positive_margins, experiment.BEHAVIOUR_MARGIN_FLOOR_QUANTILE)
    )
    hard = eligible & (parent <= floor)
    easy = eligible & (parent > np.quantile(positive_margins, 0.75))
    lines.append(
        f"  hard band: margin in (0, {floor:.8f}] -> {int(hard.sum())} rows "
        f"({hard.mean() * 100:.2f}% of all)"
    )
    lines.append(
        f"  easy band: margin > {np.quantile(positive_margins, 0.75):.8f} -> "
        f"{int(easy.sum())} rows ({easy.mean() * 100:.2f}%)"
    )
    lines.append(
        f"  parent means: hard={parent[hard].mean():.6f}  "
        f"easy={parent[easy].mean():.6f}"
    )
    result["hard_band"] = {
        "floor": floor,
        "rows": int(hard.sum()),
        "parent_mean": float(parent[hard].mean()),
    }
    result["easy_band"] = {
        "rows": int(easy.sum()),
        "parent_mean": float(parent[easy].mean()),
    }

    lines.append("")
    lines.append("per-curvature margin on the raw encoder latent")
    result["curvatures"] = {}
    for curvature in experiment.PROBE_CURVATURES:
        if curvature == 1.0:
            continue
        margins = margins_at(curvature, anchors, positives, negatives)
        hard_mean = float(margins[hard].mean())
        easy_mean = float(margins[easy].mean())
        all_mean = float(margins.mean())
        hard_ratio = hard_mean / parent[hard].mean()
        easy_ratio = easy_mean / parent[easy].mean()
        # Sign preservation: a curvature that reorders pairs so the true
        # successor falls behind has manufactured rather than amplified.
        sign_kept = float(((margins > 0) == (parent > 0)).mean())
        rank_correlation = float(
            np.corrcoef(
                np.argsort(np.argsort(parent)),
                np.argsort(np.argsort(margins)),
            )[0, 1]
        )
        hard_grew = hard_ratio >= experiment.HARD_MIN_GAIN_RATIO
        easy_stable = abs(easy_ratio) <= experiment.EASY_MAX_DRIFT_RATIO
        entry = {
            "curvature": curvature,
            "hard_mean": hard_mean,
            "easy_mean": easy_mean,
            "all_mean": all_mean,
            "hard_ratio": hard_ratio,
            "easy_ratio": easy_ratio,
            "all_ratio": all_mean / parent.mean(),
            "sign_preserved": sign_kept,
            "rank_correlation_with_parent": rank_correlation,
        }
        entry["verdict"] = {
            "hard_margin_grew": bool(hard_grew),
            "easy_stable": bool(easy_stable),
            "passes": bool(hard_grew and easy_stable),
        }
        result["curvatures"][f"{curvature}"] = entry
        lines.append(
            f"c={curvature:<5} hard {parent[hard].mean():.6f} -> {hard_mean:.6f} "
            f"({hard_ratio:.4f}x)   easy {parent[easy].mean():.6f} -> {easy_mean:.6f} "
            f"({easy_ratio:.4f}x)"
        )
        lines.append(
            f"        all {parent.mean():.6f} -> {all_mean:.6f}   "
            f"sign kept {sign_kept * 100:.2f}%   "
            f"rank corr {rank_correlation:.5f}   "
            f"-> {'PASS' if entry['verdict']['passes'] else 'fail'}"
        )

    lines.append("")
    lines.append("verdict")
    any_pass = any(
        entry["verdict"]["passes"] for entry in result["curvatures"].values()
    )
    result["any_candidate_passes"] = bool(any_pass)
    for key, entry in result["curvatures"].items():
        v = entry["verdict"]
        lines.append(
            f"   c={key}: hard_margin_grew={int(v['hard_margin_grew'])} "
            f"easy_stable={int(v['easy_stable'])} -> "
            f"{'PASS' if v['passes'] else 'fail'}"
        )
    if any_pass:
        lines.append(
            "   curvature can amplify a behaviour separation that already exists; "
            "a training run is justified"
        )
    else:
        lines.append(
            "   no curvature amplifies a weak behaviour margin without moving the "
            "strong ones, so curvature has no task-aligned degree of freedom left"
        )

    report = "\n".join(lines)
    print(report, flush=True)
    experiment.DIAGNOSTIC_JSON.write_text(
        json.dumps(result, indent=2, default=float) + "\n", encoding="utf-8"
    )
    experiment.DIAGNOSTIC_LOG.write_text(report + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()