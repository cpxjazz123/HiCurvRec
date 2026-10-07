"""Frozen test: does curvature do useful work only on ambiguous assignments?

Curvature has never helped this model when applied globally. iter96 measured it
moving behaviour distances by 0.13% on the residual side because the s-pin
cancels the metric factor against the ball depth; iter107 and iter111 moved it
anyway and lost 3.4-3.5% recall, because a global change disturbs the whole
space.

The premise here is that curvature is not useless, just misapplied: a level
where two codewords are already far apart needs no help, while the rows whose
top-1 and top-2 are nearly tied are exactly the rows where the metric factor is
the only thing left to separate them. So curvature is applied at one level and
judged only on whether it helps those rows.

Four questions, in the order they can falsify the idea:

  1. Is the effect actually selective? Absolute gap change is reported for the
     hard band and the easy band separately. A ratio of gaps would be
     meaningless here, since a hard row's parent gap is near zero and any ratio
     explodes; only the absolute difference is read.
  2. Does a bigger gap mean better separation, or just churn? Top-1 stability is
     reported, and the flipped rows are then examined rather than discarded.
  3. Do the flips point the right way for behaviour? On the hard rows, the
     prefix-agreement gain between a true successor and a random item is
     compared across curvatures. A larger gap that costs behaviour agreement is
     not disambiguation.
  4. Among the rows that did flip, how many ended up more behaviour-consistent
     than before.

Passing needs all three of: hard gap grows, easy band barely moves, behaviour
agreement improves. Nothing is trained.
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


def tokenizer_config(curvature: float) -> SimpleNamespace:
    """Parent config with a single level's assignment curvature moved."""
    curvatures = [1.0, 1.0, 1.0]
    curvatures[experiment.PROBE_LEVEL] = curvature
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
        layer_curvatures=tuple(curvatures),
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
def level_pass(
    model: RQVAE,
    embeddings: torch.Tensor,
    level: int,
    rows: np.ndarray,
    batch_size: int = 4096,
) -> dict:
    """Run the stack and return the probe level's gaps, codes and prefixes.

    The level under test keeps its own curvature for the assignment; the levels
    before it stay at the parent's, because changing them would move the input
    to the probe level and conflate the two effects.
    """
    device = embeddings.device
    levels = model.rq.vq_layers
    n = len(rows)
    gaps = np.empty(n, dtype=np.float64)
    codes = np.empty(n, dtype=np.int64)
    prefixes = np.empty((n, len(levels)), dtype=np.int64)
    for start in range(0, n, batch_size):
        chunk = torch.from_numpy(np.ascontiguousarray(rows[start : start + batch_size]))
        batch = embeddings[chunk.to(device)]
        residual = model.encoder(batch)
        previous_codes = None
        for lv, layer in enumerate(levels):
            c2 = float(layer.get_curvature())
            target_norm = model.rq._radius_for_level(lv, c2, residual)
            pinned = model.rq._pin_to_radius(residual, target_norm)
            distances = _pairwise_poincare_distance_tangents(
                pinned, layer.get_code_embs(), c2
            )
            two = torch.topk(distances, k=2, dim=1, largest=False).values
            if lv == level:
                gaps[start : start + chunk.shape[0]] = (
                    two[:, 1] - two[:, 0]
                ).cpu().numpy()
            chosen = layer._indices(
                distances, infer_use_sk=True, bucket=previous_codes
            )
            prefixes[start : start + chunk.shape[0], lv] = chosen.cpu().numpy()
            if lv == level:
                codes[start : start + chunk.shape[0]] = chosen.cpu().numpy()
            residual = model.rq._restore_norm(
                _hyperbolic_residual(pinned, layer.get_code_embs()[chosen], c2),
                residual,
                target_norm,
            )
            previous_codes = chosen
    return {"gap": gaps, "code": codes, "prefix": prefixes}


def main() -> None:
    experiment.STAGE2_LOG_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    generator = np.random.default_rng(experiment.DIAGNOSTIC_SEED)
    torch.manual_seed(experiment.DIAGNOSTIC_SEED)

    embeddings = torch.from_numpy(
        np.asarray(np.load(experiment.EMBEDDING_FILE), dtype=np.float32)
    ).to(device)
    n_items = embeddings.shape[0]
    corpus = np.arange(n_items, dtype=np.int64)

    train_frame = pd.read_parquet(experiment.TRAIN_FILE)
    source_ids, successor_ids = transition_pairs(train_frame)
    pick = generator.choice(
        len(source_ids), size=experiment.BEHAVIOUR_ROWS, replace=False
    )
    source_ids, successor_ids = source_ids[pick], successor_ids[pick]
    negative_ids = generator.integers(0, n_items, size=len(source_ids))
    clash = negative_ids == successor_ids
    negative_ids[clash] = (negative_ids[clash] + 1) % n_items
    # Union of everything the behaviour read has to look up.
    beh_rows = np.unique(
        np.concatenate([source_ids, successor_ids, negative_ids])
    )
    row_of = {int(r): i for i, r in enumerate(beh_rows)}

    checkpoint = torch.load(
        experiment.PARENT_CKPT, map_location=device, weights_only=False
    )
    level = experiment.PROBE_LEVEL
    lines: list[str] = []
    result: dict = {
        "parent_checkpoint": str(experiment.PARENT_CKPT),
        "probe_level": level,
        "hard_quantile": experiment.HARD_QUANTILE,
        "behaviour_rows": int(len(beh_rows)),
    }

    def model_at(curvature: float):
        model = RQVAE(
            tokenizer_config(curvature), in_dim=embeddings.shape[1]
        ).to(device)
        model.load_state_dict(checkpoint["state_dict"], strict=True)
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        return model

    # Parent reference on the corpus, which defines the hard band.
    parent = model_at(1.0)
    base = level_pass(parent, embeddings, level, corpus)
    threshold = float(np.quantile(base["gap"], experiment.HARD_QUANTILE))
    hard = base["gap"] <= threshold
    lines.append(
        f"probe level L{level + 1} (the only codebook with radial freedom); "
        f"corpus rows={len(corpus)}"
    )
    lines.append(
        f"hard band = parent gap <= {threshold:.8f}  "
        f"({int(hard.sum())} rows, {hard.mean() * 100:.2f}%)"
    )
    lines.append(
        f"  hard  parent gap: p50={np.median(base['gap'][hard]):.8f}  "
        f"easy  parent gap: p50={np.median(base['gap'][~hard]):.8f}"
    )
    result["hard_threshold"] = threshold
    result["hard_rows"] = int(hard.sum())
    result["parent"] = {
        "hard_gap_p50": float(np.median(base["gap"][hard])),
        "easy_gap_p50": float(np.median(base["gap"][~hard])),
    }

    # Behaviour read, on the parent's own assignment.
    base_beh = level_pass(parent, embeddings, level, beh_rows)

    def behaviour_gain(prefix_of_row, hard_row_ids):
        """Prefix-agreement gain between a true successor and a random item."""
        s = np.array([row_of[int(r)] for r in source_ids])
        b = np.array([row_of[int(r)] for r in successor_ids])
        x = np.array([row_of[int(r)] for r in negative_ids])
        keep = hard_row_ids[s] if hard_row_ids is not None else np.ones(len(s), bool)
        pos = prefix_of_row[s[keep], : level + 1]
        posb = prefix_of_row[b[keep], : level + 1]
        posx = prefix_of_row[x[keep], : level + 1]
        same_pos = np.all(pos == posb, axis=1).mean()
        same_neg = np.all(pos == posx, axis=1).mean()
        return float(same_pos), float(same_neg), float(same_pos - same_neg), int(keep.sum())

    pos, neg, gain, n_all = behaviour_gain(base_beh["prefix"], None)
    lines.append(
        f"  behaviour (all rows, n={n_all}): pos={pos * 100:.3f}%  "
        f"neg={neg * 100:.3f}%  gain={gain:+.5f}"
    )
    result["behaviour_parent_all"] = {"positive": pos, "negative": neg, "gain": gain}

    # hard-row behaviour under the parent, using the corpus-level hard flag
    hard_on_beh = hard[beh_rows]
    pos_h, neg_h, gain_h, n_h = behaviour_gain(base_beh["prefix"], hard_on_beh)
    lines.append(
        f"  behaviour (hard rows only, n={n_h}): pos={pos_h * 100:.3f}%  "
        f"neg={neg_h * 100:.3f}%  gain={gain_h:+.5f}"
    )
    result["behaviour_parent_hard"] = {
        "positive": pos_h, "negative": neg_h, "gain": gain_h, "rows": n_h,
    }

    lines.append("")
    lines.append("per-curvature comparison against the parent")
    result["curvatures"] = {}
    for curvature in experiment.PROBE_CURVATURES:
        if curvature == 1.0:
            continue
        model = model_at(curvature)
        cur = level_pass(model, embeddings, level, corpus)
        d_gap = cur["gap"] - base["gap"]
        hard_delta = float(d_gap[hard].mean())
        easy_delta = float(d_gap[~hard].mean())
        stability = float((cur["code"] == base["code"]).mean())
        stability_hard = float((cur["code"][hard] == base["code"][hard]).mean())
        flips = cur["code"] != base["code"]

        cur_beh = level_pass(model, embeddings, level, beh_rows)
        _, _, gain_c, _ = behaviour_gain(cur_beh["prefix"], None)
        pos_c, neg_c, gain_c_h, _ = behaviour_gain(cur_beh["prefix"], hard_on_beh)

        entry = {
            "curvature": curvature,
            "hard_gap_delta": hard_delta,
            "easy_gap_delta": easy_delta,
            "hard_gap_delta_p50": float(np.median(d_gap[hard])),
            "easy_gap_delta_p50": float(np.median(d_gap[~hard])),
            "top1_stability_all": stability,
            "top1_stability_hard": stability_hard,
            "flips_total": int(flips.sum()),
            "behaviour_gain_all": gain_c,
            "behaviour_gain_all_delta": gain_c - gain,
            "behaviour_gain_hard": gain_c_h,
            "behaviour_gain_hard_delta": gain_c_h - gain_h,
        }

        # (4) direction of the flips, judged against behaviour rather than
        # counted: did the new assignment agree more with the true successor?
        s_idx = np.array([row_of[int(r)] for r in source_ids])
        b_idx = np.array([row_of[int(r)] for r in successor_ids])
        x_idx = np.array([row_of[int(r)] for r in negative_ids])
        keep = hard_on_beh[s_idx]
        beh_flips = keep & (cur_beh["code"][s_idx] != base_beh["code"][s_idx])
        if beh_flips.any():
            before = (
                (base_beh["code"][s_idx[beh_flips]] == base_beh["code"][b_idx[beh_flips]]).astype(float)
                - (base_beh["code"][s_idx[beh_flips]] == base_beh["code"][x_idx[beh_flips]]).astype(float)
            )
            after = (
                (cur_beh["code"][s_idx[beh_flips]] == cur_beh["code"][b_idx[beh_flips]]).astype(float)
                - (cur_beh["code"][s_idx[beh_flips]] == cur_beh["code"][x_idx[beh_flips]]).astype(float)
            )
            delta = after - before
            entry["behaviour_flips_on_hard"] = int(beh_flips.sum())
            entry["beneficial_flip_rate"] = float((delta > 0).mean())
            entry["harmful_flip_rate"] = float((delta < 0).mean())
            entry["neutral_flip_rate"] = float((delta == 0).mean())
            entry["mean_flip_behaviour_delta"] = float(delta.mean())
        else:
            entry["behaviour_flips_on_hard"] = 0
            entry["beneficial_flip_rate"] = None
            entry["harmful_flip_rate"] = None
            entry["neutral_flip_rate"] = None
            entry["mean_flip_behaviour_delta"] = None

        result["curvatures"][f"{curvature}"] = entry
        lines.append(
            f"c={curvature}: hard dgap={hard_delta:+.6f}  easy dgap={easy_delta:+.6f}  "
            f"selective={abs(hard_delta) > 5 * max(abs(easy_delta), 1e-9)}"
        )
        lines.append(
            f"       top1 stability all={stability * 100:.2f}% hard={stability_hard * 100:.2f}%  "
            f"flips={int(flips.sum())}"
        )
        lines.append(
            f"       behaviour gain all {gain:+.5f} -> {gain_c:+.5f} "
            f"({gain_c - gain:+.5f})   hard {gain_h:+.5f} -> {gain_c_h:+.5f} "
            f"({gain_c_h - gain_h:+.5f})"
        )
        if entry["behaviour_flips_on_hard"]:
            lines.append(
                f"       flips on hard behaviour rows: {entry['behaviour_flips_on_hard']}  "
                f"beneficial={entry['beneficial_flip_rate'] * 100:.1f}%  "
                f"harmful={entry['harmful_flip_rate'] * 100:.1f}%  "
                f"neutral={entry['neutral_flip_rate'] * 100:.1f}%"
            )
        del model, cur, cur_beh
        torch.cuda.empty_cache()

    # ---- the pass condition the brief sets, read off the numbers above
    lines.append("")
    lines.append("verdict")
    for key, entry in result["curvatures"].items():
        selective = abs(entry["hard_gap_delta"]) > 5 * max(
            abs(entry["easy_gap_delta"]), 1e-9
        )
        grew = entry["hard_gap_delta"] > 0
        helped = entry["behaviour_gain_hard_delta"] > 0
        entry["verdict"] = {
            "selective": bool(selective),
            "hard_gap_grew": bool(grew),
            "behaviour_helped": bool(helped),
            "passes": bool(selective and grew and helped),
        }
        lines.append(
            f"   c={key}: selective={int(selective)} hard_gap_grew={int(grew)} "
            f"behaviour_helped={int(helped)} -> "
            f"{'PASS' if selective and grew and helped else 'fail'}"
        )
    result["verdict"] = {
        key: entry["verdict"] for key, entry in result["curvatures"].items()
    }

    report = "\n".join(lines)
    print(report, flush=True)
    experiment.DIAGNOSTIC_JSON.write_text(
        json.dumps(result, indent=2, default=float) + "\n", encoding="utf-8"
    )
    experiment.DIAGNOSTIC_LOG.write_text(report + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()