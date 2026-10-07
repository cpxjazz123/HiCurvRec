"""Do the RQ levels keep behaviour pairs together, or only split the space fine?

The retention ladder in iter114 fell 36.7% / 2.3% / 0.1% across L1 / L2 / L3, but
its "behaviour neighbour" was defined as the top-10 nearest items on the raw
encoder latent, and that definition collapses there: every pair's raw cosine is
about 0.998, so a neighbour and a random item are the same thing. A ladder built
on it measures how finely the space is subdivided, not how much behaviour
survives.

This uses the real behaviour pairs instead. A positive is an actual transition
A -> B+ taken from consecutive events in the train parquet, and the negative is a
random item for the same anchor A. Both endpoints are read at the parent's own
assignment, and the quantity reported is the difference

    G_l = P(agree on the first l codes | positive) - P(... | negative)

because G_l is what distinguishes retention from subdivision. A prefix rate that
is tiny for positives and tinier for negatives still means the level is keeping
behaviour pairs together; equal rates mean the level is not.

Two conditional rates are reported alongside it. P(agree L2 | agree L1) asks
whether the deeper levels take pairs the first level already grouped and pull
them apart again, which is the specific worry that a deep RQ undoes a shallow
one. G_l conditioned on the previous level agrees is reported for the negatives
too, so a difference at one level can be read as "the level separates
behaviour" rather than "the level separates everything".

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
    """Real behaviour positives: the last history item and the observed target."""
    sources: list[int] = []
    successors: list[int] = []
    for history, target in zip(
        train_frame["seen_history"].to_numpy(),
        train_frame["target"].to_numpy(dtype=np.int64),
    ):
        if history is None or len(history) == 0:
            continue
        source = int(history[-1])
        if source < 0:
            raise ValueError("Transition source item is negative")
        sources.append(source)
        successors.append(int(target))
    if not sources:
        raise ValueError("Training data has no usable item transitions")
    return (
        np.asarray(sources, dtype=np.int64),
        np.asarray(successors, dtype=np.int64),
    )


@torch.no_grad()
def tokens_for(model: RQVAE, embeddings: torch.Tensor, rows: np.ndarray,
               batch_size: int = 4096) -> np.ndarray:
    """Assign codes for the given items, exactly as the trainer does."""
    device = embeddings.device
    layers = model.rq.vq_layers
    out = np.empty((len(rows), len(layers)), dtype=np.int64)
    for start in range(0, len(rows), batch_size):
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
            chosen = layer._indices(
                distances, infer_use_sk=True, bucket=previous_codes
            )
            out[start : start + chunk.shape[0], level] = chosen.cpu().numpy()
            residual = model.rq._restore_norm(
                _hyperbolic_residual(
                    pinned, layer.get_code_embs()[chosen], c
                ),
                residual,
                target_norm,
            )
            previous_codes = chosen
    return out


def main() -> None:
    experiment.STAGE2_LOG_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    generator = np.random.default_rng(experiment.DIAGNOSTIC_SEED)
    torch.manual_seed(experiment.DIAGNOSTIC_SEED)

    embeddings = torch.from_numpy(
        np.asarray(np.load(experiment.EMBEDDING_FILE), dtype=np.float32)
    ).to(device)
    n_items = embeddings.shape[0]
    train_frame = pd.read_parquet(experiment.TRAIN_FILE)
    source_ids, successor_ids = transition_pairs(train_frame)
    pick = generator.choice(
        len(source_ids), size=experiment.BEHAVIOUR_ROWS, replace=False
    )
    source_ids = np.ascontiguousarray(source_ids[pick])
    successor_ids = np.ascontiguousarray(successor_ids[pick])
    # One random negative per anchor, never the anchor's own successor.
    negative_ids = generator.integers(0, n_items, size=len(source_ids))
    clash = negative_ids == successor_ids
    negative_ids[clash] = (negative_ids[clash] + 1) % n_items

    checkpoint = torch.load(
        experiment.PARENT_CKPT, map_location=device, weights_only=False
    )
    model = RQVAE(tokenizer_config(), in_dim=embeddings.shape[1]).to(device)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    tok_a = tokens_for(model, embeddings, source_ids)
    tok_p = tokens_for(model, embeddings, successor_ids)
    tok_n = tokens_for(model, embeddings, negative_ids)

    def agreement(left: np.ndarray, right: np.ndarray) -> np.ndarray:
        """(pairs, 3) booleans: pair agrees on the first l codes, l = 1..3."""
        return np.stack(
            [np.all(left[:, : l + 1] == right[:, : l + 1], axis=1)
             for l in range(3)],
            axis=1,
        )

    pos = agreement(tok_a, tok_p)
    neg = agreement(tok_a, tok_n)

    lines: list[str] = []
    result: dict = {
        "parent_checkpoint": str(experiment.PARENT_CKPT),
        "pairs": int(len(source_ids)),
        "levels": {},
    }
    lines.append(
        f"real behaviour pairs: {len(source_ids)}  "
        f"(one random negative per anchor, never its own successor)"
    )
    lines.append("")
    lines.append("prefix agreement rates")
    lines.append(
        f"   {'prefix':<10}{'positive':>12}{'negative':>12}{'G_l':>12}{'ratio':>10}"
    )
    for level in range(3):
        p = float(pos[:, level].mean())
        n = float(neg[:, level].mean())
        ratio = p / n if n > 0 else float("inf")
        lines.append(
            f"   L{level + 1}{'':<7}{p * 100:>11.4f}%{n * 100:>11.4f}%"
            f"{(p - n) * 100:>+11.4f}%{ratio:>10.1f}x"
        )
        result["levels"][f"L{level + 1}"] = {
            "positive": p, "negative": n, "gap": p - n, "ratio": ratio,
        }

    lines.append("")
    lines.append("conditional retention: does a deeper level undo the previous one?")
    lines.append(
        f"   {'conditional':<28}{'positive':>12}{'negative':>12}{'diff':>12}"
    )
    conditionals = {}
    for level in range(1, 3):
        previous = level - 1
        for label, mask_pos, mask_neg, source in (
            (f"P(L{level + 1} | L{previous + 1})", pos[:, previous], neg[:, previous], None),
        ):
            p_mask = pos[:, previous]
            n_mask = neg[:, previous]
            if p_mask.sum() == 0 or n_mask.sum() == 0:
                lines.append(f"   {label:<28}{'n/a':>12}{'n/a':>12}{'n/a':>12}")
                continue
            p_rate = float(pos[p_mask, level].mean())
            n_rate = float(neg[n_mask, level].mean())
            lines.append(
                f"   {label:<28}{p_rate * 100:>11.2f}%{n_rate * 100:>11.2f}%"
                f"{(p_rate - n_rate) * 100:>+11.2f}%"
            )
            conditionals[label] = {
                "positive": p_rate, "negative": n_rate, "gap": p_rate - n_rate,
            }
    result["conditional"] = conditionals

    # Unconditional-but-restricted view: for pairs that survived to L2, how often
    # do positives and negatives still agree at L3. Reported separately from the
    # conditional rate above so the two cannot be conflated.
    lines.append("")
    lines.append("pooled view over pairs that agree on the prefix so far")
    lines.append(
        f"   {'pool':<28}{'positive':>12}{'negative':>12}{'G':>12}"
    )
    pooled = {}
    for level in range(1, 3):
        p_mask = pos[:, level - 1]
        n_mask = neg[:, level - 1]
        p_rate = float(pos[p_mask, level].mean())
        n_rate = float(neg[n_mask, level].mean())
        lines.append(
            f"   agree L{level} -> also L{level + 1}{'':<6}"
            f"{p_rate * 100:>11.2f}%{n_rate * 100:>11.2f}%"
            f"{(p_rate - n_rate) * 100:>+11.2f}%"
        )
        pooled[f"L{level}_to_L{level + 1}"] = {
            "positive": p_rate, "negative": n_rate, "gap": p_rate - n_rate,
        }
    result["pooled"] = pooled

    lines.append("")
    lines.append("where the behaviour signal lives")
    gaps = [result["levels"][f"L{l + 1}"]["gap"] for l in range(3)]
    rates = [result["levels"][f"L{l + 1}"]["positive"] for l in range(3)]
    strongest = int(np.argmax(gaps))
    lines.append(
        f"   largest G_l at L{strongest + 1} (gap {gaps[strongest] * 100:+.4f}%, "
        f"positive rate {rates[strongest] * 100:.4f}%)"
    )
    lines.append(
        "   G_l = P(prefix | positive) - P(prefix | negative); a rate that is "
        "small for positives but smaller for negatives still means the level "
        "keeps behaviour pairs together"
    )
    result["strongest_level"] = strongest + 1

    report = "\n".join(lines)
    print(report, flush=True)
    experiment.SEPARATION_JSON.write_text(
        json.dumps(result, indent=2, default=float) + "\n", encoding="utf-8"
    )
    experiment.SEPARATION_LOG.write_text(report + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()