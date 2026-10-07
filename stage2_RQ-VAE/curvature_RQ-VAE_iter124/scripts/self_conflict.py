"""Do the behaviour constraints on one anchor agree, or cancel each other?

iter123 measured the behaviour gradient against the rest of the Stage2 objective
and found them orthogonal rather than opposed: reconstruction and quantization
are seven orders of magnitude smaller on the persistent pairs, so nothing else is
holding them back. The behaviour term has the largest gradient there of any
population and still cannot separate them after 72000 steps.

That leaves one mechanism untested. The parent's loss is a pairwise hinge, and
each (A, B+, X-) triple is optimised on its own, but the same anchor appears in
many transitions. If one anchor carries several transitions, each produces a
gradient on that anchor's latent, and those gradients can point different ways:

    triplet 1:  A pushed one way
    triplet 2:  A pushed the other

Individually every gradient is large, which is exactly what iter123 measured,
while their sum can be small. That would produce the observed combination of the
largest per-triple gradient in the persistent set with no convergence after the
whole run, and it would be invisible to a per-pair measurement.

The consistency measure is the ratio of the norm of the summed gradient to the
sum of the norms,

    C_A = || sum_i g_i || / sum_i ||g_i ||

which is 1 when every constraint wants the same change and near 0 when they
cancel. It is reported per population, since the point is whether persistent
anchors are less consistent than the ones that converged. Nothing is trained.
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


def per_triple_gradient(
    anchor: torch.Tensor,
    successor: torch.Tensor,
    negative: torch.Tensor,
    curvature: float,
) -> torch.Tensor:
    """Gradient of one triple's hinge loss with respect to the anchor's latent."""
    variable = anchor.detach().clone().requires_grad_(True)
    positive = _poincare_distance_tangent_pairs(variable, successor, curvature)
    negative_d = _poincare_distance_tangent_pairs(variable, negative, curvature)
    loss = torch.relu(positive + experiment.BEHAVIOUR_MARGIN - negative_d)
    (grad,) = torch.autograd.grad(loss, variable)
    return grad.detach()


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
    curvature = float(experiment.LAYER_CURVATURES[0])

    train_frame = pd.read_parquet(experiment.TRAIN_FILE)
    source_ids, successor_ids = transition_pairs(train_frame)

    # Group transitions by anchor so an anchor's constraints can be compared.
    order = np.argsort(source_ids, kind="stable")
    sorted_sources = source_ids[order]
    sorted_successors = successor_ids[order]
    counts = np.bincount(source_ids, minlength=n_items)
    multi = np.flatnonzero(counts >= experiment.MIN_TRANSITIONS_PER_ANCHOR)
    result_population_sample = min(
        experiment.CONFLICT_ANCHORS, len(multi)
    )
    anchors_chosen = generator.choice(multi, size=result_population_sample, replace=False)
    anchors_chosen.sort()

    lines: list[str] = []
    result: dict = {
        "parent_checkpoint": str(experiment.PARENT_CKPT),
        "anchors_with_multiple_transitions": int(len(multi)),
        "anchors_measured": int(len(anchors_chosen)),
        "min_transitions_per_anchor": experiment.MIN_TRANSITIONS_PER_ANCHOR,
    }
    lines.append(
        f"anchors with >= {experiment.MIN_TRANSITIONS_PER_ANCHOR} transitions: "
        f"{len(multi)} of {n_items} items; measuring {len(anchors_chosen)}"
    )

    # Populations come from the same five checkpoints as iter121 and iter122, so
    # an anchor can be labelled persistent, rotating or never.
    trajectory_dir = (
        experiment.REPO_ROOT
        / "results/stage2_RQ-VAE/curvature_RQ-VAE_iter120/trajectory"
    )
    paths = sorted(
        trajectory_dir.glob("step_*.pth"),
        key=lambda q: int(q.stem.split("_")[1]),
    ) + [experiment.PARENT_CKPT]
    columns = []
    for path in paths:
        other = RQVAE(tokenizer_config(), in_dim=embeddings.shape[1]).to(device)
        other.load_state_dict(
            torch.load(path, map_location=device, weights_only=False)["state_dict"],
            strict=True,
        )
        other.eval()
        with torch.no_grad():
            active = []
            for block in range(0, len(source_ids), 4096):
                idx = torch.from_numpy(
                    np.ascontiguousarray(source_ids[block : block + 4096])
                ).to(device)
                perm = torch.randperm(idx.shape[0], device=device)
                a = other.encoder(embeddings[idx])
                b = other.encoder(
                    embeddings[torch.from_numpy(
                        np.ascontiguousarray(successor_ids[block : block + 4096])
                    ).to(device)]
                )
                c = a[perm]
                positive = _poincare_distance_tangent_pairs(a, b, curvature)
                negative = _poincare_distance_tangent_pairs(a, c, curvature)
                active.append(
                    ((negative - positive) < experiment.BEHAVIOUR_MARGIN)
                    .cpu()
                    .numpy()
                )
            columns.append(np.concatenate(active))
        del other
        torch.cuda.empty_cache()
    stack = np.stack(columns, axis=1)
    pair_persistent = stack.all(axis=1)
    pair_never = (~stack).all(axis=1)
    pair_rotating = ~pair_persistent & ~pair_never
    result["pair_populations"] = {
        "persistent": int(pair_persistent.sum()),
        "rotating": int(pair_rotating.sum()),
        "never": int(pair_never.sum()),
    }
    lines.append(
        f"pairs over {len(paths)} checkpoints: persistent="
        f"{int(pair_persistent.sum())}  rotating={int(pair_rotating.sum())}  "
        f"never={int(pair_never.sum())}"
    )

    # An anchor is persistent if all of its transitions are persistent; an anchor
    # with a mix is rotating, since that is the state that fails to converge.
    def anchor_population(mask: np.ndarray) -> np.ndarray:
        population = np.zeros(len(anchors_chosen), dtype=np.int8)
        for i, anchor in enumerate(anchors_chosen):
            rows = np.flatnonzero(source_ids == anchor)
            if mask[rows].all():
                population[i] = 1
            elif mask[rows].any():
                population[i] = 2
        return population

    anchor_kind = anchor_population(pair_persistent)

    # ---- per-anchor gradient consistency
    consistency = np.zeros(len(anchors_chosen), dtype=np.float64)
    total_norm = np.zeros(len(anchors_chosen), dtype=np.float64)
    summed_norm = np.zeros(len(anchors_chosen), dtype=np.float64)
    transitions_per_anchor = np.zeros(len(anchors_chosen), dtype=np.int64)

    for i, anchor in enumerate(anchors_chosen):
        rows = np.flatnonzero(source_ids == anchor)
        transitions_per_anchor[i] = len(rows)
        anchor_latent = model.encoder(
            embeddings[torch.from_numpy(np.array([anchor], dtype=np.int64)).to(device)]
        ).detach()
        grads = []
        # One negative per transition, drawn the way the parent draws them: a
        # random permutation partner, reused so the anchor's own gradient is what
        # the hinge would push.
        partner = generator.permutation(len(rows))
        for local, row in enumerate(rows):
            successor = model.encoder(
                embeddings[
                    torch.from_numpy(
                        np.array([successor_ids[row]], dtype=np.int64)
                    ).to(device)
                ]
            ).detach()
            neg_row = rows[partner[local % len(rows)]]
            negative = model.encoder(
                embeddings[
                    torch.from_numpy(
                        np.array([source_ids[neg_row]], dtype=np.int64)
                    ).to(device)
                ]
            ).detach()
            grads.append(
                per_triple_gradient(anchor_latent, successor, negative, curvature)
            )
        stacked = torch.stack(grads, dim=0)
        total_norm[i] = stacked.norm(dim=-1).sum().item()
        summed_norm[i] = stacked.sum(dim=0).norm().item()
        consistency[i] = (
            summed_norm[i] / total_norm[i] if total_norm[i] > 0 else float("nan")
        )

    result["mean_transitions_per_anchor"] = float(transitions_per_anchor.mean())
    lines.append(
        f"mean transitions per measured anchor: {transitions_per_anchor.mean():.2f}"
    )

    lines.append("")
    lines.append("1. gradient consistency C_A = ||sum g|| / sum ||g||")
    lines.append(f"   {'population':<12}{'n':>7}{'C_A p50':>12}{'mean':>10}{'sum|g|':>12}")
    for label, kind in (
        ("persistent", 1),
        ("mixed", 2),
        ("other", 0),
    ):
        mask = anchor_kind == kind
        if mask.sum() == 0:
            continue
        c = consistency[mask]
        finite = c[np.isfinite(c)]
        if len(finite) == 0:
            continue
        entry = {
            "anchors": int(mask.sum()),
            "consistency_p50": float(np.median(finite)),
            "consistency_mean": float(finite.mean()),
            "summed_norm_mean": float(summed_norm[mask].mean()),
        }
        result.setdefault("consistency", {})[label] = entry
        lines.append(
            f"   {label:<12}{entry['anchors']:>7}{entry['consistency_p50']:>12.4f}"
            f"{entry['consistency_mean']:>10.4f}{entry['summed_norm_mean']:>12.4f}"
        )

    # ---- how much of each anchor's push survives summation
    lines.append("")
    lines.append("2. the gradient budget that survives summation")
    total_all = summed_norm.sum()
    for label, kind in (
        ("persistent", 1),
        ("mixed", 2),
        ("other", 0),
    ):
        mask = anchor_kind == kind
        if mask.sum() == 0:
            continue
        share = summed_norm[mask].sum() / total_all if total_all > 0 else float("nan")
        raw_share = total_norm[mask].sum() / (
            total_norm.sum() if total_norm.sum() > 0 else 1.0
        )
        result.setdefault("surviving_share", {})[label] = {
            "after_summation": float(share),
            "before_summation": float(raw_share),
        }
        lines.append(
            f"   {label:<12}share of sum|g_i| before summation {raw_share * 100:5.2f}%"
            f"   after {share * 100:5.2f}%"
        )

    lines.append("")
    lines.append("verdict")
    persistent_c = result.get("consistency", {}).get("persistent", {}).get(
        "consistency_p50"
    )
    mixed_c = result.get("consistency", {}).get("mixed", {}).get("consistency_p50")
    reference = mixed_c if mixed_c is not None else persistent_c
    cancels = (
        persistent_c is not None
        and reference is not None
        and persistent_c < 0.5 * reference
    )
    if cancels:
        verdict = (
            "the constraints on a persistent anchor largely cancel: each triple "
            "pushes hard, but the sum is a small fraction of the individual norms, "
            "so the pairwise form is the limit"
        )
    elif persistent_c is not None and persistent_c < 0.8:
        verdict = (
            "partial cancellation on persistent anchors, but not the dominant "
            "effect; the objective form still matters"
        )
    else:
        verdict = (
            "the constraints agree; there is no self-conflict, so the limit is "
            "what a pairwise objective can express rather than interference"
        )
    result["verdict"] = verdict
    result["self_cancelling"] = bool(cancels)
    lines.append(f"   self-cancelling: {cancels}")
    lines.append(f"   {verdict}")

    report = "\n".join(lines)
    print(report, flush=True)
    experiment.DECOMP_JSON.write_text(
        json.dumps(result, indent=2, default=float) + "\n", encoding="utf-8"
    )
    experiment.DECOMP_LOG.write_text(report + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()