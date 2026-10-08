"""Frozen one-step comparison of pairwise and hardest-negative behaviour hinges.

Only the negative count changes: the parent's single shuffled-source negative is
compared with the closest of K=16 independently shuffled sources. Both objectives
retain the same hyperbolic margin hinge. No encoder or model parameter is updated.
All measurements print to stdout; this diagnostic writes no report artifacts.
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
import train_rqvae as stage2
from model import RQVAE
from model.layers import _poincare_distance_tangent_pairs


STEP_SIZE = 0.01
N_NEGATIVES = 16
PAIR_COUNT = 20_000
SEED = 42
TRAJECTORY_DIR = (
    experiment.REPO_ROOT
    / "results/stage2_RQ-VAE/curvature_RQ-VAE_iter120/trajectory"
)


def unit(v: torch.Tensor) -> torch.Tensor:
    return v / torch.linalg.vector_norm(v, dim=-1, keepdim=True).clamp_min(1e-12)


def angular_margin(
    anchor: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor
) -> torch.Tensor:
    return (unit(anchor) * unit(positive)).sum(-1) - (
        unit(anchor) * unit(negative)
    ).sum(-1)


def angular_hard_margin(
    anchor: torch.Tensor,
    positive: torch.Tensor,
    negatives: torch.Tensor,
    curvature: float,
) -> torch.Tensor:
    expanded = anchor.unsqueeze(1).expand(-1, negatives.shape[1], -1)
    distances = _poincare_distance_tangent_pairs(expanded, negatives, curvature)
    hard_index = distances.argmin(dim=1)
    hard_negative = negatives[torch.arange(len(anchor), device=anchor.device), hard_index]
    return angular_margin(anchor, positive, hard_negative)


def distance_margins(
    anchor: torch.Tensor,
    positive: torch.Tensor,
    original_negative: torch.Tensor,
    negatives: torch.Tensor,
    curvature: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    d_pos = _poincare_distance_tangent_pairs(anchor, positive, curvature)
    d_original = _poincare_distance_tangent_pairs(anchor, original_negative, curvature)
    expanded = anchor.unsqueeze(1).expand(-1, negatives.shape[1], -1)
    d_all = _poincare_distance_tangent_pairs(expanded, negatives, curvature)
    return d_original - d_pos, d_all.min(dim=1).values - d_pos


def pairwise_gradient(
    anchor: torch.Tensor,
    positive: torch.Tensor,
    negative: torch.Tensor,
    curvature: float,
) -> torch.Tensor:
    variable = anchor.detach().clone().requires_grad_(True)
    d_pos = _poincare_distance_tangent_pairs(variable, positive, curvature)
    d_neg = _poincare_distance_tangent_pairs(variable, negative, curvature)
    loss = torch.relu(d_pos + stage2.BEHAVIOUR_MARGIN - d_neg).sum()
    (gradient,) = torch.autograd.grad(loss, variable)
    return gradient.detach()


def multi_hinge_gradient(
    anchor: torch.Tensor,
    positive: torch.Tensor,
    negatives: torch.Tensor,
    curvature: float,
) -> torch.Tensor:
    variable = anchor.detach().clone().requires_grad_(True)
    expanded = variable.unsqueeze(1).expand(-1, negatives.shape[1], -1)
    d_pos = _poincare_distance_tangent_pairs(variable, positive, curvature)
    d_neg = _poincare_distance_tangent_pairs(expanded, negatives, curvature)
    d_hard = d_neg.min(dim=1).values
    loss = torch.relu(d_pos + stage2.BEHAVIOUR_MARGIN - d_hard).sum()
    (gradient,) = torch.autograd.grad(loss, variable)
    return gradient.detach()


def main() -> None:
    rng = np.random.default_rng(SEED)
    torch.manual_seed(SEED)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    embeddings = torch.from_numpy(
        np.asarray(np.load(experiment.EMBEDDING_FILE), dtype=np.float32)
    ).to(device)
    frame = pd.read_parquet(experiment.TRAIN_FILE)
    source_ids, successor_ids = stage2._transition_pairs(frame)
    chosen = rng.choice(len(source_ids), size=PAIR_COUNT, replace=False)
    source_ids = np.ascontiguousarray(source_ids[chosen])
    successor_ids = np.ascontiguousarray(successor_ids[chosen])
    # Repeat the parent's batch-shuffled-source negative construction K times.
    negative_ids = np.stack(
        [source_ids[rng.permutation(len(source_ids))] for _ in range(N_NEGATIVES)],
        axis=1,
    )
    original_negative_ids = np.ascontiguousarray(negative_ids[:, 0])
    source_tensor = torch.from_numpy(source_ids).to(device)
    positive_tensor = torch.from_numpy(successor_ids).to(device)

    model = RQVAE(stage2._tokenizer_config(), in_dim=embeddings.shape[1]).to(device)
    parent_path = Path(experiment.RQVAE_CKPT_PATH)
    checkpoint = torch.load(parent_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    curvature = float(stage2.LAYER_CURVATURES[0])

    with torch.no_grad():
        anchor = model.encoder(embeddings[source_tensor])
        positive = model.encoder(embeddings[positive_tensor])
        negatives = model.encoder(
            embeddings[torch.from_numpy(negative_ids.reshape(-1)).to(device)]
        ).reshape(PAIR_COUNT, N_NEGATIVES, -1)
        original_negative = negatives[:, 0]

    # Preserve the exact pair populations used by iters121-125: active means
    # the original sampled negative violates the parent's hinge at that snapshot.
    snapshot_paths = sorted(
        TRAJECTORY_DIR.glob("step_*.pth"), key=lambda path: int(path.stem.split("_")[1])
    )
    if not snapshot_paths:
        raise FileNotFoundError(f"No trajectory checkpoints found in {TRAJECTORY_DIR}")
    snapshot_paths.append(parent_path)
    active_columns = []
    for path in snapshot_paths:
        snapshot = RQVAE(stage2._tokenizer_config(), in_dim=embeddings.shape[1]).to(device)
        snapshot.load_state_dict(
            torch.load(path, map_location=device, weights_only=False)["state_dict"],
            strict=True,
        )
        snapshot.eval()
        with torch.no_grad():
            a = snapshot.encoder(embeddings[source_tensor])
            p = snapshot.encoder(embeddings[positive_tensor])
            n = snapshot.encoder(embeddings[torch.from_numpy(original_negative_ids).to(device)])
            active = (
                _poincare_distance_tangent_pairs(a, n, curvature)
                - _poincare_distance_tangent_pairs(a, p, curvature)
            ) < stage2.BEHAVIOUR_MARGIN
            active_columns.append(active.cpu().numpy())
        del snapshot
        if device.type == "cuda":
            torch.cuda.empty_cache()
    active_history = np.stack(active_columns, axis=1)
    populations = {
        "persistent": active_history.all(axis=1),
        "rotating": ~(active_history.all(axis=1) | (~active_history).all(axis=1)),
        "original_never": (~active_history).all(axis=1),
    }

    base_original = np.empty(PAIR_COUNT)
    base_hard = np.empty(PAIR_COUNT)
    delta_pair_original = np.empty(PAIR_COUNT)
    delta_pair_hard = np.empty(PAIR_COUNT)
    delta_multi_original = np.empty(PAIR_COUNT)
    delta_multi_hard = np.empty(PAIR_COUNT)
    delta_distance_pair_original = np.empty(PAIR_COUNT)
    delta_distance_pair_hard = np.empty(PAIR_COUNT)
    delta_distance_multi_original = np.empty(PAIR_COUNT)
    delta_distance_multi_hard = np.empty(PAIR_COUNT)
    hard_violation = np.empty(PAIR_COUNT, dtype=bool)

    for start in range(0, PAIR_COUNT, 1024):
        stop = min(start + 1024, PAIR_COUNT)
        a, p = anchor[start:stop], positive[start:stop]
        ns = negatives[start:stop]
        n0 = original_negative[start:stop]
        d_pos = _poincare_distance_tangent_pairs(a, p, curvature)
        expanded = a.unsqueeze(1).expand(-1, N_NEGATIVES, -1)
        d_all = _poincare_distance_tangent_pairs(expanded, ns, curvature)
        hard_violation[start:stop] = (
            d_pos + stage2.BEHAVIOUR_MARGIN >= d_all.min(dim=1).values
        ).cpu().numpy()

        g_pair = unit(pairwise_gradient(a, p, n0, curvature))
        g_multi = unit(multi_hinge_gradient(a, p, ns, curvature))
        moved_pair = unit(a - STEP_SIZE * g_pair)
        moved_multi = unit(a - STEP_SIZE * g_multi)

        base_o = angular_margin(a, p, n0)
        base_h = angular_hard_margin(a, p, ns, curvature)
        pair_o = angular_margin(moved_pair, p, n0)
        pair_h = angular_hard_margin(moved_pair, p, ns, curvature)
        multi_o = angular_margin(moved_multi, p, n0)
        multi_h = angular_hard_margin(moved_multi, p, ns, curvature)
        base_d_o, base_d_h = distance_margins(a, p, n0, ns, curvature)
        pair_d_o, pair_d_h = distance_margins(moved_pair, p, n0, ns, curvature)
        multi_d_o, multi_d_h = distance_margins(moved_multi, p, n0, ns, curvature)

        base_original[start:stop] = base_o.cpu().numpy()
        base_hard[start:stop] = base_h.cpu().numpy()
        delta_pair_original[start:stop] = (pair_o - base_o).cpu().numpy()
        delta_pair_hard[start:stop] = (pair_h - base_h).cpu().numpy()
        delta_multi_original[start:stop] = (multi_o - base_o).cpu().numpy()
        delta_multi_hard[start:stop] = (multi_h - base_h).cpu().numpy()
        delta_distance_pair_original[start:stop] = (pair_d_o - base_d_o).cpu().numpy()
        delta_distance_pair_hard[start:stop] = (pair_d_h - base_d_h).cpu().numpy()
        delta_distance_multi_original[start:stop] = (multi_d_o - base_d_o).cpu().numpy()
        delta_distance_multi_hard[start:stop] = (multi_d_h - base_d_h).cpu().numpy()

    print(f"parent checkpoint: {parent_path}")
    print(
        f"pairs={PAIR_COUNT} K={N_NEGATIVES} margin={stage2.BEHAVIOUR_MARGIN} "
        f"equal unit step={STEP_SIZE} snapshots={len(snapshot_paths)}"
    )
    print(
        "populations:",
        ", ".join(f"{name}={int(mask.sum())}" for name, mask in populations.items()),
    )
    never = populations["original_never"]
    print(
        "P(K=16 hyperbolic margin violation | original never) = "
        f"{hard_violation[never].mean():.4%}"
    )
    print(
        "angular margins are cos(A,B+) - cos(A,X-); "
        "hard negative is min hyperbolic distance"
    )
    print(
        "population          n   base orig  base hard  pair dOrig  pair dHard  "
        "multi dOrig  multi dHard"
    )
    for name, mask in populations.items():
        if not mask.any():
            continue
        means = (
            base_original[mask].mean(),
            base_hard[mask].mean(),
            delta_pair_original[mask].mean(),
            delta_pair_hard[mask].mean(),
            delta_multi_original[mask].mean(),
            delta_multi_hard[mask].mean(),
        )
        print(
            f"{name:<18}{int(mask.sum()):>6}"
            + "".join(f"{value:>12.6f}" for value in means)
        )
    print("hyperbolic distance-margin changes (d_neg - d_pos):")
    print("population          pair original pair hard  multi original multi hard")
    for name, mask in populations.items():
        if not mask.any():
            continue
        values = (
            delta_distance_pair_original[mask].mean(),
            delta_distance_pair_hard[mask].mean(),
            delta_distance_multi_original[mask].mean(),
            delta_distance_multi_hard[mask].mean(),
        )
        print(f"{name:<18}" + "".join(f"{value:>15.6f}" for value in values))

    persistent = populations["persistent"]
    print(
        "persistent criteria: multi-hinge Δangular-hard > pairwise Δangular-hard, "
        "and multi-hinge Δangular-original is not worse"
    )
    print(
        "criteria pass =",
        bool(
            delta_multi_hard[persistent].mean() > delta_pair_hard[persistent].mean()
            and delta_multi_original[persistent].mean()
            >= delta_pair_original[persistent].mean() - 1e-6
        ),
    )


if __name__ == "__main__":
    main()
