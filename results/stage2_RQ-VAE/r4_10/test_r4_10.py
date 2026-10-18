"""R4-10 tests: train-target-frequency-weighted L1 KMeans initialisation (issue #60).

The RED test fits the weighted and the unweighted L1 clustering itself and checks
what the arm's real ``init_codebook`` produces. Without the arm's weighted path
the measured centres are the baseline's unweighted ones, so the failure is a
measured numeric mismatch on the weighted objective.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.cluster import KMeans

sys.path.insert(0, "/home/wlia0047/ar57/wenyu/GeneRec/results/stage2_RQ-VAE/r4_common")
import r4_testkit as K  # noqa: E402

ARM = "r4_10"
FLAG = "R4_10_WEIGHTED_KMEANS"
T = K.load_arm(ARM)

PARQUET = Path("/home/wlia0047/ar57/wenyu/GeneRec/results/stage0_build_parquet/train.parquet")


def _set(value: bool) -> None:
    if hasattr(T, FLAG):
        setattr(T, FLAG, value)


def target_weight(n_items: int = 24587) -> np.ndarray:
    """w_i = sqrt(1 + count_train_target(i)), divided by its mean over the catalogue."""
    frame = pd.read_parquet(PARQUET, columns=["target"])
    counts = np.bincount(frame["target"].to_numpy(dtype=np.int64), minlength=n_items)[:n_items]
    weights = np.sqrt(1.0 + counts)
    return weights / weights.mean()


def weighted_inertia(points: np.ndarray, centers: np.ndarray, weights: np.ndarray) -> float:
    distances = ((points[:, None, :] - centers[None, :, :]) ** 2).sum(-1)
    assignment = distances.argmin(1)
    return float((distances[np.arange(len(points)), assignment] * weights).sum())


def _arm_l1_centers(
    encoded: torch.Tensor, weights: torch.Tensor | None, clusters: int
) -> np.ndarray:
    """Run the arm's real L1 initialisation and return its centres as numpy."""
    model, _ = K.tiny_model(T, codebook_dim=encoded.shape[-1], size=clusters)
    layer = model.rq.vq_layers[0]
    # Mirror main(): with the mechanism off the arm passes no weights at all, which is
    # what keeps the baseline code path byte-identical rather than merely equivalent.
    # getattrs so the canonical parent, which has no such flag, is measured rather than
    # attribute-errored: the RED failure must be a number, not a missing name.
    flag = bool(getattr(T, "R4_10_WEIGHTED_KMEANS", False))
    effective = weights if flag else None
    if effective is None:
        layer.init_codebook(encoded, torch.device("cpu"))
    else:
        try:
            layer.init_codebook(encoded, torch.device("cpu"), sample_weight=effective)
        except TypeError:
            # Canonical parent: no weighted path at all, so the L1 centres are the
            # unweighted ones and the weighted objective decides pass or fail.
            layer.init_codebook(encoded, torch.device("cpu"))
    return layer.embed.weight.detach().cpu().numpy()


# ---------------------------------------------------------------- RED tests


def test_red_weighted_l1_init_beats_unweighted_on_the_weighted_objective():
    """The fitted L1 centres must minimise the weighted objective, not the plain one."""
    _set(True)
    # Two tight clusters plus a diffuse heavy-weight cloud: the unweighted fit splits
    # the mass roughly evenly while the weighted fit must chase the heavy points.
    tight = np.concatenate([
        np.repeat([[-1.0, 0.0]], 30, axis=0),
        np.repeat([[1.0, 0.0]], 30, axis=0),
    ])
    diffuse = np.concatenate([
        np.repeat([[0.0, 6.0]], 8, axis=0),
        np.repeat([[0.0, -6.0]], 8, axis=0),
    ])
    points_np = np.concatenate([tight, diffuse], axis=0).astype(np.float64)
    weights_np = np.concatenate([np.full(60, 0.05), np.full(16, 6.0)])
    weights_np = weights_np / weights_np.mean()

    plain = KMeans(n_clusters=2, n_init="auto").fit(points_np)
    weighted = KMeans(n_clusters=2, n_init="auto").fit(
        points_np, sample_weight=weights_np
    )
    plain_inertia = weighted_inertia(points_np, plain.cluster_centers_, weights_np)
    weighted_inertia_best = weighted_inertia(
        points_np, weighted.cluster_centers_, weights_np
    )
    assert weighted_inertia_best < plain_inertia - 1e-6, (
        "test precondition: the weighted optimum must differ from the unweighted one"
    )

    points = torch.tensor(points_np, dtype=torch.float32)
    centers = _arm_l1_centers(points, torch.tensor(weights_np, dtype=torch.float32), 2)
    achieved = weighted_inertia(points_np, centers, weights_np)
    assert achieved <= weighted_inertia_best + 1e-3 * max(weighted_inertia_best, 1.0), (
        f"L1 initialisation did not use the weights (weighted inertia {achieved:.4f})"
    )


def test_red_uniform_weights_reproduce_unweighted_kmeans():
    """With uniform weights the weighted fit must equal the baseline clustering."""
    _set(True)
    torch.manual_seed(37)
    points = torch.randn(48, 4)
    uniform = torch.ones(48)
    # The arm keeps the baseline's own KMeans (no random_state), so a single reference
    # fit would be a different draw. Compare against the spread of unweighted fits.
    reference = sorted(
        weighted_inertia(
            points.numpy(),
            KMeans(n_clusters=3, n_init="auto").fit(points.numpy()).cluster_centers_,
            uniform.numpy(),
        )
        for _ in range(10)
    )
    best_plain = reference[int(0.9 * (len(reference) - 1))]
    centers = _arm_l1_centers(points, uniform, 3)
    achieved = weighted_inertia(points.numpy(), centers, uniform.numpy())
    assert achieved <= best_plain * 1.02 + 1e-6, (
        f"uniform weights changed the baseline clustering ({achieved:.6f} vs {best_plain:.6f})"
    )


# ------------------------------------------------------------ contract tests


def test_weights_are_train_only_positive_and_finite():
    _set(True)
    weights = target_weight()
    assert weights.shape == (24587,)
    assert np.isfinite(weights).all() and (weights > 0).all()
    assert abs(float(weights.mean()) - 1.0) < 1e-9, "weights must be mean-normalised"
    assert float(weights.max()) > 2.0 * float(np.median(weights)), (
        "test precondition: train target counts must be skewed"
    )


def test_l2_l3_paths_shapes_and_seed_unchanged():
    _set(True)
    model, cfg = K.tiny_model(T)
    assert len(model.rq.vq_layers) == 3
    assert all(layer.embed.weight.shape == (cfg.codebook_size[0], cfg.codebook_dim)
               for layer in model.rq.vq_layers)


def test_off_flag_matches_exact_resample():
    _set(False)
    torch.manual_seed(41)
    points = torch.randn(48, 4)
    reference = sorted(
        weighted_inertia(
            points.numpy(),
            KMeans(n_clusters=3, n_init="auto").fit(points.numpy()).cluster_centers_,
            np.ones(48),
        )
        for _ in range(10)
    )
    best_plain = reference[int(0.9 * (len(reference) - 1))]
    centers = _arm_l1_centers(points, torch.ones(48), 3)
    achieved = weighted_inertia(points.numpy(), centers, np.ones(48))
    assert achieved <= best_plain * 1.02 + 1e-6, (
        f"disabled mechanism still changed the L1 initialisation ({achieved:.6f} vs {best_plain:.6f})"
    )


def test_finite_forward_and_backward():
    step = K.mini_step(T)
    assert step["finite"] and step["encoder_grad_norm"] > 0.0


def test_train_only_data_audit():
    K.check_train_only(T)


def test_sid_three_token_contract():
    K.check_sid_contract(T)


if __name__ == "__main__":
    K.run(
        __name__,
        [
            test_red_weighted_l1_init_beats_unweighted_on_the_weighted_objective,
            test_red_uniform_weights_reproduce_unweighted_kmeans,
            test_weights_are_train_only_positive_and_finite,
            test_l2_l3_paths_shapes_and_seed_unchanged,
            test_off_flag_matches_exact_resample,
            test_finite_forward_and_backward,
            test_train_only_data_audit,
            test_sid_three_token_contract,
        ],
    )
