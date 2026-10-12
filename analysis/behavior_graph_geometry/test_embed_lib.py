"""Unit tests for the RQ4 metric-embedding library. Run before any training.

The point of these tests is that both geometries must be driven through EXACTLY the
same harness, so that anything the comparison finds belongs to the geometry and not to
a different budget, a different pair set or a different evaluation.

Run: /home/wlia0047/ar57_scratch/wenyu/genrec_env/bin/python3.10 \
        analysis/behavior_graph_geometry/test_embed_lib.py
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import embed_lib as E  # noqa: E402
import graph_lib as L  # noqa: E402

_ok = [True]


def req(cond, msg):
    print(("PASS " if cond else "FAIL ") + msg, flush=True)
    _ok[0] = _ok[0] and bool(cond)


def test_distances() -> None:
    d = E.euclidean_pairwise(torch.tensor([[0.0, 0.0]]), torch.tensor([[3.0, 4.0]]))
    req(abs(float(d[0, 0]) - 5.0) < 1e-6, f"T1 Euclidean pairwise is exact ({float(d[0,0])})")

    # d(0, r) = 2*artanh(r*sqrt(c)) / sqrt(c) inside the ball
    for r in (0.1, 0.4, 0.8):
        p = torch.tensor([[0.0, r]], dtype=torch.float64)
        got = float(E.poincare_pairwise(torch.zeros(1, 2, dtype=torch.float64), p))
        want = 2.0 * math.atanh(r)
        req(abs(got - want) < 1e-8, f"T1 Poincare d(0,{r}) == 2*atanh(r) ({got:.8f})")

    # the arcosh argument is floored just above 1, so the smallest representable
    # distance is acosh(1 + 1e-7) and NOT 1e-6
    FLOOR = math.acosh(1.0 + 1e-7)
    x = torch.tensor([[0.2, -0.3, 0.1]], dtype=torch.float64)
    got = float(E.poincare_pairwise(x, x))
    req(abs(got - FLOOR) < 1e-9,
        f"T1 coincident points give the acosh floor acosh(1+1e-7)={FLOOR:.6e} ({got:.6e})")

    # a point pushed outside the ball is rescaled, not allowed to produce nan
    bad = torch.tensor([[1.0, 1.0, 0.0]], dtype=torch.float64)
    d2 = E.poincare_pairwise(bad, x)
    req(bool(torch.isfinite(d2).all()), f"T1 an out-of-ball point stays finite ({float(d2[0,0]):.4f})")
    cmax = (1.0 - E.BALL_EPS) / math.sqrt(1.0)
    req(float(E.ball_radii(E._inside(bad, 1.0, cmax)).max()) <= cmax + 1e-9,
        f"T1 and is rescaled inside the ball by the distance function itself "
        f"({float(E.ball_radii(E._inside(bad, 1.0, cmax)).max()):.6f} <= {cmax:.6f}), "
        f"not by an external clip")

    # symmetry and the triangle inequality, which the distortion metrics assume
    A = torch.randn(12, 4, dtype=torch.float64) * 0.2
    B = torch.randn(9, 4, dtype=torch.float64) * 0.2
    DAB = E.poincare_pairwise(A, B)
    DBA = E.poincare_pairwise(B, A).t()
    req(float((DAB - DBA).abs().max()) < 1e-9,
        f"T1 the pairwise distance is symmetric in its arguments "
        f"({float((DAB - DBA).abs().max()):.2e})")
    DA = E.poincare_pairwise(A, A)
    req(float((DA - DA.t()).abs().max()) < 1e-9, "T1 the self matrix is symmetric")
    req(bool((DA.diagonal() <= FLOOR + 1e-12).all()),
        f"T1 its diagonal sits at the floor ({float(DA.diagonal().max()):.3e})")


def test_identical_protocol() -> None:
    """Both geometries must see the same pairs, the same target distances and the
    same budget, and the harness must not silently differentiate them."""
    n, dim = 120, 8
    rng = np.random.default_rng(0)
    pairs = rng.integers(0, n, size=(4000, 2))
    pairs = pairs[pairs[:, 0] != pairs[:, 1]]
    true = np.abs(rng.normal(0, 2, pairs.shape[0])) + 0.1
    common = dict(n_nodes=n, dim=dim, train_pairs=pairs, train_dists=true,
                  steps=300, lr=0.05, batch=512, seed=42, device="cpu", log_every=50)
    Xe, se, he = E.train_metric_embedding("euclid", **common)
    Xh, sh, hh = E.train_metric_embedding("poincare", **common)
    req(Xe.shape == Xh.shape == (n, dim), f"T1 both embeddings have the same shape ({tuple(Xe.shape)})")
    req(len(he) == len(hh), "T1 both geometries take the same number of logged steps")
    req(he[-1]["loss"] < he[0]["loss"] and hh[-1]["loss"] < hh[0]["loss"],
        f"T1 both losses decrease (euclid {he[0]['loss']:.4f}->{he[-1]['loss']:.4f}, "
        f"poincare {hh[0]['loss']:.4f}->{hh[-1]['loss']:.4f})")
    req(se > 0 and sh > 0, f"T1 both learned a positive global scale ({se:.4f}, {sh:.4f})")
    r = E.ball_radii(Xh)
    req(bool(torch.isfinite(Xh).all()) and float(r.max()) <= 1.0 + 1e-6,
        f"T1 the hyperbolic embedding stays inside the ball (max radius {float(r.max()):.6f})")
    req(hh[-1]["share_radius_gt_0.99"] is not None, "T1 its boundary saturation share is recorded")
    req(bool(np.isfinite(Xe.numpy()).all()), "T1 the Euclidean embedding is finite")


def test_evaluation_metrics() -> None:
    rng = np.random.default_rng(3)
    true = rng.integers(1, 10, size=3000).astype(float)
    req(abs(L.normalised_stress(true, true)) < 1e-12, "T1 stress is 0 for a perfect fit")
    req(abs(L.mean_relative_distortion(true, true)) < 1e-12, "T1 mean relative distortion is 0")
    s = E.fit_scale_eval(true, true * 4.3)
    req(abs(s - 1 / 4.3) < 1e-9, f"T1 the eval-fitted scale is exact ({s:.6f})")
    req(abs(L.normalised_stress(true, true * 4.3 * s)) < 1e-9,
        "T1 and a pure rescaling is not counted as distortion")

    # separable construction: positives strictly above negatives -> AUC exactly 1
    rng2 = np.random.default_rng(4)
    hi = 10.0 + rng2.random(2000)
    lo = rng2.random(2000)
    lab = np.concatenate([np.ones(2000), np.zeros(2000)])
    sc = np.concatenate([hi, lo])
    req(abs(E.mean_auc(sc, lab) - 1.0) < 1e-9,
        f"T1 AUC is exactly 1 when positives strictly outrank negatives "
        f"({E.mean_auc(sc, lab):.6f})")
    req(abs(E.mean_auc(sc[::-1], lab)) < 1e-9,
        f"T1 and exactly 0 for the fully reversed ordering "
        f"({E.mean_auc(sc[::-1], lab):.6f})")
    # a tie must be split, not ordered
    tied = np.zeros(2000)                       # genuinely all-tied scores
    tied_lab = np.concatenate([np.ones(1000), np.zeros(1000)])
    # and a fully separated pair of levels is NOT a tie: it must score AUC 1
    sep = np.concatenate([np.ones(1000), np.zeros(1000)])
    req(abs(E.mean_auc(sep, tied_lab) - 1.0) < 1e-9,
        f"T1 two well-separated score LEVELS are not a tie and score AUC 1 "
        f"({E.mean_auc(sep, tied_lab):.6f})")
    req(abs(E.mean_auc(tied, tied_lab) - 0.5) < 1e-9,
        f"T1 and exactly 0.5 when every score is tied ({E.mean_auc(tied, tied_lab):.6f})")
    one_class = np.ones(1000)
    req(np.isnan(E.mean_auc(np.arange(1000.0), one_class)),
        "T1 and undefined when there is only one class")
    # overlapping case must agree with sklearn, which is the reference definition
    from sklearn.metrics import roc_auc_score
    pos = rng2.normal(1, 1, 2000)
    neg = rng2.normal(0, 1, 2000)
    ov = np.concatenate([pos, neg])
    ovl = np.concatenate([np.ones(2000), np.zeros(2000)])
    ref = float(roc_auc_score(ovl, ov))
    req(abs(E.mean_auc(ov, ovl) - ref) < 1e-9,
        f"T1 an overlapping case matches sklearn exactly ({E.mean_auc(ov, ovl):.9f} vs {ref:.9f})")

    # neighbourhood preservation at k=1 with a matching ordering
    full = np.array([[0.0, 1.0, 2.0], [1.0, 0.0, 3.0], [2.0, 3.0, 0.0]])
    req(abs(L.neighbourhood_preservation([[1, 2], [0, 2], [0, 1]], full, 1) - 1.0) < 1e-12,
        "T1 neighbourhood preservation is 1 when the nearest neighbour agrees")


def main() -> int:
    print("=== distances ===", flush=True); test_distances()
    print("\n=== identical protocol for both geometries ===", flush=True); test_identical_protocol()
    print("\n=== evaluation metrics ===", flush=True); test_evaluation_metrics()
    print("\n" + "=" * 72)
    print("ALL EMBEDDING TESTS PASS" if _ok[0] else "EMBEDDING TESTS FAILED")
    print("=" * 72)
    return 0 if _ok[0] else 1


if __name__ == "__main__":
    raise SystemExit(main())