"""Issue #111: the cheap CPU sandbox behind subspace_geometry_review.json.

Synthetic mathematics only. It implements the Poincare-ball operators the three candidates
declare (exp/log at the origin, the geodesic distance, parallel transport, the metric tensor)
and measures whether the declared invariants and the H1-H7 feasibility conditions actually
hold. No training, no checkpoint, no GPU, no candidate code: this is the verifier's own
reference implementation of the declared mathematics.

Run:  python3 results/hyperbolic_research_runs/subspace_111/sandbox/h1_h7_feasibility.py
Writes: .../sandbox/results.json
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import torch

torch.set_default_dtype(torch.float64)
OUT = Path(__file__).resolve().parent
N_ITEMS = 512
N_CODES = 256
DIM = 16


# ----------------------------------------------------------------------------- operators
def expmap0(v: torch.Tensor, c: float) -> torch.Tensor:
    """exp at the origin of the Poincare ball of curvature -c."""
    norm = torch.linalg.vector_norm(v, dim=-1, keepdim=True)
    root = math.sqrt(c)
    scale = torch.where(norm > 0, torch.tanh(root * norm) / (root * norm), torch.ones_like(norm))
    return scale * v


def logmap0(x: torch.Tensor, c: float) -> torch.Tensor:
    norm = torch.linalg.vector_norm(x, dim=-1, keepdim=True)
    root = math.sqrt(c)
    scale = torch.where(norm > 0, torch.atanh(torch.clamp(root * norm, max=1 - 1e-12)) / (root * norm),
                        torch.ones_like(norm))
    return scale * x


def ball_radius(x: torch.Tensor, c: float) -> torch.Tensor:
    return torch.linalg.vector_norm(x, dim=-1)


def ponder(x: torch.Tensor, y: torch.Tensor, c: float) -> torch.Tensor:
    """d_c(x, y), the Poincare geodesic distance."""
    nx = torch.linalg.vector_norm(x, dim=-1) ** 2
    ny = torch.linalg.vector_norm(y, dim=-1) ** 2
    diff = torch.linalg.vector_norm(x - y, dim=-1) ** 2
    arg = 1 + 2 * c * diff / ((1 - c * nx).clamp_min(1e-12) * (1 - c * ny).clamp_min(1e-12))
    return torch.acosh(arg.clamp_min(1.0)) / math.sqrt(c)


def mobius_add(x: torch.Tensor, y: torch.Tensor, c: float) -> torch.Tensor:
    x2 = (x * x).sum(-1, keepdim=True)
    y2 = (y * y).sum(-1, keepdim=True)
    xy = (x * y).sum(-1, keepdim=True)
    num = (1 + 2 * c * xy + c * y2) * x + (1 - c * x2) * y
    den = 1 + 2 * c * xy + c * c * x2 * y2
    return num / den.clamp_min(1e-12)


def gyration(a: torch.Tensor, b: torch.Tensor, v: torch.Tensor, c: float) -> torch.Tensor:
    """gyr[a, b] v = -(a+b) + (a + (b + v)) under Mobius addition."""
    return mobius_add(-mobius_add(a, b, c), mobius_add(a, mobius_add(b, v, c), c), c)


def conformal(x: torch.Tensor, c: float) -> torch.Tensor:
    """lambda_x = 2 / (1 - c||x||^2), the scale of g_x = lambda^2 I."""
    return 2.0 / (1.0 - c * (x * x).sum(-1)).clamp_min(1e-12)


def parallel_transport(v: torch.Tensor, x: torch.Tensor, y: torch.Tensor, c: float) -> torch.Tensor:
    """PT_{x->y}(v) for v in T_x, along the unique geodesic from x to y."""
    lam_x = conformal(x, c)
    lam_y = conformal(y, c)
    moved = gyration(y, -x, v, c)
    return (lam_x / lam_y).unsqueeze(-1) * moved


# ----------------------------------------------------------------------------- fixtures
def fixture(seed: int = 20261023):
    g = torch.Generator().manual_seed(seed)
    # a spread of latent scales: small, unit, and large enough to test the boundary
    scales = torch.tensor([0.25, 0.5, 1.0, 2.0]).repeat_interleave(N_ITEMS // 4)
    z = torch.randn(N_ITEMS, 2 * DIM, generator=g) * scales.unsqueeze(-1)
    codes = torch.randn(N_CODES, 2 * DIM, generator=g) * 0.5
    return z, codes


def product_assign(z, codes, c: float, s: float = 0.5):
    """The P1 assignment: argmin over the sum of the two factor geodesic distances squared."""
    zs, zb = z[:, :DIM], z[:, DIM:]
    cs, cb = codes[:, :DIM], codes[:, DIM:]
    xs, xb = expmap0(s * zs, c), expmap0(s * zb, c)
    a, b = expmap0(cs, c), expmap0(cb, c)
    ds = torch.cdist(xs, a)  # euclidean, used only to rank candidates cheaply
    db = torch.cdist(xb, b)
    # the geodesic distance is a monotone function of the euclidean one only at fixed endpoints,
    # so the distances are computed exactly, in chunks of codewords
    def exact(x, y):
        out = torch.empty(x.shape[0], y.shape[0])
        for j in range(y.shape[0]):
            out[:, j] = ponder(x, y[j].expand_as(x), c)
        return out
    ds = exact(xs, a)
    db = exact(xb, b)
    joint = (ds ** 2 + db ** 2).argmin(dim=-1)
    s_only = (ds ** 2).argmin(dim=-1)
    b_only = (db ** 2).argmin(dim=-1)
    return joint, s_only, b_only, (ds, db)


def euclidean_assign(z, codes):
    return torch.cdist(z, codes).argmin(dim=-1)


# ----------------------------------------------------------------------------- checks
def checks() -> dict:
    z, codes = fixture()
    res: dict = {"fixture": {"n_items": N_ITEMS, "n_codes": N_CODES, "dim_per_factor": DIM,
                             "item_latent_scales": [0.25, 0.5, 1.0, 2.0],
                             "seed": 20261023}}

    # ---- H1 domain: every ball point satisfies c||x||^2 < 1 by construction
    c = 1.0
    xs = expmap0(0.5 * z[:, :DIM], c)
    a = expmap0(codes[:, :DIM], c)
    max_ball = float((c * (xs * xs).sum(-1)).max())
    max_code_ball = float((c * (a * a).sum(-1)).max())
    res["H1_domain"] = {
        "max_c_squared_norm_items": max_ball,
        "max_c_squared_norm_codes": max_code_ball,
        "inside_ball_by_construction": bool(max(max_ball, max_code_ball) < 1.0),
        "max_ball_radius_items": float(ball_radius(xs, c).max()),
        "boundary_margin": 1.0 - float(ball_radius(xs, c).max()),
        "note": "exp_0 maps every tangent vector strictly inside the ball, so no projection is needed",
    }

    # ---- H7 oracle identities
    v = torch.randn(64, DIM, generator=torch.Generator().manual_seed(7))
    for cc in (1.0, 2.0, 0.25):
        back = logmap0(expmap0(v, cc), cc)
        res.setdefault("H7_oracle", {})[f"exp_log_inverse_c{cc}"] = float((back - v).abs().max())
    x = expmap0(0.5 * torch.randn(16, DIM, generator=torch.Generator().manual_seed(11)), 1.0)
    y = expmap0(0.5 * torch.randn(16, DIM, generator=torch.Generator().manual_seed(12)), 1.0)
    dxy = ponder(x, y, 1.0)
    dyx = ponder(y, x, 1.0)
    dxx = ponder(x, x, 1.0)
    res["H7_oracle"]["distance_symmetry"] = float((dxy - dyx).abs().max())
    res["H7_oracle"]["distance_self_zero"] = float(dxx.abs().max())
    # triangle inequality on 16 triples
    zt = expmap0(0.5 * torch.randn(16, DIM, generator=torch.Generator().manual_seed(13)), 1.0)
    lhs = ponder(x, zt, 1.0)
    rhs = dxy + ponder(y, zt, 1.0)
    res["H7_oracle"]["triangle_violations"] = int((lhs > rhs + 1e-9).sum())

    # ---- P2: parallel transport
    t = torch.randn(16, DIM, generator=torch.Generator().manual_seed(21)) * 0.3
    transported = parallel_transport(t, x, y, 1.0)
    norm_x = conformal(x, 1.0) * torch.linalg.vector_norm(t, dim=-1)
    norm_y = conformal(y, 1.0) * torch.linalg.vector_norm(transported, dim=-1)
    back = parallel_transport(transported, y, x, 1.0)
    at_origin = parallel_transport(t, x, torch.zeros_like(x), 1.0)
    ambient_identity = parallel_transport(t, torch.zeros_like(x), torch.zeros_like(x), 1.0)
    res["P2_transport"] = {
        "isometry_max_deviation": float((norm_x - norm_y).abs().max()),
        "roundtrip_max_deviation": float((back - t).abs().max()),
        "identity_when_both_endpoints_origin": float((ambient_identity - t).abs().max()),
        "transport_to_origin_is_not_identity_max_dev": float((at_origin - t).abs().max()),
        "norm_ratio_transport_vs_ambient_mean": float((torch.linalg.vector_norm(transported, dim=-1)
                                                      / torch.linalg.vector_norm(t, dim=-1)).mean()),
    }

    # ---- P2/H3: hold the ball points fixed and vary the curvature: the transport deviation
    # from the identity must grow with c and vanish as c -> 0 (the Euclidean limit).
    gen2 = torch.Generator().manual_seed(41)
    x0 = torch.randn(16, DIM, generator=gen2) * 0.15
    y0 = torch.randn(16, DIM, generator=gen2) * 0.15
    t0 = torch.randn(16, DIM, generator=gen2) * 0.3
    limit = {}
    for cc in (0.001, 0.25, 1.0, 4.0):
        # hold the NORMALISED radius sqrt(c)||x|| fixed: the only way to sweep the curvature
        # without also sweeping the domain. The tangent vector is rescaled with 1/sqrt(c).
        root = math.sqrt(cc)
        xx, yy, tt = x0 / root, y0 / root, t0 / root
        moved = parallel_transport(tt, xx, yy, cc)
        limit[f"c{cc}"] = {
            "relative_deviation_from_identity": float((moved - tt).abs().max() / tt.abs().max()),
            "max_c_squared_norm": float(cc * (xx * xx).sum(-1).max()),
        }
    res["P2_curvature_limit"] = limit
    # the same sweep WITHOUT the rescaling: at c=4 the fixed Euclidean points leave the ball
    res["P2_domain_violation_evidence"] = {
        f"c{cc}": {
            "max_c_squared_norm": float(cc * (x0 * x0).sum(-1).max()),
            "transport_deviation": float((parallel_transport(t0, x0, y0, cc) - t0).abs().max()),
            "finite": bool(torch.isfinite(parallel_transport(t0, x0, y0, cc)).all()),
        } for cc in (1.0, 4.0)
    }

    # ---- P1: does the geodesic assignment differ from the Euclidean one? (H2)
    joint, s_only, b_only, _ = product_assign(z, codes, c)
    euclid = euclidean_assign(z, codes)
    res["H2_essential_computation"] = {
        "geodesic_vs_euclidean_disagreement": float((joint != euclid).double().mean()),
        "joint_vs_semantic_only_agreement": float((joint == s_only).double().mean()),
        "joint_vs_behaviour_only_agreement": float((joint == b_only).double().mean()),
        "factors_disagree_share": float((s_only != b_only).double().mean()),
        "distinct_joint_codes": int(joint.unique().numel()),
        "note": ("a disagreement rate above zero is what makes the curved metric the mechanism; "
                 "a joint-vs-single-factor agreement of 1.0 is the declared inertness failure"),
    }

    # ---- H3: curvature dependence
    base = joint
    rows = {}
    for cc in (0.1, 0.25, 1.0, 4.0):
        j, _, _, _ = product_assign(z, codes, cc)
        rows[f"c{cc}"] = {
            "disagreement_vs_c1": float((j != base).double().mean()),
            "disagreement_vs_euclidean": float((j != euclid).double().mean()),
        }
    res["H3_curvature_dependence"] = rows

    # ---- H5: numerical stability at coincident / near-boundary points
    zero = torch.zeros(4, DIM)
    near = torch.full((4, DIM), 1.0 / math.sqrt(DIM)) * 0.999999
    res["H5_numerical"] = {
        "distance_zero_zero": float(ponder(zero, zero, 1.0).abs().max()),
        "distance_zero_near_boundary": float(ponder(zero, near, 1.0).max()),
        "distance_near_near": float(ponder(near, near, 1.0).abs().max()),
        "any_non_finite": bool(not torch.isfinite(ponder(zero, near, 1.0)).all()),
        "max_c_squared_norm_near_boundary": float((1.0 * (near * near).sum(-1)).max()),
    }

    # ---- P3: the Riemannian conflict set
    lam = conformal(expmap0(codes[:, :DIM], 1.0), 1.0)
    gen = torch.Generator().manual_seed(31)
    g_rec = torch.randn(N_CODES, DIM, generator=gen)
    g_beh = torch.randn(N_CODES, DIM, generator=gen)
    # plant a known conflict for half of the codewords
    g_beh[: N_CODES // 2] = -g_rec[: N_CODES // 2] * 0.5
    dot_e = (g_rec * g_beh).sum(-1)
    dot_r = (lam.unsqueeze(-1) ** 2 * g_rec * g_beh).sum(-1)
    conflict_e = dot_e < 0
    conflict_r = dot_r < 0
    denom = (lam.unsqueeze(-1) ** 2 * g_rec * g_rec).sum(-1) + 1e-12
    proj = g_beh - (dot_r / denom).unsqueeze(-1) * g_rec
    after = (lam.unsqueeze(-1) ** 2 * g_rec * proj).sum(-1)
    res["P3_metric_conflict"] = {
        "metric_min": float(lam.min()),
        "metric_max": float(lam.max()),
        "conflict_set_disagreement": float((conflict_e != conflict_r).double().mean()),
        "conflict_pairs": int(conflict_e.sum()),
        "post_projection_min_inner_product": float(after.min()),
        "curvature_zero_conflict_set_equals_euclidean": bool(
            torch.equal(((torch.full_like(lam, 2.0).unsqueeze(-1) ** 2 * g_rec * g_beh).sum(-1) < 0),
                        conflict_e)),
        "note": ("at c=0 the metric is the identity and the Riemannian inner product equals the "
                 "Euclidean one exactly, so the conflict sets coincide"),
    }
    return res


if __name__ == "__main__":
    out = checks()
    (OUT / "results.json").write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))
