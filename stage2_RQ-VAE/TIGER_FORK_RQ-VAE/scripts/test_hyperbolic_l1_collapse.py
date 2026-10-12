"""Unit tests for the hyperbolic L1 quantisation, aimed at the codebook collapse.

Every test is a small, named check with its own PASS/FAIL line. Run with:

  /home/wlia0047/ar57_scratch/wenyu/genrec_env/bin/python3.10 \
      stage2_RQ-VAE/TIGER_FORK_RQ-VAE/scripts/test_hyperbolic_l1_collapse.py

The tests are grouped by the question the programme asks:

  A  is the Poincare distance mathematically correct?
  B  is it numerically stable in float32 where it matters (near-coincident points,
     near-boundary points)?
  C  what is the gradient path? argmin, the clamp, the boundary rescale,
     the straight-through estimator, the codebook term.
  D  is the quantisation objective consistent with the assignment geometry?
  E  is the initialisation consistent with the geometry used in training?
  F  does the hard-negative hinge push a negative OUT of the shared codeword?
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

import hyperbolic as H  # noqa: E402
from model import RQVAE  # noqa: E402
from model.layers import VQLayer  # noqa: E402

torch.manual_seed(0)
DEV = torch.device("cpu")
CURV = 1.0

_RESULTS: list[tuple[str, bool, str]] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    _RESULTS.append((name, bool(cond), detail))
    print(f"{'PASS' if cond else 'FAIL'}  {name}" + (f"   [{detail}]" if detail else ""), flush=True)


def ball_point(r: float, c: float = CURV) -> torch.Tensor:
    return torch.tensor([r], dtype=torch.float64)


# ---------------------------------------------------------------- A: distance
def test_a_distance_math() -> None:
    # A1. d(0, r) == 2 * artanh(r / R) with R = 1/sqrt(c); at c = 1 that is 2*artanh(r).
    for r in (0.01, 0.2, 0.7, 0.9):
        got = float(H.poincare_distance(torch.zeros(1, dtype=torch.float64),
                                         ball_point(r), CURV))
        want = 2.0 * torch.atanh(torch.tensor(r, dtype=torch.float64))
        check(f"A1 d(0,{r}) == 2*atanh(r)", abs(got - float(want)) < 1e-9,
              f"got {got:.10f} want {float(want):.10f}")

    # A2. symmetry, and the exact floor the acosh clamp imposes on the diagonal.
    x = torch.tensor([[0.3, -0.2, 0.1]], dtype=torch.float64)
    y = torch.tensor([[-0.1, 0.4, -0.3]], dtype=torch.float64)
    dxy = float(H.poincare_distance(x, y, CURV))
    dyx = float(H.poincare_distance(y, x, CURV))
    check("A2 distance symmetric", abs(dxy - dyx) < 1e-12, f"{dxy:.12f} vs {dyx:.12f}")
    floor = float(H.poincare_distance(x, x, CURV))
    want_floor = float(torch.acosh(torch.tensor(1.0 + H.ACOSH_EPS, dtype=torch.float64)))
    check("A2 the diagonal distance is floored at acosh(1+ACOSH_EPS), not 0",
          abs(floor - want_floor) < 1e-12, f"d(x,x) = {floor:.3e}, floor {want_floor:.3e}")

    # A3. the chord closed form itself: d = acosh(1 + 2c|x-y|^2 / ((1-c|x|^2)(1-c|y|^2)))
    nn = lambda t: float(torch.linalg.vector_norm(t))  # noqa: E731
    want = torch.acosh(torch.tensor(
        1.0 + 2 * CURV * nn(x - y) ** 2 / ((1 - CURV * nn(x) ** 2) * (1 - CURV * nn(y) ** 2)),
        dtype=torch.float64))
    check("A3 matches the closed form", abs(dxy - float(want)) < 1e-9,
          f"{dxy:.10f} vs {float(want):.10f}")

    # A4. exponential map and distance agree: the geodesic from the origin is
    # 2*artanh(||x||), so d(expmap0(u), 0) == 2*||u|| (the tanh cancels the artanh).
    u = torch.tensor([[0.7, -1.3, 2.2]], dtype=torch.float64)
    mapped = H.expmap0(u, CURV)
    got = float(H.poincare_distance(mapped, torch.zeros(1, dtype=torch.float64), CURV))
    want = 2.0 * float(torch.linalg.vector_norm(u))
    check("A4 d(expmap0(u), 0) == 2*||u||", abs(got - want) < 1e-9,
          f"{got:.10f} vs {want:.10f}")

    # A5. the exponential map lands strictly inside the ball.
    big = torch.tensor([[1e3, -2e3, 3e3]], dtype=torch.float64)
    mb = H.expmap0(big, CURV)
    r_map = float(torch.linalg.vector_norm(mb))
    r_in = float(torch.linalg.vector_norm(H._inside(mb, CURV)))
    check("A5 expmap0 saturates to exactly the boundary for large norms",
          abs(r_map - 1.0) < 1e-12, f"||expmap0(u)|| = {r_map:.12f}")
    check("A5 _inside pulls it back strictly inside the ball",
          r_in <= 1.0 - H.BOUNDARY_EPS + 1e-15, f"||_inside|| = {r_in:.12f}")

    # A6. pair form equals the scalar form for every pair.
    A = torch.randn(7, 5, dtype=torch.float64) * 0.2
    B = torch.randn(4, 5, dtype=torch.float64) * 0.2
    P = H.pair_poincare_distance(A, B, CURV)
    ok = all(
        abs(float(P[i, j]) - float(H.poincare_distance(A[i], B[j], CURV))) < 1e-12
        for i in range(7) for j in range(4)
    )
    check("A6 pair_poincare_distance == scalar form", ok and tuple(P.shape) == (7, 4),
          f"shape {tuple(P.shape)}")


# ------------------------------------------------------------- B: float32 traps
def test_b_float32() -> None:
    # B1. The acosh clamp imposes a hard FLOOR on every distance. Below it the
    # distance is indistinguishable from a coincident point and carries no
    # gradient. Measure the floor and check it is dtype-independent.
    floor64 = float(torch.acosh(torch.tensor(1.0 + H.ACOSH_EPS, dtype=torch.float64)))
    x32 = torch.zeros(1, 1, dtype=torch.float32)
    d32 = float(H.poincare_distance(torch.full((1, 1), 1e-8, dtype=torch.float32), x32, CURV))
    d64 = float(H.poincare_distance(torch.full((1, 1), 1e-8, dtype=torch.float64),
                                    torch.zeros(1, 1, dtype=torch.float64), CURV))
    check("B1 the distance has a hard floor (below it points are indistinguishable)",
          floor64 > 0.0, f"floor = {floor64:.3e}")
    check("B1 float32 and float64 agree on the floor (no dtype-specific collapse)",
          abs(d32 - d64) < 0.15 * floor64,
          f"f32 {d32:.3e} vs f64 {d64:.3e}, floor {floor64:.3e}")

    # B2. gradient of acosh at argument 1 is infinite; the clamp removes it but the
    # loss then has NO gradient at all there.
    t = torch.tensor([1.0 + 1e-9, 1.000001, 1.1], dtype=torch.float32, requires_grad=True)
    ac = torch.acosh(t.clamp_min(1.0 + H.ACOSH_EPS))
    ac.sum().backward()
    g0 = float(t.grad[0])
    check("B2 acosh gradient is exactly 0 (not inf) at the clamped point",
          g0 == 0.0, f"grad = {g0:.3e}")

    # B3. points that sit at the ball boundary: is the distance finite, and is the
    # distance still informative (monotone) between two near-boundary points?
    near = torch.tensor([[0.99999, 0.0, 0.0], [0.99999, 1e-5, 0.0]], dtype=torch.float64)
    d2 = H.pair_poincare_distance(near, torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float64), CURV)
    check("B3 near-boundary distances finite", bool(torch.isfinite(d2).all()),
          f"{d2.reshape(-1).tolist()}")
    check("B3 near-boundary distance is monotone in separation",
          float(d2[1]) > float(d2[0]),
          f"{float(d2[0]):.6f} -> {float(d2[1]):.6f}")

    # B4. float32 assignment of a batch: does the geodesic change the argmax set
    # relative to float64, i.e. is the L1 assignment float32-fragile?
    torch.manual_seed(3)
    z = torch.randn(512, 8, dtype=torch.float32) * 0.3
    codes = torch.randn(256, 8, dtype=torch.float32) * 0.3
    a32 = H.pair_poincare_distance(H.encode_ball(z, CURV, 4.0),
                                   H.encode_ball(codes, CURV, 4.0), CURV).argmin(-1)
    a64 = H.pair_poincare_distance(H.encode_ball(z.double(), CURV, 4.0),
                                   H.encode_ball(codes.double(), CURV, 4.0), CURV).argmin(-1)
    check("B4 float32 and float64 pick the same L1 codeword",
          bool((a32 == a64).all()), f"{int((a32 == a64).sum())}/{len(a32)} agree")


# -------------------------------------------------------------- C: gradient path
def _layer_with(geodesic: bool) -> VQLayer:
    cfg_layer = VQLayer(16, 4, 0.25, geodesic=geodesic,
                        geodesic_normalizer=1.0 if geodesic else 1.0,
                        geodesic_curvature=CURV)
    return cfg_layer


def test_c_gradient() -> None:
    torch.manual_seed(1)
    l = _layer_with(geodesic=True)
    with torch.no_grad():
        l.embed.weight.copy_(torch.randn_like(l.embed.weight) * 0.3)
    z = torch.randn(32, 4, requires_grad=True)

    # C1. argmin carries no gradient: the index is a hard argmax of the distances.
    d = l._distances(z)
    idx = d.argmin(-1)
    sel = torch.zeros_like(d).scatter_(1, idx.unsqueeze(1), 1.0).sum()
    check("C1 the assignment carries no gradient (argmin is non-differentiable)",
          sel.grad_fn is None and not sel.requires_grad,
          "the one-hot assignment tensor has grad_fn=None, so the geodesic selection "
          "never reaches the latents or the codebook")

    # C2. the clamp on the geodesic distance has a dead zone at zero.
    tiny = torch.zeros(1, 4, requires_grad=True)
    d_tiny = l._distances(tiny + torch.randn(1, 4) * 0.0)
    check("C2 _distances clamps at 1e-6", float(d_tiny.min()) >= 1e-6,
          f"min = {float(d_tiny.min()):.3e}")

    # C3. the straight-through estimator returns the RAW codeword, not a ball point.
    z2 = torch.randn(32, 4)
    x_q, quant_loss, _, _ = l(z2)
    got = x_q[0]
    want = l.embed.weight[l(z2)[3][0]]
    check("C3 STE value equals the raw codeword (pre-map Euclidean vector)",
          torch.allclose(got, want, atol=1e-6),
          f"||x_q||={float(torch.linalg.vector_norm(got)):.4f}")

    # C4. the quantisation loss is a Euclidean squared MSE in the PRE-MAP space.
    z3 = torch.randn(32, 4)
    _, ql3, _, _ = l(z3)
    idx3 = l._distances(z3).argmin(-1)
    x_q_manual = l.embed.weight[idx3]
    euclid = torch.nn.functional.mse_loss(x_q_manual, z3.detach()) \
        + 0.25 * torch.nn.functional.mse_loss(z3, x_q_manual.detach())
    check("C4 quant_loss is exactly the Euclidean MSE (no ball metric in the loss)",
          abs(float(ql3) - float(euclid)) < 1e-9,
          f"layer {float(ql3):.9f} vs euclid {float(euclid):.9f}")

    # C5. the codebook gradient therefore comes from the Euclidean term only.
    z4 = torch.randn(32, 4)
    _, ql4, _, _ = l(z4)
    l.zero_grad(set_to_none=True)
    ql4.backward()
    gc = l.embed.weight.grad
    check("C5 codebook receives a gradient from the Euclidean MSE term",
          gc is not None and float(gc.abs().sum()) > 0.0,
          f"|grad| = {float(gc.abs().sum()):.4f}" if gc is not None else "None")


# -------------------------------------------------- D: assignment/loss mismatch
def _scaled_to_median_norm(dim: int, median_norm: float, seed: int) -> torch.Tensor:
    """A random latent block whose MEDIAN norm is exactly `median_norm`.

    The fork arms resolve their normalizer from the training median norm, so the
    training map puts the median at radius tanh(atanh(0.7)) = 0.7. A synthetic
    block with the wrong norm silently saturates the map at the ball boundary and
    every measurement taken on it is meaningless; this helper removes that trap.
    """
    g = torch.Generator().manual_seed(seed)
    z = torch.randn(4096, dim, generator=g, dtype=torch.float64)
    m = float(torch.linalg.vector_norm(z, dim=-1).median())
    return z * (median_norm / m)


def test_d_geometry_consistency() -> None:
    """The assignment is geodesic; the codebook target is Euclidean, in the
    pre-map space. Measure how differently the two geometries order codewords
    under the TRAINING map, and confirm that they are not merely a rescaling."""
    nz = 11.921758725170167          # the normalizer the fork arms resolve
    median_norm = torch.atanh(torch.tensor(0.7, dtype=torch.float64)).item() / nz
    z = _scaled_to_median_norm(16, median_norm, 5)
    codes = _scaled_to_median_norm(16, median_norm, 6)[:64]

    d_geo = H.pair_poincare_distance(H.encode_ball(z, CURV, nz),
                                     H.encode_ball(codes, CURV, nz), CURV)
    d_euc = z.pow(2).sum(1, keepdim=True) - 2 * z @ codes.t() + codes.pow(2).sum(1)
    a_geo, a_euc = d_geo.argmin(-1), d_euc.argmin(-1)
    changed = float((a_geo != a_euc).float().mean())
    check("D1 geodesic and Euclidean L1 assignments differ for a measurable share",
          0.0 < changed < 1.0, f"{changed:.3f} of items change codeword")

    # The two distances are not related by a monotone rescaling: if they were, the
    # per-item ranking of codewords would be identical. Rank-correlate the two
    # orderings per item and average.
    r_geo = d_geo.argsort(-1).argsort(-1).float()
    r_euc = d_euc.argsort(-1).argsort(-1).float()
    a = r_geo - r_geo.mean(1, keepdim=True)
    b = r_euc - r_euc.mean(1, keepdim=True)
    corr = (a * b).sum(1) / (a.norm(dim=1) * b.norm(dim=1)).clamp_min(1e-12)
    spearman = float(corr.mean())
    check("D2 geodesic distance is not a monotone rescaling of the Euclidean one",
          abs(spearman) < 0.999,
          f"mean per-item Spearman(geodesic rank, Euclidean rank) = {spearman:.4f}")

    # Median item radius under the training map is 0.7 by construction, so the
    # geometry really is being exercised and the measurements above are valid.
    r_items = float(H.encode_ball(z, CURV, nz).norm(dim=-1).median())
    check("D2 the training map really places the median item at radius 0.7",
          abs(r_items - 0.7) < 0.02, f"median ball radius = {r_items:.4f}")


# ------------------------------------------------------------- E: init ordering
def test_e_init_order() -> None:
    """KMeans runs, and the level assignment right after it, with the normalizer
    that is still the construction default (1.0). Training later replaces it with
    ~11.92. Measure whether that changes the geometry at all, or is only a
    rescaling of the same ordering."""
    import train_rqvae as T

    T.set_seed(T.SEED)
    geo = {"normalizer": 1.0, "curvature": T.L1_GEODESIC_CURVATURE}
    model = RQVAE(T._tokenizer_config(geo), in_dim=32)
    T.initialize_tiger_weights(model)
    layer = model.rq.vq_layers[0]
    check("E1 at initialisation the L1 geodesic normalizer is the construction default",
          abs(float(layer.geodesic_normalizer) - 1.0) < 1e-12,
          f"normalizer = {float(layer.geodesic_normalizer)}")

    nz_train = 11.921758725170167
    median_norm = torch.atanh(torch.tensor(0.7, dtype=torch.float64)).item() / nz_train
    lat = _scaled_to_median_norm(32, median_norm, 11)[:2048]
    ctr = _scaled_to_median_norm(32, median_norm, 12)[:256]

    def assign(nz):
        dg = H.pair_poincare_distance(H.encode_ball(lat, CURV, nz),
                                      H.encode_ball(ctr, CURV, nz), CURV)
        de = lat.pow(2).sum(1, keepdim=True) - 2 * lat @ ctr.t() + ctr.pow(2).sum(1)
        return dg.argmin(-1), de.argmin(-1), dg

    a_init, a_euc_init, _ = assign(1.0)
    a_train, a_euc_train, _ = assign(nz_train)
    agree_init = float((a_init == a_euc_init).float().mean())
    agree_train = float((a_train == a_euc_train).float().mean())

    check("E2 under the INIT normalizer the geodesic ordering IS the Euclidean ordering",
          agree_init > 0.99,
          f"agreement {agree_init:.4f}: at normalizer 1.0 the map is linear, so the "
          f"geodesic differs from the Euclidean distance only by a factor")
    check("E2 under the TRAINING normalizer the two orderings really diverge",
          agree_train < 0.9,
          f"agreement {agree_train:.4f}")

    changed = float((a_init != a_train).float().mean())
    check("E3 the L1 assignment chosen at KMeans time differs from the one training uses",
          changed > 0.0,
          f"{changed:.4f} of items get a different L1 codeword under the two maps")

    n_init = int(a_init.unique().numel())
    n_train = int(a_train.unique().numel())
    check("E4 the training map does not spread the items over MORE codes than the init map",
          n_train >= n_init,
          f"init {n_init}/256 vs train {n_train}/256")

    r_init = float(H.encode_ball(lat, CURV, 1.0).norm(dim=-1).median())
    r_train = float(H.encode_ball(lat, CURV, nz_train).norm(dim=-1).median())
    check("E5 the two maps put the same latents at very different ball radii",
          abs(r_train - 0.7) < 0.02 and r_init < 0.1,
          f"init median radius {r_init:.4f} vs train median radius {r_train:.4f}")


# ------------------------------------------------- F: hard-negative hinge sign
def test_f_hinge_sign() -> None:
    """The registered hard-negative term is relu(margin + d_shared - d_alt).

    Read the gradient w.r.t. d_shared. DESCENT moves d_shared by -grad, so a
    NEGATIVE gradient means the optimiser INCREASES the distance to the shared
    prototype, i.e. it pushes the negative OUT of the codeword it wrongly shares.
    A positive gradient pulls it IN.
    """
    margin = 0.05

    def grad_form(form, d_shared_val, d_alt_val):
        ds = torch.tensor([d_shared_val], requires_grad=True)
        da = torch.tensor([d_alt_val], requires_grad=True)
        if form == "registered":
            loss = torch.relu(margin + ds - da)
        elif form == "corrected":
            loss = torch.relu(margin + da - ds)
        else:  # "positive": same shape as registered, target is the shared code
            loss = torch.relu(margin + ds - da)
        g_ds, g_da = torch.autograd.grad(loss, (ds, da))
        return float(loss), float(g_ds), float(g_da)

    val, g_ds, g_da = grad_form("registered", 0.30, 0.32)
    print(f"     registered relu(margin + d_shared - d_alt) on a hard negative: "
          f"value {val:.4f}, d/d(d_shared) {g_ds:+.1f}, d/d(d_alt) {g_da:+.1f}", flush=True)
    check("F1 registered hard-negative hinge pulls the negative INTO the shared code "
          "(it should push out)",
          g_ds > 0.0,
          "d/d(d_shared) > 0, so descent DECREASES the distance to the shared codeword: "
          "the term actively deepens the wrong sharing it was meant to break"
          if g_ds > 0.0 else "already correct")

    val2, g_ds2, _ = grad_form("corrected", 0.30, 0.32)
    check("F2 corrected relu(margin + d_alt - d_shared) pushes the negative OUT",
          g_ds2 < 0.0,
          f"value {val2:.4f}, d/d(d_shared) {g_ds2:+.1f}: descent increases the shared "
          f"distance, i.e. the negative crosses the boundary")

    val3, g_ds3, _ = grad_form("positive", 0.30, 0.32)
    check("F3 the positive term keeps its direction (pulls toward the shared code)",
          g_ds3 > 0.0,
          f"value {val3:.4f}, d/d(d_shared) {g_ds3:+.1f}: descent decreases the shared "
          f"distance, i.e. the positive stays in its group")


def main() -> int:
    print("=== A: is the Poincare distance mathematically correct? ===", flush=True)
    test_a_distance_math()
    print("\n=== B: is it stable in float32 where it matters? ===", flush=True)
    test_b_float32()
    print("\n=== C: what is the gradient path? ===", flush=True)
    test_c_gradient()
    print("\n=== D: is the quantisation objective consistent with the assignment? ===", flush=True)
    test_d_geometry_consistency()
    print("\n=== E: is initialisation consistent with training-time geometry? ===", flush=True)
    test_e_init_order()
    print("\n=== F: does the hard-negative hinge push out or pull in? ===", flush=True)
    test_f_hinge_sign()

    failed = [n for n, ok, _ in _RESULTS if not ok]
    print("\n" + "=" * 78)
    print(f"{len(_RESULTS) - len(failed)}/{len(_RESULTS)} checks passed")
    if failed:
        print("checks that did NOT pass (each is a finding, not a test bug):")
        for n in failed:
            print("  FAIL " + n)
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())