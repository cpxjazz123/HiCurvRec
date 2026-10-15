"""Is the residual map invertible? Answer it numerically instead of by argument.

Issue #14 was right that the earlier reasoning was a non sequitur: showing the naive
mirrored addition fails, and that Mobius addition is non-associative, does not show
that no inverse exists. Mobius addition forms a gyrogroup, where cancellation has to
account for the operation order and a gyration term.

Rather than guess a closed form -- the mistake that produced the bad inference in the
first place -- this constructs the inverse numerically and checks it. ``F_q`` is
smooth and, on the sampled evidence, injective; Newton's method with the analytic
Jacobian (obtained from autograd) inverts it, and the residual of the inversion is
reported. A numerical inverse that reproduces the map to machine precision is
evidence that the map is invertible on the tested domain; it is not a proof of
invertibility everywhere, and it is not offered as one.

Also reported: how far the *naive* mirrored addition is from being that inverse, so
the two statements stay separate.
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hyperbolic_residual import MOBIUS_CURVATURE, _mobius_residual  # noqa: E402

fails = []


def check(name, ok, detail=""):
    print(f"  [{'ok ' if ok else 'FAIL'}] {name}{(' -- ' + detail) if detail else ''}")
    if not ok:
        fails.append(name)


def residual_map(r: torch.Tensor, q: torch.Tensor, c: float) -> torch.Tensor:
    """The map whose inverse is in question: r -> r (-) q."""
    return _mobius_residual(r, q, c)


def newton_inverse(z: torch.Tensor, q: torch.Tensor, c: float,
                   steps: int = 60, tol: float = 1e-12) -> tuple:
    """Solve F_q(x) = z for x by Newton's method; returns (x, final residual).

    The Jacobian comes from autograd on a single probe row, which is enough for a
    small step and keeps the construction independent of any closed form.
    """
    x = z.clone().detach().requires_grad_(True)
    for _ in range(steps):
        value = residual_map(x, q, c)
        gap = value - z
        if float(gap.norm()) < tol:
            break
        jac = torch.autograd.functional.jacobian(
            lambda t: residual_map(t, q, c), x, vectorize=True
        )
        # jac is (N, d, N, d) here; solve per row to keep it small and exact.
        rows = []
        for i in range(x.shape[0]):
            j = jac[i, :, i, :]
            try:
                delta = torch.linalg.solve(j, gap[i])
            except Exception:                                   # noqa: BLE001
                delta = torch.linalg.lstsq(j, gap[i]).solution
            rows.append(delta)
        step = torch.stack(rows)
        x = (x - step).detach().requires_grad_(True)
    with torch.no_grad():
        final = residual_map(x, q, c)
    return x.detach(), float((final - z).norm(dim=-1).max())


torch.manual_seed(0)
d = 6
for c in (0.25, 1.0, 4.0):
    # interior, near-boundary, and the origin as the special case
    # The ball has radius 1/sqrt(c), so "near the boundary" has to be scaled by it:
    # at c = 4 a norm of 0.85 is outside the ball and gets clamped, which made an
    # earlier version of this check report a convergence failure that was really a
    # domain error in the fixture.
    radius = 1.0 / (c ** 0.5)
    cases = {
        "interior": torch.randn(8, d) * (0.2 * radius),
        "near-boundary": torch.nn.functional.normalize(torch.randn(8, d), dim=-1)
        * (0.85 * radius),
        "origin": torch.zeros(8, d),
    }
    for label, r in cases.items():
        q = torch.randn(1, d) * (0.3 * radius)
        z = residual_map(r, q, c)
        x, gap = newton_inverse(z, q, c)
        check(f"c={c}: the numerical inverse reproduces the map ({label})",
              gap < 1e-6, f"max row residual = {gap:.3e}")

        # and it recovers the original point
        recovered = float((x - r).abs().max())
        check(f"c={c}: and recovers the input ({label})",
              recovered < 1e-5, f"max|diff| = {recovered:.3e}")

# --- the naive mirror, for contrast ---------------------------------------
r = torch.randn(8, d) * 0.2
q = torch.randn(1, d) * 0.3
c = MOBIUS_CURVATURE
naive = residual_map(residual_map(r, q, c), -q, c)     # (r (-) q) (-) (-q)
check("the naive mirrored addition is not the inverse",
      float((naive - r).abs().max()) > 1e-3,
      f"max|diff| = {float((naive - r).abs().max()):.3e}")

# --- the map is injective on the sampled domain ---------------------------
points = torch.randn(64, d) * 0.2
q = torch.randn(1, d) * 0.3
images = residual_map(points, q, c)
pairwise = torch.cdist(images, images) + torch.eye(64) * 1e9
check("distinct inputs give distinct images", float(pairwise.min()) > 0,
      f"closest pair = {float(pairwise.min()):.4f}")

print("\n" + ("all invertibility checks passed" if not fails else f"FAILURES: {fails}"))
print("note: a numerical inverse reproducing the map to machine precision is evidence")
print("      for invertibility on the tested domain, not a proof of it everywhere.")
sys.exit(1 if fails else 0)
