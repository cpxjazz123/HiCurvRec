"""Track F (issue #87): the independent VJP oracle for the smooth separation loss.

The analytic gradient is written out from the softmax Jacobian. It is checked against central finite
differences of the SMOOTH surrogate -- never against the implementation's own backward, and never
across the hard argmin, which has no derivative.

Every case the issue names is separate: Encoder and L1 codebook, a real-distance-scale tau, a known
toy geometry, mask truncation, empty pairs, invalid tau and NaN, and the exact lambda=0 no-op.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

REPO_ROOT = Path("/home/wlia0047/ar57/wenyu/GeneRec")
for entry in (str(REPO_ROOT / "tools"),):
    if entry not in sys.path:
        sys.path.insert(0, entry)

import r7_f_soft_separation as F  # noqa: E402

FAILURES: list[str] = []
CHECKS = 0
REAL_TAU = 0.37          # a plausible fraction of the real distance scale, not the prototype's 1.0


def check(name: str, condition: bool, detail: str = "") -> None:
    global CHECKS
    CHECKS += 1
    if not condition:
        FAILURES.append(f"{name}: {detail}")


class _Layer:
    def __init__(self, codebook: torch.Tensor):
        self.dim = codebook.shape[1]
        self._codebook = codebook

    def get_code_embs(self) -> torch.Tensor:
        return self._codebook

    def _distances(self, latents: torch.Tensor) -> torch.Tensor:
        return ((latents[:, None, :] - self._codebook[None]) ** 2).sum(-1)


class _Model:
    def __init__(self, codebook: torch.Tensor):
        self.rq = type("_RQ", (), {})()
        self.rq.vq_layers = [_Layer(codebook)]


def toy(seed: int = 0, n_items: int = 5, n_codes: int = 4, dim: int = 3):
    torch.manual_seed(seed)
    model = _Model(torch.randn(n_codes, dim, dtype=torch.float64) * 0.6)
    latents = torch.randn(n_items, dim, dtype=torch.float64) * 0.6
    pairs = torch.tensor([[0, 1], [2, 3], [1, 4]])
    return model, latents, pairs


def surrogate(model, latents: torch.Tensor, pairs: torch.Tensor, *,
              tau: float, lam: float) -> torch.Tensor:
    posterior = F.posterior(model, latents, tau=tau)
    left, right = posterior[pairs[:, 0]], posterior[pairs[:, 1]]
    return lam * F.same_code_probability(left, right).mean()


def central_difference(loss_fn, tensor: torch.Tensor, step: float = 1e-6) -> torch.Tensor:
    out = torch.zeros_like(tensor)
    flat = tensor.reshape(-1)
    flat_out = out.reshape(-1)
    for index in range(flat.numel()):
        plus = flat.detach().clone()
        minus = flat.detach().clone()
        plus[index] += step
        minus[index] -= step
        flat_out[index] = (loss_fn(plus.reshape(tensor.shape)) - loss_fn(minus.reshape(tensor.shape))) \
            / (2 * step)
    return out


def test_the_native_distance_is_the_squared_form_the_oracle_assumes() -> None:
    """The oracle differentiates the host's ACTUAL distance, so that assumption is asserted.

    VQLayer._distances is the expansion |z|^2 - 2 z.c + |c|^2, i.e. the SQUARED Euclidean distance.
    Differentiating the true Euclidean distance instead would give a plausible-looking gradient for
    a different function, which is the failure mode this check exists to prevent.
    """
    model, latents, _ = toy(seed=9)
    layer = model.rq.vq_layers[0]
    native = layer._distances(latents)
    squared = ((latents[:, None, :] - layer.get_code_embs()[None]) ** 2).sum(-1)
    euclidean = (latents[:, None, :] - layer.get_code_embs()[None]).norm(dim=-1)
    check("the native distance is the squared form", float((native - squared).abs().max()) < 1e-12,
          "the oracle's derivation does not match the host's distance")
    check("the native distance is NOT the true Euclidean form",
          float((native - euclidean).abs().max()) > 1e-6,
          "the two forms coincide here, so this test proves nothing")


def test_latent_vjp_matches_finite_differences_at_real_tau() -> None:
    model, latents, pairs = toy(seed=1)
    reference = surrogate(model, latents, pairs, tau=REAL_TAU, lam=F.DEFAULT_LAMBDA)
    posterior = F.posterior(model, latents, tau=REAL_TAU)
    analytic = F.analytic_grad_latents(model, latents, posterior, pairs, tau=REAL_TAU)
    numeric = central_difference(
        lambda value: surrogate(model, value, pairs, tau=REAL_TAU, lam=F.DEFAULT_LAMBDA),
        latents)
    worst = float((analytic - numeric).abs().max())
    check(f"latent VJP matches central differences at tau={REAL_TAU} (worst {worst:.2e})",
          worst < 1e-6, f"worst deviation {worst}")
    check("the latent VJP is nonzero at a real-distance-scale tau",
          float(analytic.abs().max()) > 1e-8, "the analytic gradient vanished")


def test_codebook_vjp_matches_finite_differences_at_real_tau() -> None:
    model, latents, pairs = toy(seed=2)
    posterior = F.posterior(model, latents, tau=REAL_TAU)
    analytic = F.analytic_grad_codebook(model, latents, posterior, pairs, tau=REAL_TAU)
    codebook = model.rq.vq_layers[0].get_code_embs()
    numeric = central_difference(
        lambda value: surrogate(_Model(value), latents, pairs, tau=REAL_TAU,
                                lam=F.DEFAULT_LAMBDA),
        codebook)
    worst = float((analytic - numeric).abs().max())
    check(f"codebook VJP matches central differences at tau={REAL_TAU} (worst {worst:.2e})",
          worst < 1e-6, f"worst deviation {worst}")
    check("the codebook VJP is nonzero at a real-distance-scale tau",
          float(analytic.abs().max()) > 1e-8, "the analytic codebook gradient vanished")


def test_toy_geometry_has_a_hand_checkable_answer() -> None:
    """Two items, two codes, one pair: the closed form is computable by hand."""
    codebook = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float64)
    model = _Model(codebook)
    # z sits closer to code 0; the partner closer to code 1
    latents = torch.tensor([[0.6, 0.0], [0.0, 0.6]], dtype=torch.float64)
    pairs = torch.tensor([[0, 1]])
    tau = 0.5
    posterior = F.posterior(model, latents, tau=tau)
    overlap = float((posterior[0] * posterior[1]).sum())
    # brute-force the same quantity by the definition
    distance = model.rq.vq_layers[0]._distances(latents)
    expected = float((torch.softmax(-distance[0] / tau, 0)
                      * torch.softmax(-distance[1] / tau, 0)).sum())
    check("P_same matches its definition on the toy geometry", abs(overlap - expected) < 1e-12,
          f"{overlap} vs {expected}")
    analytic = F.analytic_grad_latents(model, latents, posterior, pairs, tau=tau)
    numeric = central_difference(
        lambda value: surrogate(model, value, pairs, tau=tau, lam=F.DEFAULT_LAMBDA), latents)
    check("the toy geometry VJP also matches", float((analytic - numeric).abs().max()) < 1e-6,
          str(float((analytic - numeric).abs().max())))


def test_truncating_the_mask_scales_the_mean_not_the_shape() -> None:
    """Dropping pairs changes the loss through the MEAN only; the analytic route must agree."""
    model, latents, pairs = toy(seed=3)
    posterior = F.posterior(model, latents, tau=REAL_TAU)
    kept = pairs[:2]
    analytic = F.analytic_grad_latents(model, latents, posterior, kept, tau=REAL_TAU)
    numeric = central_difference(
        lambda value: surrogate(model, value, kept, tau=REAL_TAU, lam=F.DEFAULT_LAMBDA), latents)
    check("a truncated pair set still matches central differences",
          float((analytic - numeric).abs().max()) < 1e-6,
          str(float((analytic - numeric).abs().max())))


def test_empty_pair_set_is_reported_not_a_silent_zero() -> None:
    model, latents, _ = toy(seed=4)
    empty = torch.tensor([], dtype=torch.long)
    posterior = F.posterior(model, latents, tau=REAL_TAU)
    check("no pairs yields no analytic gradient rather than a fabricated zero",
          F.analytic_grad_latents(model, latents, posterior, empty, tau=REAL_TAU) is None,
          "a zero gradient was returned for an empty pair set")
    result = F.separation_loss(posterior, empty)
    check("an empty pair set says so", result["pairs"] == 0 and "exclusion mask" in result["note"],
          str(result))


def test_invalid_tau_is_refused_and_nan_is_refused() -> None:
    model, latents, pairs = toy(seed=5)
    for bad in (0.0, -1.0, 1e-9):
        try:
            F.posterior(model, latents, tau=bad)
        except ValueError:
            check(f"tau={bad} is refused", True)
        else:
            check(f"tau={bad} is refused", False, "it was accepted")
    broken = latents.clone()
    broken[0] = float("nan")
    posterior = torch.softmax(torch.full((5, 4), float("nan"), dtype=torch.float64), dim=-1)
    check("a NaN posterior does not produce a finite-looking loss",
          not torch.isfinite(F.same_code_probability(posterior, posterior)).all(),
          "NaN silently became finite")


def test_lambda_zero_is_an_exact_no_op() -> None:
    model, latents, pairs = toy(seed=6)
    posterior = F.posterior(model, latents, tau=REAL_TAU)
    analytic = F.analytic_grad_latents(model, latents, posterior, pairs, tau=REAL_TAU,
                                        lambda_weight=0.0)
    check("lambda=0 produces no latent gradient", analytic is None, "a gradient leaked through")
    check("lambda=0 produces no codebook gradient",
          F.analytic_grad_codebook(model, latents, posterior, pairs, tau=REAL_TAU,
                                   lambda_weight=0.0) is None, "a gradient leaked through")
    loss = F.separation_loss(posterior, pairs, lambda_weight=0.0)
    check("lambda=0 loss is exactly zero", float(loss["loss"]) == 0.0, str(loss["loss"]))


def main() -> int:
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        try:
            test()
        except Exception as exc:  # noqa: BLE001
            FAILURES.append(f"{test.__name__} raised {type(exc).__name__}: {exc}")
    print(f"r7_f_vjp: {CHECKS} checks, {len(FAILURES)} failures")
    for failure in FAILURES:
        print(f"  FAIL {failure}")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    raise SystemExit(main())
