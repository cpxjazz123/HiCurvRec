"""Track F (issue #85) tests: the smooth surrogate, its oracle, and its three traps.

The oracle check differentiates the SMOOTH surrogate against central finite differences. It never
touches the hard argmin, because that has no derivative and a finite difference across it is
meaningless.
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


def check(name: str, condition: bool, detail: str = "") -> None:
    global CHECKS
    CHECKS += 1
    if not condition:
        FAILURES.append(f"{name}: {detail}")


class _ToyLayer:
    dim = 2

    def __init__(self, codebook: torch.Tensor):
        self.codebook = codebook

    def get_code_embs(self) -> torch.Tensor:
        return self.codebook

    def _distances(self, latents: torch.Tensor) -> torch.Tensor:
        return ((latents[:, None, :] - self.codebook[None]) ** 2).sum(-1)


class _ToyModel:
    def __init__(self, codebook: torch.Tensor):
        self.rq = type("_RQ", (), {})()
        self.rq.vq_layers = [_ToyLayer(codebook)]


def toy_model(seed: int = 0) -> _ToyModel:
    torch.manual_seed(seed)
    return _ToyModel(torch.randn(4, 2, dtype=torch.float64) * 0.5)


def test_posterior_is_a_distribution() -> None:
    model = toy_model()
    latents = torch.randn(7, 2, dtype=torch.float64)
    posterior = F.posterior(model, latents, tau=1.0)
    check("posterior rows sum to one",
          torch.allclose(posterior.sum(dim=-1), torch.ones(7, dtype=torch.float64), atol=1e-12),
          str(posterior.sum(dim=-1)))
    check("posterior is non-negative", bool((posterior >= 0).all()), "negative probability")


def test_tau_below_the_floor_is_refused() -> None:
    model = toy_model()
    latents = torch.randn(3, 2, dtype=torch.float64)
    try:
        F.posterior(model, latents, tau=1e-6)
    except ValueError as exc:
        check("a collapsing tau is refused rather than silently clamped", True)
        check("the refusal explains that it would become a hard argmin",
              "argmin" in str(exc), str(exc))
    else:
        check("a collapsing tau is refused rather than silently clamped", False, "it was accepted")


def test_identical_items_maximise_p_same() -> None:
    """The toy case the whole lane rests on: two items in one cell must score worst."""
    model = toy_model()
    latents = torch.tensor([[1.0, 0.0], [1.0, 0.0], [-1.0, 0.0]], dtype=torch.float64)
    posterior = F.posterior(model, latents, tau=0.5)
    value = F.same_code_probability(posterior, posterior)
    check("an item paired with itself scores highest",
          bool(value[0] >= value[2]), str(value))


def test_separation_loss_gradient_matches_finite_differences() -> None:
    """Central differences on the SMOOTH surrogate only -- never on the hard argmin."""
    torch.manual_seed(3)
    model = toy_model(3)
    latents = torch.randn(6, 2, dtype=torch.float64, requires_grad=True)
    pair_index = torch.tensor([[0, 3], [1, 4]])

    def surrogate(latent_values: torch.Tensor) -> torch.Tensor:
        posterior = F.posterior(model, latent_values, tau=0.7)
        left, right = posterior[pair_index[:, 0]], posterior[pair_index[:, 1]]
        return 0.1 * F.same_code_probability(left, right).mean()

    analytical = torch.autograd.grad(surrogate(latents), latents)[0]
    step = 1e-6
    numeric = torch.zeros_like(analytical)
    with torch.no_grad():
        for index in range(latents.shape[0]):
            for coordinate in range(latents.shape[1]):
                plus, minus = latents.detach().clone(), latents.detach().clone()
                plus[index, coordinate] += step
                minus[index, coordinate] -= step
                numeric[index, coordinate] = ((surrogate(plus) - surrogate(minus))
                                               / (2 * step)).reshape(-1)
    worst = float((analytical - numeric).abs().max())
    check(f"the analytic gradient matches finite differences (worst {worst:.2e})",
          worst < 1e-5, f"worst deviation {worst}")


def test_uniform_posterior_is_detected_as_gaming() -> None:
    """The cheap pass: uniform p_i scores well on the loss and must be flagged by the diagnostics."""
    size = 8
    uniform = torch.full((5, size), 1.0 / size, dtype=torch.float64)
    concentrated = torch.softmax(torch.tensor([[8.0, 0, 0, 0, 0, 0, 0, 0]]).repeat(5, 1), dim=-1)
    collapsed = torch.zeros(5, dtype=torch.long)
    spread = torch.arange(5, dtype=torch.long)
    margins = torch.full((5,), 1e-4, dtype=torch.float64)

    gamed = F.gaming_diagnostics(uniform, collapsed, margins)
    honest = F.gaming_diagnostics(concentrated, spread, margins)
    check("a uniform posterior is reported as maximally uniform",
          abs(gamed["posterior_entropy_relative_to_uniform"] - 1.0) < 1e-6,
          str(gamed["posterior_entropy_relative_to_uniform"]))
    check("a uniform posterior collapses hard code use",
          gamed["hard_effective_codes"] < honest["hard_effective_codes"],
          f"{gamed['hard_effective_codes']} vs {honest['hard_effective_codes']}")
    check("the diagnostic says a falling loss with falling diversity is a reject",
          "reject" in gamed["reading"], gamed["reading"])


def test_exclusion_mask_removes_self_and_false_negatives() -> None:
    source = torch.tensor([0, 0, 1])
    target = torch.tensor([5, 6, 7])
    # row 0 pairs an item with itself, row 1 with its observed target (a false negative),
    # row 2 with an unrelated item that must survive
    candidates = torch.tensor([[3, 3], [0, 5], [1, 7]])
    successors = torch.tensor([[5]])
    mask = F.exclusion_mask(source, target, candidates=candidates,
                            successors=successors).reshape(-1)
    check("self pairs are excluded", not bool(mask[0]), str(mask))
    check("a false negative (the observed target) is excluded", not bool(mask[1]), str(mask))
    check("an unrelated pair is kept", bool(mask[2]), str(mask))


def test_empty_pair_set_reports_itself() -> None:
    posteriors = torch.rand(4, 4, dtype=torch.float64)
    result = F.separation_loss(posteriors, torch.tensor([], dtype=torch.long))
    check("an empty candidate set is stated, not scored as zero loss",
          result["pairs"] == 0 and "exclusion mask" in result["note"], str(result))


def main() -> int:
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        try:
            test()
        except Exception as exc:  # noqa: BLE001
            FAILURES.append(f"{test.__name__} raised {type(exc).__name__}: {exc}")
    print(f"r7_f_soft_separation: {CHECKS} checks, {len(FAILURES)} failures")
    for failure in FAILURES:
        print(f"  FAIL {failure}")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    raise SystemExit(main())
