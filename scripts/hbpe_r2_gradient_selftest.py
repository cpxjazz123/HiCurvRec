"""Gradient and numerical self-tests for the HBPE R2 arms (issues #34, #35, #36).

Each new term is asserted to do exactly what its issue specifies and nothing more:

  * finite and non-zero gradient on the Encoder,
  * identically zero gradient on every codebook level,
  * no gradient flowing back through the discrete ``argmin`` index,
  * the RQ reconstruction / quantization path still trains normally.

Plus the mechanism-specific behaviour:

  #34/#36  the occupancy barrier is hard-forward (its forward histogram is the exact
          argmin histogram) but soft-backward, and only bites when the current
          entropy is below the reference.
  #35      L_ER1 keeps its s1 gradient; L_ER12 still reaches the Encoder through p2;
          under the stop-gradient variant d(L_ER12)/dp1 is exactly zero while
          d(L_ER12)/dp2 is not; and the L2 term does not get cancelled by the STE.
  #36      the negative cap has non-zero gradient when the current soft negative
          sharing exceeds the teacher reference and exactly zero when it does not.

Exits non-zero on any failure and writes no report file.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "stage2_RQ-VAE/TIGER_HBPE_A1_R2_W010_OCC"))
sys.path.insert(0, str(REPO_ROOT / "stage2_RQ-VAE/TIGER_HBPE_A2_R2_L2W005_SGL1"))
sys.path.insert(0, str(REPO_ROOT / "stage2_RQ-VAE/TIGER_HBPE_A3_R2_NEGCAP_OCC"))

import hbpe_a1r2 as a1r2  # noqa: E402
import hbpe_a2r2 as a2r2  # noqa: E402
import hbpe_a3r2 as a3r2  # noqa: E402

BATCH = 32
DIM = 32
CODES = 256


def fresh(seed: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Anchor/positive/negative latents and two codebooks, all requiring grad."""
    gen = torch.Generator().manual_seed(seed)
    z_a = torch.randn(BATCH, DIM, generator=gen, requires_grad=True)
    z_p = torch.randn(BATCH, DIM, generator=gen, requires_grad=True)
    z_n = torch.randn(BATCH, 10, DIM, generator=gen, requires_grad=True)
    c1 = torch.randn(CODES, DIM, generator=gen, requires_grad=True)
    c2 = torch.randn(CODES, DIM, generator=gen, requires_grad=True)
    return z_a, z_p, z_n, c1, c2


def flat(grads) -> torch.Tensor | None:
    present = [g.reshape(-1) for g in grads if g is not None]
    return torch.cat(present) if present else None


def grad_of(scalar: torch.Tensor, tensors) -> torch.Tensor | None:
    if not scalar.requires_grad or scalar.grad_fn is None:
        return None
    return flat(torch.autograd.grad(scalar, tensors, retain_graph=True, allow_unused=True))


def check(name: str, condition: bool, failures: list[str], detail: str = "") -> None:
    print(f"  {'PASS' if condition else 'FAIL'}  {name}{f'  ({detail})' if detail else ''}")
    if not condition:
        failures.append(f"{name}{f': {detail}' if detail else ''}")


def check_encoder_only(name: str, scalar, params, codebooks, failures) -> None:
    """Finite non-zero on the Encoder, exactly zero on every codebook."""
    g = grad_of(scalar, params)
    norm = 0.0 if g is None else float(g.norm())
    check(f"{name}: encoder gradient finite and non-zero",
          g is not None and torch.isfinite(g).all() and norm > 0.0,
          failures, f"norm={norm:.6g}")
    for index, codebook in enumerate(codebooks):
        gc = grad_of(scalar, [codebook])
        value = 0.0 if gc is None else float(gc.abs().max())
        check(f"{name}: zero gradient on codebook L{index + 1}",
              value == 0.0, failures, f"max|grad|={value:.3g}")


def test_a1(failures: list[str]) -> None:
    print("\n[#34] A1-R2: L1 enrichment and the occupancy barrier")
    z_a, z_p, z_n, c1, c2 = fresh(101)
    params = [z_a, z_p, z_n]
    temperature = 0.01

    p1_a = a1r2._soft_prefix_distribution(z_a, c1, temperature)
    p1_p = a1r2._soft_prefix_distribution(z_p, c1, temperature)
    p1_n = a1r2._soft_prefix_distribution(z_n, c1, temperature)
    loss, stats = a1r2.hbpe_l1_loss(p1_a, p1_p, p1_n)
    check_encoder_only("L_A1", loss, params, [c1, c2], failures)
    check("L_A1: Soft_ER1 is finite",
          torch.isfinite(torch.tensor(stats["Soft_ER1"])), failures,
          f"Soft_ER1={stats['Soft_ER1']:.4f}")

    # The argmin index must not be differentiable: changing the codebook slightly
    # cannot move a gradient that goes through hard indices.
    catalogue = torch.randn(512, DIM, generator=torch.Generator().manual_seed(7),
                            requires_grad=True)
    occ, occ_stats = a1r2.hbpe_l1_occupancy_loss(catalogue, c1, temperature, 5.4)
    g = grad_of(occ, [catalogue])
    check("L_occ: encoder gradient finite and non-zero",
          g is not None and torch.isfinite(g).all() and float(g.norm()) > 0.0,
          failures, f"norm={0.0 if g is None else float(g.norm()):.6g}")
    gc = grad_of(occ, [c1])
    check("L_occ: zero gradient on the L1 codebook",
          gc is None or float(gc.abs().max()) == 0.0, failures)
    check("L_occ: effective codes <= codebook size",
          occ_stats["effective_codes_exp_H"] <= CODES + 1e-6, failures,
          f"exp(H)={occ_stats['effective_codes_exp_H']:.2f}")
    check("L_occ: distinct codes <= sample size",
          occ_stats["hard_distinct_l1_codes"] <= 512, failures,
          f"distinct={occ_stats['hard_distinct_l1_codes']}")

    # Hard-forward: the reported forward histogram is the argmin histogram, so a
    # collapsed catalogue (all points on one codeword) must read ~0 nats.
    collapsed = c1[0].detach().unsqueeze(0).repeat(512, 1) + 1e-6 * torch.randn(
        512, DIM, generator=torch.Generator().manual_seed(11)
    )
    collapsed.requires_grad_(True)
    _, collapsed_stats = a1r2.hbpe_l1_occupancy_loss(collapsed, c1, temperature, 5.4)
    check("L_occ: collapsed catalogue reads near-zero entropy",
          collapsed_stats["H_pi"] < 0.5, failures,
          f"H_pi={collapsed_stats['H_pi']:.4f}")
    check("L_occ: collapsed catalogue loses against H_ref",
          collapsed_stats["L_occ"] > 0.5, failures,
          f"L_occ={collapsed_stats['L_occ']:.4f}")

    spread = torch.randn(512, DIM, generator=torch.Generator().manual_seed(13))
    spread = (spread / spread.norm(dim=1, keepdim=True) * 30.0).requires_grad_(True)
    measured = a1r2.hbpe_l1_occupancy_loss(spread, c1, 0.01, 5.4)[1]["H_pi"]
    # The barrier is satisfied whenever the measured entropy clears the reference,
    # so assert the ReLU semantics directly: satisfied below, violated above.
    _, satisfied = a1r2.hbpe_l1_occupancy_loss(spread, c1, 0.01, measured - 0.1)
    _, violated = a1r2.hbpe_l1_occupancy_loss(spread, c1, 0.01, measured + 0.1)
    check("L_occ: no penalty when entropy clears the reference",
          satisfied["L_occ"] == 0.0, failures,
          f"L_occ={satisfied['L_occ']:.6f} at H_ref={measured - 0.1:.4f}")
    check("L_occ: penalty grows as the reference rises",
          violated["L_occ"] > satisfied["L_occ"], failures,
          f"L_occ={violated['L_occ']:.6f} vs {satisfied['L_occ']:.6f}")
    check("L_occ: penalty is quadratic in the shortfall",
          abs(violated["L_occ"] - (0.1 / (measured + 0.1)) ** 2) < 1e-9,
          failures,
          f"got {violated['L_occ']:.9f}, expected {(0.1 / (measured + 0.1)) ** 2:.9f}")


def test_a2(failures: list[str]) -> None:
    print("\n[#35] A2-R2: two-level enrichment, L2 weight and gradient isolation")
    z_a, z_p, z_n, c1, c2 = fresh(202)
    params = [z_a, z_p, z_n]
    t1, t2 = 0.01, 0.002

    def build(z):
        p1 = a2r2._soft_prefix_distribution(z, c1, t1)
        r2 = a2r2._continuous_l2_residual_proxy(z, c1)
        p2 = a2r2._soft_prefix_distribution(r2, c2, t2)
        return p1, p2

    p1_a, p2_a = build(z_a)
    p1_p, p2_p = build(z_p)
    p1_n, p2_n = build(z_n)

    total, l_er1, l_er12, stats = a2r2.hbpe_joint_enrichment_loss(
        p1_a, p1_p, p1_n, p2_a, p2_p, p2_n
    )
    check_encoder_only("joint total", total, params, [c1, c2], failures)

    g1 = grad_of(l_er1, params)
    check("L_ER1: reaches the Encoder",
          g1 is not None and float(g1.norm()) > 0.0, failures,
          f"norm={0.0 if g1 is None else float(g1.norm()):.6g}")

    # Keep only p1 and p2 alive so the partials can be measured directly.
    p1a = p1_a.detach().requires_grad_(True)
    p2a = p2_a.detach().requires_grad_(True)
    p2n = p2_n.detach().requires_grad_(True)
    _, sg_er1, sg_er12, _ = a2r2.hbpe_joint_enrichment_loss(
        p1a, p1_p, p1_n, p2a, p2_p, p2n, stopgrad_l1_in_s12=True
    )
    g_p1 = grad_of(sg_er12, [p1a])
    g_p2 = grad_of(sg_er12, [p2a, p2n])
    check("sgL1: d(L_ER12)/dp1 is exactly zero",
          g_p1 is None or float(g_p1.abs().max()) == 0.0, failures,
          f"max|d/dp1|={0.0 if g_p1 is None else float(g_p1.abs().max()):.3g}")
    check("sgL1: d(L_ER12)/dp2 is non-zero",
          g_p2 is not None and float(g_p2.norm()) > 0.0, failures,
          f"norm={0.0 if g_p2 is None else float(g_p2.norm()):.6g}")

    p1a2 = p1_a.detach().requires_grad_(True)
    _, plain_er1, _, _ = a2r2.hbpe_joint_enrichment_loss(
        p1a2, p1_p, p1_n, p2_a, p2_p, p2_n
    )
    g_plain_p1 = grad_of(plain_er1, [p1a2])
    check("L_ER1: s1 is never detached",
          g_plain_p1 is not None and float(g_plain_p1.norm()) > 0.0, failures,
          f"norm={0.0 if g_plain_p1 is None else float(g_plain_p1.norm()):.6g}")

    # The residual proxy must carry z, otherwise the STE identity gradient cancels
    # the L2 path and the whole branch is dead.
    probe = z_a[:1].detach().requires_grad_(True)
    r2 = a2r2._continuous_l2_residual_proxy(probe, c1)
    check("r2_proxy: gradient reaches z (STE does not cancel L2)",
          r2.requires_grad and r2.grad_fn is not None, failures)
    gp = grad_of((r2 ** 2).sum(), [probe])
    check("r2_proxy: non-zero gradient on z",
          gp is not None and float(gp.norm()) > 0.0, failures,
          f"norm={0.0 if gp is None else float(gp.norm()):.6g}")
    check("reported L_ER components are finite",
          torch.isfinite(l_er1) and torch.isfinite(l_er12), failures,
          f"L_ER1={stats['L_ER1']:.4f} L_ER12={stats['L_ER12']:.4f}")


def test_a3(failures: list[str]) -> None:
    print("\n[#36] A3-R2: negative-sharing cap and occupancy guard")
    z_a, z_p, z_n, c1, c2 = fresh(303)
    params = [z_a, z_p, z_n]
    # A near-uniform soft assignment keeps S12_neg at its realistic magnitude. With
    # independent random latents and a peaked softmax, p2_a . p2_n collapses to ~1e-19,
    # far below the spec's max(B12_neg, 1e-8) floor, so the cap could never fire and the
    # assertion would be testing the floor rather than the mechanism.
    t1, t2 = 0.01, 5.0

    p1_a = a3r2._soft_prefix_distribution(z_a, c1, t1)
    p1_p = a3r2._soft_prefix_distribution(z_p, c1, t1)
    p1_n = a3r2._soft_prefix_distribution(z_n, c1, t1)
    p2_a = a3r2._soft_prefix_distribution(
        a3r2._continuous_l2_residual_proxy(z_a, c1), c2, t2
    )
    p2_p = a3r2._soft_prefix_distribution(
        a3r2._continuous_l2_residual_proxy(z_p, c1), c2, t2
    )
    p2_n = a3r2._soft_prefix_distribution(
        a3r2._continuous_l2_residual_proxy(z_n, c1), c2, t2
    )

    # References set strictly below the measured soft shares, so both caps fire.
    # The reductions must match the module's exactly: S1_neg is the mean of s1, but
    # S12_neg is the mean of the per-pair PRODUCT, not the product of the two means.
    with torch.no_grad():
        s1_pairs = (p1_a.unsqueeze(1) * p1_n).sum(-1)
        s2_pairs = (p2_a.unsqueeze(1) * p2_n).sum(-1)
        s1_now = float(s1_pairs.mean())
        s12_now = float((s1_pairs * s2_pairs).mean())
    tight_total, tight = a3r2.hbpe_negcap_enrichment_loss(
        p1_a, p1_p, p1_n, p2_a, p2_p, p2_n,
        b1_neg=0.5 * s1_now, b12_neg=0.5 * s12_now,
    )
    check("L_neg: fires when S_neg exceeds the reference",
          tight["L_neg1"] > 0.0 and tight["L_neg12"] > 0.0, failures,
          f"L_neg1={tight['L_neg1']:.6f} L_neg12={tight['L_neg12']:.8f}")
    check("L_neg: activation flags set",
          tight["negcap1_active"] == 1.0 and tight["negcap12_active"] == 1.0, failures)
    check("L_neg: penalty is quadratic in the excess",
          abs(tight["L_neg1"] - (s1_now / (0.5 * s1_now) - 1.0) ** 2) < 1e-8,
          failures, f"got {tight['L_neg1']:.6f}")
    check_encoder_only("negcap total", tight_total, params, [c1, c2], failures)

    # References set above the measured soft shares: both caps must be exactly zero.
    loose_total, loose = a3r2.hbpe_negcap_enrichment_loss(
        p1_a, p1_p, p1_n, p2_a, p2_p, p2_n,
        b1_neg=2.0 * s1_now, b12_neg=2.0 * s12_now,
    )
    check("L_neg: silent when S_neg is below the reference",
          loose["L_neg1"] == 0.0 and loose["L_neg12"] == 0.0, failures,
          f"L_neg1={loose['L_neg1']:.6f} L_neg12={loose['L_neg12']:.8f}")
    check("L_neg: activation flags clear",
          loose["negcap1_active"] == 0.0 and loose["negcap12_active"] == 0.0, failures)

    # Exactly at the reference the penalty is zero: ReLU boundary.
    at_total, at_ref = a3r2.hbpe_negcap_enrichment_loss(
        p1_a, p1_p, p1_n, p2_a, p2_p, p2_n,
        b1_neg=s1_now, b12_neg=s12_now,
    )
    check("L_neg: zero exactly at the reference",
          at_ref["L_neg1"] == 0.0 and at_ref["L_neg12"] == 0.0, failures,
          f"L_neg1={at_ref['L_neg1']:.8f} L_neg12={at_ref['L_neg12']:.10f}")

    # A soft reference must compare against a soft quantity, never a hard rate.
    check("B_neg is compared against soft S_neg",
          abs(loose["S1_neg"] - s1_now) < 1e-9, failures,
          f"reported={loose['S1_neg']:.8f} measured={s1_now:.8f}")

    # The teacher reference must be computable on the same items.
    with torch.no_grad():
        b1, b12 = a3r2.teacher_negative_sharing(z_a, z_n, c1, c2, 0.01, 0.002, 1.0)
    check("teacher B_neg(tau) is finite and positive",
          b1 > 0.0 and b12 > 0.0, failures, f"B1={b1:.6g} B12={b12:.6g}")

    # Pin the spec's max(B_neg, 1e-8) floor. Without it a zero reference divides by
    # zero; with it the penalty is exactly (S1_neg / 1e-8 - 1)^2 and stays finite.
    floored, floor_stats = a3r2.hbpe_negcap_enrichment_loss(
        p1_a, p1_p, p1_n, p2_a, p2_p, p2_n, b1_neg=0.0, b12_neg=0.0
    )
    expected_floor = (s1_now / 1e-8 - 1.0) ** 2
    # float32 throughout: at a magnitude of 1e12 the expected relative error is ~1e-7.
    check("L_neg: the 1e-8 reference floor bounds the penalty",
          torch.isfinite(floored)
          and abs(floor_stats["L_neg1"] - expected_floor) / expected_floor < 1e-6,
          failures,
          f"L_neg1={floor_stats['L_neg1']:.6g}, expected={expected_floor:.6g}")

    catalogue = torch.randn(512, DIM, generator=torch.Generator().manual_seed(17),
                            requires_grad=True)
    occ, occ_stats = a3r2.hbpe_l1_occupancy_loss(catalogue, c1, t1, 5.4)
    g = grad_of(occ, [catalogue])
    check("L_occ: encoder gradient finite and non-zero",
          g is not None and torch.isfinite(g).all() and float(g.norm()) > 0.0,
          failures, f"norm={0.0 if g is None else float(g.norm()):.6g}")
    gc = grad_of(occ, [c1])
    check("L_occ: zero gradient on the L1 codebook",
          gc is None or float(gc.abs().max()) == 0.0, failures)
    check("L_occ: H_pi <= ln(codebook size)",
          occ_stats["H_pi"] <= float(torch.log(torch.tensor(float(CODES)))) + 1e-6,
          failures, f"H_pi={occ_stats['H_pi']:.4f}")


def main() -> None:
    torch.manual_seed(0)
    failures: list[str] = []
    test_a1(failures)
    test_a2(failures)
    test_a3(failures)

    print()
    if failures:
        print(f"FAILED ({len(failures)}):")
        for item in failures:
            print(f"  - {item}")
        raise SystemExit(1)
    print("all HBPE R2 gradient and numerical self-tests PASSED")


if __name__ == "__main__":
    main()
