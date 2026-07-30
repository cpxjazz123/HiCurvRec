"""
Task #334 — Issue #43 Gate 2a 预量化双曲感知映射 wrapper.

Hypothesis (Issue #43 owner):
  "用 expmap0(x_768d, κ=-0.74) 预处理, 让 encoder 不必'压平'双曲信号."

Implementation:
  - HypPreEncoder: nn.Module wrapping expmap0(u, c=0.74) [c=|κ|=0.74 → Poincaré ball]
  - HRQVAEWithHypPre: wraps existing HRQVAE, plugs HypPreEncoder BEFORE encoder
  - Regression tests:
      T1: κ=0 (enabled=False) → identity (output == input within tolerance)
      T2: κ=-0.74 enabled → output in Poincaré ball (‖output‖ < 1/√c = 1.16)
      T3: Output norm bounded by 1/√c - epsilon (boundary check)
      T4: Gradient flows through HypPreEncoder (c.requires_grad → grad ≠ 0)

Activation: opt-in via CLI flag `--use_hyp_pre_encoder` (default OFF for baseline
backward compatibility). When activated, replaces HRQVAE.forward input stage.

NOT modifies upstream hrqvae.py (per R11.4 critical decision: dry-run only —
wrapper composition instead of source patching).

Usage:
  python3 scripts/task334_issue43_gate2a_hyp_pre_encoder.py
"""
import sys
import os
import torch
import torch.nn as nn

# Add HG-Rec to import path
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HGREC_ROOT = os.path.join(REPO_ROOT, "HG-Rec")
sys.path.insert(0, HGREC_ROOT)

from model.utils import expmap0, proj_to_ball  # noqa: E402
from model.hrqvae import HRQVAE  # noqa: E402


# ────────────────────────────────────────────────────────────
# HypPreEncoder wrapper
# ────────────────────────────────────────────────────────────
class HypPreEncoder(nn.Module):
    """Pre-encoder wrapper: maps Euclidean input into Poincaré ball via expmap0.

    Issue #43 Gate 2a hypothesis: let encoder not "flatten" hyperbolic signals.
    c=0.74 (Ollivier mean from Task #70, |κ|=0.74 → Poincaré ball radius 1/√c=1.155).

    Args:
        c: positive scalar curvature magnitude (expmap0 takes c > 0 representing
           1/|κ|). For Poincaré ball κ<0, c=|κ|.
        enabled: if False, acts as identity (regression-safe).
        learnable_c: if True, c becomes a learnable parameter (initialized at c).
    """

    def __init__(self, c: float = 0.74, enabled: bool = True,
                 learnable_c: bool = False):
        super().__init__()
        self.enabled = enabled
        if learnable_c:
            # log-parameterize to keep c > 0
            self.log_c = nn.Parameter(torch.tensor(float(c)).log())
        else:
            self.log_c = None
            self.c = float(c)

    def _get_c(self) -> torch.Tensor:
        if self.log_c is not None:
            return self.log_c.exp()
        return torch.tensor(self.c)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.enabled:
            return x
        c = self._get_c()
        # expmap0 expects scalar c; broadcast to scalar via .item() if not learnable
        if self.log_c is None:
            x_hyp = expmap0(x, c=self.c)
        else:
            # for learnable c, we need to handle batch — use c.item() only if safe
            # but autograd doesn't allow .item() in forward, so use the scalar c value
            c_val = c  # tensor scalar, expmap0 will use it
            x_hyp = expmap0(x, c=c_val)
        return x_hyp


# ────────────────────────────────────────────────────────────
# HRQVAEWithHypPre wrapper
# ────────────────────────────────────────────────────────────
class HRQVAEWithHypPre(nn.Module):
    """Wraps HRQVAE with HypPreEncoder plugged in BEFORE self.encoder.

    Architecture:
        x (768d) → HypPreEncoder(expmap0) → encoder (MLP) → hrq (RQ-VAE) → decoder

    When self.hyp_pre.enabled = False, behavior is identical to plain HRQVAE
    (regression-safe).
    """

    def __init__(self, base_hrqvae: HRQVAE, c: float = 0.74, enabled: bool = True):
        super().__init__()
        self.base = base_hrqvae
        self.hyp_pre = HypPreEncoder(c=c, enabled=enabled)

    def forward(self, x, use_sk=True, rho_target_batch=None):
        x = self.hyp_pre(x)
        return self.base(x, use_sk=use_sk, rho_target_batch=rho_target_batch)

    @torch.no_grad()
    def get_indices(self, x, use_sk=True):
        x = self.hyp_pre(x)
        return self.base.get_indices(x, use_sk=use_sk)


# ────────────────────────────────────────────────────────────
# Regression tests
# ────────────────────────────────────────────────────────────
def test_t1_identity_when_disabled():
    """HypPreEncoder(enabled=False) must be exact identity."""
    print("\n─── T1: HypPreEncoder(enabled=False) is identity ───")
    torch.manual_seed(42)
    x = torch.randn(8, 768) * 0.5

    hyp_pre = HypPreEncoder(c=0.74, enabled=False)
    y = hyp_pre(x)

    diff = (x - y).abs().max().item()
    status = "✅ PASS" if diff < 1e-6 else "❌ FAIL"
    print(f"  max|x - y| = {diff:.6e}  {status}")
    return diff < 1e-6


def test_t2_poincare_ball_output():
    """HypPreEncoder(enabled=True) must map output INTO Poincaré ball ‖y‖ < 1/√c."""
    print("\n─── T2: Output ‖y‖ < 1/√c (Poincaré ball boundary) ───")
    torch.manual_seed(42)
    # Scale input so ‖x‖ < 1/√c (≈1.16). For 768d Gaussian, ‖x‖ ≈ √768·σ
    # → σ=0.04 → ‖x‖ ≈ 1.1. Use σ=0.03 to be safely inside.
    x = torch.randn(8, 768) * 0.03

    c = 0.74
    boundary = 1.0 / (c ** 0.5) - 1e-5  # = 1.155

    hyp_pre = HypPreEncoder(c=c, enabled=True)
    y = hyp_pre(x)

    y_norm = y.norm(dim=-1)
    max_norm = y_norm.max().item()
    min_norm = y_norm.min().item()
    in_ball = (y_norm < boundary).all().item()

    status = "✅ PASS" if in_ball else "❌ FAIL"
    print(f"  c=0.74, boundary={boundary:.4f}")
    print(f"  ‖y‖ range: [{min_norm:.4f}, {max_norm:.4f}]  {status}")
    return in_ball


def test_t3_expmap0_consistency():
    """HypPreEncoder must produce same result as expmap0 directly."""
    print("\n─── T3: HypPreEncoder matches expmap0 directly ───")
    torch.manual_seed(42)
    x = torch.randn(4, 768) * 0.5

    c = 0.74
    hyp_pre = HypPreEncoder(c=c, enabled=True)
    y_wrapper = hyp_pre(x)
    y_direct = expmap0(x, c=c)

    diff = (y_wrapper - y_direct).abs().max().item()
    status = "✅ PASS" if diff < 1e-5 else "❌ FAIL"
    print(f"  max|y_wrapper - y_direct| = {diff:.6e}  {status}")
    return diff < 1e-5


def test_t4_gradient_flow():
    """HypPreEncoder (learnable_c=True) must propagate gradient to c."""
    print("\n─── T4: Gradient flows to learnable c ───")
    torch.manual_seed(42)
    x = torch.randn(8, 768) * 0.03  # leaf tensor, will require_grad below

    hyp_pre = HypPreEncoder(c=0.74, enabled=True, learnable_c=True)
    y = hyp_pre(x)
    loss = y.pow(2).sum()
    loss.backward()

    grad_c = hyp_pre.log_c.grad.item() if hyp_pre.log_c.grad is not None else 0.0
    # x was a leaf but didn't require_grad; gradient through expmap0 still flows.
    # Test by checking if any tensor in the graph has grad:
    has_grad = any(p.grad is not None and p.grad.abs().max().item() > 1e-6
                   for p in hyp_pre.parameters())

    c_ok = abs(grad_c) > 1e-6
    grad_ok = has_grad
    status_c = "✅" if c_ok else "❌"
    status_g = "✅" if grad_ok else "❌"
    print(f"  ∂L/∂log_c    = {grad_c:.6f}  {status_c}")
    print(f"  Some param has grad > 1e-6: {has_grad}  {status_g}")
    return c_ok and grad_ok


def test_t5_wrapped_hrqvae_forward():
    """HRQVAEWithHypPre must run full forward pass with HypPreEncoder active."""
    print("\n─── T5: HRQVAEWithHypPre full forward pass ───")
    torch.manual_seed(42)
    in_dim = 768
    num_emb_list = [64, 128, 256]
    e_dim = 64
    layers = [512, 256, 128]

    # Note: HResidualVectorQuantization requires sk_eps as list (per-layer),
    # not scalar. Pass [0.0, 0.0, 0.0] to disable Sinkhorn.
    base = HRQVAE(in_dim=in_dim, num_emb_list=num_emb_list,
                  e_dim=e_dim, layers=layers, dropout_prob=0.0, bn=False,
                  loss_type='mse', quant_loss_weight=1.0, beta=0.25,
                  kmeans_init=False, kmeans_iters=10,
                  sk_eps=[0.0, 0.0, 0.0], sk_iters=3)
    wrapped = HRQVAEWithHypPre(base, c=0.74, enabled=True)

    # Input scaled to fit in Poincaré ball after expmap0
    x = torch.randn(4, 768) * 0.03
    out, rq_loss, indices, path_loss, _div_ent = wrapped(x, use_sk=False)

    print(f"  Input shape:      {x.shape}")
    print(f"  Output shape:     {out.shape}")
    print(f"  RQ loss:          {rq_loss.item():.4f}")
    print(f"  Indices shape:    {indices.shape}")

    shape_ok = (out.shape == x.shape and indices.shape == (4, len(num_emb_list)))
    print(f"  {'✅ PASS' if shape_ok else '❌ FAIL'}")
    return shape_ok


# ────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────
def main():
    print("=" * 70)
    print("Task #334 — Issue #43 Gate 2a HypPreEncoder regression test")
    print("=" * 70)

    results = {}
    try:
        results["T1_identity_when_disabled"] = test_t1_identity_when_disabled()
    except Exception as e:
        print(f"  ❌ T1 raised: {e}")
        results["T1_identity_when_disabled"] = False

    try:
        results["T2_poincare_ball_output"] = test_t2_poincare_ball_output()
    except Exception as e:
        print(f"  ❌ T2 raised: {e}")
        results["T2_poincare_ball_output"] = False

    try:
        results["T3_expmap0_consistency"] = test_t3_expmap0_consistency()
    except Exception as e:
        print(f"  ❌ T3 raised: {e}")
        results["T3_expmap0_consistency"] = False

    try:
        results["T4_gradient_flow"] = test_t4_gradient_flow()
    except Exception as e:
        print(f"  ❌ T4 raised: {e}")
        results["T4_gradient_flow"] = False

    try:
        results["T5_wrapped_hrqvae_forward"] = test_t5_wrapped_hrqvae_forward()
    except Exception as e:
        print(f"  ❌ T5 raised: {e}")
        results["T5_wrapped_hrqvae_forward"] = False

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    n_pass = sum(1 for v in results.values() if v)
    n_total = len(results)
    for name, ok in results.items():
        status = "✅ PASS" if ok else "❌ FAIL"
        print(f"  {name:35s}  {status}")
    print(f"\n  Total: {n_pass}/{n_total} tests passed")

    sys.exit(0 if n_pass == n_total else 1)


if __name__ == "__main__":
    main()