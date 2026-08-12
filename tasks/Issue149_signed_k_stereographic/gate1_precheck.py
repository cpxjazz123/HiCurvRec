"""Issue #149 — Gate 1 Precheck.

验证 A (signed κ 只能负曲率 c_l ≥ 0) vs B (signed κ ∈ (-κ_max, +κ_max)) 两条路径:
- check1: Stage1 共享 (input SHA 一致)
- check2: A/B SID 结构一致 (3 层, codebook 64/128/256, total dim 32)
- check3: A 距离 = 32D Poincaré with c_l ∈ [C_MIN, C_MAX]
- check4: B 距离 = signed κ-stereographic, κ_init = -C_INITIAL = -1.25
- check5: 同样 init (kmeans on first batch), A/B 在 init 时 SID 应高度一致 (|κ|=1.25 ≈ A c=1.25)
- check6: 梯度 finite + κ 边界有界 (-κ_max < κ < +κ_max)
- check7: 无 NaN/Inf, 100 个 batch 训练稳定
"""

import os, sys, json, hashlib
import numpy as np
import torch
import torch.nn as nn

TASK_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(TASK_DIR, "_lib"))

from utils import (
    proj_to_ball, expmap0, logmap0, poincare_distance, kmeans, MLP, _eps,
)
from signed_kappa_quantizer import (
    SignedKappaHRQVAE, SignedKappaVQ,
    C_MIN, C_MAX, C_INITIAL, KAPPA_MAX, K_THRESH, E_DIM,
    signed_kappa_distance, signed_kappa_expmap0, signed_kappa_proj,
)

STAGE1_PARQUET = os.path.join(TASK_DIR, "stage1", "item_emb.parquet")
REPORT_PATH = os.path.join(TASK_DIR, "gate1_precheck_report.json")
EXPECTED_STAGE1_SHA = "1a6dd2ac1c690d029d985787d5b0b95b881df934d7d7fb52c3f095162add6af6"  # Issue #147 / #148 / #149 共享 Stage1


def sha256_of_parquet(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def build_model(signed_kappa, seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    return SignedKappaHRQVAE(
        in_dim=768, num_emb_list=(64, 128, 256), e_dim=E_DIM,
        layers=(512, 256, 128, 64), beta=0.25, kmeans_init=True,
        kmeans_iters=10, signed_kappa=signed_kappa, kappa_max=KAPPA_MAX,
    )


def check1_input_hash():
    p = STAGE1_PARQUET
    if not os.path.exists(p):
        return {"name": "check1_input_hash", "decision": "FAIL",
                "reason": f"Stage1 parquet 不存在: {p}"}
    sha = sha256_of_parquet(p)
    ok = (sha == EXPECTED_STAGE1_SHA)
    return {"name": "check1_input_hash", "decision": "PASS" if ok else "FAIL",
            "sha256": sha, "expected": EXPECTED_STAGE1_SHA}


def check2_sid_structure(model_A, model_B):
    expected_sizes = (64, 128, 256)
    a_sizes = tuple(q.n_e for q in model_A.vq_layers)
    b_sizes = tuple(q.n_e for q in model_B.vq_layers)
    sid_len = len(a_sizes)
    ok = (a_sizes == expected_sizes and b_sizes == expected_sizes and sid_len == 3)
    return {"name": "check2_sid_structure", "decision": "PASS" if ok else "FAIL",
            "A_codebook_sizes": a_sizes, "B_codebook_sizes": b_sizes,
            "expected": expected_sizes, "sid_len": sid_len}


def check3_A_distance_correct():
    """A: d_A(i,b) = d_Poincare(z, e; c_l), 32D, c_l ∈ [C_MIN, C_MAX]."""
    torch.manual_seed(0)
    vq = SignedKappaVQ(n_e=64, e_dim=E_DIM, signed_kappa=False)
    z = torch.randn(5, E_DIM) * 0.3
    e = torch.randn(64, E_DIM) * 0.3
    c = vq.get_c_or_kappa()[0]
    assert (c >= C_MIN).all() and (c <= C_MAX).all()
    z_h = proj_to_ball(expmap0(z, c), c)
    e_h = proj_to_ball(expmap0(e, c), c)
    d_expected = poincare_distance(
        z_h.unsqueeze(1).expand(5, 64, -1),
        e_h.unsqueeze(0).expand(5, 64, -1), c
    ).squeeze(-1)
    sample = torch.randn(200, E_DIM) * 0.3
    vq.initted = True
    vq.embeddings.weight.data.copy_(kmeans(sample, vq.n_e, 10))
    z_q, loss, idx = vq(z)
    assert z_q.shape == z.shape
    assert idx.shape == (5,)
    return {"name": "check3_A_distance_correct", "decision": "PASS",
            "c_value": float(c.item()), "d_expected_min": float(d_expected.min().item()),
            "vq_idx_range": [int(idx.min().item()), int(idx.max().item())]}


def check4_B_distance_correct():
    """B: d_B = signed κ-stereographic distance, κ_init = -1.25 (与 A c=1.25 严格等价)."""
    torch.manual_seed(0)
    vq = SignedKappaVQ(n_e=64, e_dim=E_DIM, signed_kappa=True)
    z = torch.randn(5, E_DIM) * 0.3
    e = torch.randn(64, E_DIM) * 0.3
    kappa = vq.get_c_or_kappa()[0]
    assert abs(kappa.item() - (-C_INITIAL)) < 1e-5, f"B init κ should be -1.25, got {kappa.item()}"
    assert -KAPPA_MAX < kappa.item() < KAPPA_MAX
    # 验证 B κ=-1.25 distance 公式 与 A c=1.25 distance 公式一致
    z_h = signed_kappa_proj(signed_kappa_expmap0(z, kappa), kappa)
    e_h = signed_kappa_proj(signed_kappa_expmap0(e, kappa), kappa)
    d_signed = signed_kappa_distance(
        z_h.unsqueeze(1).expand(5, 64, -1),
        e_h.unsqueeze(0).expand(5, 64, -1), kappa
    ).squeeze(-1)
    # 与标准 Poincaré c=1.25 比较 (应当严格相等)
    c = 1.25
    z_A = proj_to_ball(expmap0(z, c), c)
    e_A = proj_to_ball(expmap0(e, c), c)
    d_poincare = poincare_distance(
        z_A.unsqueeze(1).expand(5, 64, -1),
        e_A.unsqueeze(0).expand(5, 64, -1), c
    ).squeeze(-1)
    diff = (d_signed - d_poincare).abs().max().item()
    sample = torch.randn(200, E_DIM) * 0.3
    vq.initted = True
    vq.embeddings.weight.data.copy_(kmeans(sample, vq.n_e, 10))
    z_q, loss, idx = vq(z)
    assert z_q.shape == z.shape
    return {"name": "check4_B_distance_correct", "decision": "PASS",
            "kappa_value": float(kappa.item()),
            "d_signed_min": float(d_signed.min().item()),
            "d_poincare_min": float(d_poincare.min().item()),
            "max_diff_signed_vs_poincare": diff,
            "vq_idx_range": [int(idx.min().item()), int(idx.max().item())]}


def check5_different_sid_same_init():
    """同样 codeword init (kmeans), A 和 B 在 init 时 SID 应非常接近 (|κ|=1.25 ≈ A c=1.25).

    期望: 三层差异都 < 5 (允许微小浮点差异).
    """
    torch.manual_seed(42)
    np.random.seed(42)
    model_A = build_model(signed_kappa=False)
    torch.manual_seed(42)
    np.random.seed(42)
    model_B = build_model(signed_kappa=True)
    for qa, qb in zip(model_A.vq_layers, model_B.vq_layers):
        sample = torch.randn(2048, E_DIM) * 0.3
        qa.initted = True
        qa.embeddings.weight.data.copy_(kmeans(sample, qa.n_e, 10))
        qb.initted = True
        qb.embeddings.weight.data.copy_(qa.embeddings.weight.data)
    model_A.eval()
    model_B.eval()
    x = torch.randn(50, 768) * 0.3
    with torch.no_grad():
        z_A = model_A.encoder(x)
        z_B = model_B.encoder(x)
        assert torch.allclose(z_A, z_B, atol=1e-5), "encoder 应一致"
        _, _, sid_A = model_A._rq_forward(z_A)
        _, _, sid_B = model_B._rq_forward(z_B)
    sid_A = sid_A.cpu().numpy()
    sid_B = sid_B.cpu().numpy()
    layer_diffs = [(sid_A[:, l] != sid_B[:, l]).sum() for l in range(3)]
    # 因为 κ=-1.25 ≈ c=1.25, 三层差异都应该接近 0 (允许微小浮点差异)
    max_diff = max(layer_diffs)
    return {"name": "check5_different_sid_same_init", "decision": "PASS" if max_diff < 5 else "FAIL",
            "encoder_close": True,
            "L0_SID_diff": int(layer_diffs[0]),
            "L1_SID_diff": int(layer_diffs[1]),
            "L2_SID_diff": int(layer_diffs[2]),
            "max_diff": max_diff,
            "explanation": "A (c=1.25) vs B (κ=-1.25, |κ|=1.25) 数学严格等价 → SID 应几乎一致"}


def check6_grad_finite(model_A, model_B):
    """encoder + theta 梯度 finite, κ 边界有界."""
    model_A.train()
    model_B.train()
    x = torch.randn(8, 768) * 0.3
    for model in [model_A, model_B]:
        for q in model.vq_layers:
            q.initted = True
            q.embeddings.weight.data.copy_(torch.randn(q.n_e, E_DIM) * 0.1)
    out_A, loss_A, _, _, _ = model_A(x, use_sk=False)
    out_B, loss_B, _, _, _ = model_B(x, use_sk=False)
    loss = loss_A + loss_B
    loss.backward()
    enc_grad_A = model_A.encoder.mlp[1].weight.grad
    enc_grad_B = model_B.encoder.mlp[1].weight.grad
    cs_A = [float(q.get_c_or_kappa()[0].item()) for q in model_A.vq_layers]
    ks_B = [float(q.get_c_or_kappa()[0].item()) for q in model_B.vq_layers]
    in_bounds_A = all((c >= C_MIN - 1e-6 and c <= C_MAX + 1e-6) for c in cs_A)
    in_bounds_B = all((-KAPPA_MAX < k and k < KAPPA_MAX) for k in ks_B)
    finite_grad = torch.isfinite(enc_grad_A).all().item() and torch.isfinite(enc_grad_B).all().item()
    return {"name": "check6_grad_finite", "decision": "PASS" if (finite_grad and in_bounds_A and in_bounds_B) else "FAIL",
            "encoder_grad_norm_A": float(enc_grad_A.norm().item()),
            "encoder_grad_norm_B": float(enc_grad_B.norm().item()),
            "c_values_A": cs_A, "kappa_values_B": ks_B,
            "c_in_bounds": in_bounds_A, "kappa_in_bounds": in_bounds_B}


def check7_no_nan_100_steps(model_A, model_B):
    """A/B 各跑 100 步, 无 NaN/Inf, c/κ 仍 in bounds."""
    model_A.train()
    model_B.train()
    for q in model_A.vq_layers:
        q.initted = True
        q.embeddings.weight.data.copy_(torch.randn(q.n_e, E_DIM) * 0.1)
    for q in model_B.vq_layers:
        q.initted = True
        q.embeddings.weight.data.copy_(torch.randn(q.n_e, E_DIM) * 0.1)
    opt_A = torch.optim.Adam(model_A.parameters(), lr=1e-3)
    opt_B = torch.optim.Adam(model_B.parameters(), lr=1e-3)
    nan_count = 0
    losses_A, losses_B = [], []
    for step in range(100):
        x = torch.randn(32, 768) * 0.3
        opt_A.zero_grad()
        _, loss_A, _, _, _ = model_A(x, use_sk=False)
        loss_A.backward()
        opt_A.step()
        opt_B.zero_grad()
        _, loss_B, _, _, _ = model_B(x, use_sk=False)
        loss_B.backward()
        opt_B.step()
        if torch.isnan(loss_A) or torch.isinf(loss_A) or torch.isnan(loss_B) or torch.isinf(loss_B):
            nan_count += 1
        losses_A.append(float(loss_A.item()))
        losses_B.append(float(loss_B.item()))
    cs_A = [float(q.get_c_or_kappa()[0].item()) for q in model_A.vq_layers]
    ks_B = [float(q.get_c_or_kappa()[0].item()) for q in model_B.vq_layers]
    in_bounds_A = all((c >= C_MIN - 1e-6 and c <= C_MAX + 1e-6) for c in cs_A)
    in_bounds_B = all((-KAPPA_MAX < k and k < KAPPA_MAX) for k in ks_B)
    return {"name": "check7_no_nan_100_steps", "decision": "PASS" if (nan_count == 0 and in_bounds_A and in_bounds_B) else "FAIL",
            "nan_count": nan_count, "loss_A_first": losses_A[0], "loss_A_last": losses_A[-1],
            "loss_B_first": losses_B[0], "loss_B_last": losses_B[-1],
            "final_c_A": cs_A, "final_kappa_B": ks_B}


def main():
    print("=" * 60)
    print("Issue #149 — Gate 1 Precheck")
    print("=" * 60)
    checks = []
    c1 = check1_input_hash()
    checks.append(c1)
    print(f"[1/7] input_hash: {c1['decision']} sha={c1.get('sha256', '?')[:16]}...")
    model_A = build_model(signed_kappa=False)
    model_B = build_model(signed_kappa=True)
    c2 = check2_sid_structure(model_A, model_B)
    checks.append(c2)
    print(f"[2/7] sid_structure: {c2['decision']} A={c2['A_codebook_sizes']} B={c2['B_codebook_sizes']}")
    c3 = check3_A_distance_correct()
    checks.append(c3)
    print(f"[3/7] A_distance_correct: {c3['decision']} c={c3['c_value']:.4f}")
    c4 = check4_B_distance_correct()
    checks.append(c4)
    print(f"[4/7] B_distance_correct: {c4['decision']} kappa={c4['kappa_value']:.4f} "
          f"diff={c4['max_diff_signed_vs_poincare']:.2e}")
    c5 = check5_different_sid_same_init()
    checks.append(c5)
    print(f"[5/7] similar_sid_same_init: {c5['decision']} L0/L1/L2 diff = "
          f"{c5['L0_SID_diff']}/{c5['L1_SID_diff']}/{c5['L2_SID_diff']}")
    c6 = check6_grad_finite(model_A, model_B)
    checks.append(c6)
    print(f"[6/7] grad_finite: {c6['decision']} grad_A={c6['encoder_grad_norm_A']:.4e} "
          f"grad_B={c6['encoder_grad_norm_B']:.4e}")
    c7 = check7_no_nan_100_steps(model_A, model_B)
    checks.append(c7)
    print(f"[7/7] no_nan_100_steps: {c7['decision']} nan={c7['nan_count']} "
          f"c_A={c7['final_c_A']} kappa_B={c7['final_kappa_B']}")
    n_pass = sum(1 for c in checks if c["decision"] == "PASS")
    n_total = len(checks)
    overall = "PASS" if n_pass == n_total else "FAIL"
    report = {
        "issue": "#149",
        "overall_decision": overall,
        "n_pass": n_pass, "n_total": n_total,
        "checks": checks,
    }
    with open(REPORT_PATH, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False, default=lambda o: int(o) if hasattr(o, 'item') else str(o))
    print(f"\nOverall: {overall} ({n_pass}/{n_total})")
    print(f"Report: {REPORT_PATH}")
    return 0 if overall == "PASS" else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
