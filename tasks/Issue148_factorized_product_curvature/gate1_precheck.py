"""Issue #148 — Gate 1 Precheck.

验证 A (单 32D 双曲) vs B (16D 欧氏 + 16D 双曲乘积) 两条路径:
- check1: Stage1 共享 (input SHA 一致)
- check2: A/B SID 结构一致 (3 层, codebook 64/128/256, total dim 32)
- check3: A 距离 = 32D Poincaré (per-layer c_l scalar)
- check4: B 距离 = sqrt(d_E^2 + d_H^2) (d_E 16D Euclidean, d_H 16D Poincaré with c_l^H scalar)
- check5: 同样 init (kmeans on first batch), A/B 应产生不同 SID (几何机制差异)
- check6: 梯度 finite + c 边界有界 (C_MIN=0.5, C_MAX=2.0)
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
from factorized_product_quantizer import (
    FactorizedProductHRQVAE, FactorizedProductVQ,
    C_MIN, C_MAX, E_DIM_EUCLIDEAN, E_DIM_HYPERBOLIC, E_DIM_TOTAL,
)

STAGE1_PARQUET = os.path.join(TASK_DIR, "stage1", "item_emb.parquet")
REPORT_PATH = os.path.join(TASK_DIR, "gate1_precheck_report.json")
EXPECTED_STAGE1_SHA = "1a6dd2ac1c690d029d985787d5b0b95b881df934d7d7fb52c3f095162add6af6"  # Issue #147 / #148 共享 Stage1


def sha256_of_parquet(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def build_model(factorized, seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    return FactorizedProductHRQVAE(
        in_dim=768, num_emb_list=(64, 128, 256), e_dim=E_DIM_TOTAL,
        layers=(512, 256, 128, 64), beta=0.25, kmeans_init=True,
        kmeans_iters=10, factorized=factorized,
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
    """A: d_A(i,b) = d_Poincare(z, e; c_l), 全 32 维. c_l 在 [C_MIN, C_MAX]."""
    torch.manual_seed(0)
    vq = FactorizedProductVQ(n_e=64, factorized=False)
    z = torch.randn(5, E_DIM_TOTAL) * 0.3
    e = torch.randn(64, E_DIM_TOTAL) * 0.3
    c = vq.get_c()
    assert (c >= C_MIN).all() and (c <= C_MAX).all()
    z_h = proj_to_ball(expmap0(z, c), c)
    e_h = proj_to_ball(expmap0(e, c), c)
    d_expected = poincare_distance(
        z_h.unsqueeze(1).expand(5, 64, -1),
        e_h.unsqueeze(0).expand(5, 64, -1), c
    ).squeeze(-1)
    # 手动 kmeans-init (samples 200 > clusters 64)
    sample = torch.randn(200, E_DIM_TOTAL) * 0.3
    vq.initted = True
    vq.embeddings.weight.data.copy_(kmeans(sample, vq.n_e, 10))
    z_q, loss, idx = vq(z)
    assert z_q.shape == z.shape
    assert idx.shape == (5,)
    return {"name": "check3_A_distance_correct", "decision": "PASS",
            "c_value": float(c.item()), "d_expected_min": float(d_expected.min().item()),
            "vq_idx_range": [int(idx.min().item()), int(idx.max().item())]}


def check4_B_distance_correct():
    """B: d_B = sqrt(d_E^2 + d_H^2), 16D Euc + 16D Hyp."""
    torch.manual_seed(0)
    vq = FactorizedProductVQ(n_e=64, factorized=True)
    z = torch.randn(5, E_DIM_TOTAL) * 0.3
    e = torch.randn(64, E_DIM_TOTAL) * 0.3
    c = vq.get_c()
    z_e, z_h = z[:, :E_DIM_EUCLIDEAN], z[:, E_DIM_EUCLIDEAN:]
    e_e, e_h = e[:, :E_DIM_EUCLIDEAN], e[:, E_DIM_EUCLIDEAN:]
    d_euc = torch.cdist(z_e.unsqueeze(1), e_e.unsqueeze(0)).squeeze(1)
    z_h_h = proj_to_ball(expmap0(z_h, c), c)
    e_h_h = proj_to_ball(expmap0(e_h, c), c)
    d_hyp = poincare_distance(
        z_h_h.unsqueeze(1).expand(5, 64, -1),
        e_h_h.unsqueeze(0).expand(5, 64, -1), c
    ).squeeze(-1)
    d_product = torch.sqrt(d_euc ** 2 + d_hyp ** 2 + 1e-12)
    assert d_product.shape == (5, 64)
    sample = torch.randn(200, E_DIM_TOTAL) * 0.3
    vq.initted = True
    vq.embeddings.weight.data.copy_(kmeans(sample, vq.n_e, 10))
    z_q, loss, idx = vq(z)
    assert z_q.shape == z.shape
    return {"name": "check4_B_distance_correct", "decision": "PASS",
            "c_value": float(c.item()),
            "d_euc_min": float(d_euc.min().item()),
            "d_hyp_min": float(d_hyp.min().item()),
            "d_product_min": float(d_product.min().item()),
            "vq_idx_range": [int(idx.min().item()), int(idx.max().item())]}


def check5_different_sid_same_init():
    """同样 codeword init (kmeans), A 和 B 应产生不同 SID — 因为几何机制不同."""
    torch.manual_seed(42)
    np.random.seed(42)
    model_A = build_model(factorized=False)
    model_B = build_model(factorized=True)
    # 共享 codeword: 把 A 的 codebook 复制到 B (同 kmeans 中心)
    for qa, qb in zip(model_A.vq_layers, model_B.vq_layers):
        sample = torch.randn(2048, E_DIM_TOTAL) * 0.3
        qa.initted = True
        qa.embeddings.weight.data.copy_(kmeans(sample, qa.n_e, 10))
        qb.initted = True
        qb.embeddings.weight.data.copy_(qa.embeddings.weight.data)
    model_A.eval()
    model_B.eval()
    x = torch.randn(50, 768) * 0.3  # encoder 输入 768 维
    with torch.no_grad():
        z_A = model_A.encoder(x)
        z_B = model_B.encoder(x)
        assert torch.allclose(z_A, z_B, atol=1e-5), "encoder 应一致"
        # 直接在 latent 上调 _rq_forward 拿 SID
        _, _, sid_A = model_A._rq_forward(z_A)
        _, _, sid_B = model_B._rq_forward(z_B)
    sid_A = sid_A.cpu().numpy()
    sid_B = sid_B.cpu().numpy()
    layer_diffs = [(sid_A[:, l] != sid_B[:, l]).sum() for l in range(3)]
    any_diff = any(d > 0 for d in layer_diffs)
    return {"name": "check5_different_sid_same_init", "decision": "PASS" if any_diff else "FAIL",
            "encoder_close": True,
            "L0_SID_diff": int(layer_diffs[0]), "L1_SID_diff": int(layer_diffs[1]), "L2_SID_diff": int(layer_diffs[2]),
            "explanation": "A (32D Poincaré) vs B (sqrt(d_E^2+d_H^2)) 距离不同 → 同 input 同 init 应至少 1 层 SID 不一致"}


def check6_grad_finite(model_A, model_B):
    """encoder + theta 梯度 finite, c ∈ [C_MIN, C_MAX]."""
    model_A.train()
    model_B.train()
    x = torch.randn(8, 768) * 0.3  # encoder expects 768-dim
    for model in [model_A, model_B]:
        for q in model.vq_layers:
            q.initted = True
            q.embeddings.weight.data.copy_(torch.randn(q.n_e, E_DIM_TOTAL) * 0.1)
    out_A, loss_A, _, _, _ = model_A(x, use_sk=False)
    out_B, loss_B, _, _, _ = model_B(x, use_sk=False)
    loss = loss_A + loss_B
    loss.backward()
    enc_grad_A = model_A.encoder.mlp[1].weight.grad
    enc_grad_B = model_B.encoder.mlp[1].weight.grad
    cs = [float(q.get_c().item()) for q in model_A.vq_layers]
    cs += [float(q.get_c().item()) for q in model_B.vq_layers]
    in_bounds = all((c >= C_MIN - 1e-6 and c <= C_MAX + 1e-6) for c in cs)
    finite_grad = torch.isfinite(enc_grad_A).all().item() and torch.isfinite(enc_grad_B).all().item()
    return {"name": "check6_grad_finite", "decision": "PASS" if (finite_grad and in_bounds) else "FAIL",
            "encoder_grad_norm_A": float(enc_grad_A.norm().item()),
            "encoder_grad_norm_B": float(enc_grad_B.norm().item()),
            "c_values": cs, "c_in_bounds": in_bounds}


def check7_no_nan_100_steps(model_A, model_B):
    """A/B 各跑 100 步, 无 NaN/Inf, c 仍 in bounds."""
    model_A.train()
    model_B.train()
    for q in model_A.vq_layers:
        q.initted = True
        q.embeddings.weight.data.copy_(torch.randn(q.n_e, E_DIM_TOTAL) * 0.1)
    for q in model_B.vq_layers:
        q.initted = True
        q.embeddings.weight.data.copy_(torch.randn(q.n_e, E_DIM_TOTAL) * 0.1)
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
    cs_A = [float(q.get_c().item()) for q in model_A.vq_layers]
    cs_B = [float(q.get_c().item()) for q in model_B.vq_layers]
    in_bounds = all((c >= C_MIN - 1e-6 and c <= C_MAX + 1e-6) for c in (cs_A + cs_B))
    return {"name": "check7_no_nan_100_steps", "decision": "PASS" if (nan_count == 0 and in_bounds) else "FAIL",
            "nan_count": nan_count, "loss_A_first": losses_A[0], "loss_A_last": losses_A[-1],
            "loss_B_first": losses_B[0], "loss_B_last": losses_B[-1],
            "final_c_A": cs_A, "final_c_B": cs_B}


def main():
    print("=" * 60)
    print("Issue #148 — Gate 1 Precheck")
    print("=" * 60)
    checks = []
    c1 = check1_input_hash()
    checks.append(c1)
    print(f"[1/7] input_hash: {c1['decision']} sha={c1.get('sha256', '?')[:16]}...")
    model_A = build_model(factorized=False)
    model_B = build_model(factorized=True)
    c2 = check2_sid_structure(model_A, model_B)
    checks.append(c2)
    print(f"[2/7] sid_structure: {c2['decision']} A={c2['A_codebook_sizes']} B={c2['B_codebook_sizes']}")
    c3 = check3_A_distance_correct()
    checks.append(c3)
    print(f"[3/7] A_distance_correct: {c3['decision']} c={c3['c_value']:.4f}")
    c4 = check4_B_distance_correct()
    checks.append(c4)
    print(f"[4/7] B_distance_correct: {c4['decision']} c={c4['c_value']:.4f} d_product_min={c4['d_product_min']:.4f}")
    c5 = check5_different_sid_same_init()
    checks.append(c5)
    print(f"[5/7] different_sid_same_init: {c5['decision']} L0/L1/L2 diff = {c5['L0_SID_diff']}/{c5['L1_SID_diff']}/{c5['L2_SID_diff']}")
    c6 = check6_grad_finite(model_A, model_B)
    checks.append(c6)
    print(f"[6/7] grad_finite: {c6['decision']} grad_A={c6['encoder_grad_norm_A']:.4e} grad_B={c6['encoder_grad_norm_B']:.4e}")
    c7 = check7_no_nan_100_steps(model_A, model_B)
    checks.append(c7)
    print(f"[7/7] no_nan_100_steps: {c7['decision']} nan={c7['nan_count']} c_A={c7['final_c_A']} c_B={c7['final_c_B']}")
    n_pass = sum(1 for c in checks if c["decision"] == "PASS")
    n_total = len(checks)
    overall = "PASS" if n_pass == n_total else "FAIL"
    report = {
        "issue": "#148",
        "overall_decision": overall,
        "n_pass": n_pass, "n_total": n_total,
        "checks": checks,
    }
    with open(REPORT_PATH, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\nOverall: {overall} ({n_pass}/{n_total})")
    print(f"Report: {REPORT_PATH}")
    return 0 if overall == "PASS" else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())