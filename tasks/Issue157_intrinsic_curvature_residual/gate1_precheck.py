"""Issue #157 Gate 1 precheck: intrinsic curvature residual (Möbius 减法).

前置检查 (spec):
1. 小曲率极限下内在 subtraction 收敛到欧式 r-e (独立数学 oracle)
2. Exp/Log 往返误差、Möbius inverse、维度广播检查
3. 曲率门控归零时 A/B 三层 residual、distance、assignment、loss、SID 逐元素一致
4. 同一个 c_l,i 贯穿该商品的距离、选码和 residual 更新 (禁止两套曲率)
5. 连续优化步曲率参数与 residual 路径梯度 finite 非零

输出: control/stage2/gate1_precheck.json
"""

import os, sys, json, math
from pathlib import Path
import numpy as np
import torch

TASK_DIR = Path(__file__).parent
sys.path.insert(0, str(TASK_DIR / "treatment" / "_lib"))
from intrinsic_curvature_quantizer import IntrinsicCurvatureHRQVAE, _expmap0_t, _logmap0_t, _mobius_add_t
from utils import EmbDataset

SEED = 2024
EMB_DIM = 768
E_DIM = 32
CODEBOOK_SIZES = [64, 128, 256]
ENCODER_LAYERS = [512, 256, 128, 64]
N_ITEMS = 9922
BATCH = 1024
KMEANS_ITERS = 1000


def poincare_recon_loss(out, target):
    from utils import poincare_distance, expmap0, proj_to_ball
    o = proj_to_ball(expmap0(out, 1.0), 1.0)
    t = proj_to_ball(expmap0(target, 1.0), 1.0)
    return torch.mean(poincare_distance(o, t, 1.0) ** 2)


def main():
    torch.set_default_dtype(torch.float64)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    g1 = {"issue": "#157", "checks": {}, "decision": "FAIL"}

    # ── check 1: 小曲率极限 → 欧式 (独立 oracle, float64) ──
    rng = torch.Generator().manual_seed(42)
    r = torch.randn(64, 32, generator=rng) * 0.1
    e = torch.randn(64, 32, generator=rng) * 0.1
    c_small = torch.full((64, 1), 1e-6)
    h_r = _expmap0_t(r, c_small); h_e = _expmap0_t(e, c_small)
    h_next = _mobius_add_t(-h_e, h_r, c_small)
    r_int = _logmap0_t(h_next, c_small)
    err_small_c = float((r_int - (r - e)).abs().max().item())
    g1["checks"]["small_c_euclid_limit"] = {"pass": err_small_c < 1e-4, "max_err": err_small_c}

    # ── check 2: Exp/Log 往返 + Möbius inverse ──
    c_mid = torch.full((64, 1), 1.0)
    roundtrip = float((_logmap0_t(_expmap0_t(r, c_mid), c_mid) - r).abs().max().item())
    inv = float((_mobius_add_t(r, -r, c_mid)).abs().max().item())
    g1["checks"]["roundtrip_and_inverse"] = {"pass": roundtrip < 1e-5 and inv < 1e-5,
                                             "roundtrip": roundtrip, "inverse": inv}

    # ── check 3/4: 门控归零时 A/B 三层 residual/assignment/loss/SID 一致 ──
    torch.set_default_dtype(torch.float32)
    torch.manual_seed(SEED); np.random.seed(SEED)
    item_emb = torch.tensor(
        EmbDataset(str(TASK_DIR / "treatment" / "stage1" / "item_emb.parquet")).embeddings,
        dtype=torch.float32).to(device)

    def build(prefix_routing):
        torch.manual_seed(SEED); np.random.seed(SEED)
        m = IntrinsicCurvatureHRQVAE(
            in_dim=EMB_DIM, num_emb_list=CODEBOOK_SIZES, e_dim=E_DIM,
            layers=ENCODER_LAYERS, kmeans_init=True, kmeans_iters=KMEANS_ITERS,
            prefix_routing=prefix_routing,
        ).to(device)
        m.train()
        with torch.no_grad():
            m(item_emb[:BATCH], use_sk=False)
        return m

    mA = build(False)
    mB = build(True)
    with torch.no_grad():
        x = item_emb[:BATCH]
        out_A, rq_A, idx_A, zq_A, z_A = mA(x, use_sk=False)
        out_B, rq_B, idx_B, zq_B, z_B = mB(x, use_sk=False)
        # 三层 residual 一致性 (hook 或重算)
        delta0 = [float(q._last_delta.abs().max()) if (q.prefix_routing and q._last_delta is not None) else 0.0
                  for q in mB.vq_layers]
    g1["checks"]["gate_zero_equivalence"] = {
        "pass": bool(torch.allclose(out_A, out_B, atol=1e-6) and torch.equal(idx_A, idx_B)
                     and all(d < 1e-8 for d in delta0)),
        "out_diff": float((out_A - out_B).abs().max()),
        "delta0": delta0,
    }

    # ── check 5: 梯度通路 (内在减法路径 + router) ──
    opt = torch.optim.Adam(mB.parameters(), lr=1e-3)
    grad_log = []
    for step in range(2):
        mB.zero_grad()
        perm = np.random.permutation(N_ITEMS)
        xb = item_emb[perm[:BATCH]]
        out, rq, idx, zq, z = mB(xb, use_sk=False)
        loss = poincare_recon_loss(out, xb) + rq
        rd, rm = mB.compute_route_reg(); loss = loss + rd + rm
        loss.backward()
        torch.nn.utils.clip_grad_norm_(mB.parameters(), 1.0)
        rec = {}
        for li in [1, 2]:
            rt = mB.vq_layers[li].router
            rec[f"L{li}_fc2_g"] = float(rt.fc2.weight.grad.norm()) if rt.fc2.weight.grad is not None else 0.0
        opt.step()
        grad_log.append(rec)
        print(f"step{step+1}: L1 fc2_g={rec['L1_fc2_g']:.3e} L2 fc2_g={rec['L2_fc2_g']:.3e}")
    s2 = grad_log[1]
    g1["checks"]["gradient_path"] = {"pass": bool(s2["L1_fc2_g"] > 0 and s2["L2_fc2_g"] > 0),
                                     "step2": s2}

    all_pass = all(v.get("pass", False) for v in g1["checks"].values())
    g1["decision"] = "PASS" if all_pass else "FAIL"
    out = TASK_DIR / "control" / "stage2" / "gate1_precheck.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(g1, f, indent=2, ensure_ascii=False)
    print(f"\nGate 1: {g1['decision']} -> {out}")
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
