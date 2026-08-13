"""Issue #158 Gate 1 precheck: cross-layer curvature transport.

前置检查 (spec):
1. c_l=c_(l+1) 时曲率转换为恒等映射 (有限精差容差)
2. c_a→c_b→c_a 循环转换恢复原表示 (cycle error < 容差)
3. transport gate 关闭时 A/B residual、assignment、loss、SID 逐元素一致
4. 固定 residual/prefix, 只改上一层曲率 → 下一层表示产生可测响应
5. 曲率参数/转换路径/router 梯度 finite 非零

输出: control/stage2/gate1_precheck.json
"""

import os, sys, json
from pathlib import Path
import numpy as np
import torch

TASK_DIR = Path(__file__).parent
sys.path.insert(0, str(TASK_DIR / "treatment" / "_lib"))
from cross_layer_quantizer import CrossLayerHRQVAE, _expmap0_t, _logmap0_t
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
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    g1 = {"issue": "#158", "checks": {}, "decision": "FAIL"}

    # ── check 1/2: transport 数学 (identity + cycle) ──
    rng = torch.Generator().manual_seed(42)
    r = torch.randn(100, 32, generator=rng) * 0.1
    c_eq = torch.full((100, 1), 0.61)
    h = _expmap0_t(r, c_eq); u = _logmap0_t(h, c_eq); r_back = _expmap0_t(u, c_eq)
    id_err = float((r_back - r).abs().max().item())
    c_a = torch.full((100, 1), 0.6); c_b = torch.full((100, 1), 1.5)
    u_a = _logmap0_t(_expmap0_t(r, c_a), c_a)
    h_b = _expmap0_t(u_a, c_b); u_b = _logmap0_t(h_b, c_b)
    r_cycle = _expmap0_t(u_b, c_a)
    cyc_err = float((r_cycle - r).abs().max().item())
    g1["checks"]["transport_identity"] = {"pass": id_err < 0.05, "id_err": id_err}
    g1["checks"]["transport_cycle"] = {"pass": cyc_err < 0.05, "cycle_err": cyc_err}

    # ── check 3: transport 关闭时 A/B 等价 (transport gate 由 prefix_routing=False 模拟) ──
    torch.manual_seed(SEED); np.random.seed(SEED)
    item_emb = torch.tensor(
        EmbDataset(str(TASK_DIR / "treatment" / "stage1" / "item_emb.parquet")).embeddings,
        dtype=torch.float32).to(device)

    def build(prefix_routing):
        torch.manual_seed(SEED); np.random.seed(SEED)
        m = CrossLayerHRQVAE(
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
        delta0 = [float(q._last_delta.abs().max()) if (q.prefix_routing and q._last_delta is not None) else 0.0
                  for q in mB.vq_layers]
    g1["checks"]["gate_zero_equivalence"] = {
        "pass": bool(torch.allclose(out_A, out_B, atol=1e-6) and torch.equal(idx_A, idx_B)
                     and all(d < 1e-8 for d in delta0)),
        "out_diff": float((out_A - out_B).abs().max()),
        "delta0": delta0,
    }

    # ── check 4: 固定 residual/prefix, 只改上一层曲率 → 下一层响应 (router 通路 autograd) ──
    with torch.no_grad():
        mB.eval()
        sid0 = mB.get_indices(item_emb[:BATCH])[:, 0]
        cb0 = mB.vq_layers[0].embeddings.weight[sid0]
        log_cmax = float(torch.log(torch.tensor(2.0)).item())
        c_sig_a = torch.log(torch.full((BATCH, 1), mB.vq_layers[0].get_c_global().item(), device=device).clamp(min=1e-6)) / log_cmax
        prefix_a = torch.cat([cb0, c_sig_a], dim=-1)
        d1_a = mB.vq_layers[1].router(prefix_a)
        # 恢复并重算 (同一 prefix)
        c_sig_b = torch.log(torch.full((BATCH, 1), mB.vq_layers[0].get_c_global().item(), device=device).clamp(min=1e-6)) / log_cmax
        prefix_b = torch.cat([cb0, c_sig_b], dim=-1)
        d1_b = mB.vq_layers[1].router(prefix_b)
        resp = float((d1_a - d1_b).abs().max().item())
    # 训练后 router 非零 → 跨层响应真实测量 (此处先占位, 训练后更新)
    g1["checks"]["cross_layer_response"] = {"pass": True, "max_delta_response": resp, "note": "训练后更新"}

    # ── check 5: 梯度通路 ──
    mB.train()
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

    # ── check 4b (训练后): 固定 residual/prefix, 改上一层曲率 → 下一层 router 响应 ──
    with torch.no_grad():
        mB.eval()
        cb0 = mB.vq_layers[0].embeddings.weight[mB.get_indices(item_emb[:BATCH])[:, 0]]
        log_cmax = float(torch.log(torch.tensor(2.0)).item())
        c0 = mB.vq_layers[0].get_c_global().item()
        c_sig_hi = torch.log(torch.full((BATCH, 1), c0 * 1.5, device=device).clamp(min=1e-6)) / log_cmax
        c_sig_lo = torch.log(torch.full((BATCH, 1), c0 * 0.5, device=device).clamp(min=1e-6)) / log_cmax
        d_hi = mB.vq_layers[1].router(torch.cat([cb0, c_sig_hi], dim=-1))
        d_lo = mB.vq_layers[1].router(torch.cat([cb0, c_sig_lo], dim=-1))
        resp = float((d_hi - d_lo).abs().max().item())
    g1["checks"]["cross_layer_response"] = {"pass": resp > 1e-6, "max_delta_response": resp,
                                            "note": "训练后测量 (改上层曲率→下层 router 输出)"}

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
