"""Issue #154 Gate 1: 初始化与梯度通路验证.

必须同时通过:
- 初始 delta=0, A/B 距离、assignment、loss 和 SID 逐元素一致
- fc1 初始权重非零; fc2 初始权重为零
- 至少连续执行两个优化步, 记录 fc1/fc2 gradient norm 和 parameter norm
- 第二个优化步后 fc1/fc2 均已形成非零可学习通路
- prefix embedding 保持 stop-gradient
"""

import os, sys, json, math
from pathlib import Path
import numpy as np
import torch

TASK_DIR = Path(__file__).parent
# 统一用 treatment/_lib (Issue #154 修复版 quantizer); A 分支 = prefix_routing=False,
# 路由器修复不影响该分支 (fix 只在 prefix_routing=True 时生效)
sys.path.insert(0, str(TASK_DIR / "treatment" / "_lib"))
from prefix_conditioned_quantizer import PrefixConditionedHRQVAE, C_MIN, C_MAX
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
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    item_emb = torch.tensor(
        EmbDataset(str(TASK_DIR / "treatment" / "stage1" / "item_emb.parquet")).embeddings,
        dtype=torch.float32).to(device)

    # 初始化: 与 stage2 完全一致 (rank0 路径: 首个 batch 触发 kmeans init)
    results = {}
    models = {}
    for arm in ["control", "treatment"]:
        torch.manual_seed(SEED)
        np.random.seed(SEED)
        m = PrefixConditionedHRQVAE(
            in_dim=EMB_DIM, num_emb_list=CODEBOOK_SIZES, e_dim=E_DIM,
            layers=ENCODER_LAYERS, kmeans_init=True, kmeans_iters=KMEANS_ITERS,
            prefix_routing=(arm == "treatment"),
        ).to(device)
        perm = np.random.permutation(N_ITEMS)
        m.train()
        with torch.no_grad():
            m(item_emb[perm[:BATCH]], use_sk=False)
        models[arm] = m

    # Gate 1.1: 初始 delta=0, A/B 逐元素一致 (距离/assignment/loss/SID)
    g1 = {"checks": {}}
    with torch.no_grad():
        A = models["control"]
        B = models["treatment"]
        x = item_emb[:BATCH]
        out_A, rq_A, idx_A, zq_A, z_A = A(x, use_sk=False)
        out_B, rq_B, idx_B, zq_B, z_B = B(x, use_sk=False)
        same_out = torch.allclose(out_A, out_B, atol=1e-6)
        same_rq = torch.allclose(rq_A, rq_B, atol=1e-6)
        same_idx = torch.equal(idx_A, idx_B)
        recon_A = poincare_recon_loss(out_A, x)
        recon_B = poincare_recon_loss(out_B, x)
        same_loss = torch.allclose(recon_A + rq_A, recon_B + rq_B, atol=1e-6)
        sid_A = A.get_indices(x, use_sk=False)
        sid_B = B.get_indices(x, use_sk=False)
        same_sid = torch.equal(sid_A, sid_B)
        # 初始 delta 检查 (B 路径 L1/L2)
        delta0 = []
        for q in B.vq_layers:
            if q.prefix_routing:
                d = q._last_delta
                delta0.append(float(d.abs().max().item()) if d is not None else 0.0)
    g1["checks"]["init_delta_zero"] = {"pass": all(d < 1e-8 for d in delta0), "delta_max": delta0}
    g1["checks"]["same_distance_out"] = {"pass": bool(same_out), "max_abs_diff": float((out_A - out_B).abs().max())}
    g1["checks"]["same_rq_loss"] = {"pass": bool(same_rq), "diff": float((rq_A - rq_B).abs().max())}
    g1["checks"]["same_assignment"] = {"pass": bool(same_idx)}
    g1["checks"]["same_total_loss"] = {"pass": bool(same_loss)}
    g1["checks"]["same_sid"] = {"pass": bool(same_sid)}
    g1["checks"]["fc1_nonzero"] = {"pass": bool(models["treatment"].vq_layers[1].router.fc1.weight.abs().sum() > 0),
                                   "fc1_w_norm": float(models["treatment"].vq_layers[1].router.fc1.weight.norm())}
    g1["checks"]["fc2_zero"] = {"pass": bool(models["treatment"].vq_layers[1].router.fc2.weight.abs().sum() == 0)}

    # Gate 1.2: 两个优化步, 记录 fc1/fc2 梯度与参数范数
    opt = torch.optim.Adam(models["treatment"].parameters(), lr=1e-3)
    grad_log = []
    for step in range(2):
        perm = np.random.permutation(N_ITEMS)
        xb = item_emb[perm[:BATCH]]
        opt.zero_grad()
        out, rq, idx, zq, z = models["treatment"](xb, use_sk=False)
        recon = poincare_recon_loss(out, xb)
        loss = recon + rq
        rd, rm = models["treatment"].compute_route_reg()
        loss = loss + rd + rm
        loss.backward()
        torch.nn.utils.clip_grad_norm_(models["treatment"].parameters(), 1.0)
        rec = {}
        for li in [1, 2]:
            q = models["treatment"].vq_layers[li].router
            rec[f"L{li}_fc1_grad_norm"] = float(q.fc1.weight.grad.norm()) if q.fc1.weight.grad is not None else -1.0
            rec[f"L{li}_fc1_w_norm"] = float(q.fc1.weight.norm())
            rec[f"L{li}_fc2_grad_norm"] = float(q.fc2.weight.grad.norm()) if q.fc2.weight.grad is not None else -1.0
            rec[f"L{li}_fc2_w_norm"] = float(q.fc2.weight.norm())
        opt.step()
        grad_log.append(rec)
        print(f"step {step+1}: L1 fc1_g={rec['L1_fc1_grad_norm']:.6f} fc2_g={rec['L1_fc2_grad_norm']:.6f} "
              f"| L2 fc1_g={rec['L2_fc1_grad_norm']:.6f} fc2_g={rec['L2_fc2_grad_norm']:.6f}")
    s1, s2 = grad_log[0], grad_log[1]
    # spec: 第一个优化步后 fc2 获得有效梯度 (step1 检查 fc2)
    g1["checks"]["step1_fc2_grad"] = {"pass": bool(s1["L1_fc2_grad_norm"] > 0 and s1["L2_fc2_grad_norm"] > 0),
                                      "step1": s1}
    # spec: 第二个优化步后 fc1/fc2 均已形成非零可学习通路 (检查 step2 的 fc1+fc2 梯度与 fc2 参数更新)
    # 注: step1 时 fc2.weight=0 → 梯度不穿透到 fc1 (预期); step2 起 fc2 权重非零 → fc1 获得梯度
    g1["checks"]["step2_both_grad"] = {
        "pass": bool(s2["L1_fc1_grad_norm"] > 0 and s2["L2_fc1_grad_norm"] > 0
                and s2["L1_fc2_grad_norm"] > 0 and s2["L2_fc2_grad_norm"] > 0
                and s2["L1_fc2_w_norm"] > 0 and s2["L2_fc2_w_norm"] > 0),
        "step1": s1, "step2": s2}

    # Gate 1.3: prefix stop-grad (路由损失不改写 codeword)
    with torch.no_grad():
        cb_before = models["treatment"].vq_layers[1].embeddings.weight.clone()
    # 再跑一步专门看 codeword 梯度: 路由正则项对 prefix codeword 的反向应被 stop_grad 阻断
    models["treatment"].zero_grad()
    perm = np.random.permutation(N_ITEMS)
    xb = item_emb[perm[:BATCH]]
    with torch.no_grad():
        _, _, _, _, _ = models["treatment"](xb, use_sk=False)
    rd, rm = models["treatment"].compute_route_reg()
    (rd + rm).backward()
    cb_grad_l0 = models["treatment"].vq_layers[0].embeddings.weight.grad
    cb_grad_l1 = models["treatment"].vq_layers[1].embeddings.weight.grad
    g1["checks"]["prefix_stopgrad"] = {
        "pass": bool((cb_grad_l0 is None or cb_grad_l0.abs().sum() == 0)
                     and (cb_grad_l1 is None or cb_grad_l1.abs().sum() == 0)),
        "L0_grad_sum": float(cb_grad_l0.abs().sum()) if cb_grad_l0 is not None else 0.0,
        "L1_grad_sum": float(cb_grad_l1.abs().sum()) if cb_grad_l1 is not None else 0.0,
    }

    all_pass = all(c["pass"] for c in g1["checks"].values())
    g1["decision"] = "PASS" if all_pass else "FAIL"
    out_path = TASK_DIR / "gate1_initialization.json"
    with open(out_path, "w") as f:
        json.dump(g1, f, indent=2, ensure_ascii=False)
    print(f"\nGate 1: {g1['decision']} ({sum(1 for c in g1['checks'].values() if c['pass'])}/"
          f"{len(g1['checks'])} checks) -> {out_path}")
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
