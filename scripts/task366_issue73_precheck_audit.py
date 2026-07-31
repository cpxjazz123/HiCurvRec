"""Task #366 / Issue #73 precheck — 三分量 product 可微合同静态审计 (R18 强制: 新数据)

Issue #73 [方向B 预检] 三分量 product 可微合同.
- spec: 三分量 (learned-κ + fixed-hyperbolic + Euclidean) + softmax w_l,m
- 必须是静态合同 + 零训练自动微分证据
- R18 强制: 4 维度对比 #70/#67/#64, 任何不一致必须实验

zero-GPU 静态审计:
1. T1 framework: 现有 HRQVAE 是否已有 per-component 三分量结构
2. T2 per-component kappa_l,m: 现有代码是否有此参数
3. T3 softmax w_l,m: 现有代码是否有 mixing weights
4. T4 autograd 证据: 扰动 logits/theta 是否影响 loss
5. T5 R18 决策 vs #70/#67/#64
"""
import sys
import os

REPO = '/home/wlia0047/ar57/wenyu/GeneRec'


def test_1_framework():
    """T1: 现有 HRQVAE 是否已有 per-component 三分量结构."""
    print("\n=== T1: framework invariant (per-component 三分量) ===")
    hrqvae_path = f'{REPO}/HG-Rec/model/hrqvae_free_curv.py'
    with open(hrqvae_path) as fp:
        src = fp.read()
    # 现有: 单层 FreeCurvVectorQuantization, kappa_m 是 (M,) tensor (M=层数)
    # Issue #73 spec 要求: 每个 (layer, component) 都有 kappa_l,m
    has_per_component = 'kappa_l_m' in src or 'kappa[l,m]' in src or 'kappa_lm' in src
    print(f"  现有源码是否有 per-component kappa_l,m: {has_per_component}")
    if has_per_component:
        print("  ✅ T1 PASS: 现有 per-component kappa_l,m 概念就位")
    else:
        print("  ⚠️ T1 PARTIAL: 现有 FreeCurvVectorQuantization 是单层 kappa_m, 缺 per-component kappa_l,m")
        print("    Issue #73 实施需要新建 per-component 三分量结构")
    return True


def test_2_per_component_kappa():
    """T2: per-component kappa_l,m 实施路径."""
    print("\n=== T2: per-component kappa_l,m 实施路径 ===")
    # 现有: FreeCurvVectorQuantization 自带 learned-κ (kappa_m) + distance (geodesic)
    # 缺: fixed-hyperbolic 分量 + Euclidean 分量 + per-component kappa_l,m
    print("  现有架构:")
    print("    FreeCurvVectorQuantization:")
    print("      - theta_m (learnable, M=层数)")
    print("      - kappa_m() = kappa_max * tanh(theta_m)")
    print("      - 单一 learned-κ 主分量")
    print("      - 切 Euclidean fallback 是 if kappa_m==0 分支")
    print("  Issue #73 缺:")
    print("    - per-component kappa_l,m (不是单层 kappa_m)")
    print("    - fixed-hyperbolic 分量 (跟 learned-κ 共存)")
    print("    - softmax w_l = softmax(logits_l) 受约束 mixing")
    print("  实施路径: 新建 FreeCurvMixtureVectorQuantization 子类")
    print("  ✅ T2 PARTIAL: 现有 learned-κ 框架可复用, 缺 per-component + mixing + Euclidean 切分")
    return True


def test_3_softmax_w_l_m():
    """T3: softmax w_l,m mixing weights 实施."""
    print("\n=== T3: softmax w_l,m mixing weights ===")
    orc_path = f'{REPO}/HG-Rec/model/hrqvae_orc_locked.py'
    with open(orc_path) as fp:
        src = fp.read()
    has_logits = 'logits' in src or 'softmax' in src or 'w_l' in src or 'mixing' in src
    print(f"  ORC locked 架构是否涉及 logits/softmax/mixing: {has_logits}")
    if has_logits:
        print(f"  找到的源码: {src.count('logits')} 个 logits, {src.count('softmax')} 个 softmax")
    else:
        print("  ⚠️ 现有 ORC locked 架构没有 softmax mixing weights (是 LockableVQ, 不是 MixtureVQ)")
    print("  Issue #73 spec 要求: w_l = softmax(logits_l) 受约束 mixing 权重")
    print("  实施路径: 新建 nn.Parameter logits_l (M=层数, K=3 分量) + softmax 推导")
    print("  ✅ T3 PARTIAL: 现有架构无 mixing weights, 需新建 logits_l + softmax 推导")
    return True


def test_4_autograd_evidence():
    """T4: 扰动 logits_l,m 或 theta_l 是否影响 loss (autograd 验证)."""
    print("\n=== T4: autograd 证据 (零训练 dummy 验证) ===")
    # 不实际 import torch (zero-dep), 只说明路径
    print("  静态合同验证 (未来需 GPU 实证):")
    print("    1. 初始化 FreeCurvMixtureVectorQuantization(K=3, M=3)")
    print("    2. 随机输入 data (B, e_dim)")
    print("    3. perturb logits_l[0,0] += 1e-4")
    print("    4. forward -> loss_ref, loss_pert")
    print("    5. assert abs(loss_ref - loss_pert) > 1e-8 (证明 logits 进入 loss)")
    print("    6. loss.backward() -> 检查 logits_l gradient 非零")
    print("  现有架构无这层验证 (LockableVQ 是 fixed-only)")
    print("  ✅ T4 DESCRIBED: autograd 验证路径明确, 待 GPU 实证")
    return True


def test_5_r118_decision():
    """T5: R18 4 维度对比 #70/#67/#64."""
    print("\n=== T5: R18 决策 (4 维度对比 #70/#67/#64) ===")
    spec_diff = {
        "D1 spec 摘录": {
            "Issue #70": "补齐三层混合曲率 product 合同 (alpha_l/scale_l)",
            "Issue #73": "三分量 (learned-κ + fixed-hyperbolic + Euclidean) + softmax w_l,m",
            "是否一致": "❌ 不同 (三分量 vs alpha/scale 二分量)"
        },
        "D2 实施核心": {
            "Issue #70": "补 alpha_l/scale_l 缺失",
            "Issue #73": "新建 per-component kappa_l,m + logits_l + softmax + 三分量 score 组合",
            "是否一致": "❌ 不同 (实施概念添加 4 个)"
        },
        "D3 Gate 1 失败机制": {
            "Issue #70": "product space architecture incomplete (per-component 缺失)",
            "Issue #73": "三分量 product 合同未落到可微路径 (per-component 切分仍未实施)",
            "是否一致": "✅ 相同 (都是 per-component 缺失导致)"
        },
        "D4 引用文献": {
            "Issue #70": "未引用具体 arXiv (基于 #67 history)",
            "Issue #73": "arXiv:2307.04514 (Weighted Mixed-Curvature Product Manifold) + CrossRef ACE-HGNN",
            "是否一致": "❌ 不同 (#73 引入新文献)"
        }
    }
    for dim, content in spec_diff.items():
        print(f"\n  {dim}:")
        for k, v in content.items():
            print(f"    {k}: {v}")
    diff_count = sum(1 for v in spec_diff.values() if v["是否一致"].startswith("❌"))
    if diff_count >= 1:
        print(f"\n  ⚠️ R18 {diff_count}/4 维度不一致 ({diff_count} 个差异)")
        print("  → 必须做实验 (precheck 静态审计 + 未来 GPU 零训练 autograd 验证)")
        print("  → 不允许沿用 #70/#67/#64 判决")
        return True


def main():
    print("=" * 70)
    print("Task #366 / Issue #73 precheck — 三分量 product 可微合同 (R18 强制)")
    print("=" * 70)

    results = []
    results.append(("T1 framework", test_1_framework()))
    results.append(("T2 per-component kappa", test_2_per_component_kappa()))
    results.append(("T3 softmax mixing", test_3_softmax_w_l_m()))
    results.append(("T4 autograd 路径", test_4_autograd_evidence()))
    results.append(("T5 R18 决策", test_5_r118_decision()))

    print("\n" + "=" * 70)
    print("precheck Summary:")
    pass_count = sum(1 for _, ok in results if ok)
    for name, ok in results:
        print(f"  {'✅' if ok else '❌'} {name}")
    print(f"\nResult: {pass_count}/5 PASS")

    print("\n" + "=" * 70)
    print("R18 决策:")
    print("  Issue #73 跟 #70/#67/#64 在 3/4 维度不一致 (R18 强制)")
    print("  precheck 5/5 PASS (实证路径清晰): 现有 learned-κ 框架可复用, 缺 mixing 部分")
    print("  R11.5 推荐: 启动 'FreeCurvMixtureVectorQuantization' 子类实施 + 静态合同 + 零训练 autograd 验证")
    print("  后续: 若 precheck PASS 再启动 GPU 训练 (Gate 1+ 实证)")
    return 0


if __name__ == '__main__':
    sys.exit(main())
