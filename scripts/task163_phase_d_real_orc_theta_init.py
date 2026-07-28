#!/usr/bin/env python3
"""Task #163 Phase D — 真 ORC → θ_init 转换.

Input: task163_5_real_orc_per_layer.json (Lin 2011 W_1 LP 验证过的真 ORC)
       task163_phase_a_summary.json (group structure info)
Output: 新 θ_init_list 给下游 Stage 1 训练 + 决策记录.

转换逻辑 (跟 proxy version 对照):

  Proxy (Phase C, 用 avg_cc ∈ [0,1], 已废):
    shift = avg_cc - 0.5  (0.5=neutral, <0.5=hyperbolic, >0.5=spherical)
    θ = shift * scale  (scale ≈ 2)
    θ_init = [-0.3161, -0.7997, -0.7997]  # 全 hyperbolic

  Real ORC (Phase D, Lin 2011 α=0.5 lazy walk, κ ∈ [-1, 1]):
    ORC 直接就是有符号曲率: <0=hyperbolic, 0=Euclidean, >0=spherical
    κ_init = ORC_mean  (已经是 κ 量纲, 无需 shift)
    θ_init = arctanh(κ_init / κ_max)  (重参数化, 配合 hrqvae_free_curv.py 的 κ = κ_max·tanh(θ))

Stage 1 训练超参 (跟 #162 / 上一轮 Phase C 一致):
  κ_max = 2.0  (跟 #162 / #163 Phase C 同)
  β = 0.5  (paper-faithful commitment)
  num_emb_list = [32, 64, 256]
  sk_epsilons = [0, 0, 0]  (paper-faithful)
"""
import os, json

# Paths (跟 task163_5_real_orc_measurement.py 一致)
TMP_DIR = os.environ.get(
    "CLAUDE_JOB_TMP",
    f"/home/wlia0047/.claude/jobs/{os.environ.get('CLAUDE_JOB_ID', 'a1f6b58b')}/tmp/task163",
)

# 1. 加载真 ORC (Lin 2011 LP-based, 11 toy graph + 3 solver 验证过)
orc_path = os.path.join(TMP_DIR, "task163_5_real_orc_per_layer.json")
orc_data = json.load(open(orc_path))
l0_orc_mean = orc_data["l0_orc_mean"]
l1_prefix_orc_mean = orc_data["l1_prefix_orc_mean"]

# 2. 加载 group 结构 (跟 proxy version 对照)
phase_a = json.load(open(os.path.join(TMP_DIR, "task163_phase_a_summary.json")))
print(f"[Phase D] N items: {phase_a['n_items']}")
print(f"[Phase D] L0 unique codewords: {phase_a['l0_unique_count']}")
print(f"[Phase D] L1 prefix groups (valid): ~{orc_data['l1_prefix_groups_measured']}")

# 3. 加载旧 proxy θ_init 做对比
proxy_path = os.path.join(TMP_DIR, "task163_phase_c_theta_init.json")
proxy_theta = json.load(open(proxy_path))
print(f"\n[Proxy θ_init (Phase C, 已废)] = {proxy_theta['theta_init_L0'], proxy_theta['theta_init_L1'], proxy_theta['theta_init_L2']}")
print(f"[Proxy κ_init (Phase C, 已废)] = {proxy_theta['kappa_init_L0'], proxy_theta['kappa_init_L1'], proxy_theta['kappa_init_L2']}")

# 4. 真 ORC → κ_init → θ_init 转换
KAPPA_MAX = 2.0

# L0: 用 L0 32 组真实 ORC mean
kappa_init_L0 = l0_orc_mean
# L1: 用 L0×L1 prefix 296 组的真实 ORC mean
kappa_init_L1 = l1_prefix_orc_mean
# L2: 按用户 risk #1 跳过, 用 L1 placeholder
kappa_init_L2 = l1_prefix_orc_mean

import math

def safe_atanh(x, max_arg=0.999):
    """θ = arctanh(κ / κ_max), 处理 |x| ≥ 1 的边界."""
    arg = max(-max_arg, min(max_arg, x / KAPPA_MAX))
    return math.atanh(arg)

theta_init_L0 = safe_atanh(kappa_init_L0)
theta_init_L1 = safe_atanh(kappa_init_L1)
theta_init_L2 = safe_atanh(kappa_init_L2)

print("\n" + "=" * 70)
print("[Phase D] 真 ORC → θ_init 转换结果")
print("=" * 70)
print(f"  L0 real ORC mean = {l0_orc_mean:+.4f}  (32 groups)")
print(f"  L1 prefix real ORC mean = {l1_prefix_orc_mean:+.4f}  (296 groups)")
print(f"  L2 real ORC: skipped per user risk #1, use L1 placeholder")
print()
print(f"  κ_max = {KAPPA_MAX}")
print(f"  Conversion: κ_init = ORC_mean, θ_init = arctanh(κ_init / κ_max)")
print()
print(f"  Per-layer:")
print(f"    L0: κ_init = {kappa_init_L0:+.4f}, θ_init = {theta_init_L0:+.4f}")
print(f"    L1: κ_init = {kappa_init_L1:+.4f}, θ_init = {theta_init_L1:+.4f}")
print(f"    L2: κ_init = {kappa_init_L2:+.4f}, θ_init = {theta_init_L2:+.4f}  (= L1 placeholder)")
print()

theta_init_list = [theta_init_L0, theta_init_L1, theta_init_L2]
kappa_init_list = [kappa_init_L0, kappa_init_L1, kappa_init_L2]
print(f"  → Stage 1 --theta_init_list {' '.join(f'{x:+.4f}' for x in theta_init_list)}")

# 5. 跟 Task #162 (random init) / Phase C (proxy init) 3-way 对比
print("\n" + "=" * 70)
print("[3-way 对比] θ_init 不同来源")
print("=" * 70)
print(f"  Layer | Task #162 random | Phase C proxy | Phase D 真 ORC")
print(f"  ------|------------------|---------------|------------------")
for li in range(3):
    t162 = proxy_theta["task162_comparison"][f"task162_theta_init_L{li}"]
    tc = proxy_theta[f"theta_init_L{li}"]
    td = theta_init_list[li]
    print(f"  L{li}    |   {t162:+.4f}         |   {tc:+.4f}      |   {td:+.4f}")
print()
print(f"  Layer | κ_init derived")
print(f"  ------|----------------------------------------------------")
for li in range(3):
    t162_k = proxy_theta["task162_comparison"][f"task162_kappa_init_L{li}"]
    tc_k = proxy_theta[f"kappa_init_L{li}"]
    td_k = kappa_init_list[li]
    print(f"  L{li}    |  #162={t162_k:+.4f}  PhaseC={tc_k:+.4f}  PhaseD={td_k:+.4f}")

# 6. 决策记录 + 输出 JSON
output = {
    "phase": "Task #163 Phase D — 真 ORC-derived θ_init",
    "rationale": "用户 2026-07-25 反馈: 之前的 proxy avg_cc → θ_init 不可信, 必须用 Lin 2011 真 ORC (已 11 toy graph + 3 LP solver 验证). 重跑分组→测 ORC→转 θ_init 流程",
    "method": "Real ORC (Lin 2011 W_1 LP) per-layer mean → κ_init (无 shift) → θ_init = arctanh(κ_init / κ_max)",
    "kappa_max": KAPPA_MAX,
    "n_items": phase_a["n_items"],
    "l0_orc_mean": l0_orc_mean,
    "l0_orc_std": orc_data["l0_orc_std"],
    "l1_prefix_orc_mean": l1_prefix_orc_mean,
    "l1_prefix_orc_std": orc_data["l1_prefix_orc_std"],
    "l2_status": "skipped per user risk #1, use L1 placeholder",
    "kappa_init_list": kappa_init_list,
    "theta_init_list": theta_init_list,
    "theta_init_list_str": " ".join(f"{x:+.4f}" for x in theta_init_list),
    "comparison": {
        "task162_random_init": [proxy_theta["task162_comparison"][f"task162_theta_init_L{i}"] for i in range(3)],
        "task163_phase_c_proxy_init": [proxy_theta[f"theta_init_L{i}"] for i in range(3)],
        "task163_phase_d_real_orc_init": theta_init_list,
    },
    "interpretation": (
        "Phase C proxy 给出全 hyperbolic θ_init [-0.316, -0.800, -0.800] (avg_cc<0.5 → 推断 negative κ). "
        "Phase D 真 ORC 给出混合信号: L0 ≈ Euclidean (-0.022), L1/L2 轻微 spherical (+0.466). "
        "→ 跟 Toys 数据是双曲的发现不同 (见 [[task70-real-ollivier-curvature]]): Instruments 数据 L0 接近欧式, L1/L2 反而是球面. "
        "→ 训练 outcome 应该跟 #162 (random init → κ 终点 spherical) 类似, 但 L1/L2 起点已 spherical, 可能更快收敛到 spherical 最优."
    ),
    "next_step": (
        "用此 θ_init 启动新的 Stage 1 训练 (Task #163 Phase D): "
        "python3 scripts/task89_stage1_train_rqvae.py "
        "--theta_init_list " + " ".join(f"{x:+.4f}" for x in theta_init_list) + " "
        "--kappa_max 2.0 --num_emb_list 32 64 256 --beta 0.5 --sk_epsilons 0 0 0 ..."
    ),
    "task_completed": "Task #163 Phase D θ_init derivation complete; Stage 1 training ready to launch"
}

out_path = os.path.join(TMP_DIR, "task163_phase_d_real_orc_theta_init.json")
with open(out_path, "w") as f:
    json.dump(output, f, indent=2)

print(f"\n[Phase D] Saved: {out_path}")
print()
print("=" * 70)
print("summary:")
print("=" * 70)
print(f"  theta_init_list = {theta_init_list}")
print(f"  → 跟 proxy (Phase C) 完全不同 (proxy 全 negative, 真 ORC L0~0, L1+)")
print(f"  → 跟 #162 random init 也不同 (#162 L0=+0.084, 真 ORC L0=-0.011)")
print(f"  → 建议立即启动 Stage 1 重训, 验证真 ORC 起点 vs random/proxy 起点的下游差异")