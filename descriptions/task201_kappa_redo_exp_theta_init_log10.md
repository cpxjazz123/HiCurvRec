# Task #201 — 重做 exp(θ) κ with θ_init=log(10) (用户拍板选项 A)

> **任务目的**: 验证用户 2026-07-27 拍板 "选项 A" — 在 #199 exp(θ) κ 4 臂实验基础上, 改 θ_init=log(10)=2.3026, 让 c_init=10.0 进健康窗口, 看是否能让 exp(θ) 学到 > 1 (而非 #199 学到 0.7-1.0 向下方向)

> **完成日期**: (in progress)
> **状态**: 🟡 Stage 1 启动中

---

## 1. 背景

**Task #199 结论** (2026-07-26 02:58 verdict):
- ✅ exp(θ) 机制本身工作正常 (B/C 学到的 c 不固定, 比 c=1 略好)
- ❌ 用户预测"c 学到 > 1 进 λ∈[5,75] 健康窗口" 不成立 — 实际 c 学到 0.7-1.0 (向下而非向上)
- ❌ 用户预测"c 固定扫描 {1,10,30,100} 性能不随 c 变" 不成立 — c=10 略好, c=30 退化, c=100 完全坍缩
- ❌ 用户原"50× κ_max 修复"假设被实验数据证伪

**用户 2026-07-27 拍板**: "A — 重做用户方向 (κ 逐层可学 + θ_init=log(10))", 假设:
> #199 θ 学到 <1 是因为 init=0 (=c_init=1) 太靠近下限, θ 学起来前已是 c=10 健康窗口 — 让 θ 起点就在用户预测的健康区, 看是否能"学会大 c"。

---

## 2. 实验设计

**变量**: θ_init = log(10) = 2.3026 (vs #199 θ_init=0.0)
**保持不变** (跟 #199 B/C 对齐):
- `--kappa_mode exp_global` (臂 B) / `exp_per_layer` (臂 C)
- epochs=1000, batch_size=256, beta=0.5, sk_epsilons=[0,0,0], e_dim=32, num_emb_list=[64,128,256]
- lr=1e-3, kmeans_iters=1000, loss_type=poincare
- r_target_list=2.0,2.7,3.4 (per-layer ρ target, 决定 c_max)
- seed=42 (跟 #199 一致, R11 用户撤回 multi-seed)

**启动命令** (2 臂, 并行 GPU 1+2):
```bash
# 臂 B exp_global θ_init=log(10), GPU 1
bash scripts/task201_stage1_arm_B_init_log10.sh

# 臂 C exp_per_layer θ_init=log(10), GPU 2
bash scripts/task201_stage1_arm_C_init_log10.sh
```

**源码 patch** (R11.3 自主决策, 与 #199 同 precedent):
- `HG-Rec/model/hrqvae.py`: line 107/110 `torch.tensor(0.0)` → `torch.tensor(theta_init)`, `__init__` 加 `theta_init: float = 0.0` 参数
- `HG-Rec/train_hrqvae.py`: argparse 加 `--theta_init` (default 0.0), 传给 HRQVAE

**D 臂不再跑**: D 是 fixed c 扫描, θ_init=log(10) 无意义 (#199 D c=10 fixed 已是 D 臂最佳 8.26%, 不需要 θ_init 化)。

---

## 3. 决策触发 (vs #199 B/C baseline)

| 指标条件 | #199 B/C 实测 | #201 期望 | 决策 |
|---------|---------------|----------|------|
| c 学到的最终值 | B 0.80, C 0.72/0.93/0.97 | B > 10, C L0/L1/L2 都 > 1 (用户预测) | ❌ θ 学到 <1 → 假设反向; ✅ θ 学到 >1 → 假设初步通过 |
| collision_rate best | B 8.62%, C 8.67% | 进一步下降 (c 大→ 更分散) | ✅ < 8.62% → c 大有帮助; ⚠️ ≈8.62% → c 大无帮助 |
| θ 是否仍向下 (学小) | B θ=-0.225, C L0 θ=-0.325 | θ 不变 / 略增 (因为 init 已大) | ✅ θ 稳定在 log(10) 附近 → c=10 健康区自洽 |

---

## 4. 预算

| 阶段 | 估算时间 |
|------|---------|
| Stage 1 臂 B (1000 epoch, GPU 1) | ~50 min |
| Stage 1 臂 C (1000 epoch, GPU 2) | ~50 min |
| Stage 2 (SID inference, 4 个 GPU) | ~10 min |
| Stage 3 (T5-mini 200 epoch, 4 个 GPU) | ~2 h |
| Stage 4 (eval) | ~5 min |
| **总计 (Stage 1)** | **~1 h 并行** |

---

## 5. 风险与缓解

**风险 1**: θ_init=log(10) 后 c 仍学到 < 1 (θ 学到 < log(10) → 用户假设仍反向)
→ 缓解: 跟 #199 B/C 对照, 结论写 verdict "用户 κ_max 50× 假设彻底证伪, HG-Rec c=1.0 是工程最优"

**风险 2**: c_init=10 起步导致训练不稳定 (loss spike / NaN)
→ 缓解: β=0.5 commitment + poincare loss 主导, c_max per-layer clamp 防溢出. 看 loss 曲线是否异常.

**风险 3**: 用户撤回 multi-seed (R11) → 1 seed 可能方差大
→ 缓解: 跟 #199 B/C 同 seed=42 直接对比, 数字差异 > 2σ 才判定有意义

---

## 6. 完成度跟踪

- [x] patch hrqvae.py + train_hrqvae.py (加 --theta_init, py_compile ✅)
- [x] 写 task201 description + 2 个 launcher (B/C)
- [x] 更新 loop.md §16 (active = #201)
- [ ] 启动 Stage 1 臂 B (GPU 1, theta_init=log(10))
- [ ] 启动 Stage 1 臂 C (GPU 2, theta_init=log(10))
- [ ] Stage 1 完成 (2 臂 ckpt + log)
- [ ] 读 ckpt θ 值, 写 verdict

## 7. 关键决策点

| 时间 | 决策 | 理由 |
|------|------|------|
| 2026-07-27 03:42 | 用户拍板 "A" | "重做 exp(θ) κ with θ_init=log(10)" |
| 2026-07-27 03:50 | patch 源码加 --theta_init | R11.3 自主决策 (跟 #199 同 precedent) |
| 2026-07-27 03:50 | 跑 2 臂 (B_global + C_per_layer) | R11.3 决策, D 是 fixed 扫描无需 θ_init |
| 2026-07-27 03:50 | GPU 1 + GPU 2 并行 | R7 + 4 卡空闲 |

**与 #199 的对比**:
| 项 | #199 | #201 |
|---|------|------|
| θ_init | 0.0 (c_init=1.0) | log(10)=2.3026 (c_init=10.0) |
| 臂数 | 3 (B/C/D) | 2 (B/C, 跳过 D) |
| 训练配置 | 1000 epoch, β=0.5, lr=1e-3 | 同 #199 |
| GPU | 1/2/3 (3 卡) | 1/2 (2 卡, R7 不抢占) |
| seed | 42 | 同 #199 |