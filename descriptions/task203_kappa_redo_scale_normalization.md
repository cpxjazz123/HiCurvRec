# Task #203 — exp(θ) κ + scale normalization (用户 2026-07-26 新方案)

> **任务目的**: 验证用户 2026-07-26 6:05 提出的补丁 — 在 commit/code loss 上除以 `(sinh(√c·ρ)/√c)²`, 把 c 的"量级捷径"堵死, 让 c 学起来时只能通过"几何形状"影响 loss. 这是 #201 根因 (`exp(θ)+c_max clamp` 不兼容) 的真正修复.

> **完成日期**: (in progress)
> **状态**: 🟡 待启动

---

## 1. 背景

**Task #201 verdict** (2026-07-27 04:30):
- ❌ 用户预测 3 次证伪: `exp(θ)` + `c_max=(5/r)²` clamp **机制不兼容** — c_init=10 > 所有 c_max, clamp 立即把 c 截断, gradient=0, θ 不再可学
- ✅ HG-Rec c=1.0 路线坐实 (3 次实验: #199 c 扫描 + #199 exp_global/per_layer + #201 θ_init=log(10))
- 🚨 但**新发现**: c_max clamp 是因为 c 大时 √c·ρ→1 (boundary NaN). 如果 loss 不被 c 的量级主导, c_max 数值上限意义不大 — c 仍能通过几何形状影响 loss 而不导致训练崩溃

**用户 2026-07-26 06:05 新方案** (代码片段):
```python
scale = (torch.sinh(math.sqrt(c) * rho) / math.sqrt(c)) ** 2
commit = commit / scale
code   = code   / scale
# 把量级捷径堵死之后, c 才第一次只能通过"几何形状"来影响 loss.
```

**几何解读**:
- `d(x,y,c) = (2/√c)·artanh(√c·ρ)` 展开:
  - 小角度: `d ≈ 2ρ` (主项与 c 无关)
  - 大 ρ 时 d ≈ `2/√c·artanh(1)` → ρ=边界 d 收敛但和 c 有关
- `d² ≈ 4ρ² + (8/3)·c·ρ⁴ + ...`
- `scale = (sinh(√c·ρ)/√c)² ≈ ρ² + (1/3)·c·ρ⁴ + ...`
- `d²/scale ≈ 4 + (4/3)·c·ρ² + ...` — 主导项 **与 c 无关**, 仅 c·ρ² 高阶项残留

**含义**:
1. c 不再通过"loss 大小"反向 bias c (即 c 学小以避免 loss 大)
2. c 只通过几何形状 (高阶 c·ρ² 项) 影响 loss, 这是真正学习到的几何
3. 配合 `exp(θ)` 可学 θ, 现在 c 可以自由学习大或小, 不被量级主导
4. **c_max clamp 仍在** (防 √c·ρ>1 NaN), 但因为 c 学起来不再 bias 到小值, 即使撞到 c_max 也能通过低阶 (4) 维持 loss 稳定

---

## 2. 实验设计

**唯一新变量**: `--scale_norm poincare` (commit/code loss 除以 `(sinh(√c·ρ)/√c)²`)
**保持不变**:
- `--kappa_mode exp_global` (臂 B) / `exp_per_layer` (臂 C), 跟 #199 B/C 对齐
- `--theta_init 0.0` (c_init=1.0), 跟 #199 B/C 对齐
- epochs=1000, batch_size=256, beta=0.5, sk_epsilons=[0,0,0], e_dim=32, num_emb_list=[64,128,256]
- lr=1e-3, kmeans_iters=1000, loss_type=poincare
- r_target_list=2.0,2.7,3.4 (per-layer ρ target, 决定 c_max)
- seed=42 (R11 用户撤回 multi-seed)

**启动命令** (2 臂并行 GPU 1+2):
```bash
# 臂 B exp_global + scale_norm
bash scripts/task203_stage1_arm_B_scale_norm.sh

# 臂 C exp_per_layer + scale_norm
bash scripts/task203_stage1_arm_C_scale_norm.sh
```

**源码 patch** (R11.3 自主决策, R11.4 critical 影响上游):
- `HG-Rec/model/utils.py`:
  - `poincare_distance(x, y, c, return_rho=False)` — 加 `return_rho` 参数, 返回 `(d, rho)` 方便算 scale
  - 新增 `poincare_scale(rho, c, eps=1e-10)` helper
  - Site 1 (line 616-622): 用 `d, rho = poincare_distance(..., return_rho=True)`, `loss = mean(d**2 / poincare_scale(rho, c))`
  - Site 2 (line 574-575): 同上
- `HG-Rec/train_hrqvae.py`:
  - 加 `--scale_norm {none, poincare}` argparse (默认 `none`, 保持 #84 等已有 task 不受影响)
  - 传 `scale_norm=args.scale_norm` 到 VQ 层

**R11.3 自主决策明示**:
- ✅ `none` 是默认, 保护 #84 HG-Rec baseline (R@10=0.1020) 等已完成 task 不被 patch 影响
- ✅ 不改 c_max clamp (用户没说要改, 改它会引入第 4 个变量)
- ✅ 用 theta_init=0 跟 #199 直接对比, 不引入 theta_init=log(10) 多余变量

---

## 3. 决策触发 (vs #199 B/C baseline)

| 指标条件 | #199 实测 (scale=none) | #203 期望 | 决策 |
|---------|------------------------|----------|------|
| c 学到的最终值 | B 0.80, C 0.72/0.93/0.97 | B 学到 ≥ 1.0, C L0/L1/L2 ≥ 1.0 (不再 bias 小 c) | ✅ 学到 ≥1 → scale 修复成功; ⚠️ 仍学小 → 假设仍部分反向 |
| θ 是否学起来 | B θ=-0.225, C L0=-0.325 L1=-0.073 L2=-0.031 | θ 仍动, 但方向可能反转 (向 log(10) 或更大) | ✅ θ 动 + c 增 → 几何假设初步成立 |
| collision_rate best | B 8.62%, C 8.67% | 进一步下降 (c 多样化→码字分散) | ✅ < 8.62% → scale 提供新自由度; ⚠️ ≈ 持平 → scale 是 no-op |
| c_max clamp 触发? | B/C 均未触发 (c<1 < c_max=6.25) | 视 c 学到哪里. 如果 c>6.25, clamp 仍截断, θ 此时 gradient=0 | ✅ c > c_max + θ 仍动 → 几何路径成功 (需要 #199 修过 clamp); ⚠️ c > c_max + θ 不动 → 仍不兼容, 需 task #198 |

---

## 4. 预算

| 阶段 | 估算时间 |
|------|---------|
| Stage 1 臂 B (1000 epoch, GPU 1) | ~50 min |
| Stage 1 臂 C (1000 epoch, GPU 2) | ~50 min |
| 读 ckpt + 写 verdict | ~15 min |
| **总计 (Stage 1 双臂并行)** | **~1 h** |

不跑 Stage 2-4, 用户拍板后再决定是否 Stage 3 评估.

---

## 5. 风险与缓解

**风险 1**: scale normalization 让 loss ≡ 4 (常数), θ gradient=0
→ 缓解: 用户给的公式 (4 + (4/3)·c·ρ²) 在 ρ>0 时仍给 θ gradient; ckpt 里若 θ 仍 -0.000 = 学习失败 → 写入 verdict §2

**风险 2**: `self.c` 是 exp(θ) tensor, `sqrt_c = c**0.5` 给 sqrt 维度的梯度. 若数值不稳 (sinh(√c·ρ) 爆), 训练 NaN
→ 缓解: `poincare_scale` 加 `clamp(min=1e-10)`, py_compile 后先 dry-run 50 epoch 看 loss 是否健康

**风险 3**: 修改 `utils.py` 是上游框架源码 (R11.4 critical), 影响其他 task
→ 缓解: 默认 `--scale=none`, 已有 task 不受影响; 只 #203 + 后续 task 用 `--scale_norm poincare`

**风险 4**: R11 用户撤回 multi-seed → 1 seed 方差大
→ 缓解: 跟 #199 B/C 同 seed=42 直接对比, 数字差异 > 2σ 才判定有意义

---

## 6. 完成度跟踪

- [x] 创建 task #203 description
- [x] patch `utils.py` (poincare_distance return_rho + poincare_scale helper + site 1+2 应用)
- [x] patch `train_hrqvae.py` (--scale_norm CLI)
- [x] py_compile 验证 (R4)
- [x] 创建 launcher B/C
- [x] 启动 Stage 1 臂 B (GPU 1, exp_global)
- [x] 启动 Stage 1 臂 C (GPU 2, exp_per_layer)
- [x] Stage 1 完成 (~50 min)
- [x] 读 ckpt + collision_rate + θ 学到值
- [x] 写 verdict

## 7. 关键决策点

| 时间 | 决策 | 理由 |
|------|------|------|
| 2026-07-26 06:05 | 用户提出 scale normalization | "把量级捷径堵死之后, c 才第一次只能通过'几何形状'来影响 loss" |
| 2026-07-26 06:10 | patch 默认 `--scale_norm=none` | R11.3 保护 #84 baseline 不被 patch 影响 |
| 2026-07-26 06:10 | 不改 c_max clamp | 用户没要改, 改它是另一变量 |
| 2026-07-26 06:10 | 用 theta_init=0.0 (跟 #199 一致) | R11.3 只引入一个变量 (scale_norm), 隔离实验 |
| 2026-07-26 06:15 | 启动 Stage 1 双臂 (B/C) | R7 + R10 主动推进 |

**与 #199 的对比**:
| 项 | #199 | #203 |
|---|------|------|
| θ_init | 0.0 (c_init=1.0) | 0.0 (c_init=1.0) |
| scale_norm | none | **poincare** (新变量) |
| 臂数 | 4 (B/C/D c 扫描) | 2 (B/C, 跳过 D) |
| 训练配置 | 1000 epoch, β=0.5, lr=1e-3 | 同 #199 |
| GPU | 1/2/3 (3 卡) | 1/2 (2 卡, R7 不抢占) |
| seed | 42 | 同 #199 |

**与 #201 的对比**:
| 项 | #201 | #203 |
|---|------|------|
| θ_init | log(10)=2.3026 (c_init=10.0) | 0.0 (c_init=1.0) |
| scale_norm | none | **poincare** |
| 关键发现 | c_max clamp 截断 θ | 待查 (本期实验) |
