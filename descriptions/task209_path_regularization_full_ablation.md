# Task #209 — 路径正则 + 完整消融 (Phase 0 + 1 + 1b + 2 + 3)

> **任务目的**: 在 #208 双码本 + 显式 `r_target = ρ/2` 基础上, 引入 **路径正则** (Path Regularization) 作为核心创新点. 通过 5 臂消融证明"双曲几何在 RQ 中实质生效, 几何贡献 ≠ 加权正则", 并通过 Phase 3 Head/Body/Tail 切片评估呈现层级结构对 Tail item 的优势.

> **完成日期**: (in progress)
> **状态**: 🟡 待启动

---

## 0. 任务谱系

**继承**: Task #208 (双码本几何解耦, 4 臂 B/C/D 双曲 + E 欧式加权) → 提供 `emb_geo` + `emb_rec` 双码本架构 + 显式 `r_target = ρ/2` 半径设定 + κ-Stereographic 距离 (c=1.0).

**替代原因**: 用户 2026-07-26 派工新方案, 把 #208 的 4 臂扩展为 5 臂, 加入 **路径正则** 作为核心新机制, 加入 A4 (欧式路径正则) 作为几何对照, 加入 Phase 1b 超参扫描 + Phase 2 多种子 + Phase 3 切片评估. #208 已 ⛔ 归档.

**前置**:
- Task #199/201/203/204/205 链: c ∈ [1, ~10] 健康区 (c=1.0 选定)
- Task #200/202 双码本 v5 失败但路径方向已验证 (no NaN, α_geo 修复路径走通)
- Task #181 Phase 0.6 baseline ckpt (Phase 0 init diagnostic 依赖此 baseline encoder)

---

## 1. 实验设计总览

| Phase | 内容 | 卡 | 时间 | 出口条件 |
|---|---|---|---|---|
| **0** | 实现验证 6 项 (不训练) | 0 (CPU) | 半天 | **6 项全过** |
| **1** | Stage 1 消融, 5 臂 | 4 | 2 天 | 至少 1 臂不坍缩 |
| **1b** | w_path / ρ 超参扫描 (A3 上) | 4 | 1 天 | 选定 Phase 2 主力配置 |
| **2** | 下游 Stage 2/3/4 + 3 种子 | 4 | 2 天 | — |
| **3** | 机制分析 + 切片评估 | 1 | 1 天 | — |

---

## 2. Phase 0: 实现验证 (6 项, 全过才能进 Phase 1)

> **核心警示**: 你们已经因为实现问题浪费过五轮, 这一步不能省.

| # | 检查 | 通过标准 |
|---|---|---|
| 1 | **半径换算** | 三层的 ρ 实测 = 2×切空间范数, 误差 < 1e-5 |
| 2 | **半径达标** | 三层 √c·ρ = 2.0 / 2.7 / 3.4, 误差 < 2% |
| 3 | **方向初始化** | `n_dup(cos>0.995) = 0`, `util = 100%`, `cos_mean ∈ [-0.05, 0.15]`; **输入必须中心化 (用 batch 均值, 不是未收敛的 EMA)** |
| 4 | **损失配比** | 归一化后, quant/recon 的比值与官方 baseline 同量级 (±3 倍以内) |
| 5 | **路径成本单元测试** | **人工构造一条在测地线上的路径, 绕路成本必须 ≈ 0 (< 1e-4); 人工偏离 30°, 成本必须显著为正** |
| 6 | **数值安全** | 三层 `min‖residual‖ > 1e-6`; `F.normalize(eps=1e-8)`; 跑 20 步无 NaN |

**第 5 项是这个方法专属的单元测试** —— 路径成本算错的话, 核心创新就是空的, 而且训练完全看不出来. **务必做**.

#### Phase 0 启动命令

```bash
# 实现 6 项检查 (CPU 可跑)
python3 scripts/task209_phase0_verify.py --check 1  # 半径换算
python3 scripts/task209_phase0_verify.py --check 2  # 半径达标
python3 scripts/task209_phase0_verify.py --check 3  # 方向初始化
python3 scripts/task209_phase0_verify.py --check 4  # 损失配比
python3 scripts/task209_phase0_verify.py --check 5  # 路径成本单元测试 ← 关键
python3 scripts/task209_phase0_verify.py --check 6  # 数值安全
# 或一键全跑
python3 scripts/task209_phase0_verify.py --all
```

产物目录:
- `verdicts/task209_phase0_check{1-6}_result.md` (每项一 verdict, 含 result: 行)
- `verdicts/task209_phase0_overall_result.md` (6 项汇总)

**任一项不过 → 不进入 Phase 1**.

---

## 3. Phase 1: Stage 1 消融 (5 臂)

### 五个臂

| 臂 | 半径分层 | 尺度归一化 | 路径正则 | 说明 |
|---|---|---|---|---|
| **A0** | ✗ | ✗ | ✗ | 官方 baseline (Task #181, 不重跑) |
| **A1** | ✓ | ✗ | ✗ | 只分层 |
| **A2** | ✓ | ✓ | ✗ | +校正 |
| **A3** | ✓ | ✓ | ✓ **双曲** | **完整方法** |
| **A4** | ✓ | ✓ | ✓ **欧式** | **关键对照** |

**A4 是审稿人必问的那一格: 证明提升来自几何, 而不是"加了个正则就行"**.

### 配置 (其余全部对齐官方)

```
数据集       Instruments
e_dim        32
码本         [64, 128, 256]
epochs       1000
batch_size   1024          ← 不是 256
lr           1e-3, AdamW, weight_decay 0
warmup       20, linear schedule
grad clip    1.0
beta         0.5
sk_epsilons  [0, 0, 0]
kmeans_init  True (方向 k-means), iters 1000

新增:
rho_targets  [2.0, 2.7, 3.4]
c            1.0 (固定)
w_path       1.0 (Phase 1 先用默认, Phase 1b 再扫)
alpha        1.0 (重构码本对齐权重)
```

**四张卡跑 A1/A2/A3/A4, 一轮 (A0 用 #181 baseline 不重跑)**.

### Phase 1b: 超参扫描 (A3 上)

| 参数 | 扫描值 | 目的 |
|---|---|---|
| `w_path` | 0.1 / 1.0 / 10 | 路径正则的强度 |
| `rho_targets` | [1.5,2.0,2.5] / [2.0,2.7,3.4] / [2.5,3.5,4.5] | 半径档位 |

**先扫 `w_path` (更关键), 再扫半径**.

### Phase 1 通过标准 (进 Phase 2 的门槛)

| 指标 | 标准 |
|---|---|
| 无 NaN, 训练跑完 1000 epoch | 必须 |
| `cos_max < 0.995`, 全程 | 没有方向坍缩 |
| **真实利用率 (Sinkhorn 前)** | **≥ baseline 的 90%** |
| **碰撞率** | **≤ baseline × 1.2** (即 ≤ 11%) |
| **√c·ρ** | 三层全程 ≥ 1.8 |

**不满足就不进 Phase 2 —— 别拿坏码本去烧 GPU**.

---

## 4. Phase 2: 下游

**对 Phase 1 通过的臂 (至少包括 A3 和 A4), 跑 Stage 2 + 3 + 4**.

| 项 | 设置 |
|---|---|
| Stage 2 | `sk_epsilons=[0,0,0]`, 30 轮 dedup, **记录第 4 位非零数量和最大值** |
| Stage 3 | 官方配置: 200 epoch, early_stop 20, lr 1e-4, batch 256, beam 20 |
| **种子** | **每臂 3 个 Stage 3 种子** |
| **Stage 1 种子** | **A0 和 A3 各补 3 个 Stage 1 种子** ← 你们从来没测过 Stage 1 的方差 |

**Stage 1 的种子方差必须补** —— 否则你们只有 T5 的噪声线 (±0.0014), 没有完整的噪声线, 任何"提升"都会被质疑.

**评测同时报 validation 和 test**.

---

## 5. Phase 3: 机制分析 (论文的核心章节)

### 5.1 机制指标 (证明创新点兑现)

| 指标 | 对比 | 期望 |
|---|---|---|
| **绕路成本** | A3 vs A4 vs A0 | **A3 显著最低** ← 核心证据 |
| **√c·ρ** | A3 vs A0 | 2.0+ vs 0.54 |
| **argmin 一致率** | A3 vs A0 | A0 = 99.9% (几何无效); A3 报出实际值 |
| **每层角分辨率 sinh(√c·ρ)** | A3 vs A0 | 3.6/7.4/15.0 vs 0.57/0.19/0.13 (**方向反转**) |

**"绕路成本"是这个方法专属的指标, 必须画成随 epoch 的曲线**.

### 5.2 切片评估 (重要, 可能是唯一能看到提升的地方)

**按 item 流行度分三档: Head (前 20%) / Body (中 60%) / Tail (后 20%), 分别报 R@10 和 NDCG@10**.

**理由**: 你们已经用 24 次评估证明 **整体指标对码本质量不敏感**. 但层级结构最该帮助的是 **Tail item** (数据稀疏, 只能靠语义结构). **整体指标可能把这个信号平均掉了**.

VarLenRec 正是在切片上发现了它的核心现象. **这条路走得通**.

---

## 6. 日志规格 (每次 eval)

```
per layer:
  rho_p50, sqrt_c_times_rho        # 几何是否激活
  path_cost_p50                    # 核心机制 ← 新增
  agree_hyp_vs_euc                 # 几何是否影响分配
  cos_mean, cos_max, n_dup         # 方向坍缩监控
  util_raw                         # Sinkhorn 前
  rec_err_p50                      # 重构码本的拟合质量
  resid_norm_min                   # 数值安全
global:
  collision_rate, recon_loss, quant_loss, path_loss
```

产物目录:
- `products/task209/phase0/` (6 项 check 输出)
- `products/task209/phase1/arm_A1/A2/A3/A4/` (4 臂 × 1000 epoch)
- `products/task209/phase1b/w_path_{0.1,1.0,10}/` (扫描)
- `products/task209/phase2/arm_A3_seed{1,2,3}/...arm_A4_seed{1,2,3}/` (3 种子 × 2 臂)
- `products/task209/phase3/mechanism_metrics/`, `products/task209/phase3/slice_metrics/`
- `verdicts/task209_phase0_overall_result.md` → ... → `verdicts/task209_phase3_result.md`

---

## 7. 风险与退路

| 风险 | 症状 | 应对 |
|---|---|---|
| **方向坍缩** | `cos_max → 1`, 利用率暴跌 | 方向 k-means 没做对; 或加方向多样性正则 |
| **重构变差** | recon_loss 明显高于 A0 | 调大 `alpha` |
| **零残差 NaN** | 深层 NaN | `eps=1e-8`, 监控 `resid_norm_min` |
| **整体 R@10 不动** | 大概率会发生 | **切换到切片评估 + 机制指标叙事** |

### 关于最后一条

**你们已经证明下游对码本质量不敏感. 所以"整体 R@10 不动"是很可能的结果, 要提前想好怎么写**.

**两条退路:**

1. **切片**: Tail 上有提升就够了, 而且更有说服力 (说明是结构在起作用, 不是运气)
2. **机制叙事**: 即使性能持平, "首次让双曲几何在 RQ 中实质生效, 并给出生效条件 √c·ρ ≳ 2" 本身就是贡献 —— **而 A4 那一格保证了这个主张站得住**

**建议现在就把这两条退路写进 description, 别等结果出来才想** (已写).

---

## 8. 预算

| 阶段 | 估算时间 | 卡 |
|------|---------|-----|
| Phase 0 实现验证 | ~0.5 天 | 0 (CPU) |
| Phase 1 Stage 1 消融 | 2 天 | 4 |
| Phase 1b 超参扫描 | 1 天 | 4 |
| Phase 2 下游 + 3 种子 | 2 天 | 4 |
| Phase 3 机制 + 切片 | 1 天 | 1 |
| **总计** | **~6.5 天** | - |

---

## 9. 与在跑任务的关系

- Task #208 已 ⛔ 归档 (用户 2026-07-26 新方案替代)
- Task #181 Phase 0.6 baseline (A0, 不重跑)
- Task #199/201/203/204/205 链已坐实 c ∈ [1, 10] 健康区, 本任务 c=1.0 不需要重新扫描
- Task #202 (Sinkhorn Stage 3) FAIL, 路径不进 paper recipe

---

## 10. 完成度跟踪

- [x] task209 description 登记 (loop.md §16 替换)
- [ ] Phase 0 实现 6 项检查全部通过
- [ ] Phase 0 verdict 6 项 + 汇总写入
- [ ] Phase 1 5 臂 Stage 1 训练完成 (A0 不重跑, 用 #181)
- [ ] Phase 1 4 臂主判据核验 (cos_max, util_raw, collision_rate, √c·ρ)
- [ ] Phase 1 verdict 写入
- [ ] Phase 1b w_path / ρ 扫描完成
- [ ] Phase 1b verdict 写入
- [ ] Phase 2 Stage 2 SID 推断 (通过的臂 × Sinkhorn)
- [ ] Phase 2 Stage 3 T5-mini 训练 (通过的臂 × 200 epoch × 3 种子)
- [ ] Phase 2 Stage 4 eval (通过的臂 × test R@10 × 3 种子)
- [ ] Phase 2 verdict 写入 (含均值±std)
- [ ] Phase 3.1 机制指标 (绕路成本曲线, √c·ρ, argmin 一致率, 角分辨率)
- [ ] Phase 3.2 切片评估 (Head/Body/Tail R@10, NDCG@10)
- [ ] Phase 3 verdict 写入
- [ ] 论文 paper.md §5.7 / §6.3 整合
- [ ] 最终 verdict 写入 + loop.md §16 归档

---

## 11. 关键决策点 (R11.3 自主决策留痕)

1. **编号 #209 而不是继续 #208**: 用户新方案加入路径正则 + 5 臂 + Phase 1b 扫描 + 多种子 + 切片评估, 是新的实验框架. #208 的双码本 + 显式 ρ 设定作为基础被继承, 但消融维度大幅扩展, 用新编号 #209.
2. **A0 用 #181 baseline**: 用户明示不重跑, 不需要自主决策.
3. **A4 欧式路径正则作为对照**: 用户明示 (审稿人必问的那一格), 不需要自主决策.
4. **ρ_targets = [2.0, 2.7, 3.4]**: 用户给定, 与 #208 保持一致.
5. **c = 1.0 固定**: 用户给定, 与 #199/#201/#203 链证伪的 c∈[1,10] 健康区一致.
6. **w_path = 1.0 默认**: 用户明示 Phase 1 用默认, Phase 1b 再扫.

---

**R12 checkpoint 强制**: 训练每 N=50 epoch 必须保存 + 删旧 ckpt, 即使 Phase 1 训练崩也要有 R12 best_ckpt 救场.
**R13 禁用 worktree**: 所有代码修改落共享 cwd `/fs04/ar57/wenyu/GeneRec/`.
