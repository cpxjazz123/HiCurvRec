# Task #208 — 双码本几何解耦 (Phase 0 + 1 + 2 + 3)

> **任务目的**: 把每层的分配半径从"残差幅度的副产品"变成"显式设定的设计变量" `r_target = ρ/2`, 让双曲几何首次进入有效工作区; 同时验证其效果是否可被逐层加权 (臂 E) 替代.

> **完成日期**: (in progress)
> **状态**: 🟡 待启动

---

## 1. 背景

**承接**: Task #200 双码本 v3/v5 全线 FAIL (collision 85-99.5%, Stage 3 test R@10=0.0915 vs baseline 0.1020 -10%), 即"双码本 + 残差派生半径"路线不通.

**根因假设 (用户 2026-07-26)**:
- 原 init `r = ‖residual‖ / K` 让小残差层被推近原点 → 双曲几何进入"几乎平坦"工作区 → 没起到分配半径作用
- 必须显式设定 `r_target = ρ/2` (ρ ∈ {1.5, 2.7, 4.5}), 双曲几何才能真正工作

**本任务假设**:
- R1 (双码本解耦): `emb_geo` (方向码本) + `emb_rec` (残差码本) 分离 → 几何激活
- R2 (主推 C 臂): ρ_ℓ = 2.0/2.7/3.4 在 HG-Rec Musical_Instruments 数据上**几何激活 + 性能持平** 是通过判据, **不是刷分**
- R3 (欧式对照 E): 如果 C ≈ E, 双曲几何的作用完全等价于一组由几何推导出的逐层权重 — 必须如实报告

**前置必要不充分条件**:
- Phase 0 初始化验证通过 (`cos_max < 0.95`, `n_dup = 0`, `min ‖residual‖ > 1e-6`)
- Phase 0 不通过则**严禁**进入 Phase 1/2 (会全跑废)

---

## 2. 实验设计

**唯一改动**: 双码本初始化方式 + commit/code loss 显式 `r_target = ρ/2` 控制
**保持不变**: 官方 RQ-VAE Stage 1 架构 (3 层, num_emb_list=[64,128,256], e_dim=32), 500 epoch, batch_size=1024, lr=1e-3, kmeans_iters=1000, sk_epsilons=[0,0,0] (Sinkhorn 关闭).

### Phase 0: 初始化验证 (半天, 不占 GPU)

**只实现 `init_emb`, 不训练.** 用已训好的 baseline encoder 产生真实 latent, 三层各跑一次, 报:

| 指标 | 通过标准 |
|------|---------|
| `cos_mean` | ∈ [-0.05, 0.15] |
| **`cos_max`** | **< 0.95** |
| `n_dup` (cos > 0.99) | **= 0** |
| **`min ‖residual‖`** | **> 1e-6** ⚠️ 最后一项最重要 |

```python
@torch.no_grad()
def init_emb(self, latent):
    dirs = F.normalize(latent, dim=-1, eps=1e-8)
    centers = kmeans(dirs, self.n_e, self.kmeans_iters).float().to(latent.device)
    centers = F.normalize(centers, dim=-1, eps=1e-8)
    self.emb_geo.weight.data.copy_(centers)

    assign = (dirs @ centers.t()).argmax(dim=-1)
    rec = torch.zeros_like(centers)
    for k in range(self.n_e):
        m = assign == k
        rec[k] = latent[m].mean(0) if m.any() else latent[torch.randint(len(latent),(1,))].squeeze(0)
    self.emb_rec.weight.data.copy_(rec)
    self.initted = True
```

**如果 min 范数接近 0, 先解决这个再往下走.**

### Phase 1: 冒烟测试 (1 张卡, ~1 小时)

只跑臂 C, `--epochs 1000` 但**手动停在 epoch 50**. 看三件事:
- 有没有 NaN
- `cos_max` 是否稳定 (不上升)
- collision 是否在合理范围 (< 30%)

崩了就回 Phase 0 查初始化.

### Phase 2: 主实验 (4 张卡, 1-2 天)

臂 A (官方基线) 已有, 不重跑. **新增 4 臂:**

| 臂 | 配置 | ρ 目标 | 用途 |
|----|------|--------|------|
| **B** | 双码本 + 双曲 | 1.5 / 2.0 / 2.5 | 保守 |
| **C** | 双码本 + 双曲 | **2.0 / 2.7 / 3.4** | **主推** |
| **D** | 双码本 + 双曲 | 2.5 / 3.5 / 4.5 | 探上界 |
| **E** | 双码本 + 欧式加权 | 同 C | **关键对照** |

对应的 `r_target = ρ/2`: B = 0.75/1.0/1.25, C = 1.0/1.35/1.70, D = 1.25/1.75/2.25

#### 臂 E 的精确定义

**和 C 完全相同的结构 (双码本、归一化、同一套初始化), 只把几何 loss 的距离函数换掉:**

```python
# C (双曲)
commit = mean(poincare_distance(e_h.detach(), z_h, c) ** 2)
code   = mean(poincare_distance(e_h, z_h.detach(), c) ** 2)

# E (欧式 + 几何推导的层权重)
w = [0.1352, 0.5636, 2.3013]        # ∝ sinh²(ρ_ℓ), 均值归一
commit = w[layer] * mean(((e_dir.detach() - z_dir) ** 2).sum(-1))
code   = w[layer] * mean(((e_dir - z_dir.detach()) ** 2).sum(-1))
```

**E 的意义**: 如果 C ≈ E, 说明双曲几何的作用**完全等价于一组由几何推导出的逐层权重**. 这是必须回答的问题, 不能留给审稿人问.

权重来源: 小角度下 d ≈ sinh(ρ)·θ, 所以 d² 的有效权重是 sinh²(ρ).
sinh²(2.0)=13.15, sinh²(2.7)=54.85, sinh²(3.4)=223.96 → 归一化后 0.135 / 0.564 / 2.301

### Phase 3: Stage 2+3+4 评估 (主判据通过后, 4 张卡, ~1 天)

对通过主判据的臂 (Phase 2 collision ≤ 15%, cos_max < 0.95, util_raw ≥ 90%) 跑:
- Stage 2 SID 推断 + Sinkhorn + 4th-digit dedup
- Stage 3 T5-mini 9.18M, 200 epoch, early_stop=20
- Stage 4 evaluation, test R@10

**启动命令** (待 Phase 0 通过后细化):
```bash
# Phase 0 (init + latent diagnostic, 不训练)
python3 scripts/task208_phase0_init_diagnostic.py --layer 0
python3 scripts/task208_phase0_init_diagnostic.py --layer 1
python3 scripts/task208_phase0_init_diagnostic.py --layer 2

# Phase 1 (臂 C 冒烟)
python3 scripts/task208_phase1_smoke.py --arm C --epochs 1000 --manual_stop_epoch 50

# Phase 2 (4 臂 Stage 1)
python3 scripts/task208_phase2_train.py --arm B --rho_targets 0.75,1.0,1.25
python3 scripts/task208_phase2_train.py --arm C --rho_targets 1.0,1.35,1.70
python3 scripts/task208_phase2_train.py --arm D --rho_targets 1.25,1.75,2.25
python3 scripts/task208_phase2_train.py --arm E --rho_targets 1.0,1.35,1.70  # euclidean weighted
```

---

## 3. 决策触发 (vs HG-Rec baseline)

**主判据 (Stage 1 层面):**

| 检查 | 通过标准 |
|------|---------|
| 几何激活 | ρ = 2r 恒成立 (构造保证, 验证即可) |
| **没有方向坍缩** | `cos_max` < 0.95, `n_dup` = 0, 全程 |
| 码本可用 | `util_raw` ≥ 90%, collision ≤ 15% |
| 数值稳定 | 无 NaN, `resid_norm_min` > 1e-6 |

**次判据 (Stage 3 层面, 通过主判据后再跑):**

| 结果 | 结论 |
|------|------|
| **test R@10 ≥ 0.100** | ✅ 成功 — 几何激活且性能不掉 |
| test R@10 明显下降 | ❌ 归一化损失了必要信息, 需调整 |
| **C ≈ E** | ⚠️ 双曲 ≡ 逐层加权, **如实报告** |
| **C > E** | ⭐ 几何提供了加权之外的东西 |

### ⚠️ 期望管理

**成功标准是"几何激活 + 性能不掉", 不是刷分.**

Task #188 的 24 次评估已证明: 下游性能对码本质量不敏感 (碰撞率 +48%、深层半径 +95%、首层误差 +207%, 全部无影响). 所以大幅提升本来就不该被期待. 贡献是"让一个从未生效的机制第一次生效并可测量", 不是 SOTA.

如果 Phase 2 出来性能持平, **那是通过, 不是失败**.

---

## 4. 预算

| 阶段 | 估算时间 | 卡 |
|------|---------|-----|
| Phase 0 初始化验证 | ~4 h (CPU 可跑) | 0 |
| Phase 1 冒烟 | ~1 h | 1 |
| Phase 2 主实验 (4 臂) | 1-2 天 | 4 |
| Phase 3 Stage 2+3+4 | ~1 天 | 4 |
| **总计** | **~3 天** | - |

**建议排期**:
- ✅ Phase 0 今天就能做 (几乎不占资源)
- ✅ Phase 1 等 Phase 0 通过后做
- ⏸️ Phase 2 等 Sinkhorn 实验 (用户 2026-07-25 派工的 Sinkhorn Stage 3) 出结果后再排
- ⏸️ Phase 3 等 Phase 2 主判据通过后

---

## 5. 风险与缓解

| 风险 | 症状 | 应对 |
|------|------|------|
| **方向坍缩** | `cos_max` → 1, 利用率暴跌 | 球面 k-means 没做对; 或加方向多样性正则 |
| **零残差 NaN** | L2 层 NaN | `F.normalize(eps=1e-8)`, 监控 `resid_norm_min` |
| 两码本 index 错位 | 初期 loss 不降 | 必须用同一套 `assign` 初始化 |
| 重构变差 | `recon_loss` 高于基线 | 调大 `alpha` |
| D 臂数值溢出 | ρ=4.5 → √c·r=2.25 安全; 若后续加 κ 需检查 √c·r ≲ 5 | 记录 `√c·r` |
| 词表越界 | 4 臂 SID 任何前 3 列坍缩 → 去重饱和 | Phase 2 必须检查 col0/1/2 唯一值 ≥ 50/100/200 |

---

## 6. 与在跑任务的关系

- Task #194 K0 扫描 Stage 3: 用户已决策不需要 Stage 3+4, **不冲突**
- Sinkhorn 版 Stage 3 (用户 2026-07-25 派工): **优先级高于 Phase 2** — 那个验证"用 Sinkhorn 版 SID 重跑 Stage 3 是否提 +1~3%", 关系到"论文有没有用 Sinkhorn", Sinkhorn 结果决定 Phase 2 是否值得投 GPU
- Task #202 (Sinkhorn-on Stage 3 K0=64) verdict 已 FAIL (持平预测不成立), 用户给的预测 +1~3% 是错误基线, 重做以验证"论文 Sinkhorn 价值"

---

## 7. 产物预期 / 日志规格

```
per layer (per epoch):
  cos_mean, cos_max, n_dup        # 方向坍缩监控 ← 最重要
  util_raw                        # Sinkhorn 前真实利用率
  assign_entropy
  rec_err_p50                     # ‖latent − emb_rec[idx]‖ 中位数
  resid_norm_min                  # 数值安全
global (per epoch):
  collision_rate, recon_loss, quant_loss, beta_actual
```

**半径固定后 ρ 和 λ 是常数, 不用记. 必须记新的失效模式: 方向坍缩 / 数值 NaN / 词表越界.**

产物目录:
- `products/task208/init_diagnostic/` (Phase 0)
- `products/task208/phase1_smoke_C/` (Phase 1)
- `products/task208/phase2_arm_B/...C/...D/...E/` (Phase 2, 4 臂)
- `products/task208/phase3_stage3/` (Phase 3)
- `verdicts/task208_phase0_init_result.md` (Phase 0 完成立即写)
- `verdicts/task208_phase1_smoke_result.md`
- `verdicts/task208_phase2_4arms_result.md`
- `verdicts/task208_phase3_stage3_4_result.md` (终态)

---

## 8. 完成度跟踪

- [x] task208 description 登记 (loop.md §16 替换)
- [ ] Phase 0 init_emb 实现 + 三层 latent 诊断通过 (cos_max < 0.95, min ‖residual‖ > 1e-6)
- [ ] Phase 0 verdict 写入
- [ ] Phase 1 冒烟 (臂 C, ep 50 manual stop)
- [ ] Phase 1 verdict 写入
- [ ] Phase 2 主实验 4 臂 Stage 1 完成 (500 epoch × 4)
- [ ] Phase 2 主判据核验 (cos_max, util_raw, collision_rate)
- [ ] Phase 2 verdict 写入
- [ ] Phase 3 Stage 2 SID 推断 (4 臂 × Sinkhorn)
- [ ] Phase 3 Stage 3 T5-mini 训练 (4 臂 × 200 epoch)
- [ ] Phase 3 Stage 4 eval (4 臂 × test R@10)
- [ ] C vs E 对照判定 (双曲 vs 加权欧式)
- [ ] 论文 paper.md §5.7 / §6.3 整合
- [ ] 最终 verdict 写入 + loop.md §16 归档

---

## 9. 关键决策点 (R11.3 自主决策留痕)

1. **编号 #208 而不是 #198**: 旧 #198 (Stage 3 逐层可学习 κ) 已被 #200/#201/#203 链证伪, 用户重提的 Task 实际是双码本几何解耦, 不是原 #198. 用新编号 #208 避免冲突 (R9 编号连续).
2. **肩 C ρ 目标 = 2.0/2.7/3.4**: 用户给定, 不需要自主决策.
3. **臂 E 权重 ∝ sinh²(ρ_ℓ)**: 用户给定, 不需要自主决策.
4. **Phase 2 排期等 Sinkhorn 实验**: 用户文字明示, 不需要自主决策.

---

**R12 checkpoint 强制**: 训练每 N=50 epoch 必须保存 + 删旧 ckpt, 即使 Phase 2 训练崩也要有 R12 best_ckpt 救场.
**R13 禁用 worktree**: 所有代码修改落共享 cwd `/fs04/ar57/wenyu/GeneRec/`.
