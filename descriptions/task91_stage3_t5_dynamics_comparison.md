# Task #91 — Stage 3 T5 训练动力学对比 (analytical, no GPU training)

> **任务目的**: 解析 8 个 Stage 3 T5 训练日志 (vanilla / c111 / c222 / c555 / c512 / c215 / c1055 / free-curv), 对比收敛速度 / best epoch timing / valid R@10 轨迹 / 泛化 gap. 解释 phonism R@10=0.1058 vs HG-Rec c555 R@10=0.1051 (+0.7%) 在 T5 训练阶段的差异来源.

> **完成日期**: (in progress)
> **状态**: 🟡 待启动

---

## 1. 背景

### 1.1 Task #90 闭环后的新问题

Task #90 codebook 分解显示:
- vanilla / c111 / c555 L0/L1/L2 token SET 完全共享 (Jaccard=1.000)
- 4-col SID 实质独立 (Jaccard=0.001)
- c555 L0 略集中 (normH 0.969 vs vanilla 0.983)

**新问题**: c555 R@10=0.1051 vs vanilla R@10=0.1058 (+0.7% 反向!) — 是 codebook 差异, 还是 T5 训练动力学差异?

### 1.2 8 个 Stage 3 训练日志 (已存在, 需解析)

| 配置 | 日志路径 | code_path suffix |
|------|----------|------------------|
| **vanilla (phonism)** | `logs/Instruments/Jul-23-2026_20-24-44/HG_Rec.log` | `_t5_hrqvae_poincare.npy` |
| **c111 (Task #88)** | `logs/Instruments/Jul-23-2026_23-44-03/HG_Rec.log` | `_curv_1.0_1.0_1.0_t5_hrqvae_poincare.npy` |
| **c222 (Task #88)** | `logs/Instruments/Jul-24-2026_00-06-10/HG_Rec.log` | `_curv_2.0_2.0_2.0_t5_hrqvae_poincare.npy` |
| **c555 (Task #88)** | `logs/Instruments/Jul-23-2026_23-48-37/HG_Rec.log` | `_curv_0.5_0.5_0.5_t5_hrqvae_poincare.npy` |
| **c512 (Task #88)** | `logs_curv_0.5_1.0_2.0/Instruments/Jul-24-2026_00-17-36/HG_Rec.log` | `_curv_0.5_1.0_2.0_t5_hrqvae_poincare.npy` |
| **c215 (Task #88)** | `logs_curv_2.0_1.0_0.5/Instruments/Jul-24-2026_00-58-58/HG_Rec.log` | `_curv_2.0_1.0_0.5_t5_hrqvae_poincare.npy` |
| **c1055 (Task #88)** | `logs_curv_1.0_0.5_0.5/Instruments/Jul-24-2026_00-59-22/HG_Rec.log` | `_curv_1.0_0.5_0.5_t5_hrqvae_poincare.npy` |
| **free-curv (Task #89)** | `logs_curv_free_M1/Instruments/Jul-24-2026_02-11-11/HG_Rec.log` | `_curv_free_M1_t5_hrqvae_poincare.npy` |

---

## 2. 实验设计

### 2.1 变量

**比较 8 个 T5 训练运行的 Stage 3 动力学**

### 2.2 保持不变

- T5 模型架构 (T5-small, 6 enc + 4 dec, d_model=128)
- 训练配置 (batch_size=256, lr=1e-4, 200 epochs, early_stop=20)
- 数据集 (Musical_Instruments)
- seed=42

### 2.3 测量项 (从日志解析)

**每个 epoch**:
- Training loss
- Valid R@5, R@10, R@20, NDCG@5, NDCG@10, NDCG@20

**聚合指标**:
- 收敛速度: 首次达到 R@10 ≥ 0.08 的 epoch
- Best epoch (NDCG@20 最高)
- Best epoch R@10 / R@5 / NDCG@10
- Early stop epoch (训练终止)
- 总训练时长 (epoch × epoch_time)

**比较指标**:
- Best valid R@10 排名
- 收敛 epoch 排名
- Best valid vs Stage 4 test R@10 差距 (泛化 gap)

### 2.4 启动命令

**纯 CPU 解析, 无 GPU**:

```bash
cd /home/wlia0047/ar57/wenyu/GeneRec
python3 scripts/task91_stage3_dynamics.py \
    --output_json verdicts/task91_stage3_dynamics.json \
    --output_md verdicts/task91_stage3_dynamics_result.md
```

**预期时间**: 解析 8 个 log × 200 epoch ≈ 30 sec.

---

## 3. 决策触发

| 观察 | 决策 |
|------|------|
| **vanilla 收敛显著快于 c555**: vanilla 在 epoch X 达到 R@10=Y, c555 需更多 epoch | ✅ 解释 "phonism R@10=0.1058 > c555 0.1051" 来自 vanilla 收敛速度 → 论文 Section 5.4 报告 "vanilla + Sinkhorn outperforms hyperbolic on convergence speed" |
| **vanilla 与 c555 收敛曲线几乎重合**: epoch-by-epoch R@10 差异 < 1% | ⚠️ → R@10 0.1058 vs 0.1051 差异主要来自 Stage 4 eval 噪声 → 论文报告 "statistically tied" |
| **c555 early stop 显著晚于 vanilla**: c555 训练时间多 30% | ❌ → c555 优势是 overfitting on valid set, test 上未保持 → 论文标注 "c555 may overfit valid set" |
| 全部否证 | ❌ open question, 建议 future work |

---

## 4. 预算

| 阶段 | 估算时间 | GPU |
|------|---------|-----|
| 解析 8 个 log | ~30 sec | 无 |
| 聚合指标计算 | ~10 sec | 无 |
| 写 verdict | ~10 min | 无 |
| **总计** | **~12 min** | **0 GPU** |

---

## 5. 风险与缓解

**风险 1**: `Jul-24-2026_00-06-10/HG_Rec.log` 含 c512 + c222 两个训练 (同一时间戳) → 写脚本时按时间戳分段提取
**风险 2**: 部分 log 早期被 kill (epochs=0, c111-v2 23-39-21) → 跳过异常 log
**风险 3**: Stage 4 test R@10 vs Stage 3 valid R@10 比较需从 verdicts 提取 → 脚本中 hardcode mapping

---

## 6. 完成度跟踪

- [ ] 写 `scripts/task91_stage3_dynamics.py`
- [ ] 解析 8 个 log 提取 epoch-by-epoch metrics
- [ ] 计算聚合指标 (best epoch, 收敛速度, 训练时长)
- [ ] 加载 Stage 4 test R@10 (from verdicts)
- [ ] 综合分析
- [ ] 写 verdict `verdicts/task91_stage3_dynamics_result.md`
- [ ] 更新 loop.md §16 (R8 归档)

---

## 7. 关联

- 前置: Task #84 (HG-Rec c111), Task #88 (6 curvature Stage 4), Task #89 (free-curv), Task #90 (codebook decomposition)
- 关联: Task #32 (phonism baseline)
- 后续: 论文 Section 5.4 final writeup

---

**核心问题**: phonism 0.1058 vs HG-Rec c555 0.1051 在 T5 训练阶段是收敛速度 / best epoch / 泛化 gap 哪个驱动?