# Task #54 — MCKG L3 norm regularization 重训

> **任务目的**: 从源头修复 MCKG norm 长尾 (在 margin ranking loss 之外加 L3 norm penalty), 验证训出 metric 结构健康的 MCKG embedding
> **完成日期**: (in progress)
> **状态**: 🟡 待启动

---

## 1. 背景

承接 Task #44 V5 + Task #52 终判: L1 log1p 是"临床缓解", L3 norm regularization 是"源头治疗" — 让 MCKG 训出 metric 结构健康的 embedding. 若 L3 成功, 则 SCR 4.22x → 接近 1.0x, 不再需要 log1p 后处理.

## 2. 实验设计

**变量**: MCKG 训练 loss (margin ranking → margin ranking + λ·||e||³)
**保持不变**:
- MCKG 架构 (3 κ 子空间 + fused)
- Toys 数据集 (11924 items + 19412 users + KG neighbors)
- lr, batch, epochs
- seed=42

**启动命令** (基于 `task_artifacts/scripts/mckg_model/mckg.py`):
```bash
source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh && conda activate /home/wlia0047/ar57_scratch/wenyu/kgat_tf216
cd /home/wlia0047/ar57/wenyu/MCKG_repro/knowledge_graph_attention_network
python3 -m src.train \
    config=mc_repro_v1_lambda_0.01 \
    lambda_norm=0.01 \
    seed=42 \
    task_name=task162_l3_lambda001

# λ sweep: 0.001 / 0.01 / 0.1 / 1.0
```

## 3. 决策触发

| 训出 norm max | 判定 |
|--------------|------|
| max_norm < 5.0 | ✅ L3 成功, 用此 embedding 替换 products/task99_mckg_rebuild/entity_embedding.pt |
| 5-50 | ⚠️ 部分缓解, λ 需加大 |
| > 50 | ❌ L3 不足以对抗 margin loss 内在偏好, 保持 log1p 后处理 |

## 4. 预算

| 阶段 | 估算时间 |
|------|---------|
| MCKG 重训 × 4 λ | ~4 h (每 λ ~1 h) |
| QMP 重测 | ~5 min (复用 Task #49 脚本) |
| **总计** | **~4 h** |

## 5. 风险与缓解

**风险 1**: λ sweep 资源消耗大 → 缓解: 先单 λ=0.01 试, 训出后再决定是否需要 sweep
**风险 2**: MCKG 训练无 ckpt (TensorFlow checkpoint 行为差异) → 缓解: 每 epoch 强制 save
**风险 3**: KGAT TF 216 env 启动慢 → 缓解: 后台启动 (run_in_background=true), 立即开始

## 6. 完成度跟踪

- [ ] MCKG 训练配置加 L3 term
- [ ] λ=0.01 单 run launch
- [ ] MCKG 训出 entity_embedding.pt
- [ ] QMP 重测 (复用 Task #49)
- [ ] λ sweep (按需)
- [ ] 写 verdict (L3 路线 vs log1p 路线优劣)
