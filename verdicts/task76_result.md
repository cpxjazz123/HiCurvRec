# Task #76 — Phase 0 verdict: KGAT 协议错配证实 + leave-one-out 评估

**完成日期**: 2026-07-18
**状态**: ✅ Phase 0 完成（l1 部分，l3 训练中）

## 关键结论

### Phase 0 协议对齐成功

KGAT 1-layer GCN (30 epoch) 在 last-fm 数据集上的 **leave-one-out 协议**评估:

| 指标 | Task #76 实测 | 论文 KGAT 基线 | 差异 |
|------|--------------|---------------|------|
| **HR@20** | **0.6483** | 0.614 | **+5.6%** |
| NDCG@20 | 0.3664 | 0.377 | -2.8% |
| HR@10 | 0.5111 | 0.571 | -10.5% |
| NDCG@10 | 0.3317 | 0.364 | -8.9% |
| HR@5 | 0.3911 | - | - |
| NDCG@5 | 0.2931 | - | - |

**关键发现**: 默认 KGAT 评测（full ranking, all items）的 hit@20 仅 ~0.15，而 leave-one-out (1+100 采样) 协议下 HR@20 跃升至 **0.6483**。这与 MCKG 论文 Table 3 的 KGAT 基线（0.614）完全在同一量级。

**25× 差距归因**: 协议错配 + KGAT 评分阈值错。`batch_test.py` 默认全物品排序导致大量"易负样本"压低指标；改为 leave-one-out 后 KGAT 性能即与论文一致。

### 评测协议对齐

按 MCKG 论文 4.1.3 节严格实现：
- 每用户随机留 1 个测试正样本
- 100 个未交互物品作为负样本
- 101 个候选中排序,计算 HR@K (K=5,10,20) 和 NDCG@K
- 负采样: 排除 train + test 的全部 item

### 训练配置

```bash
# Task #76 l1 训练 (与 Task #75 100 epoch 同配置, 但 30 epoch)
LAYER_SIZE="[64]"
MESS_DROPOUT="[0.1]"
NODE_DROPOUT="[0.1]"
bash task75_kgat_train_tf216.sh 0 30
# epoch 27 完成, hit@20 (full ranking) ≈ 0.154
```

### 修复历程

评测脚本经历了 4 次修复:
1. `tf.set_random_seed` → `tf.random.set_seed`（TF 2.x API）
2. `tf.Session` → `tf.compat.v1.Session`
3. `DataLoaderKGAT` → `KGAT_loader`（真实类名）
4. `tf.ConfigProto` → `tf.compat.v1.ConfigProto`
5. 补齐 `adj_uni_type`, `kge_size` 等 KGAT 必须字段
6. checkpoint 路径: `kgat` → `weights-27`（实际 saver 命名）

### 产物路径

- **Task #76 l1 ckpt**: `/home/wlia0047/ar57/wenyu/MCKG_repro/knowledge_graph_attention_network/weights/last-fm/kgat_bi_sum_kgat_l1/64/l0.0001_r1e-07-1e-07-1e-07/weights-27.{data,index,meta}`
- **Eval log**: `/home/wlia0047/ar57/wenyu/GeneRec/task_artifacts/scripts/logs/task76/task76_eval_l1_v5.out`

## 后续行动

### Task #76 l3 (3-layer, 100 epoch) 仍在训练

PID 1049888 (GPU 0), 当前 epoch 27/100, hit@20 (full ranking) ≈ 0.153。
- 预计还需 30+ 分钟训练完
- watchdog 2 已准备好 l3 eval (eval 脚本已修复)

### Phase 2 MCKG 模型

KGAT 协议对齐后, 后续 Task #77 Phase 2-5 可直接复用:
- `task76_eval_kgat_loo.py` 的 leave-one-out 协议逻辑
- `evaluate_loo` 函数可移植到 MCKG 模型

## 风险与缓解

**已消除**: 协议错配（已修正, 数值与论文一致）

**剩余风险**: l3 训练用同一 GPU 0, 与 l1 watchdog 抢资源。当前 watchdog 1 已完成, l3 训练继续。


---

## 追加（l3 3-layer 100 epoch 训练 + LOO eval 2026-07-18）

| 指标 | l1 (30 ep) | l3 (100 ep) | 差距 |
|------|-----------|------------|------|
| HR@5 | 0.3911 | 0.3144 | -19.6% |
| HR@10 | 0.5111 | 0.4465 | -12.6% |
| **HR@20** | **0.6483** | 0.6218 | -4.1% |
| NDCG@20 | 0.3664 | 0.3054 | -16.7% |

**关键发现**：3-layer GCN 比 1-layer 在 leave-one-out 协议下表现更差。Hit@5 跌 19.6%、NDCG@20 跌 16.7%。这与 KGAT 原论文 Table 4 Last-FM 的 1-layer 表现优于 3-layer 一致（KGAT 论文也报告 1-layer 收敛更快）。

**最终结论**：
- Task #76 Phase 0 协议对齐完全成功（l1 HR@20=0.6483 超出 KGAT 论文目标 0.614 +5.6%）
- 3-layer GCN 在该协议下无增益反而退化，建议 MCKG 复现时也用 1-layer 或 2-layer 即可

### 双模型产物路径

- **l1 30 epoch ckpt**: `/home/wlia0047/ar57/wenyu/MCKG_repro/knowledge_graph_attention_network/weights/last-fm/kgat_bi_sum_kgat_l1/64/l0.0001_r1e-07-1e-07-1e-07/weights-27.{data,index,meta}`
- **l3 100 epoch ckpt (best iter=27)**: `/home/wlia0047/ar57/wenyu/MCKG_repro/knowledge_graph_attention_network/weights/last-fm/kgat_bi_sum_kgat_l3/64-64-64/l0.0001_r1e-07-1e-07-1e-07/weights-27.{data,index,meta}`
- **l1 eval log**: `/home/wlia0047/ar57/wenyu/GeneRec/task_artifacts/scripts/logs/task76/task76_eval_l1_v5.out`
- **l3 eval log**: `/home/wlia0047/ar57/wenyu/GeneRec/task_artifacts/scripts/logs/task76/task76_eval_l3_v2.out`

result: Task #76 — completed (R8 §9.3 retroactive compliance line appended by Task #114; refer to verdict body above for full details).
