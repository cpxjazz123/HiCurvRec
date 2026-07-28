# Task #161 — T5-mini 9.18M d_kv fix 探测 (ladder 第 2.5 点, R11.3 主动推进)

> **任务目的**: 验证 Task #159 (T5-mini 9.18M R@10=0.0978) 反向 -5% 是否因 `num_heads × d_kv ≠ d_model` 数值不稳, 而非 T5-mini 容量本身.
> **完成日期**: (in progress)
> **状态**: 🟡 待启动 (R11.3 自主决策, 利用 GPU 1 空闲, 解释 9.18M 反向根因)

---

## 1. 背景

4 档 T5 容量 ladder 进度 2/4: Task #156 (5.5M) R@10=0.1030 ✅, Task #159 (9.18M) R@10=0.0978 ❌ **反向 -5%**. 9.18M 是异常信号, 与"容量单调 scaling"假设矛盾.

**核心嫌疑 (R11.3 假设)**: Task #159 用 `d_model=256, num_heads=6, d_kv=64`, 而 `6 × 64 = 384 > 256 = d_model`. T5 multi-head attention 严格要求 `num_heads × d_kv = d_model`. 错配触发数值不稳, 在前向/反向传播中 d_kv padding 错位 → 训练动力学崩溃 → R@10 反向.

**Task #161 假设**: 修复 `num_heads × d_kv = d_model` (选 `heads=4, d_kv=64` ⇒ `4 × 64 = 256`) → R@10 应回升至 ≥ 0.103 (超过 Task #156 5.5M baseline).

---

## 2. 实验设计

**变量 (1 变量 fix)**:
- `num_heads`: 6 → **4**
- `num_decoder_layers`: 4 (维持, 非 decoder heads)
- **总参变化**: 9.18M → ~8.5M (-7%), 仍属 T5-mini 容量档

**保持不变**:
- Stage 2 codebook: Task #156 复用 (`β=0.25, [32,64,256], sk=0.5`)
- Stage 3 模型: 4 enc + 4 dec layers, d_model=256, d_ff=1024, d_kv=64
- 训练: 200 epoch, lr=1e-4, batch=256, seed=42, 早停机制一致
- Stage 4: Recall@5/10/20 + NDCG@5/10/20, beam_size=20

**启动命令 (计划)**:
```bash
# Stage 3: 复用 task84 fork, 改 num_heads 6→4
nohup bash /home/wlia0047/ar57/wenyu/GeneRec/scripts/task161_hgrec_t5mini_dkvfix_stage3.sh > /home/wlia0047/ar57/wenyu/GeneRec/logs/task161/stage3_nohup.out 2>&1 &
# Stage 4: 复用 task159 stage4 launcher, 改 products 路径 → task161, 改 num_heads
nohup bash /home/wlia0047/ar57/wenyu/GeneRec/scripts/task161_hgrec_t5mini_dkvfix_stage4_eval.sh
```

---

## 3. 决策触发 (vs Task #159 baseline R@10=0.0978)

| 指标条件 | R@10 区间 | 决策 |
|----------|-----------|------|
| **d_kv fix 验证成功** | R@10 ≥ 0.103 (超过 Task #156 5.5M) | ✅ **d_kv 错配是 9.18M 反向根因**. ladder 结论: T5 容量单调 scaling 成立, 但需避开 heads × d_kv 错配点. |
| **d_kv fix 失败** | R@10 ≤ 0.103 (≤ 5.5M baseline) | ❌ d_kv 不是根因. R11.3 后续: 9.18M 容量本身失败, 与 d_kv 无关, 走 R3 切换调查方向 (paper GitHub diff). |
| **d_kv fix 改善但仍低** | 0.0978 < R@10 < 0.103 | ⚠️ d_kv 是部分根因. 记录到综合 verdict, 不重跑其他 d_kv 配置. |

---

## 4. 预算

| 阶段 | 估算时间 |
|------|---------|
| Stage 3 训练 (4+4 layers, 估计早停) | ~30-40 min |
| Stage 4 inference | ~1 min |
| R@10 评估 | ~1 min |
| **总计** | **~40 min** |

GPU 占用: 1 张 L40S (空 GPU 1).

---

## 5. 风险与缓解

**风险 1**: `num_heads=4` 在 d_model=256 下 attention 头数太少, 可能不足以建模 R@10 提升 (与 d_kv fix 无关). → **缓解**: 即使 R@10 仍低, 也算 d_kv 假设 falsified, 写 verdict 归因. 不再重试 heads=8 d_kv=32 (heads 加倍 → 训练动力学不同, 变量混淆).

**风险 2**: Task #159 Stage 3 跑过 NDCG@20 plateau 早停, Task #161 也会早停 → 实际跑 epoch 数不可控. → **缓解**: 不依赖 epoch 数解释, 只看最终 R@10 vs baseline.

**风险 3**: GPU 1 仍可能被 Task #160 (#160 不在 GPU 1, 在 GPU 2) 或其他新任务抢用. → **缓解**: R7 启动前 re-check nvidia-smi, 若 util > 10% 则推迟启动.

---

## 6. 完成度跟踪

- [ ] Stage 3 launcher 创建 (从 task159 fork, 改 num_heads 6→4)
- [ ] Stage 4 launcher 创建 (从 task159 fork, 改 products 路径 + num_heads)
- [ ] bash -n 验证 launcher syntax
- [ ] Stage 3 启动 (GPU 1, 监控 NDCG@20 plateau 早停)
- [ ] Stage 3 完成 (exit 0 + best ckpt 落盘 products/task161/)
- [ ] Stage 4 inference 完成 (verdict JSON 落盘)
- [ ] 比较 vs Task #159 baseline (R@10=0.0978) + Task #156 5.5M (R@10=0.1030)
- [ ] 写 verdict: `verdicts/task161_t5mini_dkv_fix_result.md`
- [ ] 更新 §16 R8 cleanup + composite verdict (与 #156/#159/#160/#157 综合到 `task161_capacity_ladder_5point_synthesis.md`)

---

## 7. R11.3 决策明示 (写入 loop.md §16 备注)

**(a) 选了哪个**: Task #161 T5-mini 9.18M d_kv fix 探测 (heads 6→4, d_kv 维持 64, 总参 9.18M → ~8.5M)

**(b) 为什么**:
- 9.18M 反向 -5% 是异常信号, 需要归因而非等待
- `heads × d_kv ≠ d_model` 是 T5 multi-head attention 已知数值陷阱
- 启动 GPU 1 空闲是 R7 + R10 主动推进要求
- ROI 高: 解释 9.18M 失败 = 理解 T5-mini 容量真伪 = 决定 4 档 ladder 结论方向

**(c) 备选方案**:
- **A**: heads=8 d_kv=32 (整除但头数加倍) — 训练动力学不同, 变量混淆
- **B**: 完整 200 epoch 重跑 Task #159 (无 d_kv 改) — R12 已说明不必要, 早停已确认 plateau
- **C**: 等 Task #160 + #157 收齐 4 档后归因 — 低 ROI 等待, GPU 1 闲置
- **D**: 跑 d_kv=42 heads=6 (凑 252, 不严格整除 256) — 数学不干净, 仍是错配版

选 A 风险高 (头数变化混淆), B 重复无效, C 浪费 GPU, D 数学不干净. → 选本次决策 (heads=4 d_kv=64).

---

result: Task #161 — Stage 3 + Stage 4 启动 + 完成 + 写 verdict. 预期 R@10 ≥ 0.103 (d_kv fix 验证成功) 或 ≤ 0.103 (d_kv 不是根因).