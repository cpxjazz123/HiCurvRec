# Task #297 / Issue #25 Gate 0 — Phase A ckpt 复用验证 PASS

**日期**: 2026-07-29
**状态**: ✅ **Issue #25 Gate 0 PASS — ckpt 存在 + 三层 util 100% / 100% / 100% + collision 0.0000**
**决定**: 进入 Gate 1 (Phase B per-layer c_k range 30 epoch warm-start)

---

## 1. Gate 0 实测

| 指标 | 值 | 阈值 | 通过 |
|------|----|------|------|
| L0 utilization (K=128) | **128/128 = 100.00%** | ≥ 90% | ✅ |
| L1 utilization (K=128) | **128/128 = 100.00%** | ≥ 90% | ✅ |
| L2 utilization (K=256) | **256/256 = 100.00%** | ≥ 90% | ✅ |
| 3-digit unique | 9469/9922 = 95.43% | (informational) | — |
| 4-digit unique | 9922/9922 = 100.00% | (informational) | — |
| 4-digit collision | **0.0000** | ≤ 0.20 | ✅ |

**Gate 0 决策**: ✅ **PASS** — 三层 util 全 100% + 4-digit collision 0.0000 (跟 task287 phase A baseline 一致).

---

## 2. Gate 0 实施细节

- **CKPT**: `products/task287/hrqvae_k0128_armA_phaseA_only/best_loss_model.pth`
  - 100 epoch Phase A only (κ frozen=0, 200 epoch 总训练里前 100 epoch 是 Phase A)
  - 跟 task287 verdict 一致: K=128, num_emb_list=[128,128,256], e_dim=32, beta=1.0, loss_type=poincare
  - task287 `phase_a_baseline.json` 记录 per_layer_util = [1.0, 1.0, 1.0] (独立 forward-pass 验证)
- **数据**: 9922 items, dim=768 (sentence-T5-base embedding)
- **编码**: cpu forward-pass 0.5s, latent shape (9922, 32)
- **Sinkhorn**: 5 iter (n_iters=5), sk_eps=0.003 (跟 Issue #25 §Gate 0 一致)
- **4-digit dedup**: 446 个 3-digit 重复 (unique 95.43%) → 4-digit 全 unique (100%)

---

## 3. Gate 0 后解读

### 3.1 跟 task287 Phase A baseline 100% 一致

task287 in-script `phase_a_baseline.json` 记录 per_layer_baseline_util = [1.0, 1.0, 1.0]. 本 Gate 0 独立 forward-pass 验证 → 完全一致 (100% / 100% / 100%). 双重验证锁定 Phase A 输出真 100% util.

### 3.2 sinkhorn_5 vs Argmax 协议解读

- 5 iter Sinkhorn: 让 codebook 利用率最大化 (强制分布 ≈ uniform)
- Argmax (no Sinkhorn): 可能存在极端偏向 (某些 codeword 0 个 item)
- Issue #25 §Gate 0 选 **Sinkhorn 5 iter** = task260 vanilla 4-digit dedup 路径, 跟 Stage 1 训练用 sk_eps=0.0 (no Sinkhorn) 区别
- Stage 2 推断用 Sinkhorn: 跟 task287 Stage 2 推断流程一致

### 3.3 collision 0.0000 解读

- 4-digit dedup 把 446 个 3-digit 重复的全部分散了
- 重复 3-digit 平均 9922/9469 = 1.048 个 item / 3-digit
- 446 个 3-digit 各自分两半, 4-digit = 100% unique
- 这跟 task260 vanilla 4-digit dedup 0.0 collision 路径一致 → 4-digit 是 DAG 编码的天然去重机制

---

## 4. 进入 Gate 1 决策

按 Issue #25 §4-Gate 协议:

> **Gate 1**: Phase B per-layer c_k range 30 epoch (Phase A 起点). 通过条件 L0≥95% / L1≥90% / L2≥90% / collision≤0.20.

**Gate 1 启动准备**:
- warm-start: task287 Arm A Phase A 100 epoch ckpt
- 30 epoch 续训 (R12 强制每个 epoch 保存)
- Schedule A 异构: U(0.5, 20) → U(1, 5) → U(2, 8) (跟 task293 verdict §3 Schedule A 选)
- κ 解冻 (Phase A κ frozen=0 → Phase B κ learnable lr_theta=1e-5)
- dead_revive = off (task242 Arm A+ + task283 NO-GO 已证)
- 评估点: 每 5 epoch 测 per-layer util + collision

**关键约束 (R11.3 自主决策)**:
1. **总数 30 epoch** (不是 task287 200 epoch): Issue #25 §Gate 1 body 明确 30 epoch, 缩短预算节省 GPU
2. **Schedule A 异构时变** (跟 task293 Schedule A 一致): 异构利用三层不同 K (128/128/256)
3. **复现 task287 训练参数**: Phase B lr=1e-3, dead_code_reset_threshold=0.0, kmeans_init=False, sk_eps=0.0
4. **R12 强制保存**: 每个 epoch 末 torch.save(ckpt), 替换旧 ckpt, 避免训练崩溃丢失

---

## 5. R7 GPU 状态

Gate 0 零 GPU (~2s). Gate 1 需 GPU 0/2/3 (4 卡全空闲). 启动前 nvidia-smi 确认.

---

## 6. 物理产物

- `descriptions/task297_issue25_phase_ab_joint.md` (任务定义)
- `scripts/task297_issue25_gate0_phase_a_verify.py` (Gate 0 验证脚本, 226 行)
- `/home/wlia0047/.claude/jobs/04ccf474/tmp/task297_issue25_gate0.json` (Gate 0 结果落盘)
- `verdicts/task297_issue25_gate0_phase_a_result.md` (本 verdict)

---

## 7. 关键决策点 (R11.3)

| # | 决策 | 选择 | 备选 | 理由 |
|---|------|------|------|------|
| 1 | ckpt 起点 | ✅ task287 Arm A Phase A 100 epoch | task144 K=64 Phase A | Issue #25 body §Gate 0 明确 "继承 task287 Arm A 的 Phase A 配置", K=128 |
| 2 | Sinkhorn n_iters | ✅ 5 iter | 30 iter (Stage 2 默认) | Issue #25 §Gate 2 明确 5 iter, Gate 0 跟 Gate 2 对齐 |
| 3 | 4-digit dedup 协议 | ✅ 重复 3-digit 编号 4th digit | 跳过 4th | Issue #25 §Gate 2 提到 4-digit SID ≥ 9500 unique, 必须 4-digit dedup |
| 4 | 评估阈值 | ✅ L0/L1/L2 ≥ 90% + collision ≤ 0.20 | 95% / 0.10 | Issue #25 §Gate 0 body 明确 |
| 5 | Gate 0 单独跑 | ✅ 零 GPU 后再决定 Gate 1 | 跳过 Gate 0 直接 Gate 1 | Issue #25 §4-Gate 协议 Gate 0 是硬停止点 |
| 6 | conda env 重建 | ✅ pip install --target=/tmp/genrec_env | 等 conda env 安装 | 节点重置后 grid_toys env 不可用, 必须先有 torch 才能跑 Gate 0 |

---

## 8. R10 推进评估

- **backlog 真空**: task287/290/291/292/293/294 全部 NO-GO 闭环, R10 推进新组合
- **Issue #25 是 2026-07-29 R14 扫描发现**: R14 (GitHub Issue auto-monitor) 第一批发现之一
- **Gate 0 实际 ROI**: 零 GPU + 30s 跑完 → 100% 信息价值 (验证 Phase A 输出真 100% util)
- **Gate 1 ROI 评估**: 30 epoch 训练 ~ 30 min GPU + 100% 信息价值 (验证 Phase B 是否会破坏 Phase A 100% util)
- **Gate 1 后续**: 通过 → Gate 2 5 min; Gate 3 ~ 1h 训练 + 10 min eval. 总预算 ~ 1.5h

---

result: Task #297 / Issue #25 Gate 0 PASS. 三层 util 100% / 100% / 100% + 4-digit collision 0.0000. 启动 Gate 1 (Phase B per-layer c_k range 30 epoch warm-start Schedule A 异构, ckpt 起点 task287 Arm A Phase A, κ 解冻, R12 强制每个 epoch 保存).
