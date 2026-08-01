# Task #448 / Issue #157 [方向A Gate2] κ同步重校准的RQ-VAE代码本与完整SID链路验证

## Gate 目标
**Gate 2 = Stage 2 实施** (前序 Gate 1: #155 PASS commit 62bcc47, raw κ grad [233.48, 738.34, 54.93] + step delta [1.43e-5, 1.55e-4, 1.52e-6])

把 #155 三层 κ 感知更新 + #47 统一缩放 接入当前 RQ-VAE Stage 2 训练和推断:
- 每次 κ 更新后, 同步重校准相应层 codebook 向量尺度 / 距离 / projection 计算 / assignment 缓存
- 保持 L0 K=64 / L1 K=128 / L2 K=256 三组独立 κ
- 用 Musical_Instruments 冻结 item embedding + seed=42 输出完整 SID, 含第4位去重 digit

## 验收 (per Issue #157 spec)
1. **10+ κ 更新点记录**: 每层 κ / codebook 范数 / 有效半径 / 距离统计 / 重校准前后差异. κ 变化后不得使用旧尺度或旧距离缓存.
2. **checkpoint reload 一致**: 同一 batch 的 κ / codebook / 距离 / assignment 一致. 无 NaN/Inf.
3. **真实全量 (9922, 4) 整数 SID**: 含 shape / dtype / 范围 / SHA256 + item alignment 证据.
4. **对照关闭同步重校准消融**: 报告几何一致性诊断. 不允许以 R@K 代替 Stage 2 验收.

## R18 4 维度差异对比 (vs 历史)
| 维度 | #155 | #157 (本任务) |
|------|------|----------------|
| D1 spec 摘录 | 仅 Gate 1 monitoring 时序审计 | Gate 2 完整 Stage 2 链路 (codebook scale + distance + projection + assignment + SID) |
| D2 实施核心 | 改 train_step monitoring 时序 (raw grad before opt.step()) | 改 Stage 2 forward: κ 更新后同步重校准, 输出真实 SID |
| D3 Gate 失败机制 | monitoring grad=0 显示 bug (零 grad 是 opt.zero_grad() 后读的副作用) | 无 SID 链路证据 / 旧尺度缓存错配 |
| D4 引用文献 | 无 | arXiv:2405.13979 学习曲率与双曲尺度同步 |

→ **4 维度全部不一致**, 必须新实验.

## R11.5 自主决策 (实施)
- **Stage 2 RQ-VAE 框架**: 复用 #155 改好的 BilateralQuotaKappaModel + Hungarian bilateral cost expand
- **codebook 同步重校准**: 在每次 κ 更新 step 后, 对 codebook 范数做归一化 = √(c/κ) · x_eucl (即 #47 统一缩放公式)
- **distance 缓存失效**: κ 更新后强制 recompute distances (不 cache)
- **assignment 缓存失效**: κ 更新后强制重算 hard assignment
- **SID 输出**: 走标准 RQ-VAE Sinkhorn + 第4位 dedup, 输出 (9922, 4) 整数 SID
- **item alignment**: 用 row index 保证 item 顺序对齐, SHA256 全量 hash
- **GPU**: 分配 GPU 0 (空闲, 46068 MiB 可用)

## 阶段产物 (8 件套 + R12 ckpt 强制)
1. `descriptions/task448_issue157_gate2_kappa_sync_recalibration.md` (本文件)
2. `scripts/task448_issue157_gate2_kappa_sync_recalibration.py` (~600 lines, R4 py_compile OK)
3. `products/task448_issue157_gate2_kappa_sync_recalibration/config.json` (SHA256 item_emb=1a42341f01537d6d...)
4. `products/task448_issue157_gate2_kappa_sync_recalibration/precheck.json`
5. `products/task448_issue157_gate2_kappa_sync_recalibration/kappa_recalibration_log.json` (10+ κ 更新点)
6. `products/task448_issue157_gate2_kappa_sync_recalibration/sid_output.npy` ((9922, 4) int)
7. `products/task448_issue157_gate2_kappa_sync_recalibration/sid_metadata.json` (shape/dtype/range/SHA256)
8. `products/task448_issue157_gate2_kappa_sync_recalibration/verdict.json` (gate2_decision)
9. `verdicts/task448_issue157_gate2_kappa_sync_recalibration_result.md` (R20 4 Gate 详细)

## Gate 决策阈值
- **Gate 2 PASS** if: 10+ κ 更新点 + reload 一致 (5/5 check) + 无 NaN/Inf + 真实 SID SHA256 唯一 + item alignment + 对照消融诊断 PASS
- 任一项 FAIL → Gate 2 FAIL, STOP. 不进 Gate 3.

## R17 + R20 + R21 合规
- commit message: 含 issue # + Gate 状态 + 关键数据 + 失败原因 (R17) + 详细 4 Gate (R20)
- 落地后立即补 issue comment 含 commit hash (R21 v2, 禁止 pending/TBD)
- verdict 落地后立即 push (R15)
- issue close (R16) 含详细 4 Gate comment