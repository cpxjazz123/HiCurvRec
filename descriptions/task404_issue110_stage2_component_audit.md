# Task #404 / Issue #110 [方向B Gate2] #108 best product checkpoint 的 SID 与 component 审计

**日期**: 2026-07-31
**任务**: 复用 task401 ckpt (Issue #108 best product checkpoint) → Stage 2 SID 9922×4 生成 + component contribution + gate/mixing 分布 + 原/均/shuffle 三组对照审计
**目的**: 验证 #108 best product checkpoint 的 mixed-curvature component + mixing/gate 是否在 Sinkhorn/SID 生成中仍有效
**不重训 Stage 1** (per Issue spec §区别于 #98/#101/#106/#107/#108)

## Gate 0 (实施前预检)

- [x] task401 ckpt 已落盘: `products/task401_issue108_best_epoch_verify/ckpt/issue108_ckpt.pt` (60701 bytes, sha256=36d68fe0...)
- [x] task401 ckpt 配置: 30 epoch early-stop, L0/L1/L2 utilization 100%, per-component κ_l,m + std norm + mean agg
- [x] HG-Rec Stage 2 Sinkhorn script 模板: `scripts/task402_stage2_codebook.py`
- [x] GPU 1 全空闲 (util 0%, mem 0 MiB) — task404 分配 GPU 1
- [x] task403 Stage 3 仍跑 GPU 0 (R7 不抢)

## Gate 1 (Stage 1 ckpt 复用)

- 直接 load task401 ckpt (不重训)
- 核对: sha256 + 3 层 K=64/128/256 + component params + mixing/gate 非零 + 无 NaN/Inf
- 不一致立即 STOP

## Gate 2 (Stage 2 SID 生成 + 审计)

按 Issue #110 spec §Gate2 要求生成 9922×4 SID, 并报告:
1. **SID unique / collision**: 期望 unique=9922/9922, collision=0%
2. **L0/L1/L2 utilization / entropy / max_load**: 期望 utilization ≥90%, max_load <5%
3. **每层各 component contribution / gate/mixing 分布 / assignment overlap**: 必须有可重复非零影响
4. **原 gate / 均匀 gate / component shuffle 三组对照**: 必须显示 component/gate 对输出有可重复非零影响
5. **item → SID 一一对齐 + 可复现命令 + 日志 + artifact + verdict + commit**

**决策**: PASS 需同时满足 unique=9922/9922, collision=0%, 三层 utilization≥90%, max_load<5%, component/gate 对输出有可重复非零影响, 无 NaN/Inf.

任一 component 静默删除 / gate 全零 / 只靠去重 digit 掩盖前三层坍缩均 FAIL.

## Gate 3 (Stage 3 条件信号接口) ⏸ STOP per spec

- Issue #110 spec 明确: 仅 Gate 2 PASS 后才定义 component/gate/learned-κ metadata 到 T5 representation 的接口, 并做 on/off / shuffle / 梯度验证
- 若 Gate 2 PASS → 触发 Gate 3 接口设计 (复用 Issue #109 / #111 的 T5 几何消费反事实 protocol)

## Gate 4 (Stage 4 R@K) ⏸ STOP per spec

- Issue #110 spec 明确: 仅 Gate 3 PASS 后才统一对 Task84 评估, R@10 > 0.1020 才是 Target reached

## R11.5 自主决策

| 决策点 | 选择 | 理由 |
|--------|------|------|
| GPU 分配 | GPU 1 (task403 用 GPU 0) | R7 规则: 不抢已占卡, GPU 1/2/3 全空闲, 默认 GPU 1 |
| SID 生成脚本 | 复用 task402_stage2_codebook.py 模板 + 改 ckpt path | task402 验证 work, 改 ckpt 即可 |
| Component 审计方式 | ckpt load 后 inspect 每一层 component params + mixing/gate weight + Sinkhorn 生成中看 component 影响 | Issue #110 spec 强制要求 |
| 决策阈值 | unique=9922/9922, collision=0%, 三层 utilization≥90%, max_load<5%, component/gate 非零影响 | Issue #110 spec §Gate2 PASS 条件 |
| 下一方向 (若 Gate 2 FAIL) | 同 task401 ckpt + #97 patch 重新跑 (Issue #110 路径必须从 #108 best product ckpt 出发) | Issue #110 spec 禁止 #100 单路 checkpoint 冒充 |

## 预期产物

- verdict: `verdicts/task404_issue110_stage2_component_audit_v2.md`
- SID file: `HG-Rec/dataset/Instruments/Instruments_t5_rqvae_task404.npy` (9922, 4)
- Stage 2 audit json: `verdicts/task404_stage2_audit.json` (SID unique + utilization + component contribution + gate/mixing 分布 + 3 组对照 logits diff)
- log: `logs/task404_stage2_<TS>.log`
- Stage 3 ckpt: N/A (本 task 只到 Gate 2)

## R18 4 维度对比 (Issue #110 vs 历史)

| 维度 | Issue #110 spec | 历史 task401 verdict | 复用? |
|------|----------------|---------------------|-------|
| D1 spec 摘录 | Stage 2 component/gate 审计 + 不重训 Stage 1 | task401 Stage 1 30 epoch early-stop PASS | 部分 (Stage 1 已 PASS, 复用 ckpt) |
| D2 实施核心 | component/gate 分布 + Sinkhorn 生成影响 + 原/均/shuffle 三组对照 | task401 仅 Stage 1 训练, 未做 Stage 2 component 审计 | 不同 (新增 Stage 2 审计维度) |
| D3 Gate 1 失败机制 | N/A (不跑 Stage 1) | USAGE-KILL @ 50 epoch → early-stop @ 30 | 不适用 |
| D4 引用文献 | arXiv:2307.04514 + ACE-HGNN DOI 10.1109/ICDM51629.2021.00021 | task401 引用 arXiv:2405.13979 | 不同 |

→ R18 判定: Issue #110 路径与 task401 不完全同, 必须做实验 (Sinkhorn 跑一次 + component 审计)
