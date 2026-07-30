# Cron Tick Runbook — task327 + task320 Arm C 完成路径

**日期**: 2026-07-30
**触发**: 每 ~5 min cron tick (`/loop 5m follow loop.md`)

## 当前活跃任务 (2026-07-30 11:13)

| 任务 | GPU | Stage | 状态 |
|------|-----|-------|------|
| **task327 (K=256 + Issue #30 synergy)** | GPU 1 | Stage 3 T5-mini 200 epoch | Ep50/200 RUNNING |
| **task320 Arm C (R-Drop)** | GPU 0 | Stage 3 T5-mini 200 epoch | Ep54/200 RUNNING |
| — | GPU 2/3 | FREE | — |

## 完成时 Runbook

### 阶段 1: 立即 Stage 4 评估

#### task327 完成时 (K=256 + Issue #30 synergy)
1. **确认 ckpt 存在**:
   ```bash
   ls -lt /home/wlia0047/ar57/wenyu/GeneRec/products/task327/t5mini_k256_issue30/Instruments/*/HG_Rec_best.pth
   ```
2. **复制 SID 到 dataset**:
   ```bash
   cp /home/wlia0047/ar57/wenyu/GeneRec/products/task327/Instruments_t5_rqvae_k256_issue30.npy \
      /home/wlia0047/arenyu/GeneRec/HG-Rec/dataset/Instruments/Instruments_t5_rqvae_k256_issue30.npy
   ```
   (路径应为 `wenyu/GeneRec/HG-Rec/dataset/Instruments/`, 注意 typo)
3. **Launch Stage 4 K=20/50/100 beam ablation**:
   ```bash
   bash /home/wlia0047/ar57/wenyu/GeneRec/scripts/task327_stage4_beam_eval.sh
   ```
4. **三个 beam size 串行**, 各 ~1 min, 总 ~3 min
5. **读出 R@10**:
   ```bash
   cat /home/wlia0047/ar57/wenyu/GeneRec/verdicts/task327_stage4_beam{20,50,100}_metrics.json
   ```

#### task320 Arm C 完成时 (R-Drop)
1. **确认 ckpt**:
   ```bash
   ls -lt /home/wlia0047/ar57/wenyu/GeneRec/products/task320/armC/Instruments/*/HG_Rec_best.pth
   ```
2. **Launch Stage 4 K=100 eval**:
   ```bash
   bash /home/wlia0047/ar57/wenyu/GeneRec/scripts/task320_armC_stage4_eval_beam100.sh 2
   ```
   (GPU 2 = free; 不需要 GPU 0 因为 GPU 1 给 task327)
3. **读出 R@10**:
   ```bash
   cat /home/wlia0047/ar57/wenyu/GeneRec/verdicts/task320_armC_beam100_metrics.json
   ```

### 阶段 2: 决策矩阵

#### task327 R@10 解读

| R@10 | 决策 | 后续 |
|------|------|------|
| **> 0.1053** | **GO ⭐⭐⭐⭐** | 跨方向 ceiling 突破, 新 anchor 锁定. Phase 2 启动 KG-enhanced SID / architecture layer |
| 0.1020 ~ 0.1053 | 中性 | 协同不放大, 锚定 K=256 vanilla 0.1053 不变 |
| ≤ 0.1020 | 退化 | NORTH STAR FULL NO-GO, Issue #26 owner decision 升级 |

#### task320 Arm C R@10 解读

| R@10 | 决策 |
|------|------|
| > 0.1020 | R-Drop 可能轻微正信号, 需 cross-verify (val/test gap known) |
| 0.0942 ~ 0.1020 | Stage 3 protocol 确认 NOT 杠杆 |
| < 0.0942 | 异常负信号, 检查 training collapse |

### 阶段 3: R8 强制清理

#### task327 verdict 落盘模板

如果 R@10 > 0.1053 → `verdicts/task327_k256_issue30_synergy_GO.md`
如果 R@10 ≤ 0.1053 → `verdicts/task327_k256_issue30_synergy_NOGO.md`

verdict 必含:
- 实测 R@10 (K=20/50/100 三个)
- val/test gap 分析
- 协同效应解读 (k=256 vanilla 0.1053 vs synergy 实际)
- 跨 Stage 1/2/3/4 协议 NO-GO 联立综合
- 关 issue / 更新 loop.md / R8 cleanup

#### task320 Arm C verdict 落盘

`verdicts/task320_armC_verdict.md` (Issue #38 第 5 Arm 闭环)

### 阶段 4: GitHub 联动

#### Issue #38 已有 verdict
- 不需要再发 GitHub 评论 (Issue #38 已 closed)
- 在 §16 表格删除 task320 行

#### Issue #39 已有 verdict
- 不需要再发 GitHub 评论
- 在 §16 表格删除 task324 行

#### task327 verdict → 通知 owner (新 comment 到 Issue #26 Conflict Report)
- 如果 task327 R@10 > 0.1053 → Issue #26 不再 need 等 owner, 启动架构层
- 如果 task327 R@10 ≤ 0.1053 → Issue #26 升级, 请 owner 决策
  ```bash
  gh issue comment 26 -b "..."
  ```

## R10 优先

| 优先级 | 任务 |
|--------|------|
| 1 | task327 Stage 3 完成 → Stage 4 K=20/50/100 |
| 2 | task320 Arm C Stage 4 K=100 |
| 3 | task327 Stage 4 结果 verdict 落盘 |
| 4 | task320 Arm C Stage 4 结果 verdict 落盘 |
| 5 | 跨方向 NORTH STAR 终极判据 R@10 > 0.1053? |
| 6 | Issue #26 owner 决策升级 (Task #34 D9 / 架构层 / 暂停) |

## R11.5 自主决策边界

✅ AI 可自主推进: 任务完成检测, Stage 4 launch, verdict 落盘, Issue 评论
❌ AI 不可自主推进: 修改 owner decision 选项, 删除 verdict, 修改 project baseline

## 关联

- verdicts/task320_issue38_5arm_nogo.md (Issue #38 4/5 Arms 闭环)
- verdicts/task324_issue39_5arm_nogo.md (Issue #39 双协议 NO-GO)
- verdicts/north_star_ceiling_status.md (跨 Stage 综合)
- descriptions/task327_k256_issue30_synergy.md (跨方向协同 决策阈值)
- loop.md (活动 §16 表格)

result: Cron tick runbook 提供 task327 + task320 Arm C 完成时 R10 优先级 + R11.5 边界 + 决策矩阵. 无新 R10 决策需求.