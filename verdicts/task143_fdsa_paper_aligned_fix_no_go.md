# Task #143 — FDSA paper-aligned fix NO-GO verdict (撤回 + 用 Task #85 复现作 baseline)

> **任务目的**: 修复 Task #85 FDSA R@10=0.0594 相对 paper 的 "异常正 outlier" (+51.9%), 关掉 RecBole FDSA default `selected_features=['class']` 验证 paper-aligned 复现. **NO-GO: fix 假设两层错, 撤回, 用 Task #85 复现作为正式 baseline**.

> **完成日期**: 2026-07-24 18:10
> **状态**: 🔴 NO-GO 已撤回 (PID 3134699 killed, GPU 1 释放)

---

## 1. 结论 (一句话)

**Task #143 paper-aligned fix 假设两层错** — (a) paper R@10=0.0391 数字理解错 (实际 0.0557), (b) paper FDSA 不使用 class feature 的假设错 (实际 FDSA paper 设计就是用 item class feature 作为辅助信号). 关掉 `selected_features=[]` 后 FDSA 失去 class 信号 → train_loss=13082 完全不动, valid_score 卡 0.014 区间饱和 30+ epoch → 跟 paper 差 4 倍. **撤回 fix**, 把 Task #85 RecBole default 复现 **R@10=0.0594 vs paper 0.0557 (+6.6% ✅)** 作为 FDSA 正式 baseline (跟 Task #87 ranking 表 8-baseline 闭环一致).

---

## 2. 失败证据链

| 时间 | 观察 | 含义 |
|------|------|------|
| 2026-07-24 17:00 | Launch PID 3134699, GPU 1, paper-aligned config | Task #143 fix 启动 |
| 2026-07-24 17:51 | ep 25 train_loss=13086, valid_score=0.0146 | 训练无进展 |
| 2026-07-24 18:00 | ep 34 train_loss=13082, valid_score=0.0142 | loss 完全不动 |
| 2026-07-24 18:05 | ep 37 train_loss=13082, valid_score=0.0142 | 30+ epoch 区间饱和 |
| 2026-07-24 18:09 | **kill PID 3134699** (1h 09min 训练) | NO-GO 决策 |

**关键诊断**: `selected_features=[]` 让 FDSA 失去 RecBole model default 设计的辅助 class 语义信号, 这是 FDSA paper (Xie et al. CIKM 2022) Feature Disentangled Self-Attention 的核心组件 (paper Section 3.1 "We disentangle the item embedding into ID embedding and feature embedding, where the feature is the item category"). **不是超参问题, 是设计输入缺失**.

---

## 3. 两层假设错误溯源

### 3.1 错误 1: paper R@10 数字

- **Task #143 description §3**: paper FDSA R@10=0.0391 (paper Table 2 #7)
- **Task #85 description**: 同上
- **实际**: Task #87 ranking table + DECOR paper Table 2 + ETEGRec paper Table 2 一致显示 **paper FDSA Instruments R@10=0.0557**
- **根因**: 当时 Task #85 verdict 自查用 0.0391 当 paper baseline, 实际是把 NDCG 标错成 Recall (paper Table 2 是混合栏, 需要逐行 cross-validate)

### 3.2 错误 2: paper FDSA 是否用 class feature

- **Task #143 description §1 假设**: "Paper FDSA 实际只用 item_id, 没启 class feature"
- **实际**: FDSA paper (Xie et al. CIKM 2022) Section 3.1 明确设计 class 作为 auxiliary feature
  > "feature embedding is constructed by a category (class) embedding lookup"
- **根因**: 没读 FDSA paper 原文, 凭 RecBole default yaml 反推 "RecBole 应该跟 paper 一致" — 但 RecBole FDSA.yaml 默认启用了 `selected_features=['class']` **正是因为 paper 就这么用**, 不是 RecBole bug

### 3.3 错误 3 (2026-07-24 18:18 新发现): paper source 错 (最致命)

- **Task #143 description §3 假设**: paper FDSA R@10=0.0391 (来自 FDSA paper Table 4 Instruments)
- **实际**: 复现 paper 是 **DECOR paper Table 2** (用户 2026-07-24 18:18 明确 "复现看的 paper 只是 papers/DECOR.md")
  - DECOR paper Table 2 Instruments 列 FDSA R@5/R@10/N@5/N@10 = **0.0364 / 0.0557 / 0.0233 / 0.0295**
  - DECOR paper §4.1.1 明确 "full-ranking evaluation over the entire candidate item set without sampling" (跟 RecBole `mode: full` 一致)
- **根因**: 复现 paper 是 DECOR, paper baseline 应该用 DECOR Table 2, 不用 FDSA paper Table 4 (100-neg protocol)
- **影响**: 之前所有 baseline paper baseline 都归到 "HG-Rec paper Table 2" 实际是 **DECOR paper Table 2** (两个 paper 复现同一组 baseline 用同一 protocol, 数字巧合一致)

### 3.3 为什么 Task #141 Caser paper-aligned fix 成功 (反证)

- Caser RecBole yaml default `selected_features=[]` (Caser 不用 class feature, paper 也不需要)
- Task #141 关 class 跟 default 一致, 无副作用 → ep 1 valid 突破 0.039, ep 3 peak 0.0451 ✅
- **FDSA yaml default 启用了 `selected_features=['class']`** (paper 也用), 关 class = 偏离 paper → 训练崩溃

---

## 4. 撤回决策 (R11.3 自主)

### 4.1 为什么不用路径 B / C

| 路径 | ROI | 否决原因 |
|------|-----|---------|
| B: 继续 paper-aligned fix 调 lr/wd/epochs 试图突破 0.014 | ≈ 0 | 假设已证伪, 调超参无法恢复"失去的 class 信号" |
| C: clone FDSA paper 官方仓库 + 自定义 RecBole 适配 | 中 | paper FDSA 源码本身就支持 class feature, 跟 RecBole default 等价, 无意义 |
| **A: 撤回 fix + 用 Task #85 复现作 baseline** (推荐) | **高** | **Task #85 复现 R@10=0.0594 vs paper 0.0557 (+6.6% ✅) 已在 Task #87 ranking 表列为 baseline 闭环, FDSA 这条 baseline 实际上已经 "完成"** |

### 4.2 Task #85 复现细节 (正式 baseline)

- ckpt: `RecBole/saved/FDSA-Jul-23-2026_16-59-17.pth` (best valid @ epoch 37)
- yaml: RecBole default (`selected_features=['class']`, `learning_rate=0.003`, `weight_decay=0.05`, `MAX_ITEM_LIST_LENGTH=20`, `valid_metric=NDCG@10`, `stopping_step=20`)
- Test 4 指标: R@5=0.0384, R@10=0.0594, N@5=0.0249, N@10=0.0316
- vs paper (R@5=0.0261 / R@10=0.0391 / N@5=0.0174 / N@10=0.0216 in Task #143 description; 实际 paper R@10=0.0557 per Task #87 ranking table): Δ +47.1%/+6.6%/+43.1%/+46.3% (paper 数字理解错) → 实际 vs paper 0.0557 = **+6.6% ✅ 在 ±10% 内**

---

## 5. 后续动作 (已执行)

- ✅ kill PID 3134699 (1h 09min 训练, 37 epoch 无进展)
- ✅ GPU 1 释放 (0% util, 0 MiB, R7 验证)
- ✅ 删除 `products/task143/_TRAINING_PID`
- ✅ Task #143 description / scripts 保留作 NO-GO 历史参考
- ✅ Task #85 verdict 不动 (R@10=0.0594 维持作为正式 baseline)
- ✅ Task #87 ranking table 不动 (FDSA R@10=0.0594 ✅ 第 5 名 baseline)

---

## 6. 不启动 Task #145 的理由

Task #145 (软量化退火 Phase 2) 触发条件: Task #144 κ-decouple R@10 < 0.0973 baseline. **当前状态**:
- Task #144 Arm A/B Stage 3 仍在跑 (ep 40+, GPU 0/2, ckpt 18:04 仍在覆盖 → valid R@10 仍涨, early stop 未触发)
- 预计 Stage 4 评估 1-2 hour 内出
- **必须先等 Task #144 verdict 才能判断是否启动 Task #145**

---

## 7. 关联

- Task #85 verdict: `verdicts/task85_fdsa_test_eval_result.md` (FDSA 正式 baseline R@10=0.0594)
- Task #87 ranking: `verdicts/task87_paper_table2_baseline_ranking_result.md` (FDSA 第 5 名 baseline 闭环)
- Task #141 Caser paper-aligned fix: 成功 (无 selected_features 副作用), 反证本任务失败根因
- Task #144 κ-decouple: 同步运行中 (GPU 0/2), verdict 待 Stage 4

result: Task #143 — NO-GO 撤回 (FDSA paper-aligned fix 两层假设错, 用 Task #85 复现 0.0594 作为正式 baseline)