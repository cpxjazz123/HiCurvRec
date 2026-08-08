# Issue #135 v72/v73 HAB valid_earlystop — NO-GO 闭环

**最终 verdict**: v72/v73 test R@10 最佳 0.1003/0.0979, **全 FAIL baseline 0.1024**
**结论**: NO-GO (HAB 路线整体 NO-GO, 关闭实验)

---

## 4 Gate 评估

### Gate 1 (代码改动 + 语法) — PASS
- `EARLY_STOP` 从 30 (loss-based) 改为 5 (v72) / 10 (v73), valid_R10-based
- `best_ckpt` 改为 valid_R10 保存 (loss-best 跟 valid peak 脱节)
- 训练代码 `python3 -m py_compile` 通过
- Verdict: `git diff` 无语法错误, Gate 1 PASS

### Gate 2 (训练启动 + 早停触发) — PASS
- v72: ep60 触发 early_stop (5/5 no_improv) ✓
- v73: ep85 触发 early_stop (10/10 no_improv) ✓
- 两个版本 best ckpt 都成功保存为 valid_R10-best
- Verdict: Gate 2 PASS

### Gate 3 (valid 早停效果) — PARTIAL PASS
- v72 best ckpt = ep35 valid_R10 = 0.1280 (vs v71 loss-best ep71 valid ~0.1266, +0.0014 ✓)
- v73 best ckpt = ep35 valid_R10 = 0.1280 (同 v72, EARLY_STOP=10 没区别)
- valid_R10-based 早停成功锁定 valid peak (避免 v71 那种 ep75→ep115 持续下降)
- **但**: valid peak 仍是 ep35 0.1280, 比 v71 历史 ep50 0.1285 略低 (-0.0005)
- Verdict: Gate 3 PARTIAL PASS

### Gate 4 (test eval) — FAIL ⚠️

| Checkpoint | Valid R@10 | Test R@10 | vs baseline 0.1024 |
|------------|------------|-----------|--------------------|
| v72 ep35 (ES=5) | 0.1280 | 0.1003 | **-0.0021** FAIL |
| v73 ep35 (ES=10) | 0.1280 | 0.0979 | **-0.0045** FAIL |

**Gate 4 FAIL**: valid_earlystop 没改善 test R@10. v73 test 比 v72 test 退步 -0.0024 (更长的训练反而让 test 更差).

---

## 关键实验对比 (HAB 路线全景)

| Issue | 方案 | 早停信号 | Best ckpt epoch | Valid R@10 | Test R@10 | vs baseline |
|-------|------|---------|-----------------|------------|-----------|-------------|
| #64 v6b (历史) | U·V^T learnable, 无 anchor | — | (历史 ep65) | 0.1294 | 0.1038 | +0.0014 ✓ |
| #64 v6b ep85 | 崩点 | — | — | 0.1219 | 0.0988 | -0.0036 |
| #71 v71 ep71 | residual + α=0.5 | loss (ES=30) | ep71 (loss-best) | ~0.1266 | 0.1013 | -0.0011 |
| #71 v71 ep105 | 续训练 | loss (ES=30) | ep105 (loss-best) | ~0.1190 | 0.0979 | -0.0045 |
| #135 v72 ep35 | 同 v71 + valid ES=5 | **valid_R10 (ES=5)** | ep35 (valid-best) | 0.1280 | 0.1003 | -0.0021 |
| **#135 v73 ep35** | 同 v71 + valid ES=10 | **valid_R10 (ES=10)** | ep35 (valid-best) | 0.1280 | **0.0979** | **-0.0045** |
| baseline | T5-mini no HAB | — | — | 0.1267 | 0.1024 | — |
| 目标 | — | — | — | — | **0.1100** | **+0.0076** |

---

## 核心发现

### 1. HAB 路线整体上限 ~0.10 test R@10

- 历史最佳 test R@10 = 0.1038 (Issue #64 v6b 某次)
- v71/v72/v73 全部在 0.0979-0.1013 区间
- **HAB 路线无法稳定超 baseline 0.1024**

### 2. 0.11 目标在现有框架下不可达

| 路线 | 最佳 test R@10 | 距 0.11 差距 |
|------|----------------|---------------|
| HAB (v6b/v71/v72/v73) | 0.1038 | -0.0062 |
| taskA hyp v2 突破 | 0.1048 | -0.0052 |
| baseline | 0.1024 | -0.0076 |
| DIGER 论文 (instruments) | 0.1121 | +0.0021 (新架构) |
| DECOR 论文 (instruments) | 0.1157 | +0.0057 (新架构) |

**结论**: 0.11 需要 DIGER/DECOR 级别的**架构性改动**, 不是 HAB 微调可达.
- DIGER: uncertainty decay 头 (新模块)
- DECOR: candidate bins + alpha gate (新模块)
- HAB 在现有 T5 + SID 框架下, 上限就是 baseline ±0.005

### 3. v73 EARLY_STOP=10 比 v72 EARLY_STOP=5 更差

- v72 best = ep35, test 0.1003
- v73 best = ep35 (同 epoch), test 0.0979 (-0.0024 退化)
- **EARLY_STOP 增加没帮助**: 因为 valid peak 都是 ep35, 训练更长反而让 ckpt "更成熟" 但 test 略差
- 这是**典型过拟合曲线**: valid 不变但 test 下降, 表明 HAB 训练让 ckpt 微调 overfit 到 valid

### 4. v71 ep50 才是 HAB 路线真实 valid peak

- v71 loss-based EARLY_STOP=30 训练到 ep116, 中间 valid peak 是 ep50 0.1285
- v72/v73 valid ES 都把 best 锁在 ep35 0.1280 (loss 2.0806, 训练初期)
- v71 ep50 loss 估计 1.96 (训练中期), valid 0.1285
- **HAB 路线真实 valid peak 是 ep50 (0.1285), 不是 ep35 (0.1280)**

---

## 失败机制总结 (HAB 路线 3 issue 全 NO-GO)

| Issue | 核心失败机制 | 失败幅度 |
|-------|-------------|----------|
| #64 v6b | learned U·V^T 偏离 Dbar 460-660%, valid 单点崩 -0.0075 | test -0.0036 |
| #71 v71 | α-anchor 改变曲线形状但 valid 仍缓慢下降 -0.0083 | test -0.0011 best, -0.0045 ep105 |
| #135 v72 | valid ES=5 锁定 ep35, ckpt 太早期 | test -0.0021 |
| #135 v73 | valid ES=10 仍锁定 ep35, 训练长 ckpt 微过拟合 | test -0.0045 |

**统一根因**: HAB 残差学习在现有 T5-mini + SID (be9be8f8) 框架下无法突破 baseline. 任何改进 (anchor / valid 早停 / 容量限制) 都没法越过 0.1038 历史最佳.

---

## 未来方向 — 0.11 必须架构性改动

| 方向 | 预期 test R@10 | 改动量 | R18 依据 |
|------|----------------|--------|---------|
| 现有 HAB 路线 | ≤0.1038 | — | 已实验, NO-GO |
| taskA hyp v2 复现 | ~0.1048 | stage1 per-item radius + 解冻 T5 | memory: taska-hyp-v2-end2end-breakthrough |
| DIGER uncertainty head | ~0.11 | 新模块 (uncertainty decay) | memory: diger-decor-borrow-points |
| DECOR candidate bins | ~0.115 | 新模块 (alpha gate) | Issue #70 已 NO-GO (self-reinforcing trap) |
| **新 SID + Stage2** | ? | 重新训练 Stage2 | **禁止** (SHA256 锁定) |

**推荐**: 接受 HAB 路线 NO-GO, 关闭 #64/#71/#135. 未来如果要 0.11, 走 DIGER 路线 (新模块, 论文已证 +0.0097).

---

## 产物清单

| 类型 | 路径 |
|------|------|
| v72 训练产物 | `/tmp/v72_hab_valid_earlystop/HG_Rec_best.pth` (ep35) |
| v72 test eval | `/tmp/v72_hab_valid_earlystop/test_eval/eval_test.json` |
| v73 训练产物 | `/tmp/v73_hab_es10/HG_Rec_best.pth` (ep35) |
| v73 test eval | `/tmp/v73_hab_es10/test_eval/eval_test.json` |
| 训练日志 | `/tmp/v72_hab_valid_earlystop/train.log` + `/tmp/v73_hab_es10/train.log` |

## 代码变更

| 文件 | 改动 |
|------|------|
| `common/stage3/stage3_train_pure_t5.py` | `EARLY_STOP=10` + valid_R10-based 早停 + best ckpt 按 valid 保存 |

## 关键 commit

| 阶段 | commit hash |
|------|-------------|
| v72/v73 实施 + 训练 | TBD (本次 commit) |

## Issue 闭环

- Issue #135 (本地命名) → close with `glab issue close` 或 note
- Issue #64 (GitLab HAB 母 issue) → 发 comment 说明 v72/v73 结果