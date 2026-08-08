# HAB 路线最终整合 verdict — 0.11 在现有框架下不可达

**最终结论**: HAB 路线 (Issue #64 v6b + #71 v71 + #135 v72/v73) 4 issue 全部 NO-GO. test R@10 上限 0.1038 (历史 v6b), 实际平均 0.0998. **0.11 目标在现有 HAB 框架下不可达, 必须架构性改动.**

---

## 4 Issue 全景

| Issue | 方案 | 关键改动 | Best Valid R@10 | Test R@10 | vs baseline 0.1024 | Verdict |
|-------|------|---------|-----------------|-----------|-------------------|---------|
| **#64 v6b** | U·V^T learnable (rank 16) | λ_max=0.20, LR ratio=100x | 0.1294 (ep65) | 0.1038 / 0.0988 (ep85) | +0.0014 / -0.0036 | NO-GO |
| **#71 v71** | residual + α=0.5 | Dbar_frozen + α·delta, α LR 10x | 0.1285 (ep50) | 0.1013 (ep71) / 0.0979 (ep105) | -0.0011 / -0.0045 | NO-GO |
| **#135 v72** | + valid_R10 ES=5 | ckpt 按 valid 保存, ES=5 | 0.1280 (ep35) | 0.1003 | -0.0021 | NO-GO |
| **#135 v73** | + valid_R10 ES=10 | ES=10 (给 valid 更多机会) | 0.1280 (ep35) | 0.0979 | -0.0045 | NO-GO |

---

## 失败机制统一根因

### 4 Issue 共享同一失败模式

1. **valid 触顶后必然崩**: 任何 learned bias 在 T5 训练中都会让 valid 偏离 baseline pattern
2. **valid peak 区间窄**: ep35-50 触顶 (0.1280-0.1294), 然后持续下降
3. **test 跟 valid 同步**: test 0.0979-0.1038, 永远在 baseline ±0.005 内
4. **valid/test ratio 失衡**: 1.250 (HAB) vs 1.237 (baseline), HAB 学到 valid pattern 但不 generalize 到 test

### HAB 上限就是 baseline ±0.005

| 实验 | Best Test R@10 | 偏离 baseline |
|------|----------------|---------------|
| v6b (某次) | 0.1038 | +0.0014 |
| v71 ep71 | 0.1013 | -0.0011 |
| v72 ep35 | 0.1003 | -0.0021 |
| v73 ep35 | 0.0979 | -0.0045 |
| 平均 | 0.1008 | -0.0016 |

**HAB 路线的"超 baseline"窗口**: 仅 v6b 某次 +0.0014. 其余都 FAIL.

---

## 0.11 目标不可达 — 距离证据

| 指标 | test R@10 | vs 0.11 |
|------|-----------|---------|
| **HAB 路线历史最佳 (v6b)** | **0.1038** | **-0.0062** |
| HAB 路线 4 issue 平均 | 0.0998 | -0.0102 |
| taskA hyp v2 (Stage1 per-item radius + Stage3 解冻 T5) | 0.1048 | -0.0052 |
| baseline T5-mini | 0.1024 | -0.0076 |
| DIGER 论文 (instruments) | 0.1121 | **+0.0021** (新模块) |
| DECOR 论文 (instruments) | 0.1157 | **+0.0057** (新模块) |

**核心证据**:
- HAB 路线最佳 +0.0014 vs baseline (历史偶然)
- taskA hyp v2 最佳 +0.0024 vs baseline (Stage1 + Stage3 联合)
- DIGER 论文 +0.0097 vs baseline (新模块 uncertainty head)
- **0.11 需要 +0.0076 vs baseline, 超 HAB 微调上限 5x**

---

## 现有框架约束 (R 规则 + SID 锁定)

| 约束 | 影响 |
|------|------|
| SID SHA256 锁定 `be9be8f8...` | 不能重新训练 Stage2 (R5 + user 指示) |
| Stage1 固化 | 不能加 per-item radius (taskA hyp v2 突破) |
| T5-mini 固定 | 不能改架构 |
| 4 阶段流水线 | 不能 joint opt |
| HAB 唯一可改 | Stage3 attention bias (已 4 issue 穷尽) |

**HAB 在现有框架下是唯一可动的杠杆**, 但 4 issue 已证明这个杠杆的上限是 baseline ±0.005.

---

## 0.11 唯一可行路径 (新架构改动)

| 路径 | 预期 test | 改动 | R18 依据 |
|------|-----------|------|---------|
| **DIGER uncertainty head** | ~0.11 | Stage3 加新模块 (uncertainty decay) | DIGER 论文报 +0.0097 |
| DECOR candidate bins (改良) | ~0.11 | 解决 Issue #70 self-reinforcing trap | 已 NO-GO, 需重新设计 |
| Stage1 per-item radius | ~0.105 | 重新训练 Stage1 (1.5h) | taskA hyp v2 突破方案 |
| Stage1 fine-tune | ? | 重新训练 Stage1 | 大工程 |
| **接受 baseline 0.1024** | 0.1024 | 无 | 务实 |

**推荐**: 接受 baseline 0.1024. 如果坚持 0.11, 走 DIGER 路线 (新模块, 不在 HAB 框架内).

---

## 关键 commit 历史

| commit | 描述 |
|--------|------|
| `fa75e99` | Issue #70 DECOR PromptFormer 移植 NO-GO |
| `1ee3567` | Issue #71 HAB 残差学习 (Dbar + α·delta) 修复 v6b 过拟合 |
| `a0aae12` | Issue #135 v72/v73 HAB valid_earlystop |

所有 commit 已 push 到 gitlab origin (`wlia0047/generec`).

## Issue #64 (HAB 母 issue)

- GitLab note_3653934069: Issue #71 v71 结果
- GitLab note_3654115562: Issue #135 v72/v73 结果
- GitLab note_3654118682: HAB 路线最终 4 issue 全景

---

## 结论

**HAB 路线整体关闭**. 4 issue (#64 v6b + #71 v71 + #135 v72 + #135 v73) 全部 NO-GO, 投入产出比归零. test R@10 上限锁在 0.1038, 0.11 目标在 HAB 框架内不可达.

**0.11 唯一可行路径**: DIGER uncertainty head (新模块, 论文级 +0.0097) — 这超出"现有 HAB 框架"边界, 应开新 issue.

**务实选择**: 接受 baseline 0.1024. HAB 已不是 hot research 方向.