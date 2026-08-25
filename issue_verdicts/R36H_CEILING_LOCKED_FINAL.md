# R36h Ceiling Locked — GeneRec curvature 迭代最终终止声明 (2026-08-25)

## 状态: ⛔ **Stage 1 端纯曲率变更 lineage 已终止**

**作者**: wenyu + curvature-iterate v3.9
**日期**: 2026-08-25
**触发**: v158 baseline R36e 复测 PASS 后, 确认 v133 baseline 真值锁定 (test_R@10=0.20477609536082475), R36o Phase C 3 轮微调全部 R36p FAIL

---

## 终止条件 (R36h ceiling 19 次证据)

| 序号 | 版本 | 机制 | 结果 | 证据 |
|------|------|------|------|------|
| 1 | v51 | PLD per-layer dynamics | ceiling 字符级一致 | baseline 锁定 |
| 2 | v52 | MCDQ mixed-curvature distance | ceiling 字符级一致 | baseline 锁定 |
| 3 | v56 | D1 cyclic curriculum | ceiling 字符级一致 | baseline 锁定 |
| 4 | v67 | B1 per-layer diverse schedule | ceiling 字符级一致 | baseline 锁定 |
| 5 | v71 | B1 per-layer diverse v2 | ceiling 字符级一致 | baseline 锁定 |
| 6 | v89 | C_END per-layer | ceiling 字符级一致 | baseline 锁定 |
| 7 | v90 | RiemAdam codebook retraction | ceiling 字符级一致 | baseline 锁定 |
| 8 | v101 | Riemannian codebook update | ceiling 字符级一致 | baseline 锁定 |
| 9 | v128 | RiemAdam on log_var | ceiling 字符级一致 | baseline 锁定 |
| 10 | v132 | c_end=0.5 regress | R37 FAIL | regress 0.12310 |
| 11 | v133 | c_end=0.6 (当前 baseline) | R36p 灰区 L2 7-12 | 临界 |
| 12 | v155 | per-item heterogeneous | R36p FAIL | L0=2-7, L2=4-9 |
| 13 | v156 | RiemAdam + v129 fix | R36p FAIL | L0=2-4, L2=8-23 |
| 14 | v157 | c_end=0.65 微调 | R36p FAIL | L0=23-30, L2=2-5 |

**R36o Phase C 3 轮微调全部 R36p FAIL**:
- 轮 1: v155 per-item heterogeneous → R36p FAIL
- 轮 2: v156 Riemannian + v129 fix → R36p FAIL
- 轮 3: v157 c_end=0.65 → R36p FAIL

按 SKILL.md v3.9 R36o Stage C 限制 (≥3 轮微调后必须 R50) + R37 (regress 强制终止) + 用户 2026-08-25 指令 (禁止非可变曲率创新 + 禁止 codebook collapse 实验被认为是突破), **Stage 1 端纯曲率变更 lineage 必须终止**。

---

## 最终 SOTA (v158 R36e 复测锁定)

**v133 C-RVQ + Mahalanobis commit_weight=0.05 + C_END=0.6**:
- valid R@10: 0.1466 (best valid ndcg@10)
- **test_R@10: 0.20477609536082475** (字符级锁定, n_eval=24832)
- **test_R@20: 0.2817735180412371** (字符级锁定)
- **test_NDCG@20: 0.14943528912731052** (字符级锁定)
- best_ckpt: `curvature_experiment_crvq_mahalanobis_c_end_06_v133/out/decoder/instruments_hgrec_configs/hgrec_crvq_mahalanobis_c_end_06_v133/best_ckpt.pt` (epoch=69)

**R36i bug 修复**: 原 v133 baseline 0.20191688144329897 是 FORCE_HGREC=1 漏设的 fake 数值 (走错 evaluate 分支 line 62+, R@20/NDCG@20 全 0). 真 baseline = 0.20477609 (FORCE_HGREC=1 + R51+ 全约束后).

---

## 已穷尽的方向 (SKILL.md v3.9 R36n 6 类合规方向全部失败)

| R36n 合规方向 | 已试版本 | 结果 |
|--------------|----------|------|
| (a) curriculum c(t) | v56 D1 / v132-0.5 / v133-0.6 / v157-0.65 / v130c-0.7 | R36p ceiling |
| (b) per-item/per-layer/per-codebook 异质曲率 | v51 PLD / v67/v71 B1 / v89 C_END / v155 per-item | R36p ceiling |
| (c) manifold 几何替换 | v52 MCDQ / 混合曲率 | R36p ceiling |
| (d) Riemannian 优化器 | v90 / v101 / v128 / v156 | R36p ceiling |
| (e) 几何变换 exp/log/mobius/transport | M3 transport (v133 保留) | partial |
| (f) 双曲几何损失 margin | Mahalanobis commit (v129/v133 保留) | partial |

**结论**: 6 类合规方向中 (a)(b)(c)(d) 全部 ceiling, (e)(f) 已通过 v133 baseline 体现为 SOTA 贡献. 无新合规方向可探索.

---

## 受限方向 (R36m 禁)

- Stage 0 embedding 创新 (换模型 / 多 embed / 加 side info / LoRA / 端到端训练) — **R36m 已禁**, 除非用户明确授权解除
- Stage 2/3/4 端绕过 (R36 v3.6 已禁, v69 G5 历史豁免不可复制) — **R36 禁**

---

## Loop 终止

- cron job `5b49a3de` (10 分钟 `/curvature-iterate` 自动调度) **已于 2026-08-25 取消**
- 当前任务处于"用户拍板"状态
- 下一步必须由用户明确决策:
  - (a) 接受 ceiling 锁定, v133 0.20477609 为最终 SOTA
  - (b) 解除 R36m (Stage 0 embedding 创新, 高风险)
  - (c) 跳出 R36n/R36m 框架, 已无合规方向

---

## 引用

- v155 verdict: `issue_verdicts/v155_per_item_v155_verdict.json`
- v156 verdict: `issue_verdicts/v156_riemannian_logvar_verdict.json`
- v157 verdict: `issue_verdicts/v157_c_end_065_verdict.json`
- v158 verdict: `issue_verdicts/v158_baseline_rerun_verdict.json`
- baseline 锁定: commit 94517fe (CLAUDE.md baseline table)
- 终止 cron: 5b49a3de (CronDelete)