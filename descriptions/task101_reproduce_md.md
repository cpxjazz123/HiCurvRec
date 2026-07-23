# Task #101 — REPRODUCE.md 复现包 + 环境验证脚本

> **任务目的**: 创建根目录 `REPRODUCE.md` 文件, 整合 Stage 1-4 完整复现指令 (verbatim from CLAUDE.md R1) + 数据集准备 + 期望产出 + 期望关键数字 (R@10=0.1020/0.1058 vs paper Table 2). 配套 scripts/task101_verify_env.py 验证 conda env + dataset + repo layout 三件套. 形成 paper §7 "All experiments reproducible with seed=42" 主张的**实证可复现** 入口.

> **完成日期**: (in progress)
> **状态**: 🟡 待启动

---

## 1. 背景

Task #98 paper.md + Task #99 paper.pdf + Task #100 paper submission 准备 都已完成. 但论文 §7 reproducibility claim ("All experiments reproducible with seed=42 on Amazon Musical\_Instruments 5-core leave-one-out") **没有可执行的 artifact** 支撑. 用户/审稿人拿到 paper.pdf 后, 没有任何快速指引告诉他在自己机器上如何跑出相同数字.

需要:
1. 根目录 `REPRODUCE.md` — 完整复现指引 (含 conda env / dataset / Stage 1-4 命令 / 期望数字 / 验证脚本)
2. `scripts/task101_verify_env.py` — Python verifier, 检查 env + dataset + repo layout (no fallback, R2 raise)
3. (可选) `scripts/download_amazon_musical_instruments.sh` — 数据集下载辅助

## 2. 实验设计 (writeup only)

### 2.1 文档结构 (REPRODUCE.md §1-§11)
1. **Hardware & Software** — L40S GPU, conda `grid_toys` env
2. **Dataset** — Amazon Musical\_Instruments 5-core, acquisition
3. **Stage 1** — sem_embeds_inference_flat, sentence-T5 768d
4. **Stage 2** — rqvae_train_flat + rkmeans_inference_flat, 3 layers + dedup digit append
5. **Stage 3** — tiger_train_flat, T5-small, num_hierarchies=4
6. **Stage 4** — tiger_inference_flat, beam search
7. **Expected Headline Numbers** — 6 个方法 R@5/R@10/R@20/NDCG@10 + 6 网格 curvature + 训练成本
8. **Verification Scripts** — scripts/ 表格
9. **Caveats** — 已知 Stage 1-4 陷阱
10. **File Layout** — products/task101_<stage>/...
11. **Citation & License** — snap-research/GRID 致谢

### 2.2 Verifier 脚本 (R2 no-fallback 严格执行)
- 三个 check: repo layout, packages (torch/lightning/transformers/hydra/omegaconf), dataset existence
- 任一 check 失败 → RuntimeError, exit 1
- 输出 `[OK] / [FAIL]` 行 + 最终汇总

### 2.3 (可选) Download 脚本
- `scripts/download_amazon_musical_instruments.sh`
- 命令: `wget https://datadryad.org/.../Musical_Instruments_5core.csv.gz -O data/amazon_data/musical_instruments/Musical_Instruments_5core.csv.gz` (若 dryad URL 失效, 则指向 HG-Rec 官方 dataset page)
- **R11.3 决策**: 不实施 optional download shell (避免 hard-code 失效 URL, 通过 REPRODUCE.md §2.2 说明手工下载路径)

## 3. 决策触发

| 条件 | 结果 | 决策 |
|------|------|------|
| REPRODUCE.md + verifier 落盘 + tasks task99-100-101 description 链条闭环 | ✅ 闭环 | 写 task101 verdict, §16 表格清空 |
| py_compile 失败 | ❌ 不闭环 | 修复 verifier 语法, 重试 |
| verifier 实际运行时 (full env) 失败 | ⚠️ 部分闭环 | 在 verdict 报告具体失败项, 但 verifier 自身在 failed env 下失败是预期的 |

## 4. 预算

| 阶段 | 估算时间 | GPU |
|------|---------|-----|
| 写 REPRODUCE.md | ~25 min | 0 |
| 写 verifier 脚本 | ~10 min | 0 |
| py_compile 验证 | ~30 sec | 0 |
| verifier 实际运行 (failed env) | ~5 sec | 0 |
| **总计** | **~40 min** | **0 GPU** |

## 5. 风险与缓解

**风险 1**: REPRODUCE.md 中 hardcode 命令路径在新机器上不对
  → **缓解**: 用 `${REPO}` 抽象路径占位符, 在 §1.3 给出 root 切换指引

**风险 2**: py_compile 通过但实际 verifier 误报
  → **缓解**: 三层 check 独立, 任一 raise 都明确指向修复路径

**风险 3**: 数据集 URL 失效
  → **缓解**: §2.2 同时给 dryad URL + HG-Rec 官方 dataset page, 用户任选一条

## 6. 产物清单

- `REPRODUCE.md` — 完整复现指引 (~ 280 行)
- `scripts/task101_verify_env.py` — 环境验证脚本 (~ 90 行)
- `descriptions/task101_reproduce_md.md` — 本任务描述
- `verdicts/task101_reproduce_md_result.md` — 闭环报告

## 7. 关联

- 前置: Task #98 paper.md + Task #99 paper.pdf + Task #100 submission 准备
- 后置: (无, paper reproducibility artifact 闭环, 后续可选 supplementary)

---

**核心交付**: REPRODUCE.md (Stage 1-4 完整复现指引) + scripts/task101_verify_env.py (env/dataset/repo 三层验证).
