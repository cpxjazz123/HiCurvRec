# Task #107 — GitHub Actions CI for Paper Defense Audits

> **任务目的**: 把 Task #101/#103/#105/#106 的 4 个 audit 脚本接到 GitHub Actions CI, 形成 paper submission defense 的 **continuous regression protection**. 每次 PR/push 自动跑 4 个 audit, fail-fast 提交保护.

> **完成日期**: (in progress)
> **状态**: 🟡 待启动

---

## 1. 背景

仓库已经累计 4 个 audit 脚本:
- Task #101: `scripts/task101_verify_env.py` (环境 + 数据验证)
- Task #103: `scripts/task103_paper_claims_audit.py` (paper claim × verdict 14/14 Δ=0)
- Task #105: `scripts/task105_ckpt_integrity.py` (R12 ckpt 4 层审计)
- Task #106: `scripts/task106_audits.py` (5 项 submission defense)

但这些 audit 都是 **manual** 跑的 (提交 PR 之前手动执行). 如果 paper.md 被改 / verdict 被删 / ckpt 路径变了, 不会自动报警.

Task #107 = 加 `.github/workflows/audits.yml`, 让 CI 自动接管. 不增加新 audit, 把现有 audit 自动化.

**Why now**: Task #106 刚闭环 5/5 audits, 趁热把它们 pipe 到 CI.

---

## 2. 实验设计 (writeup only)

### 2.1 Workflow 文件结构

`/home/wlia0047/ar57/wenyu/GeneRec/.github/workflows/audits.yml`:

```yaml
name: Paper Integrity Audits

on:
  push:
    branches: [main, master]
  pull_request:
    branches: [main, master]
  workflow_dispatch:  # allow manual trigger

jobs:
  audits:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - name: Cache HuggingFace models
        uses: actions/cache@v4
        with:
          path: ~/.cache/huggingface
          key: ${{ runner.os }}-hf-${{ hashFiles('**/requirements.txt') }}
      - name: Install deps
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements.txt
      - name: Task #101 — Environment verification
        run: python3 scripts/task101_verify_env.py
      - name: Task #103 — Paper claims audit
        run: python3 scripts/task103_paper_claims_audit.py
      - name: Task #105 — Best ckpt integrity
        run: python3 scripts/task105_ckpt_integrity.py
        continue-on-error: true  # ckpt may be absent in fresh-clone
      - name: Task #106 — Submission defense bundle
        run: python3 scripts/task106_audits.py
      - name: Upload audit reports
        if: always()
        uses: actions/upload-artifact@v4
        with:
          name: audit-reports
          path: |
            verdicts/task103_paper_claims_audit.md
            verdicts/task103_paper_claims_audit.csv
            verdicts/task105_ckpt_integrity.md
            verdicts/task106_audits.md
            verdicts/task106_audits.json
```

### 2.2 关键决策

| 决策 | 理由 |
|------|------|
| 用 `ubuntu-latest` | 不要 GPU runner (audit 脚本都是 stdlib); 节省 CI 时间 + 成本 |
| Task #105 用 `continue-on-error: true` | 第一次 fresh clone 没有 ckpt, 应 fail gracefully, 不 block |
| 不跑 `task101_verify_env.py` 的 dataset check | 大数据集 (Musical_Instruments) 数百 MB, 不在 CI 里跑. task101 改成"stub mode"允许 CI skip |
| cache HF | 节省 T5 model download 时间 |
| upload artifact 留 verdicts/ 输出 | PR review 时 reviewer 可 download 完整 audit output |

### 2.3 配套文件

- `.github/workflows/audits.yml` (主 CI workflow)
- `.github/workflows/README.md` (workflow 解释, 选填)
- 修改 `scripts/task101_verify_env.py` 加 `--skip-data` flag (允许 CI 跳过数据完整性检查)
- 修改 `task105` 加 graceful empty-ckpt handling (already raises FileNotFoundError; need to handle)

---

## 3. 决策触发

| 条件 | 决策 |
|------|------|
| CI workflow YAML 合法 | ✅ 闭环 |
| 4 个 audit 步骤都能 exit 0 | ✅ 闭环 (有 ckpt 时) / 部分 (无 ckpt 时降级) |
| task101 skip-data flag 加入 | ⏳ 检查 |
| artifact upload 工作 | ⏳ GitHub Actions runner 才能验证 |

---

## 4. 预算

| 阶段 | 估算时间 |
|------|---------|
| 写 audits.yml workflow | ~10 min |
| task101 + task105 graceful skip patch | ~10 min |
| py_compile 验证 | 3 sec |
| 工作流语法 lint | ~2 min (用 `act` 或 GitHub CLI dry-run) |
| 写 description + verdict + commit | ~5 min |
| **总计** | **~30 min, 0 GPU** |

---

## 5. 风险与缓解

**风险 1**: GitHub Actions runner 网络拉 T5 model 慢 (~ 250 MB, 5-10 min)
  → **缓解**: actions/cache 缓存 HF model, 仅首次下载

**风险 2**: ckpt 路径依赖 Task #84 已有的产品物, fresh clone 无 ckpt 时 audit fail
  → **缓解**: task105 用 `continue-on-error: true`, CI log 显示 "ckpt not present in fresh-clone, skipping" 而不是 fail

**风险 3**: task101 verify_env 需要 dataset CSV (hundreds MB, 仓库不含)
  → **缓解**: 加 `--skip-data` flag, CI 用此 flag 跳过大数据集检查 (仍 verify repo layout + packages)

**风险 4**: 维护负担 (GitHub Actions YAML 升级, dependency 漂移)
  → **缓解**: 锁版本 (actions/checkout@v4, setup-python@v5), 不跟 latest

---

## 6. 完成度跟踪

- [x] 写 `descriptions/task107_github_actions_ci.md` (本文件)
- [ ] 写 `.github/workflows/audits.yml`
- [ ] patch task101 + task105 graceful skip
- [ ] yaml syntax lint (Python yaml.safe_load)
- [ ] 写 `verdicts/task107_github_actions_ci_result.md`
- [ ] git commit
- [ ] §16 loop.md 更新 (Task #107 ✅ 已完成)

---

## 7. 关联

- 前置: Task #101/#103/#105/#106 (4 audit 脚本)
- 后置: 后续 Task #108+ (CI 跑通后加 shields.io badges / nightly scheduled run)

---

**核心交付**: `.github/workflows/audits.yml` + task101/105 graceful skip patches. CI 自动跑 4 audit, 形成 paper submission defense 的持续保护层.
