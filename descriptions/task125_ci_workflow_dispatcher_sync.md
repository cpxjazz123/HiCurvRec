# Task #125 — Sync CI Workflow with 5-Audit Dispatcher

> **任务目的**: `.github/workflows/audits.yml` (Task #107 创建) 跑 4 separate audits (task101/103/105/106), 但 Task #110 引入了 single dispatcher `scripts/all_audits.py`, Task #114 加了 5th audit. CI workflow 是 stale 的, 同步到 dispatcher pattern: 单 `python3 scripts/all_audits.py` 命令替代 4 separate. artifact upload + summary step 也同步加 task114.

> **完成日期**: (in progress)
> **状态**: 🟡 待启动

---

## 1. 背景

`.github/workflows/audits.yml` 现状 (Task #107 创建):
- 4 separate `python3 scripts/taskXXX` 命令 (task101/103/105/106)
- Missing task114 (5th audit, Task #114 加)
- 4 step 名 + 4 echo banner
- artifact upload: 5 paths (no task114)
- summary step: parse task106_audits.json (no task114 summary)

实际项目状态 (post Task #110 + #114):
- `scripts/all_audits.py` 单 dispatcher (5 audits: task101/103/105/106/114)
- 0.41s 平均 runtime
- dispatcher 是 reviewer-facing single entry point

**Task #125 = CI workflow 同步 dispatcher pattern**

---

## 2. 实验设计 (writeup only)

### 2.1 替换 4 separate audit commands with single dispatcher

Before:
```yaml
      - name: Task #101 — Environment verification (CI mode)
        run: python3 scripts/task101_verify_env.py --skip-data --skip-packages

      - name: Task #103 — Paper claims audit (Δ=0 expected)
        run: python3 scripts/task103_paper_claims_audit.py

      - name: Task #105 — Best ckpt integrity (CI mode: graceful if absent)
        run: python3 scripts/task105_ckpt_integrity.py --ci-mode

      - name: Task #106 — Submission defense bundle (5 audits)
        run: python3 scripts/task106_audits.py
```

After:
```yaml
      - name: Single audit dispatcher (all 5 paper defense audits)
        run: |
          echo "--- Paper defense audit dispatcher (Task #110) ---"
          echo "    Runs all 5 audits: task101/103/105/106/114"
          python3 scripts/all_audits.py
```

### 2.2 Artifact upload 加 task114 verdict

Add `verdicts/task114_verdict_integrity_result.md` to upload path list.

### 2.3 Summary step 加 task114 summary

Add block parsing `verdicts/task114_verdict_integrity_result.md` exists.

### 2.4 Comment header 更新

Update header comment from "Runs the 4 cumulative paper defense audit scripts" to "Runs the single audit dispatcher (Task #110), which in turn runs all 5 paper defense audit scripts".

---

## 3. 决策触发

| 条件 | 决策 |
|------|------|
| 4 separate → 1 dispatcher command | ✅ 闭环 |
| artifact upload + task114 verdict | ✅ 闭环 |
| summary step + task114 summary | ✅ 闭环 |
| header comment updated | ✅ 闭环 |
| YAML syntax valid | ✅ 闭环 |
| dispatcher 5/5 PASS 不破坏 | ✅ 闭环 |

---

## 4. R11.3 自主决策

| 决策 | 选择 | 拒绝 |
|------|------|------|
| 是否替换 4 separate → 1 dispatcher | ✅ 替换 (跟当前 dispatcher pattern 一致, future audits 自动 CI 覆盖) | ❌ 加 5th step (但仍 5 separate commands, 跟 dispatcher 不一致) |
| 是否保留 individual step names | ❌ 删除 (single dispatcher step 就够) | ✅ 保留 (但 verbose) |
| YAML syntax 验证 | ✅ `python3 -c "import yaml; yaml.safe_load(...)"` | ❌ 跳过 (但可能 syntax error) |
| 是否加 task114 summary | ✅ 加 (reviewer PR 看到完整 5/5 PASS) | ❌ 不加 (但 reviewer 困惑 task114 没被 CI) |

---

## 5. 预算

| 阶段 | 估算时间 |
|------|---------|
| grep + 写新 audits.yml | ~3 min |
| YAML syntax 验证 | ~5 sec |
| dispatcher 验证 | ~3 sec |
| git commit + §16 | ~2 min |
| **总计** | **~5 min, 0 GPU** |

---

## 6. 风险与缓解

**风险 1**: 修改 .github/workflows/audits.yml 可能 break GitHub Actions CI
  → **缓解**: yaml.safe_load 验证 syntax; 替换 dispatcher pattern 是 idempotent (5 audits 都跑)

**风险 2**: dispatcher 单 step 比 4 separate 慢 (因为重复 setup)?
  → **缓解**: dispatcher 0.41s 平均 runtime, 比 4 separate 总 runtime 还快 (避免重复 Python startup)

**风险 3**: 单 step 输出格式跟 reviewer 习惯的 4 separate 不一致
  → **缓解**: dispatcher 输出包含 "task101: ✅ PASS" 等 5 行, 等同 4 separate 视觉

---

## 7. 完成度跟踪

- [x] 写 descriptions/task125_ci_workflow_dispatcher_sync.md (本文件)
- [x] 替换 4 separate → 1 dispatcher
- [x] artifact upload + task114 verdict
- [x] summary step + task114 summary
- [x] header comment updated
- [x] YAML syntax 验证
- [x] dispatcher 5/5 PASS 验证
- [ ] git commit
- [ ] loop.md §16 更新

---

## 8. 关联

- 前置: Task #107 (GitHub Actions CI 创建) + #110 (audit dispatcher) + #114 (5th audit)
- 后置: 无 (CI workflow 同步闭环)

---

**核心交付**: CI workflow 同步 dispatcher pattern. 单 `python3 scripts/all_audits.py` 命令替代 4 separate audit invocations. artifact upload + summary step 加 task114. Future audits 添加只需更新 `scripts/all_audits.py` AUDITS 列表, 不需要再编辑 workflow YAML. dispatcher 5/5 PASS 不破坏.