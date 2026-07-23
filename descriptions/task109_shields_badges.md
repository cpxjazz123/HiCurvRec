# Task #109 — shields.io Badges + README Visual Upgrade

> **任务目的**: 在 `README.md` 顶部加 6 个 shields.io badges, 反映 paper submission defense 状态. 让 reviewer / GitHub visitor 一目了然看到 (a) baselines 复现率 (b) abstract compliance (c) R12 ckpt 完整性 (d) audits 通过情况 (e) license (f) paper 长度.

> **完成日期**: (in progress)
> **状态**: 🟡 待启动

---

## 1. 背景

`README.md` 当前只有文字, 没有可视化指标. shields.io 是 GitHub 事实标准, 加 6 个 badges:
- 让 reviewer 一眼看到 (e.g., "12/20 reproduced", "197/200 word abstract")
- 与 CI / repo 状态联动
- 不依赖外部 linter, 数据源是 `task106_audits.json` + `task105_ckpt_integrity.md`

**Why now**: Task #106 已经产生 `verdicts/task106_audits.json` 数据源, shields.io badge 可以直接读这些数据.

---

## 2. 实验设计 (writeup only)

### 2.1 6 个 badges 清单

| Badge | 数据源 | shields.io URL |
|-------|--------|----------------|
| 1. baselines reproduced | 12 of 20 (`task106_audits.json` A3) | `https://img.shields.io/badge/baselines-12%2F20-brightgreen` |
| 2. abstract compliance | 197/200 (A2) | `https://img.shields.io/badge/abstract-197%2F200%E5%AD%97-green` |
| 3. R12 ckpt integrity | 21.06 MB (Task #105) | `https://img.shields.io/badge/R12%20ckpt-21.06%20MB-green` |
| 4. paper claims audit | 14/14 Δ=0 (Task #103 csv) | `https://img.shields.io/badge/paper%20claims-14%2F14%20%CE%94%3D0-brightgreen` |
| 5. license | MIT | `https://img.shields.io/badge/license-MIT-blue` |
| 6. paper size | 14 pages | `https://img.shields.io/badge/paper-14%20pages-blue` |

### 2.2 shields.io 限制

| 限制 | 影响 |
|------|------|
| shields.io URL 必须 encode `%2F` for `/` | ✅ 用 `%2F` 没问题 |
| Dynamic badges 需要 endpoint (shields.io/endpoint) | 静态 badges 显示硬编码数字, 但仍是可读 |
| 不能 embed Python script | Markdown 直接 embed shields.io `<img>` 即可 |

### 2.3 README.md 插入位置

`README.md` 第 1 行 (TL;DR 之前):

```markdown
# GeneRec — HG-Rec Reproduction on Amazon Musical_Instruments

<p align="left">
<a href="papers/paper.pdf"><img src="https://img.shields.io/badge/paper-14%20pages-blue" alt="paper"></a>
<a href="papers/SUBMISSION_DEFENSE.md"><img src="https://img.shields.io/badge/baselines-12%2F20-brightgreen" alt="baselines"></a>
<a href="papers/SUBMISSION_DEFENSE.md"><img src="https://img.shields.io/badge/paper%20claims-14%2F14%20%CE%94%3D0-brightgreen" alt="paper claims"></a>
<a href="papers/SUBMISSION_DEFENSE.md"><img src="https://img.shields.io/badge/abstract-197%2F200%E5%AD%97-green" alt="abstract"></a>
<a href="verdicts/task105_ckpt_integrity.md"><img src="https://img.shields.io/badge/R12%20ckpt-21.06%20MB-green" alt="R12 ckpt"></a>
<a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue" alt="license"></a>
</p>

## TL;DR
...
```

### 2.4 替代方案: dynamic endpoint

如果要让 badges 反映最新 audit 状态, 可用 shields.io endpoint 模式 (Python JSON 服务):
- 提供 `https://api.shields.io/endpoint?url=...` 解析 `task106_audits.json` 的字段
- 但需要 public URL (GitHub Pages / raw.githubusercontent)
- 当前阶段先做静态 badges, 后续可升级到 dynamic

---

## 3. 决策触发

| 条件 | 决策 |
|------|------|
| README.md 顶部插入 6 badges | ✅ 闭环 |
| 每个 badge 有 link 指向 evidence file | ✅ 闭环 (reviewer 可点击 verify) |
| shields.io URL 不含 unencoded chars | ✅ 闭环 |

---

## 4. 预算

| 阶段 | 估算时间 |
|------|---------|
| 编辑 README.md 顶部 | ~5 min |
| 验证 badges URL encode | ~2 min |
| py_compile 验证 (Rule 10) | n/a (markdown) |
| 写 description + verdict + commit | ~5 min |
| **总计** | **~15 min, 0 GPU** |

---

## 5. 风险与缓解

**风险 1**: shields.io 服务偶尔 downtime → badge 显示灰色
  → **缓解**: 这是 GitHub 事实标准, 自身 uptime 99%+; 不影响 markdown 显示

**风险 2**: README.md 已较长 (200+ lines), 加 6 行 badges 不破坏 layout
  → **缓解**: 用 `<p align="left">` 包 badges, 不换行

**风险 3**: Badge URL 中 `Δ` 字符编码不一致
  → **缓解**: 用 `%CE%94` (UTF-8 encoded capital delta) 或 `'Delta` (literal)

---

## 6. 完成度跟踪

- [x] 写 `descriptions/task109_shields_badges.md`
- [ ] 编辑 `README.md` 顶部插入 6 badges
- [ ] 验证 shields.io URL 不含 bad chars
- [ ] 写 `verdicts/task109_shields_badges_result.md`
- [ ] git commit
- [ ] §16 loop.md 更新 (Task #109 ✅ 已完成)

---

## 7. 关联

- 前置: Task #106 (audits JSON 数据源), Task #105 (ckpt size 数据源)
- 后置: 后续 Task #110+ (dynamic endpoint / GitHub Pages 静态 badge 生成器)

---

**核心交付**: README.md 顶部 6 个 shields.io badges, 数据来自 task106_audits.json + task105_ckpt_integrity.md. 一眼可读 paper submission defense 全貌.
