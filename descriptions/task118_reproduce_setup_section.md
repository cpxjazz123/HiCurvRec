# Task #118 — Add src/configs/data Setup Instructions to REPRODUCE.md

> **任务目的**: 在 `REPRODUCE.md` 加 §0 "Upstream Framework Clone" 段, 给 reviewer 明确 `git clone https://github.com/snap-research/GRID.git` 步骤把 `src/` + `configs/` + `data/` 放到正确位置. 当前 README §1 + REPRODUCE.md §0 仅 mention "snap-research/GRID" 但无 explicit clone command, reviewer onboarding gap.

> **完成日期**: (in progress)
> **状态**: 🟡 待启动

---

## 1. 背景

CLAUDE.md 说 "`src/` + `data/` | 从 `snap-research/GRID` clone 的框架源码和数据 | ❌ 只读". 但 REPRODUCE.md 和 README 没有 explicit `git clone` 命令告诉 reviewer 如何获取这些 upstream framework 目录.

Result: reviewer `git clone GeneRec` 后看到 162 untracked files 包含 `src/`, `configs/`, `data/` — 不知道下一步该怎么办. 即使他们 clone 了 snap-research/GRID, 也不清楚是把 GRID 的整个目录复制过来, 还是只复制 src/ + configs/ + data/ 这三个子目录.

**Task #118 = explicit setup instructions**:
- REPRODUCE.md 加 §0 "Upstream Framework Clone" 在 §1 "Hardware" 之前
- 给明确 `git clone` + `cp -r` 命令, 精确到子目录
- 给 verification 步骤 (检查 src/ + configs/ + data/ 都存在)

---

## 2. 实验设计 (writeup only)

### 2.1 新增 §0 段 (插在 §1 之前)

```markdown
## 0. Upstream Framework Clone (prerequisite)

The `src/`, `configs/`, and `data/amazon_data/toys/` directories are
**not** tracked in this repository. They are clones of the upstream
[snap-research/GRID](https://github.com/snap-research/GRID) framework
(Apache 2.0 licensed), which provides the Stage 1–4 pipeline.

### 0.1 Clone GRID

```bash
# Clone GRID into a sibling directory
cd /home/wlia0047/ar57/wenyu
git clone https://github.com/snap-research/GRID.git
```

### 0.2 Copy three required directories

```bash
cd /fs04/ar57/wenyu/GeneRec
cp -r ../../GRID/src ./
cp -r ../../GRID/configs ./
cp -r ../../GRID/data/amazon_data ./
```

### 0.3 Verify

```bash
ls src/train.py configs/experiment/tiger_train_flat.yaml data/amazon_data/toys/
# All three should list contents (not "No such file or directory")
```

If `data/amazon_data/toys/` is missing the 5-core CSV, also see §2.2 (Dataset acquisition).
```

### 2.2 验证

```bash
# 1. REPRODUCE.md §0 段存在
grep -n "## 0\\. Upstream Framework" REPRODUCE.md

# 2. dispatcher 5/5 PASS (不破坏)
python3 scripts/all_audits.py

# 3. REPRODUCE.md 行数增加合理 (~+25 行)
wc -l REPRODUCE.md
```

---

## 3. 决策触发

| 条件 | 决策 |
|------|------|
| REPRODUCE.md 含 §0 "Upstream Framework Clone" 段 | ✅ 闭环 |
| 段含 explicit git clone 命令 | ✅ 闭环 |
| 段含 explicit cp -r 命令 | ✅ 闭环 |
| 段含 verification 步骤 | ✅ 闭环 |
| dispatcher 5/5 PASS 不破坏 | ✅ 闭环 |
| REPRODUCE.md 行数 +25 (合理范围) | ✅ 闭环 |

---

## 4. 预算

| 阶段 | 估算时间 |
|------|---------|
| grep 现状 + 起草 §0 内容 | ~2 min |
| Edit REPRODUCE.md 插 §0 | ~30 sec |
| 验证 + dispatcher | ~3 sec |
| git commit + §16 | ~2 min |
| **总计** | **~5 min, 0 GPU** |

---

## 5. 风险与缓解

**风险 1**: §0 段路径硬编码 (/home/wlia0047/ar57/wenyu/), reviewer 路径不同
  → **缓解**: 用相对路径 (../GRID/), reviewer 可自行 adapt. 或注释清楚是 example

**风险 2**: GRID upstream 改名/移动, git clone URL 失效
  → **缓解**: 引用官方 URL, 不假设 pinned commit. 如未来失效, reviewer 看 issue tracker

**风险 3**: 插入 §0 破坏其他 numbering (后续 §1 §2 §3 仍正确?)
  → **缓解**: §0 是 new section, 不动后续 §1/§2/§3 numbering. 后续若有 §0.1 §0.2 §0.3 sub-sections, 其他章节不受影响

**风险 4**: §0 段与 README §3 "Reproducing the Results" 内容重复
  → **缓解**: REPRODUCE.md §0 详细 (clone + copy + verify), README §3 简短 (引用 REPRODUCE.md). 不重复

---

## 6. 完成度跟踪

- [x] 写 `descriptions/task118_reproduce_setup_section.md` (本文件)
- [ ] Edit REPRODUCE.md 加 §0 段
- [ ] 验证 + dispatcher
- [ ] git commit
- [ ] loop.md §16 更新

---

## 7. 关联

- 前置: Task #117 (.gitignore + CLAUDE.md + requirements.txt tracked)
- 后置: Task #119+ (top-level task109/119/123/27 _*.json/csv/png 处理)

---

**核心交付**: REPRODUCE.md 加 §0 "Upstream Framework Clone" 段, reviewer onboarding gap 闭环.