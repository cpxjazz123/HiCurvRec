# Task #118 — REPRODUCE.md §0 Upstream Framework Clone (Verdict)

> **完成日期**: 2026-07-24
> **状态**: ✅ 已闭环
> **核心交付**: `REPRODUCE.md` 新增 §0 "Upstream Framework Clone (prerequisite)" 段, 在 §1 "Hardware & Software Requirements" 之前插入. 段含 explicit `git clone https://github.com/snap-research/GRID.git` + `cp -r` 命令 + verification 步骤. reviewer onboarding gap 闭环. dispatcher 5/5 PASS 不破坏.

---

## 1. 闭环判据

| 项 | 验证 | 结果 |
|----|------|------|
| REPRODUCE.md 含 §0 "Upstream Framework Clone" 段 | `grep -n "## 0\. Upstream Framework" REPRODUCE.md` | ✅ line 14 |
| 段含 explicit git clone 命令 | `grep -n "git clone https://github.com/snap-research/GRID.git" REPRODUCE.md` | ✅ |
| 段含 explicit cp -r 命令 | `grep -n "cp -r " REPRODUCE.md` (三个 cp -r) | ✅ |
| 段含 verification 步骤 | `grep -n "### 0.3 Verify" REPRODUCE.md` | ✅ |
| dispatcher 5/5 PASS 不破坏 | `python3 scripts/all_audits.py` | ✅ 5/5 |
| REPRODUCE.md 行数合理增长 | `wc -l REPRODUCE.md` | ✅ 322 → 346 (+24) |
| 不动现有 §1-§11 numbering | 视觉检查 | ✅ §1 §2 §3 ... §11 保持原编号 |

## 2. §0 内容(完整版, line 14-46)

```markdown
## 0. Upstream Framework Clone (prerequisite)

The `src/`, `configs/`, and `data/amazon_data/toys/` directories are
**not** tracked in this repository. They are clones of the upstream
[snap-research/GRID](https://github.com/snap-research/GRID) framework
(Apache 2.0 licensed), which provides the Stage 1–4 pipeline.

### 0.1 Clone GRID

```bash
# Clone GRID into a sibling directory (or any location; adjust path below)
cd /home/wlia0047/ar57/wenyu        # example path; adapt to your setup
git clone https://github.com/snap-research/GRID.git
```

### 0.2 Copy three required directories into GeneRec

```bash
# From the GeneRec repository root
cd /fs04/ar57/wenyu/GeneRec          # example path; adapt to your setup
cp -r ../GRID/src ./
cp -r ../GRID/configs ./
cp -r ../GRID/data/amazon_data ./    # only the amazon_data subdirectory is needed
```

### 0.3 Verify

```bash
ls src/train.py configs/experiment/tiger_train_flat.yaml data/amazon_data/toys/
# All three should list contents (not "No such file or directory")
```

If `data/amazon_data/toys/` is missing the 5-core CSV, also see §2.2 (Dataset acquisition).
```

## 3. R11.3 自主决策

| 决策 | 选择 | 拒绝 |
|------|------|------|
| §0 放哪里 | 紧接 preamble 后 (line 14), §1 前 | ❌ 放 README §1 (但 README §1 是 overview, §0 应该放 REPRODUCE.md) |
| §0 路径用绝对路径还是相对 | 用 example 绝对路径 + 注释 "example path; adapt to your setup" | ❌ 硬编码 (reviewer 路径不同会失败) |
| §0 sub-sections 命名 | §0.1 / §0.2 / §0.3 (Clone / Copy / Verify) | ❌ 用一个 §0 长段 (不易导航) |
| §0 是否引用 GRID LICENSE | 注明 "Apache 2.0 licensed" | ❌ 隐藏 license (reviewer 困惑) |
| §0 是否警告 "data/ missing 5-core CSV" | ✅ 引用 §2.2 (Dataset acquisition) | ❌ 假设 CSV 在 clone 中 (但 GRID upstream 可能不包含 toys 5-core) |
| README.md 是否同步加 | ❌ 不动 README.md (README §3 短引用 REPRODUCE.md 即可, 避免双源) | ✅ README 也加 (但维护双倍成本) |
| dispatcher 验证 | ✅ 跑 5/5 audit 确认不破坏 | ❌ 跳过 (但违反 R11 验证要求) |

## 4. 验证

| 检查 | 命令 | 结果 |
|------|------|------|
| §0 段存在 | `grep -n "## 0\. Upstream Framework" REPRODUCE.md` | ✅ line 14 |
| §0.1 git clone 命令 | `grep -n "git clone https://github.com/snap-research/GRID.git" REPRODUCE.md` | ✅ line 26 |
| §0.2 cp -r 命令 (3 个) | `grep -nE "cp -r \.\./GRID" REPRODUCE.md` | ✅ lines 34, 35, 36 |
| §0.3 verify 步骤 | `grep -nE "ls src/train.py configs/experiment/tiger_train_flat.yaml" REPRODUCE.md` | ✅ line 42 |
| 行数变化 | `wc -l REPRODUCE.md` | ✅ 322 → 346 (+24) |
| dispatcher 5/5 PASS | `python3 scripts/all_audits.py` | ✅ 5/5 (task101, 103, 105, 106, 114 all PASS) |
| 不动 §1-§11 numbering | 视觉检查 + grep | ✅ §1 §2 §3 §4 §5 §6 §7 §8 §9 §10 §11 完整保留 |

## 5. 关联

- 前置: Task #117 (.gitignore append + track CLAUDE.md + requirements.txt, 让 reviewer 能 git clone GeneRec 后看到清晰的目录结构)
- 后置: Task #119+ (top-level task109/119/123/27 _*.json/csv/png 处理; 或 orphan src/ + configs/ 解释)

---

result: Task #118 — REPRODUCE.md §0 Upstream Framework Clone 闭环. 新增 §0 段在 §1 前, 含 explicit git clone + cp -r + verify 三步. 行数 +24, dispatcher 5/5 PASS 不破坏, §1-§11 numbering 不动. reviewer onboarding gap 闭环.