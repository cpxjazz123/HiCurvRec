# Task #134 — CLAUDE.md R9-Enforce + audit 脚本 verdict

> **完成日期**: 2026-07-19
> **状态**: ✅ R9-Enforce 规则落地 + audit 脚本通过

---

## 1. 任务目标

解决 #127-#130 跳号事件(2026-07-19)的根因 — **在 CLAUDE.md 加 R9-Enforce 规则 + 提供 audit 脚本**,确保任务编号连续性强制执行,防止再次出现编号空洞。

## 2. 完成内容

### 2.1 CLAUDE.md 新增 R9-Enforce 子规则

**位置**: `/home/wlia0047/ar57/wenyu/GeneRec/CLAUDE.md`,R9 之后新增 "R9-Enforce" 节

**三层强制**:

- **Layer 1 (pre-creation check)**: 创建新任务文件前**必须**先跑命令获取 `NEXT_TASK_ID = MAX_ID + 1`;若发现 descriptions/ 有空洞,严禁写入新任务,必须先用 renumbering 脚本填空
- **Layer 2 (post-creation audit)**: 创建 description 后立即跑 audit 验证编号连续
- **Layer 3 (loop tick periodic audit)**: 每个 loop tick 第一步跑 `bash scripts/audit_r9_compliance.sh`,FAIL 则强制先修复

**白名单 + 历史事故记录表**: 列出 R9-Enforce 例外情况 + 2026-07-19 #127-#130 跳号事故作为反面教材。

### 2.2 audit 脚本

**位置**: `/home/wlia0047/ar57/wenyu/GeneRec/scripts/audit_r9_compliance.sh`

**3 项检查**:

| 检查 | 范围 | 严格度 |
|------|------|--------|
| [1] descriptions/ 任务编号连续 | descriptions/ task IDs 1..MAX_ID 无空洞 | **强制** (FAIL 阻塞任务) |
| [2] verdicts/ 任务编号连续 | verdicts/ task IDs 1..MAX_ID 无空洞 | warning only (允许历史空洞) |
| [3] 残余 task<old_num> 引用 | 检查 task104-117 + task127-130 (RENUMBERED_NUMBERS) 范围 | **强制** (排除自指 + renumber 脚本自身) |

**关键技术点**:
- 用 `diff` 替代 `comm` 避免 GNU comm 在 process substitution 上的"not in sorted order"误报
- 用 `awk -v n="$MAX_ID" 'BEGIN{for(i=1;i<=n;i++) print i}'` 生成 1..N 序列避免 `seq` 大数字 scientific notation
- 不使用 `set -e`,所有 `grep` 加 `|| true` 防止无匹配提前终止
- `logs/` 范围限制为顶层 `task*.log` (排除 `.hydra/` 自动生成的不可变元数据)

### 2.3 修复的真正残留引用

audit 过程中发现并修复的 renumber 残留:

| 文件 | 旧编号 | 新编号 |
|------|--------|--------|
| `descriptions/task31_baseline_seed_extension.md:37` | task108_s1 | task26_s1 |
| `scripts/task27_3tokenizer_analysis.py:141` | task108_s1 | task26_s1 |
| `verdicts/task119_3tokenizer_kNN_result.md:25` | task108_s1 | task26_s1 |
| `verdicts/task26_28_29_30_cancelled_k_ablation.md:69-72` | task108/110/111/112 | task26/28/29/30 |
| `logs/task108_s1/` → `logs/task26_s1/` | (目录 rename) | - |
| `logs/task*.log` × 10 文件内嵌 ANSI 序列包裹的 task104-107 | task104-117 | task23-25 |

**辅助脚本**: `scripts/fix_old_task_refs_in_logs.py` (一次性,处理 ANSI 包裹的 task<old>)

## 3. 最终 audit 输出

```text
===============================================
R9 合规审计 (CLAUDE.md R9-Enforce)
===============================================
时间: 2026-07-19 23:18:00

[1] descriptions/ 任务编号连续性检查
  ✅ descriptions/ 连续无空洞 (max=35, 共 35 个任务)

[2] verdicts/ 任务编号连续性检查 (warning)
  ⚠️ verdicts/ 有历史空洞 (允许): 1 2 3 4 5 6 7 8 9 11
      (历史遗留, 不在 R9-Enforce 强制范围; 但如果新任务导致空洞, 必须填补)
  ℹ️  verdict 数量 = 39, description 数量 = 36 (差值 = 分析任务无 description)

[3] 残余 task<old_num> 引用检查
  ✅ 无 task104-130 范围 renumber 残留 (排除自指)

===============================================
R9 合规审计结果
===============================================
✅ PASS: 2 项通过
EXIT: 0
```

## 4. 产物清单

| 产物 | 路径 |
|------|------|
| CLAUDE.md 新规则 | `/home/wlia0047/ar57/wenyu/GeneRec/CLAUDE.md` R9-Enforce 节 |
| audit 脚本 | `/home/wlia0047/ar57/wenyu/GeneRec/scripts/audit_r9_compliance.sh` |
| 辅助清理脚本 | `/home/wlia0047/ar57/wenyu/GeneRec/scripts/fix_old_task_refs_in_logs.py` |
| R8 §16 清理 | (本任务无活跃任务登记,无需清理) |

## 5. 后续使用规范

- **新任务登记**: `/grid-new-task` skill 必须先调用 audit 脚本确认 descriptions/ 无空洞,再取 NEXT_TASK_ID
- **loop tick 强制**: 每个 loop tick 第一步 `bash scripts/audit_r9_compliance.sh`,FAIL 则先修复
- **renumbering 后**: 必须 `grep -l "task<old_num>"` 全仓搜索残余引用并修复
- **renumber 脚本自身**: 不在 audit 检查范围 (白名单),其映射定义合法保留

---

**result:** Task #134 完成 — CLAUDE.md R9-Enforce 三层规则 + audit 脚本落地,所有现存 renumber 残留已修复,audit 跑通 PASS (descriptions/ 连续 1-35,verdicts/ 仅历史空洞 warning,task104-130 范围无残留)

result: Task #134 — completed (R8 §9.3 retroactive compliance line appended by Task #114; refer to verdict body above for full details).
