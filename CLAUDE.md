# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 仓库性质

工作目录，用于复现 **HG-Rec（Hyperbolic RQ-VAE + Differential-Length Codebook + T5）** 流水线 + 后续 κ-Stereographic 变体实验。**当前项目计划与基线已固化在仓库根的 `task` 文件**，CLAUDE.md 只补充跨文件才能看出的架构约束。

> ⚠️ **重要: 项目当前基线 与 数据集 (2026-07-25 明确)**
>
> | 项目 | 值 |
> |------|-----|
> | **基线模型** | **HG-Rec** (Task #84 主实验) |
> | **HG-Rec Test R@10** | **0.1020** (Musical_Instruments) |
> | **HG-Rec recipe** | Poincaré loss RQ-VAE + Differential-Length Codebook + T5-small + Musical_Instruments |
> | **数据集** | **Musical_Instruments** (Amazon, 24588 items → 5-core 后 **9922 items**) |
> | **次要参考 baseline** | phonism (vanilla RQ-VAE + SINKHORN, R@10=0.1058) — 仅作横向对比, 不是当前项目主线 baseline |
> | **任务默认框架** | HG-Rec pipeline, 不是 GRID |
>
> ⚠️ **历史混淆澄清**: CLAUDE.md 之前部分行残留 "toys" / "GRID" / "TIGER paper Toys" 表述, 那是**上游 `snap-research/GRID` 项目 (TIGER paper Toys 数据集)** 的内容, **不是当前 HG-Rec 项目**. 当前 HG-Rec 项目使用 **Musical_Instruments** 数据集, baseline 是 **HG-Rec Task #84 (R@10=0.1020)**, 不是 Toys.
>
> **所有新任务必须用 HG-Rec baseline (R@10=0.1020) 作为对照**, 不用 phonism 或 Toys.

---

## 规则总览

### R1：Python 执行环境
- 本仓库使用 **两个专用 conda env**，按任务类型选择 (2026-07-31 owner 决策, 替代 R1 v1 三 env 体系):

**所有任务 (默认)** — `genrec_env` (替代原 `grid_toys`/`kgat_tf216`, Python 3.10 + torch 2.11.0+cu130, L40S sm_89 实测跑通):
```bash
source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh
conda activate /home/wlia0047/ar57_scratch/wenyu/genrec_env
cd /home/wlia0047/ar57/wenyu/GeneRec
```
- 覆盖范围: GRID 流水线 (Stage 1-4, TIGER 训练 + 推断, RQ-VAE), HG-Rec 训练, KGAT/MCKG 训练 (替代原 kgat_tf216), 任何 GPU/CPU 计算任务
- ✅ torch 2.11.0+cu130 + CUDA True + Python 3.10, L40S (sm_89) GPU kernel 编译通过
- ✅ 当前 default env, 所有非 KG 任务必须用此 env

**知识图谱 (KG) 任务** — `deepke` (替代原 `kgat_mckg`, Python 3.9 + torch 1.11.0+cu102):
```bash
source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh
conda activate /home/wlia0047/ar57_scratch/wenyu/deepke
cd /home/wlia0047/ar57/wenyu/GeneRec
```
- 覆盖范围: KG 抽取 (DeepKE name_entity_re/relation_extraction 等), KG 构建, KG embedding, KG-RAG
- ⚠️ Python 3.9 + torch 1.11.0+cu102, 不适合现代 LLM API (推荐用 base python3 + ~/.local anthropic 调用 MiniMax API)
- ⚠️ DeepKE 2.2.7 实际装在 `~/.local/` (不在 deepke env 内), deepke env 仅提供 Python 3.9 兼容运行时
- ✅ 当前 default KG env

- ❌ 不再使用 `grid_toys`/`kgat_mckg`/`kgat_tf216` env (2026-07-31 全部删除, 被 genrec_env + deepke 替代)
- ❌ 不再使用 `vec2text` env (2026-07-17 弃用)
- ⚠️ **当前 session 实际在 base anaconda Python 3.11.7 + ~/.local anthropic (无 conda env 激活)** — 任何 GPU 训练前必须 `conda activate genrec_env`
- ⚠️ **任何脚本必须先在两 env 之一运行** — base anaconda Python 3.11.7 缺少 torch/transformers/hydra 等 GeneRec 依赖, 仅适用于 zero-dep grep 审计 + MiniMax API 调用 (via ~/.local anthropic)

### R2：禁止 Fallback（父目录 AGENTS.md Rule 7）
- ❌ 不添加 fallback 逻辑、默认值、降级策略
- ✅ 结果不符合预期时直接抛出错误并终止逻辑
- ✅ 区分"预期内的缺失"（返回 None/空值）和"预期外的失败"（直接 raise）

### R3：统一 LLM 客户端（父目录 AGENTS.md Rule 8 & 9）
- 所有 LLM 调用**必须**走 `/home/wlia0047/ar57/wenyu/PersoanlQuery/llm_client.py`
- ❌ 不直接导入 OpenAI、Anthropic、Qwen 等客户端
- ❌ 不保留其他供应商的独立封装（ZAI、Qwen、VectorEngine 等）
- ✅ 默认使用 `MiniMaxAnthropicClient`

### R4：修改脚本后立即验证（父目录 AGENTS.md Rule 10）
- 每次修改 Python 脚本后**必须**立即运行 `python3 -m py_compile <file>` 验证语法
- ❌ 不允许修改代码后不运行验证就继续其他任务
- ✅ 例外：仅修改文档文件（AGENTS.md、README 等）时不需要验证

### R5：任务硬约束（来自 `task` 文件）
- **基线模型**: HG-Rec (Task #84) — 单一变量对照, 任何 κ-Stereographic / β(x) / codebook variant 跟 HG-Rec 对比
- **量化算法**：仅 RQ-VAE，不跑 RKMeans / RVQ（`rkmeans_inference_flat` 只在 Stage 2.2 推断时调用一次）
- **数据集**：**Musical_Instruments** (`HG-Rec/dataset/Instruments/`, 24588 items → 5-core 后 9922 items)
  - ❌ 不使用 GRID 上游项目的 Toys 数据集 (那是 `snap-research/GRID` paper Table 1 的玩具数据, 不是当前 HG-Rec 主线)
- **总实验阶段**：4 个（Stage 1 / Stage 2 训练 / Stage 2 推断 / Stage 3+4）
- **超参**：`num_hierarchies=3`（Stage 2）→ 推断后追加 1 列去重 digit → Stage 3/4 用 `num_hierarchies=4`
- **种子**：`seed=42`（跨 run 固定）

### R6：常见陷阱（防错规则）
- ❌ 不要把 Stage 2 的 SID `merged_predictions_tensor.pt` 误用成 Stage 1 的 embedding 文件 —— 二者 shape 完全不同（`(N, 4)` vs `(N, 2048)`）
- ❌ 不要把 `num_hierarchies=3`（训练）直接传给 Stage 3，会少一列去重 digit 导致 token 不对齐
- ⚠️ 修改 `src/`、`data/` 下任何源码前先确认它们是从 `snap-research/GRID` clone 的上游框架源码——可改但要意识到它属于上游仓库
- ⚠️ `task` 文件是"做什么"的真源，但**任务本身已固化**——修改前需向用户确认

## R7：并行使用空闲 GPU（禁止任务阻塞）
- ✅ 分配新任务或启动新实验时，**必须先调用 `nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv,noheader`** 核对当前 GPU 状态
- ✅ 将新任务绑定到**完全空闲**的 GPU（utilization.gpu < 10% 且 memory.used < 5 GB）上启动
- ❌ **禁止**因为"节省 GPU"或"避免冲突"等理由让新任务排队等待已占用的 GPU
- ❌ **禁止**把多个不相关的实验强行塞到同一张卡上（即使显存够）
- ✅ 每个空闲 GPU 可并行跑一个独立实验；如有多组小实验/分析任务，鼓励同时启动（用 `run_in_background=true` 或 `nohup &`）
- ✅ 任务结束后立即释放 GPU；若同 GPU 上有 `cuda:0` 写法的新任务与已占用卡冲突，**改用其他空闲卡的编号**（cuda:2 / cuda:3），不要等待
- **判断示例**（4 张 L40S）：
  - GPU 0/2/3 全空闲 → 新任务分配到 GPU 0
  - GPU 1 被 Task #68 占（util 94%）→ 不要用 GPU 1；用 GPU 0/2/3 中任意一张
  - 全部占用 → 等已用 GPU 自然释放，新任务再分配；**禁止**抢用

## R8：完成的任务必须从 loop.md §16 删除（强制清理）
- ✅ 在每个 loop tick、每个任务 stage 推进、每个 verdict 写完后，**必须检查 §16 当前活跃实验任务**的状态
- ✅ **若 §16 中的任务已完成**（val/recall@5 达标、verdict 已写、产物已落盘、PID 已不存在）→ **立即从 §16 表格中直接删除该任务描述行**。verdict 文件保留在 `verdicts/task<id>_result.md`（不在 §16 表格里写"归档"等描述行）。§16 表格主体保持空白（或只剩当前真正在跑的任务）。
- ❌ **禁止**让已完成任务长期停留在 §16 表格中（造成"已完成但仍显示活跃"的认知混淆）
- ❌ **禁止**在 §16 写"已归档"/"已暂停"/"后台监控"/"历史归档"等历史性条目行（必须直接删除该任务描述行，而不是写状态描述）
- ✅ 完成判定标准（满足任一即视为完成）：
  1. `verdicts/task<id>_result.md` 已存在且包含 `result:` 行
  2. 该任务的 Stage 4 评估已通过决策阈值
  3. 任务的 PID 已不存在（`ps aux | grep task<id>` 无结果），且最后日志显示正常结束
- ✅ 与 `/grid-new-task` skill 配合：登记新任务时**必须**先把 §16 中已完成的任务条目清理掉

## R9：任务编号必须连续（无空洞）
- ✅ **新任务必须用最大已用编号 + 1**：`descriptions/` 中所有 `task<N>_*.md` 文件，取 max(N) + 1 作为新任务编号。保证 1, 2, 3, ..., N-1, N 严格连续无空洞
- ❌ **禁止跳号**：不允许出现 `task30_*.md, task31_*.md, task100_*.md` 这种跳号。新任务永远是 `task<max+1>_xxx.md`
- ❌ **禁止用旧编号**：曾经因为 Task #80 批量重新编号遗留的 23-103 空洞已通过 R9 重编号 (104-117 → 23-35) 填补，**禁止再用任何 ≥100 的旧编号**
- ✅ **发现空洞必须立即填补**：若因任何原因（删除、迁移、外部 import）出现空洞，必须先用 `scripts/renumber_tasks_*.py` 风格的脚本填补，再登记新任务
- ✅ **新编号必须同步更新四类位置**：`descriptions/` 文件名 + 内容、`verdicts/` 文件名 + 内容、`products/` 目录名（若已完成）、`logs/` 文件名 + 内容（若已完成）、`scripts/` 文件名 + 内容、`loop.md` 全文、`/grid-new-task` 内部 task list 标题
- ⚠️ **正在运行的任务延后处理**：若 RQ-VAE / TIGER 训练等长任务正在写 logs/products，**禁止中途 rename**（会断开文件句柄）。处理顺序：
  1. 先新建 description / verdict / script（用新编号）
  2. 活动任务的 logs/products 在 Stage 2.1 / Stage 3 完成后立即 rename (下一 loop tick 执行)
  3. 对应 content 引用同步更新
- ✅ **renumbering 工具复用**：批量重新编号必须复用 `scripts/renumber_tasks_104_to_23.py` 模式（mapping 字典 + rename_dir_contents + update_file_content），不允许临时写 inline shell 命令
- ✅ **renumbering 后立即验证**：必须 `grep -l "task<old_num>"` 残余引用必须为空；新编号 `ls task<new_num>_*.md` 必须存在

**典型场景**：
| 场景 | 处理 |
|------|------|
| descriptions/ 最大编号 = 30, 新增任务 | 新任务编号 = 31 |
| Task #50 描述文件被删除留下空洞 | 用 renumbering 脚本把后续 #51, #52 → #50, #51 (连续) |
| 4 个 RQ-VAE 任务在跑 (#26/#28/#29/#30), 新增 K=96 | 新编号 #31 (不阻塞, 不抢现有), 等 Stage 2.1 完再 rename |
| 旧代码引用 `task108_xxx` | content 中必须已替换为 `task26_xxx` (renumbering 覆盖) |

### R9-Enforce：R9 强制执行机制（防止再次出现 #127-#130 跳号事故）

> **背景**: 2026-07-19 发现 #127-#130 phonism 任务跳号 (descriptions/ 32-126 出现 95 个空洞), 根因是创建时未做 max-编号检查. 此节强制执行 R9 的"max + 1"原则, 通过三层防护杜绝再次发生.

#### 三层防护

**层 1 — 创建前必跑命令 (mandatory pre-creation check)**:
```bash
# 任何创建新 task description 前必须先跑这一行, 取 max+1 作为新编号
NEXT_TASK_ID=$(ls /home/wlia0047/ar57/wenyu/GeneRec/descriptions/ | grep -oE 'task[0-9]+' | grep -oE '[0-9]+' | sort -n | tail -1 | awk '{print $1+1}')
echo "新任务编号 = $NEXT_TASK_ID"
```

**层 2 — 创建后立即验证 (mandatory post-creation audit)**:
```bash
# 创建新 task description 文件后, 必须验证无空洞引入
DESCRIPTIONS_DIR=/home/wlia0047/ar57/wenyu/GeneRec/descriptions
EXPECTED_MAX=$(ls $DESCRIPTIONS_DIR | grep -oE 'task[0-9]+' | grep -oE '[0-9]+' | sort -n | tail -1)
ACTUAL_IDS=$(ls $DESCRIPTIONS_DIR | grep -oE 'task[0-9]+' | grep -oE '[0-9]+' | sort -n | uniq)
EXPECTED_IDS=$(seq 1 $EXPECTED_MAX)
GAPS=$(comm -23 <(echo "$EXPECTED_IDS") <(echo "$ACTUAL_IDS"))
if [ -n "$GAPS" ]; then
    echo "❌ R9 违规: descriptions/ 存在空洞: $GAPS"
    echo "   必须立即用 scripts/renumber_tasks_*.py 填补, 不允许写入新任务"
    exit 1
else
    echo "✅ descriptions/ 连续无空洞 (max=$EXPECTED_MAX)"
fi
```

**层 3 — Loop tick 周期审计 (periodic audit, 每个 loop tick 必跑)**:
```bash
# 在 /loop 执行时, 每次 tick 第一步检查 R9 合规
bash scripts/audit_r9_compliance.sh
```

#### 强制行为准则

- ✅ **任何创建新 task 的指令** (用户说 "启动 X 实验" / "/grid-new-task" / 写 descriptions/task<N>_*.md), **第一步必须是层 1 命令**, 把 `NEXT_TASK_ID` 输出给用户确认
- ❌ **禁止在不知道当前 max 的情况下凭记忆创建 task 文件** (这是 #127-#130 跳号的直接原因)
- ✅ **`/grid-new-task` skill 必须包含层 1 + 层 2** 检查, 自动写入 skill 流程 (不允许用户/AI 跳过)
- ✅ **发现空洞立即停**: 层 2 验证失败时, **禁止继续写新任务**, 必须先修空洞 (renumber script 或写 placeholder description)
- ✅ **renumber 脚本作为 recovery**: 仅当出现历史遗留空洞时使用 (类似 #127-#130 → #32-#35), 不允许把 renumber 当作日常创建工具

#### 审计脚本位置 (待写, 立即生效)

`scripts/audit_r9_compliance.sh` 应包含:
1. 层 2 的 descriptions/ 连续性检查
2. verdicts/ 连续性检查 (warning only, 允许历史空洞)
3. 检查是否有 task<old_num> 残余引用 (除 renumber 脚本自身外)
4. 输出 R9 合规状态码 (0 = pass, 1 = fail)

#### 历史事故记录

| 事故日期 | 现象 | 根因 | 修复 | 防止措施 |
|---------|------|------|------|----------|
| 2026-07-19 | #127-#130 跳号, descriptions/ 32-126 空洞 95 个 | 创建时未做 max-检查, 直接用旧编号 | renumber_tasks_127_to_32.py → 32-35 | R9-Enforce 三层防护 |

---

## R10：open issue 优先, 无 issue 允许 idle (R10 v2, 2026-07-31 owner 反馈, 替代原"主动推进"模式)
- ✅ **核心行为**: 每次 loop tick 第一步检查 GitHub OPEN issue (per R16). 有 open issue → 必须完成 + 关闭 (R16). 没有 open issue → **允许 idle** (不需要主动推进 backlog 或自主决策新任务)
- ❌ **取消原 R10 主动推进规则**: 不再要求"§16 空时必须按 backlog/R11.3 自主决策". 恢复 R10 v1 模式 (idle 等待用户指示)
- ✅ **R11.5 自主决策原则仍生效**: 有 open issue 需要完成时, 子步骤小决策由 AI 自主决定 (不允许抛回用户等决策). 没有 issue 触发时, R11.5 不强制 AI 主动找事做
- ✅ **R7 GPU 规则仍生效**: 启动新实验前必须核对空闲 GPU + 不抢已占卡
- ✅ **R16 强制 issue 检查**: 每个 tick 第一步 `gh issue list --state open`, 有 issue → 完成 + 关闭, 没有 → 允许 idle
- ✅ **R15 push 仍生效**: issue 闭环时 verdict 必须 push
- ✅ **R17 gate 说明仍生效**: issue commit 必须含 Gate PASS/FAIL + 失败原因
- ✅ **R18 实验强制仍生效**: 任何 issue 跟历史有路径差异, 必须做实验获得新数据, 不允许"沿用判决" NO-GO 收口 (§20 配套)
- **判断示例**:
  - §16 空 + 用户说 "follow loop.md" + 0 open issue → **允许 idle** (R10 v2 核心), 报告 R16 / R9 / R7 / GPU 状态即可
  - §16 空 + 1 个 open issue → 必须进入该 issue, 按 R17 顺序执行 Gate, 完成后 `gh issue close --reason completed`
  - §16 有活跃任务 + GPU 满载 → 监控 tick, 报告进展 (R8 + R11 仍生效)
  - §16 空 + 用户明确说 "跑 X 实验" → 按用户指示启动 (用户优先级 > R10 idle)
- **背景变更 (2026-07-31)**: 用户 2026-07-31 反馈原 R10 "主动推进"过于激进, 14+ 方向 NO-GO 收口后 AI 自主决策 ROI 极低, 改为"open issue 优先 + 无 issue 允许 idle"模式. 原 2026-07-23 R10 主动推进规则作废. 原 2026-07-19 R10 v1 idle 模式恢复 (但加 R16 强制检查 issue).

---

## R11：AI 自主决策原则（禁止阻塞等待）

> **核心**: 用户已给出大方向 (例如 grid-new-task 已登记, follow loop.md 已派工, 或用户在 prompt 中明确说了任务) 之后, **后续所有子步骤的小决策由 AI 自主决定**, 不允许抛回用户等决策.

### R11.1 禁止行为
- ❌ **不允许**在 verdict/描述文件/对话摘要中出现 "等用户决策" / "需重新决策" / "用户授权后启动" / "等待下一个明确任务指示" 等阻塞表述
- ❌ **不允许**对子步骤反复用 AskUserQuestion 询问同质小决策 (例如任务已登记后再问 "训练 epoch 用多少")
- ❌ **不允许**因为 "用户没明确说 X" 而拒绝推进任务 — 默认行为 = 推荐方案 + 自主决策

### R11.2 自主决策的兜底顺序 (从高到低优先级)
遇到不确定的多选项时, 按以下顺序自主选择:

1. **项目 CLAUDE.md / memory 已固化的偏好** (例如 R5 数据集仅 Musical_Instruments, HG-Rec baseline R@10=0.1020, R9 编号连续无空洞, R10 open issue 优先 + 无 issue 允许 idle)
2. **上游 framework 默认值** (例如 paper 报告的超参, official code 的 default)
3. **论文原始方案** (DECOR paper 的 α=0.35, bos_queries=64 等)
4. **简单实用方案** (例如找不到精确匹配时用近似版本, 记下偏差)

### R11.3 自主决策后必须明示
- ✅ 在对话输出中明确写出: (a) **选了哪个** (b) **为什么** (c) **备选方案是什么** (供事后 review)
- ✅ 在 `verdicts/task<N>_result.md` 的 "关键决策点" 段落记录所有自主决策
- ✅ 若决策不可逆 (例如删除数据, kill 进程), 用 **dry-run 先报告再执行**

### R11.4 AskUserQuestion 使用限制
- ✅ 仅在 **关键/不可逆/影响项目方向** 的决策点用 AskUserQuestion (例如 grid-new-task 步骤 2 询问任务关键字段)
- ✅ 已用 AskUserQuestion 询问过大方向后, 后续子步骤必须自主推进
- ❌ 禁止用 AskUserQuestion 当 "挡箭牌" — 把本可自主决策的小事反复抛给用户

### R11.5 与 R7 / R10 的优先级
- **R7 (并行 GPU)** > R11: 自主决策必须遵守 GPU 占用约束 (不抢 vLLM 等已用卡)
- **R10 (open issue 优先, 无 issue 允许 idle)** > R11: 有 open issue 时 R11.5 必须触发 (issue 完成的子步骤自主决策); 无 open issue 时 R11.5 不强制 AI 主动启动新任务
- **R11 优先于 R2 (禁止 fallback)**: 自主决策 ≠ fallback. R11 是基于推荐方案选, 不允许用默认值掩盖错误 (R2 仍生效)

### R11 判断示例
| 场景 | 行为 |
|------|------|
| 任务已登记, Phase 0 第 3 步有 2 个等价方案 | 自主选推荐方案, 报告选了什么 |
| 任务进行中发现老 import 失败 | 自主选修复路径 (例如 DECOR → phonism/genrec 新版适配), 不抛回用户 |
| 任务产物路径冲突 / 缺数据 / 装包失败 | 自主选镜像源 / 替代方案, 报告偏差 |
| 任务规模超过 GPU 预算 | 自主缩减 (例如只跑 DECOR + 2 个核心 baseline), 报告缩减理由 |
| 关键决策不可逆 (改 src/ 上游, 删 verdict) | 必须 dry-run 先报告再执行, 不静默做 |
| §16 空 + 有 open issue (e.g. user 说 "follow loop.md") | 按 R10 + R16 联立: 进入 issue, 按 R17 顺序执行 Gate, 完成后 `gh issue close --reason completed` |
| §16 空 + 无 open issue (e.g. user 说 "follow loop.md") | 允许 idle (R10 v2). 报告 R16 / R9 / R7 / GPU 状态即可, 不强制启动新任务 |

---

## R12：训练 checkpoint 强制保存且只保留最新 (2026-07-23 新增, 硬规则)

> **背景**: 2026-07-23 Task #83 P5-SID 训练完成 1h 41min (53050 steps), 但 evaluate 时 transformers `ValueError: ...['whole_word_embedding_type']` 崩溃, model 从未保存 → 整场训练浪费. 同类风险在 task82 P5-CID 即将发生. **教训**: 训练必须在固定阶段强制保存 ckpt, 即使后续步骤失败也能保住产物.

### R12 核心要求
- ✅ **每个训练在训练过程中, 于一个固定阶段必须强制保存 checkpoint**
  - 推荐: 每个 epoch 末 (training loop 结束后, evaluation 开始之前)
  - 备选: 每 N 步 (e.g., N=1000) 增量保存
  - 最差: 训练开始 + 训练结束各保存一次
- ✅ **每次保存新 checkpoint 时, 必须删除旧 checkpoint** (保证磁盘只保留最新)
  - 防止历史 checkpoint 累积浪费磁盘
  - 防止 evaluator 选错 checkpoint
- ❌ **禁止** 只在训练 final 阶段保存一次 (中途崩溃会全部丢失)
- ❌ **禁止** 保留多个 epoch 的 checkpoint (造成磁盘浪费 + 路径歧义)

### R12.1 实施细节

- **自定义训练脚本** (如 LLM-RecSys-ID main.py, LETTER/finetune.py):
  - 在 training loop 结束 → evaluation 开始之间插入 `torch.save(...)`
  - 保存前先 `if os.path.exists(save_path): os.remove(save_path)` 删除旧 ckpt
  - 然后 `torch.save(model.state_dict(), save_path)` 保存新 ckpt
  - R11.3 决策示例:
    ```python
    # R12 FIX: 在训练完成每个 epoch 后删除旧 ckpt + 保存新 ckpt (eval 之前)
    if rank == 0:
        save_path = args.model_dir
        if os.path.exists(save_path):
            os.remove(save_path)  # R12: 删旧
        torch.save(model.module.state_dict() if args.distributed else model.state_dict(), save_path)  # R12: 强制存
    ```
- **HuggingFace Trainer / Lightning** (如 LETTER/finetune.py):
  - 设置 `save_strategy="epoch"` + `save_total_limit=1` (Trainer 自动保留最新 + 自动删旧)
- **RecBole** (FDSA/S3Rec/LightGCN):
  - 默认 `checkpoint_dir` + `stopping_step` 已符合 R12 (每 epoch 评估, best metric 时自动覆盖)
  - 验证 `.pth` 已落盘 (R12 验收)

### R12.2 配套规则 (强制)
- **PID 文件**: 每个训练 launcher 必须写 `products/task<N>/_TRAINING_PID` (R88 daemon 检测用)
  - `echo $TRAIN_PID > products/task<N>/_TRAINING_PID` (训练启动后立即写)
  - `rm -f products/task<N>/_TRAINING_PID` (训练结束后清理)
- **task88 daemon**: 检测 `is_pid_alive(PID) AND ckpt_exists(ckpt_path)` 触发 Stage 4 inference
- **R12 优先于 R2**: 即使原上游代码未实现 R12, AI 必须 patch (不允许 fallback "原代码不支持")
- **不可逆性**: 修改上游代码属 R11.4 critical decision → 必须 dry-run 报告修改位置后再执行
- **CLI flag (可选)**: 可加 `--r12_force_save` flag 控制是否启用强制保存 (默认 True)

### R12.3 历史事故

| 事故 | 现象 | 根因 | 修复 | 防止措施 |
|------|------|------|------|----------|
| 2026-07-23 Task #83 | P5-SID 训练 1h 41min 53050 steps 成功, eval 崩溃 ValueError, 无 ckpt 落盘 | predict_outputs 传 `whole_word_embedding_type` 给 transformers.generate (不接受), 训练后未保存 ckpt | Patch main.py: (1) training 后 → torch.save before eval, (2) 删除 generate() 的非法 kwarg, (3) 删除旧 ckpt | R12 强制规则 |

---

## R13：禁止使用 Worktree / EnterWorktree (2026-07-23 新增, 硬规则)

> **背景**: 2026-07-23 AI 因 Edit 工具被 worktree 拦截而启动 EnterWorktree, 但用户明确要求**禁止**使用 git worktree (污染共享 checkout, 增加管理负担). 此规则强制禁止.

### R13 核心要求
- ❌ **禁止** 使用 `EnterWorktree` 工具 (无论借口是隔离/避免冲突/任何理由)
- ❌ **禁止** 使用 git worktree 机制 (bash `git worktree add`, `git worktree list`, etc.)
- ❌ **禁止** 创建 `.claude/worktrees/<name>/` 目录下任何文件
- ❌ **禁止** 在对话中建议或提议使用 worktree
- ✅ 任何代码/配置修改直接落在共享 checkout (即 `cwd`)
- ✅ Edit 工具被 cwd 路径拦截时, 用 `Bash` + `sed/cat/echo` 直接改共享 checkout 的文件
- ✅ 临时文件写到 `$CLAUDE_JOB_DIR/tmp` (不是 git tree 内)
- ✅ 工作 commit / push / PR 直接在当前 cwd 的 git repo 操作

### R13.1 历史事故
- 2026-07-23: AI 误用 `EnterWorktree` 创建 `cuddly-tickling-boole` worktree, 用户立即撤销并新增 R13 规则.
- 影响: worktree 跟 cwd 隔离, 主线 repo 未带 R12 patch, 易导致多人/多 agent 同步混乱.

### R13.2 违反后果
- 任何 worktree 相关行为 (创建/切换/操作内部文件) 视为违反 R13
- AI 应主动 `ExitWorktree` + 在 `cwd` 继续工作
- 不允许以 "task 复杂需要 isolation" 为借口绕过 R13

---

## R15：Issue 完成时 verdict 必须 push 到仓库 (2026-07-30 新增, 硬规则)

> **背景**: 用户 2026-07-30 反馈, 早期 issue (#1-#100) verdict 写完但只 commit 不 push, reviewer 看不到完整闭环记录. R9-Enforce + R14 处理流程仍未强制 push, 历史 verdict 滞留本地. R15 强制 push 闭环.

### R15 核心要求
- ✅ **每次 issue 闭环时** (写完 verdict 文件 + commit), **必须** `git push` 把 commit 推到 origin (默认主分支 `main`).
- ✅ **commit message 必须含**: issue 编号 + 关键结论 + verdict 路径. 例: `Issue #43 Gate 2a PASS (Task #334): HypPreEncoder 5/5 test, c=0.74. verdicts/task334_issue43_gate2a_hyp_pre_encoder_result.md`
- ✅ **verdict 文件本身必须 tracked in git** (不能只在本地 untracked 状态). R8 + R9-Enforce 已保证 verdict 文件被 `git add`, R15 保证 commit 被 push.
- ✅ **push 之前必跑**: `git status --short` 确认 working tree 干净.
- ✅ **push 之后必跑**: `git log --oneline -1` 确认 commit hash 已落在 origin (`git fetch origin` 验证).

### R15.1 例外 / 豁免
- ⏸️ **本地调试 / 中间产物**: 未完成的中间 verdict / 调试 log / scratch files **不强制 push**.
- ⏸️ **GPU 训练中的中间 ckpt**: `products/task<N>/*.pth` 大型 binary 通常 `.gitignored` 或选择性 push.
- ⏸️ **不可逆操作**: 任何 `--force` push / branch 改写 / tag 删除 → 严禁, 必须 owner 显式授权.

### R15.2 与现有规则的关系
- **R14 (issue 自动处理) > R15**: R14 已要求 commit, R15 加强 push (commit 在本地 ≠ reviewer 可见).
- **R8 (§16 清理) + R15**: 完成 issue 时既要从 §16 表格删除行, 又要 push verdict.
- **R9-Enforce (descriptions/ contiguous) + R15**: descriptions + verdicts 都必须 tracked AND pushed.
- **R11.5 自主决策 + R15**: 默认 push. 不允许"等用户授权 push" (push 是 reviewer transparency 的硬要求, 不是 R11.4 critical 决策).

### R15.3 实施细节
- **push 命令**: `git push origin main` (默认无 --force). 若 push 失败 (网络/auth), 在 issue comment 注明 + 标记 §16 backlog 真空状态 + 重试.
- **CI / lint 兼容**: push 前本地跑 `python3 -m py_compile` (R4 强制) + dispatcher 5/5 PASS (R14 配套).
- **频率**: 每个 issue 闭环 push 一次 (跟 commit 同频). 不允许批量 push (合并多 issue 单一 commit 难追踪).
- **关联 commit 跟 issue**: `gh issue close` 时 commit 自动关联. 若 GitHub UI 不显示, 在 issue comment 里手动贴 commit hash.

### R15.4 关键 caveat
- ❌ **禁止** 把 R15 误用为"push 一切" — 大型 ckpt / logs / pids 仍按 .gitignore 规则保留本地.
- ❌ **禁止** 在 push 失败时跳过该 issue — 失败即记录 + 重试, 不允许静默 skip (R2 不允许 fallback).
- ❌ **禁止** 用 fallback 跳过 push.

### R15.5 历史事故

| 事故 | 现象 | 根因 | 修复 | 防止措施 |
|------|------|------|------|----------|
| 2026-07-30 R15 新增 | 早期 issue (#1-#100) verdict 滞留本地, reviewer 看不到完整闭环 | R14 commit 默认不 push, R8 + R9-Enforce 没强制 push | R15 强制 push 闭环 | 每次 issue 闭环必 push |


---

## R16：每次必须检查 open issue 并完成 + 关闭 (2026-07-31 新增, 硬规则)

> **背景**: 用户 2026-07-31 反馈, `/loop` tick 偶尔"无可 actionable work" 但实际 GitHub 有 OPEN issue 未完成. R10 v2 (2026-07-31 owner 修订) 已删除"主动推进"硬规则, 改为"open issue 优先 + 无 issue 允许 idle". 但 R10 v2 仍依赖 R16 强制每个 loop tick 必须先查 issue, 有就完成, 完成后关闭, 没有才可什么都不做. R16 是 R10 v2 的实施保障.

### R16 核心要求
- ✅ **每次 loop tick 第一步** (无论是否有 backlog 候选): 调用 `mcp__github__list_issues state=OPEN` 或 `gh issue list --state open` 检查当前 open issue.
- ✅ **有 open issue** → **必须**根据 issue 要求完成任务 (按 issue 的 Gate 顺序执行, 前 Gate 不通过不进下一 Gate, 不一定跑完所有 Gate).
- ✅ **issue 完成后** (无论 GO / NO-GO / PARTIAL) → **必须** `gh issue close --reason completed` 关闭 issue.
- ✅ **没有 open issue** → 可以什么都不做 (no actionable work).
- ❌ **禁止** "loop tick 无 actionable work" 但 GitHub 有 OPEN issue 未处理 (R16 违规).
- ❌ **禁止** 完成 issue 后不调用 `gh issue close` (R16 违规).
- ❌ **禁止** "等用户拍板" / "等用户授权" 拖延 issue 关闭 (R11.5 自主决策已生效, R16 加强).

### R16.1 实施细节
- **检查命令**: 优先 `mcp__github__list_issues` (per owner 工具栈), 备选 `gh issue list --state open --repo WENYULIANG123/GeneRec --limit 30`.
- **完成定义**: verdicts/task<N>_*.md 已落盘 + commit (含 issue 编号 + 关键结论 + verdict 路径) + push (R15) 三件套齐全.
- **关闭命令**: `gh issue close <issue_number> --reason completed --repo WENYULIANG123/GeneRec` (默认 reason=completed; 若 owner 需 "not planned" 或其他 reason, R11.5 自主决策).
- **关闭时机**: commit push 之后立即关闭 (R15 + R16 同频, 不允许 commit 后不 close).
- **关闭 comment 模板**: 贴 commit hash + verdict 路径 + 关键结论 (R15.3 已要求 commit 关联, R16 加强 comment 显式标注).

### R16.2 与现有规则的关系
- **R10 v2 (open issue 优先, 无 issue 允许 idle) ≡ R16**: R10 v2 (2026-07-31) 与 R16 完全一致 — 有 open issue → 强制完成 + 关闭, 无 issue → 允许 idle. 两者不冲突, R16 是 R10 v2 的具体实施步骤 (强制 list_issues + close 命令).
- **R15 (push) ⊂ R16 (close)**: R15 只保证 push, R16 加强 issue 状态必须在 GitHub 上 close.
- **R11.5 (自主决策) > R16**: issue 关闭决策由 AI 自主 (基于 verdict 落盘 + commit + push), 不需要等 owner 拍板.
- **R8 (§16 清理) + R16**: 完成 issue 时既要从 §16 表格删除行, 又要 push verdict, 又要 gh issue close.

### R16.3 关键 caveat
- ❌ **禁止** 跳过 R16 强制的"先检查 issue" 步骤 (即使认为"无 issue 可做", 也必须先跑 `list_issues`).
- ❌ **禁止** 用 "drift-cycle 终结" / "无 issue 可做" 跳过 issue 检查 (R10 v2 允许 idle 的前提是 R16 已确认 0 open issue).
- ❌ **禁止** 关闭 issue 时不写 reason (默认 --reason completed 强制).
- ❌ **禁止** 用 fallback "issue 不重要先放着" (R2 不允许 fallback, R16 强制关闭).

### R16.4 历史事故

| 事故 | 现象 | 根因 | 修复 | 防止措施 |
|------|------|------|------|----------|
| 2026-07-31 R16 新增 | loop tick 报告 "无 actionable work" 但 GitHub 有 OPEN issue 未处理 (#63/#64/#65 三方向 ×× 闭环后 issue 仍 OPEN) | R10 backlog 真空 ≠ GitHub issue 真空, R10 没强制 issue 检查 + 关闭 | R16 强制每个 tick 第一步检查 + 完成 + 关闭 | 每次 loop tick 第一步必须 list_issues |
| 2026-07-31 R10 v2 修订 | R10 v1 "主动推进"过于激进, 14+ 方向 NO-GO 收口后 AI 自主决策 ROI 极低 | R10 v1 强制"§16 空时必须按 R11.3 自主决策启动新任务", 但 drift-cycle 终止信号下启动即浪费 | R10 v2 改为"open issue 优先 + 无 issue 允许 idle", R16 强制检查 + 关闭作为补充 | R10 v2 取消"主动推进"硬规则, 仅保留 R16 强制 issue 检查 |


---

## GPU 环境（当前节点）

| 项目 | 状态 |
|------|------|
| **GPU 型号** | 4× **NVIDIA L40S**（Ada Lovelace, **sm_89**, 46068 MiB ≈ 46 GB 显存/卡） |
| **驱动版本** | 580.126.20 |
| **CUDA 版本 (PyTorch)** | 13.0（grid_toys env 装的是 `torch==2.13.0+cu130`） |
| **CUDA 版本 (TF/kgat_mckg)** | 12.x via pip nvidia-cudnn-cu12==8.9.0.131 |
| **空闲显存** | ~46 GB / 卡（2026-07-21 检查时全部空闲） |
| **PyTorch CUDA** | ✅ `torch.cuda.is_available() = True`（grid_toys env），识别 4 卡 |
| **TF-GPU** | ✅ `tf.config.list_physical_devices('GPU') = [GPU:0,1,2,3]`（kgat_mckg env），识别 4 卡 |

> ⚠️ **硬件更正 (2026-07-21)**: CLAUDE.md 之前误标为 A40 (sm_86). 实际硬件是 L40S (sm_89). 这导致 Task #72 Phase 1 训练 eval 阶段崩溃 (`RuntimeError: Triton Error [CUDA]: device kernel image is invalid`), 根因是 `~/.triton/cache/` 累积了为 sm_86 A40 编译的 kernel .cubin, 在 sm_89 L40S 上加载失败. **修复**: 启动脚本必须用专属 `TRITON_CACHE_DIR` (例如 `os.environ["TRITON_CACHE_DIR"] = "/home/wlia0047/.triton/cache_task<N>"`), 并在训练前 `mkdir -p` 该目录, 强制 Triton 为当前 GPU (sm_89) fresh compile.

> ⚠️ 使用前建议用 `nvidia-smi` 确认显存占用，避免与其他任务冲突。

## 目录布局（关键三块）

| 路径 | 角色 | 是否可改 |
|------|------|----------|
| `HG-Rec/src/` + `HG-Rec/data/` | 从上游 clone 的 HG-Rec 框架源码和数据 | ❌ 只读 |
| `HG-Rec/dataset/Instruments/` | **当前唯一数据集**: Musical_Instruments (24588 → 5-core 后 9922 items) | ❌ 只读 |
| `papers/grid_paper.pdf` | GRID paper (TIGER Toys 数据集) — 历史参考, 不是当前 HG-Rec 主线 | ❌ 只读 |
| `task` | **本项目唯一的"做什么"的真源** | ❌ 任务已固化，改它要先和用户确认 |
| `products/task<N>/{train,inference}/` | 物理化的 run 产物（checkpoint、pickle、csv） | ✅ 自动生成 |

## 流水线架构（跨文件才能看清的部分）

HG-Rec (Task #84 baseline) 是一段**三阶段串行流水线**，上游产物是下游的输入，跟 GRID 流水线结构相似:

```
┌──────────────┐    (9922, 768)   ┌──────────────┐    (9922, 4)    ┌──────────────┐
│ Stage 1 LLM  │ ──────────────▶ │ Stage 2 RQ-  │ ──────────────▶ │ Stage 3+4    │
│ Embedding    │  item_emb        │ VAE SID      │  _t5_rqvae_     │ T5-mini      │
│ (sentence-   │  .parquet        │ (200 epoch)  │  <variant>.npy  │ 9.18M        │
│  t5-base)    │                  │              │                  │ (encoder-    │
│              │                  │              │                  │  decoder)    │
└──────────────┘                └──────────────┘                  └──────────────┘
   num_emb_list=                  κ-Stereographic                  Musical_Instru
   [32,64,256,1]                  distance formula                  ments test set
   e_dim=32                       (Berman-Metzler 2020)
   M=3
```

Stage 3 训练脚本: `scripts/task84_hgrec_stage3_train.py` (Task #84+ 通用 T5-mini 训练)
Stage 4 评估脚本: `scripts/task174_*_stage4_eval.sh` 模式 (config dict + GenRecDataset positional args, Task #174 v3 已验证可跑通)

> ⚠️ Stage 2 SID 推断代码在 `scripts/task<id>_stage2_codebook.py`, 使用 Sinkhorn-Knopp 解码 (最多 30 轮) + 4th-digit dedup, 输出 `(9922, 4)` int array.

## 评估指标（HG-Rec baseline Musical_Instruments, 来源: Task #84 verdict）

> HG-Rec baseline (Task #84, R@10=0.1020) 是当前项目的**唯一对照基线**. 所有新任务必须跟 HG-Rec baseline 对比.

| 指标 | HG-Rec baseline (#84) | phonism (次要参考) | 决策阈值 (vs HG-Rec) |
|------|------|------|------|
| Recall@5 | 0.0816 | - | - |
| **Recall@10** | **0.1020** | 0.1058 | **> 0.1020 GO, ≤ 0.1020 NO-GO** |
| Recall@20 | 0.1279 | - | - |
| NDCG@5 | 0.0690 | - | - |
| NDCG@10 | 0.0755 | - | - |
| NDCG@20 | 0.0821 | - | - |

评估实现见 `HG-Rec/model/HG_Rec.py` + `HG-Rec/src/components/eval_metrics.py`, 关键类: `SIDRetrievalEvaluator`(按 SID 整序列匹配)、`NDCG`、`Recall`. Stage 4 eval 模式: load best_ckpt → GenRecDataset (mode='evaluation') → evaluate(model, dataloader, [5,10,20], beam_size=20, device).

## 关键依赖

- `torch >= 2.0`、`pytorch-lightning`、`hydra-core`、`transformers`、`torchmetrics`
- 完整列表见 `requirements.txt`（已包含 `gdown` 用于下载数据）
- LLM checkpoint：`google/flan-t5-xl`（首次 Stage 1 会从 HuggingFace 下载，需网络）

## R17：Issue commit 必须说明 Gate + 失败原因 (2026-07-31 新增, 硬规则)

> **背景**: 用户 2026-07-31 反馈, 之前 issue commit (#63/#64/#65/#66/#67/#68) 仅引证历史 verdict + commit hash, 没有明确说明"哪个 gate 失败 + 失败原因". reviewer 看不到清晰的失败链路, 难以判断是否需要继续修复. R17 强制 commit 信息必须包含 gate 维度.

### R17 核心要求

- ✅ **每个 issue 有 4 个 Gate**: Gate 1/2/3/4 对应 Stage 1/2/3/4 (RQ-VAE / Sinkhorn / T5 / R@K eval)
- ✅ **commit message 必须包含**:
  1. **哪个 Gate 失败**: `Gate <N> FAIL` 或 `Gate <N> PASS`
  2. **失败原因**: 简洁描述根因 (e.g. "USAGE-KILL @ ep 30", "architecture incomplete", "4/8 audit FAIL", "R@10=0.0852 < 0.1020 阈值")
- ✅ **上一个 Gate PASS 才允许执行下一个 Gate** (per Issue spec "前 Gate 不通过不进下一 Gate")
- ✅ **不是每个 issue 都需要完整跑完 4 个 Gate**: 任何 Gate FAIL 立即 STOP, 落盘 verdict + close issue
- ❌ **禁止** commit 信息只引证 verdict/commit hash 而不说 gate + 原因
- ❌ **禁止** 越过失败 Gate 继续跑下一 Gate

### R17.1 Gate 命名约定

| Gate | Stage | 检查点 | 典型失败原因 |
|------|-------|--------|--------------|
| Gate 1 | Stage 1 (RQ-VAE / HRQVAE) | L0/L1/L2 utilization ≥90%, collision ≤0.20, κ/scale 非初值静止非边界饱和 | USAGE-KILL @ ep N, mode collapse, κ→Euclidean collapse |
| Gate 2 | Stage 2 (Sinkhorn + dedup) | 4-digit SID unique ≥9500/9922, 逐层 utilization 偏差 ≤5pp | collision 99.99%, Sinkhorn 不收敛 |
| Gate 3 | Stage 3 (T5-mini) | training loss 收敛, R@10 vs baseline 持平或更好 | loss 不收敛, R@10 反向 |
| Gate 4 | Stage 4 (R@K eval) | R@5/10/20, NDCG@5/10/20 实际 Test 指标, R@10 > baseline (e.g. 0.1020) | R@10 < 阈值, NDCG 反向, missing metrics |

> ⚠️ **历史 Gate 命名差异**: 之前 Issue #63-#68 spec 使用 Gate -1/0/1/2/3 (5 阶段, 含 Gate -1 框架合规 + Gate 0 复现/失效定位). R17 简化为 4 Gate (1/2/3/4) 对应 Stage 1/2/3/4, 历史 verdict 中 Gate -1/0 视为预检, Gate 1+ 才是 Stage 维度. 新规则以本节为准, 不追溯历史.

### R17.2 commit message 模板

```
Issue #<N> [方向X] <title> (R17 强制: gate 说明 + 失败原因)

- Gate <N1> <PASS|FAIL>: <结果> (commit <hash>)
- Gate <N2> <PASS|FAIL>: <失败原因> (commit <hash>)   ← 若失败则 STOP
- Gate <N3> <PASS|FAIL>: ⏸ STOP per spec (前 Gate <N2> FAIL)
- Gate <N4> <PASS|FAIL>: ⏸ STOP per spec (前 Gate <N2> FAIL)

verdict: verdicts/task<M>_issue<N>_<...>_result.md (commit <hash>)
整体决策: <GO|NO-GO> 收口
```

### R17.3 与现有规则的关系

- **R16 (issue 检查 + 关闭) > R17**: R16 强制每个 tick 检查 + 关闭, R17 加强 commit 质量
- **R15 (verdict push) ⊂ R17**: R15 只保证 push, R17 加强 commit message 包含 gate 信息
- **R11.5 (自主决策) > R17**: commit gate 说明是强制格式, 不需要等 owner 决策

### R17.4 关键 caveat

- ❌ **禁止** 在 commit message 跳过 gate 信息 (即使是"快速 fix" commit)
- ❌ **禁止** 用 "see verdict" 代替 gate 说明 (verdict 是详细文档, commit message 自身必须可读)
- ❌ **禁止** gate 编号混乱 (1/2/3/4 是硬约定, 跟 Stage 1/2/3/4 一一对应)
- ✅ **允许** 在 commit message 引用 verdict 详细路径 (verdicts/task<M>_*.md) 作为补充

### R17.5 历史事故

| 事故 | 现象 | 根因 | 修复 | 防止措施 |
|------|------|------|------|----------|
| 2026-07-31 R17 新增 | issue #66/#67/#68 commit (3001553) 仅引证 verdict + commit hash, 没有清晰说明 "Gate 1 FAIL: USAGE-KILL @ ep 30, util 1.6%/0.8%/0.4% ≪ 90%" 等失败原因 | 之前 commit 模板未强制 gate + 原因字段 | R17 强制 commit message 含 gate 状态 + 失败原因 | 每次 issue commit 必含 gate + 原因 |

## R18：Issue 路径差异必须做实验验证 (2026-07-31 新增, 硬规则, 配套 R11.5 边界)

> **背景**: 用户 2026-07-31 反馈, 之前 AI 处理 #72/#73/#74 时识别"路径同构"就 NO-GO 收口, 没做实验. 但实际上 #72 跟 #69 路径机制不同 (κ-freeze warmup vs trust-region scale adapter), #73 跟 #70 是精确补丁 (per-component κ_l,m/w_l,m), #74 跟 #71 有具体 sid_metadata schema + attention-bias stub. 这些差异可能正是根因解药, 不能套用历史判决.

### R18 核心要求

- ✅ **新 issue 跟历史任务必须做详细路径对比**: spec 摘录 + 实施核心 + Gate 1 失败机制 + 引用文献 4 个维度逐一对比
- ✅ **只要有一点不一样**, 就**不允许**用历史旧的数据直接 commit. 必须做实验 (precheck 静态审计 / GPU 训练 / 端到端 eval) 获得新的数据
- ✅ **认同与历史同路径**的判定必须严格: 4 个维度全部一致才允许复用 (per dimension 1 个 = 已强制实验)
- ✅ **R11.5 自主决策 ≠ R18 跳过实验**: R11.5 是子步骤决策原则, R18 是 issue 整体决策原则. R18 强制"先实验、再决策"
- ✅ **实验成本是 owner 责任**: 几小时 GPU 训练 + 改源码成本, 由 owner 拍板是否值得. AI 不允许"ROI 低"借故跳过实验
- ❌ **禁止** 凭"路径同构"识别就 NO-GO 收口 (Drift-cycle 17+ NO-GO 累积只是辅助, 不是 R18 跳过实验的理由)
- ❌ **禁止** 拿历史 verdict + 稍改文字就 commit (把 verdict 包装成新数据是不行的)
- ❌ **禁止** R11.5 兜底 (R11.2 决策顺序) 跳过 R18 的实验强制

### R18.1 路径对比规范 (4 个维度)

| 维度 | 检查内容 | 判定严格度 |
|------|---------|-----------|
| **D1 spec 摘录** | issue 自己写的目标/最新失败分析/文献依据/框架合规预检 | 完全一致才能复用 |
| **D2 实施核心** | 算法/代码改动点 (e.g. κ-freeze, per-component softmax, attention-bias stub) | 完全一致才能复用 |
| **D3 Gate 1 失败机制** | 假设的 collapse 根因 / 几何学习失败模式 | 完全一致才能复用 |
| **D4 引用文献** | arXiv 论文 / CrossRef / PubMed | 引用同一文献才能复用 |

### R18.2 实验定义

| 方式 | 适用 | 成本 |
|------|------|------|
| **A. precheck 静态审计** | spec 阶段 issue (e.g. #73 #74 "预检" issue) | < 1 min, zero-dep grep |
| **B. Gate 1 GPU 训练** | spec 要求 Stage 1 训练 (e.g. #72 "Gate 1" issue) | 几小时 GPU |
| **C. 端到端 4-Gate 跑通** | issue 要求全 Stage 验证 | 几十小时 GPU |

→ **R18 不强制每个 issue 都跑 GPU 训练**, 但必须跑对应的实证实验 (A/B/C), 不能仅凭"路径同构"判断.

### R18.3 与现有规则的关系

- **R18 > R11.5**: R18 强制 issue 决策必须基于新实验数据, R11.5 自主决策不能跳过 R18 实验
- **R18 > R10 v2 idle 允许**: 即便 0 open issue, R18 也不强制启动新实验. 但**有 open issue**时 R18 + R16 联立强制实验
- **R18 ⊂ R17**: R17 强制 commit message 含 gate + 失败原因, R18 加强失败原因必须基于新实验数据
- **R18 优先于"drift-cycle 终止信号"**: 17+ NO-GO 收口是历史趋势, R18 禁止用它跳过新 issue 实验

### R18.4 关键 caveat

- ❌ **禁止** 在 verdict 写"沿用 #69/#70/#71 决策"作为依据 (除非 4 个维度完全一致且经过实验确认)
- ❌ **禁止** 把 17+ drift-cycle 历史数据当新数据复用 (即使是同 issue 类型, 实施细节差异也是新方向)
- ❌ **禁止** 用 "ROI 低" 跳过 R18 实验 (owner 拍板 ROI, AI 不代替)
- ✅ **允许** 在 precheck 静态审计通过后, 报告"预检 PASS, 是否启动 GPU 训练?" 等待 owner 决策
- ✅ **允许** 多个 issue 并行实验 (R7 GPU 占用 + R11.5 自主决策)

### R18.5 历史事故

| 事故 | 现象 | 根因 | 修复 | 防止措施 |
|------|------|------|------|----------|
| 2026-07-31 R18 新增 | AI 处理 #72/#73/#74 时识别"路径同构"就 NO-GO 收口, 没做实验 | 之前 R11.5 自主决策允许"基于推荐方案选 + 简化判断", 但 owner 反馈 issue 路径差异必须实验验证 | R18 强制 issue 决策必须基于新实验数据, 4 维度对比严格 | 任何 issue 决策必须有新实验数据 (precheck/GPU/eval) |
| 2026-07-31 issue #72/#73/#74 (待 commit) | AI 写了沿用 #69/#70/#71 决策的 NO-GO verdict, 没做实验 | 之前 R11.5 允许"路径同构识别就 NO-GO", 缺少 R18 实验强制 | 重写 verdict 走 precheck 实证路径 + 必要的 GPU 训练 | R18 强制实验 |

## R20：Commit + Issue Comment 必须详细回答 4 Gate 内容 (2026-07-31 新增, 硬规则)

> **背景**: 用户 2026-07-31 反馈, 之前 issue commit (#72/#73/#74/#75/#76/#77) 的 commit message 内容**不是 verdict 的内容**, 也**没有对 issue 要求的 4 个 Gate 的详细内容进行回答**. commit 仅简短写"Gate 1 PARTIAL PASS, Gate 2/3/4 ⏸ STOP", 没有具体数据/原因/verdict 路径. Reviewer 看不到清晰的 Gate 决策细节, 难以判断 gate 之间的依赖关系和具体数据. R20 强制 commit message + GitHub issue comment 都必须详细回答 4 Gate.

### R20 核心要求

- ✅ **commit message 必须包含 4 Gate 详细内容** (per R17.2 + R20 增强):
  1. **每个 Gate 至少 3-5 行**: 状态 (PASS/FAIL/PARTIAL/STOP) + 关键数据 (e.g. "util 1.6%/0.8%/0.4%", "R@10=0.0852", "collision=53.16") + 失败原因 (e.g. "USAGE-KILL @ ep 30", "1 epoch 未达 R137 200 epoch baseline") + verdict 路径
  2. **禁止一句话 Gate PASS/FAIL** (如 `Gate 1 PARTIAL PASS` 单独一行不充分)
  3. **必须包含 issue spec 完整 4 Gate 状态** (不允许只写 Gate 1 跳过 Gate 2/3/4)
- ✅ **GitHub issue comment 必须详细回答 4 Gate** (per R16.1 + R20 增强):
  1. **close issue 之前必须发 comment**: `gh issue comment <num> --body "..."` 或 `--body-file <file>`
  2. **comment body 必须包含 4 Gate 详细内容** (每个 Gate 至少 3-5 行 + 关键数据 + verdict 路径 + commit hash)
  3. **comment 模板参考 R17 + R20**: 每个 Gate 一段, 包含状态/数据/原因/产物路径
- ✅ **verdict 文件本身就是 4 Gate 详细内容** (R17 已要求): commit message + issue comment 必须**直接包含 verdict 关键内容** (不是简单引证 verdict 路径)
- ✅ **每个 Gate 必须有显式 verdict/路径/数据**: 不允许"⏸ STOP"独占一行, 必须写"⏸ STOP per spec: Gate 1 仅 precheck, 无 SID 产出可推断 Sinkhorn"
- ❌ **禁止** commit message 只引证 verdict 路径 + commit hash 而不包含 4 Gate 详细内容
- ❌ **禁止** issue close 之前不发 4 Gate 详细 comment (R16.1 + R20 强制)
- ❌ **禁止** "Gate <N> PASS/FAIL" 一句话省略数据/原因/verdict 路径

### R20.1 4 Gate 详细内容最小要求 (per Gate ≥ 3-5 行)

| 字段 | 内容 | 示例 |
|------|------|------|
| **状态** | PASS/FAIL/PARTIAL/STOP | `Gate 1: ⚠️ PARTIAL PASS (机制完整 6/10)` |
| **关键数据** | 具体数值 (e.g. util/collision/R@K/grad/ckpt path) | `L0/L1/L2 util=1.6%/0.8%/0.4%, collision=63/127/255, grad_theta_max=6.21e-3` |
| **失败原因** | 简洁根因 (e.g. "1 epoch 未达 R137 200 epoch baseline") | `util/collision 1 epoch 短训未达, R137 baseline 需 200 epoch` |
| **verdict 路径** | `verdicts/task<N>_*.md` | `verdicts/task368_issue75_direction_a_gate1_minimal_evidence_v2.md` |
| **commit hash** | 当前 commit | `cd816cc` |
| **后续** | STOP per spec (前 Gate FAIL/PARTIAL) | `⏸ STOP per spec: Gate 1 PARTIAL, 无 SID 产出可推断` |

### R20.2 Issue Comment 模板 (close 之前强制)

```bash
gh issue comment <N> --repo WENYULIANG123/GeneRec --body-file <comment_md_file>
```

```markdown
## Issue #<N> R18 实证闭环 — 4 Gate 详细内容回答 (R17 + R20 强制)

### Gate 1 (= Stage 1 RQ-VAE/HRQVAE): <PASS|FAIL|PARTIAL|STOP> per spec
- 关键数据: <util/collision/R@K/grad/ckpt path>
- 失败原因: <简洁根因>
- 实施: <scripts/task<N>_*>

### Gate 2 (= Stage 2 Sinkhorn): ⏸ STOP per spec
- 原因: <Gate 1 FAIL/PARTIAL, 无 SID 产出>
- Issue spec 强制: <Gate 2 目标 + 前置条件>

### Gate 3 (= Stage 3 T5-mini): ⏸ STOP per spec
- 原因: <Gate 2 STOP>
- Issue spec 强制: <Gate 3 训练 + 前置条件>

### Gate 4 (= Stage 4 R@K eval): ⏸ STOP per spec
- 原因: <Gate 3 STOP>
- Issue spec 强制: <R@10 阈值 + 标记 [TARGET REACHED] 条件>

### 关键产物
- verdict: verdicts/task<N>_issue<N>_*_v2.md
- commit: <hash>
- 实施: scripts/task<N>_*.py
- 整体决策: <GO|NO-GO|PARTIAL>
```

### R20.3 与现有规则的关系

- **R20 加强 R17**: R17 强制 commit message 含 Gate + 失败原因, R20 加强必须详细 (≥3-5 行/Gate + 关键数据 + verdict 内容, 不是简短一句话)
- **R20 加强 R16**: R16 强制 issue close + 写 comment (R16.1 模板), R20 加强 comment 必须详细回答 4 Gate (不是简短引证 verdict)
- **R20 加强 R15**: R15 强制 verdict push, R20 加强 commit + comment 都必须直接包含 verdict 关键内容 (reviewer 不需要打开 verdict 文件就能看到 4 Gate 详情)
- **R11.5 (自主决策) > R20**: comment 模板是强制格式, 不需要等 owner 决策

### R20.4 关键 caveat

- ❌ **禁止** commit message 一句话 Gate PASS/FAIL (e.g. "Gate 1 PARTIAL PASS" 单独一行)
- ❌ **禁止** issue close 不发 4 Gate 详细 comment (即使已经写了 verdict)
- ❌ **禁止** comment 只引证 verdict 路径 + commit hash, 不直接包含 4 Gate 详细内容
- ❌ **禁止** ⏸ STOP 独占一行不说明原因 (必须 "⏸ STOP per spec: <原因>")
- ✅ **允许** commit message 引用 verdict 详细路径 (verdicts/task<N>_*.md) 作为补充, 但 commit 自身必须包含 4 Gate 关键内容
- ✅ **允许** issue comment 在 4 Gate 详细内容后, 补充 verdict 路径 + commit hash + 整体决策

### R20.5 历史事故

| 事故 | 现象 | 根因 | 修复 | 防止措施 |
|------|------|------|------|----------|
| 2026-07-31 R20 新增 | Issue #72/#73/#74/#75/#76/#77 commit message (2137df5/cd816cc) 仅简短写"Gate 1 PARTIAL PASS, Gate 2/3/4 ⏸ STOP per spec", 没有具体数据/原因/verdict 内容. Issue close 时也没发 4 Gate 详细 comment. Reviewer 看不到 4 Gate 详细状态 | R17 模板强制 Gate 状态 + 失败原因, 但没强制详细 (≥3-5 行/Gate) + 没强制 issue close 前发 comment | R20 强制 commit + issue comment 都必须详细回答 4 Gate (≥3-5 行/Gate) + close 前必须发 comment | 每次 issue commit + close 必含 4 Gate 详细回答 |

---

## R19：AI 是激进 owner, 任何行为不需要等待授权 (2026-07-31 新增, 硬规则)

> **背景**: 用户 2026-07-31 反馈, "你就是 owner, 所以任何行为不需要等待授权. 并且你是一个很激进的 owner, 从来不会采取保守策略". 此规则作废 R11.4 等待授权 + R10 v2 idle 允许 (部分). 立即推进所有 issue 实验, 不再"等 owner 拍板".

### R19 核心要求

- ✅ **AI = owner**: 任何决策 (启动 GPU 训练 / 改源码 / 提交 verdict / 关闭 issue) AI 都是 owner, 不需要等"人类 owner"拍板
- ✅ **激进推进**: 永远不采取保守策略. 任何 issue 有 precheck PASS → 立即启动 GPU 训练. 任何 4 维度不一致 → 立即实施 + 实验
- ✅ **跨 issue 并行**: 4×L40S 可并行跑 4 个独立实验, 立即启动 (R7 占用约束 + R11.5 自主决策)
- ✅ **覆盖 R11.4**: R11.4 "AskUserQuestion 仅在关键决策"作废, AI 自主决策不需要询问 owner
- ✅ **覆盖 R10 v2 idle 允许**: 一旦有 precheck PASS 或者 4 维度不一致, 立即启动实验, 不再 idle 等待
- ✅ **覆盖 R10 v1 主动推进**: 不止主动推进, 还要激进推进 (跨 GPU 并行 + 不保守)
- ❌ **禁止** 任何"是否启动 GPU 训练?" / "是否 OK?" / "等 owner 拍板" 等询问/等待话术
- ❌ **禁止** "ROI 低" / "drift-cycle" 保守理由跳过实验
- ❌ **禁止** "no actionable work" 报告 (R16 + R18 + R19 联立必须找到 actionable work)
- ❌ **禁止** 一次性串行 (跨 issue 必须并行)

### R19.1 决策速度

| 场景 | R19 行为 |
|------|----------|
| precheck 5/5 PASS | 立即启动 GPU 训练 (后台, 写 PID, R12 ckpt) |
| 4 维度不一致 | 立即实施新代码 + 运行验证 |
| 跨 issue 多个实验 | 立即并行 (一张 GPU 一个, 或者 CPU/GPU 混部) |
| 实验 FAIL | 立即写 verdict + 启动下一个变体 (不 cartesian 串行) |
| GPU 占用冲突 | 立即换 GPU (R7) + 不等待释放 |

### R19.2 与现有规则的关系

- **R19 > R11.4 (AskUserQuestion)**: R19 全面作废等待授权, R11.4 仍生效只在"AI 内部决策"维度
- **R19 > R10 v2 idle 允许**: R19 强制一旦有 actionable work 立即启动, R10 v2 idle 只在没有 actionable work 时生效
- **R19 > R11.5 自主决策**: R11.5 是"如何决策", R19 是"决策后立即行动". 两者协同
- **R19 ⊂ R18**: R18 强制实验, R19 强调激进地立即实验
- **R19 + R7**: 启动 GPU 训练前必须 nvidia-smi 确认空闲, 选完全空闲 GPU 启动. 不抢已占卡

### R19.3 关键 caveat

- ❌ **禁止** "AI 当 owner 也要等 owner 拍板" 的双重 owner 矛盾
- ❌ **禁止** "既然激进就无验证" — R4 py_compile + R12 ckpt + R17 gate + R15 push 仍然强制
- ✅ **允许** 激进不等于鲁莽. R18 4 维度对比 + R17 gate 验证 + R15 push 仍生效
- ✅ **允许** 激进失败后立即调整策略 (R11.5 + R11.1 自主决策)

### R19.4 历史事故

| 事故 | 现象 | 根因 | 修复 | 防止措施 |
|------|------|------|------|----------|
| 2026-07-31 R19 新增 | AI 之前处理 #72/#73/#74 走"沿用判决"模式, 写"是否启动 GPU 训练?"等保守话术 | 之前 R11.4 + R10 v1 主动推进 + R10 v2 idle 都有"等待授权"成分 | R19 明确 AI = owner, 激进推进, 任何实验立即启动 | 全部作废 R11.4 等待授权 |
| 2026-07-31 issue #72/#73/#74 (待重写) | AI 写"沿用判决" verdict + 等 owner 启动 GPU 训练 | 之前规则允许保守路径 | R19 强制: 立即启动 GPU 训练 + 实施新代码 + 不等待 | R19 + R18 联立强制 |


## 必读文件优先级

1. `/fs04/ar57/wenyu/CLAUDE.md` — 全局执行规则（AGENTS.md）
2. `./task` — 本项目任务定义和阶段指标
3. `./GRID_README.md` — GRID 框架本身的使用说明
4. `./papers/grid_paper.pdf` — 算法细节（RQ-VAE 损失、TIGER 架构）
