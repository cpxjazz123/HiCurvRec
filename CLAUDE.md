# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 仓库性质

工作目录，用于复现 Snap Research 的 GRID（Generative Recommendation with Semantic IDs）流水线。**当前项目计划与基线已固化在仓库根的 `task` 文件**，CLAUDE.md 只补充跨文件才能看出的架构约束。

---

## 规则总览

### R1：Python 执行环境
- 本仓库使用**三个专用 conda env**，按任务类型选择：

**GRID 流水线（Stage 1-4，TIGER 训练 + 推断，RQ-VAE）**：
```bash
source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh
conda activate /home/wlia0047/ar57_scratch/wenyu/grid_toys
cd /home/wlia0047/ar57/wenyu/GeneRec
```

**KGAT/MCKG 训练 v1（Task #73 早期 10 epoch baseline，仅 CPU）**：
```bash
source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh
conda activate /home/wlia0047/ar57_scratch/wenyu/kgat_mckg
cd /home/wlia0047/ar57/wenyu/MCKG_repro/knowledge_graph_attention_network
```

**KGAT/MCKG 训练 v2（Task #75+ 完整 GPU 训练，默认）**：
```bash
source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh
conda activate /home/wlia0047/ar57_scratch/wenyu/kgat_tf216
cd /home/wlia0047/ar57/wenyu/MCKG_repro/knowledge_graph_attention_network
```

- ✅ `kgat_tf216` (TF 2.16.2 + cu12 + cudnn 8.9.7.29) 是 **Task #75 起的默认 KGAT 训练 env**，所有 GPU kernel（l2_normalize/Rsqrt, norm/Sqrt, softplus/Sigmoid 等）编译通过，L40S (sm_89) 实测跑通
- ⚠️ `kgat_mckg` (TF 2.15.0 + cudnn 8.9.0.131) 在 L40S (sm_89) 上 GPU kernel JIT 编译失败（l2_normalize/Rsqrt, norm/Sqrt, SoftplusGrad 等），**仅保留**作为 Task #73 baseline 重现（10 epoch CPU 可跑），不再用于 GPU 训练
- ❌ 不再使用 `vec2text` env（2026-07-17 弃用，原 vec2text 装的是 RAG 项目依赖 vec2text/openai/sentence-transformers 等，与 KGAT 训练无关；TF-GPU 路径走的是错配 cu13 不可用）
- ✅ 新建 env 一键脚本：`task_artifacts/scripts/task74_create_kgat_env.sh` (kgat_mckg) / KGAT 训练 launcher `task_artifacts/scripts/task75_kgat_train_tf216.sh` (kgat_tf216)

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
- **量化算法**：仅 RQ-VAE，不跑 RKMeans / RVQ（`rkmeans_inference_flat` 只在 Stage 2.2 推断时调用一次）
- **数据集**：仅 `data/amazon_data/toys/`
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

## R10：§16 必须主动推进（覆盖原 R10 空闲允许等待）
- ❌ **禁止空闲等待**: §16 表格为空时**必须主动推进**下一项任务, 不允许保持空闲等待用户指示
- ✅ **主动推进优先级 (按序尝试)**:
  1. **backlog 候选**: loop.md §16 backlog 中列出的待办任务 (按 R11.5 自主决策选最高 ROI)
  2. **backlog 也没有** → 按 R11.3 自主决策推进低 ROI 实验 (数据分析 / 诊断 / 已有结果整理 / 写 verdict)
  3. **§16 有活跃任务 + GPU 满载** → 监控 tick, 报告进展 (R8 R11 仍生效)
- ✅ **GPU 占用约束**: 即使主动推进, 仍遵守 R7 (不抢已占卡) + R11.5 自主决策边界
- ✅ **不再有"等待用户指示"模式**: 即使监控 tick 期间也要尝试任务切换 (而不是只报告"无任务可推")
- ❌ **明确禁止**: 监控 tick 只报"§16 空, 等待下一个任务" — 这是被覆盖的旧行为
- **判断示例**:
  - §16 空 + 用户说 "follow loop.md" + 4 张卡全空闲 → **必须**主动尝试 backlog 候选或 R11.3 自主决策, 不可空闲
  - §16 空 + 用户明确说 "跑 X 实验" → 按用户指示启动 (最高优先级)
  - §16 有活跃任务 + GPU 满载 → 监控 + 推进任务完成度
  - §16 空 + 监控 tick → 立即尝试切换任务 (而非报告"无任务")
- **背景变更 (2026-07-23)**: 用户 2026-07-23 反馈原 R10 "空闲允许等待"过于被动, AI 应主动推进 backlog 或自主决策新任务. R7 "禁止 GPU 闲置"恢复为最高优先级. 原 2026-07-19 反馈作废.

---

## R11：AI 自主决策原则（禁止阻塞等待）

> **核心**: 用户已给出大方向 (例如 grid-new-task 已登记, follow loop.md 已派工, 或用户在 prompt 中明确说了任务) 之后, **后续所有子步骤的小决策由 AI 自主决定**, 不允许抛回用户等决策.

### R11.1 禁止行为
- ❌ **不允许**在 verdict/描述文件/对话摘要中出现 "等用户决策" / "需重新决策" / "用户授权后启动" / "等待下一个明确任务指示" 等阻塞表述
- ❌ **不允许**对子步骤反复用 AskUserQuestion 询问同质小决策 (例如任务已登记后再问 "训练 epoch 用多少")
- ❌ **不允许**因为 "用户没明确说 X" 而拒绝推进任务 — 默认行为 = 推荐方案 + 自主决策

### R11.2 自主决策的兜底顺序 (从高到低优先级)
遇到不确定的多选项时, 按以下顺序自主选择:

1. **项目 CLAUDE.md / memory 已固化的偏好** (例如 R5 数据集仅 toys, R9 编号连续无空洞, R10 必须主动推进)
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
- **R10 (必须主动推进)** > R11: §16 空时 AI 必须按 backlog/R11.3 推进, 不允许空闲等待
- **R11 优先于 R2 (禁止 fallback)**: 自主决策 ≠ fallback. R11 是基于推荐方案选, 不允许用默认值掩盖错误 (R2 仍生效)

### R11 判断示例
| 场景 | 行为 |
|------|------|
| 任务已登记, Phase 0 第 3 步有 2 个等价方案 | 自主选推荐方案, 报告选了什么 |
| 任务进行中发现老 import 失败 | 自主选修复路径 (例如 DECOR → phonism/genrec 新版适配), 不抛回用户 |
| 任务产物路径冲突 / 缺数据 / 装包失败 | 自主选镜像源 / 替代方案, 报告偏差 |
| 任务规模超过 GPU 预算 | 自主缩减 (例如只跑 DECOR + 2 个核心 baseline), 报告缩减理由 |
| 关键决策不可逆 (改 src/ 上游, 删 verdict) | 必须 dry-run 先报告再执行, 不静默做 |
| §16 真无活跃任务 + 用户说 "follow loop.md" | 按 R10 必须主动推进: 先尝试 backlog 候选, 没有则按 R11.3 自主决策 (不允许报告 "§16 空" 后空闲等待) |

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
| `src/` + `data/` | 从 `snap-research/GRID` clone 的框架源码和数据 | ❌ 只读 |
| `data/amazon_data/toys/` | 当前唯一数据集（Beauty/Sports 已删） | ❌ 只读 |
| `papers/grid_paper.pdf` | arXiv:2507.22224 GRID 论文 | ❌ 只读 |
| `task` | **本项目唯一的"做什么"的真源** | ❌ 任务已固化，改它要先和用户确认 |
| `products/task<N>/{train,inference}/` | 物理化的 run 产物（checkpoint、pickle、csv） | ✅ 自动生成 |

## 流水线架构（跨文件才能看清的部分）

GRID 是一段**三阶段串行流水线**，上游产物是下游的输入，跨阶段的 shape 约定见 `task` 第 153 行的"关键约定"：

```
┌──────────────┐    (N, 2048)    ┌──────────────┐    (N, 4)    ┌──────────────┐
│ Stage 1 LLM  │ ──────────────▶ │ Stage 2 RQ-  │ ───────────▶ │ Stage 3+4    │
│ Embedding    │  merged_        │ VAE SID      │  merged_     │ TIGER        │
│ (flan-t5-xl) │  predictions_   │ (15k steps) │  predictions_│ (T5 encoder- │
│              │  tensor.pt      │              │  tensor.pt   │  decoder)    │
└──────────────┘                └──────────────┘              └──────────────┘
   configs/                       configs/                       configs/
   experiment/                    experiment/                    experiment/
   sem_embeds_                    rqvae_train_flat               tiger_train_flat
   inference_flat                 + rkmeans_inference_flat       + tiger_inference_flat
```

> ⚠️ **非常规约定**：Stage 2 的 SID 推断**复用** `rkmeans_inference_flat` 配置（不是 `rqvae_inference_flat`），因为 SID 推断只读码本，与训练算法无关。task 文件明确写了这一点。

## 配置系统

GRID 用 Hydra 做配置组合，每个 run 形如：

```bash
python -m src.train experiment=<yaml_basename> data_dir=<path> [override_key=value ...]
```

`configs/experiment/` 下 7 个 yaml 名称即为可传入 `experiment=` 的值。`src/train.py` 和 `src/inference.py` 是仅有的两个 CLI 入口。
⚠️ `configs/` 目录已删除，如需重跑实验需从上游 `snap-research/GRID` 重新获取。历史 run 的精确配置保存在 `products/task<N>/train/*/.hydra/` 中。

## 评估指标（Toys 基线，来源：论文 Table 1 RQ-VAE 行）

> ⚠️ 以下为 **RQ-VAE** 专用指标（论文 Table 1），task 指定只用 RQ-VAE。注意 RK-Means Toys 值更高（R@5=0.0376, R@10=0.0577）—— **不可混用**。

| 指标 | 目标（RQ-VAE Toys） | RK-Means Toys（参考） | TIGER 原论文 |
|------|------|------|------|
| Recall@5 | ≥ 0.034 | 0.0376 | 0.0446 |
| Recall@10 | ≥ 0.051 | 0.0577 | 0.0679 |
| NDCG@5 | ≥ 0.022 | 0.0243 | - |
| NDCG@10 | ≥ 0.028 | 0.0308 | - |

评估实现见 `src/components/eval_metrics.py`，关键类：`SIDRetrievalEvaluator`（按 SID 整序列匹配）、`NDCG`、`Recall`。

## 关键依赖

- `torch >= 2.0`、`pytorch-lightning`、`hydra-core`、`transformers`、`torchmetrics`
- 完整列表见 `requirements.txt`（已包含 `gdown` 用于下载数据）
- LLM checkpoint：`google/flan-t5-xl`（首次 Stage 1 会从 HuggingFace 下载，需网络）

## 必读文件优先级

1. `/fs04/ar57/wenyu/CLAUDE.md` — 全局执行规则（AGENTS.md）
2. `./task` — 本项目任务定义和阶段指标
3. `./GRID_README.md` — GRID 框架本身的使用说明
4. `./papers/grid_paper.pdf` — 算法细节（RQ-VAE 损失、TIGER 架构）
