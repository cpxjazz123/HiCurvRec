# Project Rules

## 0. Stage-2 产物路径速查（2026-10-01 起强制）

机制源码与产物只有一棵工作树，**不再按 iter 编号分目录**：

- 源码：`stage2_RQ-VAE/curvature_RQ-VAE/`
- Stage-2 产物：`results/stage2_RQ-VAE/curvature_RQ-VAE/`
- Stage-3 产物：`results/stage3_T5Train/curvature_RQ-VAE/`

Stage-2 训练产物（`rqvae_best.pth` / `sids_raw.npy` / `sids_for_hgrec.npy` / `item_sids.json`）必须落在 `results/stage2_RQ-VAE/curvature_RQ-VAE/`，源码目录只留代码/`scripts/`/`logs/`/`__pycache__`/输入副本，不得再写 `.pth` / `.npy` / `item_sids.json`。**禁止**复制新目录或把 iter 编号写进任何路径、launcher 文件名、`MECHANISM_NAME`、`RQVAE_VARIANT`。启动前必须 `grep RQVAE_OUT_DIR stage2_RQ-VAE/curvature_RQ-VAE/curvature_config.py` 验证路径落在上述统一目录。迭代边界与回滚由 Git 承担，规则见 §14。

## 1. No CLI args — hard-code all parameters

所有项目内脚本（stage1_GeneEmbedding / stage2_RQ-VAE / stage3_T5Train 等）的超参、路径、常量必须直接硬编码在源码里——禁止 argparse、sys.argv、环境变量覆写、CLI flag，要改就改源码后 `python <script>`（无参）重跑；仅 `/tmp/` 下的参考比较脚本可接受 SID 路径 + 标签位置参数。

## 2. Stage 2 SID Quality Gate

**Stage 2 不设任何外层/内层 hard gate**（2026-09-19 终态）：

- **HitRate@K=50 已删除**（2026-09-19）：经验证 `/tmp/sid_metrics_any.py` 的 HR@50 算法（基于 `item_emb.npy` 余弦相似度 + train parquet 共现邻居）**完全不读 SID 文件**，因此对任何 RQ-VAE 变体恒等于常数 0.7165（curvature baseline）/ 0.5002（TIGER baseline），对 RQ-VAE 机制探索无任何区分力；作为 gate 是空操作。
- **所有其他指标**（3-token SID Gini、每层 mean Gini、collision rate、l01_unique_pairs、H(L1|L0)）只作为描述性指标记录到日志，**不作为任何层级的 gate**：经验上 L0 极端塌陷的合法变体（iter11 sk_eps=0.5 是当前最佳，full_gini=0.1143 远高于 baseline 0.0672，collision rate / l01_pairs / H 也偏高）会被 hard gate 误杀；trainer 内层 early-stop 也完全删除（`should_early_stop` 始终返回 `(False, "")`），所有候选跑满 `MAX_GLOBAL_STEPS` 由下游 stage3 `test_R@10` 决定是否采用。
- **stage3 仍可跑**：stage2 gate 全空后所有 RQ-VAE 变体都自动进入下游；裁决完全交由 stage3 完整跑完后的 `test_R@10`（硬目标 > 0.065）。

## 3. stage2 curvature_RQ-VAE 启动：cd 到 `stage2_RQ-VAE/curvature_RQ-VAE/` 后用 `nohup /home/wlia0047/ar57_scratch/wenyu/genrec_env_v2/bin/python3.9 scripts/run_stage2_curvature.py > logs/train_run.log 2>&1 &`（裸 `python3` 无 torch，会立刻 ModuleNotFoundError；launcher 经 `train_rqvae._launch_via_torchrun` 自动 fork 4 卡 DDP，训练输出落在 `logs/train_hyperbolic_migrated.log`）。也可直接跑 `scripts/train_iter.sh`：它先执行 `scripts/mvg_check.py`，MVG 报告 `MVG PASS` 后再启动 Stage2。

## 4. 未完成任务允许持续监控

未完成任务可通过读取运行日志、检查进程状态或使用 `sleep` 间隔轮询持续监控；可使用 `tail` 跟踪日志输出。任务完成后停止监控。监控不得造成重复启动训练或评测任务。

## 5. stage3 train_HG-Rec 启动：cd 到 stage3_T5Train 后用 `nohup /home/wlia0047/ar57_scratch/wenyu/genrec_env/bin/python3.10 train_HG-Rec.py > logs/_stage3_run.log 2>&1 &`（裸 `python3` 无 torch；脚本内置 `_launch_via_torchrun` 自动 fork 4 卡 DDP，硬编码 `nproc_per_node=4 / master_port=50201 / CUDA_VISIBLE_DEVICES=0,1,2,3`，启动前会注入 `NCCL_IB_DISABLE=1 / NCCL_P2P_DISABLE=1 / NCCL_SHM_DISABLE=1 / NCCL_TIMEOUT=3600 / TORCH_NCCL_BLOCKING_WAIT=1`，训练输出落在 `logs/_stage3_launcher.log`）。

## 6. stage2 实际训练前必做梯度通路检查

每次要实际跑 stage2 训练（不论是基线还是新增机制），必须先在 1 个 ckpt + 1 个 batch 上验证所有相关参数的 `loss.backward()` 通路能拿到非零梯度：(1) `total_loss.requires_grad` + `total_loss.grad_fn is not None`；(2) 逐项新机制 loss（commitment / margin / spread / anisotropy / logdet / contrastive / Möbius 等）`torch.autograd.grad(loss_item, model.parameters(), retain_graph=True, allow_unused=True)` 至少 1 个参数非零；(3) 检查 `self._last_*.detach()` 这类截断是否会让 backward 静默失效。若任一项 FAIL，禁止启动 GPU 训练，立即 R50 + 修复后再验证。

## 7. pip 重装 torch / nvidia 必须 `--target=env-site-packages --upgrade` 强制落点

`pip install torch` 在 genrec_env 下会被 `--user` 默认行为或 vllm/transformers resolver 约束把包装到 `/home/wlia0047/wenyu/.local/` 或强制回滚原版，导致 site-packages/torch 与 nvidia runtime 失配、`import torch` 报 `libcublas.so` 缺失。**唯一允许的写法**：`pip install --no-deps --ignore-installed --upgrade --target /home/wlia0047/ar57_scratch/wenyu/genrec_env/lib/python3.10/site-packages <pkg>==<ver>+<cu> --index-url https://download.pytorch.org/whl/<cu>`，必须同次装配套 nvidia-* runtime，否则 libcublas 找不到。任一步 FAIL 立即回滚到 torch==2.13.0+cu130 + cu13 nvidia 套件，绝不"先这样以后再修"。

## 8. Git 分支与推送规则

- **唯一允许的远端**：所有 Git 远端操作只针对 GitHub 仓库 `https://github.com/cpxjazz123/HiCurvRec.git`。`origin` 必须指向该 URL；不得使用、添加、访问或推送到 GitLab 或任何其他远端。
- **GitHub-only 网络操作**：`clone`、`fetch`、`pull`、`push`、`ls-remote`、远端分支/标签查询及其他会访问网络的 Git 操作，只能对上述 GitHub 仓库执行；禁止使用 `gitlab` 远端，即使该远端仍存在本地历史引用。
- **只维护 `main` 分支**：所有开发、提交和后续迭代均必须在 `main` 分支上进行，不得创建、维护或推送其他长期分支；`main` 的上游必须是 `origin/main`。
- **每次推送必须推送到 GitHub `main`**：完成一次提交后，必须使用 `git push origin main` 推送到 GitHub 远端 `main`；不得仅推送到临时分支、个人分支或其他服务。
- **禁止未经授权强推**：不得使用 `git push --force` / `--force-with-lease` 改写 GitHub 历史，除非用户明确授权。
- **推送前后必须核验**：操作前确认 `git remote -v` 中只有目标 GitHub URL；推送后比较 `git rev-parse main` 与 `git ls-remote origin refs/heads/main` 的 commit hash；不一致时不得继续下一轮任务。
- **每次迭代结束必须 commit + push（与 §13 绑定）**：不得以「训练/评测已在磁盘完成」代替 Git 同步；未 push 成功前不得宣称该 iter 闭环、不得启动下一 iter 的 mechanism 改动。

## 9. 禁止使用 git worktree

- ❌ **NEVER**：调用 `EnterWorktree` 或 `git worktree add` 进入任何 worktree，所有编辑/提交/迭代一律在主 checkout（`/home/wlia0047/ar57/wenyu/GeneRec`）下完成
- ❌ **NEVER**：在 worktree 路径（`/fs04/ar57/wenyu/GeneRec/.claude/worktrees/...`）下做任何代码改动
- ✅ **ALWAYS**：所有 Read / Edit / Write / Bash 操作的目标路径必须是 `/home/wlia0047/ar57/wenyu/GeneRec/...`
- **理由**：worktree 在共享 Git 仓库上会创建跨会话的状态分裂（未提交改动、orphan branch、stash 污染），与 §8 "只维护 main 分支" 铁律直接冲突；2026-09-22 已发生一次 EnterWorktree 误用并被强制退出
- **例外**：仅 `/tmp/` 下的参考比较脚本与 `.claude/worktrees/` 下的代码可被纯读取用于历史回查

## 10. Stage-2 产物统一目录（2026-10-01 起生效）

Stage-2 训练产物必须统一落在仓库级 `/home/wlia0047/ar57/wenyu/GeneRec/results/stage2_RQ-VAE/curvature_RQ-VAE/`，与源码目录 `stage2_RQ-VAE/curvature_RQ-VAE/` 同名。源码树下禁写 `.pth` / `.npy` / `item_sids.json` 等产物文件。目录名、`MECHANISM_NAME`、`RQVAE_VARIANT` 与 launcher 文件名一律不带 iter 编号。启动前必须 `grep RQVAE_OUT_DIR stage2_RQ-VAE/curvature_RQ-VAE/curvature_config.py` 验证路径落在该统一目录。

## 11. Stage-3 产物统一目录（2026-10-01 起生效）

Stage-3 训练产物（`HG_Rec_best.pth` / `_stage3_launcher.log` / `test_final.json` / `training_metrics.jsonl` 等）必须统一落在仓库级 `/home/wlia0047/ar57/wenyu/GeneRec/results/stage3_T5Train/curvature_RQ-VAE/`，与 Stage-2 源码与产物目录同名同前缀。Stage3 `trainer.LOG_PATH` / `SAVE_PATH` 必须解析到该子树，且不含任何 iter 编号后缀。


### Stage-3 冻结协议（人工核对；iteration 不得修改）

以下字段是 Stage3 的权威锁定值。原由 `stage3_protocol_gate.py` 自动校验；该脚本已于 2026-10-14 随 skill `curvature-rqvae-iter` 一并删除，因此现在**必须人工核对** `stage3_T5Train/train_HG-Rec.py` 与当前 launcher 是否与本表一致，不一致即不得启动 Stage3。iteration launcher 只能修改 SID/输出路径与 variant 元数据，**不得修改训练超参数**。

```text
STAGE3_NUM_EPOCHS=150
STAGE3_EARLY_STOP=DISABLED
STAGE3_NO_EVAL=true
STAGE3_SKIP_TEST=false
STAGE3_SEED=42
STAGE3_BEAM_SIZE=20
STAGE3_SCREEN_BASELINE_LOG=
```

任何 trainer/launcher 与上述值不一致时必须在 Stage3 启动前返回 `PROTOCOL_GATE_FAIL`；main 不得绕过或重新解释该失败。尤其禁止重新引入基于 train-loss 或 validation patience 的 early-stop。

## 12. GitHub-only 强制执行

本项目后续任何 Git 操作均以 `https://github.com/cpxjazz123/HiCurvRec.git` 为唯一目标。若发现 `origin` 指向其他地址，先修正为该 GitHub URL；不得通过 GitLab 或其他远端进行同步、备份、分支/标签操作或历史改写。

- **仓库内 Git 细则**：§8（分支/推送/hash 核验）、§13（每轮代码+产物 commit+push）与 §14（接受/回滚门控）即完整约束，以本文件为准。
- **curvature-RQ-VAE 迭代流程**（main-only + zero-bookkeeping-artifact：在 `stage2_RQ-VAE/curvature_RQ-VAE/` 原地迭代，研究设计、实现、验证、Stage2/Stage3、结果解释、接受或回滚、Git closure 全部由 main 直接完成；Skill 不生成任何 Markdown/JSON/JSONL/LOG/TXT 等流程记录；仅保留训练/评测程序自身必要原生产物）：见原 skill **`~/.claude/skills/curvature-rqvae-iter/SKILL.md`**（该 skill 及其项目级副本已于 2026-10-14 删除，随附的 `stage3_protocol_gate.py` 一并移除；迭代流程现以本文件 §13/§14 为唯一准则）。


### 与监控/目标的关系

流水线监控、goal complete、向用户汇报之前，必须先满足本节 push 核验；仅本地存在 `test_final.json` 而未 push 时，只能报告「评测已结束，Git 待同步」，不得标记迭代闭环。

## 15. 任务规划必须先做并行性判定（2026-10-13 起强制）

**开始一系列任务规划时，第一步就是判定并行性**：把待办逐项分类为「可并行」与「真依赖」，可并行的**必须并行**，只有真依赖才串行。禁止把「按直觉顺序一个个做」当作默认。

### 可并行（应并行执行）

- 各 Stage2 臂：每臂有独立输出目录与独立 checkpoint，互不影响；
- 跨臂筛选 / 度量 / 残差提取：embarrassingly parallel，按臂分进程；
- 与训练无关的分析：源码与历史结果勘查、指标实现、绘图、TDD；
- 写不同文件的多项工作，前提是**文件所有权不重叠**。

### 真依赖（必须串行）

- 实现 → 测试通过 → 实跑 → 评测：数据依赖，前一步未成立不得启动后一步；
- 任何依赖前一步产出定义的环节（如筛选必须先有度量定义）；
- 同一文件在同一时刻**只能有一个写入者**（例如 Stage2 trainer 的改动必须单一所有者）。

### 并行时的强制约束

1. **独占所有权**：每个并行任务预先声明它可写与不可写的文件；两个任务不得写同一文件。
2. **并行不构成绕过约束的理由**：不得以「资源互斥」「加速需要」为由解除或推翻任何已声明的等待条件、验收契约或禁止项。需要共享资源（GPU）时**排期**，不得默认可用。
3. **不得静默降精度换速度**：并行、向量化、多进程都不得下调任何精度旋钮（bootstrap 次数、采样数、评测集大小等）。确需降精度须先取得用户明确同意。
4. **加速须先验证等价**：改动计算实现（线程数、并行度、向量化）后，必须用已缓存的样本验证数值与改动前**逐位相同**；若有位移，立即回滚并上报。
5. **该并行就跑满机器**：存在并行度时须实际利用（本机 32 核）。典型反例：cKDTree 查询显式写 `workers=1`、跨臂串行 for 循环——这类自缚必须修正。
6. **长任务增量落盘**：按臂 / 按条目增量写盘并可续跑，不得只在结束时一次性输出；否则一次超时会让全部已算结果丢失。
7. **每个切片独立验证后再汇总结论**：并行切片各自的测试与数值必须先通过验收，结论不得由未验收的切片给出。
8. **大量 for 循环必须改为 GPU 向量化**：任何训练或分析循环，若每轮迭代的 Python `for` 数量达到「成千上万次」的量级（即循环体在逐样本 / 逐臂 / 逐层 / 逐邻居推进），**必须**改为一次性张量化并在 GPU 上计算，而不是靠加进程或多线程掩盖。判定与做法：
   - **识别**：先用 `py-spy dump --pid <pid>`（或等价 profiler）确认时间落在解释器循环里，而不是先猜。参照本仓库的两次实际教训：`cKDTree.query(workers=1)` 在 32 核机器上自缚单核；`_successor_overlap` 的双层 Python 循环在残差提取完成后立刻成为关键路径。
   - **改造**：用 `torch` 的批量算子 / `torch.vmap` / 张量化替代 `torch.stack` 拼接后的逐元素运算；距离、邻居、重叠度等改用 `torch.cdist` + `topk` / `torch.bincount` / `segment_reduce` / `index_add`；把 `rng` 生成的索引一次性构造成整块张量，避免每个样本单独取样。
   - **优先 GPU**：可在 GPU 完成的一律放 GPU（本机有 4 卡 Stage2 训练与 GPU 分析环境）；CPU 只承担数据搬运与落盘。
   - **反向传播约束**：涉及 loss 的向量化**不得**引入 detached 张量或切断梯度；STE 语义必须保持与原实现一致。
   - **精度与等价**：向量化后必须按本节第 4 条用已缓存样本验证数值一致；GPU 归约与 CPU 归约可能有末位差异，需明确记录容差与实际偏差，不得当作等价而略过。
   - **内存**：向量化不得以显存溢出为代价；超限时按 batch / chunk 分块（本仓库 `_local_scales` 的 1024 分块即是正确范式），分块边界不得改变结果。
   - **禁止**：不得为了"向量化"而改变机制语义、指标定义、采样分布或超参；效率优化只允许改变计算方式，不允许改变被计算的对象。


## 17. 每个新 Stage2 运行前必须通过 CGV 六项检查（2026-10-19 起强制）

**任何 Stage2 训练启动前，该机制必须先通过 genrec MCP 的 `cgv_check`，且六项全部为 PASS。**
`stage2_start` 会自行校验：证书缺失、任一门未过、或 `train_rqvae.py` 在发证后被改动，都直接拒绝启动，且拒绝发生在归档与派发之前（无任何副作用）。

标准调用序列（`source_dir` 必填，本仓库有多棵树）：

```
write xd://mcp__genrec_cgv_check
{"source_dir": "/home/wlia0047/ar57/wenyu/GeneRec/stage2_RQ-VAE/TIGER_<臂名>"}
```

返回 `passed: true` 且 `gates` 里 G1–G6 全为 `true` 后，才允许 `stage2_start`。只读查看用 `cgv_status`（不跑门、不启动）。

### 六项检查

| 门 | 验证什么 | 防止什么 |
|:--|:--|:--|
| **G1 前向一致性** | 新机制只改变它声明的计算 | 无意改变 loss、量化结果、采样 |
| **G2 梯度路径** | 机制依赖的每个 loss 项都能把梯度传到目标参数 | R5-10 那类零梯度问题 |
| **G3 梯度方向** | 实际梯度符合该机制自己的数学公式 | 投影方向、系数、符号错误 |
| **G4 参数更新** | 完整反向后 optimizer 真的执行了预期更新 | 机制存在但不起作用 |
| **G5 DDP 一致性** | 多卡梯度与独立池化参考一致 | 重复 AllReduce、不同 GPU 更新不一致 |
| **G6 真实训练入口** | 正式训练走的就是被验证的代码路径 | 测试通过但训练执行另一套逻辑 |

### 强制约束

1. **G3 fail-closed**：机制必须先在 `tools/cgv/engine.py` 的 `MECHANISMS` 注册表登记自己的公式；未登记的机制一律拒绝，并给出补登记提示。**不得**以「未被识别」蒙混过关。
2. **G2 的阻断口径**：只有机制**自己声明的参考项**梯度为零才阻断；共享装配里其它只读但静默的项以 `warnings` 上报。理由：家族级遗留问题若一并阻断，所有臂都会被挡死，工具将失去判别力。
3. **证书绑定源码哈希**：`train_rqvae.py` 一旦改动，证书立即失效，必须重跑 `cgv_check`。改完机制不重验就开跑，等于没验。
4. **禁止手工绕过**：`nohup` 直跑 `train_rqvae.py`、`tools/r5_launch_wave.py`、自建 launcher 等一切旁路一律禁止。`cgv_check` 返回 refused 时按其指出的问题修，不绕开。
5. **输出隔离是 CGV 的前置保证，不得削弱**：G6 会执行真实训练入口，`tools/cgv/engine.py::isolated_outputs` 负责把 trainer 的产物路径重定向到临时目录。重定向按「trainer 会写的文件名」发现并向下展开，**输入**（stage1 embeddings、stage0 parquet、dataset）一律不动。
   - 已知教训：`METRICS_PATH` 在导入时由 `LOG_DIR` 计算，只改 `LOG_DIR` 会让验证把真实 `training_metrics.jsonl` 截断成 2 行。任何改动隔离逻辑的提交，都必须先用 md5 前后比对证明生产产物零变化。
   - **禁止**为了让 G6 通过而放宽隔离，或把输出重定向回生产目录。
6. **门失败即结论，不得手改**：某机制被 G2/G3 拦下，说明机制本身在该处无效（如 R5-10 的参考项梯度恒为 0）。正确处置是记录发现、必要时另开 issue，禁止改门禁、放宽容差或改写已上报的指标来「通过」。

### 与 §6 的关系

§6 的梯度通路检查是单次 `loss.backward()` 的最低要求；本节的 CGV 是它的完整替代与加强，覆盖 DDP、参数更新与真实入口。二者都不可省略。
