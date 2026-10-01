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

## 4. 禁止进入 plan 模式

任何会话、任何任务（包括代码重构、规则修订、批量编辑等）都不得使用 EnterPlanMode / Plan agent，进入即视为违规，必须立即退出并直接执行；如确需事先规划，写简短要点到回复正文即可，禁止调用 Plan 工具链。

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


### Stage-3 冻结协议（机器校验；iteration 不得修改）

以下字段是 Stage3 的权威锁定值，供 `stage3_protocol_gate.py` 直接解析。iteration launcher 只能修改 SID/输出路径与 variant 元数据，**不得修改训练超参数**。

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
- **curvature-RQ-VAE 迭代流程**（main-only + zero-bookkeeping-artifact：在 `stage2_RQ-VAE/curvature_RQ-VAE/` 原地迭代，研究设计、实现、验证、Stage2/Stage3、结果解释、接受或回滚、Git closure 全部由 main 直接完成；Skill 不生成任何 Markdown/JSON/JSONL/LOG/TXT 等流程记录；仅保留训练/评测程序自身必要原生产物）：见用户 skill **`~/.claude/skills/curvature-rqvae-iter/SKILL.md`**（与 §13/§14 对齐；不再使用已删除的项目根 `SKILL.md`）。

## 13. 每次迭代完成必须 commit + push（代码 + 产物）

**迭代完成**指：在 `stage2_RQ-VAE/curvature_RQ-VAE/` 上原地修改的那一轮，其 Stage2（含 SID 导出）与 Stage3（含程序原生 `test_final.json`）均已跑完，且代码与 Stage2/Stage3 原生产物已 commit + push 并完成本地/远端 hash 核验。**Skill 本身不生成任何额外流程产物；不得新增 hypothesis / protocol / manifest / audit / decision / review / closure 等 Markdown、JSON、JSONL、LOG 或 TXT 文件。**

### 必须纳入版本库的范围

1. **代码**（`stage2_RQ-VAE/curvature_RQ-VAE/`）：`train_rqvae.py`、`curvature_config.py`、`model/`、`scripts/`（含 `run_stage2_curvature.py`、`run_stage3_curvature.py`、`mvg_check.py`）。
2. **Stage2 产物**（§0 / §10）：`results/stage2_RQ-VAE/curvature_RQ-VAE/` 下本轮训练产出（至少 `item_sids.json`、`rqvae_best.pth`、`sids_for_hgrec.npy`）。
3. **Stage3 产物**（§11）：`results/stage3_T5Train/curvature_RQ-VAE/` 下本轮 run 的 `logs/.../test_final.json`、`training_metrics.jsonl`、对应 `ckpt/.../HG_Rec_best.pth` 及 `_stage3_launcher.log`。

不得只提交代码而遗漏 `results/` 产物；不得只提交 metrics 而遗漏源码变更。**实现与其原生产物必须放进同一次 commit**，使 `git revert` 能把机制与产物作为整体撤销。

### 执行步骤（强制顺序）

1. `git status`：确认无遗漏路径；`git check-ignore -v` 排查 `.gitignore` 误排除产物（若被 ignore，须修正规则后再提交，不得静默跳过）。
2. 单次语义化 commit（建议 message：`<mechanism 简述> + stage2/3 results | test_R@10=<值> vs parent=<值>`），`git add` 覆盖上述三类路径。
3. `git push origin main`（遵守 §8，禁止未授权 force）。
4. 核验：`git rev-parse main` 与 `git ls-remote origin refs/heads/main` 一致；**不一致则本轮视为未交付**。

## 14. 迭代接受 / 回滚门控（2026-10-01 起强制）

比较基准是本树上一次未被回滚的**父条件**，指标取各 run 自己的 `test_final.json` 的 `test_recall@10`（须协议可比，`n_eval = 57439`）。

- **优于父条件** → 接受：保留新源码与其原生产物，commit + push + 核验 hash（§13）。
- **劣于父条件** → 回滚：对实现 commit 执行 `git revert`，恢复父条件源码，再 commit + push + 核验 hash。**禁止** `git reset --hard` 已推送分支，**禁止**删除历史，失败尝试保留在历史中。
- **持平或在噪声内** → 视为未改进，同样回滚，并在同一目录内换一个结构上不同的机制。

被否决的条件仍须完整跑一次 Stage2 与 Stage3，判定依据是它们的原生输出，不得用 Stage2 代理指标（collision / Gini / 熵）替代 Stage3 证据。回滚后工作树即恢复为父条件，下一轮始终从当前最佳条件出发。**不为任何一轮新建 hypothesis / protocol / manifest / audit / decision / review / closure / 失败归档等记录文件**；被否决的尝试只以 revert commit 存在，不额外落盘。

### 与监控/目标的关系

流水线监控、goal complete、向用户汇报之前，必须先满足本节 push 核验；仅本地存在 `test_final.json` 而未 push 时，只能报告「评测已结束，Git 待同步」，不得标记迭代闭环。
