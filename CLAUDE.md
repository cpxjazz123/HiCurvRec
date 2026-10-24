# CLAUDE.md — 本仓库的权威工作契约

本文件描述**当前仓库**（2026-10-24 清理后）的真实布局与规则。任何与本文件不符的路径、脚本、
目录都是过期的，以本文件为准。改动本文件即改动规则。

---

## 0. 仓库现状（清理后的最小主线）

```
stage0_build_parquet/build_recbole_parquet.py   -> results/stage0_build_parquet/{items,train,valid,test}.parquet
stage1_GeneEmbedding/generate_item_embedding.py  -> stage1_GeneEmbedding/output/{sentence_t5.npy,item_ids.json,emb_meta.json}
stage2_RQ-VAE/TIGER_RQ-VAE/                     -> results/stage2_RQ-VAE/TIGER_RQ-VAE/{rqvae_best.pth,sids_for_hgrec.npy,item_sids.json}
stage2_RQ-VAE/TIGER_BEHAVIOR_SAMPLE_RQ-VAE/     -> results/stage2_RQ-VAE/TIGER_BEHAVIOR_SAMPLE_RQ-VAE/
stage3_T5Train/train_HG-Rec.py                 -> results/stage3_T5Train/<RQVAE_VARIANT>/{logs/,ckpt/}
tools/genrec-mcp/server.py                      MCP: stage2_start/stage2_status/stage3_start/stage3_status/cgv_check/cgv_status/stage2_stability_probe
tools/cgv/                                      机制契约六门（run_cgv.py / engine.py / contract.py / oracles.py / adapters.py / family.py）
tools/stage2_stability_probe.py                 诊断探针（不是门）
tools/research/                              新科研工具：core/agents/science/workflows/integrations/tests
results/stage3_T5Train/<94 个历史变体>/         只读历史结果，不再有对应源树
papers/                                         35 份参考论文 + grid_paper.md
```

**Stage-2 源树只剩两棵**：`TIGER_RQ-VAE`（上游 TIGER 基线）与 `TIGER_BEHAVIOR_SAMPLE_RQ-VAE`
（行为采样基线）。此前的 r4/r6/r7/r8/r9/r10、c1/c106/c107、seed43、behavior 等迭代树与其
结果目录、配套工具（`tools/r4_*`、`r5_*`、`r7_*`、`r8_*`、`tools/genrec_agents/`、
`tools/mvg/`、`tools/hyperbolic_verifier/`）已全部退役删除。**不要**再引用它们。

### 解释器

| 用途 | 解释器 |
|---|---|
| Stage-0 / Stage-1 / Stage-3 | `/home/wlia0047/ar57_scratch/wenyu/genrec_env/bin/python3.10` |
| CGV 六门 / Stage-2 探针 | `/home/wlia0047/ar57_scratch/wenyu/genrec_env_v2/bin/python3.9` |
| MCP server | 裸 `python3`（仅标准库，不得往任何 env 里装包） |

裸 `python3` 没有 torch，跑训练/评测一律用上表的解释器。

---

## 1. 参数全部硬编码，禁止 CLI

所有项目脚本的超参、路径、常量**直接写死在源码里**。禁止 argparse、sys.argv、环境变量覆写、
CLI flag。要改就改源码后重跑（无参）。仅 `/tmp/` 下的参考比较脚本可接受位置参数。

## 2. Stage-2 不设任何 gate

Stage-2 **没有**外层/内层 hard gate。`HitRate@50` 之类指标对 RQ-VAE 变体没有区分力，经验上会误杀
合法变体；trainer 内层 early-stop 也已删除，训练固定跑满 `EPOCHS = 3000`。所有指标只作为描述性
记录写进日志。裁决完全交给 Stage-3 的 `test_R@10`。

## 3. 产物路径（唯一权威）

```
Stage-0  results/stage0_build_parquet/{items,train,valid,test}.parquet
Stage-1  stage1_GeneEmbedding/output/{sentence_t5.npy,item_ids.json,emb_meta.json}   (源码树内，不进 results)
Stage-2  results/stage2_RQ-VAE/<ARM>/{rqvae_best.pth,sids_for_hgrec.npy,item_sids.json}
Stage-3  results/stage3_T5Train/<RQVAE_VARIANT>/{logs/,ckpt/}
```

Stage-2 产物**必须**落在 `results/stage2_RQ-VAE/<ARM>/`，源码树只留代码/`scripts/`/`logs/`/
输入副本，不得写 `.pth`/`.npy`/`item_sids.json`。目录名与 launcher 文件名一律不带 iter 编号。
迭代边界与回滚由 Git 承担。

## 4. Stage-3 冻结协议（人工核对，iteration 不得修改）

以下字段是 Stage-3 的权威锁定值（见 `stage3_T5Train/train_HG-Rec.py`）。任何不一致都不得启动
Stage-3。iteration launcher 只能改 SID/输出路径与 variant 元数据，**不得修改训练超参**。

```text
NUM_EPOCHS          = 150
EVAL_INTERVAL       = 5        (NO_EVAL=True 时不生效)
EVAL_START_EPOCH    = 75
EARLY_EVAL_EPOCHS   = (25, 50, 75, 100, 125)   仅记录走势
SEED                = 42
BEAM_SIZE           = 20
NO_EVAL             = True
SKIP_TEST           = False
```

**禁止**重新引入基于 train-loss 或 validation patience 的 early-stop。Stage-3 launcher 由
`train_HG-Rec.py` 内的 `_launch_via_torchrun` 自动 fork 4 卡 DDP
（`nproc_per_node=4`、`master_port=50201`、`NCCL_IB/P2P/SHM_DISABLE=1`）。
Stage-2 用 `master_port=50202`，两者不冲突。

## 5. 训练启动

```bash
# Stage-3（4 卡 DDP，自动 fork）
cd stage3_T5Train && /home/wlia0047/ar57_scratch/wenyu/genrec_env/bin/python3.10 train_HG-Rec.py

# Stage-2（4 卡 DDP，自动 fork；输出写死到 results/stage2_RQ-VAE/<ARM>/）
cd stage2_RQ-VAE/TIGER_RQ-VAE && /home/wlia0047/ar57_scratch/wenyu/genrec_env_v2/bin/python3.9 train_rqvae.py
```

Stage-2 trainer 的 `EMBEDDING_FILE` 硬编码指向 `stage1_GeneEmbedding/output/sentence_t5.npy`；
`results/stage2_RQ-VAE/_input/` 已随清理删除，若有脚本依赖它需重新从 Stage-1 产物重建。

## 6. 训练前的梯度通路检查

每次实际跑 Stage-2 训练前，必须在 1 个 ckpt + 1 个 batch 上验证所有相关参数的
`loss.backward()` 通路能拿到非零梯度：

1. `total_loss.requires_grad` 且 `total_loss.grad_fn is not None`；
2. 逐项新机制 loss（commitment / margin / spread / anisotropy / logdet / contrastive / Möbius 等）
   用 `torch.autograd.grad(loss_item, model.parameters(), retain_graph=True, allow_unused=True)`
   至少 1 个参数非零；
3. 检查 `self._last_*.detach()` 这类截断会不会让 backward 静默失效。

任一项 FAIL，禁止启动 GPU 训练，先修再验。

## 7. 机制准入：CGV 六门

Stage-2 训练**必须**先通过 CGV（Contract-based Verification）。声明是**调用参数**，不是文件：
机制不向 CGV 提交任何文件，契约随调用传入，报告中给出规范化哈希（`spec_sha256`）。

七组声明（`schema_version = 1`）：

```text
mechanism / model / forward_contract / gradient_contract /
update_contract / distributed_contract / runtime_contract
```

六门及其含义：

| 门 | 验证什么 |
|---|---|
| **G1** 前向契约 | 实测前向满足声明的公式与不变量（形状、取值范围、残差递推、有限性） |
| **G2** 梯度传播 | 声明的每条 loss 路径都以声明的性质（nonzero/zero/finite）到达声明的目标参数 |
| **G3** 梯度数学 | 实测梯度等于机制自己声明的规则，且机制确实改变了梯度 |
| **G4** 参数更新 | 声明的 optimizer 与梯度裁剪产生的更新等于独立闭式预测 |
| **G5** 分布式一致性 | 多卡实测符合声明的聚合规则 |
| **G6** 真实训练入口 | 正式入口执行了声明的机制、调用真实 optimizer，且至少一个有梯度参数实际改变 |

门的四种状态必须如实区分：`passed` / `failed` / `not_applicable` / `insufficient_evidence`。
**`insufficient_evidence` 绝不等于通过**；`passed` 为真才视为验证成功。

调用方式：

```
# MCP
cgv_check(source_dir, spec)      # 实跑六门
cgv_status(source_dir, spec?)    # 只读表：每门验什么、trainer 源码哈希、声明哈希
stage2_start(source_dir, spec)   # 先验六门，再派发训练

# 命令行（等价路径）
python3.9 tools/cgv/run_cgv.py <tree> --spec -      # JSON 从 stdin
```

强制约束：

1. **声明必填**：`cgv_check` 与 `stage2_start` 缺少 `spec` 一律拒绝。
2. **独立 Oracle**：oracle 只能引用验证器自己的 `oracles:`/`adapters:` 模块，不得是机制自己声明的
   entrypoint，否则该机制一律判 `insufficient_evidence`。
3. **G3 fail-closed**：`mode: formula` 必须给出 `oracles:` 引用。
4. **无持久化凭据**：CGV 不写证书、不留裁决文件；`stage2_start` 每次启动都重新验证一遍。
5. **禁止手工绕过**：`nohup` 直跑 `train_rqvae.py`、自建 launcher 等一切旁路一律禁止。
   `cgv_check` 返回 refused 时按其指出的问题修，不绕开。
6. **门失败即结论**：记录发现、换一个结构上不同的机制；禁止改门禁、放宽容差或改写指标来「通过」。
7. **声明必须如实**：契约写实测性质（如 encoder 无梯度就声明 `zero`），不得为了让门通过而写假话。

## 8. Git 分支与推送规则

- **唯一允许的远端**：`https://github.com/cpxjazz123/HiCurvRec.git`。任何其他地址都不得使用
  （`git remote -v` 必须只有上面这个 GitHub 仓库）。
- **只维护 `main`**：所有开发、提交、推送都在 `main`；不创建、不维护其他长期分支。
- **每次迭代结束必须 commit + push**：不得以「产物已在磁盘完成」代替 Git 同步；未 push 成功前不得
  宣称该 iter 闭环、不得启动下一轮改动。
- **禁止未经授权强推**：`git push --force` / `--force-with-lease` 一律禁止。
- **推送前后必须核验**：操作前确认 `git remote -v` 只有 GitHub URL；推送后比较
  `git rev-parse main` 与 `git ls-remote origin refs/heads/main` 的 commit hash，不一致不得继续。
- **LFS**：`.pth` 与 `.npy` 由 Git LFS 跟踪（`.gitattributes`）。历史大产物已用
  `git lfs prune --verify-remote` 清理过，需要时可从 GitHub 重新拉取。


## 9. 新增自研研究工具（2026-10-24）

`tools/research/` 是清理后新建、职责分层的研究工具，不恢复已退役的控制器。`core` 维护唯一持久 DAG 和证据，`agents` 直接使用 OpenCode CLI，`science` 从 train-only 交互做行为分析并验证候选，`workflows` 串联 DATA+LITERATURE→GEOMETRY→REVIEW，`integrations` 只读对接现有 CGV/MCP，`tests` 离线验证。完整目录见 `tools/research/README.md`。状态和研究输出只存 `results/research/issue116/`。所有脚本遵守 §1 的无参入口，真人 Agent 默认不开启，任何正式 Stage2/Stage3 均只能经现有 MCP 和 CGV 流程授权运行。

