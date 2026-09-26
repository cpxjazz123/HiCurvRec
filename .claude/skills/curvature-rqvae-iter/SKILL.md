---
name: curvature-rqvae-iter
description: Iterate and optimize the variable-curvature RQ-VAE at /home/wlia0047/ar57/wenyu/GeneRec/stage2_RQ-VAE/curvature_RQ-VAE for the Amazon-2023 Instruments dataset. Each iteration clones the working directory, modifies only the RQ-VAE mechanics (NOT stage1 embeddings or stage3 T5 trainer), trains the new variant, evaluates SID quality with descriptive metrics only (no gate as of 2026-09-19), and promotes a variant to stage3 downstream training if its downstream test_R@10 > 0.065. The hard success target is downstream test metric > 0.065. Innovation focus is variable curvature + hyperbolic geometry (Poincaré ball / Lorentz / projective hyperbolic). Each completed iteration must commit and push mechanism code plus Stage2/Stage3 artifacts under results/ to GitHub origin/main before the iter is closed (GeneRec CLAUDE.md §8, §13).
---

# curvature-rqvae-iter

Iterate on the variable-curvature RQ-VAE pipeline. The deliverable is a promoted variant whose downstream GR (generative recommender) test metric exceeds **0.065** on the Amazon-2023 Instruments dataset, with valid-stage projected above **0.07**.

## Scope Boundaries (Hard Constraints)

- **Only edit mechanism code** under `/home/wlia0047/ar57/wenyu/GeneRec/stage2_RQ-VAE/curvature_RQ-VAE/` (promoted baseline) or the active clone `curvature_RQ-VAE_iter<N>/`. Do not hand-edit `results/` except via training/export scripts; **do** commit `results/stage2_RQ-VAE/curvature_RQ-VAE_iter<N>/` and `results/stage3_T5Train/curvature_RQ-VAE_iter<N>/` at iter close (CLAUDE.md §13).
- **Never edit** `/home/wlia0047/ar57/wenyu/GeneRec/stage1_GeneEmbedding/` (input embeddings).
- **Never edit** `/home/wlia0047/ar57/wenyu/GeneRec/stage3_T5Train/` (downstream trainer). Stage3 is invoked **read-only** after each SID promotion.
- **Never edit** the input `item_emb.parquet` or `Instruments.inter.json`. They are the immutable contract with stage1.

## Project Rules (mirror of /home/wlia0047/ar57/wenyu/GeneRec/CLAUDE.md)

Skill 在执行任何步骤前必须先遵守以下硬约束，违反即视为 NO-GO：

1. **No CLI args — hard-code all parameters.** 所有项目内脚本（stage1_GeneEmbedding / stage2_RQ-VAE / stage3_T5Train 等）的超参、路径、常量必须直接硬编码在源码里——禁止 argparse、sys.argv、环境变量覆写、CLI flag，要改就改源码后 `python <script>`（无参）重跑；仅 `/tmp/` 下的参考比较脚本可接受 SID 路径 + 标签位置参数。
2. **Stage 2 SID Quality Gate (2026-09-19 终态: 无任何 gate).** 不再设任何外层/内层 hard gate; 训练完直接进 stage3, 由 stage3 test_R@10 (硬目标 > 0.065) 决定是否采用. HitRate@K=50 已被证明不读 SID、对 RQ-VAE 变体无区分力 (对所有 curvature 变体恒等于 0.7165); Gini / collision / l01_pairs / H 等指标仅作描述性日志. trainer 内层 `should_early_stop` 始终返回 `(False, "")`.
3. **stage2 curvature_RQ-VAE 启动：** `cd` 到该目录后用 `nohup /home/wlia0047/ar57_scratch/wenyu/genrec_env/bin/python3.10 curvature_RQ-VAE.py > logs/train_run.log 2>&1 &`（裸 `python3` 无 torch，会立刻 ModuleNotFoundError；脚本内置 `_launch_via_torchrun` 自动 fork 4 卡 DDP，训练输出落在 `logs/train_migrated.log`）。
4. **禁止进入 plan 模式.** 任何会话、任何任务（包括代码重构、规则修订、批量编辑等）都不得使用 EnterPlanMode / Plan agent，进入即视为违规，必须立即退出并直接执行；如确需事先规划，写简短要点到回复正文即可，禁止调用 Plan 工具链。
5. **stage3 train_HG-Rec 启动（仅在 SID Gate PASS 后调用）：** `cd` 到 stage3_T5Train 后用 `nohup /home/wlia0047/ar57_scratch/wenyu/genrec_env/bin/python3.10 train_HG-Rec.py > logs/_stage3_run.log 2>&1 &`（裸 `python3` 无 torch；脚本内置 `_launch_via_torchrun` 自动 fork 4 卡 DDP，硬编码 `nproc_per_node=4 / master_port=50201 / CUDA_VISIBLE_DEVICES=0,1,2,3`，启动前会注入 `NCCL_IB_DISABLE=1 / NCCL_P2P_DISABLE=1 / NCCL_SHM_DISABLE=1 / NCCL_TIMEOUT=3600 / TORCH_NCCL_BLOCKING_WAIT=1`，训练输出落在 `logs/_stage3_launcher.log`）。
6. **stage2 实际训练前必做 4 层 Mechanism Verification Gate (MVG).** 每次要实际跑 stage2 训练（不论基线还是新机制），必须先在 1 个 ckpt + 1 个 batch 上按 4 层校验：(a) **Graph** — total_loss.requires_grad + total_loss.grad_fn 不为 None；(b) **Gradient** — 逐项新机制 loss（commitment / margin / spread / anisotropy / logdet / contrastive / Möbius / attention / curvature routing / gate 等）`torch.autograd.grad(loss_item, model.parameters(), retain_graph=True, allow_unused=True)` 每个 requires_grad 参数 grad L2 > 1e-12；(c) **Update** — 5 步后 optimizer 真更新至少 1 个新机制参数，相对变化 > 1e-7；(d) **Behavior** — 同 seed/batch 200 步 ON vs OFF 后 `|L_on - L_off|` > 1e-6，且机制直接输出（如 attention temperature、gate 分布、`std(c_i)`、`L_aux/L_total`）明显偏离恒等。任一层 FAIL → 禁止启动 GPU 训练，立即 R50 + 修复后重验。
   **§6 实施细则（避免反复犯错）**：常驻检查脚本**必须**落到 `scripts/mvg_check.py`（4 层版本）和 `scripts/grad_check.py`（layer-1+layer-2 兼容入口），禁止每次用 `python -c "..."` 临时 inline；调用方式 `PY=/home/wlia0047/ar57_scratch/wenyu/genrec_env_v2/bin/python3.9 python3 scripts/mvg_check.py <iter_id>`（脚本内部 `cd $NEXT`、加载数值超参用 `importlib.util.spec_from_file_location('rqtrain', '$NEXT/curvature_RQ-VAE.py')`）。**严禁**（1）`import curvature_RQ-VAE as m`——文件名含 `-` 不是合法 Python 标识符，必报 `SyntaxError`；（2）假设 `curvature_config.py` 暴露 `INPUT_DIM / USE_CYCLIC_CURVATURE / MIDPOINT_LAYER_MASK` 等数值超参——`curvature_config.py` **只**放路径常量，数值超参全部在 `curvature_RQ-VAE.py` 文件顶部；（3）通过复制粘贴数值到检查脚本——一旦后续调整常量就会与基线漂移。脚本输出 `MVG PASS` 才允许进入 step 3 启动训练；旧 `GRAD_CHECK PASS` 仅作为 layer-1+layer-2 子集 PASS。
7. **GitHub-only Git（CLAUDE.md §8、项目根 `SKILL.md`）**：`origin` 只能是 `https://github.com/cpxjazz123/HiCurvRec.git`；只在 `main` 上工作；禁止未授权 force push；每次 push 后 `git rev-parse main` 必须等于 `git ls-remote origin refs/heads/main`。
8. **每次迭代结束必须 commit + push 代码与产物（CLAUDE.md §13）**：Stage2（含 SID 导出）与 Stage3（含 `test_final.json`）跑完后，在**宣称 iter 完成、PROMOTE、NO-GO 清理、或启动 `i+1`** 之前，必须把本轮 **(1) `stage2_RQ-VAE/curvature_RQ-VAE_iter<N>/` 代码与 logs**、**(2) `results/stage2_RQ-VAE/curvature_RQ-VAE_iter<N>/`**、**(3) `results/stage3_T5Train/curvature_RQ-VAE_iter<N>/`** 一并提交并 `git push origin main`。仅磁盘上有 `test_final.json` 而未 push 成功 → iter **未完成**。

## Iteration Protocol (Loop Until Success or N Iterations)

每轮迭代必须严格按串行顺序执行：

```
A  literature-hunter  (logs/lit_search_iter${i}.md)
B  direction-judge    (logs/direction_decision_iter${i}.md)
C  hypothesis-designer (logs/hypothesis_iter${i}.md)
   ↓
GradCheck (scripts/grad_check.py → GRAD_CHECK_PASS)
   ↓
Stage2  (4 卡 DDP, MAX_GLOBAL_STEPS 100_000, ckpt every 10k)
   ↓
D  sid-geometry-analyst  (logs/sid_geometry_iter${i}.md)
   ↓
Export SID (4-token extension → item_sids_recbole.json)
   ↓
Stage3  (train_HG-Rec.py 150 epoch, beam=20)
   ↓
E  failure-analyst  (logs/failure_analysis_iter${i}.md + failed_mechanism_ledger.md)
   ↓
F  failure-attribution-auditor (logs/failure_attribution_iter${i}.md, 5 类标签)
   ↓
G  root-cause-gap-analyst (logs/iteration_bridge.md + gap 给下一轮 Agent A/B)
   ↓
Gate Decision (logs/gate_decision_iter${i}.md, hard target test_R@10 > 0.065)
   ↓
H  commit + push (代码 + results/stage2 + results/stage3 → origin/main, §13; gate 写入 Audit commit)
   ↓
PROMOTE 或 NO-GO cleanup（仅 push 核验通过后）
```

下一轮：

```
[上一轮 gate_decision]
   ↓
G  root-cause-gap-analyst  (在 Agent A 之前生成 iteration_bridge.md,
   主瓶颈 + forbidden next directions + 检索目标)
   ↓
A  literature-hunter  (只允许在 G 给出的 targeted bottleneck search 范围内检索)
   ↓
B  direction-judge   (必须新增 (e) Gap-closing relevance 维度,
                       任何不能明确解释"如何修复上一轮 dominant failure"的候选,
                       即使 novelty / literature 分数高, 也不能成为下一轮方向)
```

For each iteration `i = 1, 2, 3, ...`:

1. **Clone the current best working directory** to a new sibling:
   ```bash
   SRC=/home/wlia0047/ar57/wenyu/GeneRec/stage2_RQ-VAE/curvature_RQ-VAE
   NEXT=/home/wlia0047/ar57/wenyu/GeneRec/stage2_RQ-VAE/curvature_RQ-VAE_iter${i}
   cp -r "$SRC" "$NEXT"
   ```
   The previous iteration's directory is **read-only after the clone** — never edit it again.

   **1.1** **Unified Stage-2 output paths (mandatory, 2026-09-25 起生效).** 克隆后**必须立刻修改** `$NEXT/curvature_config.py` 的输出路径常量，让本轮产物**统一落在仓库级 `results/stage2_RQ-VAE/curvature_RQ-VAE_iter<N>/`**（目录名与 `stage2_RQ-VAE/curvature_RQ-VAE_iter<N>/` 完全同名同前缀，`MECHANISM_NAME` 保留完整 `iter<N>_<descriptive>` 字符串仅供 Stage3 派生 `RQVAE_VARIANT`，产物目录名固定短名 `curvature_RQ-VAE_iter<N>`），具体映射：`_CONFIG_DIR` → `/home/wlia0047/ar57/wenyu/GeneRec/results/stage2_RQ-VAE/curvature_RQ-VAE_iter<N>`；`RQVAE_OUT_DIR` → `$STAGE2_OUT_ROOT/out/rqvae/instruments/`；`SIDS_NPY` → `$STAGE2_OUT_ROOT/dataset/Instruments/sids_for_hgrec.npy`；`RAW_SIDS_NPY` → `$STAGE2_OUT_ROOT/out/rqvae/instruments/sids_raw.npy`；`ITEM_SIDS_JSON` → `$STAGE2_OUT_ROOT/item_sids.json`；改完后立刻 `mkdir -p "$STAGE2_OUT_ROOT/out/rqvae/instruments" "$STAGE2_OUT_ROOT/dataset/Instruments"` 并把 SRC 的 `item_emb.npy` 拷到 `$STAGE2_OUT_ROOT/dataset/Instruments/`；本步骤只允许编辑 `$NEXT/curvature_config.py` 的路径相关常量（不许动 SRC 的 mechanism / 数值超参 / 模型代码），iter 子树 `$NEXT/` 下只允许保留代码副本、`configs/`、`scripts/`、`logs/`、`__pycache__/` 与输入副本，**绝不允许**在 iter 子树内写 `.pth` / `.npy` / `item_sids.json` 或 `$NEXT/results/...` 任何子路径。

2. **One novelty per iteration.** Pick exactly one new mechanism from the candidate pool (see Innovation Focus) and apply it to `curvature_RQ-VAE.py` + `curvature_config.py`. Do not stack multiple novelties in one iteration.

### 2a. Agent C — Hypothesis Designer（在 Agent B 之后、GradCheck 之前）
- **唯一输入**：(a) `logs/direction_decision_iter${i}.md`（唯一推荐机制）；(b) `references/mechanism_pool.md` 当前候选清单；(c) Agent A 报告。
- **唯一任务**：把“推荐机制”翻译为可证伪的 Stage2 几何假设，并把假设拆分为：
  1. **直接效应（mechanism effect）**——机制本身应触发的可观察现象（例如 `temperature_scale` 应随 step 变化、`attention_logits` 应被 cyclic `c(t)` 联动、key 矩阵应有非零更新量）。
  2. **Proxy 假设（proxy hypothesis）**——Stage2 描述性指标的方向变化（`H(L1|L0)` 方向、`coarse-fine balance` 方向、`L0 oracle` 方向）。
- **输出**：`logs/hypothesis_iter${i}.md`，必须列出：
  - 机制 X 的直接效应（D 必须命中才算 ALIGNED）；
  - 机制 X 的 proxy 假设（D 仅作 PARTIAL / METRIC_MISMATCH 判定依据，不强制 NO-GO）；
  - 对照 iter11 `geom_L0_util` 等已有 fingerprint 的预期变化。
- **禁止**：出现“可能提升”“也许”“视情况而定”等不可证伪词；不得修改 Python。

3. **Run the iteration in the cloned directory.** Use the `train_iter.sh` helper script (see `scripts/`). It pins:

3. **Run the iteration in the cloned directory.** Use the `train_iter.sh` helper script (see `scripts/`). It pins:
   - Working directory = `$NEXT`
   - 解释器硬编码 `genrec_env` 的 python3.10（裸 `python3` 无 torch）；launcher 由 `curvature_RQ-VAE.py` 内置 `_launch_via_torchrun` 自动 fork 4 卡 DDP，NCCL 友好 env 已在 launcher 内注入。
   - Log to `$NEXT/logs/train_iter${i}.log`（wrapper）与 `$NEXT/logs/train_migrated.log`（launcher 子进程）
   - Output ckpt to **仓库级统一目录** `results/stage2_RQ-VAE/curvature_RQ-VAE_iter<N>/out/rqvae/instruments/`（由 step 1.1 重定向后的 `RQVAE_OUT_DIR`，目录名固定为 `curvature_RQ-VAE_iter<N>` 与 `stage2_RQ-VAE/` 下子目录同名）；启动前 `grep RQVAE_OUT_DIR $NEXT/curvature_config.py` 强制验证路径正确性。
   - `MAX_GLOBAL_STEPS` from config (default 100_000, ~8.5 min on 4 卡 DDP)

   唯一调用方式（0 CLI flag, 与 Project Rules §3 一致）：
   ```bash
   bash scripts/train_iter.sh ${i}
   # 等价于 (硬编码在脚本里):
   #   cd stage2_RQ-VAE/curvature_RQ-VAE_iter${i}
   #   nohup /home/wlia0047/ar57_scratch/wenyu/genrec_env/bin/python3.10 curvature_RQ-VAE.py \
   #       > logs/train_iter${i}.log 2>&1 &
   ```

   **3a. Mid-training early-stop judgement (mandatory, time-saving).** The trainer is allowed — even *encouraged* — to terminate a step 3 run **before** `MAX_GLOBAL_STEPS` if any of the following signals appear in the log tail (`tail -50 $NEXT/logs/train_iter${i}.log`):

   - **(a) Codebook collapse observed.** Per-layer unique codes drops below `0.85 × codebook_size` for ≥ 2 layers in any 500-step window (e.g. for 256 codes, L1 < 218 AND L2 < 218 simultaneously). The run is dead, do NOT wait it out.
   - **(b) Convergence plateau.** Both `loss` and `vl` (vq loss) do not improve by more than 1% over a 2k-step sliding window. Compare last 2000 steps vs the previous 2000 steps.
   - **(c) Step budget already sufficient.** If `MAX_GLOBAL_STEPS = 50_000` and the iteration reaches `step >= 30_000` with `unique codes ≥ 240/256` on every layer and `loss` is monotonically decreasing, the trainer is free to declare "30k is enough, the SID quality won't improve past this" and skip the remaining 20k steps.
   - **(d) Numerical instability.** NaN/Inf in any reported metric, `c` collapsing to 0 or exploding > 5, `margin` becoming negative, or `loss` flipping sign twice within 1k steps.
   - **(e) Time-budget hard cap.** If the iteration has already consumed > 35 minutes wall-clock on a single GPU and the loss curve looks no better than the current promoted baseline trajectory, stop. There is no value in running 50 more minutes for a 0.1% delta.

   **How to early-stop cleanly:**
   ```bash
   # Find the trainer PID
   pgrep -f "curvature_RQ-VAE.py" | head -1
   # SIGTERM first (let trainer save final ckpt if it has a signal handler); if not, SIGKILL
   kill -TERM <PID>; sleep 5; kill -KILL <PID> 2>/dev/null
   # Verify last partial ckpt exists
   ls -lt $NEXT/out/rqvae/instruments/rqvae_step*.pt | head -3
   ```
   Then export SIDs from the latest `rqvae_step*.pt` (the trainer saves every `CKPT_EVERY` steps so a partial ckpt is always available). If no partial ckpt exists (i.e. terminated before first `CKPT_EVERY`), fall back to `rqvae_final.pt` from the previous best iteration.

   **Mandatory log entry on early-stop:** append a one-line note to `$NEXT/logs/train_iter${i}.log` saying `EARLY-STOP TRIGGERED at step <N>: <reason>`, plus the wall-clock time and the saved ckpt that downstream will use. This goes into the gate decision file as the explanation for why `MAX_GLOBAL_STEPS` was not hit.

4. **Quick SID quality diagnostic (descriptive only, 2026-09-19 终态: 无任何 gate)**:

   优先调用 `/tmp/sid_metrics_any.py`（CLAUDE.md §2 钦点，Project Rules §1 例外：`/tmp/` 比较脚本可接受位置参数），但**不再作为 gate**——纯日志输出供迭代调试：
   ```bash
   python3 /tmp/sid_metrics_any.py \
       "$NEXT/dataset/Instruments/sids_for_hgrec.npy" \
       "iter${i}"
   ```
   4 项指标（HitRate@K=50、3-token SID Gini、per-layer mean Gini、collision rate）全部仅作为描述性日志，不触发任何 early-stop / 外层 gate。trainer 内层 `should_early_stop` 始终返回 `(False, "")`。所有候选无条件进 stage3，由 stage3 `test_R@10` 决定是否采用。

   辅助：`scripts/eval_sids.py`（本 skill 自带，硬编码路径 + 0 CLI flag）会同时打印 iter vs TIGER 4 指标的对比表，便于日志归档。
   ```bash
   ITER_ID=iter${i} ITER_DIR="$NEXT" python3 scripts/eval_sids.py
   ```
   
   Promotion requires **stage3 完整跑完** + `test_R@10 > 0.065`:
   - stage2 SID 描述性指标 (HitRate@K=50 / full_gini / per_layer / l01_pairs / H) 仅用于日志/调试
   - stage3 `test_R@10` 是唯一硬裁决指标
   - 解锁 12 iter 上限后所有候选均自动进入 stage3

5. **Agent D — SID Geometry Analyst（在 Stage2 完成后、`export_sids_for_stage3.py` 之前）**
- **唯一输入**：(a) `logs/hypothesis_iter${i}.md`；(b) Stage2 真实训练日志与 `quality_step*.json`；(c) `logs/failed_mechanism_ledger.md` 历史结构化失败记录；(d) 现有 `sid_metrics_collector/` 的几何指纹工具（不得修改）。
- **唯一任务**：分析新 SID 的层级结构，回答“这个机制到底让 SID 变成了什么”，并对照 Agent C 的预测给出三态判定：
  - `MECHANISM_FAIL` → 直接 NO-GO（不进入 Stage3）；
  - `PARTIAL` / `METRIC_MISMATCH` → 继续 Stage3，仅记录不强制；
  - `ALIGNED` → 优先 Stage3。
- **判定规则**（必须严格按 C 的“直接效应”清单逐项检查）：
  - 直接效应列表中**任一项反向或完全不发生** → `MECHANISM_FAIL`；
  - 直接效应均发生但 proxy 假设不达成 → `PARTIAL` / `METRIC_MISMATCH`；
  - 直接效应与 proxy 假设均达成 → `ALIGNED`。
- **输出**：`logs/sid_geometry_iter${i}.md`，必须包含：
  - `L0/L1/L2 utilization`、`H(L1|L0)`、`H(L2|L0)`、`coarse-fine ratio`、`prefix collision`、`L0 oracle`；
  - 与 iter11 baseline / 上轮 iter4 的指纹对照；
  - 三态判定与触发依据。
- **禁止**：不得修改 Stage2 源码或训练 ckpt；不得调用 Stage3 trainer；不得跳过直接效应检查直接给 ALIGNED。
- **硬 gate**：Agent D 给出 `MECHANISM_FAIL` → 本轮 NO-GO，禁止进入 Stage3；`PARTIAL / METRIC_MISMATCH / ALIGNED` → 继续 Stage3。

6. **Stage3 downstream evaluation** (unconditional after stage2 completes, unless Agent D → MECHANISM_FAIL):
   
   按 Project Rules §5（CLAUDE.md 已钦定的 stage3 启动方式）调用 `stage3_T5Train/train_HG-Rec.py`，Stage3 `trainer.LOG_PATH` / `SAVE_PATH` 必须解析到仓库级短名 `curvature_RQ-VAE_iter<N>/`（目录名与 `stage2_RQ-VAE/curvature_RQ-VAE_iter<N>/`、`results/stage2_RQ-VAE/curvature_RQ-VAE_iter<N>/` 完全同名同前缀；详见 Project Rules §11）：
   ```bash
   cd /home/wlia0047/ar57/wenyu/GeneRec/stage3_T5Train
   ITER_ID=iter${i} ITER_DIR="$NEXT" \
   nohup /home/wlia0047/ar57_scratch/wenyu/genrec_env/bin/python3.10 train_HG-Rec.py \
       > logs/_stage3_run_iter${i}.log 2>&1 &
   ```
   训练器硬编码 `nproc_per_node=4 / master_port=50201 / CUDA_VISIBLE_DEVICES=0,1,2,3`，自动 fork 4 卡 DDP，launcher 内注入 NCCL 友好 env；日志落在 `logs/_stage3_launcher.log`。stage3 训练器在迭代中**只读**，不要修改 train_HG-Rec.py。

7. **Agent E — Failure Analyst / Memory（在 Stage3 完成后、Agent F 之前）**
- **唯一输入**：(a) `logs/sid_geometry_iter${i}.md`；(b) Stage3 `test_R@10` / `valid_ndcg@20`；(c) `logs/hypothesis_iter${i}.md` 的预期失败原因集合；(d) `logs/failed_mechanism_ledger.md` 历史条目。
- **唯一任务**：把失败归到三类之一，并把结论写入结构化 ledger：
  - 类 1 **机制未生效**（Agent D 已记录为 `MECHANISM_FAIL`）；
  - 类 2 **机制生效但 SID geometry 方向错**（Agent D 为 `PARTIAL` / `METRIC_MISMATCH`，且与本类失败机制失败 fingerprint 高度重合）；
  - 类 3 **SID geometry 看起来对但 downstream 不吃**（Agent D `ALIGNED`，但 Stage3 `test_R@10 ≤ 0.065`）。
- **输出**：
  - `logs/failure_analysis_iter${i}.md`：归类 + 失败 fingerprint + 对照 iter11 的几何差距；
  - 追加 `references/failed_mechanism_ledger.md` 一行：`机制名 | 归类 | SID 几何指纹 | Stage3 test_R@10 | 同类累计次数`。
- **禁止**：不得修改 Python；不得替换 Stage3 输入协议；不得把“描述性指标改善”当成几何方向正确。
- **硬约束（ledger 喂给下轮 Agent A）**：
  - 当 `同类机制 + 同类失败原因 ≥ 2 次`，下轮 Agent A **默认禁止**再次把它列为候选；
  - 例外：新方案必须**结构性改变导致过去失败的核心因果假设**，且必须由 Agent B 复核例外是否成立；
  - 例外成立时，Agent A 必须显式写出：过去为什么失败 / 这次改了哪个关键假设 / 为什么不算简单重复。

8. **Agent F — Failure Attribution Auditor（在 Agent E 之后、`gate_decision` 写入之前）**
- **唯一任务**：把一次失败的根因**拆开**：是机制本身无效，还是实现/接线/训练过程出了问题？Agent E 已经给出三段事实，但只有 4 个硬条件**全部**满足才能把本次失败正式记成机制失败并写进 ledger：
  1. **实现正确（Implementation Auditor）**：前面 `MVG PASS` 的 4 层（Graph/Gradient/Update/Behavior）已通过；counterfactual rollback test 显示同一代码路径关闭机制后能恢复到 baseline；
  2. **机制激活**：Agent C 预注册的**直接效应**（DE-1~3）在 Agent D 中确实发生；
  3. **几何方向匹配**：Agent D 报告为 `ALIGNED`，或 `PARTIAL` 但与机制预期方向一致；
  4. **管线一致**：Stage3 输入 SID、export pipeline、Stage3 配置与 iter4 baseline 完全一致。
- **5 类失败标签（必选其一）**：
  - `IMPLEMENTATION_FAIL`：代码或数学实现有问题（违反条件 1）；不能判机制失败，不写 ledger。
  - `ACTIVATION_FAIL`：代码正确，但机制实际上没产生足够影响（违反条件 2）；不能判机制失败，不写 ledger。
  - `GEOMETRY_MISMATCH`：机制激活但没有产生 Agent C 预期的 SID 结构（违反条件 3）；不能判机制失败，不写 ledger。
  - `PIPELINE_FAIL`：SID/export/Stage3 接口或实验协议存在问题（违反条件 4）；不能判机制失败，不写 ledger。
  - `TRUE_MECHANISM_FAIL`：4 个条件全部满足——实现正确、机制激活、几何变化符合预期、下游无获益。**只有此标签**允许写进 `failed_mechanism_ledger.md` 作为下轮 Agent A 的硬约束。
- **counterfactual rollback test**（推荐执行）：保留本轮代码与环境，仅把新机制关掉（用 MVG layer-4 的 ON/OFF 路径），再跑一次 Stage2→Stage3；若 OFF 恢复到 baseline 而 ON 确实产生了预期 SID 变化但 `test_R@10` 下降，则证据链最强，直接进入 `TRUE_MECHANISM_FAIL`。
- **输出**：`logs/failure_attribution_iter${i}.md`，必须包含：5 类标签判定 + 4 个条件逐项证据 + 是否触发 counterfactual rollback + 是否写入 ledger。
- **禁止**：不得修改 Python；不得跳过条件直接判 `TRUE_MECHANISM_FAIL`；不得把 `IMPLEMENTATION_FAIL` 误记为机制失败而污染 ledger。

9. **Agent G — Root-Cause & Gap Analyst（在 Agent F 之后、`gate_decision` 之前；下一轮 Agent A 之前也必须再次产出 iteration_bridge 增量）**
- **唯一输入**：(a) `logs/hypothesis_iter${i}.md` 原始假设；(b) `mvg_check.py` 输出（Implementation Auditor）；(c) `logs/sid_geometry_iter${i}.md`（Agent D）；(d) Stage3 `test_R@10` / `valid_ndcg@20` / 关键指纹；(e) `failed_mechanism_ledger.md`；(f) 当前最佳 baseline iter 的指标。
- **唯一任务**：回答“为什么失败 → 距离目标差在哪里 → 下一轮只解决哪个瓶颈”，并强制要求只选一个 **Dominant Failure Cause**。
- **固定输出结构**：
  1. **机制是否成功执行**：Implementation / Activation / Geometry alignment 三态 PASS/FAIL（Implementation 由 MVG 输出，Activation 由 Agent D DE-1~3 判定，Geometry 由 Agent D 状态决定）；
  2. **与最佳 baseline 差分**：`ΔU_L0`、`ΔH(L1|L0)`、`ΔH(L2|L0)`、`Δcollision`、`Δoracle`、`Δtest_R@10`；
  3. **Dominant bottleneck**：**只能写一句**；禁止"可能 A/B/C 都有问题"；明确禁止方向（forbidden next directions）；
  4. **Next iteration objective**：可证伪的目标（条件 + 数值区间 + 禁止条件）。
- **输出文件**：`logs/iteration_bridge.md`，连接本轮与下一轮。下一轮 Agent A 在检索前必须先读 `git log -1 -- logs/iteration_bridge.md` 与最近一次 `failed_mechanism_ledger.md` 条目，然后只搜索能修复上一轮 dominant failure 的机制。
- **硬约束（写入项目规则）**：任何下一轮机制候选，如果不能明确解释"它如何修复上一轮 dominant failure"，禁止进入候选池；该约束在 Agent B 的 (e) 评分维度"Gap-closing relevance"上强制执行。

10. **Decision** (stage3 完整跑完后, Agent F 给出 5 类标签之后):
   - 先写完 `gate_decision_iter${i}.md`（GO / NO-GO + `test_R@10` 数值），再执行 **Commit Discipline（§13）**：代码 + Stage2/Stage3 `results/` 必须已 push 且 hash 核验通过。
   - If `test_R@10 > 0.065` (硬目标) → **PROMOTE**: copy the iteration directory back to `curvature_RQ-VAE/` (overwrite), **再 commit + push promotion 变更**, log promotion. Stop the loop.
   - If `test_R@10 ≤ 0.065` → **archive iteration** (`rm -rf` `stage2_RQ-VAE/curvature_RQ-VAE_iter${i}/` **only after** audit push; **`results/.../curvature_RQ-VAE_iter${i}/` 保留在 Git 历史中，不得删除**), continue with `i+1`.
   - (历史决策规则已废弃: valid_ndcg@20 阈值和 SID gate 失败分支均已删除, 因 stage2 已无任何 gate.)

## Innovation Focus: Variable Curvature + Hyperbolic

The novelty MUST touch curvature geometry. Candidate mechanisms (rotate through these, do NOT stack):

- **Cyclic c(t) curriculum** — `c(t) = c_min + (c_max - c_min) * |sin(πt/T)|`, period T ∈ {25k, 50k, 100k}.
- **Layer-wise learned curvature** — each RQ layer has its own `c_l`, learned via reparameterization (sigmoid) with momentum-restricted drift.
- **Lorentz (hyperboloid) model instead of Poincaré ball** — replace `exp_0` / `log_0` with `exp_x^c` / `log_x^c` in the Lorentz formulation. Keep `c` learnable.
- **Projective hyperbolic (Hesse/Poincaré half-plane)** — alternative manifold model with Möbius addition.
- **Adaptive margin loss** with curvature-dependent target: `margin = α / c_l` (sharper separation at high c).
- **Negative c (spherical interpolation)** — partially spherical, partially hyperbolic for items that lie in low-density regions.
- **Mixed-curvature product manifold** — `M = H^{c1} × H^{c2} × S` (product of hyperbolic + spherical).
- **Curriculum on quantization difficulty** — anneal codebook temperature by `1/c`.
- **Riemannian Adam** instead of plain Adam (gradient rescaling by inverse metric `1/c`).
- **Sinkhorn-Knopp regularization on codebook with c-dependent epsilon** `ε = ε_0 / c`.

When picking, the priority is: (a) mechanisms with prior success in published papers, (b) mechanisms that interact with the existing curriculum rather than orthogonal axes. **Do not invent novel loss terms** unless they directly modulate curvature behavior.

## Literature Search (Mandatory Each Iteration — 双 Agent 角色分工)

每次迭代 step 2 设计机制之前，**必须并行派 2 个 subagent**，角色严格分离、互不交叉：

### Agent A — `literature-hunter`（**只**负责搜索，绝不评判）
- **唯一输入**：(a) 上一轮失败原因（从 `$PREV/logs/gate_decision_iter${i-1}.md` / `$PREV/logs/sid_quality_iter${i-1}.json` 抽取；(b) `references/mechanism_pool.md` 当前候选清单。
- **唯一任务**：围绕"上一轮失败的根因"精准搜索，**禁止**泛 query（如纯 `"hyperbolic recommender"`）或扩散到 5 个无关方向。
- 必查 query 模板（按失败原因挑选，**不限定**清单）：
  - 失败类 A（codebook collapse / utility 低）→ `"<manifold> codebook collapse remedy"`、`"residual quantization utility diversity"`
  - 失败类 B（silent no-op / gradient 截断）→ `"STE backward differentiable quantization"`、`"commitment loss gradient flow"`
  - 失败类 C（test_R@10 regress）→ `"<失败 mechanism 名> recommender ablation"`、`"<失败 mechanism 名> downstream task improvement"`
  - 失败类 D（Stage 1 端纯曲率变更被 Stage 3 T5 表征锁死 → 历史 ceiling 反复被证伪）→ `"<embedding model> + RQ-VAE"`，`"<text encoder> + generative recommender tokenization"`
  - 失败类 F（数值不稳定 / artanh saturation）→ `"artanh saturation hyperbolic recommender"`、`"fixed c curvature recommender"`
- 输出（落到 `$NEXT/logs/lit_search_iter${i}.md`）：(1) 每个 query 的 top-3 命中；(2) 候选编号 P1/P2/P3（≥3 个候选机制）；(3) 每个候选 1-2 句"为什么可能解决上轮失败"。
- **禁止**：Agent A 输出任何"建议选哪个 / 哪个最优"的评判；评判属于 Agent B。

### Agent B — `direction-judge`（**只**负责评判方向合理性，绝不搜索）
- **唯一输入**：(a) 上轮失败原因与 Agent G `logs/iteration_bridge.md` 的 dominant bottleneck；(b) Agent A 输出的 `$NEXT/logs/lit_search_iter${i}.md`；(c) 当前"机制池"清单（已尝试 + 候选）；(d) `references/failed_mechanism_ledger.md`。
- **唯一任务**：基于 Agent A 提供的候选清单 + 已尝试清单，按下表评分并**给出唯一推荐**：
  - (a) 是否对上轮失败根因有**明确**修复路径（必须能 1 句话说清"为什么能修"）？
  - (b) 是否在 2023+ 论文中有实证（不能仅靠经验/直觉）？
  - (c) 与现有 curriculum 的兼容性（与 cyclic / midpoint 冲突越少越好）。
  - (d) Stage 1 端纯曲率变更 ceiling 风险评估（已多次证伪失败）：若仍属"Stage 1 端纯曲率变更"，必须给出 WARNING 并建议改走 Stage 0 embedding 创新。
  - (e) **Gap-closing relevance（强制）**：候选必须**明确**解释它如何修复上一轮 `iteration_bridge.md` 中的 dominant bottleneck；不能修复则直接淘汰，不论其它维度分数多高。已被 `failed_mechanism_ledger.md` 标记 `TRUE_MECHANISM_FAIL` 的同类方向若不满足结构性 exception，禁止再次成为推荐。
  - (f) 创新与曲率相关性合规：必须命中以下 6 类中 ≥1 类（曲率 c 在训练中动态变化 / per-item·per-layer·per-codebook 异质曲率 / manifold 几何替换 / Riemannian 优化器 / 几何变换 / 双曲几何损失），且论文标题/abstract 命中 `geometric` / `hyperbolic` / `manifold` / `Riemannian` / `curvature` / `distance metric` 中 ≥1 关键词。
- 输出（落到 `$NEXT/logs/direction_decision_iter${i}.md`）：评分表 + 最终推荐 1 个机制 + 推荐理由（3 行以内）+ 若有多个备选请标 `BACKUP`。
- **禁止**：Agent B 调用任何 `web_search` / `WebFetch` / 读论文外部资料；所有判断依据只来自 Agent A 的输出与项目内已有产物；不得让 (e) 不通过的候选成为推荐。

### 硬约束
- **两 agent 必须串行**：先 Agent A 落盘 `lit_search_iter${i}.md`，再 Agent B 读该文件后落盘 `direction_decision_iter${i}.md`，父 agent 不允许跳过任一环节。
- 缺 `direction_decision_iter${i}.md` 即视为 NO-GO，禁入 step 3 训练。
- 若 Agent A 输出 < 3 个候选，父 agent 必须重新派 Agent A 补搜索，不得让 Agent B 在 < 3 个候选下决断。
- 若 Agent A 与 Agent B 推荐不一致，**以 Agent B 为准**（评判是 Agent B 的职责），但必须在 `direction_decision_iter${i}.md` 末尾记录差异与最终理由，便于下一轮 Agent A 针对性补搜。

## Hard Targets

- **Final target**: downstream `test_R@10 > 0.065` on Amazon-2023 Instruments.
- **Valid-stage projection**: `valid_ndcg@20 >= 0.07` (typical valid→test drift ~−15%, so 0.07 → ~0.06-0.07 test).
- **SID quality floor (2026-09-19 终态: 无 gate)**: 已全部删除, 无任何外层/内层 hard gate. 所有候选无条件进 stage3, 由 stage3 test_R@10 裁决.
- **Iteration cap**: 12 iterations. If after 12 iterations the target is not met, halt and surface a NO-GO summary.

## Files Layout Per Iteration

```
/home/wlia0047/ar57/wenyu/GeneRec/stage2_RQ-VAE/curvature_RQ-VAE_iter${i}/     (代码副本, git 可访问, 不再写产物)
├── curvature_RQ-VAE.py            (modified — exactly one new mechanism)
├── curvature_config.py            (modified — new mechanism constants + 路径重定向到 results/stage2_RQ-VAE/<MECHANISM_NAME>/)
├── configs/                        (unchanged unless mechanism requires gin update)
├── dataset/                        (input — unchanged, cloned)
└── logs/
    ├── train_iter${i}.log
    ├── lit_search_iter${i}.md     (Agent A 搜索节流: query + top-3 + ≥3 候选)
    ├── direction_decision_iter${i}.md  (Agent B 评判节流: 评分表 + 唯一推荐)
    ├── sid_quality_iter${i}.json  (FORGE metrics)
    ├── gate_decision_iter${i}.md  (GO / NO-GO)
    └── train_migrated.log         (DDP launcher 子进程 stdout/stderr)

/home/wlia0047/ar57/wenyu/GeneRec/results/stage2_RQ-VAE/curvature_RQ-VAE_iter<N>/  (Stage2 产物, 必须入库; item_sids.json, sids_for_hgrec.npy, rqvae_best.pth, …)

/home/wlia0047/ar57/wenyu/GeneRec/results/stage3_T5Train/curvature_RQ-VAE_iter<N>/  (Stage3 产物, 必须入库; logs/.../test_final.json, training_metrics.jsonl, ckpt/.../HG_Rec_best.pth, _stage3_launcher.log)
```

## Commit Discipline（与 CLAUDE.md §13 一致，迭代完成的硬门槛）

**迭代完成** = Stage2 + Stage3 跑完 + Agent 审计文件落盘 + **下列三类路径均已 commit 且 `git push origin main` 成功**。

### 必须纳入版本库（三类，缺一不可）

| 类 | 路径 |
|----|------|
| 机制代码与 iter 日志 | `stage2_RQ-VAE/curvature_RQ-VAE_iter${i}/`（含 `scripts/run_stage3_iter${i}.py` 等） |
| Stage2 产物 | `results/stage2_RQ-VAE/curvature_RQ-VAE_iter${i}/`（至少 `item_sids.json`、最终/best ckpt、`sids_for_hgrec.npy`；`rqvae_step_*.pt` 过大可只保留 best+末 step，须在 commit message 说明） |
| Stage3 产物 | `results/stage3_T5Train/curvature_RQ-VAE_iter${i}/`（`test_final.json`、`training_metrics.jsonl`、`HG_Rec_best.pth`、launcher 日志） |

另须纳入同一 commit 或紧邻 follow-up commit：(a) Agent C/D/E/F/G 的 `logs/*.md`；(b) `references/failed_mechanism_ledger.md` 仅 `TRUE_MECHANISM_FAIL` 追加；(c) `gate_decision_iter${i}.md` 含 **Audit commit** 哈希。不得只提交日志而漏 `results/`；不得只提交 metrics 而漏 mechanism 源码。

下轮 Agent A 在检索前必须 `git log -1 -- logs/iteration_bridge.md && git log -1 -- references/failed_mechanism_ledger.md`；Agent B 强制执行 Gap-closing relevance (e)。

### 执行脚本（GO 与 NO-GO 相同；必须在 PROMOTE / `rm -rf` iter 代码目录 / 启动 `i+1` 之前跑通）

```bash
cd /home/wlia0047/ar57/wenyu/GeneRec
git remote -v   # 仅 https://github.com/cpxjazz123/HiCurvRec.git
git branch --show-current   # 必须为 main

ITER="${i}"   # 本轮编号, 如 14
CODE="stage2_RQ-VAE/curvature_RQ-VAE_iter${ITER}"
S2="results/stage2_RQ-VAE/curvature_RQ-VAE_iter${ITER}"
S3="results/stage3_T5Train/curvature_RQ-VAE_iter${ITER}"

git check-ignore -v "$S2/item_sids.json" "$S3/logs" 2>/dev/null || true
# 若被 .gitignore 误排除, 修正 ignore 后再 add; 不得静默跳过产物

git add "$CODE/" "$S2/" "$S3/"
# 若 ledger / references 有变更: git add references/failed_mechanism_ledger.md

git commit -m "iter${ITER}: <mechanism-name> | test_R@10=<value> (PASS|FAIL) | stage2+stage3 results"
AUDIT_COMMIT="$(git rev-parse HEAD)"
printf '\n- Audit commit: `%s`\n' "$AUDIT_COMMIT" >> "${CODE}/logs/gate_decision_iter${ITER}.md"
git add "${CODE}/logs/gate_decision_iter${ITER}.md"
git commit -m "iter${ITER}: record audit commit hash"

git push origin main
LOCAL_HEAD="$(git rev-parse main)"
REMOTE_HEAD="$(git ls-remote origin refs/heads/main | awk '{print $1}')"
test "$LOCAL_HEAD" = "$REMOTE_HEAD"
printf 'ITER%s_COMMIT_PUSH_PASS %s\n' "$ITER" "$LOCAL_HEAD"
```

`AUDIT_COMMIT` 不得为 `pending` / `TBD`；hash 不一致 → iter **未完成**，禁止 NO-GO 删目录、禁止下一轮、禁止对用户声称「iter 闭环」。禁止 `git push --force`。PROMOTE 时覆盖 `curvature_RQ-VAE/` 的额外 commit 也需 push 并再次核验 hash。

## Cleanup Between Iterations

Once an iteration is archived as NO-GO **and** `ITER${i}_COMMIT_PUSH_PASS` 已打印:
- `rm -rf` **仅** `stage2_RQ-VAE/curvature_RQ-VAE_iter${i}/`（代码副本；无本地备份）。
- **不得** `rm -rf` `results/stage2_RQ-VAE/curvature_RQ-VAE_iter${i}/` 或 `results/stage3_T5Train/curvature_RQ-VAE_iter${i}/`（产物只留在 Git + 磁盘，供对照与复现）。
- The active iteration directory under `stage2_RQ-VAE/` is always the **latest one**; previous best lives in `curvature_RQ-VAE/` until promoted.
- 审计轨迹 = Git 历史（含已 push 的 `logs/` + `results/`）；删代码目录前必须已完成 Commit Discipline。

## Pre-Flight Before Each Iteration

Before launching training:
1. `nvidia-smi` — confirm GPU util < 10% and mem < 5GB on the chosen GPU.
2. `git status` — working tree clean **or** only the new `iter${i}` tree in progress; **上一轮**必须已在 `origin/main`（`git rev-parse main` == `git ls-remote origin refs/heads/main`）且 `gate_decision_iter${i-1}.md` 含有效 Audit commit。
3. `cat $NEXT/curvature_config.py | grep MECHANISM_NAME` — confirm mechanism is switched.
4. Check that `MAX_GLOBAL_STEPS` is set and `CKPT_EVERY` is reasonable.

## Stopping Conditions

Stop the loop and surface a summary when:
- (a) Downstream `test_R@10 > 0.065` AND `valid_ndcg@20 >= 0.07` achieved → **SUCCESS**.
- (b) 12 iterations exhausted without success → **HARD STOP**, write summary of best 3 iterations by valid_ndcg@20.
- (c) Same mechanism tried with 3 different hyperparameter settings, all NO-GO → drop the mechanism from the candidate pool.
- (d) **Agent D `MECHANISM_FAIL` 触发**：机制直接效应完全未发生，禁止再尝试该方向；写入 ledger 并立即开始下一轮。
- (e) **Ledger blocked trigger**：同一 `机制名 + 失败原因组合 ≥ 2 次` 后，第 3 次默认 NO-GO；必须由 Agent B 复核是否构成结构性 exception 才能继续。



### Stop Triggers (Mid-Iteration, Skill-Local)

- (d) **Codebook collapse**: per-layer unique < 85% × codebook_size in ≥ 2 layers → SIGTERM, do NOT complete the step budget.
- (e) **Loss plateau**: last 2k steps Δloss < 1% → SIGTERM, ckpt at the plateau is enough.
- (f) **Step already sufficient**: ≥ 30k steps with healthy unique + monotone loss → SIGTERM, no need to drain to MAX_GLOBAL_STEPS.
- (g) **Time budget exceeded**: > 35 min wall-clock with no downstream-projected improvement → SIGTERM.

These triggers OVERRIDE the "run to MAX_GLOBAL_STEPS" default. The default `MAX_GLOBAL_STEPS` is an upper bound, not a target. Always prefer to ship a clean 30k-step run + 1 quality check over a 50k-step run with no quality check at all.

## Reference Files

- `references/baseline_metrics.md` — baseline numbers for descriptive logging (Gini, l01_pairs, H, valid_ndcg@20, test_R@10; 2026-09-19 起 HitRate 不再作 gate).
- `references/mechanism_pool.md` — the 10 candidate mechanisms with paper citations.
- `scripts/train_iter.sh` — Project-Rules-compliant helper: 用 `genrec_env` 的 `python3.10` + `nohup ... curvature_RQ-VAE.py` 启动 4 卡 DDP（脚本内置 `_launch_via_torchrun`），仅接受位置参数 `<iter_id>`。
- `scripts/eval_sids.py` — Project-Rules-compliant FORGE-style SID 评估器：硬编码路径、0 CLI flag、对当前 iter 与 TIGER baseline 算 4 指标并打印对比表。
