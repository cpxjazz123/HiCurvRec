# Project Rules

## 1. No CLI args — hard-code all parameters

所有项目内脚本（stage1_GeneEmbedding / stage2_RQ-VAE / stage3_T5Train 等）的超参、路径、常量必须直接硬编码在源码里——禁止 argparse、sys.argv、环境变量覆写、CLI flag，要改就改源码后 `python <script>`（无参）重跑；仅 `/tmp/` 下的参考比较脚本可接受 SID 路径 + 标签位置参数。

## 2. Stage 2 SID Quality Gate

每次跑完 curvature stage2 RQ-VAE，必须用 `python /tmp/sid_metrics_any.py <sids.npy> "<label>"` 在新产出的 SID 上算 **Embedding HitRate@K=50** 一项硬指标，且必须优于 `results/stage2_RQ-VAE/TIGER_RQ-VAE/sids_for_hgrec_recbole.npy`，否则迭代重跑，禁入 stage3。

**Gini 类指标**（3-token SID Gini、每层 mean Gini）与 **collision rate** 只作为描述性指标记录到日志，**不作为 gate**：经验上 L0 极端塌陷的合法变体（iter11 sk_eps=0.5 是当前最佳，full_gini=0.1143 远高于 baseline 0.0672，collision rate 也偏高）会被这些 hard gate 误杀；改用 collapse-aware 的 l01_unique_pairs 与 H(L1|L0) 联合结构 gate（见 `curvature_RQ-VAE/modules/sid_quality.py`）。

## 3. stage2 curvature_RQ-VAE 启动：cd 到该目录后用 `nohup /home/wlia0047/ar57_scratch/wenyu/genrec_env_v2/bin/python3.9 curvature_RQ-VAE.py > logs/train_run.log 2>&1 &`（裸 `python3` 无 torch，会立刻 ModuleNotFoundError；脚本内置 `_launch_via_torchrun` 自动 fork 4 卡 DDP，训练输出落在 `logs/train_migrated.log`）。

## 4. 禁止进入 plan 模式

任何会话、任何任务（包括代码重构、规则修订、批量编辑等）都不得使用 EnterPlanMode / Plan agent，进入即视为违规，必须立即退出并直接执行；如确需事先规划，写简短要点到回复正文即可，禁止调用 Plan 工具链。

## 5. stage3 train_HG-Rec 启动：cd 到 stage3_T5Train 后用 `nohup /home/wlia0047/ar57_scratch/wenyu/genrec_env/bin/python3.10 train_HG-Rec.py > logs/_stage3_run.log 2>&1 &`（裸 `python3` 无 torch；脚本内置 `_launch_via_torchrun` 自动 fork 4 卡 DDP，硬编码 `nproc_per_node=4 / master_port=50201 / CUDA_VISIBLE_DEVICES=0,1,2,3`，启动前会注入 `NCCL_IB_DISABLE=1 / NCCL_P2P_DISABLE=1 / NCCL_SHM_DISABLE=1 / NCCL_TIMEOUT=3600 / TORCH_NCCL_BLOCKING_WAIT=1`，训练输出落在 `logs/_stage3_launcher.log`）。

## 6. stage2 实际训练前必做梯度通路检查

每次要实际跑 stage2 训练（不论是基线还是新增机制），必须先在 1 个 ckpt + 1 个 batch 上验证所有相关参数的 `loss.backward()` 通路能拿到非零梯度：(1) `total_loss.requires_grad` + `total_loss.grad_fn is not None`；(2) 逐项新机制 loss（commitment / margin / spread / anisotropy / logdet / contrastive / Möbius 等）`torch.autograd.grad(loss_item, model.parameters(), retain_graph=True, allow_unused=True)` 至少 1 个参数非零；(3) 检查 `self._last_*.detach()` 这类截断是否会让 backward 静默失效。若任一项 FAIL，禁止启动 GPU 训练，立即 R50 + 修复后再验证。

## 7. pip 重装 torch / nvidia 必须 `--target=env-site-packages --upgrade` 强制落点

`pip install torch` 在 genrec_env 下会被 `--user` 默认行为或 vllm/transformers resolver 约束把包装到 `/home/wlia0047/wenyu/.local/` 或强制回滚原版，导致 site-packages/torch 与 nvidia runtime 失配、`import torch` 报 `libcublas.so` 缺失。**唯一允许的写法**：`pip install --no-deps --ignore-installed --upgrade --target /home/wlia0047/ar57_scratch/wenyu/genrec_env/lib/python3.10/site-packages <pkg>==<ver>+<cu> --index-url https://download.pytorch.org/whl/<cu>`，必须同次装配套 nvidia-* runtime，否则 libcublas 找不到。任一步 FAIL 立即回滚到 torch==2.13.0+cu130 + cu13 nvidia 套件，绝不"先这样以后再修"。
