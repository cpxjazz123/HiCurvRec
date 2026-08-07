# ETEGRec 迁移复现 verdict — Amazon 2018 Musical_Instruments

**日期**: 2026-08-07
**论文**: ETEGRec — "Generative Recommender with End-to-End Learnable Item Tokenization" (SIGIR'25, RUCAIBox/BishopLiu)
**上游 repo**: https://github.com/BishopLiu/ETEGRec
**目标**: 从论文默认 Amazon 2023 Scientific 迁移到我们的 Amazon 2018 Musical_Instruments (24772 users / 9922 items), 与 HG-Rec Task #84 基线对比
**结论**: **NO-GO** — test R@10=0.0763 vs 基线 0.1024 (**-25.5%**)

---

## 最终指标

| 指标 | Valid (best E17) | Test (best E17) | HG-Rec 基线 test | Δ vs 基线 |
|---|---|---|---|---|
| R@1 | 0.025553 | 0.021516 | — | — |
| R@5 | 0.077749 | 0.059543 | 0.0819 | **-27.3%** |
| R@10 | 0.096884 | **0.076336** | **0.1024** | **-25.5%** |
| NDCG@5 | 0.052992 | 0.041173 | — | — |
| NDCG@10 | 0.059155 | 0.046597 | 0.0755 | **-38.3%** |

**valid/test ratio (R@10)** = 0.096884 / 0.076336 = **1.269** (过拟合显著; HG-Rec v78 = 1.225, v74 = 1.234)

---

## 4-Gate 审计

### Gate 1 — 数据迁移与语义 embedding: **PASS**

**状态**: PASS
**关键数据**:
- 数据源: `/fs04/ar57/wenyu/DIGER/dataset/instruments/` (与 DIGER/DECOR 复现完全同源, 保证可比)
- symlink 到 `ETEGRec/dataset/instruments/`: `instruments.{train,valid,test}.jsonl` + `instruments.emb_map.json`
- item 数: 9923 (1 PAD + 9922 items, IDs 0..9922) — 与 HG-Rec 基线 9922 items 一致
- 语义 embedding: `instruments_emb_256.npy`, shape **(9922, 256)** float32, mean abs 0.0146, 无 NaN/Inf, 9.69 MB
- 生成方式: 本地 `sentence-transformers/sentence-t5-base` (offline cache `fc5d4628481a...`), 文本模板 `title:..., description:..., brand:..., categories:...`, mean-pooling 768d → 取前 256 维 (ETEGRec `semantic_hidden_size=256` 要求)

**无失败**: embedding 生成后校验通过 (shape/dtype/NaN 全 PASS)。

**verdict 路径**: 本文件
**commit hash**: 见下方

---

### Gate 2 — RQ-VAE 预训练 tokenizer: **PASS**

**状态**: PASS
**关键数据**:
- 训练: 3000 epochs, ~7 min (单卡 A100)
- best ckpt: `rqvae_ckpt/instruments/Aug-07-2026_14-12-17/best_collision_model.pth` (epoch 2049)
- **best collision_rate = 0.0956** (~9.6% 冲突, 论文级健康范围)
- train_loss 0.0051 → 0.0060, recon_loss 0.0008 → 0.0005 (recon 单调下降)
- 部署到 `dataset/instruments/256-256-256-128.rqvae.pth` (15 tensors: encoder / decoder / 3× VQ layers)
- 主训练加载后 code 使用率 (Epoch -1 初始): **level 1 = 232/256, level 2 = 256/256, level 3 = 256/256** codes
- code balance: 0.625 / 0.6805 / 0.6691
- code conflict: Max=23, Min=1, Type=8973, **90.44%** items 有唯一 3-digit code

**无失败**: tokenizer 质量健康, 非塌缩。

---

### Gate 3 — 端到端训练收敛: **PASS**

**状态**: PASS (训练本身收敛正常, 无 NaN / 无塌缩 / loss 单调)
**关键数据**:
- 环境: **单卡 A100-SXM4-40GB** (注意: 本 host `m3u009` 只有 1 张 GPU, 非 CLAUDE.md 描述的 4× L40S)
- 配置: bf16 mixed precision + `batch_size=128` + `gradient_accumulation_steps=4` (effective batch = 512, 与论文默认等价) + `eval_batch_size=16` + `num_workers=4`
- alternating optimization: `cycle=2` (REC epoch / ID epoch 交替), `warm_epoch=10` (前 10 epoch 关闭对齐损失)
- 单 epoch 耗时: **REC epoch ~42s / ID epoch ~100s** (cycle ~143s)
- `code_loss` 轨迹: 5.6134 (E0) → 3.6016 (E1) → 2.8798 (E2) → 2.5700 (E4) → 2.4834 (E8) → 2.4322 (E10) → 2.3435 (E18) → 2.3626 (E26) — **单调下降后进入 2.36-2.60 平台期**
- `kl_loss`: 0.0061 → 0.0003 (收敛)
- `dec_cl_loss`: 9.796 → 8.63 (warm_epoch=10 后从 ~9.9 掉到 ~8.6, 证明对齐损失确实在 E10 后生效)

**val NDCG@10 完整轨迹** (每 2 epoch eval):

| Epoch | 1 | 3 | 5 | 7 | 9 | 11 | 13 | 15 | **17** | 19 | 21 | 23 | 25 | 27 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| NDCG@10 | .0485 | .0554 | .0554 | .0556 | .0533 | .0557 | .0556 | .0543 | **.0592** | .0538 | .0512 | .0489 | .0501 | .0546 |

**best = Epoch 17 (val NDCG@10 = 0.059155, val R@10 = 0.096884)**

**终止决策**: Epoch 17 之后连续 5 次 eval (E19/21/23/25/27) 均未超越 best, val 从峰值明确回落 (0.0592 → 0.0489 最低), 同时 train `code_loss` 已在平台期震荡 → **判定过拟合**, 按 R23 (7 信号之一: val loss 反向) 于 epoch 27 手动终止, 未等 `early_stop=15` 触发的 epoch 47。GPU 已释放 (0 MiB)。

---

### Gate 4 — 与基线对比: **FAIL**

**状态**: **FAIL**
**关键数据**: test R@10 = **0.076336** vs HG-Rec 基线 **0.1024** → **-0.0261 (-25.5%)**

**失败原因分析**:

1. **valid/test gap 过大 (ratio 1.269)** — 显著高于我们所有 HG-Rec 变体 (v74 1.234 / v77 1.215 / v78 1.225)。说明模型在 valid 上过拟合到 leave-one-out 的倒数第二个交互, 泛化到最后一个交互 (test) 时崩塌。

2. **收敛过早 (E17/400)** — 论文默认 400 epochs, 我们在 27 epoch 就过拟合。ETEGRec 的 end-to-end 联合优化 (RQ-VAE tokenizer 与 T5 同时更新) 在 9922 items 的小数据集上参数量相对过剩, 而 Scientific (论文默认) item 数更多, tokenizer 有更多信号约束。

3. **语义 embedding 维度截断损失** — ETEGRec 硬要求 `semantic_hidden_size=256`, 而 sentence-t5-base 输出 768 维。我们取前 256 维, 丢弃了 2/3 的语义信息。论文原始 Amazon 2023 pipeline 用的是 256 维原生 embedding (不同 encoder), 这是**迁移引入的信息损失**, 也是最可能的主因。

4. **effective batch 等价但优化动态不同** — 我们用 `batch=128 × accum=4 = 512` 而非原生 `batch=512` (受 40GB 单卡显存限制)。梯度累积下 BatchNorm/dropout 的统计量与原生大 batch 不严格等价, 虽然 ETEGRec 用的是 T5 (LayerNorm, 无 BN), 影响应该有限, 但不能完全排除。

5. **单卡 vs 论文多卡** — 论文 accelerate DDP 多卡, 我们单卡。seed=42 固定但 dataloader worker 顺序与 DDP 分片不同, 存在不可消除的随机性差异。

**结论**: ETEGRec 在 Amazon 2018 Musical_Instruments 上**显著弱于 HG-Rec 基线**。相比同数据集其它复现:
- DIGER FrqUD test R@10 = **0.1121** (+9.5% vs 基线)
- DIGER SDUD test R@10 = **0.1096** (+7.0%)
- DECOR test R@10 = **0.1157** (+13.0%)
- HG-Rec 基线 = **0.1024**
- HG-Rec 最佳变体 v78 = **0.1092** (+6.6%)
- **ETEGRec = 0.0763 (-25.5%)** ← 本轮, 最差

**verdict 路径**: `verdicts/etegrec_instruments_migration_nogo.md`

---

## 工程修复记录 (迁移必需的 3 处 patch)

1. **`model.py:27`** — transformers 4.40+ 移除了 `_supports_cache_class` 属性, 原代码 `self._supports_cache_class = model._supports_cache_class` 直接 crash。
   修复: `getattr(model, '_supports_cache_class', False)`

2. **`utils.py:safe_load()`** — RQ-VAE ckpt 保存格式是 `{args, epoch, state_dict, best_collision_rate, ...}` 嵌套 dict, 但 `safe_load` 直接把顶层 key (args/epoch/...) 往 model 里 load → **实际 0 个权重被加载**, `model_id` 用随机初始化 → RQ-VAE 产生退化 code (level 1 只有 **1 个** unique code) → `ValueError: max_conflict=7014 > code_num=256` 崩溃。
   修复: 检测并 unwrap `'state_dict'` key。修复后 code 使用率恢复 232/256/256。

3. **`metrics.py:6,11`** — `np.asfarray` 在 NumPy 2.0 已移除, 第一次 eval 时 crash (`AttributeError`)。
   修复: `np.asarray(x, dtype=float)`

4. **`accelerate_config_single.yaml`** — 新建。原计划 4 卡 DDP, 但 NCCL 报 `Duplicate GPU detected (all 4 ranks on cudaDev 0)` — 本 host 只有 1 张 A100。
   配置: `distributed_type: 'NO'` (注意必须大写, accelerate enum 大小写敏感), `num_processes: 1`, `mixed_precision: 'bf16'`

5. **`test_only.py`** — 新建独立 test 评估脚本 (R30 硬编码超参 + 仅 `--config`/`--ckpt` 为 argparse 路径, R32 直接 `python3` 执行)。复用 `main.py` 完全相同的模型构造路径, 调用 `trainer.test(model_file=...)` 指定 best ckpt。

---

## 复现命令 (完整链路)

```bash
# Step 1: 语义 embedding (768d sentence-t5-base → 前 256 维)
python3 gen_instruments_emb_t5base_256.py

# Step 2: RQ-VAE 预训练 (3000 epochs, ~7min)
CUDA_VISIBLE_DEVICES=0 python3 -u main.py --config ./config/rqvae_instruments.yaml

# Step 3: 端到端主训练
accelerate launch --config_file accelerate_config_single.yaml main.py \
  --config ./config/instruments.yaml \
  --lr_rec=0.005 --lr_id=0.0001 --cycle=2 --eval_step=2 \
  --rec_kl_loss=0.0001 --rec_dec_cl_loss=0.0003 \
  --id_kl_loss=0.0001 --id_dec_cl_loss=0.0003

# Step 4: test 评估 (指定 best ckpt)
accelerate launch --config_file accelerate_config_single.yaml test_only.py \
  --config ./config/instruments.yaml \
  --ckpt ./myckpt/instruments/Aug-07-2026_14-44-9c24d0/17.pt
```

产物目录: `/home/wlia0047/ar57/wenyu/ETEGRec/myckpt/instruments/Aug-07-2026_14-44-9c24d0/`
日志: `/home/wlia0047/.claude/jobs/91631871/tmp/etegrec_v2.log` (训练) + `etegrec_test.log` (评估)

---

## 后续可选方向 (若要继续 ETEGRec 路线)

**P0 — 修复 256 维截断** (最可能的主因): 训一个 768→256 的投影层, 或改用原生输出 256 维的 encoder, 或直接把 ETEGRec 的 `semantic_hidden_size` 改成 768 (需同步改 RQ-VAE `in_dim` 与 `layers=[512,256]`)。预期收益最大。

**P1 — 正则化对抗过拟合**: `dropout_rate` 0.1 → 0.2, `weight_decay` 0.05 → 0.1, 或加 label smoothing。参考 HG-Rec v74 (WD=0.01 + dropout=0.20 使 test +0.0039)。

**P2 — 缩小模型**: `encoder_layers`/`decoder_layers` 6→4, `d_model` 128→96。9922 items 的数据规模可能撑不住 6+6 层 T5。

**当前判定**: 以上均为推测性改动, 而 DECOR (0.1157) / DIGER (0.1121) 已在同数据集上验证有效且远超 ETEGRec, ETEGRec 路线的 ROI 明显低于继续优化 DECOR/DIGER 借鉴点。**建议不继续投入 ETEGRec**。
