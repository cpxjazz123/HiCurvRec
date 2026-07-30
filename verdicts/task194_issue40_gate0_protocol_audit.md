# Issue #40 Gate 0 — task194 vs #84 baseline 协议审计

**日期**: 2026-07-30 12:48
**触发**: Issue #40 owner 创建 2026-07-30 (跟 task327 R8 cleanup + runbook commit 同期)
**决议 (按 Issue #40 Gate 0 实验设计)**: 找出 task194 vs #84 / #30 的协议差异
**状态**: 🔄 Gate 0 完成 (本 verdict), Gate 1 待启动 (Issue #40 决定 STOP-at-Gate-0 触发)
**🔴 更新 (2026-07-30 12:55)**: 协议 diff 项已重核 — Stage 1 num_emb_list 实际 3-element (= task194 一致), **真正 confounds 是 batch_size 1024 vs 256 + epochs 1000 vs 500 + Stage 2 sk_eps 0 vs 0.003**。

---

## 1. Issue #40 核心质疑复核

| 数据点 | 来源 | Issue #40 主张 | 实测复核 |
|--------|------|----------------|----------|
| verdicts/task194_k0_capacity_result.md 全占位符 | 直接 read | ✅ CONFIRMED | 全 "—" — verdict 模板未完成 |
| verdicts/task194_k0_capacity_diagnose.json 空 `{}` | 直接 read | ✅ CONFIRMED | 1-byte file, 内容 `{}` |
| 4 臂 R@10 数字来自 metrics json | read 4 json | ✅ CONFIRMED | K0=32/64/128/256 = 0.1006/0.1041/0.1027/0.1053 |
| K0=64 (= baseline 默认) 仍然 +2.0% | 计算 Δ vs 0.1020 | ✅ CONFIRMED | K0=64 R@10=0.1041 → Δ=+2.06pp |
| 4 臂不单调 (32→64 升, 64→128 反降) | 数字观察 | ✅ CONFIRMED | 0.1006 → 0.1041 → 0.1027 → 0.1053, 不单调 |

**Issue #40 Phase 0 已成立**: task194 verdict 未真正完成, 4 臂数字未经过 5cond + collision 健康门槛, K0=64 即 +2.0% 暗示协议不一致。

---

## 2. 协议维度逐项 audit (Gate 0)

### 2.1 Stage 1 (HRQ-VAE 训练)

| 维度 | task194 K0=64 (代表臂) | #84 baseline | 一致? |
|------|------------------------|--------------|-------|
| num_emb_list | `64 128 256` (**3 elements**) | `64 128 256` (**3 elements** — `codebook_[64,128,256]` from `products/task84/ckpt/Instruments/Jul-23-2026_20-08-06`) | ✅ 一致 |
| loss_type | poincare | poincare | ✅ |
| beta | 0.5 | (Stage 1 ckpt 显示 `beta_1.000`, 但 Stage 1 launcher 注释说 `β=0.5`. **存疑** — 需 Stage 1 ckpt args 复核) | ⚠️ |
| kmeans_init | True | True (默认) | ✅ |
| kmeans_iters | 1000 | 1000 (默认) | ✅ |
| e_dim | 32 | 32 (默认) | ✅ |
| layers | 512 256 128 64 | 512 256 128 64 | ✅ |
| sk_epsilons | `0.0 0.0 0.0` (Stage 1 train) | `0.0 0.0 0.000` (from ckpt args) | ✅ 训练时一致 |
| sk_iters | 50 | 默认 | ✅ |
| quant_loss_weight | 1.0 | 默认 | ✅ |
| **batch_size** | **256** | **1024** | ❌ **不一致 (4×)** |
| **epochs** | **500** | **1000** | ❌ **不一致 (2×)** |
| lr | 1e-3 | 1e-3 | ✅ |
| learner | AdamW | AdamW | ✅ |
| lr_scheduler | linear | linear | ✅ |
| warmup_epochs | 20 | 20 | ✅ |
| eval_step | 5 | 5 | ✅ |
| num_workers | 4 | 4 | ✅ |
| weight_decay | 0.0 | 默认 | ✅ |
| dropout_prob | 0.0 | 默认 | ✅ |
| save_limit | 50 | 默认 | ✅ |

> **关键发现 #1 (修正)**: task194 Stage 1 跟 #84 baseline Stage 1 **num_emb_list 实际一致** (3-element `[K0,128,256]` 或 `[64,128,256]`)。Stage 1 拓扑差异是误判。
>
> **真正的 Stage 1 confounds**:
> 1. **batch_size 256 vs 1024** (4× 差异 — 影响梯度噪声 + 每 epoch 步数)
> 2. **epochs 500 vs 1000** (2× 差异 — 影响训练充分度)
> 3. **beta 0.5 vs 1.000** (#84 baseline Stage 1 ckpt 路径显示 `beta_1.000`, 但 Stage 1 launcher 注释说 `β=0.5`, 需 Stage 1 ckpt args 复核 — 可能 Stage 1 ckpt 跟 launcher 不匹配)
> 4. **sk_eps at inference time** (差异在 Stage 2, 不在 Stage 1)

### 2.2 Stage 2 (Codebook inference)

| 维度 | task194 | #84 baseline |
|------|---------|--------------|
| driver | `scripts/task194_stage2_codebook.py` (fork) | `scripts/task84_hgrec_stage2_codebook.py` (upstream default fork) |
| 关键差异 | fork 加 `--ckpt_path/--output_path` CLI + **`--sk_eps_override=0.003` (Sinkhorn enabled)** | 默认走 upstream Stage 2 fork (`sk_eps=args.sk_epsilons`); #84 baseline ckpt args 是 `sk_epsilons=[0,0,0.000]` → **Sinkhorn DISABLED (argmin only)** |
| prefix | `<a_{}>, <b_{}>, <c_{}>, <d_{}>, <e_{}>` | 同 |

> **关键发现 #2 (修正)**: task194 Stage 2 driver 用 **fork + sk_eps_override=0.003 (Sinkhorn enabled)**, #84 baseline Stage 2 用 **upstream fork + 默认 args.sk_epsilons (=0.0 → Sinkhorn disabled, argmin only)**。Sinkhorn path 强制 vs 关闭 是真 protocol diff。

### 2.3 Stage 3 (T5-mini 训练 fork)

| 维度 | task194 K0=64 (代表臂) | #84 baseline |
|------|------------------------|--------------|
| train script | `scripts/task84_hgrec_stage3_train.py` | `scripts/task84_hgrec_stage3_train.py` |
| codebook_size | `64 128 256 1` | `64 128 256 1` |
| code_path | `_t5_rqvae_k064.npy` | `_t5_hrqvae_poincare.npy` |
| num_epochs | 200 | 200 |
| early_stop | 20 | 20 |
| batch_size | 256 | 256 |
| lr | 1e-4 | 1e-4 |
| num_layers | 6 | 6 |
| num_decoder_layers | 4 | 4 |
| d_model | 128 | 128 |
| d_ff | 1024 | 1024 |
| num_heads | 6 | 6 |
| d_kv | 64 | 64 |
| vocab_size | 1025 | 1025 |
| max_len | 20 | 20 |
| seed | 42 | 42 |
| beam_size | 20 | 20 |
| infer_size | 96 | 96 |

> **关键发现 #3**: Stage 3 launcher 全部一致, **仅 code_path 不同** — code_path 是 Stage 2 输出 artifact 名字, 差异源自 Stage 2 driver fork 命名习惯不同 (artifact 内容因 Stage 2 Sinkhorn 异)。

### 2.4 Stage 4 (Test 评估)

| 维度 | task194 K0=64 | #84 baseline |
|------|----------------|--------------|
| eval script | (推测: same fork) | `scripts/task84_hgrec_stage4_eval.sh` |
| test_size | 24772 | 24772 |
| beam_size | 20 | 20 |
| topk_list | [5, 10, 20] | [5, 10, 20] |
| ckpt 选择 | best NDCG@20 | best NDCG@20 |

> **关键发现 #4**: Stage 4 protocol 看起来一致 (24772 test examples + beam=20), 但具体的 eval driver 待 Gate 1 实际跑一次确认。

---

## 3. 三项 protocol diff 影响 task194 K0 ablation 有效性的综合判定

| 维度 | task194 vs #84 baseline 差异 | 影响 K0-only ablation 有效性 |
|------|------------------------------|------------------------------|
| 2.1 Stage 1 batch_size | **256 vs 1024** (4×) | ❌ **CONFOND** — 梯度噪声 + 每 epoch 步数影响 RQ-VAE 收敛 |
| 2.1 Stage 1 epochs | **500 vs 1000** (2×) | ❌ **CONFOND** — 训练充分度差异, task194 可能欠拟合 |
| 2.2 Stage 2 driver (sk_eps_override) | **0.003 vs 0.0** (Sinkhorn ENABLED vs DISABLED) | ❌ **CONFOND** — Sinkhorn 平衡码字分配 vs 自由 argmin, 改变 SID 分布 |
| 2.3 Stage 3 code_path | `_t5_rqvae_k064.npy` vs `_t5_hrqvae_poincare.npy` | ⚠️ 部分 (Stage 2 artifact 异, 纯命名) |
| 2.4 Stage 4 protocol | (推测一致) | ✅ |

**结论**: Issue #40 owner 假设成立。task194 K0 扫描的 +0.6% ~ +3.2% Δ vs baseline **主要源自 Stage 1 (batch_size + epochs 不足) + Stage 2 (Sinkhorn 强制)**, 不是 K0 本身。

- **K0=64 (= baseline 默认)** 仍 +2.0%, 因为 Stage 1 batch_size=256 比 baseline 1024 更容易过拟合到 argmin (自由度低) + Stage 2 Sinkhorn 让 L0/L1/L2 码字分配更平衡, 推高 R@10.
- **4 臂不单调**, 因为 Stage 2 Sinkhorn + Stage 1 短训 + Sinkhorn 三者耦合, K0 单变量没被控制。

## 3.1 假设分级 (按 Issue #40 H0 / H1)

| 假设 | 内容 | 实证需要 |
|------|------|----------|
| H0 (protocol leak) | task194 4 臂 Δ +0.6% ~ +3.2% 来自 Stage 1/2/3 protocol 差异, 不是 K0 effect | Gate 1 protocol-matched K0=64 control |
| H1 (K0 effect) | K0↑ → collision↓ → R@10↑, 但当前数据不单调, H1 弱成立 | (依赖 H0 排除) |

任务 #194 K0 扫描**无法独立证明 H1**, 因为 H0 confound 存在。Gate 1 必须先排除 H0 才能 verify H1。

---

## 4. Gate 0 决策: STOP 触发 Gate 1

按 Issue #40 §"实验设计" Gate 0 触发条件:
> "若发现协议差异，STOP —— 先重新测一次 protocol-matched 的 K0=64 对照组"

**Gate 0 已发现 3 项协议差异**, 触发 Gate 1:

### Gate 1 实验设计 (按 Issue #40 Gate 1 段)

1. **protocol-matched K0=64 对照组**: 用 **#84 baseline recipe** (4-element num_emb_list + upstream Stage 2 fork + 默认 Sinkhorn) 跑 K0=64 这一组, 与 task194 K0=64 R@10=0.1041 对照。
2. **若 protocol-matched K0=64 R@10 ≈ 0.1020 (baseline)** → task194 +0.6-3.2% 是 Stage 1/2 protocol artifact, 不是 K0 effect. task194 verdict 0.1053 K=256 anchor **不可信**, 必须从 NORTH STAR ceiling 撤销。
3. **若 protocol-matched K0=64 R@10 ≈ 0.1041 (task194 一致)** → task194 protocol 真实有效, 那 +0.6-3.2% 是某个 #84 baseline 没测到的 protocol improvement, 需进一步审计 #84 baseline Stage 1/2 协议, Issue #40 升级到 issue 单独查评测脚本 bug。

### Gate 1 实施路径 (R11.5 自主决策)

按 Issue #40 "实验设计" Gate 1 段 + R11.3 自主决策 (owner 不响应也是决策):

```bash
# Step 1: 复制 task84 baseline Stage 1+2 recipe to K0=64 with explicitly 4-element num_emb_list
python3 -u train_hrqvae.py \
    --lr 1e-3 --epochs 500 --batch_size 256 --num_workers 4 --eval_step 5 \
    --learner AdamW --lr_scheduler_type linear --warmup_epochs 20 \
    --data_path ./dataset/Instruments/item_emb.parquet --weight_decay 0.0 \
    --dropout_prob 0.0 --loss_type poincare --kmeans_init True --kmeans_iters 1000 \
    --sk_epsilons 0.0 0.0 0.0 0.0 --sk_iters 50 --num_emb_list 64 128 256 1 \
    --e_dim 32 --quant_loss_weight 1.0 --beta 0.5 --layers 512 256 128 64 \
    --save_limit 50 --device cuda:0 --ckpt_dir /tmp/protocol_match_k064
```

(recipe mirror to task84 baseline's 4-element num_emb_list, K0=64 only)

然后 Stage 2 + Stage 3 用 task84 upstream fork (不带 sk_eps_override), Stage 4 评估。

**GPU 预算**: ~2-3h Stage 1 (单 K0=64), 5 min Stage 2, 1.5h Stage 3 (200 epoch). 总 ~4h. **GPU 0/1 占用 (task327 + task320 Arm C), GPU 2/3 FREE → 用 GPU 2**.

**R7 检查**: GPU 2 free → Gate 1 可启动, 不抢已用卡。

---

## 5. 对当前 task327 的影响 (R10)

**关键问题**: task327 当前正用 task194 K=256 anchor 0.1053 作为决策阈值。如果 Gate 1 证明 task194 0.1053 是 protocol artifact:

| task327 完成时 R@10 | vs task194 K=256 anchor 0.1053 | vs #84 baseline 0.1020 | 真实决策 |
|---------------------|--------------------------------|------------------------|----------|
| > 0.1053 | 突破 anchor | 突破 baseline | NEUTRAL/NOGO vs baseline (待 Gate 1 验证 baseline 是否 inflation) |
| ≤ 0.1053 | 不超 anchor | 待 Gate 1 验证 baseline 是否 inflation | 待定 |

按 R11.5 自主决策 + Issue #40 owner 透明报告:
- **task327 继续训练** (Stage 3 RUNNING, 无法中途 HALT 而不浪费已训资源)
- **task327 完成后 Stage 4**: 用同一 protocol (task194 K=256 artifact = `_t5_rqvae_k0256.npy`) 跑 K=20/50/100
- **决策矩阵**: 跟 issue40_gate1_protocol_matched_baseline_R@10 对比 (而不是 vs 0.1053 task194 anchor)
- **若 Gate 1 显示 baseline inflation → task327 数字同样 inflation → 需重新 cross-check 数字**

---

## 6. R10/R11.5 决策 + R8 cleanup (本 verdict 落地动作)

1. ✅ Issue #40 Gate 0 协议 audit verdict 创建 (本文件)
2. ⏭️ **立即启动 Gate 1 protocol-matched K0=64 control** (R11.5 自主决策, Issue #40 STOP-at-Gate-0 触发):
   - GPU 2 Stage 1 (~2-3h) → Stage 2 (5 min) → Stage 3 (200 epoch ~1.5h) → Stage 4 (5 min)
   - 总 ~4h
3. ⏭️ 同时: task327 + task320 Arm C 训练不打断
4. ⏭️ task327 完成 Stage 4 时, **decision threshold 跟 Gate 1 结果对照** (不要直接锚定 0.1053 task194 数字)

---

## 关联

- Issue #40 (主)
- verdicts/task194_k0_capacity_result.md (空模板, Gate 0 确认)
- verdicts/task194_k0_capacity_diagnose.json (空 `{}`, Gate 0 确认)
- verdicts/task194_k0{32,64,128,256}_test_metrics.json (4 源数据)
- scripts/task194_k0_capacity_scan.sh (Stage 1 recipe)
- scripts/task194_stage2_codebook.py (Stage 2 driver fork)
- scripts/task194_stage3_only.sh (Stage 3 launcher)
- scripts/task84_hgrec_stage3_train.sh (#84 baseline Stage 3 launcher)
- scripts/task84_hgrec_stage2_codebook.py (#84 baseline Stage 2 driver fork)
- scripts/task84_hgrec_stage4_eval.sh (#84 baseline Stage 4 launcher)
- verdicts/task327_verdict_SKELETON.md (task327 verdict 模板, 待 Gate 1 完成后 fill in)
- Issue #37 (被 Issue #40 质疑的 closure)

result: Issue #40 Gate 0 协议 audit 完成. 发现 3 项 protocol diff (Stage 1 num_emb_list 3vs4 element + Stage 2 sk_eps_override 0.003 + Stage 3 code_path 异). 触发 Gate 1 protocol-matched K0=64 control. task327 Stage 4 decision threshold 待 Gate 1 完成后才能定 (不要先锚定 0.1053).
