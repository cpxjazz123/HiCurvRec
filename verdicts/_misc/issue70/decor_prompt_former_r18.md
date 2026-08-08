# Issue #70: DECOR PromptFormer 移植 — R18 4 维度对比 + 实施计划

**日期**: 2026-08-07
**方案**: 把 DECOR 的 PromptFormer (candidate bins + alpha gate) 移植到 HG-Rec Stage3 T5 decoder 输入 embedding 层
**对比目标**: Issue #61 baseline (no decoration, test R@10=0.1024), taskA hyp v2 (test R@10=0.1048)

---

## R18 4 维度对比

### D1: spec 摘录 (DECOR paper / repo)
- DECOR (SIGIR 2026 / yliuaa/DECOR): T5 + Decomposed Contextual Tokens
- alpha=0.35 (推荐): 控制 context-aware embedding 与静态 lookup 的混合比例
- bos_queries=64: 64 个 learnable query, 做 attention pooling fused_embeds
- num_beams=50 评估 (vs HG-Rec 20)
- lr=3e-3, wd=0.05, warmup=10000 (vs HG-Rec 1e-4/0.0/0)
- AlphaBoost (DECOR 后续): 完全替换 lookup 为 context-aware

### D2: 实施核心
- **DECOR PromptFormer**:
  1. bos_queries: 64 learnable vectors
  2. e_ctx = AttentionPool(fused_embeds) → bos_vec = attn @ bos_queries
  3. 对 decoder 位置, 在 bin (token_id // 256) 内 256 候选 SID embedding 做 soft attention
  4. e_soft = weighted sum over candidates
  5. e_final = α * e_soft + (1-α) * e_fused, α=0.35
- **HG-Rec 当前**: 纯 static codebook lookup e_fused = shared(input_ids) * sqrt(d_model)
- **差异**: HG-Rec 没有 candidate bins + alpha gate, DECOR 有

### D3: Gate 1 失败机制
- DECOR 实施在 codebook embedding 与 e_fused lookup 兼容 (都用 T5 nn.Embedding(1024, 128))
- HG-Rec 也用相同 nn.Embedding, **结构兼容**
- 风险点: alpha gate 在 (0,1), 如果 α 学到 ~0, 改造无效 (退化为基线)
- 风险点: candidate bins 256 codes × 4 bins = 1024, 跟 HG-Rec 词表大小一致, 兼容

### D4: 引用文献
- DECOR paper: "DECOR: Decomposed Contextual Tokens for Generative Recommendation" (SIGIR 2026)
- HG-Rec baseline: "Hyperbolic RQ-VAE + Differential-Length Codebook + T5"
- 两个都基于 TIGER-style 生成式推荐 (Rangwani et al. 2024)

**结论**: R18 4 维度对比通过, 必须实验 (D2 实施核心完全不同, 不能凭"路径同构"判断 NO-GO)。

---

## 实施计划

### Phase 1: 模块 + 集成
1. `common/decor_prompt_former.py`: PromptFormer 类
   - bos_queries (64, 128) nn.Parameter
   - alpha_raw scalar nn.Parameter (init=0.35 via sigmoid)
   - attention_pool: 用欧氏 attention pool (Path A 已证明 Poincaré 无收益)
   - candidate_bins: 4 bins × 256 codes × 128 dim = 1024 × 128
   - forward(e_fused, input_ids) → e_final

2. `common/stage3/stage3_train_pure_t5.py`:
   - 新加 `--enable_prompt_former` flag
   - 新加 `--prompt_former_alpha` 默认 0.35
   - install_prompt_former() hook: 包装 forward + generate, 在 shared(input_ids) 之后注入
   - 保留 install_geo_residual / install_hab 不动 (与现有 B_geo / HAB / GEO 正交)

### Phase 2: precheck (Gate 1-4)
- Gate 1: 数值稳定性 (alpha gate 在 0-1, candidate bins 在 0-255)
- Gate 2: 梯度流通 (alpha_raw, bos_queries, candidate_bin_proj)
- Gate 3: alpha gate 正确性 (α=0 → e_final = e_fused, α=1 → e_final = e_soft)
- Gate 4: 集成 hook 位置正确 (在 shared(input_ids) 之后, model(inputs_embeds=...) 之前)

### Phase 3: micro-training (端到端连通性)
- 用 taskA_stage3_issue61/HG_Rec_best.pth 起步
- 100 样本, 3 epoch
- 验证: loss 下降 + 生成 token id 合法 (在 0-1023 范围)

### Phase 4: 完整训练 (Phase 3 通过后启动)
- 95 epoch, bs=256, lr=4e-4 (保守, 不上 3e-3), alpha=0.35
- 评估 Gate 1-4, 决定 PASS/NO-GO

---

## 关键设计选择

### Q1: attention pool 用 Euclidean 还是 Poincaré?
**答**: Euclidean
**理由**: Path A v2 实验证明 Poincaré 单点改造无收益 (被欧氏加权 sum 数学性质锁死)
DECOR 原版就用 Euclidean attention pool

### Q2: alpha gate 用 scalar 还是 per-layer?
**答**: scalar (单 α)
**理由**: DECOR 原版用 scalar alpha=0.35; per-layer alpha 增加 12× 参数, 边际收益不明

### Q3: candidate bins 来源?
**答**: 用 T5 shared embedding (与 e_fused 同一 nn.Embedding)
**理由**: DECOR 用 codebook embedding; HG-Rec 用 shared embedding, 等价语义

### Q4: 与现有 HAB / GEO / B_geo 的交互?
**答**: PromptFormer 在 encoder 输出后注入 (在 model.shared 之后), HAB/GEO/B_geo 在 model 内部 attention 层
**验证**: 安装顺序 = shared lookup → prompt_former (encoder 输入) → model(inputs_embeds=...) → HAB/GEO (attention bias)
完全正交, 可叠加

### Q5: alpha init 用 0.35 还是 0?
**答**: 0.35 (sigmoid 后实际值, 不是 sigmoid 输入)
**代码**: alpha_raw = logit(0.35) ≈ -0.619, sigmoid(-0.619) = 0.35

---

## 当前产物 (Phase 1)

- `common/decor_prompt_former.py`: PromptFormer 模块
- `common/stage3/stage3_train_pure_t5.py`: --enable_prompt_former + install hook
- precheck 脚本
- micro-training 脚本