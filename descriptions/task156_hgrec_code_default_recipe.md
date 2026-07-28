# Task #156 — HG-Rec code-default recipe (gen_codebook.py 直接 hardcoded)

> **任务目的**: 用 `HG-Rec/gen_codebook.py:46-48` 实际 hardcoded 的参数重训 HG-Rec, 验证是否拉近 R@10 跟 paper 差距
> **完成日期**: (in progress)
> **状态**: 🟡 待 Task #155 释放 GPU 3 后启动

---

## 1. 背景

**用户 2026-07-24 20:35 关键质疑**: "为什么明明有 HG-Rec 代码, 还是不能确认超参"

**审计发现** (R11.3 立即响应): `HG-Rec/gen_codebook.py:46-48` 直接 hardcoded HG-Rec authors 实际跑出 paper 数字的 recipe:

```python
dataset = "Games"
ckpt_path = f"./ckpt/{dataset}/Nov-24-2025_14-24-21_beta_0.250_codebook_[32,64,256]_sk_0.500/epoch_1869_collision_0.2345_model.pth"
output_path = f"./{dataset}_t5_rqvae_beta_0.250_codebook_[32,64,256]_sk_0.500.npy"
```

**Task #84 baseline vs code-faithful recipe 对照** (4 个关键参数全部不一致):

| 参数 | Task #84 (我们) | HG-Rec code (gen_codebook.py) | 差距 |
|------|-----------------|-------------------------------|------|
| β | 1.0 (R11.3 默认) | **0.250** | 4× 太大 |
| num_emb_list | [64, 128, 256] | **[32, 64, 256]** | L0 2× 太大, L2 同 |
| sk_epsilons | [0.0, 0.0, 0.000] | **[0.500]** | sinkhorn 完全不同 |
| Stage 1 epochs | 1000 | **1869** | 训练时间不足 |

**Stage 3 T5 默认** (HG-Rec/train_HG-Rec.py):
- num_epochs=200 (我们 early stopped at 95)
- early_stop=20
- seed=**2025** (我们用 42)
- batch_size=256, lr=1e-4 (我们一致 ✓)
- num_layers=6, num_decoder_layers=4, d_model=128 (我们一致 ✓)
- max_len=20, beam_size=20 (我们一致 ✓)

**T5 backbone**: HG-Rec 用 `T5Config` from scratch (5.5M params) — 我们一致 ✓

---

## 2. 实验设计

**变量**: β=1.0 → **0.250**, num_emb_list=[64,128,256] → **[32,64,256]**, sk_epsilons=[0,0,0] → **[0.5]**, epochs=1000 → **1869** (跟 code hardcoded 一致)
**保持不变** (跟 Task #84 baseline 其他参数一致):
- loss_type=poincare, e_dim=32, layers=[512,256,128,64], batch_size=1024, lr=1e-3, learner=AdamW, lr_scheduler=linear, warmup_epochs=20
- Stage 2 SID + Stage 3 T5 (num_epochs=200) + Stage 4 eval (跟 Task #84 pipeline 一致)

**启动命令**: `scripts/task156_hgrec_code_default_pipeline.sh`

**R7 GPU**: GPU 3 (等 Task #155 完成释放)

---

## 3. 决策触发 (vs Task #84)

| Task #156 R@10 | Δ vs Task #84 | Δ vs paper | 解读 |
|----------------|---------------|-----------|------|
| [0.1107, 0.1315] | +8% ~ +29% | paper ±25% 内 | code-faithful 是主因 ✅ |
| [0.0986, 0.1107) | -3% ~ +8% | -25% ~ -3% | code-faithful 是部分根因 |
| [0.0821, 0.0986) | -19% ~ -3% | -37% ~ -25% | 不是主因, 还有别处 |
| < 0.0821 | < -19% | < -37% | ❌ 异常, 调查 |

---

## 4. 预算

| 阶段 | 估算 |
|------|------|
| Stage 1 RQ-VAE 训练 (1869 ep) | ~45 min (vs Task #84 25 min for 1000 ep) |
| Stage 2 SID | ~3 min |
| Stage 3 T5 (num_epochs=200) | ~140 min |
| Stage 4 test eval | ~30 sec |
| **总计** | **~3h** |

**注**: 1869 epoch 比 1000 epoch 长 ~87%. 总时间 ~3h 比 Task #84 长.

---

## 5. 风险与缓解

- **codebook collision rate 23%**: HG-Rec authors 用 Games 数据 collision 23%, 我们 Instruments 可能不同. 不影响 R@10, 但监控.
- **sinkhorn epsilon=0.5**: 强 sinkhorn regularization → SID 分布更均匀 → 跟 [0,0,0] 完全不同 → R@10 可能变化大
- **epochs=1869 训练久**: 可能 over-train → early stop 监控 (best_loss 5 epoch 不降则停)

---

## 6. 完成度跟踪

- [ ] Stage 1 RQ-VAE 训练 (β=0.25, [32,64,256], sk=0.5, 1869 epoch)
- [ ] Stage 2 SID 落盘
- [ ] Stage 3 T5 (num_epochs=200) + early stop
- [ ] Stage 4 test eval R@10 落盘
- [ ] 写 verdict `verdicts/task156_hgrec_code_default_result.md`
- [ ] 更新 loop.md §16 (R8 清理)

---

## 7. 关键决策点 (R11.3)

| 决策 | 选择 | 理由 |
|------|------|------|
| 是否 kill Task #153 β=0.5 | ❌ 不 kill | β=0.5 是 paper Table 6 值, 仍 informative (paper vs code 对比) |
| 是否 kill Task #154 200 epoch | ❌ 不 kill | 200 epoch 跟 num_emb_list 无关, 仍是有效 ablation |
| Task #156 启动时机 | 等 Task #155 完成 (~5-12 min) 释放 GPU 3 | R7 强制空闲 |
| sinkhorn_epsilons 写法 | [0.5] 单元素 vs [0.5, 0.5, 0.5] 三元素 | HG-Rec code 是单元素, 跟 train_hrqvae.py argparse 设计兼容 (nargs='+') |

---

## 8. 后续 (ROI 排序)

1. **等 Task #153/#154/#155 完成** (~1.5h 后开始 Stage 3+4)
2. **Task #156 启动** (GPU 3 释放后立即)
3. **3 个 ablation 综合分析**: Task #153 (paper β=0.5) + Task #154 (paper 200 ep) + Task #156 (code β=0.25+[32,64,256]+sk=0.5) → 哪个最接近 paper R@10=0.1315 → 主因定位
4. **论文 Section 5 更新**: 增加 "HG-Rec code-faithful recipe" 段, 解释 paper vs code 不一致

---

## 9. 结论 (待写)

result: (待 Stage 4 R@10 落盘后 fill)