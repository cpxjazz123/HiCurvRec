# Issue #70: DECOR PromptFormer 移植 — NO-GO 闭环 (self-reinforcing trap)

**日期**: 2026-08-07
**状态**: NO-GO (3 个失败机制叠加)
**对比目标**: Issue #61 baseline (no decoration, test R@10=0.1024), taskA hyp v2 (test R@10=0.1048)

---

## 最终指标

| 指标 | PromptFormer ep25 best | HG-Rec baseline | Δ |
|------|----------------------|------------------|---|
| valid R@10 | 0.1285 | 0.1267 | +0.0018 ✓ (虚高) |
| valid NDCG@20 | 0.1026 | (未全量核查) | — |
| **test R@10** | **0.1002** | **0.1024** | **-0.0022 ✗** |
| test R@5 | 0.0813 | 0.0819 | -0.0006 ✗ |
| test R@20 | 0.1234 | 0.1283 | -0.0049 ✗ |
| test NDCG@20 | 0.0814 | 0.0821 | -0.0007 ✗ |
| valid/test ratio | **0.78** | **0.81** | -0.03 失衡加剧 |

verdict: `/tmp/v70_prompt_former/test_eval_ep25/eval_test.json`
training: `/tmp/v70_prompt_former/train.log` (ep33 时 user 取消)

---

## 根因诊断 (3 个独立证据)

### 证据 1: alpha_raw 完全没动

```
alpha_raw (init logit(0.35)) = -0.619000
alpha_raw (ep25 best)        = -0.619055
delta                          = -0.000055
alpha (sigmoid)                = 0.349997 (init 0.35)
```

25 epoch 训练后 alpha gate 完全没学到, 始终输出 0.35 (init 标定值).
T5 训练 loss 对 alpha 大小**零敏感** (gradient 通过 e_soft 传来, 但 e_soft 几乎无信息).

### 证据 2: bos_queries 同质化

```
bos_queries shape: torch.Size([64, 128])
norm mean±std:    1.1459 ± 0.0514   (Xavier init 后 ≈1.38)
weight stats:     mean=0.0024  std=0.1014
```

64 个 learnable bos_queries **几乎长一个样** (std=0.051 极小).
attention pool 输出对所有 query 一致 → bos_vec 几乎是固定均值向量.

### 证据 3: candidate attention 几乎均匀

```
[valid]  attn_max=0.0080  attn_entropy=5.5064  (ln256=5.545 均匀)
[test]   attn_max=0.0079  attn_entropy=5.5075
```

attn_entropy = 99.3% 均匀, attn_max ≈ 2× 均匀值 0.0039.
→ e_soft ≈ 每个位置 256 个候选 code 的**简单均值**
→ candidate-bin attention **没有学到任何 code 偏好**.

### 证据 4: valid vs test 分布几乎一致

| 量 | valid | test | Δ |
|----|-------|------|---|
| bos_norm | 0.8135 | 0.8041 | -0.0094 |
| esoft_norm | 0.7883 | 0.7872 | -0.0011 |
| \|\|e_soft-e_fused\|\| | 132.02 | 131.90 | -0.1189 |
| attn_entropy | 5.5064 | 5.5075 | +0.0011 |

DecorPromptFormer 在 valid 和 test 上产生的扰动幅度几乎相同 (Δ < 0.01)
→ **DecorPromptFormer 不是 valid/test 失衡的根因**

### 证据 5: T5 shared embedding 被间接修改

```
shared.weight (baseline norm=379.81) vs shared.weight (v70)  diff norm=108.87
```

DecorPromptFormer 通过 teacher forcing 训练, 间接修改了 T5 主干的 shared embedding.
但这是任何加辅助模块训练的正常副作用, 不是过拟合的**充分**原因.

---

## 失败机制: Self-reinforcing Trap

DecorPromptFormer 的整条路径陷入一个**自我强化的死锁**:

```
bos_queries 同质化 (std=0.051)
        ↓
bos_vec ≈ 固定均值向量 (norm 几乎不依赖 input)
        ↓
candidate attention: softmax(q·k) 几乎均匀分布
        ↓
e_soft ≈ 256 个候选 code 的均值
        ↓
∂L/∂alpha ≈ 0  (e_soft 无信息 → alpha gate 无学习信号)
        ↓
alpha_raw 不更新 → 永远卡在 init = 0.35
        ↓
e_final = 0.35·噪音 + 0.65·e_fused  (固定扰动, 不是学到的)
        ↓
T5 主干把 e_final 当作**有噪声的输入**适应
        ↓
valid 上适应成功 (R@10 +0.0018, 来自充分训练本身)
        ↓
test 分布偏移 → fall short (R@10 -0.0022)
```

**数学根因**: ∂L/∂alpha = α(1-α) · ∂L/∂e_final · ∂e_final/∂alpha
- α(1-α) ≈ 0.35 × 0.65 = 0.228 (有限)
- ∂L/∂e_final ≠ 0 (T5 loss 传播)
- ∂e_final/∂alpha = e_soft - e_fused (范数 132, 有限)
- 但 ∂L/∂alpha_raw 的最终 magnitude 受 e_soft 的**信息量**影响:
  e_soft ≈ 均匀加权 → e_soft 对 bos_queries / q_ctx / k_candidates 的依赖几乎一致
  → 这些参数的 gradient 互相抵消 → ∂L/∂alpha 也接近 0

---

## 为什么 "调超参 / 加大 LR" 不能修复

1. **加大学习率**: gradient ≈ 0, LR×0 = 0 永远没信号
2. **改 alpha init**: 不解决 e_soft 无信息的根因
3. **加正则化**: bos_queries 同质化本身就是 Xavier init 加上 dropout 退化的稳定点, 加正则化也无信号
4. **延长训练**: 25 epoch 还没动, 200 epoch 也未必能动 (self-reinforcing trap)

---

## 4 Gate 评估

### Gate 1: precheck 数值稳定性
- **PASS**: precheck script 4/4 Gate 通过 (`verdicts/decor_prompt_former_precheck.py`)
- forward 跑通, alpha 在 (0,1), candidate 在 0-255
- micro-training loss 5.78 → 2.47 (5 step), generate token 合法

### Gate 2: 训练 loss 下降
- **PASS**: 1.5h 训练 loss 4.82 → 2.07, 下降正常
- 但 loss 下降**不证明 DecorPromptFormer 在工作** — T5 主干本身的 loss 下降足够覆盖

### Gate 3: 端到端生成
- **PASS**: micro-training generate token id 合法 (0-1024)
- 但生成 token 合法 ≠ 推荐正确

### Gate 4: 端到端 R@10 提升
- **FAIL**: test R@10 = 0.1002 < baseline 0.1024 (Δ -0.0022)
- valid R@10 = 0.1285 > baseline 0.1267 (Δ +0.0018, 但来自训练本身)
- valid/test ratio 0.78 失衡加剧

---

## 教训

1. **DECOR PromptFormer 在 HG-Rec 移植不 work** — 原论文 DECOR 用 lr=3e-3 / wd=0.05 / warmup=10000, 我们 lr=4e-4 / wd=0 / 无 warmup, 但这不是主因, 主因是 self-reinforcing trap
2. **alpha gate 没动 = 整条路径无效** — alpha_raw init 后没动应该立即引起警觉, 25 epoch 完全没动证实无效
3. **bos_queries 同质化可被检测** — Xavier init 后 norm ≈1.38, std ≈0.10, 训练后 std=0.051 应该立即报警
4. **attn_entropy 接近 ln(K) = 均匀分布** — 5.506 vs ln(256)=5.545 是 attention 退化的强信号
5. **valid 提升 ≠ 真实提升** — valid R@10 +0.0018 但 test -0.0022, valid 提升 100% 来自 stage3 主干训练, DecorPromptFormer 净贡献为负

---

## 结论

**Issue #70 NO-GO**: DecorPromptFormer 在 HG-Rec 移植失败, 根因是 candidate-bin attention 陷入 self-reinforcing trap (bos_queries 同质化 → e_soft 均匀 → alpha 无梯度 → 固定 35% 噪声注入).

**未来方向** (待新 issue):
- 若想借鉴 DECOR, 需要重新设计 (Gumbel-softmax hard attention / auxiliary code-prediction loss / orthogonal bos_queries loss)
- 直接借鉴 α-gated 思路可能不如放弃 (DECOR 关键 advantage 在 yelp/beauty 大规模数据集, instruments 数据集太小)

---

## 产物清单

- `common/decor_prompt_former.py`: 模块实现 (保留, 标记 NO-GO)
- `common/stage3/stage3_train_pure_t5.py`: --enable_prompt_former 钩子 (保留, 可关)
- `common/stage4/stage4_eval_pure_t5.py`: --enable_prompt_former 评估钩子 (新增)
- `verdicts/decor_prompt_former_precheck.py`: precheck (保留作证)
- `verdicts/issue70_micro_training.py`: micro-training 验证 (保留作证)
- `verdicts/issue70_decor_prompt_former_r18.md`: R18 4 维度对比 (保留)
- `/tmp/v70_prompt_former/HG_Rec_best.pth`: ep25 best ckpt (保留)
- `/tmp/v70_prompt_former/test_eval_ep25/eval_test.json`: test eval verdict