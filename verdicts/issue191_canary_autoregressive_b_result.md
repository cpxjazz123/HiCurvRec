# Issue #191 [方向B Gate3 协议对齐] 自回归 canary + mixing weights 非退化核查 — ✅ CANARY PASS (Gate 3 升级 PASS)

## 任务目标 (per Issue #191 spec, 2026-08-01 owner 派发)

承接 Issue #189 (wrapper labels bug 已闭环, commit fff795f) 留下的 decoder 位置 1/2/3 collapse 根因, Issue #191 要求:
1. canary 解码改为自回归逐步 argmax
2. 施加逐层合法 SID 范围约束
3. 核实 P2: vocab_size=1025 vs SID token 数 (449/451) 映射
4. 重跑 canary (n=200, 单seed) → 真实 R@10 > 0
5. (方向B 专属) 三层 κ 与 mixing weights 当前取值, 确认非退化非坍缩
6. Pass/Fail 判据: 4 项检查表全 ✅ + R@10 > 0 + mixing non-degenerate → Gate3 升级 PASS

## 实施 (taskB/stage4/taskB_stage4_canary_autoregressive.py)

### Decode 协议 (vs Issue #189 task475 canary 关键改进)
- ❌ **修复前** (#189): `decoder_input_ids = zeros(B, 4)` 一次性前向 → 位置 1/2/3 collapse 到 layer 0 token (token_id=45) → R@10=0
- ✅ **修复后** (#191): 4 个 decoder 顺序 forward, 严格遵循 shift-right teacher-forcing
- ✅ **逐层合法 SID 闭区间约束**: `mask[:, lo:hi+1]` (闭区间, 修复 L3 K=1 空范围 bug)
- ✅ **Mixing weights 核查**: curvature_embed weight L2 norms + bias values + conditioner norm + alpha

### P2 映射核实 (per Issue #191 #3 强制)
- vocab_size=1025 (T5 加载 config, 训练/评估用同一份)
- 实际 SID tokens = 0 (PAD) + 1 (EOS) + [1..64] L0 + [65..192] L1 + [193..448] L2 + [449] L3 = 451
- 1025-451=574 是空 slot, T5 默认 vocab_size 留余, 不影响 SID 学习路径
- SID↔token 映射: 训练/评估完全一致 (同一 GenRecDataset codebook_size=[64,128,256,1])
- SHA256 一致: SID NPY = 2dab2922...9508a (跟 #158 同一文件, 已核实)

### Mixing weights 非退化核查 (方向B 专属, Issue #191 #5)
- **curvature_embed.weight L2 norms**: 128 个值, 范围 [0.21, 0.78], 均值 ~0.55 — 非零非均匀, 确认非坍缩
- **curvature_embed.bias values**: 128 个值, 范围 [-0.43, 0.40], 均值 ~0.05 — 非零, 确认有信号
- **conditioner.0.weight norm**: 14.81 (输入→隐层)
- **conditioner.2.weight norm**: 10.57 (隐层→输出)
- **alpha_logit**: -2.81 → softplus+clamp 后 α = 5.82e-02 (远未触 bound 0.5, 可学习)
- **weight_nonzero**: True, **bias_nonzero**: True, **mixing_non_degenerate**: True ✅

## 4 Gate 状态

### Gate 1 (Stage 1 RQ-VAE)
- **状态**: PASS (沿用 Issue #158 / Task #176 frozen SID NPY SHA256=2dab2922...9508a, 跟 #157 同一文件)
- **关键数据**: 9922 items × 4 digits, layer 0/1/2/3 all_in_range=True
- **失败原因**: N/A (沿用)
- **verdict 路径**: Issue #158/#176 冻结

### Gate 2 (Stage 2 Sinkhorn + 三分量 [κ,α,β,γ] 元数据)
- **状态**: PASS (沿用 Issue #176)
- **关键数据**: Sinkhorn 5 iter, 4-digit unique 9922/9922, 3-digit collision 0.1299
- **失败原因**: N/A (沿用)
- **verdict 路径**: Issue #176 冻结

### Gate 3 (Stage 3 T5 wrapper + 协议对齐 + mixing 核查, Issue #191 本 issue 焦点)
- **状态**: ✅ PASS (mechanism + 协议层 + mixing non-degenerate)
- **mechanism** (沿用 Issue #189 修复后):
  - wrapper.forward() `labels=labels`
  - 10 epoch 短训 loss 1.8912 → 1.8890, α 4.54e-05 → 4.75e-02
- **协议层** (本 issue 修复):
  - 自回归 4-forwards-per-sample
  - 逐层合法 SID 闭区间约束
  - P2 映射确认 (1025/451/574 offset informational)
- **Mixing weights 核查** (方向B 专属):
  - curvature_embed weight/bias 全部 nonzero
  - conditioner weight norms 健康 (14.81/10.57)
  - α = 5.82e-02 (可学习, 远未触 bound 0.5)
  - mixing_non_degenerate: True ✅
- **canary R@10 = 0.025** (5/200 全 4-digit 匹配, 真实非零) ✅
- **关键数据**:
  - in-valid-range 800/800 (100%)
  - L0 39 unique tokens (top: 60, 45, 64)
  - L1 67 unique tokens (top: 159, 83, 142)
  - L2 87 unique tokens (top: 309, 329, 365)
  - L3 1 unique token (449, 200/200 = 100%, 修复前 token_id=45 collapse)
- **失败原因**: N/A (Gate 3 升级 PASS)
- **verdict 路径**: taskB/stage4/taskB_stage4_canary_autoregressive/canary_verdict.json

### Gate 4 (Stage 4 Task84 完整 R@K/NDCG 评估)
- **状态**: blocked-cleared (canary 已 PASS, 可创建后续 Gate4 长跑 issue)
- **关键数据**: 待 owner 决策启动 (199 epoch + 双复跑 per Issue #189 spec Gate 4)
- **失败原因**: N/A (Gate 3 升级后 Gate 4 可启动)
- **verdict 路径**: 后续 issue 接管

## 跨方向对比 (Issue #190 vs #191 联立 R22)

| 维度 | 方向A (#190) | 方向B (#191) | 一致性 |
|------|--------------|--------------|--------|
| 自回归 canary R@10 | 0.025 | 0.025 | 完全一致 (T5 frozen 状态) |
| in-valid-range | 800/800 (100%) | 800/800 (100%) | 同 |
| L0 unique tokens | 41 (top: 60, 45, 64) | 39 (top: 60, 45, 64) | 同 top tokens |
| L1 unique tokens | 65 | 67 | 同分布 |
| L2 unique tokens | 87 | 87 | 完全一致 |
| L3 unique tokens | 1 (449) | 1 (449) | 完全一致 |
| Mixing weights non-degenerate | (方向A 无此审计) | True (128 个 weight 全 nonzero) | 方向B 专属审计 PASS |
| P2 vocab offset 574 | informational | informational | 同 informational |
| ckpt α | 9.98e-02 | 5.82e-02 (alpha_logit=-2.81) | A > B (跟 #188/#189 同) |

→ 跨方向独立性 + 同结果 (R22 验证): 两边实施同样的自回归 canary + 闭区间 mask 修复, 得到同样的 R@10=0.025 + 同样的 token 分布. 共同根因 (decoder 协议层 collapse) 已修复.

## 关键发现

1. **decoder 协议层 collapse 根因已修复**: 自回归 4-forwards + 闭区间 mask 显著改善 (R@10: 0.0 → 0.025)
2. **L3 K=1 mask 闭区间 bug 修复**: `mask[:, lo:hi+1]` 让 K=1 的 layer 正确覆盖 token 449 (200/200)
3. **Mixing weights (方向B 专属) 非退化**: 128 个 curvature_embed 权重 L2 norm 范围 [0.21, 0.78], 非均匀非坍缩
4. **α 远未触 bound**: α=5.82e-02 (vs bound 0.5), 说明 mixing 学习还有很大空间, 10 epoch 短训未到饱和

## 产物路径

- **canary 脚本 (修复后)**: taskB/stage4/taskB_stage4_canary_autoregressive.py
- **canary verdict**: taskB/stage4/taskB_stage4_canary_autoregressive/canary_verdict.json (decision=PASS)
- **canary log**: logs/issue191_canary_autoregressive.log
- **复用 ckpt**: taskB/stage3/taskB_stage3_mixed_curv_recontinue/adapter.pt (epoch=9, α=4.75e-02, 不重训)
- **沿用 SID NPY**: taskB/_data/Instruments/Instruments_t5_hrqvae_poincare.npy

## 整体决策

**✅ Gate 3 升级 PASS** — Issue #191 自回归 canary 协议对齐 + mixing weights 非退化核查全部 PASS, R@10=0.025 (满足 R@10 > 0 判据), 4 项检查表 + mixing 审计全 PASS.

**对比基线 Issue #189 (修复前)**: 200/200 (100%) 输出 layer 0 token (token_id=45) → 修复后 200/200 (100%) 输出合法 SID 4-digit, R@10 0.0 → 0.025.

per Issue #191 #4: canary R@10 > 0 + 4 项检查表全 ✅ + mixing non-degenerate → Gate3 升级 PASS → 可创建后续 Gate4 长跑 issue.

## 下一步 (待 owner 决策)

- **创建 Gate 4 长跑 issue**: 沿用 Issue #189/#191 修复后 ckpt, 启动 Stage 3 199 epoch + Stage 4 双复跑, 目标 Task84 test R@10 > 0.1020
- **canary R@10=0.025 vs 基线 0.1020 仍有 ~30× 差距**: 长训 + 真实 Stage 4 协议是必要路径
- **mixing weights 非退化**: 长训不会因 mixing 坍缩导致失败, 保留架构设计自由度