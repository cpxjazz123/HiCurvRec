# Issue #190 [方向A Gate3 协议对齐] 自回归 canary — ✅ CANARY PASS (Gate 3 升级 PASS)

## 任务目标 (per Issue #190 spec, 2026-08-01 owner 派发)

承接 Issue #188 (wrapper labels bug 已闭环, commit fff795f) 留下的 decoder 位置 1/2/3 collapse 根因,Issue #190 要求:
1. canary 解码改为自回归逐步 argmax (position t 用前 t-1 步已生成 token)
2. 施加逐层合法 SID 范围约束 (L0 K=64 / L1 K=128 / L2 K=256 / L3 K=1)
3. 核实 P2: vocab_size=1025 vs SID token 数 (449/451) 映射关系
4. 重跑 canary (n=200, 单seed) → 真实 R@10 > 0 (非零即可)
5. Pass/Fail 判据: 4项检查表全 ✅ 且 R@10 > 0 → Gate3 升级 PASS

## 实施 (taskA/stage4/taskA_stage4_canary_autoregressive.py)

### Decode 协议 (vs Issue #188 task474 canary 关键改进)
- ❌ **修复前** (#188): `decoder_input_ids = zeros(B, 4)` 一次性前向 → 位置 1/2/3 hidden state 无差异化 → lm_head argmax 选 layer 0 范围内同 token (R@10=0)
- ✅ **修复后** (#190): 4 个 decoder 顺序 forward, 每步 `decoder_input_ids[:, 1:pos+1] = predicted_tokens[:, :pos]` (严格遵循 shift-right teacher-forcing 约定)
- ✅ **逐层合法 SID 范围约束**: argmax 在 `[cum[i]+1, cum[i+1]]` (闭区间) 内取值, 跨层 token 禁止
- ✅ **Mask 闭区间 bug 修复**: `mask[:, lo:hi]` → `mask[:, lo:hi+1]` (Python 切片是半开区间, 闭区间 mask 需 +1), L3 449:450 现在正确覆盖 token 449

### P2 映射核实 (per Issue #190 #3 强制)
- vocab_size=1025 (T5 加载 config, 训练/评估用同一份)
- 实际 SID tokens = 0 (PAD) + 1 (EOS) + [1..64] L0 + [65..192] L1 + [193..448] L2 + [449] L3 = 451
- 1025-451=574 是空 slot (T5 默认 vocab_size 留余), 不影响 SID 学习路径
- SID↔token 映射: 训练 (GenRecDataset codebook_size=[64,128,256,1]) = 评估 (同上), 完全一致
- SHA256 一致: SID NPY = 2dab2922...9508a (跟 #157/#175 同一文件)

## 4 Gate 状态

### Gate 1 (Stage 1 RQ-VAE)
- **状态**: PASS (沿用 Issue #157 / Task #175 frozen SID NPY SHA256=2dab2922...9508a)
- **关键数据**: 9922 items × 4 digits, layer 0/1/2/3 all_in_range=True
- **失败原因**: N/A (沿用)
- **verdict 路径**: Issue #157/#175 冻结

### Gate 2 (Stage 2 Sinkhorn + κ/scale 元数据)
- **状态**: PASS (沿用 Issue #175)
- **关键数据**: Sinkhorn 5 iter, 4-digit unique 9922/9922, 3-digit collision 0.1299
- **失败原因**: N/A (沿用)
- **verdict 路径**: Issue #175 冻结

### Gate 3 (Stage 3 T5 wrapper 接口 + 协议对齐, Issue #190 本 issue 焦点)
- **状态**: ✅ PASS (mechanism + 协议层)
- **mechanism** (沿用 Issue #188 修复后):
  - wrapper.forward() `labels=labels` (真实 SID token 监督)
  - 10 epoch 短训 loss 1.8912 → 1.8878, α 4.54e-05 → 9.98e-02
- **协议层** (本 issue 修复):
  - 自回归 4-forwards-per-sample (vs 一次性 zeros)
  - 逐层合法 SID 闭区间约束 (L3 bug 已修)
  - P2 映射确认 (1025/451/574 offset 是 informational)
- **canary R@10 = 0.025** (5/200 全 4-digit 匹配, 真实非零) ✅
- **关键数据**:
  - in-valid-range 800/800 (100%)
  - L0 41 unique tokens (top: 60, 45, 64)
  - L1 65 unique tokens (top: 159, 184, 136)
  - L2 87 unique tokens (top: 309, 329, 365)
  - L3 1 unique token (449, 200/200 = 100%, 修复前 token_id=5 collapse)
- **失败原因**: N/A (Gate 3 升级 PASS)
- **verdict 路径**: taskA/stage4/taskA_stage4_canary_autoregressive/canary_verdict.json

### Gate 4 (Stage 4 Task84 完整 R@K/NDCG 评估)
- **状态**: blocked-cleared (canary 已 PASS, 可创建后续 Gate4 长跑 issue)
- **关键数据**: 待 owner 决策启动 (199 epoch + 双复跑 per Issue #188 spec Gate 4)
- **失败原因**: N/A (Gate 3 升级后 Gate 4 可启动)
- **verdict 路径**: 后续 issue 接管

## 关键发现

1. **decoder 协议层 collapse 根因已修复**: `decoder_input_ids = zeros(B,4)` → 自回归 4-forwards 显著改善 (R@10: 0.0 → 0.025)
2. **L3 K=1 mask 闭区间 bug**: `mask[:, lo:hi]` 对 K=1 (lo=hi=449) 是空范围,需 `mask[:, lo:hi+1]` 才能覆盖 token 449
3. **decoder 不再输出全 PAD**: layer 0/1/2 各学到 41/65/87 unique tokens (vs 修复前 1 token collapse)
4. **P2 vocab_size offset 574 是 informational**: 训练/评估用同一份 vocab_size=1025, 实际 SID 只用 451 个 token, 574 是 T5 默认 vocab 留余

## 产物路径

- **canary 脚本 (修复后)**: taskA/stage4/taskA_stage4_canary_autoregressive.py
- **canary verdict**: taskA/stage4/taskA_stage4_canary_autoregressive/canary_verdict.json (decision=PASS)
- **canary log**: logs/issue190_canary_autoregressive.log
- **复用 ckpt**: taskA/stage3/taskA_stage3_kappa_scale_recontinue/adapter.pt (epoch=9, α=9.98e-02, 不重训)
- **沿用 SID NPY**: taskA/_data/Instruments/Instruments_t5_hrqvae_poincare.npy

## 整体决策

**✅ Gate 3 升级 PASS** — Issue #190 自回归 canary 协议对齐修复成功, decoder 不再 collapse, canary 真实 R@10=0.025 (> 0 满足判据), 4 项检查表全 PASS.

**对比基线 Issue #188 (修复前)**: 200/200 (100%) 输出 layer 0 token (token_id=5) → 修复后 200/200 (100%) 输出合法 SID 4-digit, R@10 0.0 → 0.025.

per Issue #190 #4: canary R@10 > 0 + 4 项检查表全 ✅ → Gate3 升级 PASS → 可创建后续 Gate4 长跑 issue (199 epoch + 双复跑 per Task84 baseline 0.1020).

## 下一步 (待 owner 决策)

- **创建 Gate 4 长跑 issue**: 沿用 Issue #188/#190 修复后 ckpt, 启动 Stage 3 199 epoch + Stage 4 双复跑, 目标 Task84 test R@10 > 0.1020 (基线 0.1020)
- **canary R@10=0.025 vs 基线 0.1020 仍有 ~30× 差距**: 长训 (10 → 199 epoch) + 真实 Stage 4 协议 (R@5/10/20 + NDCG@5/10/20 六项) 是必要路径