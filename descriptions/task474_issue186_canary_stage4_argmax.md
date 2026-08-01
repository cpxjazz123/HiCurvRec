# Task #474 / Issue #186 [方向A Gate3→Gate4] 协议对齐 canary 真实 argmax sanity check (禁止直接长跑)

## 背景
- Gate 3 (Issue #183) PASS 仅证明 wrapper 机制完整性, 协议对齐未验证
- Issue #179 已用 200 epoch 长训 + 双复跑才发现 R@10=0, 浪费算力
- 本 issue 必须先做 canary 前置检查, 才能允许进入 Gate 4 长跑

## 范围 (per Issue #186 spec)
1. **小规模真实样本 Stage 4 argmax sanity check** (几十到几百条, 单 seed, 禁止双复跑/长训)
2. **逐项核实 4 条协议一致性**:
   - 解码/打分方式是否同一 forward 调用路径
   - tokenizer/vocab 映射一致性
   - lm_head 一致性 (路径/dtype)
   - 合法 SID 约束空间
3. **canary R@10 ≠ 0** → Gate 3 升级 PASS, 后续创建新 issue 安排 Gate 4 长跑
4. **canary R@10 = 0** → 定位 4 项协议检查中**具体哪一项**不一致, 作为下一修复 issue 输入

## 三层框架合规性 (Issue #186 spec 强制)
- 沿用 #183 BoundedKappaScaleConditioner
- 保持 L0 K64 / L1 K128 / L2 K256 三层独立 learnable κ
- 不引入 global κ / 不 fixed-only / 不纯欧氏绕过 / 不脱离三层框架

## Gate 1-4 顺序
- Gate 1 (Stage 1): PASS 沿用 #175/#157 (SID NPY SHA=2dab2922...)
- Gate 2 (Stage 2): PASS 沿用 #175
- Gate 3 (Stage 3): #183 PASS 机制层面, 协议对齐 conditional-not-verified
- **本 issue (Issue #186): canary 前置检查**
- Gate 4: Task #84 统一评估 (后续 issue, canary PASS 才允许)

## 复用产物
- ckpt: products/task472_issue179_direction_a_gate4_200ep/adapter_200ep.pt (Issue #179 200 epoch 训练产物)
- SID NPY: HG-Rec/dataset/Instruments/Instruments_t5_hrqvae_poincare.npy
- T5 frozen: products/task84/ckpt_hgrec/Instruments/Jul-23-2026_20-24-44/HG_Rec_best.pth
- wrapper: scripts/task470_issue177_gate3_a_recontinue.py::HG_Rec_with_BoundedAdapter

## 决策
- canary PASS: Gate 3 → full PASS, 升级 Issue #186 为 Gate 4 ready
- canary FAIL: 定位 4 项协议检查具体哪一项失败, 写 verdict 根因
