# Task #475 / Issue #187 [方向B Gate3→Gate4] 协议对齐 canary 真实 argmax sanity check (禁止直接长跑)

## 背景
- Gate 3 (Issue #184) PASS 仅证明 wrapper 机制完整性, 协议对齐未验证
- Issue #181 已用 200 epoch 长训 + 双复跑才发现 R@10=0, 浪费算力
- 本 issue 必须先做 canary 前置检查, 才能允许进入 Gate 4 长跑

## 范围 (per Issue #187 spec)
1. **小规模真实样本 Stage 4 argmax sanity check** (几十到几百条, 单 seed, 禁止双复跑/长训)
2. **逐项核实 4 条协议一致性**:
   - 解码/打分方式是否同一 forward 调用路径
   - tokenizer/vocab 映射一致性
   - lm_head 一致性 (路径/dtype)
   - 合法 SID 约束空间
3. **canary R@10 ≠ 0** → Gate 3 升级 PASS, 后续创建新 issue 安排 Gate 4 长跑
4. **canary R@10 = 0** → 定位 4 项协议检查中**具体哪一项**不一致, 作为下一修复 issue 输入

## 三层框架合规性 (Issue #187 spec 强制)
- 沿用 #184 BoundedWeightedMixedCurvatureConditioner
- 三层各自独立 learnable κ + 固定双曲/欧氏分量 + 可学习 mixing 权重
- 不退化为纯欧氏 / 不使用固定权重替代可学习 mixing / 不脱离三层框架

## Gate 1-4 顺序
- Gate 1 (Stage 1): PASS 沿用 #176/#158 (SID NPY SHA=4654f3e2...)
- Gate 2 (Stage 2): PASS 沿用 #176
- Gate 3 (Stage 3): #184 PASS 机制层面, 协议对齐 conditional-not-verified
- **本 issue (Issue #187): canary 前置检查**
- Gate 4: Task #84 统一评估 (后续 issue, canary PASS 才允许)

## 复用产物
- ckpt: products/task473_issue181_direction_b_gate4_200ep/adapter_200ep.pt (Issue #181 200 epoch 训练产物)
- SID NPY: HG-Rec/dataset/Instruments/Instruments_t5_hrqvae_poincare.npy
- T5 frozen: products/task84/ckpt_hgrec/Instruments/Jul-23-2026_20-24-44/HG_Rec_best.pth
- wrapper: scripts/task471_issue178_gate3_b_recontinue.py::HG_Rec_with_BoundedWeightedMixedAdapter

## 决策
- canary PASS: Gate 3 → full PASS, 升级 Issue #187 为 Gate 4 ready
- canary FAIL: 定位 4 项协议检查具体哪一项失败, 写 verdict 根因
