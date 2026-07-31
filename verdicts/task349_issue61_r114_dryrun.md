# Task #349 / Issue #61 R11.4 Dry-run Report — hyp_c=-1.0 + Riemannian retraction

**日期**: 2026-07-31
**触发**: Issue #61 [方向C Bug 修复] hyp_c=-1.0 + Riemannian retraction — R10 backlog 候选 #1
**类型**: R11.4 dry-run 报告 (NOT 实施, 仅报告修改位置 + 风险评估)
**状态**: 🟡 等待 owner 拍板 (R11.4 critical decision)

---

## 1. Issue #61 实施基础

**Issue #61 触发** (Issue #164 Gate 0 wrapper audit verdict 关键 bug 发现):
> HG_Rec_Issue57._init_sid_embedding_hyperbolic 调用 `expmap0(codebook, c=hyp_c)`. 当 hyp_c=0.74 (正数) 时, `expmap0` 走 Euclidean branch `x / (1 + sqrt(1 + c·‖x‖²))`, 实际是 sphere init 不是 hyperbolic init.
>
> 真 hyperbolic 需要 hyp_c < 0 (例如 -1.0), 走 tanh 分支 `tanh(sqrt(|c|) · ‖x‖/2) · x / (sqrt(|c|) · ‖x‖)`.

**Issue #61 修复方案**:
1. 修复 hyp_c 默认值从 0.74 → -1.0, 走 tanh 分支 (真 hyperbolic)
2. 添加 post-step Riemannian retraction hook (训练 step 后把 SID token embedding 投影回 Poincaré ball ‖x‖ < 1/√|c|)
3. Stage 3 训练 + Stage 4 eval, 跟 #158/#159 Stage 4 结果交叉对比

**Issue #61 §验证标准**:
> "若 R@10 > 0.1030 (+1pp 显著增益) 且优于 #158 random baseline, 则方向 C 真 hyperbolic 路径 GO"
> "若 R@10 ≤ 0.1020 baseline, 方向 C 真 hyperbolic 路径 NO-GO"

---

## 2. R11.4 Dry-run: 修改位置

### 修改 #1: HG_Rec_issue57.py hyp_c 默认值

**位置**: `HG-Rec/model/HG_Rec_issue57.py:104`

**当前代码**:
```python
hyp_c = config.get('hyp_c', 1.0)
```

**修改为**:
```python
hyp_c = config.get('hyp_c', -1.0)  # Issue #61 fix: hyperbolic branch
```

### 修改 #2: HG_Rec_issue57.py config dict default

**位置**: `HG-Rec/model/HG_Rec_issue57.py:161`

**当前代码**:
```python
'hyp_c': 1.0,
```

**修改为**:
```python
'hyp_c': -1.0,  # Issue #61 fix: hyperbolic branch
```

### 修改 #3: 添加 Riemannian retraction hook

**位置**: `HG-Rec/model/HG_Rec_issue57.py` (新增方法)

**新增方法**:
```python
def _riemannian_retract_sid_embedding(self):
    """Post-step retraction: 把 SID token embedding 投影回 Poincaré ball.

    T5 训练用 AdamW (Euclidean), 每 step 后会把 SID tokens 飘出 ball.
    配套 Riemannian retraction: norm = min(norm, (1-eps)/sqrt(|c|)).
    """
    if self.model.shared.weight is None:
        return
    # ... 投影代码 ...
```

**注册 hook 位置**: 训练 loop (HG_Rec.py:training_step 或 trainer.after_step)

### 修改 #4: task159 launcher hyp_c

**位置**: `scripts/task159_issue57_stage3_hyp_init.sh:63`

**当前代码**:
```bash
--hyp_c 0.74 \
```

**修改为**:
```bash
--hyp_c -1.0 \
```

### 修改 #5 (新增 wrapper): Issue #61 实施脚本

**新增文件**: `scripts/task349_issue61_hyp_c_neg1_retraction.py` (~250 行)

**内容**:
- 修改 HG_Rec_issue57.py 上游 (备份 + 修改)
- 添加 Riemannian retraction hook (post-step)
- Stage 3 launcher 改造
- 5/5 sanity 测试 (跟 Issue #62 Gate 0 同模式)

---

## 3. 风险评估

| 风险 | 影响 | 缓解 |
|------|------|------|
| 修改 upstream HG_Rec_issue57.py | 中-高 (跨任务引用) | 不改 HG_Rec.py 主体, 只改 HG_Rec_issue57.py (issue-specific wrapper) |
| 添加 Riemannian retraction hook | 中 (训练 loop 性能) | hook 只在 training step 后调用一次, O(K) 计算, K=9922 |
| hyp_c=-1.0 数值稳定性 | 中 (tanh 分支) | expmap0 已 clamp_min(1e-8), 数值安全 |
| 跟 #158/#159 结果对照 | 低 (独立 run) | 复用同 baseline + Stage 2 SID npy 文件 |
| 训练时间 | 低 (1.5h vs #159 2.3h, 更简单协议) | 200 epoch T5-mini |

---

## 4. ROI 评估

**Issue #61 ROI** (per task482 verdict §后续方向建议 #1):
> "**hyp_c=-1.0 (真双曲)**: 重训 #159 用 hyp_c=-1.0 走 hyperbolic branch, 真正测 hyperbolic init
>   - 时间成本: 2.3h Stage 3 + 30s Stage 4 = ~2.5h
>   - ROI: **极低** (本 verdict 已证 sphere ≈ random, hyp 即使有 marginal 增益也不会突破 baseline)"

**Issue #61 自己的 §验证标准**:
> "若 R@10 > 0.1030 (+1pp 显著增益) → GO"

**R11.5 综合**:
- Issue #61 ROI 极低 (per task482 verdict)
- 但 issue body 已明确实施基础 (#164 audit + R10 backlog #1)
- Owner 接受 Issue #61 (issue 已在 GitHub open)
- 实施成本 ~2.5h Stage 3 + 30s Stage 4 = ~3h GPU

**ROI 不确定性**: task482 verdict 说"极低", 但 #164 找到的是 #57 的实现 bug, 修复后是否产生 hyp vs random 真差异不可预知. 实证主义视角: 真双曲 + retraction 可能产生 marginal gain. 这是 GO 候选但成功概率 < 20%.

---

## 5. 决策请求

**问题**: Issue #61 是否启动 Gate 1 (Stage 3 训练 hyp_c=-1.0 真双曲 + Riemannian retraction)?

**选项 A**: 启动 Gate 1 (~2.5h GPU, ROI 极低但实证主义)
**选项 B**: 跳过 Issue #61 Gate 1, 关闭 issue, 转 Issue #62 Gate 1 (Arm C/D 联合 ablation, ROI 较高)
**选项 C**: 同时启动 Issue #61 Gate 1 + Issue #62 Gate 1 (4 GPU × L40S 全用, ~3h)

**默认建议**: 选项 A (R11.5 自主决策 + owner feedback "每次 loop 必须推进目标")
**备选**: 选项 C (如果 owner 想最大化 GPU 利用率 + 并行 Issue #62 Gate 1)

---

## 6. R11.5 透明决策

**本 dry-run 不实施代码修改**, 仅报告修改位置 + 风险评估.

**为什么**: R11.4 critical decision (modify upstream HG_Rec_issue57.py + 添加 retraction hook) 需要 owner 拍板, 不可 AI 自主.

**下一步**: 等 owner 在本报告下评论 (选项 A/B/C), 或默认按 R11.5 推进选项 A.

---

result: Issue #61 R11.4 dry-run 报告完成, 不实施代码修改. 修改位置 4 处 (#1-#4 upstream HG_Rec_issue57.py + task159 launcher) + 新增 1 文件 (Issue #61 实施脚本). ROI 极低 per task482 verdict, 但 owner accepted Issue #61 需实证. 等 owner 拍板 选项 A/B/C (默认 A: 启动 Gate 1 ~2.5h).