# Task #348 / Issue #62 Gate 0 — #30+#43 联合 wrapper 设计 + sanity 5/5

**日期**: 2026-07-31
**触发**: Issue #62 [放弃方向 C + #30+#43 联合 ablation] 27+ 方向 NO-GO 收口后唯一 ROI > 0 路径
**类型**: Issue #62 Gate 0 (零 GPU, low-cost)
**R11.4 dry-run 报告**:
- 修改位置: 组合 Issue #30 wrapper (`PerLayerCodebookTransformHRQVAE`, scripts/task301_issue30_gate0_codebook_transforms.py) + Issue #43 wrapper (`HRQVAEWithHypPre`, scripts/task334_issue43_gate2a_hyp_pre_encoder.py), 不修改 HG-Rec upstream
- 新增脚本: `scripts/task348_issue62_gate0_joint_wrapper.py` (~250 行)
- 实施方式: 复用两个 PASS wrapper 作为组件, 新建组合 wrapper `HRQVAEWithHypPreAndPerLayerTransforms`
- 风险评估: 两个 wrapper 都是 PASS 状态, 组合是单调扩展, 不引入新公式 bug

---

## 1. 设计目标

**组合 Issue #30 + Issue #43 两个 GO 端点**:
- **Issue #30** (per-layer Codebook Transforms r_l=[0.1,1,10]+s_l=[2,2,2]): Stage 1/2 per-layer 异构 codebook 几何, R@10=0.1022 marginal GO
- **Issue #43** (HypPreEncoder expmap0 c=0.74): Stage 1/2 预量化双曲感知映射, R@10=0.1042 best GO ⭐

**两个端点独立有效, 从未联合测试过**. 假设两者在 Stage 1/2 范畴内互补增益.

---

## 2. 联合 wrapper 设计

```python
class HRQVAEWithHypPreAndPerLayerTransforms(nn.Module):
    """Joint Issue #30 + #43 wrapper, composition only (no upstream patch).

    Pipeline:
        x_768d → [Issue #43] HypPreEncoder(expmap0 c=0.74) → encoder
              → [Issue #30] per-layer Codebook Transforms (radius+rotation+scale)
              → hrq (RQ-VAE) → decoder
    """
    def __init__(self, base_hrqvae, hyp_c=0.74, hyp_enabled=True,
                 radius_list=[1.0, 1.0, 1.0],  # identity when Issue #43 only
                 rotation_list=None,            # identity by default
                 scale_list=[1.0, 1.0, 1.0]):  # identity when Issue #43 only
        super().__init__()
        self.base = base_hrqvae
        self.hyp_pre = HypPreEncoder(c=hyp_c, enabled=hyp_enabled)
        # Per-layer transform matrix (e_dim × e_dim) for each layer
        n_layers = len(base_hrqvae.num_emb_list)
        e_dim = base_hrqvae.e_dim
        if rotation_list is None:
            rotation_list = [torch.eye(e_dim) for _ in range(n_layers)]
        self.radius_list = list(radius_list)
        self.rotation_list = list(rotation_list)
        self.scale_list = list(scale_list)

    def forward(self, x, use_sk=True):
        # Step 1: HypPre (Issue #43)
        x = self.hyp_pre(x)
        # Step 2: Issue #30 per-layer transforms (apply to vq_layers forward)
        # Compose: HypPre output → encoder → per-layer transform
        return self._patched_vq_forward(x, use_sk=use_sk)

    def _patched_vq_forward(self, x, use_sk=True):
        # Save originals
        originals = []
        for li, q in enumerate(self.base.hrq.vq_layers):
            originals.append(q.forward)
            eff = (self.scale_list[li] * self.radius_list[li]) * self.rotation_list[li]
            eff = eff.to(device=q.embeddings.weight.device, dtype=q.embeddings.weight.dtype)
            orig_fwd = q.forward
            def make_patched(orig, ef):
                def patched(_self, x, use_sk=True):
                    orig_w = _self.embeddings.weight.data.clone()
                    _self.embeddings.weight.data = orig_w @ ef.t()
                    try:
                        return orig(x, use_sk=use_sk)
                    finally:
                        _self.embeddings.weight.data = orig_w
                return patched
            q.forward = make_patched(orig_fwd, eff).__get__(q, type(q))
        try:
            return self.base(x, use_sk=use_sk)
        finally:
            for li, q in enumerate(self.base.hrq.vq_layers):
                q.forward = originals[li]
```

---

## 3. Gate 0 Sanity 测试 (5/5)

| Test | 内容 | 期望 |
|------|------|------|
| T1 | baseline (no wrapper) | output.shape == input.shape, RQ loss finite |
| T2 | HypPre only, per-layer identity | output == HypPre direct, ‖y‖ < 1/√c |
| T3 | Per-layer identity (r=1, R=I, s=1) only | output == Issue #30 regression (Issue #30 wrapper alone) |
| T4 | Both wrappers active (Issue #30 design + Issue #43) | output != baseline, both transforms compose |
| T5 | monkey-patch 干净恢复 (no leakage) | second call identical to first |

Gate 0 PASS 条件: 5/5 PASS.

---

## 4. 不通过决策

- Gate 0 FAIL → Issue #62 关闭, 写 verdict 承认联合 wrapper 实施基础有 bug, 不进入 Gate 1.
- Issue #30 + Issue #43 各自仍独立 GO 端点 (R@10=0.1022 / 0.1042), 不影响.

---

## 5. 后续阶段

- Gate 0 PASS → Gate 1 (Stage 1/2 重新训练 ~6h, 2 arms × 3h):
  - Arm C: #30 + #43 联合 wrapper
  - Arm D: #30 + #43 + K0=256 三联合 (K-sweep 探索)
- Gate 1 Stage 1 完成 → Gate 2 Stage 3 T5-mini 训练 + Stage 4 eval (~3h)
- 总 GPU 时间: ~9h (4 GPU × L40S 可并行)

---

result: Issue #62 Gate 0 PASS (5/5 sanity) → 进入 Gate 1 Stage 1/2 重新训练 (~6h). Gate 0 FAIL → Issue #62 关闭.