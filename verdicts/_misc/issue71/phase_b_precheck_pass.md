# Issue #71 Phase B Precheck PASS

## ΔD 数值审计 (Stage2 issue61 ckpt, κ=[-0.229, -0.187, -0.093])

| Layer | κ | D_hyp median | D_flat median | ΔD median | ΔDbar range |
|-------|-----|--------------|---------------|-----------|-------------|
| L0 | -0.229 | 0.5384 | 0.5330 | 0.0046 | [0.18, 3.33] |
| L1 | -0.187 | 0.2881 | 0.2879 | 0.00018 | [0.22, 1.70] |
| L2 | -0.093 | 0.2210 | 0.2210 | 0.000041 | [0.20, 2.15] |

**全 finite / sym_err < 1e-6 / diag_max < 1e-7, PASS**

## ΔD 设计论证

- **ΔD = D_hyp(c_l) - D_flat(c_flat=1e-6)** 剥离码字间几何距离的尺度, 只保留曲率对距离的非线性贡献
- **L0 曲率最强 (κ=-0.229)** → ΔD 最大 (median 0.0046) → T5 学到强曲率信号
- **L2 曲率最弱 (κ=-0.093)** → ΔD 几乎为 0 (median 4.1e-5) → 接近 flat, 贡献微乎其微
- **median 标准化**: ΔDbar = ΔD / median(|ΔD|_nonzero), 范围 [0.18, 3.33] 量纲与历史 Dbar 接近, T5 λ_max=0.20 仍有效

## 为什么 ΔD 比 D_hyp 更有意义

1. **完整 D_hyp 包含两层信息**: 码字间几何距离 + 曲率贡献. T5 学 bias 时两个信号混杂.
2. **ΔD 只保留曲率贡献**: 剥离码字距离后, T5 学到的是纯曲率几何. valid/test 相关性更纯净.
3. **匹配 λ_max 甜点**: 完整 D_hyp 范围 [0, 1.5] (归一化后), ΔD 范围 [-, 3.3] 略大但 median=1 与历史对齐.

## Phase B v80 配置 (基于 v77 + ΔD)

```
torchrun --nproc_per_node=4 common/stage3/stage3_train_pure_t5.py \\
  --product_dir taskA/_history/issue71_v80_delta_curvature_hab \\
  --sid_npy taskA/_history/taskA_stage2_issue61/sid_output.npy (not 06af0fed npy from taskA_stage2_hyp_v2_capmatch_1000ep/) \\
  --enable_residual_hab --residual_alpha_init -20.0 --hab_lambda_max 0.20 \\
  --hab_delta_curvature
```

**注意**: SID 06af0fed 由 taskA_stage2_hyp_v2_capmatch_1000ep/sid_output.npy 产生, 但历史 v77/v78/v79 实际 Stage4 eval 走默认 --hab_stage2_ckpt (issue61/hrqvae_kappa_sync.ckpt, κ=[-0.23, -0.19, -0.09]). Phase B v80 必须保持一致才能复现.

## 产物

- precheck 脚本: verdicts/issue71_phase_b_precheck.py (PASS)
- 代码改动: common/hyperbolic_attention_bias.py + common/stage3/stage3_train_pure_t5.py + common/stage4/stage4_eval_pure_t5.py (待 commit)
- 预期: 曲率信号比 D_hyp 更纯粹, ratio 维持 ~1.21, test R@10 突破 v77 0.1080 → 0.11

## 时间

2026-08-07 13:35 — Phase B precheck 完成, 数值稳定, 启动 v80 训练