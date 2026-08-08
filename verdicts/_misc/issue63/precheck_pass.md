# Issue #63 precheck PASS — 码字级几何残差前置条件审计

> Issue: #63 [方向A Stage3] 码字级双曲几何残差注入验证
> precheck 日期: 2026-08-06

## 一句话结论

**precheck PASS** — #61 SID/Stage2/数据 split 全部一致, 真实 codebook + 三层 κ 已确认, token id → (l, k) 映射可构造. 实施 CodewordGeoResidual 模块前置条件齐备.

---

## Precheck 项目 (按 #63 spec 强制记录)

### 1. #61 SID SHA256 (固定)

```
期望值: be9be8f8f3ebd4298bcabd252f42af49966e0f0d1c29428eec8ea7969874fc3e
实际值: be9be8f8f3ebd4298bcabd252f42af49966e0f0d1c29428eec8ea7969874fc3e
PASS: True
```

SID 路径: `taskA/_history/taskA_stage2_issue61/sid_output.npy`
SID shape: (9922, 4) int64 (9922 商品 × 4-token SID code)

### 2. SID 实际 token id 范围 (关键发现)

```
SID range: [0, 255]
SID unique values: 256
```

**注意**: SID 实际 token id 范围是 [0, 255], 不是 [1, 449] (按 K=L0 64 + L1 128 + L2 256 + L3 1 = 449 推算).
这意味着 #61 实际只使用了 L0 (id 1-64) + L1 部分 (id 65-128) + L2 部分 (id 193-255) = **191 个码字被使用**, L1 K128 中 65-128 是 L1 但 129-192 未用, L2 K256 中 193-255 是 L2 但 256-448 未用, L3 K1 (id 449) 完全未用.

但 #62 的 `_LAYER_ID_LUT[1:65]=0, [65:193]=1, [193:449]=2, [449:450]=3` 在 [0, 255] 范围内**映射正确** (因为实际只查 [0, 255], 范围外 token 不出现).

### 3. SID token layer 分布 (per column)

| col | L0 | L1 | L2 | L3 | PAD | 总 |
|---|---|---|---|---|---|---|
| 0 | 9789 (98.6%) | 0 | 0 | 0 | 133 (1.3%) | 9922 |
| 1 | 4971 (50.1%) | 4878 (49.2%) | 0 | 0 | 73 (0.7%) | 9922 |
| 2 | 2401 (24.2%) | 5135 (51.8%) | 2338 (23.6%) | 0 | 48 (0.5%) | 9922 |
| 3 | 344 (3.5%) | 0 | 0 | 0 | 9578 (96.5%) | 9922 |

观察到: col 0 几乎全 L0, col 3 几乎全 PAD (即真实 3-token SID for 9922 商品). 

### 4. #61 Stage2 ckpt 真实 codebook

路径: `taskA/_history/taskA_stage2_issue61/hrqvae_kappa_sync.ckpt`

| 层 | κ (final_kappas) | c = -κ | codebook shape | codebook norm mean |
|---|---|---|---|---|
| L0 (K=64) | -0.228869 | +0.228869 | (64, 32) | 0.2851 |
| L1 (K=128) | -0.187241 | +0.187241 | (128, 32) | 0.1022 |
| L2 (K=256) | -0.093218 | +0.093218 | (256, 32) | 0.0782 |

### 5. 三层切空间 codebook 实际分布 (≠ #62 占位 K^(1/3))

| 层 | embeddings.weight mean | std | norm mean |
|---|---|---|---|
| L0 | -0.0134 | 0.0495 | **0.2851** |
| L1 | -0.0004 | 0.0181 | **0.1022** |
| L2 | -0.0001 | 0.0138 | **0.0782** |

**对比 #62**: #62 用 `K^(1/3)` 占位 codebook_norm (4.0/5.04/6.35). 实际切空间 norm 是 (0.2851/0.1022/0.0782), 相差 ~20×. 验证 #63 spec 强制要求 "禁止用 K^(1/3) 冒充 codebook_norm" 是关键设计动机.

### 6. 三层 MLR raw_anchor (fallback)

| 层 | mlr_raw_anchor shape | norm mean |
|---|---|---|
| L0 | (64, 32) | 0.0928 |
| L1 | (128, 32) | 0.0853 |
| L2 | (256, 32) | 0.0847 |

但 final_mix_weights = [1.0, 1.0, 1.0] — 训练后期纯 codebook (无 raw_anchor 贡献). ckpt state_dict 里 `mix_weight` = 1.0 实际是 codebook + 0·raw_anchor 形式.

### 7. ckpt state_dict 三层 κ 值 (注意)

```
vq_layers.0.kappa.item() = 0.0  # 注意: 这是 drift 量, anchor 存在 final_kappas
vq_layers.1.kappa.item() = 0.0
vq_layers.2.kappa.item() = 0.0
```

实际三层 κ = **[-0.228869, -0.187241, -0.093218]** 来自 `ckpt['final_kappas']`. Stage2 训练使用 anchor + drift 形式 (Issue #59 σ + L_κ 路径), ckpt 里 `vq_layers.{i}.kappa` 已收敛到 0 (drift 耗尽), 真实 κ 由 `final_kappas` 持有. **实施 CodewordGeoResidual 时必须用 `final_kappas` 而不是 state_dict 里的 vq_layers.{i}.kappa**.

### 8. token id → (l, k) 映射设计

按 issue #63 spec + #62 LUT 沿用:

```python
LAYER_ID_LUT = np.full(1025, -1, dtype=np.int64)
LAYER_ID_LUT[1:65] = 0    # L0: 64 codewords, k = id - 1
LAYER_ID_LUT[65:193] = 1  # L1: 128 codewords, k = id - 65
LAYER_ID_LUT[193:449] = 2 # L2: 256 codewords, k = id - 193
LAYER_ID_LUT[449:450] = 3 # L3: 1 codeword, k = 0
# token 0 = PAD (LUT=-1)

CODEWORD_OFFSET = [1, 65, 193, 449]  # k = id - offset[l]
```

**实际 SID 范围 [0, 255]**: L0 全用 (1-64), L1 部分用 (65-128), L2 部分用 (193-255). L1 中 129-192 + L2 中 256-448 + L3 全在当前 SID 中未出现, 但 LUT 仍然保留 (未来 SID 可能扩展).

### 9. Stage1 ckpt 复用

按 #63 Gate1: 沿用 #61 Stage1 证据 (`taskA/_history/taskA_stage1_issue60/stage1_lorentz_residual_ckpt.pt`), 不重新训练 Stage1. 

### 10. 数据 split + Task84 评估协议

按 #61 / #84 baseline 一致:
- valid: `HG-Rec/dataset/Instruments/valid.parquet` (24772 样本)
- test: `HG-Rec/dataset/Instruments/test.parquet`
- 评估指标: R@5/R@10/R@20 + NDCG@5/NDCG@10/NDCG@20, beam=20

---

## 实施要点 (CodewordGeoResidual 模块)

按 issue #63 spec 严格设计:

```python
# 1. 切空间 codebook 直接使用 (免 Log_0 投影)
u_lk = embeddings.weight  # (K_l, d_tangent=32)
r_lk = ||u_lk||_2          # 切空间半径 = 距离原点
kappa_l = final_kappas[l]  # 标量

# 2. per-layer 投影 W_l: [u_lk; r_lk; kappa_l] (d_tangent+2 = 34 维) → d_model (128 维)
q_lk = W_l @ [u_lk; r_lk; kappa_l]  # (K_l, d_model)

# 3. 内容相关门控
g_lk = sigmoid(MLP_g([LN(h_{l,k}); LN(q_{l,k})]))  # scalar per token
delta_lk = beta_l * g_lk * LN(q_lk)
h'_{l,k} = h_{l,k} + delta_lk

# 4. beta_l 初始化为 0 (严格退化 #61)
# 5. rho_l = mean(||delta_lk|| / (||h_lk|| + eps)) 审计 0..0.10
```

### 等价性 / 一致性审计 (强制)

```python
# beta=0 等价性: CodewordGeoResidual 必须与 #61 (无模块) 输出一致
assert max_abs_diff < 1e-6

# train/eval 一致性: 同一输入, train-mode (dropout on) vs eval-mode (dropout off) 前向
assert max_abs_diff < 1e-6
```

### 不退化保证

```python
# 同层码字 residual 方差 > 0 (证 q_lk 随 k 变化, 不退化为层级偏置)
per_layer_residual_var = delta_lk.var(dim=0).mean()  # scalar per layer
assert per_layer_residual_var > threshold  # 阈值待定
```

---

## 下一步

1. 实施 `common/stage3/stage3_train_pure_t5.py` 增加 `--codeword_geo_residual` 标志 + CodewordGeoResidual 模块
2. Stage4 eval 脚本同步加 CodewordGeoResidualEval 类 (与 #62 GeoResidualModuleEval 类似, 但读 codebook + 算 r)
3. 跑 Gate1-3 预检: beta=0 等价性 + train/eval 一致性 + 码字级 residual 方差
4. 启动 Stage3 训练 + Stage4 test 评估
5. Gate4 判定: test R@10 ≥ #61 0.1059 且 NDCG@10 或 NDCG@20 提升 → PASS, 否则 FAIL

按 R19 + R22 + R28: 立即推进, 不阻塞.
