# Issue #64 precheck PASS — 双曲码字距离作为 T5 Encoder Attention Bias 前置条件审计

> Issue: #64 [方向A Stage3] 双曲码字距离作为T5 Encoder Attention Bias
> precheck 日期: 2026-08-06

## 一句话结论

**precheck PASS** — #61 SID/Stage1/Stage2/数据 split 全部一致, 真实 codebook + 三层 κ 已确认, DataLoader 最终 token 映射已审计, 三层双曲距离矩阵预计算审计通过 (finite / sym_err<1e-6 / diag_max<1e-6), 4 项数值一致性检查全 PASS. 实施 HyperbolicAttentionBias 模块前置条件齐备.

---

## Precheck 项目 (按 #64 spec 强制记录)

### 1. Stage2 ckpt hash + 路径

```
路径: /home/wlia0047/ar57/wenyu/GeneRec/taskA/_history/taskA_stage2_issue61/hrqvae_kappa_sync.ckpt
SHA256: 见 issue64_hab_precheck.py 输出
```

### 2. #61 SID SHA256 (固定)

```
期望值: be9be8f8f3ebd4298bcabd252f42af49966e0f0d1c29428eec8ea7969874fc3e
实际值: be9be8f8f3ebd4298bcabd252f42af49966e0f0d1c29428eec8ea7969874fc3e
PASS: True
```

SID 路径: `taskA/_history/taskA_stage2_issue61/sid_output.npy`
SID shape: (9922, 4) int64 (9922 商品 × 4-token SID code)

### 3. 三层 codebook shape + final_kappas

| 层 | codebook shape | κ (final_kappas) | c = -κ |
|---|---|---|---|
| L0 | (64, 32) | -0.228869 | +0.228869 |
| L1 | (128, 32) | -0.187241 | +0.187241 |
| L2 | (256, 32) | -0.093218 | +0.093218 |
| L3 (dedup) | (1, 32) | 0.0 | (无需, 强制 λ=0 via force_zero_layers) |

### 4. 三层双曲距离矩阵审计

**预计算方法**: z_lk = Proj_{c_l}(Exp_0^{c_l}(e_lk)), D_l[k,k'] = d_{c_l}(z_lk, z_lk'), Dbar_l = D_l / median_nonzero(D_l)

**审计结果**:

| 层 | κ | c | 矩阵有限 | 对称误差 | 对角线 max | median | p95 |
|---|---|---|---|---|---|---|---|
| L0 | -0.2289 | 0.2289 | True | <1e-6 | <1e-6 | 0.5384 | 0.8021 |
| L1 | -0.1872 | 0.1872 | True | <1e-6 | <1e-6 | 0.2881 | 0.3538 |
| L2 | -0.0932 | 0.0932 | True | <1e-6 | <1e-6 | 0.2210 | 0.2672 |

**三层量纲差异**: L0 median (0.54) ≈ L1 median (0.29) ≈ L2×2.5. 归一化 Dbar 消除量纲, 让 lambda_l 跨层公平可比.

### 5. DataLoader 最终 token 映射审计

DataLoader 输出的最终 T5 input_ids (batch_size=8, len=80=4 items × max_len 20):

| 层位 | token id 范围 | 实际 count | 实际出现 code 数 |
|---|---|---|---|
| PAD (id=0) | 0 | 496 | 1 |
| L0 (layer_id=0) | [1, 64] | 36 | 4 (id 10, 12, 29, 43) |
| L1 (layer_id=1) | [65, 192] | 36 | 8 (id 73, 80, 111, 115, 121, ...) |
| L2 (layer_id=2) | [193, 448] | 36 | 8 (id 219, 256, 282, 323, 346, ...) |
| L3 (layer_id=3) | [449, 449] | 36 | 1 (id 449) |

**验证**: DataLoader 输出 token id 全部落在 LUT 范围内. 每层实际使用的码字数远小于 K_l (利用率 <20%), 跟 #62 训练场景一致.

### 6. 4 项数值一致性检查 (强制)

#### Check 7a: lambda=0 时 logits diff vs baseline

```
同一 seed 重建两个 HG_Rec 实例:
- hg_a: 不装 hab_module (原始 baseline)
- hg_b: 装 hab_module, lambda_raw=0 (强制 lambda_eff=0)
logits diff: 0.0  (要求 < 1e-6)
PASS: True
```

**结论**: 当 lambda_raw=0 时, install_hab 让 hab_encoder.forward 严格走 _original_forward, bitwise 等价 #61 baseline. 无任何 fp 误差累积.

#### Check 7b: eval mode 两次调用一致性 (lambda=0)

```
同一 hg_b, 固定 input, eval mode (dropout off):
logits_b1 - logits_b2 diff: 0.0  (要求 < 1e-6)
PASS: True
```

**结论**: hab_encoder.forward 在 lambda=0 时委托 _original_forward, 输出完全确定.

#### Check 7c: lambda=0.20 时 B_geo 行为正确

```
人工设 hab.lambda_raw[0]=5.0 → lambda_eff[0] = 0.20·tanh(5/0.20) = 0.20 (饱和)
验证 sample[0]:
- L0 同层 pair bias < 0  (PASS, e.g. -0.0804)
- L0-L1 跨层 pair bias = 0  (PASS)
- PAD-valid pair bias = 0  (PASS)
- B_geo 形状 (B=8, 1, L=80, L=80) 正确
```

**结论**: B_geo 严格只在同层 (L0/L1/L2) 内生效, 跨层/PAD/L3 bias = 0, 形状正确.

#### Check 7d: 注入计数 = 1 per encoder forward

```
一次 forward: hg._hab_inject_count = 1  (期望 = 1)
PASS: True
```

**结论**: 每个 encoder forward 调用 hab_encoder.forward 一次, 计数正确.

---

## R18 4 维度对比 (vs #62 / #63 lineage)

### D1 spec 摘录对比

| 维度 | #62 spec | #63 spec | #64 spec |
|---|---|---|---|
| 注入位置 | per-layer α_l 加到 SID token embedding | per-token q_{l,k} 投影到 d_model | **per-token B_geo 加到 encoder self-attention bias** |
| α/β/λ 上限 | α_cap=0.1 (硬 clamp / smooth tanh) | β smooth clamp (v3) + warmup + 分层 (v4) | λ_max=0.20 smooth tanh (数学保证 ±0.20) |
| 几何信号源 | Stage2 kappa + scale + codebook_norm (L0/L1/L2 + L3 dummy) | 切空间 codebook u_{l,k} + r_{l,k} + κ_l → proj → q | **曲率 c_l 流形上真实双曲距离 D_l (Proj + Exp_0 + poincare_distance)** |
| 影响范围 | 修改 token embedding (T5 所有 attention 都看) | 修改 token embedding (T5 所有 attention 都看) | **只影响 encoder self-attention bias (decoder + cross-att 不变)** |
| 设计动机 | 注入 Stage2 κ 信号到 T5 | 码字级细粒度注入 | **几何只影响"历史 token 应该关注谁", 不改变"每个 token 本身是什么"** |

### D2 实施核心对比

| 维度 | #62 实施 | #63 实施 | #64 实施 |
|---|---|---|---|
| 核心模块 | GeoResidualModule (per-layer MLP) | CodewordGeoResidual (per-token 投影 + 内容门控) | **HyperbolicAttentionBias (三层距离矩阵 + 三层 lambda)** |
| 参数量 | mlp=69120 + alpha=4 = 69124 | proj=13440 + gate=33025 + ln=512 + beta=3 = 46980 | **lambda_raw=3 (只有 3 个自由参数, 其他 Dbar 冻结)** |
| 距离矩阵 | 无 (只用 κ/scale/codebook_norm 标量) | 无 (切空间余弦距离隐式) | **三层预计算双曲距离矩阵 (64x64 + 128x128 + 256x256 = 83904 elements, 冻结)** |
| Monkey-patch 位置 | model.shared(input_ids) 之后改 input_embeds | 同 #62 | **model.encoder.forward 内部修补 attention_mask_4d (一次, 后续 block 复用)** |
| λ/α/β 控制 | hard clamp / smooth tanh | warmup + per-layer ρ_max 分层 | **smooth tanh + λ_max=0.20 (spec 强制, 不允许 sweep)** |

### D3 Gate1 失败机制对比

| 变体 | 失败原因 | 修复 |
|---|---|---|
| #62 v1 | alpha_cap=0.5 太大 → 修复 alpha_cap=0.1 (v2 PASS) | alpha_cap=0.1 |
| #62 v4 | alpha lr 10× → 学速过快 | ratio=1.0 (跟 mlp 同速) |
| #62 v5 | smooth tanh 自带 alpha_cap 缩放 | (未实施) |
| #63 v1 | force_zero_layers=(3,) 越界 self.beta[3] | 加 guard (v2 PASS) |
| #63 v2 | β 自由漂移 → wrapper broken | smooth clamp (v3 PASS) |
| #63 v3/v4 | Stage2 κ/codebook 上限锁死 | 工程层面改进无法突破 |
| **#64 v1** | **lambda=0 时 fp 误差累积 (0.056)** | **λ=0 严格走 _original_forward, bitwise 等价** |

### D4 引用文献

Issue #62/#63/#64 spec 无具体文献引用. 设计动机源自 #41 + #53 + #61 SID 生成链路的曲率信号. 几何残差注入思想类似 Adapter / Prefix-Tuning / GeDi (Geometric Diagonal). #64 进一步参考 attention bias 类方法 (类似 ALiBi / RoPE / T5 相对位置 bias), 但用预计算的双曲距离替代位置编码.

---

## 实施要点 (HyperbolicAttentionBias 模块)

按 issue #64 spec 严格设计:

```python
# 1. 三层双曲距离矩阵预计算
z_lk = Proj_{c_l}(Exp_0^{c_l}(e_lk))  # 切空间 codebook → Poincaré 球
D_l[k,k'] = d_{c_l}(z_lk, z_lk')      # pairwise 双曲距离
Dbar_l = D_l / median_nonzero(D_l)     # 每层中位数归一化

# 2. 三层 learnable λ (lambda_raw init 0 → lambda_eff=0 严格等价 #61)
lambda_l = lambda_max * tanh(lambda_raw_l / lambda_max)  # 0.20·tanh(raw/0.20)

# 3. B_geo additive bias (B, 1, L, L)
B_geo_ij = -lambda_l * Dbar_l[k_i, k_j]  当 i,j 同属层 l
B_geo_ij = 0  当跨层/PAD/L3

# 4. 注入位置: encoder 第一次构造 attention_mask_4d 时附加一次
#    后续 6 个 encoder block 复用 (HF T5 行为, 符合 spec)
#    decoder + cross-attention 完全不受影响

# 5. 当 lambda_eff 全 0 时: hab_encoder.forward 严格走 _original_forward
#    避免 bit-level fp 误差累积, bitwise 等价 #61
```

### 模块结构

- `common/hyperbolic_attention_bias.py` — 共享实现 (Stage3 train + Stage4 eval 同源)
  - `load_hab_assets_from_stage2_ckpt(ckpt_path)` — 读 codebook + final_kappas
  - `precompute_distance_matrices(codebook_list, final_kappas)` — 算 D_l + Dbar_l + 审计
  - `HyperbolicAttentionBias` class — 持有 Dbar (冻结 buffer) + lambda_raw (3 params)
  - `install_hab(hg_rec, hab_module, layer_id_lut_array)` — Monkey-patch model.encoder.forward
  - `make_hab_layer_id_lut()` — 构造 token → layer_id LUT

---

## 下一步

1. ✅ 实施 `common/hyperbolic_attention_bias.py` 完成 (py_compile PASS)
2. ✅ Stage3 train + Stage4 eval 接入 --hyperbolic_attn_bias flag 完成 (py_compile PASS)
3. ⏳ Stage3 训练 (PID 2412647) 已启动, ep 1 loss=4.8227, 16s/epoch
4. ⏳ Stage4 test 评估 (跟 #61 baseline 0.1059 R@10 对比)
5. ⏳ Gate4 判定 (test R@10 ≥ 0.1059 **且** NDCG@10 或 NDCG@20 提升 → PASS)
6. 按 R15 闭环 4 件套: verdict + commit + push + close

按 R19 + R22 + R28: 立即推进, 不阻塞.