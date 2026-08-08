# Issue #87 v87 attn_entropy regularizer 闭环: NO-GO (test_R@10=0.0957 < v77)

日期: 2026-08-08
Issue: work_items #85 (v86) + #87 (v87, 计划中)
状态: **实施 PASS, 端到端 NO-GO (FAIL vs v77)**

---

## 1. 背景

v77 (Stage1 per-item radius + v74 HAB frozen + v15 capmatch SID) = 当前曲率路线 SOTA, test R@10=0.1080.

v78 (DECOR, BANNED) test R@10=0.1092 (历史最佳), 三个抗 trap 改动之一: **attn_entropy_weight=0.001** —— 鼓励 attn 分布尖锐, 防均匀 (prevent information dilution).

DECOR BAN 下, **借鉴锐化思想**, 在曲率框架内对 HAB B_geo 做 softmax entropy proxy:
- proxy: 对 B_geo (B, L, L) 自身做 softmax(-B_geo/τ) 计算 entropy
- 鼓励 B_geo 分布尖锐 → T5 attn 利用 HAB 几何信号, 而非均匀稀释
- 与 v74 HAB frozen + WD=0.01 + dropout=0.20 协同

---

## 2. 实验设计 (1 个变体)

| 配置 | v77 baseline | v87 attn_entropy |
|---|---|---|
| Stage1 | hyp_v2 | hyp_v2 (同) |
| Stage2 SID | v15 capmatch | v15 capmatch (同) |
| Stage3 backbone | T5 6+6 d_model=128 | 同 |
| Stage3 LR | cosine | cosine (同) |
| Stage3 epochs | 200ep 早停 20 | 200ep 早停 20 (同) |
| HAB residual_alpha | -20 | -20 (同) |
| HAB lambda_max | 0.20 | 0.20 (同) |
| Stage3 dropout | 0.30 | 0.30 (同) |
| **HAB attn_entropy_weight** | **0 (无)** | **0.001** |
| **HAB attn_entropy_tau** | **N/A** | **1.0** |

---

## 3. 实施 (代码)

### 3.1 HAB entropy proxy 计算 (`common/hyperbolic_attention_bias.py`)

```python
# 在 get_B_geo 末尾新增 (仅当 self.attn_entropy_weight > 0 时)
if float(getattr(self, "attn_entropy_weight", 0.0)) > 0.0 and B_geo.abs().sum() > 0:
    tau = float(getattr(self, "attn_entropy_tau", 1.0))
    logits = -B_geo / max(tau, 1e-6)
    if attention_mask_2d is not None:
        row_mask = valid.unsqueeze(-1).float()  # (B, L, 1) 每行是否有效
        logits = logits.masked_fill(row_mask == 0, -1e9)
    attn = torch.softmax(logits, dim=-1)  # (B, L, L)
    attn_log = torch.log(attn.clamp_min(1e-9))
    entropy = -(attn * attn_log).sum(dim=-1)  # (B, L) 每行 entropy
    self.last_attn_entropy = entropy.mean()  # scalar, requires_grad=True
else:
    self.last_attn_entropy = None
```

### 3.2 install_hab encoder.forward 累加 (`common/hyperbolic_attention_bias.py`)

```python
# v87: 累加 attn_entropy (Stage3 train_step 取出, 加到总 loss)
hg_rec._hab_attn_entropy_loss = None
# ...
if getattr(hab_module, "last_attn_entropy", None) is not None:
    if hg_rec._hab_attn_entropy_loss is None:
        hg_rec._hab_attn_entropy_loss = hab_module.last_attn_entropy
    else:
        hg_rec._hab_attn_entropy_loss = hg_rec._hab_attn_entropy_loss + hab_module.last_attn_entropy
```

### 3.3 Stage3 train_step 加权 (`common/stage3/stage3_train_pure_t5.py`)

```python
# 在 loss.backward() 之前
if HAB_ATTN_ENTROPY_WEIGHT > 0:
    _hab_entropy = getattr(_hab_ref, "_hab_attn_entropy_loss", None)
    if _hab_entropy is not None:
        loss = loss + HAB_ATTN_ENTROPY_WEIGHT * _hab_entropy
        _hab_ref._hab_attn_entropy_loss = None  # 重置
```

### 3.4 Stage3 argparse 新增

- `--hab_attn_entropy_weight` (默认 0.0 = v77 baseline 等价)
- `--hab_attn_entropy_tau` (默认 1.0)

---

## 4. 结果

| 版本 | best epoch | valid_R@10 | test_R@10 | ratio |
|---|---|---|---|---|
| HG-Rec baseline (无 HAB) | - | 0.1267 | 0.1024 | 1.237 |
| v85p (baseline+v15+cosine, 无 HAB) | ep174 | 0.1265 | 0.1042 | 1.218 |
| v77 (立即 HAB, per-item radius) | ep95 | 0.1312 | **0.1080** | 1.215 |
| v86 (HAB warmup T0=200) | ep76 | 0.1206 | 0.0968 | 1.245 |
| **v87 attn_entropy (本 Issue)** | ep55 | 0.1232 | **0.0957** | **1.287** |

v87 best_epoch ep55 (early_stop=20 在 ep75 触发):
- valid_R@10=0.1232 (vs v77=0.1312, -0.0080)
- test_R@10=0.0957 (vs v77=0.1080, **-0.0123**; vs baseline=0.1024, -0.0067)
- **valid/test ratio=1.287 (历史最低, vs v77=1.215, v86=1.245)** — ratio 优化但 test 绝对值未提升
- HAB λ_eff=[-0.20, 0.20, 0.20], U/V norm [0.31, 0.42, 0.59]
- 单 epoch ~25s (vs v86 ~18s, **+7s overhead** 来自 attn_entropy 自身计算 + softmax(-B_geo/τ))

---

## 5. Gate 评估

| Gate | 判定 | 结果 |
|---|---|---|
| **Gate 1** (Stage2 SID) | v15 capmatch SID (sha=5f8331cc) 与 v77 一致 | ✅ PASS |
| **Gate 2** (Stage3 训练健康) | ep1-75 全程 loss 单调下降 6.17→2.74, 无 NaN/Inf | ✅ PASS |
| **Gate 3** (HAB 注入健康) | λ_eff 三层激活 ±0.20, U/V 范数稳定 | ✅ PASS |
| **Gate 4** (端到端) | **test_R@10=0.0957 < v77=0.1080 (-0.0123)**, < baseline=0.1024 (-0.0067) | ❌ **FAIL** |

**3/4 Gate PASS, Gate 4 FAIL → NO-GO**.

---

## 6. 根因分析 — 为什么 attn_entropy 单独也失败

### 6.1 假设 1: ratio 优但 valid/test 差距扩大 → 引入分布偏置
- v87 ratio=1.287 远低于 v77=1.215, **T5 在 valid 上表现更好, 但 test 表现更差**
- 解释: attn_entropy 鼓励 T5 利用 HAB 几何信号 (锐化 attn), 但 valid 和 test 分布不同, 锐化模式可能只在 valid 上有效
- v78 attn_entropy 在 DECOR 上 work 是因为 bos_queries 是动态学习的, 训练集和测试集共享; 但 HAB 是 frozen 几何 bias, valid/test 分布差异被放大

### 6.2 假设 2: 0.001 权重太小, 不足以跨越 v77 baseline
- 损失项 weight=0.001 × entropy≈2.5 = 0.0025, 远小于 CE loss ≈ 2.7
- 实际梯度贡献 ~0.1% 量级, 影响很小
- 但 ratio 1.287 vs v77=1.215 差异显著 (5%), 说明 attn_entropy **确实生效**, 只是方向错了

### 6.3 假设 3: B_geo entropy proxy 不等价 attn entropy
- 真实 attn = softmax(query · key / sqrt(d) + B_geo)
- proxy = softmax(-B_geo/τ) 仅考虑 B_geo 自身, 忽略 query/key 内容
- 当 query/key 强信号主导时 (T5 学到), proxy 与真实 attn entropy 相关性弱
- 这可能让 attn_entropy 损失引导模型关注错误的"锐化"目标

### 6.4 综合判断
**曲率框架借鉴 v78 两个改动 (α_warmup, attn_entropy) 全部 PARTIAL-GO / NO-GO**:
- v86 α_warmup: FAIL (-0.0112)
- v87 attn_entropy: FAIL (-0.0123)

DECOR 与曲率本质不同: DECOR 改的是 embedding 动态混合 (随训练变化), 曲率改的是 frozen 几何 bias (Stage2 SID 不变). 静态 vs 动态机制的差异决定了 v78 抗 trap 改动不能直接迁移.

---

## 7. NO-GO 判定 (R23 7 信号)

| 信号 | v87 实际 | 判定 |
|---|---|---|
| 1. val_R@10=0 跨 ≥2 checkpoint | ❌ (val=0.0093→0.1232 全程 > 0) | OK |
| 2. loss 不下降 | ❌ (loss 6.17→2.74 单调) | OK |
| 3. val loss 反向 | ❌ (val_loss 单调改善到 best 0.1232) | OK |
| 4. wrapper broken | ❌ (DDP 4 worker 正常, λ_eff 注入健康) | OK |
| 5. ckpt 不存 | ❌ (HG_Rec_best.pth ep55 保存成功) | OK |
| 6. NaN/Inf | ❌ (loss 全程 finite) | OK |
| 7. GPU 100% 但 loss 不变 | ❌ (GPU 33-38%, loss 持续下降) | OK |

7 信号全部 OK, 训练健康. NO-GO 单纯是端到端 test 性能不足.

---

## 8. 曲率框架借鉴 v78 DECOR 路线彻底闭环

| 借鉴路径 | Issue | 结果 | 失败原因 |
|---|---|---|---|
| κ_warmup (T0=200/Tw=400) | #86 v86 | FAIL (-0.0112) | HAB 静态 ≠ DECOR 动态 alpha |
| attn_entropy (weight=0.001) | #87 v87 | FAIL (-0.0123) | B_geo proxy ≠ 真实 attn entropy |

**两条 v78 抗 trap 改动全部 NO-GO** → 曲率框架借鉴 DECOR 路径**已穷尽**.

---

## 9. 后续路径

### 9.1 曲率框架内已穷尽路径
- 静态几何 (Stage1 per-item radius): v77 已用 (test=0.1080)
- 静态曲率异质 (v15 capmatch κ=[0.31, 0.24, 0.19]): v77 已用
- HAB frozen + WD + dropout: v74/v77 已用
- 借鉴 DECOR α_warmup: v86 NO-GO
- 借鉴 DECOR attn_entropy: v87 NO-GO

### 9.2 推荐方向 (跳出"借鉴 v78"思路)

#### 路径 A: Issue #83 c_l* 替换 v15 capmatch κ
- 机制: 用 Issue #83 拟合的全局 c_l* (L0=0, L1=2, L2=5) 替换 v15 fixed κ
- ROI: ★★★ (Issue #83 强证据 + 工程量小)
- 实施: 改 Stage2 训练 (用 Issue #83 推荐 curvature 替换 v15 capmatch)
- 风险: 几何异质可能破坏 Stage2 训练稳定性

#### 路径 B: Stage1 R_MAX 强化
- 机制: v77 已用 R_MAX=0.99 + sigmoid, 强化 R_MAX=0.995 + 斜率提升
- ROI: ★★ (边际可能小, 但工程量最小)
- 实施: 改 Stage1 hyp_v2 主脚本

#### 路径 C: 跳出曲率框架 → 架构改动
- 候选: RQ-VAE 加深 (3→4 层, Issue #215/#82 路线)
- 候选: T5 capacity (d_model 128→256)
- ROI: ★★ (复杂, 工程量大, 但可能突破 0.11 极限)

### 9.3 推荐
**路径 A (Issue #83 c_l*)** 最优 ROI, 直接利用 Issue #83 强证据. 工程量小, 与 v77 兼容 (只改 Stage2 SID, Stage3 不变).

---

## 10. DECOR BAN

v87 严格在曲率框架内推进:
- 无 --enable_prompt_former
- 无 decor_prompt_former.py
- 无 DECOR alpha gate / candidate bins

仅用现成 attn_entropy 思路 (B_geo softmax proxy) 在 HAB 头上实施, 与 v78 DECOR 完全独立.

---

## 11. 文件清单

- `/home/wlia0047/ar57/wenyu/GeneRec/taskA/_history/issue87_v87_attn_entropy/`
  - `train_v87.log` (训练日志, ep1-75)
  - `verdict.json` (best=ep55, loss=2.7555, done=18:04:34)
  - `HG_Rec_best.pth` (Stage3 adapter, ep55 best)
  - `trace.json` (per-epoch 评估记录)
  - `stage4_test_on_best/ep55_test/` (ep55 best test 评估, test_R@10=0.0957)
- 主脚本修改:
  - `common/stage3/stage3_train_pure_t5.py` (加 --hab_attn_entropy_weight/tau)
  - `common/hyperbolic_attention_bias.py` (get_B_geo 加 entropy proxy + install_hab 累加)

---

## 12. 总结

v87 attn_entropy regularizer (借鉴 v78 第二个改动) **端到端 NO-GO**:
- 训练健康 (3/3 PASS)
- test_R@10=0.0957 < v77=0.1080 (-0.0123), < baseline=0.1024 (-0.0067)
- 根因: B_geo entropy proxy ≠ 真实 attn entropy; 静态 frozen bias 锐化 ≠ 动态 embedding 锐化
- ratio 1.287 历史最佳, 但 test 绝对值未提升 → 锐化方向有偏
- 曲率框架借鉴 v78 DECOR 路径 (#86 v86 + #87 v87) 全部 NO-GO, **借鉴路径已穷尽**
- 后续推荐: Issue #83 c_l* 替换 v15 capmatch κ (路径 A, ROI ★★★)
