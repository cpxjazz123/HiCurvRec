# Task #151 — Issue #57 方向C: T5 混合曲率 attention Gate 0 设计 (zero-GPU prep)

**日期**: 2026-07-31 00:05
**状态**: 📝 DESIGN — Gate 0 启动就绪 (Stage 3 训练 ~2h 单 GPU)
**Issue**: https://github.com/WENYULIANG123/GeneRec/issues/57

## 1. 目的

Issue #57 验证"传输损耗在 T5 侧"假说: Stage 1/2 在双曲空间调好几何 → 量化生成离散 SID token → Stage 3 喂给标准欧式 T5 → SID token 对 T5 是普通 categorical token, T5 完全不知道几何信息. Gate 0 (低成本): SID token embedding 用双曲坐标初始化 vs 随机初始化, 对比 test R@10.

## 2. Gate 0 实施方案

### 2.1 改动点 (单点 patch, ~10 lines)

`HG-Rec/model/HG_Rec.py` line 25 之后插入:

```python
# Issue #57 Gate 0: SID token embedding initialization with hyperbolic coords
if config.get('sid_embedding_init', 'random') == 'hyperbolic':
    code_path = config['code_path']  # baseline SID file
    sid_arr = np.load(code_path)  # (N, 4) int array, K=256 + 4 digits
    codebook = build_codebook_from_sid(sid_arr, K=config['codebook_size'][0])  # (K, d_model)
    # Apply expmap0(c·x) to push to Poincaré ball
    c = config.get('hyp_c', 1.0)
    codebook_hyp = expmap0(torch.tensor(codebook * c))
    # Initialize shared.weight (T5 token embedding) with hyperbolic coords for SID tokens
    sid_token_start = config.get('sid_token_start', 1)  # 0 = pad
    sid_token_end = sid_token_start + config['codebook_size'][0] * config['num_layers']
    with torch.no_grad():
        self.model.shared.weight[sid_token_start:sid_token_end] = codebook_hyp
        self.model.lm_head.weight[sid_token_start:sid_token_end] = codebook_hyp  # T5 tied embedding
```

### 2.2 CLI flag

```python
parser.add_argument('--sid_embedding_init', type=str, default='random', choices=['random', 'hyperbolic'])
parser.add_argument('--hyp_c', type=float, default=1.0)
```

### 2.3 数学基础

- **Random init**: T5 default Xavier/Kaiming uniform init for token embedding
- **Hyperbolic init**: 把 SID token 对应的 codebook 向量先 expmap0(c·x) 推到 Poincaré ball, 然后填入 embedding. 维度必须匹配 (d_model=128 for T5-mini 5.5M, d_model=512 for T5-small).

### 2.4 兼容性约束

- T5-mini 5.5M: d_model=128 (跟 task84 / Issue #320 baseline 对齐)
- 4-layer codebook × K=256 = 1024 SID tokens
- Hyperbolic init 需要从 codebook (SID integer → 32-d embed → expmap0(c·) → 128-d) 重构

**注意**: task84 用 sentence-T5 编码 item → 32-d. 但 SID 是 int array (9922, 4). Gate 0 需要从 Stage 1/2 产物反推: Stage 1 RQ-VAE codebook 是 (256, 32-d), SID token 应该对应到这个 codebook 的向量, 不是从 Stage 1 训练后的 Sentence-T5 embedding 反推.

### 2.5 决策阈值

| 实测 R@10 | 决策 |
|-----------|------|
| Hyperbolic init > Random init | 🟢 GO - T5 能利用几何信息, 验证"传输损耗在 T5 侧" |
| Hyperbolic init ≈ Random init (Δ < 0.001) | 🟡 NEUTRAL - init 方式无关, 几何信息在训练中被丢弃 |
| Hyperbolic init < Random init | ❌ NO-GO - 双曲 init 反而扰动训练 |

## 3. Gate 1 设计 (中等成本)

**Gate 1**: 在 T5 embedding 层之后, attention 之前, 插入轻量双曲感知投影层 (复用 Issue #43 HypPreEncoder 思路).

**改动**:
```python
# 在 HG_Rec.forward() 中, T5 input embedding 之后
input_embeds = self.model.shared(input_ids)
if config.get('use_hyp_projection', False):
    # Apply expmap0(c·x) per token
    c = config.get('hyp_proj_c', 1.0)
    hyp_embeds = expmap0(c * input_embeds)
    input_embeds = hyp_embeds
# 然后送进 T5 encoder
```

**Gate 1 决策阈值**: 投影层 R@10 > 0.1020 (baseline), 验证 Stage 3 输入侧双曲投影有效.

## 4. Gate 2 设计 (高成本)

**Gate 2**: 替换 T5 attention 为混合曲率 attention (Curve Your Attention 论文做法).

**改动**: 修改 T5 attention 计算, attention score 基于双曲距离而非欧式点积. 需要改 `transformers/models/t5/modeling_t5.py` 或 wrapper.

**Gate 2 决策阈值**: Gate 0 + Gate 1 显示有效才启动.

## 5. R10/R11.5 决策 + ROI

| Gate | 改动成本 | Stage 3 训练 | 验证 ROI |
|------|---------|-------------|---------|
| Gate 0 (SID init) | ~10 lines patch | ~2h 单 GPU | **高** (低成本验证假说) |
| Gate 1 (HypPre projection) | ~30 lines | ~2h 单 GPU | 中 (跟 Issue #43 重复) |
| Gate 2 (mixed-curv attention) | ~200 lines + 修改 transformers | ~3h 4 GPU | 中-低 (高实施风险) |

**R10 推荐**: 启动 Gate 0 (Issue #57) + 候选 (b) Issue #43 × R-Drop 联合 (Task #147 重启) 并行. ~4h GPU. Gate 1 跟 Issue #43 HypPreEncoder 重叠, 等 Gate 0 落地后再决定.

## 6. 启动 Gate 0 命令模板 (等 owner 拍板或 R10 自主启动)

```bash
# Patch HG_Rec.py + run baseline (random init)
nohup python3 scripts/task84_hgrec_stage3_train.py \
    --sid_embedding_init random --task_id 151 \
    --gpu 1 > logs/task151_baseline_$(date +%Y%m%d_%H%M%S).log 2>&1 &
# Patch HG_Rec.py + run hyperbolic init
nohup python3 scripts/task84_hgrec_stage3_train.py \
    --sid_embedding_init hyperbolic --hyp_c 1.0 --task_id 151 \
    --gpu 2 > logs/task151_hyperbolic_$(date +%Y%m%d_%H%M%S).log 2>&1 &
# Stage 4 eval (both arms)
for arm in baseline hyperbolic; do
    python3 scripts/task84_hgrec_stage4_eval.py \
        --ckpt_path products/task151/$arm/.../HG_Rec_best.pth \
        --beam_size 50 --output verdicts/task151_$arm_beam50.json
done
```

**Reproducibility triangle (R12+C16 invariant)**: ckpt + SID + eval script SHA256 落盘.

result: Issue #57 方向C Gate 0 设计就绪. 2-arm 对比 (random vs hyperbolic init), ~4h GPU 2-GPU 并行. Gate 0 ROI 最高 (低成本验证"传输损耗在 T5 侧"假说). 决策阈值明确. 等 owner 拍板或 R10 自主启动.