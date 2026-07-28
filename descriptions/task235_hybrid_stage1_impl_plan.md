# Task #235 — Issue #9 Gate 1 实现 plan: per-layer assignment_mode in HG-Rec

## 来源
- Issue #9 H2: hybrid 在 Stage 1 训练中是否继承 Gromov 坍缩
- Task #234 Gate 0 PASS (verdicts/task234_hybrid_gate0_result.md) — H1 成立
- Gate 1 需要修改 HG-Rec/model/hrqvae.py 加 per-layer assignment_mode 支持

## 当前 HRQVAE 代码状态

`HG-Rec/model/hrqvae.py:163-165`:
```python
if assignment_mode not in ('shared', 'per_codeword_kappa', 'gromov'):
    raise ValueError(f"assignment_mode must be shared/per_codeword_kappa/gromov, got {assignment_mode!r}")
self.assignment_mode = assignment_mode
```

`HG-Rec/model/hrqvae.py:320`: `assignment_mode=self.assignment_mode` 透传到 HResidualVectorQuantization.
全层使用同一 mode, 无 per-layer override.

## Gate 1 改动需求

### 改动 1: HRQVAE __init__ 接受 `assignment_mode_list` (per-layer)

```python
# 新增 CLI flag:
parser.add_argument("--assignment_mode_list", type=str, default=None,
                    help="Per-layer assignment modes, comma-separated e.g. 'per_codeword_kappa,gromov,gromov'")

# HRQVAE __init__:
if assignment_mode_list is not None:
    modes = assignment_mode_list.split(',')
    assert len(modes) == len(num_emb_list), f"assignment_mode_list length {len(modes)} != num_emb_list {len(num_emb_list)}"
    for m in modes:
        assert m in ('shared', 'per_codeword_kappa', 'gromov')
    self.assignment_mode_list = modes
else:
    self.assignment_mode_list = [assignment_mode] * len(num_emb_list)
```

### 改动 2: HResidualVectorQuantization 接受 per-layer mode

在 `HG-Rec/model/hrqvae.py:319-320`:
```python
# 旧: assignment_mode=self.assignment_mode
# 新: assignment_mode_per_layer=self.assignment_mode_list
```

VQ 层 per-layer 接受自己的 mode, 在 `find_assignment` / `compute_distance` 内 dispatch.

### 改动 3: launch 配置

```bash
python3 train_hrqvae.py \
    --assignment_mode_list per_codeword_kappa,gromov,gromov \
    --c_k_min 1.0 --c_k_max 5.0 --c_k_seed 42 \
    --gromov_weight_l 0.5 \  # 新增 CLI flag
    --num_emb_list 64 128 256 \
    # 其余跟 task222 一致
```

## Gate 1 pass bar (Issue #9)

| 指标 | 阈值 | 含义 |
|------|------|------|
| collision @ best_collision epoch | ≤ 0.3706 (task222 best known) | 训练不坍缩 |
| L0 utilization | ≥ 90% | §6.7.4 stop-loss (i) |
| unique SID | > 6245 (62.94%, task225) | codebook 健康 (Stage 2) |

## 风险点 (R11.4 critical decision)

| 风险 | 评估 |
|------|------|
| 上游 src/ 改动 | R6 允许, 但需要 PUSH 时显式声明 |
| 影响 task218-234 已有的 `--assignment_mode` CLI 兼容性 | 必须保持 (default 行为不变) |
| Gromov 权重 weight_l 透传 | 新增 CLI flag, 不破坏旧用法 |
| Gate 1 wall-clock | task222 已证 40 epoch early-stop ~30s. 重复跑无差异 |
| Hybrid 是否真在训练中保持 OPEN | Gate 0 已验证 Phase 0, Gate 1 是训练 dynamics 验证 (H2) |

## 实施步骤 (R11.4 dry-run → execute)

1. **Dry-run**: 在 verdict 写明改动 1-3 的具体 diff (已在本 description)
2. **Execute**: 修改 HG-Rec/model/hrqvae.py + train_hrqvae.py
3. **Verify**: py_compile + 跑 sanity check (1 epoch Stage 1)
4. **Launch**: scripts/task235_hybrid_stage1_train.sh (40 epoch early_stop)
5. **Evaluate**: collision + L0 util 跟 task222 best row 对照
6. **Pass → continue Stage 2+3+4**: Issue #9 Gate 2/3
7. **Fail → close Issue #9 with FULL NO-GO** (跟 task221 Gromov-collapse 证据一致)

## 状态
Plan 文档, 未执行. 等 Issue #9 close comment 完成, R10 下一 tick 启动 Gate 1.
