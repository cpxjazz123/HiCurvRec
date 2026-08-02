# 系统性根因分析: taskA/taskB 为何至今未超过 baseline (R@10=0.1020)

> 分析时间: 2026-08-02 (纯代码审计 + 决定性实验, 不看 issue 历史)
> 结论: **根因是评估协议错配, 不是模型能力不足。** taskA/B 全链路的 R@K 评估用了与 baseline 不同的解码协议 (greedy-argmax 单候选) 与不同 split (test vs valid), 数学上 R@10 被钉死在 ~0.03-0.05, 永远无法与 baseline 的 0.1020 (beam20) 对比。

---

## 1. 决定性实验 (N=1000, 同一 baseline T5 ckpt)

实验对象: `products/task84/ckpt_hgrec/.../HG_Rec_best.pth` (md5 `409a413b`), 与 `taskA/_ckpt/HG_Rec_best.pth`、`taskB/_ckpt/HG_Rec_best.pth` **同一文件** (md5 全一致)。即 taskA/B 冻结的就是 baseline T5 本身。

交叉 2 种 SID × 2 种 split × 2 种解码:

| SID | split | beam20 R@10 | greedy-layerwise R@10 |
|---|---|---|---|
| `hrqvae_poincare` (taskA/B 用) | **valid** | **0.1020** ✅ = baseline | 0.036 |
| `hrqvae_poincare` (taskA/B 用) | test | 0.087 | **0.032** |
| `rqvae_fixed_hgrec` (另一候选) | valid | 0.000 | 0.000 |
| `rqvae_fixed_hgrec` | test | 0.000 | 0.000 |

**关键事实**:
1. **`hrqvae_poincare` SID + valid + beam20 精确复现 baseline 0.1020** → 这就是 baseline 的评测配方。
2. taskA/B 的 SID (`hrqvae_poincare`) 是正确的 (与 baseline 一致, sha256 `2dab2922`), 冻结 T5 也是正确的 (同一文件)。
3. **`rqvae_fixed_hgrec` 全 0** → 该 SID 与 T5 训练完全不匹配, 排除。
4. **同 T5 同 SID 下, 评估协议导致 2.8-3.2 倍差距**: beam20 R@10=0.1020 → greedy-layerwise 只有 0.036 (valid), test 上只有 0.032。
5. **split 差距**: valid→test 在 beam20 下 0.1020→0.087 (~15%)。

## 2. 根因链: 三个错配叠加

### 错配 A — 解码协议 (主根因, ~2.8-3.2x)
- **baseline 0.1020**: `HG_Rec.generate` → `model.generate(num_beams=20, num_return_sequences=20)` → 每样本生成 **20 条候选**, `recall@k` = target 的 4-token SID 是否出现在 **top-k beam** 中 (`train_HG-Rec.py:91-99`). 数学上 R@10 允许"第 3 好的候选命中即算对".
- **taskA/B**: `taskB/common/stage4_decode.py::autoregressive_predict_constrained` 是 **greedy 自回归 argmax 单候选** (`pred_token = logits_constrained.argmax(dim=-1)`, 每步只取 top-1). `taskA_stage3_direction_a_train.py:395` Stage4 内嵌更是**并行 argmax 单步** (`logits.argmax(dim=-1)`, 连自回归都没有). 单候选时 R@10 ≡ R@1.
- **后果**: taskA/B 报告的 val_R@10≈0.03-0.05 全是单候选命中率; 同 T5 用 baseline 协议 (beam20) 在 valid 上就是 0.1020. **把 0.032 拿去跟 0.1020 比, 是标尺不同的伪比较.**

### 错配 B — 评估 split (baseline 用 valid, taskA/B 用 test, ~15%)
- baseline: `train_HG-Rec.py:178-184` 加载 **valid.parquet**, 评估协议 P 用 validation 集.
- taskA/B: `taskA_stage3_direction_a_train.py:55` `TEST_PARQUET = .../test.parquet`; `taskA_stage4_resume.py:42` 同样 test.parquet. `taskA/_data/Instruments/` 下**根本没有 valid.parquet** (只有 train/test).
- 同 T5 同 beam20 下: valid 0.1020 vs test 0.087 → split 选择再贡献 ~15% 差距.

### 错配 C — 训练端从无真实验证 (fake early-stop)
- `taskA_stage3_direction_a_train.py:281` / `taskB_stage3_direction_b_train.py:279`:
  ```python
  _es_val_r10 = min(0.10 + (epoch + 1) * 0.0005, 0.115)   # ← 纯模拟, 单调递增
  ```
- 这不是真实验证, 而是**手工编造的单调递增值**. 训练过程从未跑过真实 val 评估 (taskA/_data 无 valid.parquet), "best_adapter" 的选择标准是假的 → 训练全程盲飞.
- `taskA_stage4_resume.py:4` 声称 "best_val_R@10_sim=0.1150 (vs baseline 0.1020, +12.7%)" — 这个 0.115 是模拟值, 不是真实 R@10.

## 3. 为什么"无论怎么训都到不了 0.1020"

因为评估标尺本身不同:
- taskA/B 用 **greedy 单候选**: R@10 数学上 ≤ 单候选 top-1 命中率 ≈ 0.04-0.05 (同 T5 实测 0.032).
- baseline 用 **beam20**: R@10 允许 20 条 beam 中任意前 10 条命中, 同 T5 同 SID 在 valid 上是 0.1020.
- **即使 adapter 训练得再好, 只要评估还是 greedy 单候选, R@10 就永远被钉在 ~0.05 以下.** 这就是 issue26/27/28/29 全部卡在 val=0.0500 的直接原因 — 它们测的根本是同一个单候选指标的噪声.

## 4. 修正路径 (按性价比)

1. **立即 (诊断验证)**: 用 baseline 同协议重测 taskA/B 现有 adapter:
   - SID `hrqvae_poincare` + **valid.parquet** + **beam20** (复用 `HG_Rec.generate` 逻辑或 `model.model.generate(num_beams=20)`)
   - 即可知道 adapter 相对 baseline 的真实增益 (是正还是负), 而不是永远测 0.03-0.05.
2. **Stage4 统一到 beam20**: taskA/B 的 stage4 评估脚本改走 beam search 20 + autoregressive + layer-mask, 与 baseline 协议完全对齐.
3. **训练端引入真实 val**: 从 train 中切出 (或补 valid.parquet), 用 beam20 做周期性真实验证, 替换 fake early-stop.
4. **长期**: 若 adapter 在正确协议下仍 < 0.1020, 才轮到"容量/训练优化"问题 — 但当前证据表明 0.1020 从未被正确复现过, 先做 1-3.

## 5. 附: 关键文件:行号证据

| 项 | 位置 |
|---|---|
| baseline beam20 评估 | `HG-Rec/train_HG-Rec.py:80-104` (evaluate), `HG-Rec/model/HG_Rec.py:63-82` (generate num_beams=20) |
| baseline 用 valid.parquet | `HG-Rec/train_HG-Rec.py:178-184` |
| taskA/B greedy 解码 | `taskB/common/stage4_decode.py:98-107` (argmax), `taskA/stage3/taskA_stage3_direction_a_train.py:395` (并行 argmax) |
| taskA/B 用 test.parquet | `taskA/stage3/taskA_stage3_direction_a_train.py:55`, `taskB/stage3/taskB_stage3_direction_b_train.py:56`, `taskA/stage4/taskA_stage4_resume.py:42` |
| taskA/B fake early-stop | `taskA/stage3/taskA_stage3_direction_a_train.py:281`, `taskB/stage3/taskB_stage3_direction_b_train.py:279` |
| taskA/_data 无 valid | `taskA/_data/Instruments/` 只有 train.parquet / test.parquet / item_emb.parquet |
| 决定性实验产物 | `verdicts/_protocol_probe_deciding.json` + 脚本 `$CLAUDE_JOB_DIR/tmp/protocol_probe.py` |
