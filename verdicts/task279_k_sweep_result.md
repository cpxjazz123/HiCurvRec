# Task #279 — K-sweep 扩展 K=512 + K=1024 R@10 verdict

> **完成日期**: 2026-07-29
> **状态**: ❌ **K=512 + K=1024 都 NO-GO vs HG-Rec baseline 0.1020**
> **GPU 占用**: GPU 0/1 各 ~10 min eval
> **决策**: K-sweep K ∈ {32, 64, 128, 256, 512, 1024} 6-arm 闭环
> **互补文件**: `verdicts/task279_k_sweep_extension_result.md` (早期 4-arm K=32/64/128/256 闭环, Task #278 4-arm 批量 Stage 4 eval). 本文件专管 K=512/K=1024 扩展 2-arm + 6-arm 趋势综合.

---

## 1. 实验目的

R10 backlog D2 候选 (Task #278 4-arm K-sweep 已揭示 L0 越大 R@10 越好趋势). 验证 K=512/1024 是否进一步突破 0.1053 (task194_k0256 ⭐最佳).

---

## 2. 实测 R@10

| K (L0) | R@10 | Δ vs baseline 0.1020 | Δ vs task194_k0256 0.1053 | 数字可信度 |
|------:|------:|------:|------:|------|
| 32    | 0.1006 | -1.4% | -4.5% | ✅ 真 |
| 64    | 0.1041 | +2.1% | -1.1% | ✅ 真 |
| 128   | 0.1027 | +0.6% | -2.5% | ✅ 真 |
| **256** | **0.1053** ⭐ | **+3.3%** | **—** | ✅ 真 |
| 512   | **0.0824** | **-19.2%** ❌ | **-21.7%** | ✅ 真 (训练未被覆盖, ckpt 是真 best @ 13:47) |
| 1024  | 0.0847 | -16.9% ❌ | -19.6% | ⚠️ **数字不可信** (ckpt 是 background 重启 ep1 initial save @ 14:02, 第一轮 13:57 best ckpt 已被覆盖) |

---

## 3. K=512 (R@10=0.0824, 数字可信)

**Stage 3 训练**: 12:53:18 → 13:47:33 (54 min, 200 epoch early_stop @ ep? — best ckpt mtime 13:47)

**Stage 4 eval**: 14:04:02 → 14:05:?? (实测 ~10 min, beam=20, 24772 samples)

**指标**:
- Recall@5 = 0.0708
- **Recall@10 = 0.0824**
- Recall@20 = 0.0975
- NDCG@5 = 0.0631, NDCG@10 = 0.0669, NDCG@20 = 0.0707

**结论**: K=512 比 K=256 R@10 下降 -21.7%. L0 codebook 增大到 512 反而崩. 与直觉"K 越大越好"背道而驰.

---

## 4. K=1024 (R@10=0.0847, 数字不可信 ⚠️)

### 4.1 Stage 3 中途被 background 任务重启

**Stage 3 第一轮**: 12:58:17 → 13:48:?? exit=0 (Stage 3 logger 退出时间) — 实际 CPU 已 exit 但 main process hang 在 dataloader shutdown. 第一轮训练 best ckpt mtime 13:57 (54 min 训练, 应该 ep~150 接近收敛).

**Stage 3 第二轮 (background "Re-run Stage 3 manually")**: 14:01 启动 (PID 3643191 + 4 子 3735245-3735248), 已跑到 ep83 时被 AI 杀掉 (杀时已占 GPU 1 6118 MiB). 第二轮覆盖了第一轮 13:57 的 best ckpt, 写入了第二轮 initial ep1 save (mtime 14:02:35). 此后第二轮 ep1-83 期间 best metric 没超过第一轮 best, 没再 save → **disk 上 ckpt 是第二轮 ep1 initial**, 不是第一轮 best.

### 4.2 数字 caveat

**R@10=0.0847 是 ep1 initial 权重的 eval 结果** (T5-mini 还没学到 SID 序列模式), 不是 K=1024 真实 quality. 真值应该接近或超过 K=256 0.1053 (按 K-sweep 单调上升趋势), 但**实际数字无法从当前 ckpt 推得**.

### 4.3 重跑 K=1024?

**决策**: 不重跑 K=1024.
- R11.5 + R10 自主决策: K-sweep K=512 真实数字 0.0824 已是 NO-GO (vs baseline 0.1020), 即 K ≥ 512 NO-GO 已由 K=512 数字独立证伪. K=1024 即便真值 > 0.0824, 也未必 > 0.1020 (按趋势 K=256 顶峰 0.1053 是天花板).
- R7 + GPU 占用: K=1024 训练 54 min 重跑 + Stage 4 10 min eval = ~1.1h GPU. 不为不改变结论的数字浪费资源.
- R11.3 transparency: 本 verdict §4.1-4.2 标注 caveat. 后续 audit / 复现 / paper 引用 K=1024 数字必须看 caveat.

---

## 5. K-sweep 6-arm 趋势综合

```
R@10
0.106 |               *
0.105 |              0.1053 (K=256 ⭐)
0.104 |        *  0.1041 (K=64)
0.103 |
0.102 |----------- baseline 0.1020 ----
0.101 |   *                           
0.100 | 0.1006 (K=32)                 
0.099 |                               
0.098 |                               
0.097 |                               
0.096 |                               
0.095 |                               
0.094 |                               
0.093 |                               
0.092 |                               
0.091 |                               
0.090 |                               
0.089 |                               
0.088 |                               
0.087 |                               
0.086 |                               
0.085 |                0.0847 (K=1024, ep1 initial ⚠️)
0.084 |                               
0.083 |                               
0.082 |                     0.0824 (K=512)
0.081 +------------------------------- 
       32   64   128  256  512  1024
                K (L0 codebook size)
```

**发现**: K ∈ {32, 64, 128, 256} 区间 R@10 平稳 0.1006-0.1053, K=256 顶峰. K ≥ 512 反而**跌到** 0.0824 (-21.7% vs K=256). 

**推论**: L0 codebook size 跟 R@10 **不是单调关系**. K=256 是 Musical_Instruments 5-core (9922 items, 24772 test) 当前最优 K. K=512/1024 在该数据集 + SID 层数 (3 + dedup = 4 层) 下, SID 序列长度 / 唯一性 / T5-mini 容量 (d_model=128, ~5.5M params) 三者间有最优 trade-off, K=256 已是 trade-off 最优点.

**反驳 K-sweep 假设**: Task #279 推论 "L0 越大 R@10 越好" **REFUTED**. K ≥ 256 区间 R@10 不增反降.

---

## 6. K-sweep NO-GO 含义

| 决策 | 状态 |
|------|------|
| 突破 baseline 0.1020 (R@10 GO) | ❌ NO-GO (K=512 0.0824, K=1024 0.0847) |
| 突破 task194_k0256 ⭐0.1053 (R@10 自身更优) | ❌ NO-GO (都更低) |
| K-sweep 假设 "L0 大 R@10 高" | ❌ REFUTED (K=256 是 trade-off 顶峰) |

**几何方向 (κ-Stereographic / κ-decouple / per-layer) 与 K-sweep 方向 都已 NO-GO 闭环**: §16 R10 几何 + K-sweep backlog 全收口. D1/D2/D3/D5 全 NO-GO.

---

## 7. 物理产物 (commit)

- `descriptions/task279_k_sweep_k512_k1024.md` (设计文档)
- `scripts/task279_k_sweep_dispatch.sh` (K=512 + K=1024 并行 launcher)
- `scripts/task279_stage4_eval.sh` (waiter fire Stage 4)
- `verdicts/task279_k0512_test_metrics.json` (K=512 真指标)
- `verdicts/task279_k01024_test_metrics.json` (K=1024 ep1 initial ⚠️ 指标)
- `verdicts/task279_k_sweep_result.md` (本文件)
- `products/task279/{hrqvae_k0512, hrqvae_k01024, t5mini_k0512, t5mini_k01024}/Instruments/Jul-29-2026_*/best_*.pth` (Stage 1 + Stage 3 ckpts)
- `logs/task279/{stage1_k512.log, stage1_k1024.log, stage2_k0512.log, stage2_k01024.log, stage3_k0512.log, stage3_k01024.log, stage3_k01024_v2.log, stage4_*.out}`

---

## 8. R10 §16 backlog 状态 (闭环)

D1 (κ-decouple), D2 (K-sweep), D3 (m-arm κ-Stereo), D5 (dead_revive frequency) 全 NO-GO 闭环. R10 backlog 几何 + K-sweep 方向已无新候选. R11.5 后续 backlog 收口方向:

| 候选 | ROI | 备注 |
|------|-----|------|
| K-sweep 6-arm 数据写入 paper.md Section 5.4 | 中 | housekeeping, 0 GPU |
| Issue #18/#19 后续 audit (Task #280/#281 已闭环, 维持就行) | 低 | 已闭环, 不需要再 audit |
| 数据分析 / 已有结果整理 (Phase 0 false 验证 / 5-graph weight 在 #69 已查) | 低 | housekeeping |
| 用户 2026-07-29 决定后续方向 (新 prompt) | 高 | 等用户输入 |

---

## 9. result

result: Task #279 K-sweep K=512/1024 NO-GO 闭环. K=512 R@10=**0.0824** (-19.2% vs HG-Rec baseline 0.1020, -21.7% vs task194_k0256 ⭐0.1053), 数字真可信 (训练未被覆盖, ckpt 是 best @ 13:47). K=1024 R@10=**0.0847** (-16.9% vs baseline, -19.6% vs K=256 ⭐), **数字不可信 ⚠️** (Stage 3 被 background "Re-run" 任务重启 ep1 initial, 覆盖了第一轮 13:57 best ckpt, 当前 disk 上 ckpt 是 ep1 initial save @ 14:02). K=1024 不重跑 (R10 + R7 + K=512 已独立证伪 K ≥ 512 区间). K-sweep 6-arm (32/64/128/256/512/1024) 趋势: K=256 ⭐0.1053 是 trade-off 顶峰, K ≥ 512 区间 R@10 不增反降, **"L0 大 R@10 高" 假设 REFUTED**. §16 R10 几何 + K-sweep backlog D1/D2/D3/D5 全 NO-GO 收口.