# Issue #139 — the goal, recorded

The owner's goal, verbatim and repeated across sessions:

> 目标是完成5个机制的stage2，然后把这5个stage2得到的SID，记录失真度，并且完成对应的stage3。

That is three acceptance criteria, in this order, and nothing here is complete
until all three hold for **all five** mechanisms:

1. **Five Stage 2 runs complete** — F0, G, L, E, A, each 3000 epochs on four
   cards with the frozen recipe, each exporting a real `item_sids.json`.
2. **The structural distortion recorded for each of the five SIDs** — measured
   on one footing by `measure_structure.py`: the same training graph, the same
   support, the shared readout frozen by F0 at its 1500th epoch, the graph's own
   Huber scale. The record also splits the continuous (pre-quantization) from the
   discrete (post-quantization) distortion, so the quantization's own cost is a
   separate number.
3. **The corresponding Stage 3 for each** — one run per SID under the frozen
   protocol, giving a `test_recall@10` per arm. The learned collision gate is
   excepted for these five arms by name (`ISSUE139_STAGE3_EXCEPTION`), recorded
   in `CLAUDE.md` and in the admission evidence; the collision stays a recorded
   diagnostic.

## Where it stands now

```
{
  "F0": {
    "epochs_recorded": 3000,
    "finished": true,
    "outer_loop_rounds": 0,
    "sid": true,
    "layered_records": 2
  },
  "G": {
    "epochs_recorded": 3000,
    "finished": true,
    "outer_loop_rounds": 4,
    "sid": true,
    "layered_records": 2
  },
  "E": {
    "epochs_recorded": 1499,
    "finished": false,
    "outer_loop_rounds": 0,
    "sid": false,
    "layered_records": 0
  },
  "A": {
    "epochs_recorded": 0,
    "finished": false,
    "outer_loop_rounds": 0,
    "sid": false,
    "layered_records": 0
  },
  "L": {
    "epochs_recorded": 1499,
    "finished": false,
    "outer_loop_rounds": 0,
    "sid": false,
    "layered_records": 0
  }
}
```

## Where the artifacts land

| what | where |
|---|---|
| a Stage 2 run's SID | `results/stage2_RQ-VAE/TIGER_mixed_rq_<ARM>/item_sids.json` |
| its layered SID record | the same directory under `stage2_metrics/<sid_sha256>.json` |
| its structural distortion | `results/research/issue139/structural_fidelity.json` |
| its Stage 3 result | `results/stage3_T5Train/TIGER_mixed_rq_<ARM>/logs/*/*/test_final.json` |
| the pipeline's own record | `results/research/issue139/pipeline_report.json` |
| the durable status | `results/research/issue139/STATUS.md` |

## What is deliberately not part of the goal

No arm is tuned, no seed is repeated, no ablation is added, and no arm is
dropped because its diagnostics look poor. The five comparisons the issue names
(F0 vs G, G vs L, L vs E, L vs A) are read from the numbers these five runs
produce, not chosen afterwards.

## 7. 单卡阶段的执行计划与预算（2026-11-08 晚）

主机只剩一张 A100，四个 Stage2 臂（G/E/A/L）按新规则**并发**运行（各 1 process、全局 batch 4096、约 3.5 GB 显存），F0 已单独完成（collision 0.2572）。

**共享校准**（τ = 0.374305，b = −0.044043，来源 F0@1500）已固定，五臂共用；溯源摘要改为**记录、不拦截**（owner 2026-11-08 指令），因此不再有"改代码即重训 F0"的循环。

**自动结算**：`run_after_stage2.py` 等五臂各自的新一轮结束后，自动 ① 记录分层 SID 指标 → ② 五臂同口径测结构失真（同图、同 940k 支撑、同校准、同 Huber 尺度）→ ③ 重生成 `FINAL_REPORT.md`。数字全部由代码产生，不手填。

**Stage3 预算（单卡，硬上限）**：冻结协议是 150 epoch、`EVAL_START_EPOCH=75`、`EARLY_EVAL_EPOCHS=(25,50,75,100,125)`、beam 20。四卡时每臂约 2.5 h，单卡约 8–10 h ⇒ **五臂 ≈ 40–55 h**，且每臂必须独占整卡（Stage3 仍保留框架的全局单例守卫）。`run_stage3.py` 会在五臂 Stage2 齐备后依次启动并记录 `test_recall@10`。

**因此在当前硬件上，"五个 Stage3" 是两天量级的工作**；在此之前不应声称任何召回结论。
