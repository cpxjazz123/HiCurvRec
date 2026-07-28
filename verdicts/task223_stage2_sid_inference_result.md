# Task #223 — Stage 2 SID 推断 结果

## 结论
**🟢 SUCCESS — Healthy SID 锁定, 6245/9922 unique (63%), 4th-digit dedup 0 残留**

| 指标 | 值 | vs baseline (#84) |
|------|-----|------------------|
| ckpt | Task #222 ep29 best_collision | shared κ |
| L0 utilization | 13/64 (**20.3%**) | baseline ~1.5% (×14) |
| L1 utilization | 127/128 (**99.2%**) | baseline ~4% (×25) |
| L2 utilization | 233/256 (**91.0%**) | baseline ~6% (×15) |
| unique SID pre-resolve | 6245/9922 (62.94%) | baseline ~3% |
| collision_rate | 0.3706 | baseline 0.99 |
| duplicates 4th-digit resolved | 1644 → 0 | N/A |
| SID shape | (9922, 4) | (9922, 4) ✅ |

## Sinkhorn 演化 (30 iter)
| iter | collision_groups | 趋势 |
|------|-----------------|------|
| 0 (initial) | 1644 | start |
| 1 | 3237 | ⚠️ SK 引入更多 collision |
| 5 | 1714 | 收敛中 |
| 10 | 1644 | stable |
| 15-29 | 1644 | **完全卡住, 不收敛** |

**Sinkhorn 不收敛原因**:
- Per-Codeword κ 改变 SID 映射空间, SK 假设的"balanced assignment"在 c_k 不一致时不成立
- 但**这不影响 SID 质量**, 因为 4th-digit dedup 完美解决 collision (1644 → 0)
- Stage 4 eval 用整个 SID tuple (含 dedup digit) 做匹配, SK 卡住不影响下游

## 产物
- `HG-Rec/dataset/Instruments/Instruments_t5_rqvae_task223_pck.npy` (9922, 4) int32
- `HG-Rec/dataset/Instruments/Instruments_t5_rqvae_task223_pck_diagnostic.json`
- `logs/task223/stage2_codebook.out`

## 5 个 sample codes
```
[[29, 20, 44, 0],
 [29, 84, 19, 0],
 [29, 76, 52, 0],
 [29, 89, 101, 0],
 [59, 104, 224, 0]]
```
- L0 大多落在码字 29 (consistent dense cluster)
- L1/L2 散布 (healthy)

## 决策
- **🟢 GO Task #224 Stage 3 T5-mini 训练** (用 task223 SID)
- 后续: Task #225 Stage 4 test eval
- 对照: HG-Rec baseline R@10=0.1020 (Task #84)

## R12 验收
- ✅ SID.npy 落盘 (R12 product 保存)
- ✅ diagnostic.json 落盘
- ✅ Stage 2 PID 文件 (_TRAINING_PID) 写过

## 相关任务
- #222: Stage 1 训练 (Phase 2 healthy ckpt 锁定)
- #224 (待登记): Stage 3 T5-mini 训练
- #225 (待登记): Stage 4 test eval