# Task #254 — Issue #13 Gate 3: Hamming vs pool/distance argmin (零 GPU 前向)

## 背景

Issue #13 Gate 2 (Task #253) NO-GO: Möbius 残差 argmin 几何对齐 (Task #249 一致率 89.3%) ≠ 下游 R@10 提升 (R@10=0.000403).

Issue #13 Gate 3 换思路: **不改残差算子, 改 argmin 距离公式** 本身. 用 logmap_c(x; y) = artanh(√c · ‖y⊕x‖) / √c · ((y⊕x) / ‖y⊕x‖) 在 quantizer 的 forward argmin 时替代欧式 ‖·‖². product_manifold 兼容 (hyp part 用 logmap, euc part 保留欧式).

## Gate 3 设计

**Phase 1 (零 GPU 前向)**: 与 Gate 1 (Task #249) 同模式, 1 个 quantizer 1 个 epoch 前向, 28 个 batch × 64 item = 1792 query, 验 argmin 一致率 (Euclidean vs Logmap). 假设 Logmap 与 Euclidean 一致率 70-80% (跟 Gate 1 残差结论同量级).

**Phase 2 (如 Phase 1 失败则 STOP)**: 不写.

**Phase 3 (待 Phase 2 启动)**: 需 Phase 2 决策.

## Gate 3 决策准则

- Phase 1 一致率 <70%: Issue #13 整体 NO-GO FULL, 关闭 issue
- Phase 1 一致率 70-90%: Issue #13 Gate 3 PASS, 推进 Phase 2 改 argmin → Stage 3
- Phase 1 一致率 >90%: Issue #13 Gate 3 STRONG PASS, 立即跑 Phase 2 50 epoch

## 关键决策点 (R11.3)

- **改哪个位置**: quantizer.forward 的 argmin_distance (utils.py:?, 跟 Gate 2 同一文件 utils.py)
- **不动残差**: Gate 2 已经 NO-GO, 不再重复
- **不动 Stage 3**: 仅 Phase 1 验证, Phase 2 决策后再上 Stage 3
- **不抢 GPU**: Phase 1 零 GPU
- **不动 utils.py 残差**: 恢复 Gate 2 patch, 留 backup

## 物理产物

```
descriptions/task254_issue13_gate3_logmap_argmin.md (本文件)
verdicts/task254_issue13_gate3_logmap_argmin_result.md
scripts/task254_issue13_gate3_logmap_argmin.py (1800 query 前向)
```

## References

- Task #249 Issue #13 Gate 1 PASS — 复现同样前向方法
- Task #253 Issue #13 Gate 2 NO-GO — 不复用 Möbius 残差 patch
- HG-Rec/model/utils.py:1795 — 上一版 Möbius 残差 patch (待回滚)
- Berman-Metzler 2020 / HRQ arXiv:2505.12404 — logmap 公式
