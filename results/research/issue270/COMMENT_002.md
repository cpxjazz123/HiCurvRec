## #270 verdict — 观测性关系不可识别（analysis-only）

**队列**：204 个含 `test_final.json` 的 Stage3 run；42 个可用几何臂（同一 57,439 行 target 快照、同一图资产）；12 个 immutable-run 记录经哈希核验，30 个 `arm_dir_legacy`（标记，不伪造哈希）。

### P1（曲率改善 → R 单调正相关）：不支持
- Spearman **−0.068**（family-cluster bootstrap CI **[−0.51, +0.35]**，置换 p 0.66，n=42）；Kendall −0.060。
- Pearson +0.41 由尺度离群臂（A.1 MAE 在数百）拉动，与所有秩统计量矛盾，不作为单调关系证据。
- Leave-one-family-out ∈ [−0.26, +0.01]；单卡层 n=3，退化，不构成协议证据。

### P2（以 SID 结构保持为条件）：不支持
- 压缩层 ρ **−0.90**（n=5）vs 保持层 ρ **+0.065**（n=37）；差值 CI **[−0.85, +1.44]**，含 0。

### P3（#269 的损失由 SID 容量/前缀压缩解释）：仅在 Σ 绝对标尺上成立
- #265→#269：A.1 **−0.249**（更好），L2 组 **170→113**（127 拆分、93 合并），ΔR **−0.00736**，hit10 gain/loss 1118/1691。
- item 级 ΔA.1 vs Δhit ρ **−0.009**，81% 零 Δhit 平局 → 无 item 级规律。
- 标尺反转：J（Σ）0.183→**0.326**，#269 自身拟合的 κ_t 标尺 0.168→**0.288** → “几何改善”本身依赖标尺；不支持因果归因。
- 观测性中介：curvature→SID ρ −0.28，SID→ΔR ρ −0.47，偏相关 −0.25；不是因果估计。

### P4（family/协议主导）：支持
- 22 个 family 的 between SD **0.00356** vs within SD **0.00094**（≈3.8×）。

### C2 处置
预注册规则只在 P1 为正时授权 #265/#269 重跑；P1 为 null，重跑不检验该命题，故 **C2 不触发**。因果设计（冻结 checkpoint、同一 codebook/容量，只改导出几何，配对 Stage3）写在 `VERDICT.md`，作为下一轮独立 issue 提案，不在此执行。

**产物**：`results/research/issue270/{REPORT,VERDICT,DIAGNOSIS}.md`、`analysis/RELATION.json`、`CONTRACT.md` §8、`assets/geometry/`（42×npz + 42×J + 4 κ_t）、`receipts/`。
**结论**：“折叠/几何改善 → 推荐收益”在现有独立 run 上不可识别；唯一稳定的是协议/机制族与 SID 压缩的伴随关系。关闭本 Issue，不新增 42 臂扫描。
