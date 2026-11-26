# Issue #266 run log

- Issue: #266 — Analysis only: evaluate frozen curvature-loss/L1-gradient score on R2 bundles 108–109
- Issue state at freeze: OPEN
- Issue body sha256: `09c266af34eef4d0ffdb1a70d1dcdc6254b9bcb6acb3b15ba4f400ee4d0c5952`
- Frozen at: 2026-11-26
- Mode: **FROZEN**. The issue explicitly says "允许issue专属分析脚本及测试，公共Infra与研究机制不动" and "不进行Stage2/Stage3/TIGER/GNN训练". It does not authorize public-infra construction or repair.
- Training required: **No**. The issue is analysis-only; all requested quantities are checkpoint forward/autograd diagnostics and existing Stage3 test records. Issue boundary explicitly forbids Stage2/Stage3/TIGER/GNN training and any parameter update.

## Acceptance checklist (verbatim scope, C1–C9)

| ID | Requirement | Criterion | Data / output | Status |
|---|---|---|---|---|
| C1 | 只读核验 R2 108/109 完整四件套、checkpoint、SID、源码、token/mask、训练图资产 hash 与下游精确指标身份 | all listed files exist and SHA-256 match the frozen ledger; four pieces present; recall identity matches `test_final.json` | `_r2/108`, `_r2/109`, `INFRA.md`, `ENDPOINT_108.json`, `ENDPOINT_109.json` | **PASS** |
| C2 | 复用 #262 `quant_grad_switch.py` 的测量定义：固定 4096 商品、seed 42、`draw(0,0,1,track=False)`、同一前向路径；核对 batch IDs 与 sample identity；逐项标记定义差异 | reproduced batch ID SHA equals `251dced0…`; sample IDs/triples/references/target hashes identical; differences listed, not hidden | `measure_endpoint.py` output in endpoint JSON; `SCORE_CONTRACT.md` | **PASS** |
| C3 | 从最终 checkpoint 真实 forward + autograd 计算原始未加权 C 与 G1；不得用代理；保存梯度形状、None/非零检验、源码/checkpoint hash；参数 hash 前后一致且 optimizer steps=0 | real model forward; C `grad_loss_curvature`; G1 `grad_curvature_codebook0`; parameters unchanged; steps 0 | `ENDPOINT_108.json`, `ENDPOINT_109.json`, raw logs | **PASS** |
| C4 | 若 #265 与历史臂结构目标不同义，原生 loss 只作外部可迁移性检验并注明限制；辅助诊断单列 | explicit comparability section; no claim of same objective for 109; optional auxiliary diagnostic labelled separately | `SCORE_CONTRACT.md`, `REPORT.md`, `VERDICT.md` | **PASS** |
| C5 | 复算八臂原始分数并加 108/109，输出十臂 C、G1、归一化分量、S、Recall@10 表 | ten rows; eight values reproduced from frozen C/G1; 108/109 computed; scores use frozen formula without refit/clip | `SCORES.json`, `SCORES.csv`, `REPORT.md` | **PASS** |
| C6 | 计算 Pearson、Spearman、Kendall tau、严格相邻单调性、逆序对数量与违反臂；单列原 8 臂与扩展 10 臂；报告两新臂是否在冻结区间及相互顺序 | all statistics computed; ties/missing handled explicitly; violations named | `SCORES.json`, `REPORT.md`, `VERDICT.md` | **PASS** |
| C7 | 使用既有持久化接口保存命令/日志/退出码/hash 与独立手算或参考测试；校验超出归一化范围、排序并列、逆序与缺失 | `execution.run` command/log/exit; independent score/stat reference tests with hand-computed values | `logs/`, `tests/`, `reference.py`, `tests/test_reference.py` | **PASS** |
| C8 | 不训练、不更新参数、不重写 SID、不新增训练臂或种子；公共 Infra 与机制不动 | no optimizer; SID/checkpoint hashes unchanged; FROZEN infra snapshot compare | `INFRA.md`, `INFRA_SNAPSHOT_END.json`, `VERDICT.md` | **PASS** |
| C9 | 结论区分排序预测、原生 loss 跨机制尺度可比性、梯度敏感性与实际优化收益；完成后提交推送并关闭 | all four separated; monotonic failure accepted as valid result; commit/push/close evidence | `VERDICT.md`, close comment | **PASS** |

## Modification scope

New issue-local analysis only: `results/research/issue266/{measure_endpoint.py,reference.py,aggregate.py,tests,...}` plus JSON/CSV/Markdown evidence. No candidate model, public tool, asset, or frozen baseline is modified. No Stage2/Stage3 model is loaded with an optimizer; all autograd calls are diagnostic only.

## Notes

- R2 bundles were copied read-only into `_r2/108` and `_r2/109`; all 112/113 listed files match the ledger SHA-256.
- The ten-arm historical table is reconstructed from `results/research/issue262/MATRIX.json`; no historical result is refitted.
