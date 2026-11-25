# Issue #258 RUN_LOG — R05 role-conditioned level-2 mixed-curvature refinement

## 1. Issue identity (frozen)

| field | value |
|---|---|
| number / state at lock | 258 / OPEN |
| title | R05: role-conditioned L2 mixed-curvature refinement using frozen existing iteration Infra only |
| body bytes / sha256 | 6429 / `a244f1b4314956f1d22622f3f06502b685d9da2e3cdc8801d128b7735931891e` |
| body captured at | 2026-11-25 (see `COMMENT_1.md`) |
| parent research | `results/research/issue256/rounds/r05/CONTRACT.md` sha256 `1ebc1eef161e3cf29806db5689230fbbcefae5f653f4d07e89989e83a9cfd752` |
| parent diagnosis | `results/research/issue256/rounds/r05/DIAGNOSIS.json` sha256 `4fb0499a146b8352bdb1d0affc5266444ff561f581c8464d5faf5f2e7f11ce51` |
| parent status | `results/research/issue256/rounds/r05/STATUS.json`: `REGISTERED_NOT_DISPATCHED`, "no R05 worker was started, no workspace was reserved and no GPU budget was consumed" |
| public iteration Infra | `tools/iteration/` (Issue #257) |

Corroboration that the preregistered R05 run never started:

```
$ ls results/stage2_RQ-VAE/_runs/ | grep r05        # no output
$ find . -maxdepth 4 -name '*role_conditioned_l2*'  # no output
$ ls stage2_RQ-VAE/                                 # _archive  baseline
```

## 2. INFRA_MODE = FROZEN (default; no BUILD authorization)

Issue §2 says verbatim: “本轮只使用既有 Infra,**严禁开发、修改、扩充任何新的 Infra**”, and
“禁止新建针对本轮的通用控制器、launcher、评估器、统计器、依赖检测器、报告生成器、运行状态机、
数据资产或重复实现公共功能”. There is no sentence authorizing infrastructure build or repair, so the
mode is `FROZEN` and the file boundary of the skill applies: only the candidate mechanism, its dedicated
tests and the experiment evidence may change.

## 3. Training decision — required

Issue §3 demands, verbatim: “真实现有准入通过后,执行一次四卡Stage2 3000epochs/global4096/seed42,
合法checkpoint重导出、三码碰撞<0.4,再执行一次冻结Stage3 150epochs/beam20/seed42/完整57439测试.”
The acceptance metrics (`validation R@10`, `test R@10`, distinct L2 prefixes, collision) are produced by new
weights and cannot be recomputed from existing artefacts, so this is a training round, not an analysis round.

## 4. Frozen acceptance list (C1..C13)

Each item is quoted from the issue and carries criterion / threshold / data / accounting.

| id | issue requirement (frozen) | criterion / threshold | data + sha256 | accounting |
|---|---|---|---|---|
| C1 | “复核 #256 R05 已冻结 CONTRACT.md / DIAGNOSIS.json / STATUS.json,确认未启动任何正式 worker、未消耗训练预算;…不重复注册同一正式 run” | no R05 workspace, no arm tree, STATUS says not dispatched; exactly one registration for `issue256_r05_formal` | r05 contract/diagnosis/status hashes above; `_runs/` listing | `INFRA.md`, `COMMENT_1.md` |
| C2 | “R05 单一主要机制是将角色混合应用层从L3迁至L2:`m_r=softmax(theta_r)`,`D_2(i,c)=sum_k m_{r(i),k} d_k^2(residual_i^2,c)`。L1/L3继续使用商品级分层GNN Cross-Attention门控” | code path: role mixture at level index 1 only; levels 0 and 2 use the gate; three switches exist and are wired | candidate source + mechanism tests | `CONTRACT.md` §2, `MODULE_CHECKS.json`, candidate tests |
| C3 | “角色表仅使用既有冻结训练可见图资产,不新建图/Orbit/采样缓存。明确多角色压缩方式为原合同既定定义” | same `role_table.json` bytes as R04 (`32f986da…`) with `split_rule` training-only; no new graph/orbit/cache file | role table + `results/research/graphlet_roles/` asset | `INFRA.md`, `tests/test_role_mixture.py` |
| C4 | “保持公共源码、配置、Skill、CLAUDE.md、数据生成器、准入、缓存、评估器、归档器不变;必要的正常运行输出…不属于新Infra”; “只修改候选机制与机制专属测试” | public component `path:sha256` identical before/after; only candidate mechanism + mechanism tests + evidence changed | `INFRA_BASELINE.json` and end-of-round `INFRA_FINAL.json` | `INFRA.md`, `VERDICT.md` |
| C5 | “仅重验实际受影响项,必须验证R05 L2 cost、角色条件化开关、硬SID、真实曲率梯度到L2 role table与GNN/其他层门控、实际分配、真实四卡/entry/behavior graph、性能。已有同源有效证据复用;不得复用不匹配或缺失依赖的PASS” | each of the listed items has current evidence or a `reused:<fingerprint>` receipt | per-check outputs | `CHECK_1.json`, `CHECK_1.md` |
| C6 | “保持0.8s/epoch已授权上限,0.4–0.5s目标如实记录” | inclusive 31–60 epoch mean ≤ 0.8 s; report target/expected status truthfully | `benchmark_entry.py` four-rank run | `PERFORMANCE.json` |
| C7 | “不同二码前缀≤6000、L1 carry 0.0496–0.0696、validation R@10≥0.0583;失败线二码前缀≥7000或validation R@10≤0.0576。三者均记录” + behaviour-prefix selectivity, collision, reconstruction, quantisation error | values recorded, targets unchanged; failure line applied literally | exported validation SID | `SIGNATURE.json`, `VALIDATION.json` |
| C8 | “执行一次四卡Stage2 3000epochs/global4096/seed42,合法checkpoint重导出、三码碰撞<0.4,再执行一次冻结Stage3 150epochs/beam20/seed42/完整57439测试。禁止TIGER重训、多种子、早停、追分重跑” | exactly one formal run; collision < 0.4 before Stage3; n_eval = 57439 | run workspace + SID + Stage3 `test_final.json` | `FORMAL_RUN.json`, `FINAL_STAGE2.json`, `FINAL_STAGE3.json` |
| C9 | “调用现有分析工具…现有工具不能输出某项时如实标不可得,不新写公共分析组件、不伪造” | `tools/iteration/{analysis,replay}.py` outputs; missing items marked `NOT_AVAILABLE` | run identity + artefacts | `REPORT.md`, `VERDICT.md` |
| C10 | “统计本轮从登记到完成的全部墙钟…使用既有TIMELINE工具…对照#256约4h53m…目标≤2h,报告是否真实达标” | phase spans summed by `tools/iteration/timeline.py`; target comparison reported | timeline JSONL | `TIMELINE.json` |
| C11 | “自动报告、最终VERDICT、R2精确四件套与已有台账,由现有接口生成/核验。保存合法负结果…R05与R04差异仅解释为角色权重应用层迁移的观察,不能将与TIGER比较声称为单因素因果” | report/verdict/archive generated by existing interfaces; negative result preserved as negative | report dir, R2 ledger | `REPORT.md`, `VERDICT.md`, `ARCHIVE.json` |
| C12 | “最终列出公共Infra前后哈希/变更清单证明未开发新Infra,提交机制代码及证据并核对远端” | before/after public hashes equal; pushed; local main == remote main | `INFRA_BASELINE.json`, `INFRA_FINAL.json`, git SHA | `VERDICT.md`, close comment |
| C13 | “本轮完整真实Stage2/Stage3及独立归约和归档完成,预注册结果如实报告,且整个过程未新增/修改Infra。若合法入口或已有工具存在实际不可恢复阻塞,保持OPEN并报告” | all of the above true, or the issue stays OPEN with the actual blocker | all evidence files | `VERDICT.md` |

## 5. Method notes fixed before the run

- The R05 contract in `results/research/issue256/rounds/r05/CONTRACT.md` is the research contract; this round
  does not rewrite it. `results/research/issue258/CONTRACT.md` adds the implementation-level tensor/level
  contract and the paper comparison required by the skill.
- Only one formal run id: `issue256_r05_formal`; arm: `TIGER_role_conditioned_l2_rq`.
- No ablation, tuning, seed sweep, early stop, TIGER retraining, or second formal run.
- The final test is read once, after the validation decision is frozen; the selection-bias disclosure is in
  `VERDICT.md`.

## 6. Execution outcome

### Stage2 (formal, single run)

| item | value |
|---|---|
| run id / arm | `issue256_r05_formal` / `TIGER_role_conditioned_l2_rq` |
| epochs | 3000 / 3000, four ranks, global batch 4096, seed 42, no early stop |
| training wall | 2005.5 s; exit 0 |
| checkpoint sha256 | `f93da1a7466d62f07c6482c64cf1a8efdd9dd1fa7ccdf7891be98d6d37bc7f26` |
| SID sha256 | `dc206dba7fbdadc28bbdedb582c7c755e412d70bdfec930b380a2bd5ea9d5ba6` |
| legal re-export | 0 differing items; SID reproduced from the checkpoint |
| collision | 0.055273 (< 0.40); quantisation error 0.007134; effective rank 4.333 |
| distinct L2 prefixes | **9609** (prediction <= 6000; failure line >= 7000) |
| absolute L1 prefix carry | **0.059959** (inside 0.0496..0.0696) |
| H(L3 | L1 L2) / distinct triples | 1.851570 / 23228 of 24587 |
| verdict | preregistered mechanism signature **FALSIFIED** at Stage2 |

Validation Recall@10 is **NOT_AVAILABLE** (FAULT-1 in `INFRA_FAULTS.md`).

### Stage3

Pending the single frozen 150-epoch run; result is exploratory and does not change the falsified Stage2
signature.

