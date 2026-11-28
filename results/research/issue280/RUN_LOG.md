# Issue #280 — RUN_LOG (frozen acceptance list)

| item | value |
|---|---|
| issue | #280 decode unseen directed behavioural transitions from discrete SIDs and quantify higher-order predictive fidelity |
| body sha256 | `489c08c24a8af33f6f5abc7778b40aee0a56c71476279962af7a817bf4aaa9be` |
| pinned inputs | #276 `SID_TABLE.json` `4c927914…`, `FOLDS.json` `1a8fb81d…`; #277 `PREDICTOR.json` `ec339f4a…`; #278 tables; #279 `RELATION_LEVEL.json`, `SID_FEATURES.json` |
| INFRA_MODE | **FROZEN** — issue text: "单Agent，analysis-only，INFRA_MODE=FROZEN；复用#276–279的253 SID、129标签、固定SID组折分、train/valid 有向图、现有评估器与资产；仅允许研究专属只读分析、轻量统计解码器（不是新推荐模型）" |
| training needed | **NO** — issue text: "不修改公共infra/原资产，不训练RQ-VAE/T5、不重训TIGER、不启动Stage2/3、不运行新最终test" |

## Frozen acceptance list

| C | requirement |
|---|---|
| C1 | Re-verify #279 (57 439 rows, 92.41 % unseen, direct edge and three higher-order AUCs), graph direction, legal negatives, item mapping, SID hashes; freeze the 129-SID queue, 120/9 identity split, 6 strict high-recall; evaluate all 253 SIDs where possible |
| C2 | Prediction unit = directed pair (a,b); decoder fit/calibration split inside training by source-anchored edge groups; report reverse-edge overlap and node overlap; the main validation set contains only unseen positive transitions; never build the graph from validation, never leak target features; negatives from the original legal candidates deterministically, excluding observed positives and the current target; fixed uniform and degree/popularity-matched negatives plus a reverse-edge diagnostic; identical candidate set for every SID |
| C3 | Every SID read from its real first three codes and correct item identity; no continuous z/q, item-id embedding, text or graph adjacency may enter the SID-only decoder; Recall stays hash-bound to its SID; duplicate exports are not independent samples |
| C4 | SID-only directional decoder: baselines = longest common prefix / hierarchical sharing, uniform chance, training popularity; main model = directional code-transition likelihood ratio with frozen smoothing/sparse fallback; trained from training counts only, never memorising item pairs; a low-capacity ordered-code interaction diagnostic with a fixed parameter cap; direction asymmetry via swapped scores; permutation-consistency check |
| C5 | Higher-order diagnostics on the same unseen candidates: correlation of the SID-only scores with the graph 2-hop/common-neighbour scores and stratification; optional offline decoding of higher-order relations from SID; the graph reference is an information ceiling, never mixed with SID-only |
| C6 | Relation-level metrics: unseen AUC/PR-AUC, same-anchor MRR/Hits@K, degree-matched AUC, direction accuracy, anchor macro + clustered CI; negative counts/distribution; MRR/Hits are fixed-candidate decoding metrics, not Stage3 Recall |
| C7 | Freeze 1–2 SID-level features (e.g. per-SID unseen degree-matched AUC, same-anchor MRR or directional gain) before any Recall association; keep all diagnostics without selecting on outer results |
| C8 | Prediction increment: #277 folds, leak-free preprocessing; compare Ridge6, Ridge6 + ≤2 decoder features, single-metric, and the #279 higher-order sharing features; inner-only alpha/feature selection; one OOF per SID; collinearity with L1 pos/neg/ER, group size, popularity and #279 features; cross-fitted residual analysis and counterexamples |
| C9 | Criterion: same-queue OOF MAE decrease vs Ridge6 with SID-group bootstrap 95 % CI lower bound > 0; Spearman/Kendall not worse; >0.06 FP/FN not worse; strict subset separately; 0.065 NOT_EVALUABLE; relation-level success does not imply Recall success; no causal claims; item pairs are not independent SID samples |
| C10 | Freeze a new version only if the criterion passes (formula, decoding contract, parameters, hashes, domain, prediction interval, old/new timestamped predictions for truly unknown candidates); known labels never count as blind tests; otherwise keep #277 |
| C11 | Deliverables: DIAGNOSIS/HYPOTHESES/CONTRACT, identity/isolation audit, fixed sample lists + manifest, per-SID decoder metrics and strata, code/tests (hand-count direction, smoothing, leakage, renaming, same-SID control), persisted logs, OOF/CI/ablations/error counterexamples, new predictor only if it passes, REPORT/VERDICT, infra zero drift, commit+push; final answers: can pure SID predict unseen directed edges, what it retains beyond prefix sharing, can higher-order structure be decoded from SID, does it improve Recall prediction |
