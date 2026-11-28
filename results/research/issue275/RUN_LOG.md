# Issue #275 — RUN_LOG

## Contract freeze (2026-11-28 12:35)

Issue: #275 "Analysis-only: PySR symbolic regression for prospective Stage2-to-Stage3 Recall prediction"
Body sha256: computed at freeze time by `sha256sum` over the fetched body file (results/research/issue275/issue275.json).

### INFRA_MODE
`FROZEN`. Issue text (verbatim): "分析-only；INFRA_MODE=FROZEN，复用已有训练图、R2四件套、
SID/几何/重建评估器、历史Stage3结果。不改变公共infra、训练流程、量化机制或已存资产。"
The same paragraph allows: "允许独立分析环境安装固定版本PySR/Julia及研究专属脚本，不改变训练环境".
Therefore: a dedicated analysis venv under `results/research/issue275/` is permitted; `tools/`,
`assets/`, training entries, evaluators and Stage3 remain untouched.

### Training decision
**No training.** Issue text: "不训练任何RQ-VAE/T5、不重新训练TIGER、不启动Stage3、不新增神经模型基线。"
This is an analysis-only issue; §4 of the skill is skipped by the issue's own wording.

### Acceptance list (verbatim requirement -> id)
C1  RUN_INVENTORY: enumerate all existing Stage3 results and R2 (not only the 12 high-recall arms or
    #270's 42); each row binds actual SID hash, checkpoint, Stage3 evaluation target identity,
    protocol, mechanism family, R2 id, Rec@10/NDCG, timestamps; duplicate SIDs grouped; same SID with
    several Stage3 records all kept with a pre-frozen aggregation; single/quad GPU and protocol
    differences recorded; legacy identity gaps marked; no fabricated values.
C2  FEATURE_CONTRACT + joint metric table + manifest: all features recomputed with existing
    evaluators (L1 pos/neg shared rate and Hard ER, L1 occupancy/entropy/effective use/max share,
    L2/L3 prefix and conditional refinement, L1+L2 ER, behavioural neighbourhood, SID collision
    (first three codes), PR(z) and PR(q) separately, quantisation error, semantic reconstruction
    MSE/cosine, curvature MAE/RMSE/Spearman/sign and within-centre differences, global and 1/2/multi-hop
    J, Graphlet/Orbit fidelity); unified definitions/samples/scales/denominators; forward pass with bias
    and correct codebook; no mixing of reconstruction MSE with quantisation error, PR(z) with PR(q), or
    different collision definitions; re-check the 12 high-recall arms and the #274 candidate identities.
C3  Missing/identity audit: core-SID queue, vector queue, geometry-extended queue; report samples,
    families, missing and overlap per queue; compare only within the same matched queue; never fill
    missing features with Recall; do not exclude arms whose Recall+SID/checkpoint bind strictly.
C4  Pre-registered analysis contract: labels, features, family grouping, protocol handling, time split,
    formula operators, complexity, selection, metrics and budget frozen before search; first-round core
    features <= 6-8; separate SID-only, vector+SID, geometry-increment analyses; feature screening/
    scaling/missing handling only on the training fold; outer LOFO by family and time split, inner
    grouped tuning; outer labels never used for selection; controls: training-fold mean/median,
    single-indicator ranking, linear/ridge regression, prior combined formulas on the same folds;
    primary metrics MAE/RMSE, Spearman/Kendall, rank reversals, relative-to-TIGER direction accuracy;
    secondary 0.06 threshold; family-cluster bootstrap CIs and sample sizes.
C5  PySR: fixed version, operators +,-,*, safe division, square (optional safe log1p/exp with domain
    guards), maxsize ~8-15, maxdepth <=4, bounded searches, recorded random state; Pareto front;
    hall_of_fame, SymPy expressions, constants, domains, search log, per-fold predictions and selection
    process; independent recomputation of expressions must match PySR predictions.
C6  Hypothesis tests within the search: PR as a window not a monotone trend; quantisation error and
    effective rank complementarity; L1 positive/negative rates vs ER; whether curvature/distance adds
    out-of-SID outer-fold increment; report partial derivatives/monotone intervals, extremes, counter-
    examples, family/protocol sensitivity.
C7  Frozen success criterion: outer LOFO MAE stable improvement over the strongest simple control
    (paired differences + family bootstrap CI), improved ranking metrics, no clear failure under time
    extrapolation, simple expression without numeric singularities; CI covering zero or clear
    counterexamples => insufficient evidence; do not expand the search space to force monotonicity.
C8  Prospective prediction (only if a usable formula exists): freeze formula, parameters, feature
    definitions, data hashes, prediction intervals and candidate ranking BEFORE reading any new Stage3
    labels; predict completed Stage2 candidates without Stage3 only when identity and features are
    complete; timestamped immutable prediction table and pre-registered error criteria; candidates with
    published results are marked historical; predictions never close #274 nor replace real Recall.
C9  Deliverables: RUN_INVENTORY, FEATURE_CONTRACT, full joint metric table + manifest, missing/identity
    audit, PySR environment note, independent analysis code and necessary tests, persistent commands/
    exit codes/logs, per-fold formulas/predictions, Pareto table, simple statistical controls, LOFO/time
    validation, ablations and CIs, counterexamples, frozen prospective predictions (if legal unknowns),
    REPORT/VERDICT, infra zero-drift evidence, commit and push.
C10 Scope guards: no RQ-VAE/T5 training, no TIGER retraining, no Stage3, no new neural baselines, no
    changes to public infra/training/quantisation/assets; dependency versions, source commit, random
    state, CPU budget and persistent logs recorded.

### Mechanism modification scope
None (analysis-only): no candidate tree, no mechanism code, no training. Only new files under
`results/research/issue275/`.

### Verification plan
All evidence is recomputed from stored artifacts with existing evaluators; no training evidence is
generated. Infra integrity: snapshot the public component hashes at the start and before closing
(`tools/iteration/integrity.py`), and assert zero drift except files under `results/research/issue275/`.
