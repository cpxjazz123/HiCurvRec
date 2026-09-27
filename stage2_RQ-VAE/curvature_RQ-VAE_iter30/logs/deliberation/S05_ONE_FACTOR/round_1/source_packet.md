# S05_ONE_FACTOR — canonical source packet

STAGE_ID=S05_ONE_FACTOR
ROUND=1
ITERATION=30

## Objective
Independently audit and register the exact one-factor contrast for the prospectively matched iter30 replication. Identify parent/source lineage, historical control mapping, inherited mechanisms, source/code/settings differences and protocol routing. Verify that within each future seed pair, the **only conceptual/treatment difference** will be the fixed curvature mapping/vector. Do not edit/implement code or authorize training.

## Governing canonical artifacts
- S00: `stage2_RQ-VAE/curvature_RQ-VAE_iter30/logs/source_snapshot_iter30.md`.
- S01: `stage2_RQ-VAE/curvature_RQ-VAE_iter30/logs/protocol_manifest_iter30.md` and `logs/deliberation/S01_PROTOCOL_LOCK/round_1/judge.md`.
- S02: `stage2_RQ-VAE/curvature_RQ-VAE_iter30/logs/hypothesis_iter30.md` and Judge record.
- S03: `stage2_RQ-VAE/curvature_RQ-VAE_iter30/logs/mechanism_manifest_iter30.md` and Judge record.
- S04: `stage2_RQ-VAE/curvature_RQ-VAE_iter30/logs/mechanism_contract_iter30.json` and Judge record.
- Active `skill://curvature-rqvae-iter` §5; root `CLAUDE.md` paths/launch constraints.

## Required parent/baseline roles
```text
PARENT_ITER=iter29 (implementation/source-lineage parent)
PARENT_COMMIT=ecd01e4712a1badd38a0338255f4b2ec7b030aff (current source revision in S01)
CANONICAL_BASELINE_ITER=iter26 (fresh paired-control mapping and exact historical baseline)
EXPERIMENT_TYPE=single_factor; prospective matched-seed replication of two existing FCCR-1 maps
ACTIVE_MECHANISMS_BEFORE=FCCR-1 fixed closed-form curvature plus the unchanged inherited RQ-VAE behavior/quantization mechanisms
NEW_MECHANISM=No new conceptual mechanism; compare the two S14-approved existing fixed FCCR-1 mapping settings in fresh matched-seed pairs
ACTIVE_MECHANISMS_AFTER=FCCR-1 fixed closed-form curvature under either registered mapping; no stack or third map
```
S01 records iter29 Stage2 source/config SHA256 `2469b193bee1c4352ba180540c305c9ea837043afafd1b1585ea7b8aff805f4d` / `b41bf1fff2b481abf846d71a43c30c7ba34d0d25bd93bc04f70df486e3163f7a`, iter26 historical-control source/config SHA256 `ed629824d5a0308f4018b6dbaaa24303690a3597def7d1bcbf9abd38c3aca10f` / `4d6b3e06aa2f408a00664d971d64ed83d0e88ef5f88b852be97302a0e2786086`, and current Stage3 trainer SHA256 `9861eb6b2a5b0bfac546524c50d402aba5827e5079a228e2ae5501909bc937cb`. Verify source claims against primary code and the prior `one_factor_diff_iter29.md`, not only copied declarations.

## Exact two mapping settings
Common `[L0,L1,L2]` formula inputs are the S03-approved historical method/value inputs, with limitations attached:
```text
B = [19.324911558712664, 1.4605688962651735, 1.0148104414712726]
raw_residual_medians = [1.0, 0.10941, 0.09331]
```
S03 PASS is limited to historical method/value provenance; raw residual confidence is MEDIUM, with no replay/byte identity. Use explicit `raw_residual_medians` in iter30 and fail closed if missing; never use `residual_norm` or normalized `[0.001,0.932889,1.0]` as the raw formula input.

**Control — iter26 map**
```text
s=log1p(B)/log1p(m_raw/min(m_raw))
z=(s-mean(s))/(std_population(s)+1e-12)
c=clip(0.5*exp(0.2*z),0.05,1.5)
c26=[0.6145357379232853,0.5333020920777128,0.3814078098431606]
```
**Candidate — iter29 map**
```text
x=B/(B+2.0); y=m_raw/(m_raw+0.1); u=(x+y)/2; c=0.05+1.45*u
c29=[1.3660953164241916,0.7347829661951981,0.6439958072706683]
```
Only c mapping/vector differs within pair. `d_c=c29-c26=[0.7515595785009063,0.2014808741174853,0.2625879974275077]`. Both are fixed before Stage2, non-trainable/time-invariant, and pass unchanged through the same consumers.

## Inherited mechanisms and candidate confounds to audit
Directly compare iter26 and iter29 sources or their canonical one-factor audits to verify all of these are inherited identically in the prospective pair: model architecture/dimensions, 3 quantizer layers×256, input/hidden/embedding sizes, data order and sampling, optimizer/learning rate/weight decay, commitment/reconstruction/behavior losses and weights, temperature, Sinkhorn configured epsilon/iterations, M2 residual update, M3 transport, initialization/warm-start, number of Stage2 steps, gradient clipping, output/export behavior (apart from isolated output paths), Stage3 trainer/evaluation, and all non-map runtime settings. Existing curvature consumers—including Poincaré distance, curvature-scaled effective epsilon `sk_eps*(c/c_cyclic_max)`, residual geometry/transport, behavior contrastive distances, and any curvature-dependent reconstruction/quantization loss—may respond to `c`; their equations/parameters must remain the same in both arms. Such downstream response is mediated by the mapping and is not an added mechanism.

Check for historical policy/reporting-only source differences, especially the iter29 migration removing the non-SID-dependent HitRate@50 subprocess and making Stage2 quality metrics descriptive-only/no early stop under root §2. The future iter30 pair must share one common implementation/policy path; any reporting hygiene must apply identically and must not change model/training behavior. Do not confuse source/reporting differences between historic iterations with an allowed treatment factor.

One relevant historical data-wiring distinction: iter26's closed-form consumer used ambiguous JSON `residual_norm`; S03 explicitly requires iter30 control and candidate to share the same explicit `raw_residual_medians` input. This is the approved semantic/key hygiene needed for both arms and not a treatment difference.

## Locked prospective protocol and paths
- Seeds 43/44/45; matching Stage2 and Stage3 seeds per pair; six complete serial runs; no seed 42 pooling/replacement/omission.
- Stage2 100,000 steps; batch 640/GPU, 4 ranks; same immutable iter8 warm start, input identities/order, all settings from S01. Each run has its own hardcoded profile and distinct outputs; fail-closed warm start and a per-run gradient-path gate are later implementation/verification requirements.
- Stage3 trainer remains byte-identical to S01 SHA; matching seeds and Stage2 SID path, same unchanged protocol, final test for all six runs. Only wrapper labels, SID paths, and unique result roots vary as required for protocol routing.
- Use exactly the six unique short-root paths from S01. No CLI or environment parameter selection; no output collisions/overwrites. Stage2/Stage3 outputs are not inside the source iter tree.
- User target is inclusive `test_recall@10 >=0.065`, assessed for every valid result; complete the whole block regardless of scores.

## Required candidate artifact
Each candidate independently writes a complete proposed `one_factor_diff_iter30.md` with parent/commit/baseline/experiment type, active mechanisms before/after, full exact mapping-only difference, changed source/config/wrapper/output routing and policy hygiene, inherited mechanism inventory, optimizer/loss/Stage1/Stage3 differences, direct comparison of source differences and risks, and an explicit statement whether the prospective pair is truly single-factor. Distinguish historical code lineage from the common future implementation. If any source inconsistency shows an unavoidable second treatment change, recommend blocking before implementation; do not hand-wave it.

## Independence and boundary
A and B receive this identical packet, independently inspect primary iter26/iter29 sources and canonical records, and write only their role-specific candidate under `logs/deliberation/S05_ONE_FACTOR/round_1/`. Do not read the other draft or modify source/config/canonical files. Judge C adjudicates and materializes only `logs/one_factor_diff_iter30.md` and its Judge record. S05 does not authorize S06 implementation, preflight, MVG, Stage2, Stage3, or GPU work.