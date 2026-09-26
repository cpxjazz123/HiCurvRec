# S02 Hypothesis Source Packet — iter29

STAGE_ID=S02_HYPOTHESIS
ROUND=1

## Canonical context

- Controlling contract: FCCR-1, fixed closed-form curvature. Follow `logs/source_snapshot_iter29.md`, `logs/protocol_manifest_iter29.md`, and the S01 Judge decision. Only judge-approved canonical artifacts may propagate.
- Canonical S01 protocol: iter26 is parent and sole direct control; its exact Stage3 `test_recall@10=0.057017009349048554`, `n_eval=57439`. Iter18's exact `0.05988962203380978` is historical-only and `HISTORICAL_NONCOMPARABLE` for direct ranking.
- Approved S14 direction: one bounded alternative FCCR-1 mapping using the same behavior-branching and raw-residual-median inputs, holding all other iter26 conditions fixed. Do not select or stack behavior-loss, optimizer, Sinkhorn, cyclic/learnable curvature, Stage1, Stage3, or other mechanisms.
- User's inclusive success target: `test_recall@10 >= 0.065`. This is an evaluation criterion, not a predicted outcome. Stage2 SID quality metrics are descriptive only.

## Iter26 parent mapping and recorded inputs

The canonical iter26 hypothesis and `scripts/compute_closed_form_curvature.py` record:

```text
B = [19.324911558712664, 1.4605688962651735, 1.0148104414712726]
raw_residual_medians = [1.0, 0.10941, 0.09331]
c_base = 0.5
alpha = 0.2
m_min = 0.09331
s_l = log1p(B_l) / log1p(m_l / m_min)
z_l = (s_l - mean(s)) / (std(s) + 1e-12)
c_l = clip(c_base * exp(alpha * z_l), 0.05, 1.5)
parent_c_l = [0.6145357379232853, 0.5333020920777128, 0.3814078098431606]
```

Primary records: `stage2_RQ-VAE/curvature_RQ-VAE_iter26/logs/hypothesis_iter26.md`, `mechanism_manifest_iter26.md`, `mechanism_contract_iter26.json`, `scripts/compute_closed_form_curvature.py`, and `scripts/computed_behavior_branching.json`.

## Provenance caveat that must remain explicit

S01 did not certify raw-residual provenance. Historical `calibrate_residual_scales.log` files and `hypothesis_iter1.md` report `[1.0, 0.10941, 0.09331]` as raw medians and `[0.001, 0.932889, 1.0]` as their normalized layer-scale transform. The referenced baseline checkpoint is not currently present; iter26's computed JSON aggregates inputs from multiple sources. S03_PROVENANCE must independently adjudicate whether these records and code establish the declared semantics and source well enough to proceed. Do not claim checkpoint-level reproduction or treat this provisional input record as S03 approval.

## Candidate task

Agent A and Agent B independently propose one exact, falsifiable, mathematically valid, bounded, closed-form mapping

`(behavior_branching B_l, raw_residual_median m_l) -> fixed c_l`

that is a real alternative to the iter26 mapping and uses both declared inputs. Provide the full equation, all constants/ranges, and numeric substitution using the recorded vectors above; outputs must be finite and positive within the FCCR-1 implementation's supported bounds. Curvature is computed before Stage2 and remains non-trainable and identical at every training step. State why the proposed mapping may alter quantization geometry and plausibly downstream retrieval, plus exact implementation/mechanism falsifiers and risk limits. Treat numerical inputs provisionally pending S03. No source edits, Stage2 execution, GPU work, code/build/test/formatter validation, or protocol changes.

Each candidate writes only `logs/deliberation/S02_HYPOTHESIS/round_1/agent_a.md` or `agent_b.md` and begins with the required role, independence declaration, source-packet path, and stage ID. Include assumptions, primary evidence, exact formula and computed outputs, risks, self-rejection conditions, and the concrete next action. Do not read the other candidate or overwrite a canonical file. Skip formatters, linters, project-wide tests, and all training.