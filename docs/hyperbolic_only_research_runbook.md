# The hyperbolic-only research runbook (issue #110)

How the campaign is run, what may and may not be launched, and what each artefact means.
The scientific objective is unchanged from #109 and is not negotiable: one genuinely new,
non-equivalent, **inherently hyperbolic** mechanism whose single seed-42 export satisfies
**all three** frozen measurements at once.

| Frozen measurement | Strict threshold |
|---|---:|
| Hard-L1 enrichment ratio | > 16.38953942232631 |
| Stage-3 Recall@10 | > 0.06406796775709883 |
| Stage-3 NDCG@10 | > 0.03530651701940677 |

Control: the archived `r8_control` (ER 16.334505666275888, Recall@10 0.06406796775709883,
NDCG@10 0.03530651701940677). The R4-08 ER of 16.402627511591962 is descriptive, not a gate.

## The flow

```
bootstrap_archives_and_prior_art
 → scientific_analysis_pre
 → literature_search_or_verified_cache
 → hyperbolic_hypotheses
 → math_prior_art_review + freeze_proposal
 → hyperbolic_gate_H1_to_H7
 → implement_once
 → CGV_G1_to_G6 + complete_H_gates + cheap_probe_once
 → seal_implementation
 → one_official_cold_stage2
 → frozen_ER_and_D2_once + fixed_G_H1_once
 → if ER strictly qualifies: one_official_Stage3
 → scientific_analysis_post + ledger_update
 → next genuinely novel hyperbolic portfolio
```

Each candidate is an independently durable state machine (`tools/harness2_genrec/hyperbolic_campaign.py`):
`proposed → reviewed → implemented → screened → sealed → stage2_reserved → stage2_running →
stage2_complete → evaluated → stage3_reserved → stage3_running → stage3_complete → analysed`,
with the stops `rejected`, `novelty_unknown`, `blocked_literature`, `blocked_resources` and
`reconciliation_needed`. A research planning cycle is **not** a measure of completed
experiments, and a resource pause is **not** a failed science.

## What may be launched, and when

| Launch | Requires | Never |
|---|---|---|
| a probe | the arm's export exists, seed 42, the frozen probe budget | a second probe for the same export, under any name |
| the official cold Stage-2 | the proposal frozen, the H1–H7 certificate passed, the CGV G1–G6 certificate passed, the implementation sealed, the prior-art registry present and unedited | a second cold run for the same identity, a warm start, a control rerun |
| the frozen evaluation | the Stage-2 export final | a second read of the frozen pairs for the same arm |
| the official Stage-3 | the frozen ER strictly above the gate | a Stage-3 for an arm below the gate, a second Stage-3 for an arm |

The identity of an official experiment is the tuple
`(proposal_sha256, implementation_sha256, frozen_protocol_hash, seed=42, kind)`. It is
reserved before the launch and may never be reserved twice.

## The seeds

Seed 42 only, for every official Stage-2, Stage-3 and GRU probe. No seed 43 or 44, no seed
sweep, no ensemble. The historical multi-seed archives are read-only.

## The namespaces

* `results/hyperbolic_research_runs/` — this campaign: the archive import, the prior-art
  registry, the candidates, the audit log, the reserved experiments, the certificates.
* `results/harness3_runs/` and everything older — read-only. The import records what it
  found and hashes the frozen assets; it writes nothing there.

## The commands

```bash
# the forensic import and the prior-art registry
python3 -m tools.harness2_genrec.archives
python3 -m tools.harness2_genrec.prior_art

# the #109 lessons, re-checked
python3 -m tools.harness2_genrec.forensics_109

# the historical evidence baseline of section 4
/home/wlia0047/ar57_scratch/wenyu/genrec_env_v2/bin/python3.9 \
    tools/harness2_genrec/historical_evidence.py --out results/harness3_runs/historical_evidence

# the tests
/home/wlia0047/ar57_scratch/wenyu/harness2/.venv/bin/python -m pytest tools/harness2_genrec/tests -q
/home/wlia0047/ar57_scratch/wenyu/genrec_env/bin/python -m pytest \
    tools/harness2_genrec/tests/test_hyperbolic_geometry.py \
    tools/harness2_genrec/tests/test_hyperbolic_gates.py -q
```

## What counts as success

`is_verified_triple_win` in `tools/harness2_genrec/success.py`, and nothing else. It requires
the same proposal, implementation, protocol and seed; the seven hyperbolic gates and the six
CGV gates passed with none of them not-applicable or insufficient-evidence; the frozen ER
strictly above its gate with both shares recorded; and a complete audited Stage-3 whose
Recall@10 and NDCG@10 are both strictly above theirs. An ER-only win is not a win, a
Stage-3-only win is not a win, and a tie is not a win.

On a verified win: pause, publish the artefacts and their hashes, and **request the user's
explicit approval before closing the issue**. The user may also close it administratively at
any time; that is recorded as `closed_by_user_without_scientific_success`.
