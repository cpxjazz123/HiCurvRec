# The Stage2→Stage3 transfer failures, read against the subspace hypothesis (issue #111, Agent A)

**Scope.** A retrospective, evidence-specific diagnosis of the #109 cohort, written to test whether
#111's hypothesis — that *forcing semantic hierarchy, behaviour discrimination, history
predictability and uniqueness into one undifferentiated geometry* produces conflicting objectives —
is supported by the measurements, and what a factorised (subspace / product-manifold) representation
would have to change to be worth building.

**Everything below is measured or derived from files already on disk.** No number in this document
was produced by a new run, and none is estimated. Every claim carries its epistemic label:

| label | meaning |
|:---|:---|
| `measured` | read directly from a named file and field |
| `derived` | a computation over measured values, with the method stated |
| `inferred` | a hypothesis about what caused the measurement |
| `unknown` | not present in the evidence |

## 0. Provenance

| what | where |
|:---|:---|
| per-arm diagnostics (ER, shares, G_H1, occupancy, branching, downstream) | `results/hyperbolic_research_runs/analysis/failure_analysis.json` → `arms[].diagnostics` |
| cohort control and thresholds | same file → `control`, `thresholds` |
| the research questions #110 already framed | `results/hyperbolic_research_runs/analysis/research_questions.json` |
| the causal hypotheses already tested | `results/hyperbolic_research_runs/analysis/causal_hypotheses.json` |
| the #109 forensics | `results/hyperbolic_research_runs/issue109_forensics.json` |
| the archived manifests behind the diagnostics | `results/harness3_runs/**` (read-only) |

**Control** (`failure_analysis.json` → `control`): Hard-L1 ER `16.334505666275888`, numerator positive
share `0.07908428720083246`, denominator negative share `0.004841547630309337`, L0 distinct codes
`245`, L0 Gini `0.21624933700653148`, L0 largest cell `237`, L1→L2 branches `65.90612244897959`,
Stage3 Recall@10 `0.06406796775709883`, NDCG@10 `0.03530651701940677`. The control's `g_h1` is
`null`: the control was never probed.

**Frozen thresholds** (`thresholds`): ER `> 16.38953942232631`, Recall@10 `> 0.06406796775709883`,
NDCG@10 `> 0.03530651701940677`.

**Cohort shape** (`measured`): 32 arms, 30 measured, 14 with a downstream pair, 15 with any
downstream metric; failure kinds — `mechanism_falsified` 14, `statistical_low_support` 13,
`no_failure_identified` 3, `implementation_failure` 2.

**Derived identity.** For every arm that reports both shares, ER is exactly
`positive_share / negative_share` (checked on 8 arms; e.g. control `0.07908428720083246 /
0.004841547630309337 = 16.334505666275888`; `behaviour_residual_geometry_transport`
`0.07859237536656892 / 0.004729921483303377 = 16.616`). So **ER is a ratio of two shares that can
move independently**: an arm can raise it by making the positive pairs share more (numerator up) or by
making the negative pairs share less (denominator down). That distinction is not recorded for 4 of
the 14 gate-clearing arms (their `numerator_positive_share` / `denominator_negative_share` are
`null`), so the *mechanism* of their gain cannot be read. `unknown`

## 1. The one fact that dominates everything

`derived`, from `failure_analysis.json` → `arms[]`:

* **14 arms cleared the ER gate. Zero of them beat the control on Recall@10. Zero of them beat the
  control on NDCG@10.** Every arm with a downstream measurement sits *below* the control on both
  metrics.
* Spearman rank correlation over the arms that carry both fields (mid-rank Spearman, ties averaged):
  * `ρ(Recall@10, ER) = -0.064` (n = 14)
  * `ρ(Recall@10, G_H1) = +0.462` (n = 13)
  * `ρ(ER, L0 distinct codes) = +0.696` (n = 30)
  * `ρ(ER, L0 largest cell) = -0.387` (n = 30)
  * `ρ(ER, G_H1) = -0.329` (n = 27)

`inferred`: the enrichment ratio is not merely a weak predictor of the downstream — in this cohort it
is **uncorrelated** with it, and it moves most strongly with *how many* distinct level-0 codes the
arm uses. The ratio rewards a distributional coincidence between the train transition pairs and the
L1 assignment; nothing in the cohort demonstrates that the coincidence is the thing Stage3 needs.

`inferred`: `ρ(ER, L0 distinct codes) = +0.696` with `ρ(ER, largest cell) = -0.387` says the
ER-winning arms are the ones with a **finer** level-0 partition, not a coarser one. That kills the
intuitive story ("ER went up because the L1 partition got coarser") and leaves the harder one: a
finer L1 partition raised ER and *still* lost downstream. `unknown` whether the arms' finer
partitions are finer in the *history-relevant* direction or merely in the *frequency* direction.

## 2. Case studies

### Case A — the highest ER in the cohort, and the worst downstream among the well-behaved arms

`directed_assignment_prefix_preservation` — `measured`:

| quantity | value | source field |
|:---|---:|:---|
| Hard-L1 ER | `17.298273785628265` (cleared) | `diagnostics.enrichment_ratio.value` |
| numerator positive share | `null` | `diagnostics.enrichment_ratio.numerator_positive_share.value` |
| denominator negative share | `null` | `diagnostics.enrichment_ratio.denominator_negative_share.value` |
| G_H1 | `0.12126740354150722` | `diagnostics.history_gain.value` |
| GRU probe CE / frequency-prior CE | `4.741277180168897` / `5.395585869099887` | `history_gain.probe_ce_l1`, `history_gain.prior_ce_l1` |
| probe top-1 / top-5 / top-20 | `0.0886079208940619` / `0.21975588401915847` / `0.41046505639388164` | `history_gain.top1_acc_l1`, `top5_acc_l1`, `top20_acc_l1` |
| L0 distinct codes / Gini / largest cell | `255` / `0.22925840771904804` / `374` | `diagnostics.level0_usage.*` |
| Δ largest cell vs control | `+137` | `level0_usage.delta_largest_cell_vs_control` |
| L1→L2 branches (Δ) | `63.76470588235294` (`-2.1414165666266456`) | `diagnostics.prefix_tree.*` |
| L2→L3 branches | `1.4544280442804427` | `prefix_tree.l2_to_l3_branches` |
| sequence information | `0.10452139644614881` | `diagnostics.sequence_information.value` |
| Recall@10 (Δ vs control) | `0.060185588189209424` (`-0.0038823795678894074`) | `diagnostics.downstream.*` |
| NDCG@10 (Δ vs control) | `0.032655356455760634` (`-0.0026511605636461363`) | `diagnostics.downstream.*` |
| recorded verdict | `mechanism_falsified` | `failure.kind` |

**Observation** (`measured`): the cohort's best ER arm is 0.909 above the gate and 0.964 above the
control, and it is 0.0039 *below* the control on Recall@10 and 0.0027 below on NDCG@10 — the
largest downstream loss among the arms whose L0 usage is still near the control's.

**Candidate explanation** (`inferred`): the arm's "prefix preservation" objective bought ER by
re-arranging which train pairs collide at L1 in a way that *reduces the information the L1 token
carries about the next item*. Its largest L0 cell grew by 137 items (374 vs 237) and its L1→L2
branch count fell by 2.14 — so the L1 partition did get coarser in the *tail* even though the
distinct-code count stayed at 255.

**Alternative explanation not involving any subspace or entanglement story** (`inferred`): the arm
is simply **a better ER optimiser**. The ER gain needs no structural explanation: any objective
that makes the transition pairs collide more, relative to the random pairs, raises the ratio. The
downstream loss then follows from the *control* being a better ranking model, not from any conflict
between objectives inside the arm. This explanation requires no geometry claim at all.

**Prediction if the subspace hypothesis is right** (`inferred`): a factorised representation that
gives the hierarchy its own factor and the behaviour its own factor should let the ER be raised
*without* the L1 tail absorbing the transition pairs — i.e. the L0 largest cell and the L1→L2
branch count should stay at the control's values (`237`, `65.91`) while the ER rises.

**Falsifying test** (`measured` — checkable on the next arm's artifacts): if a factorised arm raises
ER above `16.38953942232631` *and* its L0 largest cell exceeds `374` or its L1→L2 branch count falls
below `63.76`, then the factorisation bought its ER the same way Case A did, and the subspace
hypothesis has not addressed the transfer failure.

### Case B — an ER gain with no structural damage, and still no transfer

`directed_assignment_prefix_information` — `measured`:

| quantity | value |
|:---|---:|
| ER | `16.946938775510205` (cleared) |
| numerator / denominator shares | `null` / `null` |
| G_H1 | `0.12137791725688264` |
| probe CE / prior CE | `4.720922157006461` / `5.3730975464074655` |
| probe top-1 / top-5 / top-20 | `0.08482257815316475` / `0.215275274244219` / `0.42241334912705364` |
| L0 codes / Gini / largest cell | `256` / `0.23518923236873146` / `253` (Δ `+16`) |
| L1→L2 branches (Δ) | `63.58984375` (`-2.3162786989795876`) |
| L2→L3 branches | `1.4572762454696235` |
| sequence information | `0.10020646856287865` |
| Recall@10 (Δ) | `0.06375459182785216` (`-0.00031337592924667035`) |
| NDCG@10 (Δ) | `0.03467522673433716` (`-0.0006312902850696134`) |
| verdict | `mechanism_falsified` |

**Observation** (`measured`): this arm's L0 usage is essentially the control's (255/256 codes, Gini
`0.235` vs `0.216`, largest cell `253` vs `237`) and its downstream loss is the *smallest* of the
gate-clearing cohort — and it is still a loss. So there exists an ER gain that damages nothing
measurable and still does not transfer.

**Candidate explanation** (`inferred`): the ER gain here is a *denominator* effect — fewer random
pairs share a code — and the downstream metrics are simply insensitive to it. Whatever the L1 token
is for Stage3, "random pairs share a code less often" is not it.

**Alternative explanation** (`inferred`): the downstream metrics are noisy at the ±0.0003–0.0006
level, and the arm's loss is within that noise, so the arm is neither a win nor a loss. This
explanation is consistent with every number here and does not need a geometric cause.

**Prediction if the subspace hypothesis is right** (`inferred`): if the factors are separated, the
ER gain and the downstream gain should be *jointly* achievable — the factorised arm should be the
first in the cohort with Recall@10 and NDCG@10 above the control *while* ER is above the gate. If a
factorised arm produces the same "clean but useless" ER gain as Case B, then the factorisation is
orthogonal to the quantity Stage3 needs.

**Falsifying test**: `measured` — a factorised arm that clears the ER gate and whose Recall@10
remains ≤ `0.06406796775709883` reproduces Case B and refutes the subspace hypothesis as an
explanation of the transfer failure.

### Case C — the highest history gain in the cohort comes from a degenerate partition

`behavior_directed_code_assignment_cross_entropy` — `measured`:

| quantity | value | source field |
|:---|---:|:---|
| Hard-L1 ER | `1.478111600364957` | `enrichment_ratio.value` |
| numerator positive share | `0.7754611673446221` | `enrichment_ratio.numerator_positive_share.value` |
| denominator negative share | `0.5246296471478573` | `enrichment_ratio.denominator_negative_share.value` |
| G_H1 | `0.28951281755814723` | `history_gain.value` |
| probe CE / prior CE | `0.47341677238915775` / `0.66632697125103` | `history_gain.probe_ce_l1`, `prior_ce_l1` |
| probe top-1 / top-5 / top-20 | `0.779368594530566` / `1.0` / `1.0` | `history_gain.top1_acc_l1` … |
| L0 distinct codes | `2` | `level0_usage.distinct_codes.value` |
| L0 Gini / largest cell | `0.01462154797250581` / `12653` (Δ `+12416`) | `level0_usage.*` |
| L1→L2 branches (Δ) | `203.5` (`+137.5938775510204`) | `prefix_tree.*` |
| L2→L3 branches | `39.76904176904177` | `prefix_tree.l2_to_l3_branches` |
| sequence information | `0.18766254613143885` | `sequence_information.value` |
| Recall@10 / NDCG@10 | `null` / `null` | `downstream.*` |

**Observation** (`measured`): the arm with the **highest G_H1 in the cohort** (`0.28951281755814723`, 2.3× the
typical well-behaved arm's ≈ `0.123`) is the arm whose L0 partition **collapsed to two codes**: 2
distinct codes, largest cell 12653, Gini `0.0146`. Its GRU probe reaches top-1 accuracy `0.779` and
top-5/top-20 accuracy `1.0` — because the L1 token has almost no entropy left to predict.

**Candidate explanation** (`inferred`): **G_H1 is a partition-entropy measurement, not a
predictability measurement.** A degenerate L0 partition makes the probe's conditional entropy
collapse toward a two-way prediction, which maximises the gain over the frequency prior. The gain
then says nothing about whether the SID is informative.

**This is the decisive confound for #111's diagnosis.** If a factorised representation is judged by
"a higher G_H1", it can be maximised by the same degeneracy that destroyed the SID. Any
factorisation claim must therefore report G_H1 **at matched L0 occupancy**, and the cohort's G_H1
values are not comparable across arms with different code counts.

**Alternative explanation** (`inferred`): the arm is simply broken as an experiment — a
cross-entropy assignment objective that collapsed the codebook is an implementation failure that
happens to have been measured, and its G_H1 is an artefact rather than a signal. The recorded
failure kind for this arm is `statistical_low_support`, which is itself a mis-classification: the
arm has no downstream measurement, but it has a *falsified* mechanism (2 codes).

**Prediction if the subspace hypothesis is right** (`inferred`): factorising the representation
should make the L0 usage *more* constrained, not less — a factor dedicated to uniqueness should
keep the L0 codes used at or above the control's `245` while the hierarchy factor changes the
branching. If a factorised arm's G_H1 rises while its L0 distinct codes fall below `245`, the gain
is the Case C artefact and not a disentanglement.

**Falsifying test**: `measured` — report `history_gain.value` **together with**
`level0_usage.distinct_codes.value` and `level0_usage.largest_cell.value` for every candidate, and
treat any G_H1 gain accompanied by a distinct-code count below `245` or a largest cell above `560`
(`directed_capacity_preserving_assignment_repartition`'s value, the cohort's worst well-populated
tail) as uninformative.

### Case D — two arms that are bit-identical to the control, and one that is not the control it claims

`behavior_conditioned_pairwise_margin` and `curvature_weighted_transition_transport` — `measured`:
both report ER `16.334505666275888`, positive share `0.07908428720083246`, negative share
`0.004841547630309337`, L0 codes `245`, Gini `0.21624933700653148`, largest cell `237`, L1→L2
branches `65.90612244897959`, L2→L3 `1.4651018765095682`, sequence information
`0.10444776286452304` — **identical to the control to the last digit**, the same numbers the
control's own record carries.

But `behavior_conditioned_pairwise_margin` additionally reports `G_H1 = 0.12409012631042572` with a
probe CE of `4.684429098171414` against a prior CE of `5.348072032159319`, while the control's
`g_h1` is `null`. Two arms whose Stage-2 artefacts are bit-identical to the control cannot have
different history probes over the same export.

**Observation** (`derived` + `inferred`): either those arms' mechanisms never fired (a no-op
arm, and their "ER" is the control's), or their probe was computed against a different artefact
than the one the diagnostics report. This is an **evidence-provenance inconsistency** that must be
resolved before any of these arms is used as evidence for or against a hypothesis. `unknown`

**Prediction / falsifying test** (`measured`): a factorised arm must show a *non-control* L0 usage
fingerprint — a distinct-code count, Gini, largest cell or branch count that differs from the
control's — otherwise its mechanism is a no-op and its ER is the control's. The
`directed_separated_codeword_geometry` arm is the precedent: its recorded failure kind is
`implementation_failure` with the evidence `the independent checks failed: ['gradient_target']`,
and its Stage-2 fields are all `null` — it was caught. The two identical-to-control arms were not.

### Case E — an arm whose ER gain came from the numerator, with a damaged tail

`behaviour_conditioned_residual_target` — `measured`: ER `16.503445774392453` (cleared), positive
share `0.08608457099612146` (control `0.07908428720083246`, `+0.007`), negative share
`0.005216157411786965` (`+0.000374609781477628`), G_H1 `0.12575796162314637`, L0 codes `243`, Gini
`0.35277299506363646` (Δ `+0.13652365805710498`), largest cell `382` (Δ `+145`), L1→L2 branches `61.629629629629626`
(Δ `-4.276492819349961`), Recall@10 `0.0610038475600202` (Δ `-0.0030641201970786347`), NDCG@10
`0.03381020016512428` (Δ `-0.0014963168542824895`).

**Observation** (`measured`): here the *positive* share rose (more transition pairs sharing their L1
code) — the mechanism made the L1 partition coarser in the transition-relevant sense — and the
downstream loss is again the second largest in the gate-clearing cohort.

**Alternative explanation** (`inferred`): the positive share rising and the largest cell rising by
145 are the *same fact* seen twice. The arm did not gain "enrichment"; it concentrated transition
pairs into fewer L1 codes. That is a structural statement about the partition, and it is exactly the
kind of thing a factorisation is supposed to fix — but it is equally explainable as "the arm's
objective rewards concentration", with no entanglement claim required.

## 3. What the evidence supports, and what it does not

**Supported (`derived` / `inferred`):**

1. In this cohort the three objectives were never jointly achieved: 14 arms cleared ER, 0 improved
   Recall@10, 0 improved NDCG@10.
2. The ER gain is not tied to the L1 partition's *informativeness*: the ER-winning arms have *more*
   distinct L0 codes (`ρ = +0.696`), and Case B shows a gain with control-like occupancy that still
   does not transfer.
3. G_H1 is confounded by partition degeneracy (Case C) and therefore cannot be read across arms
   with different code counts.
4. D2 is not usable as a discriminator here: `conditional_l2_discrimination` is `null` for every arm
   except `directed_tangent_codebook_rotation` (`0.04646483838959613`), whose failure kind is
   `no_failure_identified` — the conditional support is too thin to order the cohort.

**Not supported / not explainable by the subspace hypothesis:**

1. The retrospective evidence does **not** establish the *cause* of the transfer failure. Every
   failure above has a non-geometric explanation that fits the same numbers: the ER ratio rewards a
   distributional coincidence; the downstream metrics are insensitive to it; G_H1 rewards degeneracy;
   two arms are no-ops with a provenance inconsistency.
2. The "one undifferentiated geometry" story would predict that arms touching *different* parts of
   the objective (assignment, prefix, codebook geometry, capacity, residual transport) fail in
   *different* ways if the conflict is inside the geometry. The cohort shows a *uniform* failure:
   every measurable arm loses downstream. A uniform failure is more consistent with "the ER proxy and
   the downstream metric measure different things" than with a specific intra-geometry conflict.
3. Nothing in the cohort measures a *factor* structure at all. There is no decomposition to
   disentangle, so the hypothesis has no positive evidence in this data — it has only the absence of
   a joint win.

**Therefore** (`inferred`): the subspace hypothesis is **not established** by the retrospective
cohort; it is a *design hypothesis* that the cohort is consistent with but does not test. The
cohort's actual lesson is narrower and harder: **the ER proxy needs a structural guardrail** (the
L0 occupancy, the branch counts and the G_H1 degeneracy check above are exactly such guardrails),
and any candidate — factorised or not — must clear them *before* its ER is read as progress.

## 4. The tests a factorised candidate must pass (falsifiable, and checkable on artifacts)

| # | test | measured threshold / source | what it refutes |
|:--:|:---|:---|:---|
| T1 | the arm's L0 usage differs from the control's | codes ≠ `245`, largest cell ≠ `237`, Gini ≠ `0.21624933700653148`, branches ≠ `65.90612244897959` | a no-op mechanism (Case D) |
| T2 | G_H1 is reported only at matched occupancy | `history_gain.value` printed together with `level0_usage.distinct_codes.value` ≥ `245` and `largest_cell.value` ≤ `560` | the Case C degeneracy artefact |
| T3 | the ER gain's *source* is recorded | both `numerator_positive_share` and `denominator_negative_share` present (non-`null`) | an unreadable ER gain (4 of 14 gate-clearing arms) |
| T4 | the ER gain does not come from concentration | largest cell ≤ `374` and L1→L2 branches ≥ `63.76` (Case A's values) | the Case A / Case E route |
| T5 | the downstream pair moves together with ER | Recall@10 > `0.06406796775709883` **and** NDCG@10 > `0.03530651701940677` **and** ER > `16.38953942232631` | the whole cohort's uniform failure |

T1–T4 are *pre-registration* guardrails for the next candidate; T5 is the frozen success predicate
and is not a proxy.

## 5. Hygiene

* The cohort has already been inspected by the #109 campaign and by this analysis, so every
  correlation here is **exploratory**. The `ρ(Recall, G_H1) ≈ +0.46` (n = 13) against
  `ρ(Recall, ER) ≈ -0.06` (n = 14) is a hypothesis about ordering, **not** a certified oracle, and
  #110 already records it as such.
* #109's campaign used frozen Stage3 outcomes adaptively while designing later mechanisms. That
  makes the multi-round optimisation exploratory even with a single seed. This document therefore
  does **not** nominate a mechanism, and the candidates in `subspace_mechanism_proposals.json` are
  selected on mathematics and prior art, never on a frozen Stage3 number.
* No frozen test label may enter a training loss. A factorised candidate may use **train-split
  behaviour pairs only**; the GRU probe stays an independent frozen measurement (history length 20,
  embedding/hidden 128, 1 layer, 6 epochs, batch 512, Adam 1e-3, user-disjoint split, seed 42) and
  is never replaced by a train-time auxiliary CE.
* The `geometry_transport_seed43` and `r8_control_seed43` rows are seed-43 diagnostics; the frozen
  protocol is seed 42 only, and they are not admissible as official comparisons.
