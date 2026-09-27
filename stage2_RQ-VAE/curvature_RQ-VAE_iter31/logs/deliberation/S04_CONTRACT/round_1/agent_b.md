ROLE=AGENT_B
INDEPENDENCE_DECLARATION=I did not read the other candidate before completing this artifact.
SOURCE_PACKET=stage2_RQ-VAE/curvature_RQ-VAE_iter31/logs/deliberation/S04_CONTRACT/round_1/source_packet.md
STAGE_ID=S04_CONTRACT
ROUND=1
RECOMMENDATION=MERGE-ELIGIBLE CONTRACT TRANSITION; S04 candidate recommendation only, not yet canonical until Judge C adjudicates
VERDICT=ACCEPT_CONTRACT_TRANSITION_WITH_DUAL_CONTRACTS
CONFIDENCE=MEDIUM
USER_INPUT_REQUIRED=NO
NO_CODE_OR_CHECKER_EDIT_AUTHORIZATION=TRUE
NO_IMPLEMENTATION_AUTHORIZATION=TRUE
NO_STAGE2_OR_STAGE3_AUTHORIZATION=TRUE
NO_TRAINING_AUTHORIZATION=TRUE
NO_PERFORMANCE_CLAIM=TRUE

## Decision

A sound contract path exists. The user-authorized Iter31 direction can be activated as a **between-iteration contract transition** because Iter30 was explicitly cancelled by user direction, has `SCIENTIFIC_RESULT=NONE`, and executed no Stage2, Stage3, or GPU training. Iter31 is a new registered iteration, not a continuation or repair of Iter30. The skill allows a canonically adjudicated between-iteration transition to replace the active hypothesis. S00–S03 independently established the Iter31 HRA Step6 hypothesis and provenance; S04 is the first stage that may formally change the contract. S03 itself did not authorize that change.

The transition must be narrow: **FCCR-1 continues as the inherited, unchanged fixed-curvature substrate; the new Iter31 top-level mechanism contract adds only the S02 HRA-inspired Step6 aggregation.** This does not reclassify the paper’s full HRA method as implemented, does not claim its exact telescope, and does not alter curvature, residual quantization/transport, or other mechanisms. The fixed FCCR-1 contract is necessary but not sufficient for Iter31 HRA compliance.

The mandatory `logs/mechanism_contract_iter31.json` should remain the exact ten-field FCCR-1 schema expected by both the Iter29 loader pattern and the current FCCR-1 skill preflight. Do not add HRA keys to that JSON: Iter29’s `_load_closed_form_curvatures` explicitly rejects any field set other than its expected fields plus `final_curvature_values`. A separate HRA contract and separate HRA checker avoid falsely treating a successful FCCR-1 check as an HRA check, avoid editing or disabling the shared skill checker, and make the two independent obligations machine-visible.

This is a contract-transition recommendation only. It authorizes no source edit, checker implementation/edit, implementation plan execution, Stage2/Stage3 run, or training. S05 onward and Judge C adjudication remain required.

## Proposed canonical files and complete draft schemas

Judge C should list both JSON files below in S04 `CANONICAL_ARTIFACT` if it accepts this design. The first is required by the skill and gate; the second is the explicit additional S04 canonical contract.

### A. `logs/mechanism_contract_iter31.json` — retained FCCR-1 substrate

This payload deliberately has exactly the loader-compatible fields; preserve values and key spellings exactly:

```json
{
  "contract_version": "FCCR-1",
  "curvature_source": "closed_form",
  "curvature_trainable": false,
  "curvature_time_varying": false,
  "uses_cyclic_schedule": false,
  "uses_curvature_regularization": false,
  "new_curvature_conditioned_optimizer": false,
  "new_curvature_conditioned_aux_loss": false,
  "formula_inputs": ["behavior_branching", "raw_residual_median"],
  "final_curvature_values": [1.3660953164241916, 0.7347829661951981, 0.6439958072706683]
}
```

These are the canonical L0/L1/L2 values from Iter29, not refreshed measurements. The formula inputs remain the recorded historical `branching` and explicit `raw_residual_medians` values; do not substitute normalized layer scales, `residual_norm`, or a fallback. The values and the distinction among raw median, normalized scale, and learnable scale must remain as in S03. Existing source buffer behavior and residual geometry are inherited unchanged.

### B. `logs/step6_aggregation_contract_iter31.json` — new HRA-inspired Step6 contract

The following is a proposed complete schema. It describes the registered S02 mechanism without making unsupported claims about exact HRA equivalence or runtime success.

```json
{
  "contract_version": "HRA-STEP6-1",
  "contract_status": "ACTIVE_FOR_ITER31_AFTER_S04_ADJUDICATION",
  "contract_transition": {
    "from_iteration": 30,
    "from_status": "ITERATION_CANCELLED_BY_USER_DIRECTION_CHANGE",
    "from_scientific_result": "NONE",
    "to_iteration": 31,
    "authority": "canonically_adjudicated_between_iteration_transition",
    "supersedes_only": "FCCR-1_MAPPING_ONLY_SCOPE_FOR_ITER31_MECHANISM_SELECTION",
    "preserves": "FCCR-1_FIXED_CURVATURE_SUBCONTRACT"
  },
  "mechanism_id": "HRA_INSPIRED_COMMON_REFERENCE_STEP6_AGGREGATION",
  "equation_id": "S02_ITER31_HRA_COMMON_C0_RIGHT_NESTED_V1",
  "paper_equivalence": "NOT_CLAIMED",
  "exact_telescope_claim": false,
  "d_hste_included": false,
  "layer_order": ["L0", "L1", "L2"],
  "inputs": {
    "e_l": {
      "semantic_type": "tangent_coordinate_quantizer_embedding",
      "source": "Quantize.forward selected embedding, including existing training STE forward value",
      "not": ["Poincare_ball_point", "semantic_id", "token_id"]
    },
    "c_l": {
      "semantic_type": "inherited_fixed_layer_curvature",
      "source_contract": "FCCR-1",
      "values_layer_order_L0_L1_L2": [1.3660953164241916, 0.7347829661951981, 0.6439958072706683],
      "trainable": false,
      "time_varying": false
    },
    "c0": {
      "semantic_type": "alias_of_c_l_layer_L0",
      "value": 1.3660953164241916,
      "independent_parameter": false
    }
  },
  "operations": {
    "q_l": "exp0(c_l, e_l)",
    "q_l_common": "exp0(c0, log0(c_l, q_l))",
    "h": "mobius_add(c0, q_L0_common, mobius_add(c0, q_L1_common, q_L2_common))",
    "z": "log0(c0, h)",
    "output_semantic_type": "tangent_coordinate_decoder_input",
    "decoder_consumer": "existing_decoder"
  },
  "evaluation_order": [
    "q_L0_common = exp0(c0, log0(c_L0, exp0(c_L0, e_L0)))",
    "q_L1_common = exp0(c0, log0(c_L1, exp0(c_L1, e_L1)))",
    "q_L2_common = exp0(c0, log0(c_L2, exp0(c_L2, e_L2)))",
    "inner = mobius_add(c0, q_L1_common, q_L2_common)",
    "h = mobius_add(c0, q_L0_common, inner)",
    "z = log0(c0, h)"
  ],
  "domain_and_numerics": {
    "exp0_output_projection": "existing helper projection to radius (1-eps)/sqrt(c), default eps=1e-6",
    "log0_input_clamp": "existing helper clamps sqrt(c)*norm argument to at most 1-1e-5",
    "origin_map_identity": "conditional_on_valid_unclipped_domain_only",
    "mobius_add_output_projection": "existing helper does not itself project output",
    "runtime_finiteness_and_ball_domain": "must_be_verified_in_S07_S08",
    "projection_and_clamp_incidence": "must_be_measured_in_S08",
    "no_new_scientific_numeric_threshold": true
  },
  "scope": {
    "replaces_only": "Iter29 Euclidean Step6 tangent sum e_L0 + e_L1 + e_L2",
    "preserves": [
      "FCCR-1 curvature source, values, and fixed-buffer behavior",
      "Step4 residual update",
      "Step5 cross-layer residual transport",
      "quantizer, codebooks, assignments, and existing STE",
      "all existing losses",
      "optimizer and Sinkhorn configuration",
      "encoder and decoder definitions",
      "Stage1 inputs and Stage3 model/data/evaluator/protocol"
    ],
    "forbidden_additions": [
      "d-HSTE",
      "curvature learning or scheduling",
      "new loss",
      "optimizer or Sinkhorn mechanism change",
      "residual update or transport change",
      "Stage1 or Stage3 change",
      "any second new mechanism"
    ]
  },
  "mathematical_limits": {
    "pointwise_exp_log_identity": "only_in_valid_unclipped_domain",
    "cross_curvature_transfer_homomorphism": "not_claimed",
    "heterogeneous_curvature_residual_inversion": "not_claimed",
    "exact_residual_reconstruction": "not_claimed",
    "exact_hra_telescope": "not_claimed"
  },
  "provenance": {
    "curvature_input_data_status": "historical_method_value_provenance_only",
    "historical_replay_or_hash_proof": false,
    "raw_residual_json_key": "raw_residual_medians",
    "raw_residual_fallback_allowed": false,
    "normalized_layer_scale_substitution_allowed": false,
    "fresh_measurement_allowed": false
  },
  "preflight": {
    "required_fccr_checker": ".claude/skills/curvature-rqvae-iter/scripts/preflight_contract.py",
    "required_hra_checker": "scripts/preflight_hra_step6_iter31.py",
    "checker_relationship": "both_must_pass_independently",
    "shared_fccr_checker_modification_allowed": false,
    "hra_check_may_be_omitted_if_fccr_passes": false,
    "hra_checker_authorization_stage": "S07_PREFLIGHT",
    "runtime_activation_and_gradient_gate": "S08_MVG"
  },
  "runtime_gate_requirements": {
    "verify_exact_expression_and_layer_order": true,
    "verify_tangent_to_ball_type_and_curvature_arguments": true,
    "verify_decoder_input_shape_and_finiteness": true,
    "verify_all_relevant_values_finite_and_domain_valid": true,
    "measure_projection_and_log_clamp_incidence": true,
    "compare_registered_step6_output_against_euclidean_sum_on_same_inputs": true,
    "invented_activation_threshold_allowed": false,
    "verify_intended_model_and_codebook_gradients": true,
    "verify_fixed_curvature_nontrainable_optimizer_excluded_and_invariant": true
  },
  "stage2_authorization": "requires_all_later_canonical_gates_and_root_CLAUDE_gradient_checks",
  "stage3_authorization": "requires_separate_later_canonical_S11_route_and_evaluation_gates"
}
```

Interpretation requirements for this schema:

- `h` is the exact S02 right-nested expression in ascending L0/L1/L2 leaf order. Do not reassociate it, reverse its leaves, or flatten it into a left fold.
- `q_l` explicitly names the ball-valued codeword before `log0`; `e_l` is tangent-valued. Never apply `log0` directly to `e_l`.
- Both members of each `q_l` conversion must use the stated source (`c_l`) and target (`c0`) curvatures. `c0` is exactly the L0 fixed curvature, not a new fit.
- The pointwise origin-map identity is conditional because the helpers project exponential outputs and clamp logarithm inputs. The Mobius helper itself does not project its output.
- This contract intentionally calls the method HRA-inspired. It does not claim to reproduce the paper’s shared-curvature paired-residual telescope, establish a cross-curvature homomorphism, or invert Iter29’s heterogeneous-curvature residual cascade.
- No numerical activation threshold is present in the frozen S02 packet. S08 may report measured differences and domain/clipping observations but must not invent a scientific cutoff after seeing results. An inability to establish a meaningful registered direct effect under the authorized protocol blocks Stage2; if the registered mechanism is demonstrated inactive and fixing that would alter the registered equation/parameters/protocol, apply the skill’s autonomous infeasibility/abort rule. If evidence is inconclusive without demonstrating inactivity, record the uncertainty and do not claim MVG PASS; do not silently retune or invent a cutoff.

## Evidence-backed rationale: facts versus proposal

### Direct repository and packet facts

1. **S02 freezes the equation and scope.** The frozen packet and canonical `hypothesis_iter31.md` specify tangent `e_l`; `q_l=exp0^{c_l}(e_l)`; radial transfer `q_l^0=exp0^{c0}(log0^{c_l}(q_l))`; exact `h=q_0^0 ⊕_{c0}(q_1^0 ⊕_{c0} q_2^0)`; `z=log0^{c0}(h)`; and replacement of only the Euclidean Step6 sum. No equation edits, layer reorder, reassociation, d-HSTE, or other new mechanism is authorized.
2. **FCCR preflight is specific and currently does not inspect HRA.** `.claude/skills/curvature-rqvae-iter/scripts/preflight_contract.py` pins `contract_version=FCCR-1`, the exact formula-input list, and positive finite curvature values; it scans curvature trainability, step-dependence, fixed-buffer presence, and curvature regularization. It prints `MECHANISM_CONTRACT_PASS`. There is no Step6 expression/HRA validation in that checker.
3. **The Iter29 loader enforces an exact FCCR schema.** `stage2_RQ-VAE/curvature_RQ-VAE_iter29/curvature_RQ-VAE.py::_load_closed_form_curvatures` checks the exact field set and values, requires explicit `branching` and `raw_residual_medians`, recomputes the fixed mapping, and compares its output. Arbitrary HRA fields added to that JSON fail the exact field-set test. Iter31 implementation must preserve that contract interface; any path adaptation belongs to a later implementation adjudication, not S04.
4. **The existing Step6 is Euclidean sum and downstream consumer is the unchanged decoder.** Iter29 `modules/rqvae.py::_step6_sum_embeddings` sums the layer embeddings; `forward` passes the result to `decode` and uses layer-0 curvature in reconstruction loss. S02’s proposed intervention changes only this aggregation.
5. **Existing helpers impose qualifications.** Iter29 `modules/hyperbolic.py` projects exp-map outputs, clamps log-map inputs, and provides Möbius addition without a final projection. Thus the unclipped identity is conditional and domain/finiteness/clipping are runtime questions, not facts proven by the equation alone.
6. **S03 provenance is explicitly bounded.** Curvatures and input values are inherited historical records with method/value provenance, not replayed historical data or fresh measurements. S04 must preserve, not strengthen, that claim.
7. **No paper/HRA exact telescope claim is supported.** Canonical S00–S03 explain that the cited paper’s telescope pairs left residual subtraction and reverse-nested aggregation at a shared curvature. Iter29’s residual path has layer-varying curvatures and explicit cross-layer transport. The Iter31 equation is an HRA-inspired common-reference Step6 proposal; exact inversion/telescoping is not claimed.
8. **The S04 deliberation gate requires the mandatory contract path.** `.claude/skills/curvature-rqvae-iter/scripts/deliberation_gate.py` requires `logs/mechanism_contract_iter31.json` as the S04 canonical artifact before Stage2. It does not itself run either mechanism preflight and does not block declaring the FCCR checker as the only checker; S07 must explicitly establish both checks and record their outputs.
9. **S08/Stage2 are not authorized by a contract.** The root rules and skill require further adjudications, runtime/gradient verification, identity/path checks, and all other gates before any training. S04 only recommends a contract design.

### Proposal/inference

- The logical transition from FCCR-1 to an Iter31 composite contract is valid only because the current skill explicitly permits a canonically adjudicated **between-iteration** transition. The transition is between canceled Iter30 (no scientific result) and new Iter31; it does not treat Iter30 as evidence or parent.
- Calling the resulting arrangement “FCCR-1 substrate + HRA-STEP6-1 mechanism contract” preserves the truth of the inherited fixed-curvature contract while formally superseding only the earlier restriction that the new mechanism must be a curvature-mapping change. The FCCR-1 substrate remains in force for all curvature properties; the new contract alone registers the aggregation intervention.
- Separate files are the narrowest compatible design: existing contract consumers remain fixed-schema, while a distinct contract/checker pair can fully represent and validate the new mechanism. No repository evidence establishes that the current generic FCCR checker can validate HRA, so treating it as sufficient would be an invalid false pass.
- S07 is the correct later authorization point for an iteration-local static checker. S07 A/B/Judge must agree on its scope and implementation/check method before the checker is created or run. It may inspect actual Iter31 source and the additional JSON, but must not rewrite, weaken, bypass, or change the behavior of the shared skill FCCR preflight. S07 then executes the unchanged FCCR preflight **and** the distinct HRA preflight. Both results and evidence are recorded in `logs/preflight_contract_iter31.log`; both must pass before MVG can proceed.
- S08 independently establishes runtime properties the static check cannot: actual tensor type/shape, curvature arguments, finiteness and domain validity, clipping/projection incidence, decoder-facing value, same-input comparison to the Euclidean Step6 output, intended gradients through model/codebook paths, and immutability/non-optimizer status of the fixed curvatures. Any static or runtime uncertainty blocks a PASS; no invented activation threshold is supplied here.

## Preflight and downstream execution path

### S04 now

1. Judge C independently checks this candidate and its sibling against the frozen packet and primary sources.
2. If accepted, Judge C writes both contracts as canonical S04 artifacts and lists both paths in `CANONICAL_ARTIFACT`. The required FCCR file remains at exactly `logs/mechanism_contract_iter31.json` with the exact ten-field schema.
3. No source or checker is changed or executed at S04. Stage2 remains prohibited.

### S05–S06

- S05 must independently lock the one-factor boundary: only Step6 Euclidean aggregation is replaced. Any inherited mechanism discrepancy or second conceptual delta blocks progression.
- S06 may plan/apply the model implementation only after its own Judge decision. The loader must consume an iteration-specific FCCR JSON preserving the exact schema; the existing Iter29 loader path currently names `mechanism_contract_iter29.json`, so an Iter31 implementation must resolve that path without adding HRA fields or weakening the validation. This is a later implementation concern, not an S04 source edit.
- S06 must implement exactly the S02 expression, preserve model/decoder interfaces and unchanged components, and not include d-HSTE or paper-level telescope logic.

### S07 static/contract preflight

- The current shared skill script remains authoritative for FCCR and is run unchanged, with no CLI arguments, from the Iter31 iteration directory. Required output: `MECHANISM_CONTRACT_PASS`.
- A separately adjudicated Iter31 checker (proposed path `scripts/preflight_hra_step6_iter31.py`) is implemented/authorized only through the S07 A/B/Judge process. It validates the additional contract against actual Iter31 source: HRA equation and exact right-nested L0/L1/L2 order; tangent versus ball semantics; source/target curvatures; only Step6 changed; decoder-facing tangent output; no d-HSTE/extra mechanism; fixed FCCR linkage; no unsupported telescope claims; and absence of FCCR checker edits. It cannot establish runtime activation, clipping rates, or gradient health.
- Run both checks as distinct commands, no CLI arguments, from the Iter31 directory. S07 must record command, exit status, full output, checker identity, and source/contract identity. The log may report the shared `MECHANISM_CONTRACT_PASS` alongside an explicit separate HRA pass marker; it must never represent the shared output as an HRA pass.
- If the HRA checker cannot be soundly implemented or its checks require an equation/scope change, do not treat the FCCR pass as sufficient. Return to adjudication; if the registered mechanism itself cannot be represented faithfully, Judge C applies `ABORT_ITERATION` under the applicable feasibility criteria rather than silently changing the equation.

### S08 MVG and S09 launch boundary

- S08 requires a separately adjudicated runtime procedure and evidence. It must inspect same-checkpoint/same-batch outputs for the registered aggregation and Euclidean comparator, verify finiteness and valid curvature/domain use, record helper projection/log-clamp incidence, verify output shape and decoder consumption, inspect intended encoder/codebook gradients, and establish that fixed curvature remains non-trainable, outside optimizer groups, and invariant. Root `CLAUDE.md` §6 gradient checks remain mandatory before actual training.
- The frozen S02 packet defines no numerical scientific activation cutoff. S08 may not invent one. It must report observed differences/limits and adjudicate whether it can establish the registered direct effect. A demonstrated inactive mechanism that would require a changed equation, parameter, or protocol to activate is an infeasibility/abort condition. Inconclusive evidence is not a license to claim MVG PASS; it remains blocked pending a sound adjudicated verification route.
- Only after both S07 static checks pass, S08 MVG passes, and all other pre-Stage2 requirements/identity/path/gradient gates are canonically satisfied may S09 consider Stage2. This candidate provides no such authorization.
- Stage3 remains unchanged and separately gated. Preserve S01’s unresolved route conflict for its later adjudicated resolution; no S04 permission to edit Stage3 or launch it.

## Hard gaps, failure/abort criteria, self-rejection

1. **Checker authorization is still pending.** The Iter31 HRA checker is proposed, not existing or approved. If S07 cannot independently specify a direct source-to-contract check without modifying/weakening the shared FCCR checker, HRA is not preflight-verified; the shared FCCR pass alone must block progression.
2. **Loader path needs later implementation compatibility.** Iter29 loader code points to its Iter29-specific FCCR contract path and validates an exact schema. If Iter31 implementation cannot use the Iter31 FCCR artifact while preserving exact loader invariants and fixed data inputs, the implementation is blocked. Do not put HRA fields into the FCCR JSON or relax the loader’s field check.
3. **Equation and domain are not runtime evidence.** Projection, clamp incidence, Möbius denominator/domain validity, output shape/finiteness, gradient continuity, and actual same-input direct effect are unobserved. S04 does not imply any of them pass.
4. **No activation threshold is authorized.** Do not invent a numerical delta/clip-frequency threshold. The absence of a preregistered numerical threshold is a genuine S08 decision constraint: the judge must derive an evidence-grounded, non-retuned verification interpretation or block/abort; it cannot manufacture a cutoff after observing outputs.
5. **Provenance remains historical.** No replay or fresh curvature measurement is permitted by this S04 proposal. Any new measurement, substitution, fallback, or claimed hash proof changes the registered inputs and is a hard contract/provenance failure.
6. **One-factor failure.** Any Step4/Step5, curvature, codebook, optimizer, loss, Stage1, decoder, or Stage3 change beyond S02’s decoder-facing Step6 input is a scope failure. A second mechanism is not a permissible repair.
7. **Mathematical overclaim.** Claiming the paper’s exact HRA telescope, universal cross-curvature homomorphism, or exact inversion of Iter29’s heterogeneous transported residual path self-invalidates this contract interpretation.
8. **Abort versus block.** A missing checker, static mismatch, implementation defect, or unresolved evidence blocks progression and may be repairable without changing the registered mechanism. Use `ABORT_ITERATION` only when direct evidence shows the registered mechanism itself is infeasible/inactive or continuing would require changing its equation, key constants, mapping, protocol, or one-factor boundary, consistent with the skill’s abort criteria. Do not label this canceled Iter30 as scientifically failed.
9. **Self-rejection condition.** I reject this recommendation if direct primary-source review shows no authority for a between-iteration contract transition, if the mandatory FCCR artifact cannot remain loader-compatible and independently checked, if HRA cannot receive its own explicit S07 checker, or if the proposed operation needs any change to canonical S02 equation, fixed values, or one-factor scope. In those cases, do not bypass the checker or silently retain an HRA label; Judge C should choose `ABORT_ITERATION` if the registered Iter31 mechanism is shown infeasible, otherwise `REJECT_BOTH`/replan as appropriate.

AUTONOMOUS_NEXT_ACTION=Judge C adjudicates the two independent S04 candidates against the frozen packet and primary checker/loader evidence, then canonically writes the required FCCR substrate contract and any approved HRA contract. If accepted, proceed to S05 only; do not edit code/checkers or launch training from S04.
