ROLE=AGENT_A
INDEPENDENCE_DECLARATION=I did not read Agent B’s S07 candidate or any other S07 candidate before completing this audit. I used the frozen shared source packet, primary source/contracts and canonical S00–S06 Judge artifacts.
SOURCE_PACKET=stage2_RQ-VAE/curvature_RQ-VAE_iter31/logs/deliberation/S07_PREFLIGHT/round_1/source_packet.md
STAGE_ID=S07_PREFLIGHT

## Scope, independence, authorization
- Read the frozen S07 packet in full before repository evidence, after first reading the matching `skill://curvature-rqvae-iter` skill as required.
- Read canonical S00–S06 Judge decisions, S05 one-factor artifact and S06 implementation plan; inspected Iter31 applied source and the Iter29 parent Step6 implementation.
- Assignment authorization permits only the two separate static checker commands. It does not authorize edits, checker changes, tests, builds, MVG, GPU use, Stage2/Stage3, or runtime/performance inference. None of those were performed.
- `USER_INPUT_REQUIRED=NO`.

## Canonical S00–S06 and parent boundary
- S05 Judge, `logs/deliberation/S05_ONE_FACTOR/round_1/judge.md`, canonical decision: only scientific change is the registered HRA-STEP6-1 replacement in `_step6_sum_embeddings`; source/path changes are wiring and the new checker is audit tooling. It expressly leaves runtime activation and Stage3 route approval pending.
- S06 Judge, `logs/deliberation/S06_IMPLEMENTATION/round_1/judge.md`, verdict `MERGE_AB`, `S06_PLAN_APPROVAL=APPROVED`, and `HRA_CHECKER_CREATION_AUTHORIZATION=APPROVED_ONCE; EXECUTION=NOT_AUTHORIZED` at that stage. Its canonical plan authorizes the S07 audit/run, not MVG/training.
- S04 Judge authorizes the between-iteration HRA contract transition while preserving FCCR-1 as the fixed-curvature substrate. S01/S02/S03/S05 decisions preserve the parent/comparator, no-telescope caveat and historical provenance limits. S06 reconciles direct Stage3 path wiring but requires S11 to verify all consumers before any Stage3 run.
- Direct comparison: Iter29 `modules/rqvae.py:319–326` sets `embeddings = quantized.embeddings.sum(dim=0).transpose(0, 1)` and validates the rank/embedding dimension and finiteness. Iter31 changes this method’s operation. Iter31’s HRA checker source also compares the full `RqVae` module AST after replacing only this method with `pass`, and requires the parent’s Euclidean-sum AST (`scripts/preflight_hra_step6_iter31.py:245–270` approx.; see exact symbol references below). The full actual checker was inspected, but not executed.

## HRA Step6: exact structure, shape, order and guards
Iter31 `modules/rqvae.py`:
- `:319–330`: `_step6_sum_embeddings`; accepts only rank-3 input, exactly three layers and `embed_dim` in axis 1, with error text specifying `(3, embed_dim, batch)`. This is the required `(L,D,B)` layout.
- `:332–334`: reads batch size from axis 2, gathers each layer’s `get_c()` as `(1,1)`, and aliases `c0` to layer 0’s curvature; no independent reference-curvature parameter.
- `:336–342`: fixed loop `(0,1,2)` (L0,L1,L2); each `e_l` is transposed before geometry helpers, converting `(D,B)` to `(B,D)`. It applies source `exp0(c_l)` to tangent `e_l`, then source `log0(c_l)` to the ball point and target `exp0(c0)` to make `q_l^0`. It does not apply `log0` directly to tangent `e_l`.
- `:344–346`: first computes `q_1^0 ⊕_{c0} q_2^0`, then `q_0^0 ⊕_{c0} inner`, then final `log0(c0)`. This is the required exact right-nested tree, without reordering/reassociation.
- `:347–351`: checks output `(batch_size, embed_dim)`, checks finiteness, returns the tangent-coordinate result.
- `:386–390`: unchanged `forward` consumer assigns `_step6_sum_embeddings(quantized)` and passes it to `self.decode`; existing reconstruction curvature remains layer-0 curvature.
- Parent Iter29 source has the Euclidean sum shown above. The registered FCCR vector and S02/S04 equation agree with the observed source form.

**Domain limit:** these are source-level operations only. No observation was made of actual exp projection, log clamp, Möbius output ball membership, HRA-vs-Euclidean direct effect, gradients, runtime finiteness or downstream outcome. The registered HRA contract and checker expressly reserve those measurements for S08; no exact paper telescope, universal cross-curvature homomorphism, or performance conclusion is claimed.

## FCCR-1 / HRA contract separation and loader
- `logs/mechanism_contract_iter31.json:1–16`: exactly ten top-level fields: the eight FCCR fixed-curvature flags/source fields, `formula_inputs`, and `final_curvature_values`; vector `[1.3660953164241916, 0.7347829661951981, 0.6439958072706683]`. Formula declaration is `behavior_branching` plus `raw_residual_median`.
- `logs/hra_step6_contract_iter31.json:1–22`: separate `HRA-STEP6-1`, status active for Iter31 by S04 between-iteration transition, mechanism/equation ID, `paper_equivalence_claimed=false`, `d_hste_included=false`. `:25–47` links the inherited FCCR contract, fixed vector, L0/L1/L2 order and historical input JSON keys `branching` and `raw_residual_medians`. `:50–95` records the registered tangent/ball types, per-layer source curvature, target `c0`, leaf order, right-nested additions and final log. `:96–134` records unchanged/forbidden factors; `:136–154` declares both separate no-argument checkers and their roles.
- `curvature_RQ-VAE.py:305–323`: `_load_closed_form_curvatures()` points to `logs/mechanism_contract_iter31.json`, checks that the input file exists and explicitly rejects missing `branching` or `raw_residual_medians`, without fallback.
- `:326–367` recomputes the mapping and validates registered formula values/constants; `:369–382` requires exactly the ten-field FCCR schema and compares the recomputed final vector. Thus the FCCR loader is not expanded to absorb HRA fields.
- Historical input replay/hash limits remain as registered; this audit did not make a new measurement or claim a replay.

## HRA checker implementation (source audit only)
- `scripts/preflight_hra_step6_iter31.py:1–15`: static checker docstring says runtime proof is S08-only; resolves root relative to `__file__` and identifies both contracts, current and parent models; uses `ast`, JSON and `Path` without CLI parsing or environment override code.
- `:74–243`: checks FCCR exact field set/values/vector, HRA contract identity/transition/equation, FCCR linkage, layer/evaluation order, semantic types, helper caveats, one-factor scope and checker requirements.
- `:245–342`: source AST checks parent Euclidean sum; expected Step6 expressions and layer loop; type/orientation, result-shape and finite guards; prohibits direct `log0(e_l)`; verifies existing `forward` decoder consumer; requires byte equality for inherited `hyperbolic.py`, `quantize.py`, `loss.py` against Iter29. It also AST-normalizes `rqvae.py` after removing Step6 and compares to parent, enforcing no other RqVae AST change. This is exactly what the implementation contains as checks; the checks’ actual runtime verdict is unavailable because it was not executable here.
- `:344–359`: no-argument `main()` calls contract/source checks and, if those succeed, prints a static-pass line plus explicit runtime disclaimer and S08 requirements; no `argparse`, `parse_args`, `sys.argv`, `getenv`, or environment-override interface was found by targeted search.
- Scope nuance: the checker’s parent byte comparisons are explicit for the three named geometry/quantizer/loss modules and its AST comparison for `rqvae.py`; it does not byte-compare every copied file in `data/`, `init/`, or other module paths. The source-tree inventory and S05/S06 record classify those as unchanged, but I do not represent that as a result proven by this checker.

## Inherited source / one-factor evidence
- Direct parent/current reads show Iter29’s Euclidean Step6 and Iter31’s exact HRA replacement; the canonical S05 diff classifies this as the sole conceptual delta.
- The HRA checker’s source comparison covers the complete `RqVae` AST outside Step6 and exact bytes for `modules/hyperbolic.py`, `modules/quantize.py`, `modules/loss.py` (`preflight_hra_step6_iter31.py:245–342`). Those checks were read, not executed. S05/S06 classify the other copied modules, data, init and scripts as unchanged or separately non-scientific identity/audit tooling.
- No Step4, Step5, encoder, decoder, loss, quantizer, optimizer, Sinkhorn, schedule, Stage1, Stage3-model or evaluator change is visible in the registered Step6 operation. The model-behavior claim stays limited to Step6.

## Stage2 paths and source-tree product exclusion
- `curvature_config.py:10–16`: `_CONFIG_DIR`, `SAVE_DIR_ROOT`, `CONFIG_PATH`, `BEST_CKPT_PATH` point under `results/stage2_RQ-VAE/curvature_RQ-VAE_iter31/` with descriptive mechanism name `iter31_hra_step6_common_reference`.
- `:30–35`: `RQVAE_OUT_DIR` is the Iter31 results subtree; `RQVAE_CKPT_PATH` and `RAW_SIDS_NPY` derive from it; `SIDS_NPY` and `ITEM_SIDS_JSON` point to Iter31 results subtree. `curvature_RQ-VAE.py:448–455` shows export consumes those configured paths.
- Read-only source-tree listing showed code, configs’ references, scripts, input copy, logs, modules/data/init only; targeted glob for `.pth`, `.pt`, `.npy`, and JSON product-like paths found only the frozen input JSON under `scripts/` and two contract JSON files under `logs/`. No model checkpoint, generated SID array, or `item_sids.json` was present in the Iter31 source tree at inspection. This is a current listing observation, not a guarantee about future writes.
- The constants establish intended static path placement; I did not execute a resolver or launch Stage2. The later S09 mandatory path and current-identity checks remain outstanding.

## Stage3 direct-path wiring (static readiness only)
- `stage3_T5Train/train_HG-Rec.py:125`: `CODE_PATH` is exactly the Iter31 results `item_sids.json` path.
- `:132–156`: the variant map has an exact full-path key for Iter31 SID and maps it to `iter31_hra_step6_common_reference`; `RQVAE_VARIANT` derives from that lookup at `:156`.
- `:158–160`: `LOG_PATH` and `SAVE_PATH` use the required short `curvature_RQ-VAE_iter31` Stage3 result root; seed remains hardcoded 42.
- `:567–573`: the constants feed the trainer configuration; `:665–670` adds dataset/timestamp beneath those roots; `:827–845` places training metrics and DDP sync artifacts beneath log path; `:885–889` and `:957–966` save `HG_Rec_best.pth` under checkpoint path; `:1143–1145` writes `test_final.json` below log path.
- `:1175–1181`: launcher script remains direct `os.path.abspath(__file__)`; launcher log is the root-level Iter31 short results `_stage3_launcher.log`. The approved plan specifies the eventual outer `nohup` stdout/stderr redirection to Iter31 results `logs/_stage3_run.log`; this is not executed or encoded by the trainer itself.
- Static constants and consumer paths appear aligned with the approved direct route, but this is **not S11 approval or runtime route proof**. Do not infer SID existence/consumption, training outputs, or Stage3 readiness beyond static wiring.

## CLI / override scope
- Targeted searches found no `argparse`, `parse_args`, `sys.argv`, or `os.getenv` use across Iter31 Stage2 sources/scripts, Stage3 trainer, or the HRA checker.
- `curvature_config.py:38–52` sets hardcoded deterministic/cache/GPU environment values; these are fixed source literals, not environment-driven model-parameter overrides. Stage2 has standard DDP rank reads (`curvature_RQ-VAE.py:206–209`) and Stage3 launcher assembles a fixed subprocess environment from hardcoded launcher constants (`train_HG-Rec.py:1197–1211`); no user-facing CLI parameter override path was found. Checker commands are prescribed with no arguments and no environment overrides.

## Authorized commands, exact attempts and checker verdicts
Required commands, each exactly once from cwd `/home/wlia0047/ar57/wenyu/GeneRec/stage2_RQ-VAE/curvature_RQ-VAE_iter31`:
1. `python /home/wlia0047/.claude/skills/curvature-rqvae-iter/scripts/preflight_contract.py`
2. `python scripts/preflight_hra_step6_iter31.py`

**Execution status:** neither command was executed. The available tool inventory in this session has file read, grep, glob, web search and yield only; there is no shell/terminal/command-execution tool. I did not substitute a checker, use a different interpreter/path, or infer an exit code/output.

**Path/resource attempts:**
- Read `skill://curvature-rqvae-iter` first; the skill resource loaded.
- Read `skill://curvature-rqvae-iter/scripts/preflight_contract.py`; the skill-resource mapping returned the shared FCCR checker source, which I inspected. This proves source access via the resource mapping, not ability to execute the command.
- Attempted to read `/home/wlia0047/.claude/skills/curvature-rqvae-iter/scripts/preflight_contract.py`; the resource returned `Path ... not found` in this reader. The prescribed command path was not substituted or bypassed.
- The HRA checker source at `scripts/preflight_hra_step6_iter31.py` was directly readable in the Iter31 tree.

| Authorized checker | Invocation count | Full stdout | Exit status | Verdict |
|---|---:|---|---|---|
| Shared FCCR-1 preflight | 0 (not run) | unavailable; no command executed | unavailable | NOT RUN / UNKNOWN; do not record PASS or FAIL |
| Iter31 HRA Step6 preflight | 0 (not run) | unavailable; no command executed | unavailable | NOT RUN / UNKNOWN; do not record PASS or FAIL |

No expected output has been fabricated. Therefore this candidate cannot satisfy the packet’s required “both independently pass” evidence; S07 checker-execution evidence is blocked by the absence of a command-execution capability in this session. Static source review is separate from that missing proof.

## Findings and limits
- **Source equation / shape / order:** source inspection supports conformity to the registered HRA Step6 structure.
- **FCCR/HRA distinction and loader:** source inspection supports separate schemas, the fixed vector, explicit `branching` and `raw_residual_medians`, recomputation and exact FCCR schema validation.
- **One-factor / inherited sources:** canonical S05/S06 and parent/current source evidence identify Step6 as the sole model-behavior change; the HRA checker contains AST/byte-compare enforcement for its specified modules, but it was not run.
- **Paths:** Stage2 constants resolve by construction under the Iter31 results root; the source tree contains no discovered Stage2 result products. Stage3 route appears wired directly to Iter31 SID and short result roots from static constants/consumers only; later S11 remains mandatory.
- **No CLI:** no command-argument or environment-parameter override interface was found in target sources/checker.
- **Unresolved blocker/deviation:** both mandated checkers were not run and have no output or exit code because no execution tool is available. This is an execution-capability limitation, not a checker failure and not evidence of success. No checker substitution or bypass was attempted.
- **No edits/no training:** no files written or modified; no tests, build, formatter, MVG, GPU, Stage2 or Stage3 execution.

USER_INPUT_REQUIRED=NO