# Iter31 S10 Stage2 SID/geometry analysis — frozen source packet

`STAGE_ID=S10_STAGE2_ANALYSIS`
`ROUND=1`
`ITERATION=31`
`ITERATION_PURPOSE=PERFORMANCE_SEEKING_MECHANISM`
`SWEEP_OR_REPLICATION_ITERATION=NO`
`ROOT_CAUSE_ITERATION=NO`
`PACKET_STATUS=FROZEN`

## Active canonical contract and question

- Registered experiment: HRA-STEP6-1, common-reference right-nested Step6 aggregation; FCCR-1 remains fixed and unchanged.
- Registered equation and limitations: `logs/hypothesis_iter31.md`, `logs/hra_step6_contract_iter31.json`, `logs/mechanism_contract_iter31.json`; one-factor boundary: `logs/one_factor_diff_iter31.md`.
- Active Stage2 execution decision: `logs/stage2_execution_plan_iter31.md` and `logs/deliberation/S09_STAGE2_EXECUTION/round_1/judge.md`, with S09 gate-compatible evidence-only amendment in round_2 and full prelaunch evidence in `logs/stage2_preflight_iter31.md`.
- S09 authorized one Stage2 run after every immediate check passed. It did not authorize Stage3.
- S02's falsifiable downstream prediction is protocol-valid Iter31 `test_R@10` improvement over Iter29's exact `0.05921064085377531` and strict `test_R@10 > 0.065`. This is not established by Stage2 SID metrics.

## Completed run evidence

- `logs/stage2_run_iter31.md` and `logs/stage2_completion_iter31.md` capture the exact launch, supervisor, and terminal record.
- `logs/train_migrated.log` ends with `global_step=100000`, total time 1157.0 seconds, and successful final export; supervisor `Iter31Stage2` exited 0 with no restarts.
- Final FCCR-1 vector remained `[1.366095, 0.734783, 0.643996]`, equal to closed-form values within `1e-6` at step 100000.
- Final output routing is the Iter31 short results root; no model/SID products were written under the source tree.

## Stage2 output artifact identities

| Artifact | SHA-256 |
|---|---|
| `results/stage2_RQ-VAE/curvature_RQ-VAE_iter31/out/rqvae/instruments/rqvae_best.pth` | `d07953be968ea02cb8a6183935264c2af6315ea5195bd90c40764f39279961b7` |
| `results/stage2_RQ-VAE/curvature_RQ-VAE_iter31/out/rqvae/instruments/sids_raw.npy` | `bc5db3c9cc28e490cec8ecef0ec14ad135df027b8841d61399cafffde95cfaa5` |
| `results/stage2_RQ-VAE/curvature_RQ-VAE_iter31/dataset/Instruments/sids_for_hgrec.npy` | `398c13bd7439d87df01e197dbdf447162c5a32e1149be11d11a16c389166f138` |
| `results/stage2_RQ-VAE/curvature_RQ-VAE_iter31/item_sids.json` | `d4b100f68fcce4f5185f56f77fb8eef898b2b1dfa0d61d258422801239716dd7` |

Checkpoint CPU inspection through the prescribed Stage2 Python environment confirmed `global_step=100000`, keys `global_step/model/optimizer`, and 14 model tensors. Artifact validation confirmed raw SID `int32 [24587,3]`, final NPY `int64 [24587,4]`, JSON keys exactly `0..24586`, four values per item, full JSON/NPY equality, exact raw/final first-three-column equality, raw codes in `[0,255]`, and 24,587 unique extended rows. The final extension range is `[768,778]`.

## Registered S08 direct-effect evidence

`logs/mvg_check_iter31.log` is the hashed S08 `MVG PASS` artifact. On the same quantized embeddings and 640 ordered source/target pairs, HRA and the Euclidean comparator both returned finite `[640,32]` values; observed max absolute difference `0.0719804987`, L2 difference `2.01821685`, relative L2 difference `0.08106279`. This establishes a nondegenerate direct Step6 output difference on that registered batch, not a Stage3 benefit. S08 also verified the exact Iter8 checkpoint warm start and intended nonzero gradients; do not repeat a GPU/MVG run.

## SID observations to independently audit

Canonical metric implementation: `modules/sid_quality.py::evaluate_sid_quality` (descriptive only; `should_early_stop` always returns false). A fresh CPU invocation through the prescribed Python 3.9 environment produced:

- `n_items=24587`, `n_unique_full=22441` (2,146 duplicate three-token rows; collision-item fraction `0.0872818969`).
- `full_gini=0.0824256891`.
- `per_layer_gini=[0.1736523302, 0.2500036541, 0.2905509962]`.
- `l01_unique_pairs=13052`; `H(L1|L0)=5.3437931264` bits.
- All 256 codes appeared in each layer. Occupied-code count ranges: L0 `[29,197]`, L1 `[4,289]`, L2 `[1,284]`. Largest raw three-token tuple multiplicity was 11.

The trainer's rounded final log line matches: `full_gini=0.0824 per_layer=[0.1737/0.2500/0.2906] unique=22441/24587 l01_pairs=13052 H_l1_given_l0=5.3438`. All occupancy, collision, Gini, uniqueness and entropy statistics are descriptive only. No threshold or promotion gate applies to Stage2.

## Required independent analysis

Agent A and Agent B independently audit the same exact run and artifacts, including output identities/shape/wiring, Stage2/FCCR-1 contract compliance, S08 direct-effect evidence, and the canonical SID statistics. Separate observed facts, the preregistered Stage2/geometry prediction, and causal interpretation limits. Identify any mismatch or invalidity with exact evidence. Do not infer Stage3 performance from Stage2 proxies, do not call cleaner statistics a success, and do not make a promotion decision in S10.

The canonical output is `logs/sid_geometry_iter31.md`. It must distinguish (1) fixed-curvature contract compliance, (2) direct HRA-versus-Euclidean Step6 effect, (3) descriptive SID observations, and (4) interpretation limits. A numerically valid and contract-valid, non-aborted candidate proceeds to separately adjudicated S11 Stage3 wiring/evaluation. S10 grants no Stage3 authorization.

No source/model/output edits, training reruns, GPU use, Stage3 launch, or Stage2 metric gate is authorized in this analysis stage. No project-wide tests/formatters/linters.
