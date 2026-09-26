# SID geometry and Stage2 analysis — iter29

Canonical S10 analysis, adjudicated from `logs/deliberation/S10_STAGE2_ANALYSIS/round_1/source_packet.md`, both completed candidates (`agent_a.md`, `agent_b.md`), and the primary records and artifacts cited below. Facts and interpretation are separated; Stage2 SID metrics are descriptive, never gates.

## 1. Contract compliance and execution record

**Observed.** `logs/stage2_output_integrity_iter29.log:3-10,17-22` records one supervised no-argument invocation, `restarts=0`, terminal `exit=0`, 17m16s uptime, and completion at `global_step=100000` (worker-reported total 1013.1s, four ranks, per-GPU batch 640 / total batch 2560). Preserve the operational warning: the TCP readiness check on port **50200 timed out after 120 seconds while the process remained running**. The record reports no restart or second launch; `logs/train_migrated.log:5-27,260-269` subsequently shows DDP rank `0/4`, worker progress, and normal completion. The readiness timeout is not the terminal process result.

`logs/train_migrated.log:8-16,19-27,260-268` records the configured Stage1 embedding, 339,519 train transitions, 11 tensors loaded from the registered iter8 warm start, four-rank DDP, and the locked vector `[1.3660953164241916, 0.7347829661951981, 0.6439958072706683]`. The worker logs fixed-curvature invariant checks at steps 0, 25,000, 50,000, and 100,000, then saves the final checkpoint/raw SIDs and exports both SID formats. `logs/stage2_output_integrity_iter29.log:18-19,31` reports checkpoint step 100,000 and fixed-c buffers `[1.3660953044891357, 0.7347829937934875, 0.6439958214759827]`, the float32 representations of that vector.

**Input provenance limits.** The separate S09 immediate-prelaunch audit `logs/stage2_preflight_iter29.log:15-20` records these hashes as matching canonical S01: `sentence_t5.npy` `6490c71753952b2d8788bb7e4e52cff6a5b0b59819a2cfbd737bee7bc07a77fb`; `item_ids.json` `3df1f3ce6a468ae541c929a149f4dbd5e806f545b89d786b9c2d6801563b7c30`; `train.parquet` `80597e38d081029e57e7434c92ff7be2c872f82aebc2e5f072ef973d73e88815`; iter8 warm-start checkpoint `189e0affb156b9cdbd11f38a6fdb1127804e3f3095edd23c55f84825beb56c3b`. These are S09 prelaunch identity records, not S08-time hashes. S08 `logs/mvg_check_iter29.log:7-11` / `logs/deliberation/S08_MVG/round_2/judge.md:22-25` explicitly has no batch fingerprint and did not rehash Stage1/Stage0 files. S09 does not retroactively fingerprint the MVG batch; do not claim exact batch-row identity.

## 2. Direct curvature and assignment evidence

**Observed.** The worker's closed-form and live curvature snapshots match to the logged `1e-6` tolerance at all four recorded training milestones (`logs/train_migrated.log:15,19-20,84,141-142,260-261`). The final checkpoint-buffer values and step are recorded by the output-integrity smoke at `logs/stage2_output_integrity_iter29.log:31`; that smoke's report is the checkpoint-content evidence here, while the checkpoint file itself is independently hash-checked below.

S08's bounded MVG record reports candidate quantizer loss `2.10162615776062`, control loss `0.8013777732849121`, delta `1.300248384475708`, and assignment fraction changed `0.4062500298023224` (`logs/mvg_check_iter29.log:21`). `logs/mvg_check_iter29.log:20-24` and the adjudicated scope in `logs/deliberation/S08_MVG/round_2/judge.md:17-25` support the logged invariance, gradient/update, and counterfactual evidence as an implementation/activation check. This is not efficacy or downstream performance evidence.

## 3. Independently recomputed SID observations

Metric definitions were read from `modules/sid_quality.py:90-144` (sorted occurrence-count Gini, per-column occupancy Gini, unique complete tuples, unique `(L0,L1)` pairs, and weighted base-2 `H(L1|L0)`). Recomputed from the final raw NPY:

| Observation | Independent recomputation |
|---|---:|
| Raw SID shape / dtype / token range | `(24587, 3)` / `int32` / `[0,255]` |
| Full 3-token tuple Gini | `0.0877262221` |
| L0 / L1 / L2 occupancy Gini | `0.1749172899` / `0.3076309699` / `0.3610293021` |
| Unique 3-token tuples | `22301 / 24587` |
| Unique `(L0,L1)` pairs | `12960` |
| `H(L1|L0)` | `5.3012415701` bits |
| Excess rows in repeated 3-token tuples | `2286` |

The values round to the worker's final report at `logs/train_migrated.log:263-265` and the integrity summary at `logs/stage2_output_integrity_iter29.log:20`.

**Four-token assignment and JSON checks.** The exporter `curvature_RQ-VAE.py:428-443,465-510` groups equal raw tuples in item order and assigns extension `768 + occurrence` (`sum([256,256,256])` plus zero-based occurrence). Reapplying that rule independently to all raw rows reproduces the stored four-token NPY exactly. The NPY has shape/dtype `(24587,4)` / `int64`; its first three columns exactly equal raw SIDs cast to `int64`; all four-token rows are unique. There are 1,645 colliding tuple groups, maximum multiplicity 13, and extension IDs 768–780 with counts `{768:22301, 769:1645, 770:386, 771:138, 772:58, 773:29, 774:12, 775:8, 776:5, 777:2, 778:1, 779:1, 780:1}`. JSON contains exactly string keys `0..24586`, and every indexed JSON row equals its four-token NPY row. These complete-array checks agree with, and go beyond, the exporter's sample-row check and match `logs/stage2_output_integrity_iter29.log:31`.

**Output path/hash evidence.** SHA-256 was recomputed from the on-disk artifacts and matches `logs/stage2_output_integrity_iter29.log:24-28`:

| Artifact | SHA-256 |
|---|---|
| `results/stage2_RQ-VAE/curvature_RQ-VAE_iter29/out/rqvae/instruments/rqvae_best.pth` | `3a464a8de3ba3cec8dab0223b0705abb6e485567f19ddc1bf02db70a49df35bb` |
| `results/stage2_RQ-VAE/curvature_RQ-VAE_iter29/out/rqvae/instruments/sids_raw.npy` | `902fb9f72276d55433029a9f75b9e426518d22cd4277b2ee0fe5200fff158402` |
| `results/stage2_RQ-VAE/curvature_RQ-VAE_iter29/dataset/Instruments/sids_for_hgrec.npy` | `a8efe6e1315166486aff162e1bc6a5722b5d5d9ac1b537b83bc003ec0ea7cde1` |
| `results/stage2_RQ-VAE/curvature_RQ-VAE_iter29/item_sids.json` | `58665e08a97e122f47a77ed3616b67efdeb8eadba8253771a1498cb5e57487ff` |

The same primary integrity record reports `STAGE2_OUTPUT_INTEGRITY_PASS`, fresh outputs, and no forbidden model/array/JSON artifacts under the source iter29 tree (`:24-33`).

## 4. Interpretation limits and next action

**Interpretation.** The recorded execution, fixed-curvature snapshots, final checkpoint report, and aligned SID exports satisfy S10 execution/export analysis. Gini, collision/uniqueness counts, occupancy, and conditional entropy are descriptive only under the project policy and FCCR-1; `modules/sid_quality.py:147-148` also confirms no SID-metric early-stop. They must not be gates, quality rankings, or proxies for recall. The S08 quantizer counterfactual does not establish Stage2 efficacy. S09 prelaunch file hashes and S08 MVG evidence do not remove the batch-fingerprint limitation.

No Stage3 run or result exists (`logs/stage2_output_integrity_iter29.log:34`). No downstream efficacy, recall, or promotion claim follows. The one authorized next action after this S10 decision is **S11 Stage3 wiring/evaluation audit and adjudication**. This S10 artifact does not authorize or execute a Stage3 launch.
