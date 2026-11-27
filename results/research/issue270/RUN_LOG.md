# Issue #270 run log

- Issue: #270 — Identify the curvature–SID–recommendation relationship using archived runs and
  triplet-level analysis
- State at freeze: OPEN
- Body sha256: `04c90c24012592cf6928530abbe9f2eed93f20c5632807d9611cc6ada24b0f39` (2,837 chars)

## INFRA_MODE

`FROZEN`. The issue states: "单 Agent，INFRA_MODE=FROZEN，只复用既有图、全量三元组、Graphlet/Orbit、
训练/验证数据、R2四件套与已有评估工具。不修改公共Infra、不重训TIGER、不新增模型训练、不启动
Stage2/Stage3、不修改机制；允许研究专属分析脚本。" No sentence authorises building or repairing
public infrastructure. Candidate-local analysis scripts live under `results/research/issue270/`.

## Does this issue require training

**No.** The issue is analysis-only: it re-reads archived R2 four-piece sets and computes geometry,
SID and recommendation-prediction linkages; it explicitly forbids Stage2/Stage3 and retraining.

## Acceptance checklist C1..C10

| ID | Requirement | Output | Status |
|---|---|---|---|
| C1 | Inventory all completed GNN/gating/sectional-curvature R2 sets (#259–#269 incl. #265/#267/#269): run, checkpoint/SID hashes, mechanism, protocol, GPU, split, curvature loss, Recall/NDCG, predictions; mark missing/duplicate SIDs; TIGER baseline and rerun range; single vs four cards; freeze common triples/references/mask/scale/target; separate J / A.1 error / fitted kappa | `RUN_INVENTORY.json`, `COHORT.json`, `INVENTORY.md`, `CONTRACT.md` | PENDING |
| C2 | Full-triplet geometry per recoverable candidate: hard q from checkpoint/SID, fixed common scale, curvature error/sign/within-centre ordering/Pearson/Spearman/J; #269 kappa_t as extra common ruler; z separately | `assets/geometry/`, `GEOMETRY.md` | PENDING |
| C3 | Strata: directed graphlet, orbit role, direction, sign/magnitude, degree, transition frequency, co-occurrence, closure, centre participation; macro (centre-equal)/micro (row-equal)/frequency-weighted; per-candidate TIGER deltas; full shards + manifest | `assets/geometry/*.npz` + manifest, `STRATA.md` | PENDING |
| C4 | Curvature -> SID: per-triplet L1/L1L2/L1L2L3 sharing, LCP, collision, role relations; chain/fork/join; edge/no-edge; per-item purity, group size, L2 split, L3 refinement, rank, quant error; #265 vs #269 lost prefixes/merged items; degree/popularity strata | `SID_LINK.md`, `assets/sid/*.npz` | PENDING |
| C5 | SID -> recommendation with saved predictions/labels; paired user-context/target; HIT@5/10, NDCG, rank; target/last-history/train-edge/triplet linkage; multiplicity weights; hit->miss and reverse; four quadrants; confounders; test predictions explanatory only | `PREDICTION_LINK.md`, `assets/pairs/*.npz` | PENDING |
| C6 | Cross-run conditional relation: Spearman/Kendall/Pearson, per-item/transition deltas, run count, independent SIDs, CI, effect size, counterexamples; centre/user cluster bootstrap; FDR; pre-defined indicators incl. SID-preservation conditional; train/valid only for development; LOO by family; no weight search | `RELATION.md`, `REPORT.md` | PENDING |
| C7 | RUN_INVENTORY, CONTRACT, HYPOTHESES, logs, full triplet + item/transition assets + manifest, strata tables, correlations/CI, LOO, counterexamples, #265/#269 diagnosis, VERDICT, code/tests, infra zero drift, push | this directory + commits | PENDING |
| C8 | Answer: which curvature gains are recommendation-irrelevant; which structures associate stably; is SID an observable intermediate; is there a conditional relation and its counterexamples; what to optimise next and how to test causality | `VERDICT.md` | PENDING |
| C9 | No training, no mechanism change, TIGER reused read-only | `PROGRESS.json`, `INFRA.md`, `INFRA_SNAPSHOT*.json` | PENDING |
| C10 | Complete reproducible analysis; honest verdict; positive correlation not required | `VERDICT.md` | PENDING |

## Mechanism/compute scope

- Research-only scripts under `results/research/issue270/`: inventory/cohort resolution, full-geometry
  evaluation on the fixed ruler, SID linkage, prediction pairing, statistics.
- All assets reused read-only: `assets/train/visible_undirected.npz`,
  `assets/v8_runtime_layout/{targets,target_offsets}.npy`, `assets/node_moments.npz`,
  `assets/index.npz`, `assets/shards/*`, `stage1_GeneEmbedding/output/sentence_t5.npy`,
  `results/research/behavior_graph/`, archived checkpoints/SIDs/predictions.
- No new public infra; no `tools/`, `assets/` writes; no Stage2/Stage3.

## Verification plan

- Every computed artefact is produced through `tools/iteration/execution.py::run` (command.json,
  log, exit code, duration) or the frozen MCP for archived data.
- Arm identity: SID/checkpoint sha256 recorded; immutable-run bindings marked `record_verified`,
  arm-directory legacy bindings marked as such, missing/mismatched SIDs excluded from geometry.
- Headline recomputation uses the existing `tools/iteration/replay.py` for Stage3 numbers; no new
  public analyser.
- Infra hash snapshot at start and before close (`INFRA_SNAPSHOT*.json`).

## 2026-11-27T05:5x — final run

- All 42 geometry arms rebuilt under one Σ ruler (39 first batch + 2 codebook arms repaired for the
  `rq.levels.*` state-dict naming); 42/42 `*.npz`.
- Distortion J: 42/42 arms on the 20k×256 sample; κ_t ruler for the 4 primary arms (#265/#267/#269/TIGER).
- `analyse.py` PASS (26.5 s, project interpreter); `RELATION.json` written; `write_report.py` →
  `REPORT.md` (270 lines); `VERDICT.md` written; `CONTRACT.md` §8 appended; `DIAGNOSIS.md` E1–E5 resolved.
- Headline: arm Spearman −0.068 (CI [−0.51,+0.35]); P2 difference CI [−0.85,+1.44]; #265→#269 A.1
  −0.249, L2 170→113, ΔR −0.00736; κ_t J 0.168 vs 0.288 reverses the sign; between/within family SD
  0.00356/0.00094.
