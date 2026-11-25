# Verdict: TIGER_role_conditioned_l2_rq

Status: **COMPLETED_NEGATIVE**

## Checks

| check | expected | actual | status |
|---|---|---|---|
| C1 | R05 preregistration intact, nothing dispatched, one registration | PASS | PASS |
| C2 | role mixture at level 2; levels 1 and 3 use the item gate; three switches wired | PASS | PASS |
| C3 | frozen training-only role table, no new graph/orbit/cache | PASS | PASS |
| C4 | public Infra byte-identical; only candidate mechanism/tests/evidence changed | PASS | PASS |
| C5 | dependency-aware revalidation of all affected items | PASS | PASS |
| C6 | 0.8 s/epoch ceiling, 0.4-0.5 s target recorded truthfully | PASS | PASS |
| C7 | preregistered predictions recorded unchanged and the failure line applied; unavailable metrics marked, never changed | PASS | PASS |
| C8 | one formal four-card 3000-epoch Stage2 + one frozen Stage3 150 epochs, collision < 0.4 | PASS | PASS |
| C9 | analysis/replay through existing tools; missing metrics marked NOT_AVAILABLE | PASS | PASS |
| C10 | wall clock via existing TIMELINE, comparison with #256 4h53m, target <= 2h | PASS | PASS |
| C11 | reports/VERDICT/R2 four-piece set via existing interfaces; negative result preserved; causal wording limited | IN_PROGRESS | IN_PROGRESS |
| C12 | before/after public hashes equal; pushed; local main == remote main | IN_PROGRESS | IN_PROGRESS |
| C13 | complete real Stage2/Stage3 + independent reduction + archive; pre-registered results truthfully reported | IN_PROGRESS | IN_PROGRESS |

## Limits

validation Recall@10 NOT_AVAILABLE (FAULT-1); Stage3 test exploratory; R05-vs-baseline is not single-factor causal
