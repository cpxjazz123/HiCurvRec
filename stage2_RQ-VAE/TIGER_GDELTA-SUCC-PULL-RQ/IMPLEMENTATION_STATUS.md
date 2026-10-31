# GDELTA-SUCC-PULL-RQ -- round seven, implementation state

Selected by `round_20261031_163630_776` (rank 3 of 6). Implemented, registered
against tree digest `f45c4ae4e2387184`, **not trained**: the verification pass
returned `AGENT_INFRA_ERROR` twice with no verdict on disk, so the round gate
refuses both acceptance and the Stage-2 job. No recall value exists for round 7.

## Implemented

* `prepare_behaviour.py` -- train-only `n_ij`, `d_beh`, `N_i` (4 successors),
  `w_ij`, `S_i` with the shortest-path closure, the local Gromov delta over
  quadruples, `delta_ref = 0.3997`, `K_i = clip(delta_ref/(delta_ref+delta_i),
  0.1, 2)` in (0.140, 1.0), and the successor-pull tables `g0/g1/g2` built from
  the frozen baseline code vectors. Runs in 20 s over 24,556 items.
* `model/layers.py` -- `_poincare_k`, `set_behaviour`, `gdelta_cost`
  (`d_K(x,e)/s1 + d_K(e,g_i)/s2` with detached batch medians) and the early
  branch in `VQLayer.forward`; `BEHAVIOUR_OFF` and `GEOMETRY_OFF` switches exist.
* `train_rqvae.py` -- builds the constants at startup, maps skus to embedding
  rows through the frozen baseline SID order, binds table and curvature per level.

## Smoke result (honest)

A 64-row forward on the untrained model succeeds: tokens `(64, 3)`,
loss 0.7657, deterministic across two calls. But **neither switch changed the
assignment** on that synthetic case (`behaviour-off` and `geometry-off` both
produced the same tokens as both-on). With random weights every codeword is
nearly equidistant, so equal argmins are expected rather than diagnostic -- but
this means the switch wiring has **not** been demonstrated, and a test on a case
where the scores differ is still owed before the mechanism is claimed correct.

## Blocker

`research_round_verify` -> `AGENT_INFRA_ERROR` (two passes, no verdict files);
`research_round_accept` refuses ("only a verified candidate can be accepted");
`research_round_launch` queues nothing. Per the policy this is an execution
failure to retry, never a mechanism defect -- but until the verifier service
returns a verdict, round seven cannot reach Stage 2 through the framework.
