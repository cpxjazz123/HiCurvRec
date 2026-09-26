STAGE_ID=S08_MVG
ROUND=2
VERDICT=ABORT_ITERATION

HARD_GATE_A=PASS
HARD_GATE_B=PASS

EVIDENCE_FOR_A=Registered `trust_radius_fraction = 0.5` produces clip_frac = 0 across 200 MVG steps AND 1000 follow-up diagnostic steps; activation requires tightening the constant to ≤ 0.05 (10× tighter).
EVIDENCE_FOR_B=Independently confirmed the same scaling-sweep table; the cumulative 200-step ON/OFF effect (loss_delta = 0.018, codebook max-abs-diff ~0.1 per layer) confirms the mechanism runs but never clips.
PROBLEMS_A=None
PROBLEMS_B=None

WHY_NOT_A=Both drafts independently reach the same ABORT_ITERATION conclusion.
WHY_NOT_B=See above.

CANONICAL_DECISION=ABORT_ITERATION.  Iter27 ends with STATUS=ITERATION_ABORTED_INFEASIBLE.  Canonical artifact is logs/iteration_abort_iter27.md (to be written).  The Stage2 full run is NOT launched; no Stage3 run is launched; no `rqvae_best.pth` is produced for iter27.
CANONICAL_ARTIFACT=iteration_abort_iter27.md
CONFIDENCE=HIGH

ABORT_REASON=User-fixed τ_l = D_l / 2 produces clip_frac = 0 across all tested regimes (200-step MVG + 1000-step probe); making the mechanism active requires changing the registered constant, which violates SKILL.md §2.9 "registered-spec immutability after hypothesis lock".
ABORT_EVIDENCE=mvg_check_iter27.log (5 ON steps clip_frac = 0); mvg_check_iter27.md Layer D (1000-step probe: trust_frac=0.5 → clip_frac=[0.000, 0.000, 0.000], trust_frac=0.05 → clip_frac=[0.000, 0.008, 0.065], trust_frac=0.02 → [0.072, 0.445, 0.914]); 200-step ON/OFF loss_delta = 0.018 with codebook max-abs-diff ~0.1 per layer (mechanism runs but never clips).
NEXT_ITERATION_CONSTRAINTS=Parent remains iter18.  A future iteration (e.g. iter28) MAY register a tighter `trust_radius_fraction ∈ {0.05, 0.02, 0.01}` under its own S00–S08 deliberation cycle.  Do NOT retune inside iter27.  Do NOT inherit the abortive iter27 mechanism as a parent.  Do NOT silently launch Stage2 with a different constant.

MERGE_COMPONENTS_A=agent_a.md (recommend ABORT_ITERATION)
MERGE_COMPONENTS_B=agent_b.md (independently recommends ABORT_ITERATION)

REPLAN_CONSTRAINTS=