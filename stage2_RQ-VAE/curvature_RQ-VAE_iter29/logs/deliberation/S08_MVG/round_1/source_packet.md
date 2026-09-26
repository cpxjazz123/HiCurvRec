# S08_MVG round 1 source packet

ROLE=AGENT_CANDIDATE
ROUND=1
STAGE_ID=S08_MVG

## Canonical authority

Use only Judge-approved S00–S07 artifacts, root `CLAUDE.md`, `skill://curvature-rqvae-iter`, and primary current source. S07 round-2 Judge accepts the exact registered-input lock. Its official preflight log records `MECHANISM_CONTRACT_PASS`; `logs/registered_input_smoke_iter29.log` records canonical formula success and rejection of a valid compensated noncanonical input pair. Do not use rejected S07 candidate drafts as instructions.

S01 locks the single warm-start checkpoint at `/home/wlia0047/ar57/wenyu/GeneRec/results/stage2_RQ-VAE/curvature_RQ-VAE_iter8/out/rqvae/instruments/rqvae_best.pth`, SHA-256 `189e0affb156b9cdbd11f38a6fdb1127804e3f3095edd23c55f84825beb56c3b`. A direct pre-S08 hash check observed that exact digest. The digest must be rechecked immediately before the one authorized MVG run and recorded; `scripts/mvg_check.py::_load_checkpoint()` itself checks existence/shape but not hash.

## S08 target

Independently audit whether `scripts/mvg_check.py` correctly designs one lightweight, pre-Stage2 verification and whether its acceptance criteria distinguish fixed-curvature contract validity from scientific performance. No Stage2 or Stage3 run is authorized here. The helper must execute at most once, only after Judge C authorizes the canonical method.

Inspect the helper and called source, including:
- locked iter8 checkpoint path and transfer semantics (`scripts/mvg_check.py:11-23,53-93`);
- one real deterministic batch from the registered Stage1 embedding/Stage0 training transition data (`:35-50`);
- candidate versus iter26 control fixed curvature with shared non-curvature checkpoint state (`:182-224,266-303`);
- fixed buffers, nontrainable/non-optimizer curvature, train/eval plus 0/25k/50k/100k invariance (`:96-148,227-263`);
- finite differentiable total loss and explicit `loss.backward()` with finite nonzero model gradients (`:151-179`);
- measurable counterfactual activation without an invented threshold or SID quality gate (`:195-224`);
- no modification of the canonical training or evaluation protocol.

The Stage2 root rule requires pre-training gradient-path verification on one checkpoint and one batch. Under FCCR-1 the fixed curvature is intentionally nontrainable; require gradients on trainable model state, not on curvature. Stage2 descriptive SID metrics are not gates. The iter29 contract is fixed FCCR-1 with curvature `[1.3660953164241916, 0.7347829661951981, 0.6439958072706683]`; control vector is the registered iter26 baseline in the helper.

## Candidate boundaries and run protocol

A and B independently review the same source and propose/assess the method, expected outputs, and any blocking concern. They must not execute MVG or run tests/build/formatter, preflight, deliberation gate, Stage2, or Stage3. They write only `agent_a.md` or `agent_b.md` under this round directory with exact role/independence/source-packet/stage headers. Judge C decides the canonical method. After acceptance, the orchestrator rechecks the S01 checkpoint digest and launches `scripts/mvg_check.py` once with the approved Python 3.9 interpreter and no arguments, then records actual output as `logs/mvg_check_iter29.log`. No duplicate GPU execution.

## Acceptance rubric

A valid `MVG PASS` must establish all of the following with observed output: exact fixed formula values loaded from source; one locked checkpoint and one real batch; train/eval and multistep curvature invariance; fixed curvature as buffers, not parameters/optimizer entries; finite loss and nonzero finite gradients on trainable model parameters after explicit backward; nonzero model updates with curvature unchanged; and a measurable candidate/control quantizer effect. A PASS validates implementation/activation only, not expected Stage3 success. Any mismatch or no-effect result blocks Stage2; do not retune the registered mechanism inside iter29.
