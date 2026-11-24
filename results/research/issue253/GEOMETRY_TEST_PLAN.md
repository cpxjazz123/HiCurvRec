# Issue253 geometry / attention TDD plan

Scope: C2 attention, C3 three-geometry assignment/fixed metric, C11 three-expert public contracts, and the mathematical/routing/mutation portion of C12. GNN kernel equivalence, original sampler bytes, real A.1/reconstruction loss module gradients, integrated checkpoint/export, four-rank execution and public machine admission remain separately required; these CPU mathematical tests do not substitute for them.

## Ordering and evidence status

The independent `tools/issue253/reference.py` ambient Lorentz/sphere and scalar attention oracle existed before the new tests; it imports no production modules and was not modified by this slice. The candidate's existing `tests/reference.py` remains the differentiable independent ambient and probability-domain Sinkhorn oracle.

Parent observed the initial real RED: `tools/issue253/test_three_contract.py`, three tests, three designated assertion failures from five-versus-three distances/query shape, exit 1. Evidence: `results/research/issue253/THREE_EXPERT_RED_2.log`. Parent then authorized implementation in this same slice. No tests, builds, linters, formatters, GPU jobs or profiling were executed by this worker. Mutation kills and GREEN were NOT_RUN at initial worker handoff; subsequent parent execution is recorded below. Parent must bind final evidence hashes after all fixes.

Parent's first integrated execution is recorded in `INITIAL_NUMERICAL_GREEN.log`: all new three-expert geometry/attention tests, the nine mutation checks and compiler isolation checks passed; the old controlled intervention fixture failed with IDs `[0,0]` because it depended on removed curvature -3. The source correction below is not yet rerun. Unrelated ego-kernel failures in that log remain the kernel owner's responsibility.

## Test/reference changes

- `tests/test_three_geometry.py`: exact curvature identity and calls (no compute-five-then-slice); independent NumPy matrix/broadcast/row-pair agreement in float32/64; one common scale and joint scaling invariance; effective zero-curvature limit via `s² D_±1(x/s,y/s)²`; ambient directional finite differences for x/y/weights; fixed equal-three reference and absence of a trainable gate bypass; nonfinite norms/nonpositive or nonfinite common scale fail closed; independent scalar attention on all four two-token masks; exact mask zeros; expert permutation without hardcoded topology roles; shared K/V/score capacity; three-query strict state-dict roundtrip; exact behavior-off 1/3 for repeated IDs, tensor/list IDs and empty populations.
- Adapted `tests/test_attention.py` while preserving hand calculations, padding, permutation, single-token zero-Q-gradient, fixed null, empty batches, independent trainable queries, all attention parameter gradient groups, optimizer updates and directional derivatives. Hand answer now uses the actual two real tokens and three independent queries. Independence is rank/trainability, not an unnecessary mandated orthogonal initializer.
- Adapted `tests/test_reference.py`, `test_geometry_memory.py`, `test_known_answers.py`: exactly (-1,0,+1), three-element weights, correct Euclidean/sphere indices and equal 1/3. Preserve differentiable ambient gradient comparisons, coordinate-volume allocation rejection, hard/soft assignments, exact hard forward, scale, saturated radial distances, zero/coincident finite gradients, spherical wraps and exact/near antipodes. Existing radial known answers also exercise Euclidean because it agrees on these collinear tangents.
- Adapted `tests/test_attention_pipeline.py`: three-way null weights and actual Euclidean-versus-hyperbolic Query intervention on hard selected IDs.
- Corrected all three copies of the old ±3-specific intervention fixture to `r=(1,0)`, `e0=(0,1)`, `e1=(2.45,0)`. Hand-auditable distances: Euclidean `[2,2.1025]`, hyperbolic -1 `[acosh(cosh(1)^2)^2,2.1025]=[2.2903008838419554,2.1025]`. Thus the actual selected ID must change `0→1`, with non-near-tie margins .1025 and .1878008838419554. The direct and actual-Query pipeline tests additionally assert these independent ambient closed-form costs before hard decisions; no production arithmetic or tolerance was changed.
- Adapted `tests/test_mutations.py`: three-expert known answer, output under issue253 (never overwrite historical issue252 evidence), and mandatory unmutated checker success before each injection. Catch only `AssertionError`; import errors, exceptions, crashes and timeouts propagate and cannot count as kills.
- `tests/test_cuda_math.py`: regression coverage for issue253-owned default cache identity, all compiler/temporary directories under the repository-owned cache, unchanged fullgraph/dynamic/default compilation options, and compatible compile_geometry/compile_probabilities/compile_dense APIs. Compilation itself is mocked only in this configuration/API regression, never in numerical or gradient acceptance tests.

## Mutation → designated assertion

All nine mutations are executable in `test_mutations.Mutations.test_designated_assertions_detect_mutations`; evidence is written to `results/research/issue253/MUTATIONS.json` only after all assertions complete. Each checker must pass first without mutation. A surviving mutation fails the suite.

| Mutation | Class | Designated assertion |
|---|---|---|
| Double squared-distance coefficient | numerical | Origin-to-radius .5 gives exactly [.25,.25,.25] within declared tolerance |
| Balanced hard assignment replaced by argmin | reduction/rule | Tiny frozen example hard IDs are [1,0], not [0,0] |
| Detach selection cost before soft probabilities, retain zero graph edge | gradient routing | Actual soft-assignment probe has strictly nonzero derivative w.r.t. selection cost |
| Flip A.1 overall sign | numerical/sign | Three-point full-reference A.1 equals 4/3 |
| Ignore invalid target mask | mask/reduction | Valid-only centre Huber equals 4.5, invalid 999 never contributes |
| Replace hard forward by soft codeword mixture | hard/soft routing | `q` is bit-exact actual selected hard codeword |
| Expert weights normalize over item axis | reduction | Every item's expert weights sum to 1 |
| Ignore token attention mask | mask | Masked token receives exactly zero attention |
| Detach Query from attention, retain zero graph edge | gradient routing | Two unequal valid tokens produce strictly nonzero Query gradient |

These controlled operators preserve executable shapes and differentiable zero edges where appropriate. No crash-as-capture. Their assertions are in addition to independent numerical checks, not comparisons between two production paths.

## Production edits authorized after parent RED

- `model/geometry.py`: `CURVATURES=(-1,0,1)`; only one -1 hyperbolic calculation, existing Euclidean and one +1 sphere calculation; retain shared radii/directions, stable logarithmic hyperbolic and chord-atan2 sphere arithmetic and compiled entry contracts; three-wide curvature-off. The fixed metric is exactly their equal mean and never accepts gate inputs. Reject invalid host scale; CUDA scale checks and already-computed radial-norm checks use asynchronous assertions, never device-to-host synchronization. CPU invalid values raise ValueError. Curvature-off validates finite tangents without computing unused radial/directional experts.
- `model/attention.py`: three independently RNG-initialized trainable Queries, same shared K/V/score, token softmax then expert softmax, unchanged fixed zero null/masks.
- `model/structure_gate.py`: exact behavior-off `1/3` on all restored global-item routes, unchanged duplicate restoration and diagnostic shapes deriving from actual attention.
- Parent owns mechanism/trainer/diagnostic/checkpoint integration and has been notified to update `tests/test_mechanism.py`'s old uniform-five integration assertion.
- Parent additionally authorized new `tools/issue253/cuda_math.py`: same compiled APIs and frozen configuration as the historical helper, with all generated artifacts isolated under `results/research/issue253/compiler_cache`. Geometry now imports this owned helper rather than causing writes into issue252 historical evidence. GNN ownership was notified to migrate its compile_dense import; parent owns other caller migrations. No tools/issue252 source or historical evidence was edited.

## Exact commands for parent

Run from the repository root; CPU tests create no temporary files outside the repository. The interpreter and recipe are frozen. No command below dispatches GPU training.

```bash
PY=/home/wlia0047/ar57_scratch/wenyu/genrec_env_v2/bin/python3.9
TREE=stage2_RQ-VAE/TIGER_three_expert_attention_rq
export PYTHONPATH="$PWD:$PWD/$TREE:$PWD/$TREE/tests${PYTHONPATH:+:$PYTHONPATH}"

# Original RED replay on the pre-implementation source; on integrated source this is GREEN.
"$PY" -m unittest tools.issue253.test_three_contract -v

# Focused mathematical GREEN plus nine assertion-only mutation kills.
"$PY" -m unittest test_three_geometry test_attention test_attention_pipeline \
  test_reference test_geometry_memory test_known_answers test_mutations test_cuda_math -v

# Required integrated candidate suite, once all ownership slices have landed.
"$PY" -m unittest discover -s "$TREE/tests" -p 'test_*.py' -v
```

Ordinary smooth float64 geometry uses atol 1e-9 / rtol 1e-8; independent ambient directional finite differences use atol 1e-6 / rtol 1e-5; float32 uses atol 1e-5 / rtol 1e-4. Preserve existing stricter hand-answer tolerances where applicable. Boundary answers use explicit collinear/wrap formulas rather than numerically cancelling ambient acosh/acos references. Exact IDs/masks/shapes/hard q remain exact. The common-scale zero-curvature test is smooth and does not assert nonexistent differentiability at sphere antipodes.
