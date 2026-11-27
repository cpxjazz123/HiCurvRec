# Mutation evidence (#269 candidate)

Production module: `model/stereographic.py`. The reference is an independent
mathematical path: the arccosh identity for the ball and the ambient
dot-product/arccos formula for the sphere, not the asinh/asin branch under test.

| mutation | production text changed | designated check | result |
|---|---|---|---|
| `scale_4_to_2` | `return 4.0 * values` -> `return 2.0 * values` | `check_zero_exact` (k=0 must equal `4*||x-y||^2`) | caught |
| `negative_sign_flip` | `denominator=(1+k*x2)*(1+k*y2)` -> `(1-k*x2)*(1-k*y2)` | `check_reference` (arccosh/arccos reference, `rtol=atol=1e-8`) | caught |
| `sphere_asinh` | `sphere = torch.asin(` -> `torch.asinh(` | `check_reference` | caught |
| `unsigned_curvature` | `magnitude = k.abs()...` -> `magnitude = k.clamp_min(...)` | `check_reference` | caught |
| `single_branch` | the three-branch `torch.where` -> `values = sphere` | `check_reference` | caught |

A crash is not a catch: each mutated module must load and the named check must
return False (the test asserts `caught is False`). Executed by
`tests/test_mutations.py` as part of the 121-test candidate suite.
