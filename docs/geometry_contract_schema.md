# The geometry contract schema (issue #110)

What a hyperbolic mechanism has to declare, and what the independent verifier checks
against it. The declaration is **caller-supplied data**: a missing or partial contract is a
blocking condition, and a generic template is never substituted for one.

## The seven contract groups

| Group | What it declares | The group it is checked against |
|---|---|---|
| `mechanism` | the id, the type, the entry points, the activation | H1, H2 |
| `model` | the adapter and the parameter groups the gradient must reach | H4 |
| `forward_contract` | the observables, the invariants, the declared effect | H1, H7 |
| `gradient_contract` | the mode, the target, the rule, the expected paths | H2, H4 |
| `update_contract` | the optimizer, the learning rate, the clipping | H4 |
| `distributed_contract` | the world size, the aggregation, the rank identity | G5 |
| `runtime_contract` | the entry point, the isolation, what is observed | G6 |

The CGV contract and the hyperbolic declaration are separate objects: the first is verified
by the existing CGV (`tools/cgv/contract.py`, `tools/cgv/run_cgv.py`), the second by the
hyperbolic gates (`tools/hyperbolic_verifier/gates.py`).

## The hyperbolic declaration

| Field | Meaning |
|---|---|
| `hyperbolic_model` | `poincare_ball` or `lorentz_hyperboloid` |
| `curvature` | written as `-c` with `c > 0` |
| `basepoints` | the tangent-space base points, or where the source names them |
| `hyperbolic_operator` | the genuinely hyperbolic operator the loss uses, and why the mechanism changes without it |
| `literature_ids` | the verified sources, or an empty list for an original inference |

## The gates

| Gate | Verifiable condition | How it is checked |
|---|---|---|
| H1 | the manifold, the curvature, the point constraint, the base points, the domains | the declared fields and the source |
| H2 | a genuinely hyperbolic operator is in the actual loss, and the mechanism needs it | a **trace** of the candidate's source against the operator names; a Euclidean mechanism with a hyperbolic title is refused here, before any GPU time |
| H3 | the geometry changes with the curvature on fixed points, and not only by a scale | two curvatures, the same points, the change compared with the scale |
| H4 | finite, correct gradients reach exactly the declared parameters | the caller's gradient evidence |
| H5 | the coincident, zero and near-boundary cases are finite and inside the domain | the verifier's own geometry |
| H6 | the operator is enabled in a run and the exported hard SID level changes | the caller's activation evidence |
| H7 | the claimed identities hold against an independent oracle | the verifier's own geometry, which never imports the candidate |

H4 and H6 need evidence the caller supplies by running the mechanism. **Without it their
state is `insufficient_evidence`, which is not a pass.** A gate that is declared
not-applicable never passes either.

## The oracle

`tools/hyperbolic_verifier/geometry.py` holds the verifier's own Poincaré ball and Lorentz
conversion: the distance, the exponential and logarithm maps at the origin, the Busemann
function, the conformal factor, the Riemannian gradient, the retraction and the domain
safeguard. It is written from the definitions and **never imports a mechanism's code** — an
oracle that did would only confirm that the mechanism agrees with itself.

Its own checks are identities that do not come from its formulas: the distance from the
origin is twice the artanh of the norm; the distance from the origin to the exponential map's
image is the norm of the tangent vector; the maps are inverses to machine precision; the
Busemann function diverges at its base point and is finite elsewhere; and the Riemannian
gradient differs from the Euclidean one by exactly the conformal factor. An operator whose
convention could not be verified is deliberately **absent** from the oracle.
