# Hyperbolic geometry contract (poincare-contract/v1)

Status: **normative**. Every research agent reads this file, and every candidate
implements its mechanism *on top of* the operators defined here.

> This contract defines the geometry. A proposal may introduce a new hyperbolic
> mechanism, but it may not silently redefine the meaning of curvature, distance,
> exp/log maps, projection, or tangent norms. Any alternative convention must be
> proposed explicitly as a different mathematical mechanism and must provide a
> complete, internally consistent operator family (manifold + metric + exp + log +
> distance + projection + invariants); changing a single function is not a
> deviation, it is a defect.

## 1. Notation

* The Poincaré ball model of curvature $\kappa = -c$ with $c > 0$:
  $\mathbb B_c^d = \{x \in \mathbb R^d : c\lVert x\rVert^2 < 1\}$.
* The ball radius is $R_c = 1/\sqrt{c}$. **A clamp to a fixed `0.999` regardless
  of $c$ is forbidden**: it silently redefines the model.
* Conformal factor: $\lambda_x^c = 2 / (1 - c\lVert x\rVert^2)$.
* $c = 0$ is a first-class branch, not a tiny $c$: the model is $\mathbb R^d$ with
  the Euclidean metric. It is the $c \to 0$ LIMIT of the family below, so
  $d_0(x,y) = 2\lVert x-y\rVert$ (the factor 2 is the metric's normalisation and
  changes no argmin); $\exp_x^0(v) = x+v$ and $\log_x^0(y) = y-x$.

## 2. Canonical operators

$$
x \oplus_c y =
\frac{(1 + 2c\langle x,y\rangle + c\lVert y\rVert^2)x + (1 - c\lVert x\rVert^2)y}
     {1 + 2c\langle x,y\rangle + c^2\lVert x\rVert^2\lVert y\rVert^2}
$$

$$
d_c(x,y) = \frac{2}{\sqrt c}\operatorname{artanh}\!\left(\sqrt c\,\lVert (-x)\oplus_c y\rVert\right)
$$

$$
\exp_x^c(v) = x \oplus_c \left[\tanh\!\left(\frac{\sqrt c\,\lambda_x^c \lVert v\rVert}{2}\right)\frac{v}{\sqrt c\,\lVert v\rVert}\right]
\qquad
\log_x^c(y) = \frac{2}{\sqrt c\,\lambda_x^c}\operatorname{artanh}\!\left(\sqrt c\lVert(-x)\oplus_c y\rVert\right)\frac{(-x)\oplus_c y}{\lVert(-x)\oplus_c y\rVert}
$$

At the origin these reduce to

$$
\exp_0^c(v) = \tanh(\sqrt c\lVert v\rVert)\frac{v}{\sqrt c\lVert v\rVert},
\qquad
\log_0^c(y) = \operatorname{artanh}(\sqrt c\lVert y\rVert)\frac{y}{\sqrt c\lVert y\rVert}.
$$

The Fréchet mean of a set $\{v_i\}$ with weights $w_i$ is the point $\mu$ that
minimises $\sum_i w_i d_c(\mu, v_i)^2$; a solver must state its initialisation,
iteration and tolerance, and a coordinate average may not be called a Fréchet
mean.

## 3. Tangent convention (the origin factor 2)

A tangent vector's **Euclidean coordinate norm is not its Riemannian norm**:
$\lVert v\rVert_{0,c} = \lambda_0^c \lVert v\rVert = 2\lVert v\rVert$. Hence
$d_c(0, \exp_0^c(v)) = 2\lVert v\rVert$ and

$$
\log_0^c(\exp_0^c(v)) = v
$$

both hold. **`d(0, exp(v)) != ||v||` is not a defect**: reporting it as one is a
false positive under this contract.

## 4. Numerical contract

* Projection uses the true radius: $r_{\max} = (1-\epsilon)/\sqrt c$. Only a point
  with $\lVert x\rVert > r_{\max}$ is rescaled; a point inside keeps its value.
* The `artanh` argument is $\sqrt c\,\lVert(-x)\oplus_c y\rVert$, clipped to
  $1-\epsilon$ — never the unscaled norm.
* Zero-norm limits are taken explicitly through $\operatorname{tanhc}(x)=\tanh(x)/x$
  and $\operatorname{atanhc}(x)=\operatorname{artanh}(x)/x$, both equal to 1 at
  $x=0$; no `+1e-12` guard may stand in for the limit, because it changes the
  gradient at the origin.
* Default precision: float64 for scalar reductions (distances, barycentre steps),
  float32 for tensors that feed a network.

## 5. Required identities

For every $c \in \{0.01, 0.1, 0.25, 1, 4, 10\}$ and every point of the ball:

1. $0 \oplus_c x = x$ and $x \oplus_c 0 = x$
2. $\log_x^c(\exp_x^c(v)) = v$ and $\exp_x^c(\log_x^c(y)) = y$
3. $d_c(x,y) = d_c(y,x)$ and $d_c(x,x) = 0$
4. $d_c(0, \exp_0^c(v)) = 2\lVert v\rVert$
5. every returned point satisfies $c\lVert x\rVert^2 < 1$
6. $c \to 0$ recovers the Euclidean branch, including its factor 2

## 6. Forbidden implementations

* a fixed unit-ball clamp when $c \neq 1$
* mixing $c$ and $\sqrt c$ in a map and its distance
* unexplained factors of two, or an $\exp$ paired with a different $\log$
* a candidate-local reimplementation of `exp`/`log`/`distance`/`project` without
  a declared, complete `GEOMETRY_CONTRACT_DEVIATION`

## 7. Implementation and enforcement

* The only implementation is `tools/research/geometry/poincare_contract.py`.
  Candidates receive a copy of that module, not a hand-written variant.
* Every agent reads this file; agents that inspect implementations also read the
  module. The execution receipt records `geometry_contract.sha256`, and a change
  to this file invalidates cached mathematical reviews.
* `tools/research/tests/test_geometry_contract.py` runs the operator family over
  the curvatures and points above. A candidate that fails it cannot be reviewed.
