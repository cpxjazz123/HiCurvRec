# Iter29 FCCR-1 Hypothesis — S02 Canonical

## A. Research question

Does replacing only iter26’s per-layer mapping from the same behavior-branching and provisionally recorded raw-residual-median inputs to fixed curvature with the bounded rational/additive map below change quantization geometry and improve protocol-matched `test_recall@10` versus iter26, while meeting the user target `test_recall@10 >= 0.065`?

## B. Exact mapping and constants

For each layer `l`, define:

```text
x_l = B_l / (B_l + B_ref)
y_l = m_l_raw / (m_l_raw + m_ref)
u_l = (x_l + y_l) / 2
c_l = c_min + (c_max - c_min) * u_l

B_ref = 2.0
m_ref = 0.1
c_min = 0.05
c_max = 1.50
```

Equivalently, `c_l = 0.05 + 1.45 * 0.5 * (B_l/(B_l+2.0) + m_l_raw/(m_l_raw+0.1))`. This is the sole registered mapping change from iter26; it uses both same-layer inputs, separately and monotonically, with diminishing response and equal weight. For finite `B_l >= 0` and finite `m_l_raw > 0`, both features are in `[0,1)`, so `c_l` is strictly inside the supported `[0.05, 1.50]` interval. No cross-layer normalization, clipping, learned value, or schedule is part of this mapping.

## C. Actual numeric substitution (provisional pending S03)

The iter26 primary records and the S02 packet list the following layer-ordered values. The branching vector is the recorded `behavior_branching`; the residual vector is **`PROVISIONAL_PENDING_S03`** and is not asserted to have certified raw-residual provenance:

```text
B = [19.324911558712664, 1.4605688962651735, 1.0148104414712726]
raw_residual_median = [1.0, 0.10941, 0.09331]  # PROVISIONAL_PENDING_S03
```

Substituting into the registered equation gives:

| layer | `x_l = B_l/(B_l+2.0)` | `y_l = m_l_raw/(m_l_raw+0.1)` | `u_l=(x_l+y_l)/2` | fixed `c_l=0.05+1.45*u_l` |
|---:|---:|---:|---:|---:|
| 0 | 0.9062129756321139 | 0.9090909090909091 | 0.9076519423615115 | 1.3660953164241916 |
| 1 | 0.42206034326942476 | 0.5224678859653312 | 0.4722641146173780 | 0.7347829661951981 |
| 2 | 0.3366083742817442 | 0.48269618747090165 | 0.40965228087632294 | 0.6439958072706683 |

```text
intermediate_terms = {
  x: [0.9062129756321139, 0.42206034326942476, 0.3366083742817442],
  y: [0.9090909090909091, 0.5224678859653312, 0.48269618747090165],
  u: [0.9076519423615115, 0.4722641146173780, 0.40965228087632294]
}
fixed_curvature = [1.3660953164241916, 0.7347829661951981, 0.6439958072706683]
```

These are independently recomputed equation outputs from the recorded decimal vectors, not checkpoint-level reproduction or provenance approval. S03 must audit the residual semantics and provenance before these values can be approved for propagation/execution; failure there blocks the next stage.

## D. Direct fixed-curvature effects

Compute the three values once before Stage2 from the registered input vectors and constants, in fixed layer order, and store/use them as non-trainable fixed curvature. They must equal the preregistered vector above, be finite and within bounds, and remain identical across training steps, optimizer updates, and train/eval mode. They must not depend on step, optimizer state, model/batch activations, or mutable state; do not learn, schedule, normalize across layers, or silently substitute/clamp invalid inputs. The formula’s direct effects are its deterministic per-layer outputs and their use by the unchanged curvature-consuming quantization path. A matched-checkpoint/batch counterfactual against iter26’s fixed-curvature configuration must show a measurable difference in the preregistered geometry/assignment/loss signal to establish mechanism activation.

## E. Downstream rationale and one-factor lock

Recorded input vectors have a large first-layer branching value (`19.3249`) and substantially smaller second/third-layer values (`1.4606`, `1.0148`); their provisionally recorded residual values are likewise larger at layer 0 (`1.0`) than at layers 1–2 (`0.10941`, `0.09331`). With `B_ref=2.0` and `m_ref=0.1`, both transformed features remain active in every layer and the resulting values are interior rather than bound-pinned. **Inference, not established evidence:** these per-layer curvature differences could alter curvature-dependent distances and quantization assignment/residual allocation, which could in turn affect the unchanged Stage3 recommender’s retrieval. The reference scales are not empirically calibrated optima; their proximity to the lower-layer input magnitudes is a defensible operating-scale rationale, not proof of superiority over another scale.

Iter26 is the sole direct control (`test_recall@10=0.057017009349048554`, `n_eval=57439`). All other iter26 conditions remain locked: parent and protocol, inputs and their identities/use, seed policy, warm-start, Stage2 steps/layers/codebook, unchanged losses/optimizer/Sinkhorn, Stage1, Stage3 code/settings, and evaluation protocol. The only conceptual delta is this mapping; do not stack another mechanism. Iter18 remains `HISTORICAL_NONCOMPARABLE`, not a direct comparator. Stage2 SID metrics are descriptive only.

## F. Falsification and promotion criteria

- **Mapping/contract falsifiers:** equation recomputation from the registered vectors and constants fails to reproduce the listed intermediates and fixed-curvature values within the implementation’s declared floating-point tolerance; an output is non-finite/out of supported bounds; either input does not affect its formula output; or curvature is trainable, changes with step/optimizer update/train-eval mode, or differs from its preregistered vector. Any such result invalidates implementation/contract compliance, not the scientific performance hypothesis.
- **One-factor/provenance falsifiers:** changing any iter26 condition other than the mapping confounds this test. S03 failure to establish the declared raw-residual provenance blocks propagation and makes the proposed raw-residual mapping unverified; the recorded residual vector remains `PROVISIONAL_PENDING_S03` until that audit. Do not claim checkpoint-level reproduction or S03 approval here.
- **Activation falsifier:** no measurable change in the preregistered curvature-consuming geometry/assignment/loss signal in the matched counterfactual means this registered mechanism was not shown active; it is not by itself a retrieval result or evidence against all FCCR-1 mappings.
- **Scientific and promotion falsifiers:** after a contract-, provenance-, and protocol-valid Stage3 comparison, a score no higher than iter26’s exact direct-control `test_recall@10=0.057017009349048554` fails the hypothesized retrieval improvement for this mapping. Separately, `test_recall@10 < 0.065` fails the user’s inclusive promotion target; a score at or above `0.065` meets that target, but improvement over iter26 and causal interpretation must still be reported separately. A single stochastic run does not estimate run-to-run noise; near-noise differences are not a discovery, and a below-target result alone does not establish mechanism inactivity or invalidate the FCCR-1 family.