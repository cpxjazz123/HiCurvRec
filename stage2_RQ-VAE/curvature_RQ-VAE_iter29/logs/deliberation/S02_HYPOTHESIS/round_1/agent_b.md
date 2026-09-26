ROLE=AGENT_B
INDEPENDENCE_DECLARATION=I did not read the other candidate before completing this artifact.
SOURCE_PACKET=stage2_RQ-VAE/curvature_RQ-VAE_iter29/logs/deliberation/S02_HYPOTHESIS/round_1/source_packet.md
STAGE_ID=S02_HYPOTHESIS

# Independent FCCR-1 hypothesis candidate

## Research question and scope

Does replacing only iter26's fixed curvature mapping with a bounded additive-saturation map of the same behavior-branching and raw-residual-median inputs change quantization geometry enough to improve protocol-matched retrieval, with the user success criterion `test_recall@10 >= 0.065`? This is a hypothesis, not a promised result. Iter26 is the sole direct control; preserve every other iter26 condition and mechanism. No behavior-loss, optimizer, Sinkhorn, Stage1, Stage3, learnable/scheduled curvature, or auxiliary-loss change is proposed.

## Exact mapping and constants

For each layer `l`, use the following dimensionless saturating responses and equal-weight blend:

```text
u_l = B_l / (B_l + B_ref)
v_l = m_l / (m_l + m_ref)
q_l = (u_l + v_l) / 2
c_l = c_min + (c_max - c_min) * q_l

B_ref = 4.0
m_ref = 0.2
c_min = 0.05
c_max = 1.50
```

Equivalently,

```text
c_l = 0.05 + 1.45 * 0.5 * (B_l/(B_l+4.0) + m_l/(m_l+0.2))
```

This is a bounded rational/additive map, not iter26's ratio followed by layerwise log-standardization and an exponential. It directly uses both declared inputs; increasing either input at fixed other input increases curvature, with diminishing response. For finite `B_l >= 0` and finite `m_l > 0`, `0 < u_l < 1`, `0 < v_l < 1`, hence `0.05 < c_l < 1.50`. Thus values are valid and strictly inside the packet's supported `[0.05, 1.5]` bounds; no clipping/saturation plateau or per-layer normalization is needed for the stated domain.

## Exact provisional numeric substitution

Using exactly the source packet vectors, provisionally (not as S03-certified provenance):

```text
B = [19.324911558712664, 1.4605688962651735, 1.0148104414712726]
raw_residual_medians = [1.0, 0.10941, 0.09331]
u = [0.828509531968, 0.267475591648, 0.202362672192]
v = [0.833333333333, 0.353608480657, 0.318127578330]
q = [0.830921432651, 0.310542036153, 0.260245125261]
fixed_curvature = [1.254836077344, 0.500285952421, 0.427355431628]
```

The computed values are all finite, positive, and within the stated supported bounds. They are the proposed pre-training values, not outputs claimed from a modified source implementation.

## Deterministic fixed implementation behavior

Compute once before Stage2 from the preregistered input vectors and constants, using the stated expression in a fixed layer order. Use deterministic ordinary arithmetic; do not use random state, training step, optimizer state, batch/model activations, train/eval mode, or mutable state. Reject non-finite inputs, `B_l < 0`, or `m_l <= 0` as invalid input rather than silently substituting or clipping them. Verify finite outputs and the strict supported bounds, then store the resulting three values as a non-trainable fixed buffer; do not recompute or update it during training. The decimal outputs above are the preregistration reference (implementation comparison may allow only the explicitly chosen floating-point representation's rounding tolerance). No other mechanism or protocol setting changes.

## Mechanistic rationale and downstream prediction

Iter26's layerwise ratio/standardization couples each layer's result to the other layers and then exponentially rescales centered scores. This proposal instead gives each layer a transparent bounded response to the two measured layer inputs independently: larger branching and larger raw residual magnitude each contribute positively, with diminishing returns controlled by the fixed reference scales. The blend preserves both signals without allowing the large branching value alone to make an unbounded curvature; the residual term remains material rather than being discarded by a branching-only rule. The resulting precomputed geometry may alter curvature-dependent distances and therefore the relative quantization assignments/coarse-to-fine residual allocation. If that allocation better preserves item distinctions useful to the unchanged Stage3 recommender, recall may improve. This is an unverified causal prediction; Stage2 SID statistics are descriptive, and only a completed protocol-valid Stage3 comparison can assess the downstream outcome against iter26 (`R@10=0.057017009349048554`, `n_eval=57439`). The inclusive user threshold is `>= 0.065`; neither crossing it nor improvement over control is assured.

## Falsification and risk limits

- **Mapping/contract falsifier:** recomputation from the exact provisional vectors and constants does not match the listed `u`, `v`, `q`, and curvature values within the declared arithmetic tolerance; any output is non-finite/out of bounds; the values are trainable, change with training step/optimizer update/train-eval mode, or differ from the pre-run values.
- **One-factor falsifier:** any change besides this mapping (including altered inputs, their provenance/transform, optimizer, losses, Sinkhorn, Stage1/Stage3, or protocol settings) means the proposed contrast is not this hypothesis.
- **Mechanism-activation falsifier:** a matched-checkpoint/batch counterfactual against the declared iter26 curvature configuration shows no measurable change in the preregistered curvature-consuming geometry/assignment/loss signal. This would fail activation, not establish a negative retrieval effect.
- **Scientific falsifier:** if a contract-valid, provenance-valid, protocol-valid run produces no meaningful change in the intended quantization behavior and no Stage3 improvement relative to the direct control, the proposed rationale is unsupported for this mapping. A below-target score alone is promotion failure, not by itself proof that the mechanism is inactive or that all FCCR-1 mappings fail. Results near historical run noise should be described as near-parity/uncertain, not a discovery.
- **Risks:** the positive monotone response is an assumption, not established evidence; the chosen reference scales and equal weights are hypotheses and have no empirical calibration in this packet. The values may shift geometry in an unhelpful direction or become effectively weak/strong in actual downstream computation despite remaining mathematically unsaturated. A single run cannot estimate noise. The input provenance remains unresolved.

## Assumptions, caveat, and self-rejection conditions

Assume only for this candidate's arithmetic that the packet's recorded `B` values are the intended behavior-branching inputs and its listed residual values are positive raw residual medians, and that `[0.05, 1.5]` is the supported curvature interval. The packet explicitly says the residual provenance is unresolved: historical calibration logs report these medians and a distinct normalized-layer-scale vector, the cited baseline checkpoint is absent, and iter26's input artifact aggregates sources. These listed raw residual numbers are provisional; this candidate does **not** assert checkpoint-level reproduction, semantic/provenance certification, or S03 approval. S03 remains required and must independently decide whether these values genuinely have the declared raw-residual-median meaning before this map can propagate or be implemented.

I would reject this proposal before propagation if S03 cannot establish valid declared inputs, if the downstream curvature consumer does not support the stated bounds, if exact recomputation fails, or if a valid implementation requires changing the formula/constant/domain assumptions or another iter26 factor. Do not rescue it by retuning constants within this registered candidate; Judge C should reject it or adjudicate the other candidate. If all gates pass, the concrete next action is for Judge C to compare the independent candidates against the same packet and either select a complete candidate or reject both; this file is noncanonical pending that decision.
