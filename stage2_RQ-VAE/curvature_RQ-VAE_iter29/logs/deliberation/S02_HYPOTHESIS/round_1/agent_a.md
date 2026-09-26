ROLE=AGENT_A
INDEPENDENCE_DECLARATION=I did not read the other candidate before completing this artifact.
SOURCE_PACKET=stage2_RQ-VAE/curvature_RQ-VAE_iter29/logs/deliberation/S02_HYPOTHESIS/round_1/source_packet.md
STAGE_ID=S02_HYPOTHESIS

# Candidate hypothesis — noncanonical pending Judge C

## Hypothesis and mechanism

For iter29, replace only iter26's map from the declared per-layer inputs `(B_l, m_l)` to curvature `c_l` with a bounded, separable-feature arithmetic map:

\[
x_l=\frac{B_l}{B_l+2},\qquad y_l=\frac{m_l}{m_l+0.1},\qquad
u_l=\frac{x_l+y_l}{2},\qquad
c_l=0.05+1.45\,u_l.
\]

Constants are fixed: branching half-scale `2`, residual half-scale `0.1`, equal feature weights `1/2`, lower curvature bound `0.05`, and range width `1.45` (thus upper bound `1.50`). This is a rational saturating map of each input followed by an affine blend; it has no cross-layer statistics, ratio of the two inputs, standardization, or exponential. It is therefore meaningfully different from iter26's `log1p(B_l)/log1p(m_l/m_min)` followed by across-layer z-scoring and exponentiation. Both declared inputs directly affect every layer's output.

### Exact provisional substitutions

Using exactly the packet vectors

`B=[19.324911558712664, 1.4605688962651735, 1.0148104414712726]`

`m=[1.0, 0.10941, 0.09331]`,

the calculations are:

| layer | `B/(B+2)` | `m/(m+0.1)` | `u=(x+y)/2` | `c=0.05+1.45u` |
|---|---:|---:|---:|---:|
| 0 | 0.9062129756321139 | 0.9090909090909091 | 0.9076519423615115 | 1.3660953164241916 |
| 1 | 0.42206034326942476 | 0.5224678859653312 | 0.4722641146173780 | 0.7347829661951981 |
| 2 | 0.3366083742817442 | 0.48269618747090165 | 0.40965228087632294 | 0.6439958072706683 |

Candidate vector: `[1.3660953164241916, 0.7347829661951981, 0.6439958072706683]`. All values are finite, positive, strictly inside the supported `[0.05, 1.50]` interval; none is clipped or at a saturation endpoint. The broad interior spread `[0.6439958072706683, 1.3660953164241916]` gives this map a nontrivial per-layer geometry change rather than a nearly uniform or bound-pinned vector.

## Mechanistic rationale and downstream hypothesis

The proposal treats branching and residual magnitude as two bounded, independently interpretable signals: increasing either increases curvature, with diminishing marginal influence as that signal exceeds its fixed reference scale. The scales place the largest layer's two normalized features near 0.91 and the other layers' features around 0.34–0.52, so neither feature is ignored and all layers remain inside the curvature range. The equal-weight blend avoids allowing a large raw magnitude on one input to dominate by units alone. Unlike the iter26 relative ratio/z-score map, a layer's value here is invariant to changes in the other layers' inputs.

The falsifiable downstream hypothesis is that this alternative allocation of stronger curvature to the layer with jointly largest branching and residual feature values, while retaining distinct intermediate curvatures for the other layers, changes quantization geometry/code assignments and may improve Stage3 retrieval versus the direct iter26 control. This is a mechanism hypothesis, not a predicted score: the inclusive user criterion is `test_recall@10 >= 0.065`, and neither reaching it nor improvement over iter26 is guaranteed. Only the locked, completed, protocol-valid Stage3 evaluation can determine the result; Stage2 SID metrics are descriptive, not gates.

## FCCR-1 implementation contract and falsifiers

- Compute this vector once before Stage2 training from the declared per-layer `behavior_branching` and `raw_residual_median` inputs. Store/use it as fixed, non-trainable curvature; it must be bitwise identical at every training step and unaffected by optimizer state, batch, epoch, or runtime behavior.
- For finite `B_l >= 0` and finite `m_l > 0`, the normalized features are in `[0,1)` and `c_l` is in `[0.05,1.50)`. Reject non-finite values, negative branching, nonpositive residuals, missing layers, or wrong vector lengths explicitly; do not silently substitute, clamp invalid inputs, or fall back to another mapping. Use deterministic scalar/tensor arithmetic with the constants above, fixed layer order, and a documented numeric dtype; no random operation or cross-layer reduction is involved.
- The mechanism is falsified before launch if recomputation from those inputs/constants does not reproduce the listed outputs within the implementation's declared floating-point tolerance; any output is non-finite/outside bounds; either input has no effect on its formula/output; or curvature changes after initialization/during training. It is falsified as the proposed single-factor treatment if any iter26 condition other than this mapping is changed.
- The scientific retrieval hypothesis is not supported if a protocol-valid iter29 Stage3 result fails to exceed iter26's exact direct-control `test_recall@10=0.057017009349048554`; the user success criterion separately fails if the result is below `0.065`. A below-control outcome rejects the claimed improvement but does not establish that all FCCR-1 maps fail. Any protocol/input identity failure makes the comparison uninterpretable rather than a mapping success or failure.

## Assumptions, evidence, risks, and self-rejection

Primary evidence: shared packet `source_packet.md` records the iter26 formula, constants, provisional input vectors, exact parent curvature and direct-control score; canonical S00 snapshot `logs/source_snapshot_iter29.md` establishes FCCR-1 and the inclusive target; canonical S01 lock `logs/protocol_manifest_iter29.md` sets iter26 as the sole direct control and requires all other conditions held fixed. These are the basis for this candidate, not a claim of independent raw-input reproduction.

Assumptions: the stated `B_l` and `m_l` correspond in layer order and have the semantics declared by FCCR-1; higher values of each can reasonably motivate greater curvature; and curvature affects quantization sufficiently for a retrieval response to be possible. The fixed scales `2` and `0.1` are proposal constants, not fitted or verified optima. This positive-monotonicity assumption may be wrong, and the bounded feature transform may compress useful magnitudes or overemphasize the largest layer. A single stochastic run cannot establish general causal improvement or run-to-run reliability.

**Raw-residual provenance remains unresolved.** The packet and S01 manifest report historical calibration medians `[1.0, 0.10941, 0.09331]`, note the referenced baseline checkpoint is absent and iter26's inputs aggregate sources, and explicitly reserve provenance adjudication to S03. The numeric substitutions above are provisional only: they do not establish checkpoint-level reproduction, raw-residual semantics, or S03 approval. Do not propagate/use this mapping canonically until Judge C approves S02 and S03 independently approves the specific raw-residual evidence.

Self-reject this candidate if a primary audit shows the listed residuals are not the declared raw per-layer medians, if branching/residual layer alignment or units invalidate the stated inputs, if the implementation cannot honor finite-domain rejection, bounds, determinism, and time invariance, or if review shows the scales/monotonic direction violate FCCR-1 or the approved one-mapping-only direction. These conditions block propagation; they must not be patched by changing constants after seeing results. Next action: Judge C adjudicates this candidate against Agent B and primary evidence; independently complete S03 provenance review before any canonical propagation or execution.
