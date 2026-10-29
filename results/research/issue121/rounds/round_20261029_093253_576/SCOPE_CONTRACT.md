A proposal is IN SCOPE only when all of this holds:
  1. BEHAVIOUR: it names one training-only behaviour signal (directed item transitions, user-interest branching, sequential co-consumption, popularity-adjusted graph neighbours, behaviour-conditioned assignment weights, ...) and says how it is computed.
  2. HYPERBOLIC GEOMETRY: it names an explicit curved operation (trainable or fixed negative curvature, Poincare or Lorentz distance, exp/log maps, geodesic scoring, curved residual, curved quantization, curved regularization) and where in training or quantization it acts.
  3. REAL COUPLING: one connected computational pathway uses both, so that removing either changes the core computation. A behaviour suffix on an unchanged Euclidean quantizer, a standalone hyperbolic loss beside an unrelated behaviour loss, a geometry name without geometric computation, or inherited unused hyperbolic code do NOT qualify.
  4. TESTABLE EFFECT: it states what should observably change (SID assignment, gradient or parameter update, prefix behaviour predictability, or Stage3 Recall) and names the behaviour-off, geometry-off/Euclidean and both-on comparisons. The three switches must be part of the mechanism, so that the coupling can be shown to be real, but the round RUNS no ablation arm: each candidate is trained once.
  5. MINIMAL SPECIFICATION: equations or pseudocode, inputs and outputs, numerical safeguards and the training cost. No proof and no paper section are required.
SID quality (ER, prefix counts, collisions) is NOT a scope criterion and NOT an admission rule.

EXPERIMENT POLICY (binding on every proposal, implementation and review):
  - ONE candidate, ONE training run, on the frozen budget (3000 Stage 2 epochs on four cards, the frozen Stage 3 protocol, no early stopping).
  - NO ablation experiments: the behaviour-off, geometry-off and both-on switches must EXIST in the code and must be correct, but no arm is trained with them switched off.
  - NO tuning experiments: no sweep over the coupling weight, the clip bound, the temperature, the usage weight or any other constant. A constant that enters the mechanism has to be DERIVED (a scale-relative bound, a measured quantity, a documented default), not chosen by trying values.
  - NO multi-seed experiments: a candidate is not repeated with other seeds, so a single run decides it and the seed noise is part of the uncertainty.
  - The behavioural diagnostics (coherence, adjacency, transfer predictability) are recorded, never used as an admission gate; Stage 3 test_recall@10 is the judgement.
  - A proposal that needs an ablation, a sweep or a seed repetition to make its case is out of programme: it has to make the case with one run.
