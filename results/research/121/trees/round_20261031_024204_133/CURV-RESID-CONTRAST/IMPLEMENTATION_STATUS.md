# CURV-RESID-CONTRAST -- implementation status

Selected by the round `round_20261031_024204_133` (three scorers: two first
places, mean rank 1.33, IN_SCOPE).

## Done

* `model/curved_chain.py` -- the mechanism's mathematics: `exp_0`, `log_p`,
  `exp_p`, the Mobius addition, the specification's parallel transport, the
  conformal factor, and `CurvedChain` with the contrast objective
  `J = lambda(p)^-2 (alpha a^2 + b)` and the `argmin` over codewords, the prefix
  advancing through `exp_p`.
* `tests/test_curved_chain.py` -- 12 tests, all passing, written before the
  implementation: the exponential and logarithm are inverse, the points stay
  inside the ball, the transport is the identity at the origin and matches the
  specification's formula term by term, behaviour off reduces to the isotropic
  objective, geometry off reproduces the Euclidean chain token for token, the
  geometry changes the tokens when the codebook spans the radii, the gain changes
  the tokens, the prefix advances through the exponential map, the gradient
  reaches the codebooks and the encoder input, and the chain is deterministic.
* `prepare_behaviour.py` -- the behaviour constants, exactly the proposal's
  recipe (training split only): `S_i` capped at 32 successors, `N_i` 32 items
  drawn from the same popularity decile excluding i, its 1-hop neighbours and its
  successors, `u_i` the difference of the mean log maps, `u_hat_i` the unit
  direction and `alpha_i = clip(||u_i||_{x_i} / median, 0.5, 2.0)`.
  Built: **23,944 items with a direction, 643 isotropic fallbacks**, radial scale
  0.2484, median tangent norm 0.1666, gain range [0.5, 2.0].
* `model/model.py` -- the curved branch of `RQVAE` with the three switches
  (`curved`, `behaviour_on`, `geometry_on`), the reconstruction target being the
  same ball point the chain quantizes, and the commitment term on the curved
  prefix.

## Remaining

1. `train_rqvae.py`: carry the item ids through the data loader (the constants
   are indexed by item id), set `curved`/`behaviour_on`/`geometry_on` and
   `curved_radial_scale` in `_tokenizer_config`, and pass `direction`/`gain` to
   `forward` and to `get_indices` at export.
2. The one-epoch smoke on the real entry point.
3. `research_round_register_implementation`, the five verifiers,
   `research_round_accept`.
4. Stage 2 (four GPUs, 3000 epochs, global batch 4096, port 50202) and Stage 3
   (frozen protocol, port 50201), then the `test_recall@10` reading.

Nothing in this tree has been trained yet, and no recall value exists.
