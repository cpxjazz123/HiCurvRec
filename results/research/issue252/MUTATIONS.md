# Issue252 mutation detection

Actual deliberate mathematical, structural and routing mutations are caught by designated assertions, not crashes/import failures/timeouts. Original full-objective and new stochastic-estimator paths are separate; no claim of their loss/gradient equivalence.

| Mutation | Category | Observed designated assertion | Raw evidence |
|---|---|---|---|
| squared-distance coefficient doubled | numeric | `Tensor-likes are not close! /  / Mismatched elements: 5 / 5 (100.0%) / Greatest absolute difference: 0.2500000000000002 at index (0, 0, 1) (up to 1e-07 allowed) / Greatest relative difference: 1.0000000000000009 at index (0, 0, 1) (up to 1e-07 allowed)` | MUTATIONS.json |
| balanced rule replaced by argmin | structural | `Lists differ: [0, 0] != [1, 0] /  / First differing element 0: / 0 / 1 /  / - [0, 0] / ?  ^ /  / + [1, 0] / ?  ^ /  : balanced hard IDs must be [1,0]` | MUTATIONS.json |
| cost detached before soft assignment | routing | `0.0 not greater than 0.0 : selection cost must receive nonzero gradient` | MUTATIONS.json |
| A.1 overall sign flipped | numeric | `-1.3333333333333335 != 1.3333333333333333 within 7 places (2.666666666666667 difference) : full-reference A.1 must be 4/3` | MUTATIONS.json |
| invalid mask ignored | structural | `252.25 != 4.5 within 7 places (247.75 difference) : invalid target cannot contribute to centre loss` | MUTATIONS.json |
| hard forward replaced by soft codewords | routing | `False is not true : hard forward must equal real selected codewords` | MUTATIONS.json |
| geometry weights normalized over items | structural | `Tensor-likes are not close! /  / Mismatched elements: 2 / 2 (100.0%) / Greatest absolute difference: 1.7676745111027943 at index (1,) (up to 1e-07 allowed) / Greatest relative difference: 1.7676745111027943 at index (1,) (up to 1e-07 allowed)` | MUTATIONS.json |
| subgraph attention mask ignored | structural | `2.6215669315439545 != 0.0 : masked subgraph must receive zero attention` | MUTATIONS.json |
| expert Query detached from attention | routing | `0.0 not greater than 0.0 : expert Query must receive gradient` | MUTATIONS.json |
| sampled curvature sign reversed | numeric | `-1.3333333333333335 != 1.3333333333333333 within 12 places (2.666666666666667 difference) : sampled A.1 with complete references must equal4/3` | ESTIMATOR_MUTATIONS.json |
| reference mean replaced by sum | structural | `2.0 != 1.3333333333333333 within 12 places (0.6666666666666667 difference) : sampled A.1 with complete references must equal4/3` | ESTIMATOR_MUTATIONS.json |
| pair coefficient doubled | numeric | `2.3333333333333335 != 1.3333333333333333 within 12 places (1.0000000000000002 difference) : sampled A.1 with complete references must equal4/3` | ESTIMATOR_MUTATIONS.json |
| legal centre reference silently masked | structural | `4.0 != 1.3333333333333333 within 12 places (2.666666666666667 difference) : sampled A.1 with complete references must equal4/3` | ESTIMATOR_MUTATIONS.json |
| hard-q structural gradient detached | routing | `reference and endpoint gradients must equal the closed-moment derivative` | ESTIMATOR_MUTATIONS.json |
| centre schedule permanently repeats one centre | structural | `Items in the second set but not the first: / 1 / 5 : one complete centre cycle must cover every frozen valid centre` | ESTIMATOR_MUTATIONS.json |

The 15 observed mutants all triggered the specified assertion. Final-source suite receipts, command/output hashes and admission binding must be regenerated after remaining integration edits; these raw intermediate results alone do not authorize formal training.
