# The arm's own §6 gradient-pathway check (the product-geodesic arm)

`scripts/grad_check.py` in the sealed tree, on the real frozen data (24556 train items, 339519
transition pairs), run under the project's torch environment. It is the arm's own check, not the
CGV's: it exercises the TRAINING path, so it is an independent confirmation that the mechanism is
wired into the graph the run will actually use.

**status: PASS, failures: []**

| observation | value |
|:---|:---|
| device | cuda:0 |
| residual trace shape | [3, 1024, 32] (three levels) |
| the contrastive term's gradient | nonzero on 8 parameters, and it touches the encoder |
| the first active epoch / the ramp weight there | 1501 / 1.33e-04 |
| the total loss's grad_fn | AddBackward0 |
| the total gradient's nonzero parameters | 19 |

The three verifications of this candidate are now independent of each other: the smoke test (the
patched assignment equals a direct computation of the declared formula, exactly), the CGV's G1-G6
on the identical frozen tree, and this training-path check.
