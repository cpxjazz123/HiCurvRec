# EVIDENCE

- `spec.json:2-5` describes `hyp_role_assigned_layout` as a role-assigned token layout: levels 0 and 1 are the host residual chain; level 2 quantises the behaviour factor. It does **not** specify a hyperbolic distance/operator in the mechanism summary.
- `train_rqvae.py:152-155` imports the hyperbolic arm table and assignment installers. `_HYP_CONFIG` is selected by exact arm-name lookup at `train_rqvae.py:275`.
- The only distance-based assignment installation in the trainer targets **level 0**: the ordinary hyperbolic assignment installer at `train_rqvae.py:1368-1372`, or the product assignment installer at `1373-1379`, each conditional on configuration flags. There is no corresponding installation on `vq_layers[1]`.
- `layers.py:63-66` initializes every layer's `hyperbolic_distance_fn` to `None`. `layers.py:102-113` uses that callback only when non-`None`; otherwise `_distances` computes squared Euclidean distances. Assignment then takes the minimum at `layers.py:115-122`, and `VQLayer.forward` calls this path at `layers.py:124-130`.
- The role assignment is explicitly set to level 2 (`train_rqvae.py:1173-1188`). `layers.py:266-269` documents the role-level behavior, and `layers.py:306-319` routes role inputs only to that level while levels 0 and 1 quantise/subtract the ordinary residual chain.
- The RQ-VAE forward computes encoder output, invokes the quantizer stack, then decodes its quantized output (`model.py:26-31`). In training, `train_rqvae.py:1553-1556` calls that forward with role inputs. The role signal is derived from successor pairs in ordinary latent space (`train_rqvae.py:1191-1228`).
- The hyperbolic loss branch is separate/additive: `train_rqvae.py:1796-1813` computes a mechanism loss only if `_HYP_CONFIG` is non-`None` and the behavior weight is positive. It does not itself install a per-layer callback; assignment installation is the separate conditional at lines 1368-1379.

# VERDICT

**VESTIGIAL** — based on the staged implementation evidence, the candidate's operative levels 0 and 1 are the host's residual quantizers, and their normal assignment is squared Euclidean unless a callback is installed. The staged trainer contains no level-1 hyperbolic assignment installation; its optional assignment wiring is restricted to level 0. The role-assigned level is level 2, not a hyperbolic operator on either residual level. Thus the declared/claimed geometry does not establish a nontrivial hyperbolic operator on levels 0 and 1 in the trained forward pass.

# RISK

`layers.py` supports a genuine hyperbolic assignment callback, and `train_rqvae.py` conditionally wires one for level 0 (including product-geodesic assignment). But the arm configuration table is imported from `scripts.hyp_losses` and is not among the staged files, so this three-file review cannot prove whether this exact arm name resolves to a configuration that activates the level-0 product branch, nor whether the external installer assigns the callback exactly as its name suggests. It also cannot establish the runtime callback value from a checkpoint/run. This uncertainty does not supply evidence for level 1: no level-1 installer appears in the trainer. To close the runtime/configuration gap, inspect the resolved `_HYP_CONFIG` and `vq_layers[0/1].hyperbolic_distance_fn` in the actual training process.
