# The layout mechanism's seam in the sealed tree

`stage2_RQ-VAE/TIGER_hyp_role_assigned_layout/model/layers.py`, class `RQLayer`, the level loop at
lines 291-295: `residual = x`, then per level `quant, quant_loss, unused, indices = vq_layer(residual,
infer_use_sk)` and `residual = residual - quant`.

The mechanism replaces the level-2 INPUT with the behaviour factor and stops the reconstruction chain
there:

```
for level, vq_layer in enumerate(self.vq_layers):
    if level == 2 and self.role_layout is not None:
        quant, quant_loss, unused, indices = vq_layer(self.role_layout(x), infer_use_sk)
        continue          # level 2 contributes no reconstruction: the chain stops at level 1
    quant, quant_loss, unused, indices = vq_layer(residual, infer_use_sk)
    residual = residual - quant
```

`role_layout` is the behaviour factor b(x) = normalise(mean_{y in succ(x)} phi(y) - phi(x)), supplied by
the trainer's pair batch (the same signal the role supervision already computes, the same path).
The level-2 codebook is then trained by its own commitment term on b(x).

NOT YET IMPLEMENTED: the codebook's level-2 quantisation loss must be added to the trainer's loss
assembly (the level-2 `quant_loss` is currently returned by the loop and would be dropped), and the
export path must emit the level-2 index without adding it to the reconstruction.
