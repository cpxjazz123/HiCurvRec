# QUESTION (C): Does `no_grad` cut the declared encoder gradient?

## EVIDENCE

- `spec.json` sets `gradient_contract.target` to `codebook` (lines 72–78), and declares `base_loss` and `role_loss` nonzero (lines 79–85). The configured role parameter group is `rq.vq_layers.2.embed` (lines 24–27). It does **not** specify an encoder gradient path for `role_loss`.
- `spec.json` separately sets `distributed_contract.verify_encoder_and_codebook_gradients` to `true` (line 115). This asks for encoder and codebook gradient verification at the distributed-contract level; it does not assign both gradients to the role loss.
- In `train_rqvae.py`, `_behaviour_factors` computes `latent = model.encoder(embeddings)` inside `torch.no_grad()` (lines 1207–1208), then builds the successor factors from that latent (lines 1218–1228). Consequently, `role_inputs` has no autograd path back to encoder parameters.
- `_role_level_loss` passes those factors to the role level's VQ layer and returns its quantization loss (lines 1242–1252). Thus the role term can train that level's codebook parameters, but cannot train the encoder through the role factors. The role executor returns zero because the role commitment is already included in the host quantization loss (lines 1271–1282).
- The ordinary model forward computes the encoder output with gradients enabled (`model.py`, lines 26–31). Therefore the base/reconstruction path can still supply encoder gradients; the `no_grad` in `_behaviour_factors` only blocks the role-factor path.

## VERDICT

**CONSISTENT.** The role term's declared target is the codebook, and its configured role-level parameter group is the level-2 embedding table. The `no_grad` scope means that this term does not deliver encoder gradients, but `spec.json` does not declare an encoder gradient specifically from `role_loss`. The distributed encoder-and-codebook check is compatible with encoder gradients coming from the base loss and codebook gradients from the role loss (as well as the host quantization loss).

## REQUIRED_FIX

No fix is required for the stated gradient contract. For clarity, document that `role_loss` is expected to reach the role codebook only, while encoder-gradient verification is against the full/base training objective. If the intended contract is instead that the role term itself updates the encoder through the behaviour factors, remove the `torch.no_grad()` around the encoder call in `_behaviour_factors` and update the loss-specific gradient contract accordingly.
