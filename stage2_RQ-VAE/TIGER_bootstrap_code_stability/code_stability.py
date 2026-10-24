"""Train-only bootstrap agreement feature for frozen RQ-VAE codes."""
import torch


def stability_tokens(embeddings, train_ids, quantize, *, seed=42, sigma=1.0, draws=8):
    """Return [N,1] stability bins; quantize maps latent batches to code tuples."""
    train = embeddings[torch.as_tensor(train_ids, device=embeddings.device)]
    variance = train.var(dim=0, unbiased=False).clamp_min(0)
    scale = variance.sqrt() * sigma
    generator = torch.Generator(device=embeddings.device).manual_seed(seed)
    n = embeddings.shape[0]
    codes = []
    for _ in range(draws):
        noise = torch.randn(embeddings.shape, generator=generator, device=embeddings.device, dtype=embeddings.dtype) * scale
        codes.append(quantize(embeddings + noise).reshape(n, -1))
    stack = torch.stack(codes, dim=1)
    # Compare complete tuples by pairwise equality; agreement is modal frequency / B.
    matches = (stack[:, :, None, :] == stack[:, None, :, :]).all(dim=-1).sum(dim=-1)
    agreement = matches.max(dim=1).values.to(torch.float32) / draws
    # Explicit specified boundaries; endpoint equality belongs to higher bin.
    token = torch.where(agreement < .5, 0, torch.where(agreement < .75, 1, torch.where(agreement < .95, 2, 3)))
    return token[:, None], agreement
