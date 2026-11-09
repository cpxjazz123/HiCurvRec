"""CPU-only probe: what does the structure term do at a collision vs a real pair?

Not a suite. It answers one question with numbers: two items that share all three
codes have a structure gradient of exactly zero (squared stress is smooth there),
while two distinct items receive a non-zero structure gradient; the reconstruction
gradient is non-zero in both cases and the codebook gets its gradient through the
gather path.
"""
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import mixed_geometry as G
import structure as S
import training_hooks as H
from model import RQVAE

SEED = 42


def build_model():
    cfg = SimpleNamespace(hidden_sizes=[512, 256, 128], codebook_num=3, codebook_size=(256, 256, 256),
                          codebook_dim=32, dropout=0.0, beta=0.25, vq_type="vq",
                          ema_decay=0.99, fix_code_embs=False, sk_epsilon=0.003, sk_iters=5)
    torch.manual_seed(SEED)
    return RQVAE(cfg, in_dim=768)


def grad_norms(term, model, modules):
    params = [p for module in modules for p in module.parameters() if p.requires_grad]
    grads = torch.autograd.grad(term, params, retain_graph=True, allow_unused=True)
    per_module = {}
    for module in modules:
        module_params = [p for p in module.parameters() if p.requires_grad]
        module_grads = torch.autograd.grad(term, module_params, retain_graph=True, allow_unused=True)
        per_module[module.__class__.__name__] = float(sum(float(g.pow(2).sum()) for g in module_grads if g is not None)) ** 0.5
    return per_module


def main():
    embeddings = torch.from_numpy(np.load(str(Path(__file__).resolve().parents[3]
        / "stage1_GeneEmbedding" / "output" / "sentence_t5.npy"))[:64]).float()
    model = build_model()
    model.eval()
    with torch.no_grad():
        _, _, _, tokens, _ = model.forward_with_codes(embeddings)

    same = (0, 0)          # the same item twice: shares all three codes
    distinct = None
    with torch.no_grad():
        for a in range(len(embeddings)):
            for b in range(a + 1, len(embeddings)):
                if tokens[a, 0].item() != tokens[b, 0].item():
                    distinct = (a, b)
                    break
            if distinct:
                break
    if distinct is None:
        report_distinct_missing = True
        distinct = (1, 2)
    report = {"collision": same == (0, 0), "items": len(embeddings)}

    for name, (a, b) in (("collision", same), ("distinct", distinct)):
        # positions must be built with grad enabled, otherwise the term has no graph
        _, _, _, pair_tokens, pair_levels = model.forward_with_codes(embeddings[[a, b]])
        pair_positions = pair_levels[-1] + (model.codes_to_positions(pair_tokens)
                                            - model.codes_to_positions(pair_tokens).detach())
        left = pair_positions[0:1]
        right = pair_positions[1:2]
        target = float(torch.sqrt(G.mixed_sq(left, right).clamp_min(0)).item())
        scale = max(target, 1.0)
        term = S.pair_stress(left, right, scale=scale, h=1.0, geometry_on=False)
        recon, quant_loss, _, _, _ = model.forward_with_codes(embeddings[[a, b]])
        base, _ = model.compute_loss(embeddings[[a, b]], recon, quant_loss)
        report[name] = {
            "distance": target,
            "structure_term": float(term),
            "structure_grad": grad_norms(term, model, [model.encoder, model.rq]),
            "recon_grad": grad_norms(base, model, [model.encoder, model.rq]),
        }

    # one structure-only step: do the tokens of the two items move?
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    with torch.no_grad():
        _, _, _, before, _ = model.forward_with_codes(embeddings[[0, 1, 2]])
    _, _, _, step_tokens, step_levels = model.forward_with_codes(embeddings[[1, 2]])
    step_positions = step_levels[-1] + (model.codes_to_positions(step_tokens)
                                       - model.codes_to_positions(step_tokens).detach())
    term = S.pair_stress(step_positions[0:1], step_positions[1:2], scale=1.0, h=1.0, geometry_on=False)
    optimizer.zero_grad()
    term.backward()
    optimizer.step()
    with torch.no_grad():
        _, _, _, after, _ = model.forward_with_codes(embeddings[[0, 1, 2]])
    report["tokens_before"] = before.tolist()
    report["tokens_after_one_structure_step"] = after.tolist()
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
